import hashlib
import pathlib
import threading
import os
import datetime
import concurrent
import random
from os import devnull
from functools import cached_property
from typing import List
from contextlib import contextmanager, redirect_stderr, redirect_stdout

from pydantic import BaseModel, Field

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationChain
from langchain.output_parsers import PydanticOutputParser
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.vectorstores import Chroma
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.schema import Document

from re4.knowledge_graph import KnowledgeGraph, Triple


class Reflection(BaseModel):
    to_learn: List[Triple] = Field(
        description="new knowledge triples you learned (only the most important ones and very brief, a list of (subject, relation, object)). try to reuse entity names if applicable."
    )
    to_forget: List[Triple] = Field(
        description="knowledge triples in your prior knowledge that are no longer true (maybe because they're better expressed by the ones you learned). empty list if none!"
    )


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


def gen_hash(codebase_path: str) -> str:
    return hashlib.sha256(codebase_path.encode()).hexdigest()


def collect_files(folder):
    files_dict = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith((".py", ".rb", ".ts", ".tsx", ".js", ".md")):
                file_path = os.path.join(root, file)
                file_type = file.split(".")[-1]
                with open(file_path, "r") as f:
                    content = f.read()
                files_dict.append(
                    {
                        "page_content": content,
                        "metadata": {"filename": file_path, "extension": file_type},
                    }
                )
    return [Document(**kwargs) for kwargs in files_dict]


class Knowledge:
    def __init__(self, entities: List[str], triples: List[Triple]):
        self.entities = entities
        self.triples = triples

    def __str__(self):
        subjects = set([triple[0] for triple in self.triples])
        return "\n".join(
            [
                f"On {subject}: "
                + ", ".join(
                    [
                        f"{triple[0]}->({triple[1]})->{triple[2]}"
                        for triple in self.triples
                        if triple[0] == subject
                    ]
                )
                for subject in subjects
            ]
        )


class Agent:
    def __init__(self, codebase_path: str, verbose: bool):
        self.codebase_path = str(pathlib.Path(codebase_path).absolute().resolve())
        self.hash = gen_hash(codebase_path)
        self.verbose = verbose

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        self.lock = threading.Lock()

        self.index_codebase()

    @cached_property
    def history(self) -> ConversationBufferMemory:
        return ConversationBufferMemory()

    @cached_property
    def gpt35(self) -> ChatOpenAI:
        return ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    @cached_property
    def gpt3516k(self) -> ChatOpenAI:
        return ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0)

    def gpt4(self, streaming=False) -> ChatOpenAI:
        kwargs = {}
        if streaming:
            kwargs = {
                "streaming": True,
                "callbacks": [StreamingStdOutCallbackHandler()],
            }
        return ChatOpenAI(model_name="gpt-4", temperature=0, **kwargs)

    @cached_property
    def embeddings(self) -> OpenAIEmbeddings:
        return OpenAIEmbeddings()

    @cached_property
    def knowledge_graph(self):
        return KnowledgeGraph(
            uri=os.getenv("NEO4J_DB_URI"),
            user=os.getenv("NEO4J_DB_USER"),
            password=os.getenv("NEO4J_DB_PASS"),
            index_name=self.hash,
        )

    @property
    def index_path(self):
        return f".re4/{self.hash}/db"

    def index_codebase(self):
        if not os.path.exists(self.index_path):
            print(f"Indexing codebase at {self.codebase_path} ({self.index_path})...")

            docs = collect_files(self.codebase_path)

            print(f"Indexing {len(docs)} docs...")

            docs = CharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=1000, chunk_overlap=0
            ).split_documents(docs)

            db = Chroma.from_documents(
                docs,
                self.embeddings,
                persist_directory=self.index_path,
            )
            db.persist()

    @cached_property
    def codebase(self) -> SelfQueryRetriever:
        db = Chroma(
            persist_directory=self.index_path, embedding_function=self.embeddings
        )

        metadata_field_info = [
            AttributeInfo(name="filename", description="The file name", type="string"),
            AttributeInfo(
                name="extension", description="The extension of the file", type="string"
            ),
        ]

        return SelfQueryRetriever.from_llm(
            self.gpt35,
            db,
            "the code",
            metadata_field_info,
            verbose=False,
            enable_limit=True,
        )

    def research(self, n):
        docs = collect_files(self.codebase_path)
        random.shuffle(docs)
        for doc in docs[:n]:
            print(f"Researching {doc.metadata['filename']}...")
            prior_knowledge = self.recall(doc.page_content)

            parser = PydanticOutputParser(pydantic_object=Reflection)

            output = (
                (
                    PromptTemplate(
                        input_variables=["code", "knowledge"],
                        partial_variables={
                            "format_instructions": parser.get_format_instructions()
                        },
                        template="""You are an observant knowledge graph engineer. You explore some codebase with some prior knowledge (in the form of knowledge graph triples).
                        Now, reflect on the code you're reading, and identify any new knowledge graph triples you want to add, and any prior triples you want to remove from your knowledge graph.
                        Try to merge the new knowledge with the prior knowledge, if possible, to keep a consistent graph over time. Respond ONLY in JSON, no text before or after.

                Code:
                {code}

                Prior knowledge:
                {knowledge}

                {format_instructions}""".strip(),
                    )
                    | self.gpt4()
                )
                .invoke({"knowledge": str(prior_knowledge), "code": doc.page_content})
                .content
            )

        parsed = None

        try:
            parsed = parser.parse(output)
        except Exception as e:
            print(f"Error parsing reflection output: {e}\n\nOutput:\n{output}\n\n")
            self.log(f"Error parsing reflection output: {e}\n\nOutput:\n{output}\n\n")
            return

        logs = []

        for subject, relation, object in parsed.to_learn:
            self.knowledge_graph.add_triple(subject, relation, object)
            logs.append(f"[Research] Learned that {subject} {relation} {object}.")

        for subject, relation, object in parsed.to_forget:
            self.knowledge_graph.remove_triple(subject, relation, object)
            logs.append(
                f"[Research] Learned that {subject} no longer {relation} {object}."
            )

        for log in logs:
            self.log(log)

    def recall(self, input: str) -> Knowledge:
        triples = []

        entities = (
            (
                PromptTemplate(
                    input_variables=["input"],
                    template="""You are an observant knowledge graph engineer. Extract the most salient 3 entities or concepts mentioned in the text.
            Output them separated by commas.

            Text:
            {input}

            Entities/concepts:""".strip(),
                )
                | self.gpt35
            )
            .invoke({"input": input})
            .content.split(",")
        )

        for entity in entities:
            triples.extend(self.knowledge_graph.get_triples(entity))

        return Knowledge([x.strip() for x in entities], triples)

    def retrieve(self, input: str) -> str:
        with suppress_stdout_stderr():
            return "\n--\n".join(
                [
                    f"{doc.metadata['filename']}:\n{doc.page_content}"
                    for doc in self.codebase.get_relevant_documents(input, k=3)
                ]
            )

    def respond(self, input: str) -> str:
        documents = self.retrieve(input)

        knowledge = self.recall(f"{input}\n{documents}")

        template = """Given the following conversation history, extracted parts of a codebase, current knowledge, and a question, answer it.

        Chat history:
        {history}

        Code:
        {documents}

        Current Knowledge:
        {knowledge}

        Question: {input}
        Answer:"""

        prompt = PromptTemplate(
            input_variables=["history", "input"],
            template=template,
            partial_variables={"documents": documents, "knowledge": str(knowledge)},
        )

        conversation = ConversationChain(
            prompt=prompt,
            llm=self.gpt4(streaming=True),
            verbose=self.verbose,
            memory=self.history,
        )

        return conversation.predict(input=input), documents, knowledge

    def log(self, content: str):
        with self.lock:
            with open("debug.log", "a") as f:
                now = datetime.datetime.now()
                timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
                log_line = f"{timestamp} - {content}"
                f.write(log_line + "\n")

    def reflect(
        self,
        question: str,
        response_given: str,
        documents_retrieved: str,
        prior_knowledge: Knowledge,
    ):
        parser = PydanticOutputParser(pydantic_object=Reflection)

        output = (
            (
                PromptTemplate(
                    input_variables=["question", "code", "knowledge", "response"],
                    partial_variables={
                        "format_instructions": parser.get_format_instructions()
                    },
                    template="""You are an observant knowledge graph engineer. You were given a question, and with some prior knowledge (in the form of knowledge graph triples) and some code you've read,
                    you've given an answer. Now, reflect on all these and identify any new knowledge graph triples you want to add, and any prior triples you want to remove from your knowledge graph.
                    Try to merge the new knowledge with the prior knowledge, if possible, to keep a consistent graph over time. Respond ONLY in JSON, no text before or after.

            Question:
            {question}

            Code:
            {code}

            Prior knowledge:
            {knowledge}

            Response:
            {response}

            {format_instructions}""".strip(),
                )
                | self.gpt4()
            )
            .invoke(
                {
                    "question": question,
                    "response": response_given,
                    "knowledge": str(prior_knowledge),
                    "code": documents_retrieved,
                }
            )
            .content
        )

        parsed = None

        try:
            parsed = parser.parse(output)
        except Exception as e:
            print(f"Error parsing reflection output: {e}\n\nOutput:\n{output}\n\n")
            self.log(f"Error parsing reflection output: {e}\n\nOutput:\n{output}\n\n")
            return

        logs = []

        for subject, relation, object in parsed.to_learn:
            self.knowledge_graph.add_triple(subject, relation, object)
            logs.append(f"Learned that {subject} {relation} {object}.")

        for subject, relation, object in parsed.to_forget:
            self.knowledge_graph.remove_triple(subject, relation, object)
            logs.append(f"Learned that {subject} no longer {relation} {object}.")

        for log in logs:
            self.log(log)

        return logs

    def chat(self):
        print(f"Chatting with codebase at {self.codebase_path}.")

        while True:
            try:
                q = input("\n> ").strip()
            except EOFError:
                print("Awaiting for any pending learnings...")
                self.executor.shutdown(wait=True)
                exit(0)

            if q == "forget":
                self.knowledge_graph.drop_index()
                self.knowledge_graph.create_index()
                print("Forgotten everything.")
                continue

            response, documents_retrieved, prior_knowledge = self.respond(q)

            self.executor.submit(
                self.reflect, q, response, documents_retrieved, prior_knowledge
            )
