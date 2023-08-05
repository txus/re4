# re4: Q&A over documents with knowledge graphs & ontology discovery
Re4 is a Q&A chatbot agent backed by an ever-growing knowledge graph. It does 4 things:

- Retrieve: Like any Q&A over documents, when it receives a question it retrieves documents from a vector store.
- Respond: It then responds with this limited context, adding in any prior knowledge it may have in the knowledge graph.
- Reflect: It updates entries in the knowledge graph, adding or removing as needed according to what it learned from the interaction.

In agent parlance, this makes it very good at exploitation: primed by user interaction, it'll learn better and better ontologies
about the set of concepts that it is asked about.

What about exploration? We need intrinsic curiosity, and the easiest proxy is random exploration! Here comes the 4th thing:

- Research: On demand or on a schedule, the agent will consult random documents and start reflecting on them, growing the knowledge graph.

With these 4 steps, an ontology of facts and relations about your documents (or codebase) will grow naturally, and the agent
will become ever better at answering any questions.

## How to use it?

You'll need OPENAI_API_KEY as an environment variable. Copy `.env.example` to `.env` and fill it in.

```bash
# to bring up Neo4j:
docker-compose up

poetry install

# to run a chat session over a document folder (for now it consumes codebases
# with files that end in (".py", ".rb", ".ts", ".tsx", ".js", ".md"), but you
# can change it!)

poetry run re4 chat --codebase ../path/to/my/codebase

# to run research (random exploration on 4 documents):

poetry run re4 research --codebase ../path/to/my/codebase -n 4
```

Tail `debug.log` to see what's going on. Every time it learns anything, it'll put it there.
You can also check the Neo4j graph by opening http://localhost:7474/ and authenticating with `neo4j` as user and `password` as password.
