from typing import Tuple

from neo4j import GraphDatabase


Triple = Tuple[str, str, str]


class KnowledgeGraph:
    def __init__(self, uri, user, password, index_name):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.index_name = index_name
        self.create_index()

    def close(self):
        self.driver.close()

    def drop_index(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def create_index(self):
        with self.driver.session() as session:
            session.run(
                f"CREATE INDEX knowledge_graph_{self.index_name} IF NOT EXISTS FOR (n:Node) ON (n.name)"
            )

    def add_triple(self, subject, relation, object):
        with self.driver.session() as session:
            session.run(
                "MERGE (a:Node {name: $subject}) "
                "MERGE (b:Node {name: $object}) "
                "MERGE (a)-[r:RELATION {name: $relation}]->(b)",
                subject=subject,
                relation=relation,
                object=object,
            )

    def remove_triple(self, subject, relation, object):
        with self.driver.session() as session:
            session.run(
                "MATCH (a:Node {name: $subject})-[r:RELATION {name: $relation}]->(b:Node {name: $object}) "
                "DELETE r "
                "WITH a, b "
                "MATCH (a)-[r]->() "
                "WITH a WHERE count(r) = 0 "
                "DELETE a "
                "MATCH (b)-[r]->() "
                "WITH b WHERE count(r) = 0 "
                "DELETE b",
                subject=subject,
                relation=relation,
                object=object,
            )

    def get_triples(self, entity):
        with self.driver.session() as session:
            result = session.run(
                "MATCH (a:Node {name: $entity})-[r]->(b) "
                "RETURN a.name, r.name, b.name "
                "UNION "
                "MATCH (a)-[r]->(b:Node {name: $entity}) "
                "RETURN a.name, r.name, b.name",
                entity=entity,
            )
            return [(record[0], record[1], record[2]) for record in result]
