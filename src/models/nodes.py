from typing import Any, Dict, Generic, TypeVar
from uuid import uuid4

from langchain_community.graphs import Neo4jGraph
from pydantic import BaseModel, Field

MetadataT = TypeVar("MetadataT", bound=Dict[str, Any])


class NodeRelationship(BaseModel, Generic[MetadataT]):
    id: str = Field(default_factory=lambda: str(uuid4()))
    labels: list[str]
    metadata: MetadataT = {}  # type: ignore
    source_node_id: str
    target_node_id: str

    def save_to_neo4j(self):
        graph = Neo4jGraph(
            url="bolt://localhost:7687", username="neo4j", password="password"
        )
        labels_str = ":" + ":".join(self.labels) if self.labels else ""
        properties = {"id": self.id, **self.metadata}
        properties_str = ", ".join([f"{k}: ${k}" for k in properties.keys()])
        query = f"""
        MATCH (source_node {{ id: $source_node_id }})
        WITH source_node
        MATCH (target_node {{ id: $target_node_id }})
        WITH source_node, target_node
        MERGE (source_node)-[r{labels_str} {{ {properties_str} }}]->(target_node)
        RETURN source_node, r, target_node
        """
        params = {
            "id": self.id,
            **self.metadata,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
        }
        graph.query(query, params)


class Node(BaseModel, Generic[MetadataT]):
    id: str = Field(default_factory=lambda: str(uuid4()))
    labels: list[str]
    metadata: MetadataT = {}  # type: ignore
    text: str = ""

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Node) and self.id == other.id:
            return True
        return False

    def get_relationships(
        self, relationships: list[NodeRelationship]
    ) -> list[NodeRelationship]:
        return [
            rel
            for rel in relationships
            if rel.source_node_id == self.id or rel.target_node_id == self.id
        ]

    @property
    def labels_str(self) -> str:
        return ":" + ":".join(self.labels) if self.labels else ""

    @property
    def properties_str(self) -> str:
        properties = {"id": self.id, "text": self.text, **self.metadata}
        return ", ".join([f"{k}: ${k}" for k in properties.keys()])

    def save_to_neo4j(self):
        graph = Neo4jGraph(
            url="bolt://localhost:7687", username="neo4j", password="password"
        )
        query = f"""
        MERGE (n{self.labels_str} {{ {self.properties_str} }})
        RETURN n
        """
        params = {"id": self.id, "text": self.text, **self.metadata}
        graph.query(query, params)
