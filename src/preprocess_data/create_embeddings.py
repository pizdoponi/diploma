import os
from uuid import uuid4

from langchain_community.graphs import Neo4jGraph
from sentence_transformers import SentenceTransformer

from src.models.nodes import Node
from src.preprocess_data.chunk_laws import chunk_unstructured_documents
from src.utils.init_chroma import init_collection

SKIP_ALREADY_EMBEDDED = True


def embed_neo4j_elements():
    """This function embeds all Element nodes from a neo4j graph db into a Chroma collection.

    The Element nodes are structured units of information, unlike the unstructured chunks.
    First, all Element nodes are queried from the graph db.
    Alraedy embedded Element nodes are skipped if SKIP_ALREADY_EMBEDDED is set to True.
    """
    collection = init_collection("law")
    graph = Neo4jGraph("bolt://localhost:7687", "neo4j", "password")
    query = f"""
        match (e:Element)
        return 
          e.id as id, 
          labels(e) as labels, 
          e.entire_text as text, 
          apoc.map.fromPairs([key IN keys(e) WHERE NOT key IN ['id', 'text'] | [key, e[key]]]) as metadata
    """
    element_nodes = [Node(**node) for node in graph.query(query)]

    already_embedded_ids = set(collection.get()["ids"])
    not_embedded_element_nodes = [
        e for e in element_nodes if e.id not in already_embedded_ids
    ]

    if SKIP_ALREADY_EMBEDDED:
        element_nodes = not_embedded_element_nodes

    ids = [e.id for e in element_nodes]
    metadatas = [e.metadata for e in element_nodes]
    documents = ["passage: " + e.text for e in element_nodes]

    if len(ids) == 0:
        print(
            "All embeddings already generated. Set SKIP_ALREADY_EMBEDDED=False to re-embed everything."
        )
    else:
        collection.upsert(ids=ids, metadatas=metadatas, documents=documents)


def embed_unstructured_chunks():
    """This function takes all law documents, (naively) chunks them into smaller pieces, and embeds them into a Chroma collection.

    This is only used to be able to compare how much do structured Elements (and relationships between them) benefit the generation.
    """
    collection = init_collection("unstructured")
    # delete all prior documents in collection
    try:
        collection.delete(ids=collection.get()["ids"])
    except Exception:
        print("No prior documents in collection. Proceeding to generate embeddings.")

    file_names = ["laws/" + f for f in os.listdir("laws")]
    model = SentenceTransformer(".models/multilingual-e5-large")
    chunks = chunk_unstructured_documents(file_names, model.tokenizer, max_tokens=500)

    ids = [str(uuid4()) for _ in chunks]
    documents = ["passage: " + e for e in chunks]

    collection.add(ids=ids, documents=documents)
