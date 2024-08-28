from textwrap import dedent

import torch
from langchain_community.chat_models import ChatOllama
from langchain_community.graphs import Neo4jGraph
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.models.nodes import Node
from src.utils.init_chroma import init_collection


def retrieve_documents(
    user_input: str, collection_name: str, n_results: int = 5
) -> list[Document]:
    """A simple function to retrieve `n_results` documents from `collection_name` based on `user_input` query."""
    # the query should start with "query: "
    query = (
        "query: " + user_input if not user_input.startswith("query: ") else user_input
    )
    collection = init_collection(collection_name)
    matches = collection.query(
        query_texts=query, n_results=n_results, include=["documents"]
    )
    ids = matches["ids"][0]
    texts: list[str] = matches["documents"][0]  # type: ignore
    # remove "passage: " at the beginning of texts
    texts = [t[9:] for t in texts]
    documents = [
        Document(page_content=text, metadata={"id": id}) for text, id in zip(texts, ids)
    ]
    return documents


def transform_query(user_input: str) -> str:
    """Transform user input with an llm into a more suitable "law" query.

    Does not prepend the 'query: ' prefix or anything like that. Just transforms the query.
    """
    INSTRUCTION = dedent(
        f"""
        # Navodilo
        Pretvori  besedilo spodaj v pravni jezik primerem za iskanje po pravnih dokumentih (zakonih).

        Osredotoči se, da iz besedila vzameš vse pomembne informacije in jih pretvoriš v pravno terminologijo.
        Pravni dokumenti so napisani v pravni terminologiji. Dokumente iščemo s primerjanjem sematičnosti besedil
        (tj. rag vectorstore retrieval). V ta namen želimo spodnje besedilo pretvoriti v ustrezno pravno obliko,
        da najdemo vsebine, ki so najbolj primerne za uporabnikovo poizvedbo. 

        Pretvorba naj bo v preprosti tekstovni obliki, brez kakršnegakoli formatiranja.

        # Besedilo
        {user_input}
        """.strip()
    )
    query_transformer = ChatOllama(model="gemma2:9b-instruct-q6_K")
    str_output_parser = StrOutputParser()
    return (query_transformer | str_output_parser).invoke(INSTRUCTION)


def rerank_documents(user_input: str, documents: list[Document]) -> list[Document]:
    """A function to rerank documents based on user input."""
    tokenizer = AutoTokenizer.from_pretrained(".models/bge-reranker-v2-m3")
    reranker = (
        AutoModelForSequenceClassification.from_pretrained(".models/bge-reranker-v2-m3")
        .to("mps")
        .eval()
    )

    sentence_pairs = [[user_input, doc.page_content] for doc in documents]

    with torch.no_grad():
        inputs = tokenizer(
            sentence_pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        ).to("mps")
        scores = (
            reranker(**inputs, return_dict=True)
            .logits.view(
                -1,
            )
            .float()
        )
        # print(scores)

        reranked_documents = [
            doc
            for _, doc in sorted(
                zip(scores, documents), key=lambda x: x[0], reverse=True
            )
        ]
        return reranked_documents


def get_full_context(id: str) -> str:
    """This function retrieves the entire (relevant) text for a node given its id.

    First, an Element node with the given id is queried.
    If there is no such node, an AssertionError is raised.

    Then, the topmost parent Element of the node is queried, and its entire_text is returned.
    """
    graph = Neo4jGraph("bolt://localhost:7687", "neo4j", "password")
    query = f"""
        MATCH (n:Element {{ id: "{id}" }})
        RETURN n.id as id, labels(n) as labels, n.text as text, apoc.map.fromPairs([key IN keys(n) WHERE NOT key IN ['id', 'text'] | [key, n[key]]]) as metadata
    """
    matches = graph.query(query)
    assert (
        len(matches) == 1
    ), f"No Element node with id {id} found, or multiple nodes found where there should be only one."
    node: Node = Node(**matches[0])

    # There are two options to get more context:
    # 1. get the topmost parent Element of node and return its full_text
    # 2. get the section that node belongs to, and concat its top level documents
    # rn, go with the first option
    query = f"""
        MATCH (e1:Element {{ id: "{node.id}" }})-[:IS_PART_OF*0..]->(e2:Element)-[:IS_PART_OF]->(:Section)
        RETURN e2.id as id, labels(e2) as labels, e2.text as text, apoc.map.fromPairs([key IN keys(e2) WHERE NOT key IN ['id', 'text'] | [key, e2[key]]]) as metadata
    """
    matches = graph.query(query)
    assert (
        len(matches) == 1
    ), "There should be only one top level Element for any other Element (an element cannot have two parents)"
    return matches[0]["metadata"]["entire_text"]


def get_referenced_text(id: str) -> list[str]:
    """Given an id of the node, get all outbound references from it and return their entire_text.

    For example, if node.text is something like "..., če je to v skladu s 42. členom tega zakona.",
    then the referenced text would be the entire_text of the 42. člen node.

    If the referenced node is an Element node, return its entire_text.
    If the referenced node is a Section node, return the entire_text of all its (topmost) Elements.
    """
    referenced_texts = []
    graph = Neo4jGraph("bolt://localhost:7687", "neo4j", "password")
    elements_query = f"""
        MATCH (n {{ id: "{id}" }})-[r:REFERENCES]->(e:Element)
        RETURN e.id as id, labels(e) as labels, e.text as text, apoc.map.fromPairs([key IN keys(e) WHERE NOT key IN ['id', 'text'] | [key, e[key]]]) as metadata, r.citing_text as citing_text
    """
    matched_elements = graph.query(elements_query)
    for element in matched_elements:
        referenced_texts.append(
            f'*{element["citing_text"]}*\n{element["metadata"]["entire_text"]}'
        )
    sections_query = f"""
        MATCH (n {{ id: "{id}" }})-[r:REFERENCES]->(s:Section)
        OPTIONAL MATCH (s)<-[:IS_PART_OF]-(e:Element)
        WITH s, r.citing_text AS citing_text, e
        ORDER BY e.index
        WITH s, COLLECT(e.entire_text) AS entire_text, citing_text
        RETURN s.id AS id, labels(s) AS labels, s.text AS section_text, 
               apoc.map.fromPairs([key IN keys(s) WHERE NOT key IN ['id', 'text'] | [key, s[key]]]) AS metadata, 
               citing_text, 
               apoc.text.join(entire_text, '\n') AS entire_text"""
    matched_sections = graph.query(sections_query)
    for section in matched_sections:
        referenced_texts.append(f'*{section["citing_text"]}*\n{section["entire_text"]}')

    return referenced_texts
