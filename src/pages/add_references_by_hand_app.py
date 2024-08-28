import streamlit as st
from langchain_community.graphs import Neo4jGraph

from src.models.nodes import Node, NodeRelationship

graph = Neo4jGraph("bolt://localhost:7687", "neo4j", "password")

if "source_document_node" not in st.session_state:
    st.session_state.source_document_node = None  # Node
if "source_section_node" not in st.session_state:
    st.session_state.source_section_node = None  # Node
if "source_element_node" not in st.session_state:
    st.session_state.source_element_node = None  # Node
if "target_document_node" not in st.session_state:
    st.session_state.target_document_node = None  # Node
if "target_section_node" not in st.session_state:
    st.session_state.target_section_node = None  # Node
if "target_element_node" not in st.session_state:
    st.session_state.target_element_node = None  # Node
if "citing_text" not in st.session_state:
    st.session_state.citing_text = None  # str


st.session_state.source_document_node = st.selectbox(
    "Select a document",
    options=[
        Node(**node)
        for node in graph.query(
            "MATCH (n:Document) RETURN n.id as id, labels(n) as labels, n.text as text, apoc.map.fromPairs([key IN keys(n) WHERE NOT key IN ['id', 'text'] | [key, n[key]]]) as metadata"
        )
    ],
    format_func=lambda x: x.text,
    index=None,
    key="source_document_node_input",
)

if st.session_state.source_document_node is not None:
    st.session_state.source_section_node = st.selectbox(
        "Select a section",
        options=[
            Node(**node)
            for node in graph.query(
                f"""
                MATCH (:Element)-[:IS_PART_OF]->(n:Section)-[:IS_PART_OF*]->({{ id: '{st.session_state.source_document_node.id}' }})
                WITH DISTINCT n
                RETURN n.id as id, labels(n) as labels, n.text as text, apoc.map.fromPairs([key IN keys(n) WHERE NOT key IN ['id', 'text'] | [key, n[key]]]) as metadata
                ORDER BY n.index
                """
            )
        ],
        format_func=lambda x: x.text,
        index=None,
        key="source_section_node_input",
    )

if st.session_state.source_section_node is not None:
    st.session_state.source_element_node = st.selectbox(
        "Select an element",
        options=[
            Node(**node)
            for node in graph.query(
                f"""
                MATCH (n:Element)-[:IS_PART_OF*]->({{ id: '{st.session_state.source_section_node.id}' }})
                RETURN n.id as id, labels(n) as labels, n.text as text, apoc.map.fromPairs([key IN keys(n) WHERE NOT key IN ['id', 'text'] | [key, n[key]]]) as metadata
                ORDER BY n.index
                """
            )
        ],
        format_func=lambda x: x.text,
        index=None,
        key="source_element_node_input",
    )

if st.session_state.source_element_node is not None:
    st.write(st.session_state.source_element_node.text)  # type: ignore
    st.session_state.citing_text = st.text_input("Citing text")
else:
    st.stop()


if st.session_state.citing_text is not None:
    # preset target_document_node, target_section_node, target_element_node
    # to the same values as source_document_node, source_section_node, source_element_node
    st.session_state.target_document_node = st.session_state.source_document_node_input
    st.session_state.target_section_node = st.session_state.source_section_node_input
    st.session_state.target_element_node = st.session_state.source_element_node_input

    st.session_state.target_document_node = st.selectbox(
        "Select a document",
        options=[
            Node(**node)
            for node in graph.query(
                "MATCH (n:Document) RETURN n.id as id, labels(n) as labels, n.text as text, apoc.map.fromPairs([key IN keys(n) WHERE NOT key IN ['id', 'text'] | [key, n[key]]]) as metadata"
            )
        ],
        format_func=lambda x: x.text,
        index=None,
        key="target_document_node_input",
    )

if st.session_state.target_document_node is not None:
    st.session_state.target_section_node = st.selectbox(
        "Select a section",
        options=[
            Node(**node)
            for node in graph.query(
                f"""
                MATCH (n:Section)-[:IS_PART_OF*]->({{ id: '{st.session_state.target_document_node.id}' }})
                WITH DISTINCT n
                RETURN n.id as id, labels(n) as labels, n.text as text, apoc.map.fromPairs([key IN keys(n) WHERE NOT key IN ['id', 'text'] | [key, n[key]]]) as metadata
                ORDER BY n.index
                """
            )
        ],
        format_func=lambda x: x.text,
        index=None,
        key="target_section_node_input",
    )

if st.session_state.target_section_node is not None:
    st.session_state.target_element_node = st.selectbox(
        "Select an element",
        options=[
            Node(**node)
            for node in graph.query(
                f"""
                MATCH (n:Element)-[:IS_PART_OF*]->({{ id: '{st.session_state.target_section_node.id}' }})
                RETURN n.id as id, labels(n) as labels, n.text as text, apoc.map.fromPairs([key IN keys(n) WHERE NOT key IN ['id', 'text'] | [key, n[key]]]) as metadata
                ORDER BY n.index
                """
            )
        ],
        format_func=lambda x: x.text,
        index=None,
        key="target_element_node_input",
    )

if (
    st.session_state.target_document_node is not None
    or st.session_state.target_section_node is not None
    or st.session_state.target_element_node is not None
):
    if st.button("Save reference"):
        source_node = st.session_state.source_element_node
        # TODO: make it possible to save references to documents
        target_node = (
            st.session_state.target_element_node
            if st.session_state.target_element_node is not None
            else st.session_state.target_section_node
        )
        relationship = NodeRelationship(
            labels=["REFERENCES"],
            source_node_id=source_node.id,  # type: ignore
            target_node_id=target_node.id,  # type: ignore
            metadata={"citing_text": st.session_state.citing_text},
        )
        relationship.save_to_neo4j()
        st.session_state.citing_text = None
        st.session_state.target_document_node = None
        st.session_state.target_section_node = None
        st.session_state.target_element_node = None
        st.toast("Reference saved", icon="🎉")
