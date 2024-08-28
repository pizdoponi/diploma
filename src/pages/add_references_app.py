import streamlit as st
from langchain_community.graphs import Neo4jGraph

from src.models.nodes import Node, NodeRelationship

graph = Neo4jGraph("bolt://localhost:7687", "neo4j", "password")
if "element_nodes" not in st.session_state:
    st.write("Booting application up, this might take a while...")
    document_id = "7e47da77-a93d-4cce-a085-065b42f8164d"
    query = f"""
        match (s:Section)-[:IS_PART_OF*]->(d:Document {{id: "{document_id}"}})
        where not (:Section)-[:IS_PART_OF]->(s)
        with s
        match (e:Element)-[:IS_PART_OF*]->(s)
        return 
          e.id as id, 
          labels(e) as labels, 
          e.text as text, 
          apoc.map.fromPairs([key IN keys(e) WHERE NOT key IN ['id', 'text'] | [key, e[key]]]) as metadata,
          s.text as section_title
        order by s.index, e.index
    """
    st.write("Finding relavant nodes in the graph")
    matched = graph.query(query)
    st.session_state.element_nodes = [Node(**node) for node in matched]
    st.session_state.section_titles = [e["section_title"] for e in matched]
    st.session_state.element_nodes_index = 0
    st.write("Extracting references")
    with open("./tmp_references.pkl", "rb") as f:
        import pickle

        st.session_state.reference_groups = pickle.load(f)
        st.session_state.reference_groups = [
            e for e in st.session_state.reference_groups
        ]

    st.success("App is ready to use")
    st.rerun()

if "target_document_node" not in st.session_state:
    st.session_state.target_document_node = None
if "target_section_node" not in st.session_state:
    st.session_state.target_section_node = None
if "target_element_node" not in st.session_state:
    st.session_state.target_element_node = None

section_titles = st.session_state.section_titles
element_nodes = st.session_state.element_nodes
index = st.session_state.element_nodes_index
references = [list(e.keys()) for e in st.session_state.reference_groups]


def get_nested_nodes(node: Node) -> list[Node]:
    if len(st.session_state.selected_nodes_stack) == 0:
        top_level_nodes_query = f"""
            MATCH (n)
            WHERE NOT (n)-[:IS_PART_OF]->()
            RETURN 
              n.id as id, 
              labels(n) as labels, 
              n.text as text, 
              apoc.map.fromPairs([key IN keys(n) WHERE NOT key IN ['id', 'text'] | [key, n[key]]]) as metadata
        """
        query = top_level_nodes_query
    else:
        last_selected_node = st.session_state.selected_nodes_stack[-1]
        nested_nodes_query = f"""
            MATCH (n)-[:IS_PART_OF]->({{ id: '{last_selected_node.id}' }})
            RETURN 
              n.id as id, 
              labels(n) as labels, 
              n.text as text, 
              apoc.map.fromPairs([key IN keys(n) WHERE NOT key IN ['id', 'text'] | [key, n[key]]]) as metadata
          """
        query = nested_nodes_query
    nested_nodes = graph.query(query)
    return [Node(**node) for node in nested_nodes]


if len(references[index]) == 0:
    st.session_state.element_nodes_index += 1
    print("skipping")
    st.rerun()

# st.write(section_titles[index])
st.header(section_titles[index])
st.write(element_nodes[index].text)
# st.html(
#     "<p>"
#     + element_nodes[index].text.replace(
#         references[index], f"<mark>{references[index]}</mark>"
#     )
#     + "</p>"
# )
st.write(references[index])


if st.button("Skip reference"):
    st.session_state.element_nodes_index += 1
    st.rerun()

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
        source_node = element_nodes[index]
        target_node = (
            st.session_state.target_element_node
            if st.session_state.target_element_node is not None
            else st.session_state.target_section_node
        )
        relationship = NodeRelationship(
            labels=["REFERENCES"],
            source_node_id=source_node.id,  # type: ignore
            target_node_id=target_node.id,  # type: ignore
            metadata={"citing_text": references[index]},
        )

        relationship.save_to_neo4j()
        st.success("Reference saved")
