import copy
import re
from typing import Union

import streamlit as st
from bs4 import PageElement
from langchain_community.graphs import Neo4jGraph
from tqdm import tqdm

from src.models.nodes import Node, NodeRelationship
from src.preprocess_data.parse_laws import extract_content, is_html_tag_section

st.session_state.doc_id = "01c2828b-ed63-46d1-8e7b-e5a7b0667c88"

# the user first has to upload a law document
# only then does execution continue
if "html_tags" not in st.session_state:
    st.session_state.html_tags = None  # list[PageElement]
    st.session_state.html_tags_index = 0  # int


def preprocess_tags(tags: list[PageElement]) -> list[PageElement]:
    """This function preprocesses law document by trying to merge naslovi clenov with their enumeration.

    Thus, when there are two consecutive <a> tags, and the first one is an article,
    while the second one is an article title; for example:

        1. clen
        (definicije)

    they are merged as one section, i.e. one tag to be processed later.
    Instead of two and having to merge them.
    """
    better_tags = []
    i = 0
    while i < len(tags):
        if (
            is_html_tag_section(tags[i])
            and i + 1 < len(tags)
            and is_html_tag_section(tags[i + 1])
            and re.match(r"\(.+\)", tags[i + 1].text.strip())
        ):
            fixed_text = tags[i].text.strip() + "\n" + tags[i + 1].text.strip()
            tags[i].string = fixed_text  # type: ignore

            # in special cases, where section text starts with »
            # change the tag to <p> tag, because it is a changed or added article
            # and not a section. (it is a part of the previous section)
            if fixed_text.startswith("»"):
                tags[i].name = "p"  # type: ignore

            better_tags.append(tags[i])
            i += 2
        else:
            better_tags.append(tags[i])
            i += 1
    return better_tags


if st.session_state.html_tags is None:
    uploaded_file = st.file_uploader("Upload a law document", type=["html"])
    if uploaded_file is not None:
        html = uploaded_file.read()
        tags = extract_content(html)  # type: ignore
        # try to merge naslovi clenov with cleni to reduce the amount of work
        better_tags = preprocess_tags(tags)
        st.session_state.html_tags = better_tags
        st.rerun()
    else:
        st.stop()

if "section_stacks" not in st.session_state:
    # we will iterate over the tags to classify sections and elements
    # sections are also sorted by their depth level, yielding a tree structure
    # elements are nested in sections
    # for each new tag, the user determines the depth level of the section
    # if the tag is not a section, it is automatically added to the current section as an element
    # we keep track of section stack on every iteration
    # this makes it easy to undo anything, or skip over it
    st.session_state.section_stacks = [[]]  # list[list[Node]]
    # we also track the nodes and relationships between them
    # like above, this is list of lists, where nodes[i] are all of the nodes added in the i-th iteration
    # and relationships[i] are all of the relationships added in the i-th iteration
    # again, this makes it easy to undo anything, or skip over it
    st.session_state.nodes = [[]]  # list[list[Node]]
    st.session_state.relationships = [[]]  # list[list[NodeRelationship]]


def update_section_stack(tag: PageElement):
    stack = copy.deepcopy(st.session_state.section_stacks[-1])
    print(stack)

    # st.header("session_state")
    # st.write(st.session_state)
    st.header("stack")
    st.write([node.text for node in stack])
    st.header("text")
    st.write(tag.text.strip())

    # buttons to set the depth level
    # there are as many buttons + 1 as there are sections in the stack
    # +1 is because it must be possible to increase the depth level
    num_cols = len(stack) + 1
    # cols = st.columns(num_cols)
    # for i, col in enumerate(cols):
    for i in range(num_cols):
        # if col.button(label=str(i)):
        if st.button(label=str(i)):
            new_node = Node(labels=["Section"], text=tag.text.strip())
            new_rel = NodeRelationship(
                labels=["IS_PART_OF"],
                source_node_id=new_node.id,
                target_node_id=stack[i - 1].id if i > 0 else st.session_state.doc_id,
            )

            new_stack = stack[:i] + [new_node]
            st.session_state.section_stacks.append(new_stack)
            st.session_state.nodes.append([new_node])
            st.session_state.relationships.append([new_rel])

            st.session_state.html_tags_index += 1
            st.rerun()

    # col1, col2 = st.columns(2, gap="small")
    if st.button("merge with previous section"):
        node = stack[-1]
        node.text += "\n" + tag.text.strip()
        stack[-1] = node

        nodes = copy.deepcopy(st.session_state.nodes[-1])
        nodes = [n for n in nodes if n.id != node.id] + [node]
        relationships = copy.deepcopy(st.session_state.relationships[-1])

        st.session_state.section_stacks.append(stack)
        st.session_state.nodes.append(nodes)
        st.session_state.relationships.append(relationships)

        st.session_state.html_tags_index += 1
        st.rerun()
    if st.button("this is not a section. convert to element"):
        element_nodes, element_relationships = split_vsebina_clena(
            tag.text.strip(), clen_id=st.session_state.section_stacks[-1][-1].id
        )

        st.session_state.section_stacks.append(stack)
        st.session_state.nodes.append(element_nodes)
        st.session_state.relationships.append(element_relationships)

        st.session_state.html_tags_index += 1
        st.rerun()

    # col3, col4, col5 = st.columns(3, gap="small")
    if st.button("undo"):
        # get the html tag index of the previous section
        i = st.session_state.html_tags_index - 1
        while i >= 0 and not is_html_tag_section(st.session_state.html_tags[i]):
            i -= 1
        # reset the index to the previous section
        st.session_state.html_tags_index = i
        # remove the last section stack, nodes and relationships, respecting i
        st.session_state.section_stacks = st.session_state.section_stacks[: i + 1]
        st.session_state.nodes = st.session_state.nodes[: i + 1]
        st.session_state.relationships = st.session_state.relationships[: i + 1]

        st.rerun()

    if st.button("skip"):
        # copy the last section stack, nodes and relationships
        st.session_state.section_stacks.append(stack)
        st.session_state.nodes.append(copy.deepcopy(st.session_state.nodes[-1]))
        st.session_state.relationships.append(
            copy.deepcopy(st.session_state.relationships[-1])
        )

        st.session_state.html_tags_index += 1
        st.rerun()

    if st.button("end"):
        st.session_state.html_tags_index = len(st.session_state.html_tags)
        st.rerun()


def split_vsebina_clena(
    vsebina_clena: str,
    clen_id: str,
) -> tuple[list[Node], list[NodeRelationship]]:
    nodes = []
    relationships = []

    lines = vsebina_clena.split("\n")
    lines = [line.strip() for line in lines if line.strip() != ""]

    # Regular expressions for hierarchical levels
    odstavek_regex = r"^\(\d+\) .*?"
    tocka_stevilka_regex = r"^\d+\.[a-zčšž]? .*?"
    tocka_crka_regex = r"^[a-zčšž]+\) .*?"
    alineja_regex = r"^[-–—] .*?"
    regexes = [odstavek_regex, tocka_stevilka_regex, tocka_crka_regex, alineja_regex]

    # if the first line is plaintext, convert it to odstavek
    if not any([re.match(regex, lines[0]) for regex in regexes]):
        lines[0] = "(1) " + lines[0]

    element_nodes_index = 0
    stack: list[Union[Node, None]] = [None for _ in range(len(regexes))]

    for line in lines:
        matched = False
        for i, regex in enumerate(regexes):
            if re.match(regex, line):
                new_node = Node(
                    labels=["Element"],
                    text=line,
                    metadata={"entire_text": line, "index": element_nodes_index},
                )
                element_nodes_index += 1
                stack[i] = new_node
                matched = True

                nodes.append(new_node)
                # append the text of the line to all the previous elements,
                # to make up the entire_text
                for j in range(i):
                    if stack[j] is not None:
                        stack[j].metadata["entire_text"] += "\n" + line  # type: ignore

                # add the relationship
                # to the clen node if this is the first matched regex (top level element)
                # or to the last non None element
                if i == 0:
                    relationships.append(
                        NodeRelationship(
                            labels=["IS_PART_OF"],
                            source_node_id=new_node.id,
                            target_node_id=clen_id,
                        )
                    )
                else:
                    last_node = None
                    while last_node is None:
                        i -= 1
                        last_node = stack[i]
                    relationships.append(
                        NodeRelationship(
                            labels=["IS_PART_OF"],
                            source_node_id=new_node.id,
                            target_node_id=last_node.id,
                        )
                    )
        if not matched:
            print("error matching:", line)

    return nodes, relationships


# main loop
if st.session_state.html_tags_index < len(st.session_state.html_tags):
    tag = st.session_state.html_tags[st.session_state.html_tags_index]

    if is_html_tag_section(tag):
        # the user should update the section stack
        update_section_stack(tag)
    else:
        element_nodes, element_relationships = split_vsebina_clena(
            tag.text.strip(), clen_id=st.session_state.section_stacks[-1][-1].id
        )

        st.session_state.section_stacks.append(st.session_state.section_stacks[-1])
        st.session_state.nodes.append(element_nodes)
        st.session_state.relationships.append(element_relationships)

        st.session_state.html_tags_index += 1
        st.rerun()

# save the nodes and relationships
if st.session_state.html_tags_index == len(st.session_state.html_tags):
    nodes: list[Node] = []
    for node_list in st.session_state.nodes:
        nodes += node_list
    relationships: list[NodeRelationship] = []
    for rel_list in st.session_state.relationships:
        relationships += rel_list

    graph = Neo4jGraph("bolt://localhost:7687", "neo4j", "password")

    for node in tqdm(nodes, desc="Saving nodes"):
        node.save_to_neo4j()
    for rel in tqdm(relationships, desc="Saving relationships"):
        rel.save_to_neo4j()
