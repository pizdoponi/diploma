import re
from typing import Union

import streamlit as st
from termcolor import colored, cprint

from src.models.nodes import Node, NodeRelationship
from src.preprocess_data.parse_laws import extract_content, is_html_tag_section

with open("laws/2011-01-0554.html", "r") as f:
    html = f.read()

elements = extract_content(html)

better_elements = []
i = 0
while i < len(elements):
    e = elements[i]
    if (
        is_html_tag_section(elements[i])
        and i + 1 < len(elements)
        and is_html_tag_section(elements[i + 1])
        and re.match(r"\(.+\)", elements[i + 1].text.strip())
    ):
        fixed_text = elements[i].text.strip() + "\n" + elements[i + 1].text.strip()
        elements[i].string = fixed_text  # type: ignore
        better_elements.append(elements[i])
        i += 2
    else:
        better_elements.append(elements[i])
        i += 1
elements = better_elements


def get_user_input(text: str, section_stack: list[Node]) -> int:
    remove_newlines = lambda text: text.replace("\n", "\\n")
    print()
    print("section_stack:")
    for i, n in enumerate(section_stack):
        cprint(f"{' ' * (i*4)} ({i}) {remove_newlines(n.text)}", "light_blue")
    print("text: ", end="")
    cprint(remove_newlines(text), "green")
    user = input("> ")
    while user not in [
        "-1",
        "0",
        *[str(i + 1) for i in range(len(section_stack))],
        "p",
        "",
        ">",
        "---",
        "u",
    ]:
        if user == "i" or user == "help" or user == "-h":
            print(
                "instruction: Determine how to update the section stack. You can choose:"
            )
            print(
                colored("[0.." + str(len(section_stack)) + "]", "light_yellow"),
                "to hard set the depth level",
            )
            print(
                colored("[>]", "light_yellow"),
                "to increase the depth level by one, i.e. append to the stack",
            )
            print(
                colored("[<cr>]", "light_yellow"),
                "to set the same level as the last element in section stack, i.e. replace the last element",
            )
            print(
                colored("[p]", "light_yellow"),
                "if the element is part of the [p]revious element. the text will be added to the previous node",
            )
            print(
                colored("[---]", "light_yellow"),
                "to stop the parsing of the document",
            )
        else:
            print("Invalid input:", user, "\nType i or help for instructions.")
        user = input("> ")
    if user == "---":
        return -2
    elif user == "p":
        return -1
    elif user == "u":
        return -3
    elif user == "":
        if len(section_stack) == 0:
            return 0
        return len(section_stack) - 1
    elif user == ">":
        return len(section_stack)
    else:
        return int(user)


def get_user_input_streamlit(text: str, section_stack: list[Node]) -> int:
    remove_newlines = lambda text: text.replace("\n", "\\n")

    if "user_input" not in st.session_state:
        st.session_state.user_input = None

    for i, n in enumerate(section_stack):
        st.write(f"{' ' * (i*4)} ({i}) {remove_newlines(n.text)}", "light_blue")

    for i in range(len(section_stack) + 1):
        if st.button(label=str(i), key=f"button_{i}_{text}"):
            st.session_state.user_input = i

    if st.button(label="p", key=f"button_p_{text}"):
        st.session_state.user_input = -1

    if st.button(label="stop", key=f"button_stop_{text}"):
        st.session_state.user_input = -2

    if st.session_state.user_input is not None:
        user_input = st.session_state.user_input
        st.session_state.user_input = None  # Reset the user input after using it
        return user_input
    else:
        st.stop()


def update_section_stack(
    section_stack: list[Node], new_node: Node, undo_stack: list[list[Node]]
) -> list[Node]:
    user_input = get_user_input(text=new_node.text, section_stack=section_stack)
    # user_input = get_user_input_streamlit(
    #     text=new_node.text, section_stack=section_stack
    # )
    if user_input == -2:
        raise StopIteration()
    elif user_input == -3:
        if undo_stack:
            section_stack = undo_stack.pop()
            return update_section_stack(
                section_stack, new_node, undo_stack
            )  # Recursive call to reprocess the current element
    elif user_input == -1:
        undo_stack.append(section_stack[:])
        section_stack[-1].text += "\n" + new_node.text
    else:
        undo_stack.append(section_stack[:])
        section_stack = section_stack[:user_input] + [new_node]
    return section_stack


# def update_section_stack(section_stack: list[Node], new_node: Node) -> list[Node]:
#     user_input = get_user_input(text=new_node.text, section_stack=section_stack)
#     # user_input = get_user_input_streamlit(
#     #     text=new_node.text, section_stack=section_stack
#     # )
#     if user_input == -2:
#         raise StopIteration()
#     elif user_input == -1:
#         section_stack[-1].text += "\n" + new_node.text
#     else:
#         section_stack = section_stack[:user_input] + [new_node]
#     return section_stack


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


doc_id = "7e47da77-a93d-4cce-a085-065b42f8164d"
nodes: list[Node] = []
relationships: list[NodeRelationship] = []


section_stack: list[Node] = []
section_nodes_index = 0
undo_stack: list[list[Node]] = []

for e in elements:
    try:
        if is_html_tag_section(e):
            maybe_new_node = Node(
                labels=["Section"],
                text=e.text.strip(),
                metadata={"index": section_nodes_index},
            )
            section_stack = update_section_stack(
                section_stack=section_stack,
                new_node=maybe_new_node,
                undo_stack=undo_stack,
            )
            new_node = section_stack[-1]
            # if the user has decided to append the text to the previous node
            # we don't want to create a new node
            if maybe_new_node != new_node:
                continue

            new_rel = NodeRelationship(
                labels=["IS_PART_OF"],
                source_node_id=new_node.id,
                target_node_id=(
                    doc_id if len(section_stack) == 1 else section_stack[-2].id
                ),
            )
            nodes.append(new_node)
            section_nodes_index += 1
            relationships.append(new_rel)
        else:
            element_nodes, element_relationships = split_vsebina_clena(
                vsebina_clena=e.text, clen_id=section_stack[-1].id
            )
            nodes += element_nodes
            relationships += element_relationships
    except StopIteration:
        print(
            f"The user has signaled to stop parsing. Parsed {len(nodes)} nodes and {len(relationships)} relationships."
        )
        break

from tqdm import tqdm

confirm_adding = input(
    f"Add {len(nodes)} nodes and {len(relationships)} relationships to the database? [y/n] "
)
if confirm_adding == "y":
    from langchain_community.graphs import Neo4jGraph

    graph = Neo4jGraph("bolt://localhost:7687", "neo4j", "password")
    graph.query(
        "MATCH (n)-[:IS_PART_OF*]->({ id: $doc_id }) DETACH DELETE n",
        params={"doc_id": doc_id},
    )
    for n in tqdm(nodes, desc="adding nodes"):
        n.save_to_neo4j()
    for r in tqdm(relationships, desc="adding relationships"):
        r.save_to_neo4j()

import pickle

with open("tmp_nodes.pkl", "wb") as f:
    pickle.dump(nodes, f)
with open("tmp_relationships.pkl", "wb") as f:
    pickle.dump(relationships, f)
