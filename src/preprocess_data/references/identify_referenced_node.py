from textwrap import dedent
from typing import Union

from langchain_community.chat_models import ChatOllama
from langchain_community.graphs import Neo4jGraph
from langchain_core.output_parsers import BaseOutputParser

from src.models.nodes import Node

graph = Neo4jGraph("bolt://localhost:7687", "neo4j", "password")


class ValidChoiceOutputParser(BaseOutputParser[Union[str, None]]):
    choices: list[str]

    def parse(self, text: str) -> Union[str, None]:
        if text in self.choices:
            return text
        return None


def _identify_document_node(
    reference: str,
    reference_group: str,
    citing_node_document_title: str,
    document_titles: list[str],
) -> Union[str, None]:
    document_titles_str = "\n".join(document_titles)
    INSTRUCTION = dedent(
        f"""
        # Instruction
        You are an expert in identifying the target law document for a reference.
        You will receive the following pieces of crucial information:
            1. the information about the citing node, i.e. the node that contains the reference. This will contain:
                - in what legal document is this found
            2. the reference itself
            3. the titles of (all) other legal documents
        Your task is to determine to what legal document does the reference point.
        The reference is composed of two parts, the reference itself, and the reference group.
        The reference group may contain more than one reference, but some additional information.
        You should focus mostly on the individual reference, but take reference group into consideration when appropriate.
        If the reference contains text such as "... tega zakon" you can be sure that it references the same law document.
        Similarly, if information about law title is omitted, you can deduct it is in the same legal document as well.
        Answer by returning an exact title of the legal document, and just the title.
        If it references a legal document that is not provided to you, respond with an empty string. This will signalise the absence of a found legal document.

        # Legal Document Titles
        {document_titles_str}
        """.strip()
    )
    EXAMPLES = [
        (
            dedent(
                """
                Document: "Zakon o davku na dodano vrednost" 
                Reference group: "točke a) drugega odstavka 63. člena tega zakona"
                Reference: "točka a) drugega odstavka 63. člena tega zakona"
                """.strip()
            ),
            "Zakon o davku na dodano vrednost",
        ),
        (
            dedent(
                """
                Document: "Zakon o davku na dodano vrednost" 
                Reference group: "13. točko 50. člena, 52. do 57. členom ali 58. členom tega zakona"
                Reference: "13. točka 50. člena"
                """.strip()
            ),
            "Zakon o davku na dodano vrednost",
        ),
        (
            dedent(
                """
                Document: "Zakon o davku na dodano vrednost" 
                Reference group: "1. do 5. točko prvega odstavka tega člena"
                Reference: "4. točka prvega odstavka tega člena"
                """.strip()
            ),
            "Zakon o davku na dodano vrednost",
        ),
        (
            dedent(
                """
                Document: "Zakon o davku na dodano vrednost" 
                Reference group: "196. členom Direktive Sveta 2006/112/ES"
                Reference: "196. člen Direktive Sveta 2006/112/ES"
                """.strip()
            ),
            "",
        ),
        (
            dedent(
                """
                Document: "Zakon o davku na dodano vrednost" 
                Reference group: "1., 2. in 3. točke tega odstavka"
                Reference: "1. točka tega odstavka"
                """.strip()
            ),
            "Zakon o davku na dodano vrednost",
        ),
    ]
    messages = [("system", INSTRUCTION)]
    for human, ai in EXAMPLES:
        messages.append(("human", human))
        messages.append(("ai", ai))
    messages.append(
        (
            "human",
            dedent(
                f"""
                Document: "f{citing_node_document_title}"
                Reference group: "f{reference_group}"
                Reference: "f{reference}"
                """.strip()
            ),
        )
    )
    model = "command-r:35b-v0.1-q4_K_S"
    llm = ChatOllama(model=model, temperature=0)
    chain = llm | ValidChoiceOutputParser(choices=document_titles)
    return chain.invoke(messages)


def _identify_section_node() -> Node:
    raise NotImplementedError()


def _identify_elemen_node() -> Node:
    raise NotImplementedError()


def identify_referenced_node(
    citing_node: Node, reference_group: str, reference: str
) -> Node:
    raise NotImplementedError()


if __name__ == "__main__":
    document_titles = ["Zakon o davku na dodano vrednost"]
    citing_node_document_title = "Zakon o davku na dodano vrednost"
    reference_group = "1. in 2. točke 46. člena tega zakona"
    reference = "1. točka 46. člena tega zakona"
    res = _identify_document_node(
        reference, reference_group, citing_node_document_title, document_titles
    )
    print(res)
