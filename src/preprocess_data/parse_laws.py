from bs4 import BeautifulSoup, PageElement
from langchain_experimental.llms.ollama_functions import OllamaFunctions

from src.models.nodes import Node


def is_html_tag_section(tag) -> bool:
    # a tag is a section if it is an <a> tag
    return tag.name == "a"


def is_html_tag_element(tag) -> bool:
    # a tag is a section if it is not an <a> tag
    return tag.name != "a"


def preprocess_html(html: str) -> list[PageElement]:
    soup = BeautifulSoup(html, "html.parser")
    # content is inside the div with selector #divSection > div > div
    # the actual content are the children of the div
    div = soup.select_one("#divSection > div > div")
    if div is None:
        raise ValueError("Could not find the content div")
    elements = div.contents
    # remove empty lines
    elements = [element for element in elements if element != "\n"]
    # remove the first element, which is a dummy <a> element
    elements = elements[1:]
    return elements


def extract_head(html: str) -> list[PageElement]:
    elements = preprocess_html(html)
    # head are all of the elements before the first section
    head = []
    for element in elements:
        if is_html_tag_section(element):
            break
        elif is_html_tag_element(element):
            head.append(element)
        else:
            raise ValueError("Unexpected tag")
    return head


def extract_content(html: str) -> list[PageElement]:
    elements = preprocess_html(html)
    head = extract_head(html)
    # content is everything after the head
    return elements[len(head) :]


def extract_hierarchy(elements: list[PageElement]):
    # returns just the text of all section elements
    import re

    hierarchy = []
    for element in elements:
        if is_html_tag_section(element):
            # if the text is of the form (naslov clena), add it to the previous item
            if re.match(r"\(.+\)", element.text.strip()):
                hierarchy[-1] += "\\n" + element.text.strip()
            else:
                hierarchy.append(element.text.strip())
        elif is_html_tag_element(element):
            continue
        else:
            raise ValueError("Unexpected tag")
    return hierarchy


def extract_sections(elements: list[PageElement]) -> list[Node]:
    llm = OllamaFunctions(
        temperature=0, model="phi3:14b-medium-128k-instruct-q6_K", format="json"
    )

    sections = []
    sections_stack = []
    return sections


def parse_law(html: str) -> Node:
    head = extract_head(html)
    content = extract_content(html)

    sections = extract_sections(content)
    return Node(id="TODO", labels=["TODO"], text="TODO")


def parse_laws(htmls: list[str]) -> list[Node]:
    laws = []
    for html in htmls:
        law_title = "TODO"
    return laws
