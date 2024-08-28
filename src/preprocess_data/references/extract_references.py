from textwrap import dedent

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable


class NewlineOutputParser(BaseOutputParser[list[str]]):
    def parse(self, text: str) -> list[str]:
        if text.strip() == "" or text.strip() == " ":
            return []
        lines = text.split("\n")
        if len(lines) == 1 and lines[0] == "":
            return []
        return [line.strip() for line in lines if not line.strip() != ""]


def _create_extract_reference_groups_chain() -> Runnable:
    INSTRUCTION = dedent(
        """
        You are an expert in extracting references from Slovenian law.

        # Objective:
        You will be given a short passage (one clause of an article) of Slovenian law and are tasked with extracting all references from it.

        # Types of References:
        - An entire article in the same or another law document
            - 51. člen tega zakona
            - 12. člen Zakona o davku na dodano vrednost
        - A specific clause of an article
            - 2. točka prvega odstavka tega člena
            - 3. točka 2. odstavka 12. člena Zakona o prometu
            - prva alineja 1. točke 3. odstavka 46. člena tega zakona
        - An entire chapter or subchapter of a law document
            - 2. poglavje tega zakona
            - 3. podpoglavje 2. poglavja Direktive 2006/112/ES
        - Other legal references

        The references may be stated one after another, but correspond to the same part of the law.
        In this case, you should group them together and extract them as a single group.
        Examples:
        - 2. in 3. točka tega odstavka
        - 52., 53., 54. in 55. členom tega zakona
        - drugi odstavek 90. člena in prvi odstavek 91. člena

        # Task:
        Identify the relevant text(s) that references something else and echo it back without any changes or interpretations.
        Just copy and paste the references as they are. Do not change the text in any way!

        # Formatting Instructions:
        - Each reference (group) found should be placed on a new line (the response will be parsed that way).
        - If there are no references in the text, respond with an empty string. Do not respond with "no references found" or anything else.

        # Additional Notes:
        - The passage will be in Slovenian language.
        - Follow the instructions precisely, as the response will be processed by a program that will crash if your response is not ok.

        # IMPORTANT:
        - Do not change the text in the reference in any way.
        - It there is no reference, respond with an empty string!!! Do not say "there are no references" or something similar.
        - Beware of false positives, but there must be no false negatives.
        """.strip()
    )
    EXAMPLES = [
        (
            "3. opravljanje storitev, ki jih davčni zavezanec opravi v okviru opravljanja svoje ekonomske dejavnosti na ozemlju Slovenije za plačilo;",
            [],
        ),
        (
            "(2) Za namene 2. b) točke prvega odstavka tega člena se za »prevozna sredstva« štejejo naslednja prevozna sredstva, namenjena za prevoz oseb ali blaga:",
            ["2. b) točke prvega odstavka tega člena"],
        ),
        (
            "(1) »Dobava blaga« pomeni prenos pravice do razpolaganja z opredmetenimi stvarmi kot da bi bil prejemnik lastnik.",
            [],
        ),
        (
            "e) dobave tega blaga, ki jo opravi davčni zavezanec pod pogoji iz 46., 52., 53., in 54. člena tega zakona;",
            ["46., 52., 53., in 54. člena tega zakona"],
        ),
        (
            "– nižje od tržne vrednosti in dobavitelj te dobave nima pravice do celotnega odbitka DDV v skladu z 62., 63., 65., 66., 74. in 74.i členom tega zakona ter za dobavo velja oprostitev po prvem odstavku 42. člena, 44. členu in drugem odstavku 49. člena tega zakona;",
            [
                "62., 63., 65., 66., 74. in 74.i členom tega zakona",
                "prvem odstavku 42. člena, 44. členu in drugem odstavku 49. člena tega zakona",
            ],
        ),
        (
            "(4) V davčno osnovo se ne vštevajo:",
            [],
        ),
        (
            "e) opravljanje storitev, vključno s prevoznimi storitvami in pomožnimi storitvami, razen storitev, ki so oproščene v skladu z 42. in 44. členom tega zakona, če so neposredno povezane z izvozom ali uvozom blaga v smislu drugega in tretjega odstavka 31. ter uvozom blaga v skladu s prvim odstavkom 58. člena tega zakona.",
            [
                "42. in 44. členom tega zakona",
                "drugega in tretjega odstavka 31. ter uvozom blaga v skladu s prvim odstavkom 58. člena tega zakona",
            ],
        ),
        (
            "a) vsaka oseba iz 2. do 4. točke prvega odstavka ter drugega in tretjega odstavka 76. člena tega zakona;",
            [
                "2. do 4. točke prvega odstavka ter drugega in tretjega odstavka 76. člena tega zakona"
            ],
        ),
        (
            "13. točko 50. člena, 52. do 57. členom ali 58. členom tega zakona",
            ["13. točko 50. člena, 52. do 57. členom ali 58. členom tega zakona"],
        ),
    ]

    messages: list[BaseMessage] = [SystemMessage(content=INSTRUCTION)]
    for user, ai in EXAMPLES:
        messages.append(HumanMessage(content=user))
        messages.append(AIMessage(content="\n".join(ai)))

    prompt = ChatPromptTemplate.from_messages(messages + [("human", "{law_text}")])

    model = "gemma2:9b-instruct-q6_K"
    llm = ChatOllama(model=model, temperature=0)

    chain = prompt | llm | NewlineOutputParser()
    return chain


def _extract_reference_groups(law_text: str) -> list[str]:
    chain = _create_extract_reference_groups_chain()
    return chain.invoke({"law_text": law_text})


def _create_extract_references_from_reference_group_chain() -> Runnable:
    INSTRUCTION = dedent(
        """
    You are an expert in slovenian law and slovenian grammar.
    You will receive a passage of slovenian law that contains one or more references to some other law text.
    As there may be more than one reference, you can think of the passage that you will receive as a group of references.
    Your task is to separate this group of references into singular references.
    You should try to conjugate each reference into tha base / first tense (imenovalnik) in slovenian language.
    You must include all of the references, leave no one out.
    Format your response by writing each reference on its own line.
    If there is only one reference, simply echo the exact thing back (the one and only reference).
    You must answer back in slovenian language.
    """.strip()
    )
    EXAMPLES = [
        (
            "1., 2. in 3. točke tega odstavka",
            [
                "1. točka tega odstavka",
                "2. točka tega odstavka",
                "3. točka tega odstavka",
            ],
        ),
        ("81. členom tega zakona", ["81. člen tega zakona"]),
        (
            "52., 53., 54. in 55. členom tega zakona",
            [
                "52. člen tega zakona",
                "53. člen tega zakona",
                "54. člen tega zakona",
                "55. člen tega zakona",
            ],
        ),
        (
            "drugi odstavek 90. člena in prvi odstavek 91. člena",
            ["drugi odstavek 90. člena", "prvi odstavek 91. člena"],
        ),
        ("87., 88., 88.a člen", ["87. člen", "88. člen", "88.a člen"]),
        (
            "2. do 4. točko prvega odstavka 76. člena tega zakona",
            [
                "2. točka prvega odstavka 76. člena tega zakona",
                "3. točka prvega odstavka 76. člena tega zakona",
                "4. točka prvega odstavka 76. člena tega zakona",
            ],
        ),
    ]

    messages: list[BaseMessage] = [SystemMessage(content=INSTRUCTION)]
    for user, ai in EXAMPLES:
        messages.append(HumanMessage(content=user))
        messages.append(AIMessage(content="\n".join(ai)))

    prompt = ChatPromptTemplate.from_messages(
        messages + [("human", "{reference_group}")]
    )

    model = "llama3:8b-instruct-q8_0"
    llm = ChatOllama(model=model, temperature=0)

    chain = prompt | llm | NewlineOutputParser()
    return chain


def _extract_references_from_reference_group(
    reference_group: str,
) -> list[str]:
    chain = _create_extract_references_from_reference_group_chain()
    return chain.invoke({"reference_group": reference_group})


def extract_references(law_text: str) -> dict[str, list[str]]:
    reference_groups = _extract_reference_groups(law_text)
    return {
        reference_group: _extract_references_from_reference_group(reference_group)
        for reference_group in reference_groups
    }


if __name__ == "__main__":
    law_text = "V prvem odstavku 141. člena se na koncu 17. točke pika nadomesti s podpičjem, za njim pa se doda nova 18. točka, ki se glasi:"
    # res = extract_references(law_text)
    res = _extract_reference_groups(law_text)
    print(res)
