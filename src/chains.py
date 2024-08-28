from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import chain

from src.utils.rag_helpers import (
    get_full_context,
    get_referenced_text,
    rerank_documents,
    retrieve_documents,
    transform_query,
)

llm = ChatOllama(model="gemma2:9b-instruct-q6_K")
str_output_parser = StrOutputParser()


@chain
def baseline(user_input: str) -> str:
    messages = [
        (
            "system",
            "Si pravni pomočnik. Uporabnik ti bo postavil vprašanje. Nanj odgovori po najboljših zmogljivostih.",
        ),
        ("human", user_input),
    ]
    return (llm | str_output_parser).invoke(messages)


@chain
def naive_rag(user_input: str) -> str:
    documents = retrieve_documents(user_input, "unstructured", n_results=5)
    messages = [
        (
            "system",
            "Si pravni pomočnik. Uporabnik ti bo postavil vprašanje. Nanj odgovori z znanjem iz spodnjih podatkov:\n\n"
            + "\n\n".join([doc.page_content for doc in documents]),
        ),
        ("human", user_input),
    ]
    return (llm | str_output_parser).invoke(messages)


@chain
def advanced_rag(user_input: str, should_transform_query: bool = True) -> str:
    # transform query
    if should_transform_query:
        _query = transform_query(user_input)
        tranformed_query = "query: " + _query
    else:
        tranformed_query = "query: " + user_input

    documents = retrieve_documents(tranformed_query, "law", n_results=20)
    reranked_documents = rerank_documents(user_input, documents)
    # take only the top 5 reranked documents
    documents = reranked_documents[:5]
    # get the entire_text for each of the documents
    # so that if a subpoint is matched, the entire paragraph is returned
    entire_texts = [get_full_context(doc.metadata["id"]) for doc in documents]

    messages = [
        (
            "system",
            "Si pravni pomočnik. Uporabnik ti bo postavil vprašanje. Nanj odgovori z znanjem iz spodnjih podatkov:\n\n"
            + "\n\n".join(entire_texts),
        ),
        ("human", user_input),
    ]
    return (llm | str_output_parser).invoke(messages)


@chain
def kg_rag(user_input: str, should_transform_query: bool = True) -> str:
    if should_transform_query:
        transformed_query = "query: " + transform_query(user_input)
    else:
        transformed_query = "query: " + user_input

    documents = retrieve_documents(transformed_query, "law", n_results=20)
    reranked_documents = rerank_documents(user_input, documents)
    # take only the top 5 reranked documents
    documents = reranked_documents[:5]
    # get the entire_text for each of the documents
    # so that if a subpoint is matched, the entire paragraph is returned
    entire_texts = [get_full_context(doc.metadata["id"]) for doc in documents]
    # for every document, get the referended texts that it contains
    referenced_texts = [get_referenced_text(doc.metadata["id"]) for doc in documents]
    # the referenced texts are appended after every document's entire text
    # to provide the llm with full context
    for i in range(len(entire_texts)):
        entire_texts[i] += "\n\n" + "\n\n".join(referenced_texts[i])

    messages = [
        (
            "system",
            "Si pravni pomočnik. Uporabnik ti bo postavil vprašanje. Nanj odgovori z znanjem iz spodnjih podatkov:\n\n"
            + "---".join(entire_texts),
        ),
        ("human", user_input),
    ]
    return (llm | str_output_parser).invoke(messages)


chains = [baseline, naive_rag, advanced_rag, kg_rag]
