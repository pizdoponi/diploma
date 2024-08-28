from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.preprocess_data.parse_laws import extract_content


def chunk_unstructured_documents(
    file_names: list[str], tokenizer, max_tokens=512
) -> list[str]:
    """Chunk law documents into "stupid" chunks.
    Used for naive rag approach.
    No metadata, no connections, just text.
    The file names are paths to the law documents in the html format, downloaded using download_laws.py.
    """
    texts = []
    for file_name in file_names:
        with open(file_name, "r") as f:
            text = f.read()
            content = extract_content(text)
            texts.append("\n".join([element.text.strip() for element in content]))

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n", " ", ".", ",", ""],
        keep_separator=True,
        is_separator_regex=True,
        chunk_size=max_tokens,
        chunk_overlap=0,
        length_function=lambda x: len(tokenizer.tokenize(x)),
    )

    chunks = text_splitter.create_documents(texts)
    return [e.page_content for e in chunks]
