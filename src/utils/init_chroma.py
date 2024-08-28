import functools
from typing import Union

import chromadb
import ollama
from chromadb.api.types import EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class OllamaEmbed(EmbeddingFunction):
    def __call__(self, input: Union[str, list[str]]) -> Embeddings:
        if isinstance(input, str):
            input = [input]
        embeddings = []
        for e in tqdm(input, desc="Generating embeddings"):
            embeddings.append(
                ollama.embeddings(model="mxbai-embed-large", prompt=e)["embedding"]
            )
        return embeddings


class SentenceTransformerEmbed(EmbeddingFunction):
    model: str = ".models/multilingual-e5-large"

    def __call__(self, input_texts: Union[str, list[str]]) -> Embeddings:
        if isinstance(input_texts, str):
            input_texts = [input_texts]
        for text in input_texts:
            assert text.startswith("passage: ") or text.startswith(
                "query: "
            ), "Text must start with 'passage: ' or 'query: '"
        embeddeng_model = SentenceTransformer(self.model)
        embeddings = embeddeng_model.encode(
            input_texts,
            normalize_embeddings=True,
            device="mps",
            convert_to_numpy=False,
            show_progress_bar=True,
        )
        # convert to list of lists, current type is list of tensors
        if isinstance(embeddings, list):
            return [e.tolist() for e in embeddings]  # type: ignore
        return list(embeddings)  # type: ignore


init_client = lambda: chromadb.PersistentClient(path=".chroma")


init_collection = functools.partial(
    init_client().get_or_create_collection,
    embedding_function=SentenceTransformerEmbed(),
)
