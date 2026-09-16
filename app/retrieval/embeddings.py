"""Dense-эмбеддинги через sentence-transformers.

По умолчанию BAAI/bge-m3: мультиязычная, 8192 токена контекста (чанки до 800 токенов не обрезаются),
префиксы не нужны. Для семейства e5 префиксы обязательны: без «query:» / «passage:» качество
поиска заметно падает, потому что модель так обучалась.
"""

from collections.abc import Sequence
from typing import Protocol

_PREFIXES = {
    # подстрока в имени модели -> (префикс запроса, префикс документа)
    "e5": ("query: ", "passage: "),
}


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...

    def token_spans(self, text: str) -> list[tuple[int, int]]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 16):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_embedding_dimension()
        self.batch_size = batch_size
        self.query_prefix, self.passage_prefix = next(
            (p for key, p in _PREFIXES.items() if key in model_name.lower()), ("", "")
        )

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self.model.encode(
            [self.passage_prefix + t for t in texts],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(self.query_prefix + text, normalize_embeddings=True).tolist()

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        encoded = self.model.tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True, verbose=False
        )
        return [tuple(span) for span in encoded["offset_mapping"]]
