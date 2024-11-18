import base64
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Self

import numpy as np
from numpy.typing import NDArray


@dataclass
class Embedding:
    data: NDArray[np.float64]

    def to_str(self) -> str:
        return base64.b64encode(self.data.tobytes()).decode("utf-8")

    @classmethod
    def from_str(cls, string: str) -> Self:
        return cls(data=np.frombuffer(base64.b64decode(string), dtype=np.float64))

    def distance(self, other: "Embedding") -> float:
        vec1 = self.data
        vec2 = other.data

        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_a * norm_b)
        cosine_distance = 1 - cosine_similarity
        return cosine_distance


class EmbeddingDB:
    def __init__(self, *, db_path: Path) -> None:
        self._db_path = db_path

        self._current_json_name: str | None = None
        self._current_json_data: dict[str, str] | None = None

        self._db_path.mkdir(parents=True, exist_ok=True)

    def _get_json_by_word(self, *, word: str) -> str | None:
        first_letter = word[0].lower()

        return self._db_path / f"{first_letter}.json"

    def _load_json(self, *, json_name: str) -> None:
        if self._current_json_name != json_name:
            self.flush()

            self._current_json_name = json_name

            json_path = Path(self._db_path) / json_name
            if json_path.exists():
                with open(json_name, "r") as f:
                    self._current_json_data = json.load(f)
            else:
                self._current_json_data = {}

    def flush(self) -> None:
        logging.info("Flushing DB")
        if self._current_json_name is not None and self._current_json_data is not None:
            with open(self._current_json_name, "w") as f:
                json.dump(self._current_json_data, f)

    def get_embedding(self, *, word: str) -> Embedding | None:
        json_name = self._get_json_by_word(word=word)
        self._load_json(json_name=json_name)

        emb = self._current_json_data.get(word)

        if emb is None:
            return None
        else:
            return Embedding.from_str(emb)

    def set_embedding(self, *, word: str, embedding: Embedding) -> None:
        json_name = self._get_json_by_word(word=word)
        self._load_json(json_name=json_name)

        self._current_json_data[word] = embedding.to_str()

    def get_all(self) -> Generator[tuple[str, Embedding], None, None]:
        for json_path in self._db_path.glob("*.json"):
            with open(json_path, "r") as f:
                data = json.load(f)

            for word, emb in data.items():
                yield word, Embedding.from_str(emb)

    def __del__(self) -> None:
        self.flush()
