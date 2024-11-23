import logging
import os

import numpy as np
import openai

from db import EmbeddingDB, Embedding


class EmbeddingGenerator:
    def __init__(self, *, db: EmbeddingDB, logger: logging.Logger, batch_size: int) -> None:
        self._db = db
        self._logger = logger
        self._batch_size = batch_size

        self._client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        self._batch = []

    def generate_and_set(self, *, word: str) -> None:
        if len(self._batch) >= self._batch_size:
            self.flush()
        else:
            self._batch.append(word)

    def flush(self) -> None:
        if not self._batch:
            return

        embeddings = self._generate_embeddings_for_batch()

        for word, embedding in zip(self._batch, embeddings):
            if embedding is not None:
                self._db.set_embedding(word=word, embedding=embedding)
                self._logger.info(f"[+] Embedding for word {word} generated")
            else:
                raise ValueError(f"Error generating embedding for word: {word}")

        self._batch = []

    def _generate_embeddings_for_batch(self) -> list[Embedding]:
        response = self._client.embeddings.create(
            input=self._batch,
            model="text-embedding-ada-002"
        )

        return [
            Embedding(data=np.array(d.embedding))
            for d in response.data
        ]
