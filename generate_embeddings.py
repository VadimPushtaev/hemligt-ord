import argparse
import base64
import logging
import os
from pathlib import Path

import numpy as np
import openai

from db import EmbeddingDB, Embedding

CURRENT_DIR = Path(__file__).parent


def generate_embedding(*, client: openai.Client, word: str) -> Embedding:
    # Generate embedding using the new API
    response = client.embeddings.create(
        input=word,
        model="text-embedding-ada-002"
    )
    # Extract the embedding vector
    embedding = response.data[0].embedding
    return Embedding(data=np.array(embedding))


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--limit", required=False, type=int, default=None)
    args = arg_parser.parse_args()

    words_txt = CURRENT_DIR / "words.txt"
    db = EmbeddingDB(db_path=CURRENT_DIR / "embeddings")
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    i = 0
    with open(words_txt, "r") as f:
        for word in sorted(f):
            word = word.strip()
            if not word:
                continue

            if db.get_embedding(word=word) is not None:
                logging.info(f"Embedding for word {word} already exists")
                continue

            embedding = generate_embedding(client=client, word=word)
            if embedding is not None:
                db.set_embedding(word=word, embedding=embedding)
                logging.info(f"Embedding for word {word} generated")
            else:
                raise ValueError(f"Error generating embedding for word: {word}")

            i += 1
            if args.limit is not None and i >= args.limit:
                break

    db.flush()


if __name__ == '__main__':
    main()
