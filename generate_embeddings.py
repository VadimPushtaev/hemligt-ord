import argparse
import logging
from pathlib import Path


from db import EmbeddingDB
from embedding_generator import EmbeddingGenerator

CURRENT_DIR = Path(__file__).parent


def main() -> None:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--limit", required=False, type=int, default=None)
    arg_parser.add_argument("--batch-size", required=False, type=int, default=10)
    args = arg_parser.parse_args()

    words_txt = CURRENT_DIR / "words.txt"
    db = EmbeddingDB(db_path=CURRENT_DIR / "embeddings")
    embedding_generator = EmbeddingGenerator(db=db, logger=logger, batch_size=args.batch_size)

    i = 0
    with open(words_txt, "r") as f:
        for word in sorted(f):
            word = word.strip()
            if not word:
                continue

            if db.get_embedding(word=word) is not None:
                logger.info(f"[~] Embedding for word {word} already exists")
                continue

            embedding_generator.generate_and_set(word=word)

            i += 1
            if args.limit is not None and i >= args.limit:
                break

    embedding_generator.flush()
    db.flush()


if __name__ == '__main__':
    main()
