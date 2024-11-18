import argparse
from pathlib import Path

from db import EmbeddingDB, Embedding


def print_sorted(word: str) -> None:
    db = EmbeddingDB(db_path=Path(__file__).parent / "embeddings")
    all_words: dict[str, Embedding] = {word: emb for word, emb in db.get_all()}

    if word not in all_words:
        raise ValueError(f"Word {word} not found in the database")

    root_embedding = all_words[word]

    for word, emb in sorted(all_words.items(), key=lambda x: root_embedding.distance(x[1])):
        print(f"{word}: {root_embedding.distance(emb)}")


def main() -> None:
    # argparse here , get --word

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--word", required=True, type=str)
    args = arg_parser.parse_args()

    print_sorted(word=args.word)



if __name__ == '__main__':
    main()