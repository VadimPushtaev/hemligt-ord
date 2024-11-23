import argparse
import sys
from typing import Generator


def _unimorph_formatter() -> Generator[str, None, None]:
    for line in sys.stdin:
        line = line.strip()

        word, form, tags = line.split("\t")
        tags_set = set(tags.split(";"))

        if "N" in tags_set and "NOM" in tags_set and "SG" in tags_set and "INDF" in tags_set:
            yield word


def _get_formatter(from_format: str) -> Generator[str, None, None]:
    if from_format == "unimorph":
        return _unimorph_formatter()
    else:
        raise ValueError(f"Unknown format: {from_format}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-format", required=True, type=str)
    args = parser.parse_args()

    formatter = _get_formatter(from_format=args.from_format)
    for line in formatter:
        print(line)


if __name__ == '__main__':
    main()
