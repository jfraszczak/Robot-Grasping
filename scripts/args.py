import argparse


def get_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose",
        action="store_true"
    )
    return parser.parse_args()
