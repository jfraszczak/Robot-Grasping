import argparse


def get_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument(
        "--images_path",
        type=str,
        help="Path to input images directory"
    )
    parser.add_argument(
        "--aruco_markers",
        action="store_true",
        help="Use AruCo markers for pose estimation"
    )
    parser.add_argument(
        "--verbose",
        action="store_true"
    )
    return parser.parse_args()
