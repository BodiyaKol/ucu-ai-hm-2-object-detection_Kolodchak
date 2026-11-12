"""M1: dataset statistics + plots saved to ../images/.

TODO plots: class distribution, box size (area / w,h) histograms, objects per frame,
per-clip stats, k-means anchors, GT visualization examples.
"""
import argparse

from src.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="../configs/protocol.yaml")
    args = parser.parse_args()
    cfg = load_config(args.protocol)

    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    main()
