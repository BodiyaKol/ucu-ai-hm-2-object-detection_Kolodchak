"""M1: convert raw VisDrone-VID -> YOLO / COCO, subsample frames, mask ignore regions.

Usage:
    python prepare_data.py --protocol ../configs/protocol.yaml
"""
import argparse

from src.config import load_config, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="../configs/protocol.yaml")
    args = parser.parse_args()

    cfg = load_config(args.protocol)
    set_seed(cfg["seed"])

    # TODO:
    # 1. read raw annotations per clip (src.data.convert)
    # 2. subsample frames by cfg["data"]["frame_step"]
    # 3. mask / record ignore regions (src.data.ignore)
    # 4. write YOLO labels + COCO json for train and val
    raise NotImplementedError


if __name__ == "__main__":
    main()
