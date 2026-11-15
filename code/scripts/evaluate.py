"""COCO-style evaluation: predictions json -> mAP / AP50 / AP_small (+ optional FPS).

Usage:
    python evaluate.py --gt <coco_gt.json> --pred <preds.json> --out ../results/<exp_id>.json
"""
import argparse
import json

from src.eval.coco_eval import coco_evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    metrics = coco_evaluate(args.gt, args.pred, verbose=True)    with open(args.out, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
