"""Predictions -> Kaggle submission CSV (+ format check).

Usage (from code/):
    python scripts/make_submission.py --pred outputs/<exp>/test_preds.json --out outputs/<exp>/submission.csv
    python scripts/make_submission.py --out outputs/submission_empty.csv   # no --pred: empty submission
"""
import argparse
import json

from src.config import CODE_DIR
from src.submission import check_submission, predictions_to_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", default=None, help="COCO results json with test image_id strings")
    parser.add_argument("--test-csv", default=str(CODE_DIR / "data/annotations/test_images.csv"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    preds = args.pred if args.pred else []
    n = predictions_to_csv(preds, args.test_csv, args.out)
    stats = check_submission(args.out, args.test_csv)
    print(f"wrote {n} rows -> {args.out}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
