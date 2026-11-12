"""Predictions -> Kaggle submission CSV.

Usage:
    python make_submission.py --pred <preds.json> --out ../outputs/submission.csv
"""
import argparse

from src.submission import predictions_to_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    predictions_to_csv(args.pred, args.out)


if __name__ == "__main__":
    main()
