import argparse
import json

from .pipeline import train


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--output", default="results/telco")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    print(json.dumps(train(a.csv, a.output, a.seed), indent=2))


if __name__ == "__main__":
    main()
