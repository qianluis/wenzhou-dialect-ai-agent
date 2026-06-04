from __future__ import annotations

import argparse


def levenshtein(a: str, b: str) -> int:
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, start=1):
            old = dp[j]
            dp[j] = min(
                dp[j] + 1,
                dp[j - 1] + 1,
                prev + (ca != cb)
            )
            prev = old
    return dp[-1]


def cer(pred: str, ref: str) -> float:
    ref = ref.strip()
    pred = pred.strip()
    if not ref:
        return 0.0 if not pred else 1.0
    return levenshtein(pred, ref) / len(ref)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--ref", required=True)
    args = parser.parse_args()
    score = cer(args.pred, args.ref)
    print(f"CER={score:.4f}")


if __name__ == "__main__":
    main()
