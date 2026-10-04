"""Prepare or recompute-check every latest saved native TreeSHAP explanation.

From backend: python -m research.build_attribution_snapshots [--check]
This never fits a model and never reads the separate prospective holdout.
"""
import argparse

from attribution_snapshot import load_attributions, write_attributions
from serving_snapshot import sha256_file
import feature_attribution as attribution


def run(variant, check=False):
    sources = attribution._paths(variant)
    before = {path.name: sha256_file(path) for path in sources}
    attribution._latest_evidence.cache_clear()
    _, _, latest, _ = attribution._latest_evidence(variant, attribution._signatures(variant))
    rows = [attribution.get_native_attribution(ticker, variant) for ticker in sorted(latest.index)]
    if before != {path.name: sha256_file(path) for path in sources}:
        raise ValueError("Attribution sources changed during preparation.")
    if check:
        saved = load_attributions(variant, sources, attribution.DATA / "serving")
        if set(saved) != set(latest.index) or any(saved[row["ticker"]] != row for row in rows):
            raise AssertionError("Prepared explanations differ from native model computation.")
        print(f"Verified {variant}: {len(rows)} native explanations and exact historical inputs")
    else:
        write_attributions(variant, sources, rows, attribution.DATA / "serving")
        print(f"Prepared {variant}: {len(rows)} native explanations")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for variant in attribution.VARIANTS:
        run(variant, args.check)


if __name__ == "__main__":
    main()
