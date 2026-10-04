"""Offline, reproducible period returns derived from an existing backtest run.

Run from backend: python -m backtesting.periods [--check]
No fitting, price downloads, trade simulation or holdout access.
"""
import argparse
import csv
from datetime import date
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

OUTPUT = Path(__file__).resolve().parents[1] / "data/backtests"
FILENAME = "period_returns.json"


def content_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@lru_cache(maxsize=64)
def _text_digest(path, modified_ns, size):
    # Hash identical text equally on Windows and Linux.
    return hashlib.sha256(Path(path).read_text(encoding="utf-8").encode()).hexdigest()


def text_digest(path):
    path = Path(path)
    stat = path.stat()
    return _text_digest(str(path), stat.st_mtime_ns, stat.st_size)


def inside(output, relative):
    path = (Path(output) / relative).resolve()
    if not path.is_relative_to(Path(output).resolve()):
        raise ValueError("Artifact path is outside the backtest directory")
    return path


def daily_path(output, report, key):
    if not key or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in key):
        raise ValueError("Invalid strategy key")
    return inside(output, f"{report['run_id']}/{key}_daily.csv")


def read_daily(path, report, strategy):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    evaluation = report["evaluation"]
    if len(rows) != evaluation["sessions"] or not rows:
        raise ValueError("Daily series length differs from the saved evaluation")
    dates = [row["date"] for row in rows]
    if dates != sorted(set(dates)) or any(date.fromisoformat(d).isoformat() != d for d in dates):
        raise ValueError("Daily dates must be unique and ordered ISO dates")
    if dates[0] != evaluation["start"] or dates[-1] != evaluation["end"]:
        raise ValueError("Daily dates differ from the saved evaluation")
    curve = strategy["curve"]
    previous = float(report["config"]["initial_capital"])
    if not math.isfinite(previous) or previous <= 0 or len(curve) != len(rows) + 1:
        raise ValueError("Invalid starting equity or saved curve")
    for row, saved in zip(rows, curve[1:]):
        equity, ret = float(row["equity"]), float(row["return"])
        if not math.isfinite(equity) or not math.isfinite(ret) or equity <= 0 or ret <= -1:
            raise ValueError("Invalid equity or daily return")
        if (saved["date"] != row["date"] or
                not math.isclose(equity, saved["equity"], rel_tol=1e-10, abs_tol=1e-6) or
                not math.isclose(ret, equity / previous - 1, rel_tol=1e-9, abs_tol=1e-12)):
            raise ValueError("Daily accounting does not reconcile with the saved curve")
        row["return"] = ret
        previous = equity
    total = math.prod(1 + row["return"] for row in rows) - 1
    if not math.isclose(total, strategy["metrics"]["total_return"], rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("Compounded returns differ from the saved total return")
    return rows


def aggregate(rows, length):
    groups = {}
    for row in rows:
        groups.setdefault(row["date"][:length], []).append(row)
    return [{"period": key, "start": group[0]["date"], "end": group[-1]["date"],
             "sessions": len(group), "net_return": math.prod(1 + row["return"] for row in group) - 1}
            for key, group in groups.items()]


def build(output=OUTPUT):
    output = Path(output)
    report_path = output / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["schema_version"] != 1:
        raise ValueError("Unsupported backtest schema")
    daily, sources = {}, {"report.json": text_digest(report_path)}
    for strategy in report["strategies"]:
        key = strategy["key"]
        path = daily_path(output, report, key)
        if key in daily:
            raise ValueError("Duplicate strategy")
        daily[key] = read_daily(path, report, strategy)
        sources[path.relative_to(output.resolve()).as_posix()] = text_digest(path)
    for rows in daily.values():
        if [r["date"] for r in rows] != [r["date"] for r in daily["spy"]]:
            raise ValueError("Strategy and benchmark sessions differ")
    strategies = {}
    for key, rows in daily.items():
        periods = {}
        for label, length in [("years", 4), ("months", 7)]:
            selected = aggregate(rows, length)
            spy, universe = aggregate(daily["spy"], length), aggregate(daily["universe"], length)
            for row, s, u in zip(selected, spy, universe):
                row.update(spy_return=s["net_return"], universe_return=u["net_return"],
                           difference_vs_spy=row["net_return"] - s["net_return"])
            periods[label] = selected
        months = periods["months"]
        periods["summary"] = {
            "months": len(months),
            "months_beating_spy": sum(r["difference_vs_spy"] > 1e-12 for r in months),
            "months_tied_spy": sum(abs(r["difference_vs_spy"]) <= 1e-12 for r in months),
            "positive_months": sum(r["net_return"] > 0 for r in months),
            "best_month": max(months, key=lambda r: r["net_return"])["period"],
            "worst_month": min(months, key=lambda r: r["net_return"])["period"],
        }
        strategies[key] = periods
    result = {"schema_version": 1, "run_id": report["run_id"],
              "backtest_generated_at": report["generated_at"],
              "methodology": "Geometrically compounded saved net daily returns; no boundary rebalancing or annualization. Dates are observed study sessions, including partial months/years. Descriptive comparisons, not prediction accuracy or statistical significance.",
              "sources": sources, "builder_sha256": text_digest(__file__), "strategies": strategies}
    result["content_sha256"] = content_digest(result)
    return result


def load(output, report):
    """Serve only a checked, source-matching sidecar; leave the original report usable."""
    path = Path(output) / FILENAME
    if not path.is_file():
        return {"available": False, "reason": "missing"}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        checksum = data.pop("content_sha256")
        if data["schema_version"] != 1 or checksum != content_digest(data):
            raise ValueError("Invalid period report")
        expected = {"report.json"} | {
            daily_path(output, report, s["key"]).relative_to(Path(output).resolve()).as_posix()
            for s in report["strategies"]}
        if (data["run_id"] != report["run_id"] or set(data["sources"]) != expected or
                data["builder_sha256"] != text_digest(__file__) or
                set(data["strategies"]) != {s["key"] for s in report["strategies"]} or
                any(text_digest(inside(output, p)) != digest for p, digest in data["sources"].items())):
            return {"available": False, "reason": "stale"}
        data["content_sha256"] = checksum
        return {"available": True, "report": data}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return {"available": False, "reason": "invalid"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = build(args.output)
    path = args.output / FILENAME
    if args.check:
        if json.loads(path.read_text(encoding="utf-8")) != result:
            raise SystemExit("Period report does not match the saved run; regenerate it.")
        print(f"Verified period returns for {len(result['strategies'])} strategies.")
    else:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        temporary.replace(path)
        print(f"Published {path}")


if __name__ == "__main__":
    main()
