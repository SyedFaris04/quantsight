"""Serve precomputed evaluations; web requests never run research jobs."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "backend/data/backtests"
REPORT = OUTPUT / "report.json"
router = APIRouter(prefix="/backtest", tags=["Backtesting"])


@lru_cache(maxsize=64)
def _hash_file(path, modified_ns, size):
    # Stat arguments invalidate the cache. Git autocrlf changes Windows CSVs
    # to LF on Linux; accept either newline representation of identical text.
    # Content/column/value changes must still invalidate the research report.
    lf, crlf = hashlib.sha256(), hashlib.sha256()
    with Path(path).open("r", encoding="utf-8", newline=None) as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), ""):
            lf.update(chunk.encode("utf-8"))
            crlf.update(chunk.replace("\n", "\r\n").encode("utf-8"))
    return {lf.hexdigest(), crlf.hexdigest()}


def load_report():
    if not REPORT.exists():
        return None
    try:
        data = json.loads(REPORT.read_text(encoding="utf-8"))
        if data.get("schema_version") != 1 or not data.get("strategies"):
            raise ValueError("Unsupported report")
        stale = []
        for source in data["sources"]:
            path = (ROOT / source["path"]).resolve()
            if not path.is_relative_to(ROOT.resolve()):
                raise ValueError("Invalid source path")
            if not path.is_file():
                stale.append(source["path"])
                continue
            stat = path.stat()
            if source["sha256"] not in _hash_file(str(path), stat.st_mtime_ns, stat.st_size):
                stale.append(source["path"])
        data["freshness"] = {"matches_current_files": not stale, "changed_sources": stale}
        return data
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(503, "Backtest report is invalid or unreadable; regenerate it.") from exc


@router.get("")
def get_backtest():
    report = load_report()
    if report is None:
        return {"available": False, "message": "No backtest report has been generated yet."}
    return {"available": True, "report": report}


@router.get("/export")
def export_backtest(kind: str = Query("metrics", pattern="^(metrics|trades|daily|report)$"), strategy: str = "ensemble"):
    report = load_report()
    if report is None:
        raise HTTPException(404, "No report available")
    if strategy not in {s["key"] for s in report["strategies"]}:
        raise HTTPException(400, "Unknown strategy")
    if kind == "report":
        relative = f"{report['run_id']}/report.json"
    elif kind == "metrics":
        relative = report["exports"]["metrics"]
    else:
        relative = f"{report['run_id']}/{strategy}_{kind}.csv"
    path = (OUTPUT / relative).resolve()
    if not path.is_relative_to(OUTPUT.resolve()) or not path.is_file():
        raise HTTPException(404, "Export file unavailable")
    return FileResponse(path, filename=f"quantsight-{path.name}",
                        media_type="application/json" if kind == "report" else "text/csv")
