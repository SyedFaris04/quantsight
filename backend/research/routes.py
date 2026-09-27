"""Serve aggregate research evidence; never load an NLP model in the API."""
import json
from pathlib import Path
from fastapi import APIRouter, HTTPException

REPORT = Path(__file__).resolve().parents[1] / 'data/research_reports/news_comparison.json'
router = APIRouter(prefix='/research', tags=['Research'])


@router.get('/news-comparison')
def news_comparison():
    if not REPORT.exists():
        return {'available': False}
    try:
        report = json.loads(REPORT.read_text(encoding='utf-8'))
        if report['schema_version'] != 1 or report['status'] != 'exploratory' or not report['models']:
            raise ValueError('Invalid research report')
        return {'available': True, 'report': report}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(503, 'Research report is unavailable or invalid.') from exc
