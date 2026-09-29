"""Serve aggregate research evidence; never load an NLP model in the API."""
import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from research.development_report import validate_report

REPORT = Path(__file__).resolve().parents[1] / 'data/research_reports/news_comparison.json'
DEVELOPMENT_REPORT = REPORT.with_name('development_models.json')
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


def load_development_report():
    if not DEVELOPMENT_REPORT.exists():
        return None
    try:
        return validate_report(json.loads(DEVELOPMENT_REPORT.read_text(encoding='utf-8')))
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
        raise HTTPException(503, 'Development comparison is unavailable or invalid.') from exc


@router.get('/development-models')
def development_models():
    report = load_development_report()
    return {'available': report is not None, **({'report': report} if report else {})}


@router.get('/development-models/export')
def export_development_models():
    report = load_development_report()
    if report is None:
        raise HTTPException(404, 'Development report is unavailable.')
    return Response(json.dumps(report, indent=2, allow_nan=False), media_type='application/json',
                    headers={'Content-Disposition': 'attachment; filename="quantsight-development-models.json"'})
