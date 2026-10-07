"""Standard-library GitHub runner; never logs raw HTTP bodies or credentials.

Retries wait beyond the backend's 60-second failed-panel cache. The backend
owns forecast timing; this runner never supplies a synthetic forecast date.
"""
import json
import http.client
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

PROTOCOL = "nyse-close-5-v2"
RETRY_HTTP = {408, 409, 429, 500, 502, 503, 504}
STAGES = {"resolution", "inference", "recording", "coverage"}
CODES = {"exact_prices_missing", "resolution_failed", "market_panel_unavailable",
         "ticker_unavailable", "stale_session", "inference_failed",
         "required_features_missing", "model_unavailable", "calibrator_unavailable",
         "signal_contract_failed", "signal_unavailable", "recording_failed",
         "database_read_failed", "invalid_probability", "invalid_price"}
PERMANENT = {"model_unavailable", "calibrator_unavailable",
             "signal_contract_failed", "invalid_probability", "invalid_price"}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward the admin header to a redirected destination.
        return None


class HTTPTransport:
    def __init__(self, base_url, secret):
        parsed = urllib.parse.urlsplit(base_url)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username or
                parsed.password or parsed.query or parsed.fragment):
            raise ValueError("Invalid backend URL configuration")
        self.base_url, self.secret = base_url.rstrip("/"), secret
        self.opener = urllib.request.build_opener(NoRedirect())

    def __call__(self, path, method="GET", timeout=45):
        headers = {"Accept": "application/json", "User-Agent": "QuantSight-collector-v1"}
        if method == "POST":
            headers["X-Admin-Key"] = self.secret
        request = urllib.request.Request(self.base_url + path, method=method, headers=headers)
        try:
            response = self.opener.open(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            response = exc
        except (OSError, http.client.HTTPException):
            return 0, None
        with response:
            status = response.code
            try:
                body = json.loads(response.read(256_001))
            except (ValueError, OSError, http.client.HTTPException):
                body = None
        return status, body


def safe_result(body):
    """Only validated contract fields enter logs and the Actions summary."""
    if not isinstance(body, dict) or body.get("protocol_version") != PROTOCOL:
        raise ValueError("Unexpected collector response")
    result = {"protocol_version": PROTOCOL}
    for key in ("logged", "resolved", "duplicates", "skipped", "tickers_processed"):
        value = body.get(key)
        if type(value) is not int or value < 0:
            raise ValueError("Invalid collector counts")
        result[key] = value
    expected = result["tickers_processed"]
    coverage = body.get("cohort_logged")
    result["cohort_logged"] = coverage
    if expected == 0 or (coverage is not None and (type(coverage) is not int or coverage < 0 or coverage > expected)):
        raise ValueError("Invalid collector coverage")
    missing = body.get("missing_tickers")
    if coverage is None:
        if missing is not None or body.get("status") != "partial_failure":
            raise ValueError("Unverified collector coverage")
    elif (not isinstance(missing, list) or any(not isinstance(t, str) or
            not re.fullmatch(r"[A-Z0-9.^_-]{1,15}", t) for t in missing) or
            len(set(missing)) != len(missing) or len(missing) != expected - result["cohort_logged"]):
        raise ValueError("Invalid missing-ticker count")
    result["missing_tickers"] = missing
    if body.get("status") not in {"ok", "skipped", "partial_failure"}:
        raise ValueError("Invalid collector status")
    result["status"] = body["status"]
    for key in ("forecast_date", "record_before"):
        value = body.get(key)
        pattern = r"\d{4}-\d{2}-\d{2}" if key == "forecast_date" else r"[0-9T:+.Z-]{20,40}"
        if not isinstance(value, str) or not re.fullmatch(pattern, value):
            raise ValueError("Invalid collector timing metadata")
        result[key] = value
    details, errors = body.get("error_details"), body.get("errors")
    if (not isinstance(details, list) or not isinstance(errors, list) or len(details) != len(errors)
            or (bool(errors) != (body["status"] == "partial_failure"))):
        raise ValueError("Invalid collector errors")
    result["error_details"] = []
    for detail in details:
        if (not isinstance(detail, dict) or detail.get("stage") not in STAGES or
                detail.get("code") not in CODES or not isinstance(detail.get("ticker"), str) or
                not re.fullmatch(r"[A-Z0-9.^_-]{1,15}", detail["ticker"])):
            raise ValueError("Invalid error category")
        result["error_details"].append({k: detail[k] for k in ("ticker", "stage", "code")})
    if coverage is None and not any(d["stage"] == "coverage" and d["code"] == "database_read_failed" for d in result["error_details"]):
        raise ValueError("Missing coverage failure category")
    return result


def collect(transport, sleep=time.sleep, emit=lambda report: None):
    report = {"runner_version": 1, "readiness": [], "attempts": [], "success": False}
    for attempt in range(4):
        status, body = transport("/health/ready")
        ready = status == 200 and isinstance(body, dict) and body.get("ready") is True
        report["readiness"].append({"http_status": status, "ready": ready})
        emit(report)
        if ready:
            release = body.get("release", "unknown")
            report["release"] = release if isinstance(release, str) and re.fullmatch(r"[a-f0-9]{40}", release) else "unknown"
            break
        if status not in RETRY_HTTP | {0, 200}:
            report["failure"] = "readiness_http_error"
            return report
        if attempt < 3:
            sleep(15)
    else:
        report["failure"] = "backend_not_ready"
        return report
    for attempt in range(3):
        status, body = transport("/admin/run-daily-predictions", method="POST", timeout=240)
        entry = {"http_status": status}
        report["attempts"].append(entry)
        if status not in {200, 503}:
            entry["failure"] = "authentication_failed" if status == 403 else "http_or_transport_error"
            retry = status in RETRY_HTTP | {0}
        else:
            try:
                result = safe_result(body)
            except ValueError:
                entry["failure"] = "backend_unavailable" if status == 503 else "invalid_response_contract"
                retry = status == 503
            else:
                entry["result"] = result
                complete = result["cohort_logged"] == result["tickers_processed"]
                if status == 200 and complete and result["status"] in {"ok", "skipped"}:
                    report["success"] = True
                    emit(report)
                    return report
                entry["failure"] = "missed_window" if result["status"] == "skipped" else "incomplete_collection_or_resolution"
                retry = result["status"] != "skipped" and not any(d["code"] in PERMANENT for d in result["error_details"])
        emit(report)
        if not retry or attempt == 2:
            report["failure"] = entry["failure"]
            return report
        sleep(75)
    return report


def summary(report):
    lines = ["## Daily forward collection", "", f"- Result: {'passed' if report['success'] else 'failed'}",
             f"- Backend release: {report.get('release', 'unknown')}",
             f"- Readiness checks: {len(report['readiness'])}"]
    for i, attempt in enumerate(report["attempts"], 1):
        lines.append(f"- Attempt {i}: HTTP {attempt['http_status']}")
        result = attempt.get("result")
        if result:
            lines.extend([f"  - Forecast session: {result['forecast_date']}; deadline: {result['record_before']}",
                          f"  - Verified saved coverage: {result['cohort_logged']}/{result['tickers_processed']}",
                          f"  - New: {result['logged']}; existing: {result['duplicates']}; resolved: {result['resolved']}"])
            if result["missing_tickers"]:
                lines.append("  - Missing: " + ", ".join(result["missing_tickers"]))
            for detail in result["error_details"]:
                lines.append(f"  - {detail['ticker']}: {detail['stage']} / {detail['code']}")
        if attempt.get("failure"):
            lines.append(f"  - Failure category: {attempt['failure']}")
    if report.get("failure"):
        lines.append(f"- Final failure category: {report['failure']}")
    lines.extend(["", "Retries preserve recorded forecasts. Missed dates are never backfilled.", ""])
    return "\n".join(lines)


def main():
    base, secret = os.getenv("BACKEND_URL", ""), os.getenv("ADMIN_KEY", "")
    if not base or not secret:
        print("Missing BACKEND_URL or ADMIN_KEY configuration.")
        return 1
    try:
        transport = HTTPTransport(base, secret)
    except ValueError:
        print("Invalid backend URL configuration; HTTPS is required.")
        return 1
    report = collect(transport, emit=lambda r: print(json.dumps(r), flush=True))
    print(json.dumps(report), flush=True)
    path = os.getenv("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(summary(report))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
