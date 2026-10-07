"""Read-only smoke check of the real serving app and repository-owned assets.

Run from backend: python -m research.verify_runtime
No provider requests, database writes or model fitting are performed.
"""
import hashlib
import json
import sys
import time


def working_set_mb():
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Memory(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in
                ("peak_ws", "ws", "peak_pool", "pool", "peak_nonpool", "nonpool",
                 "pagefile", "peak_pagefile")]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Memory), wintypes.DWORD]
        memory = Memory()
        memory.cb = ctypes.sizeof(memory)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(memory), memory.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return round(memory.ws / 1048576, 1)
    # Current resident pages, not peak RSS. Linux CI / Render.
    from pathlib import Path
    import os
    pages = int(Path("/proc/self/statm").read_text().split()[1])
    return round(pages * os.sysconf("SC_PAGE_SIZE") / 1048576, 1)


def main():
    started = time.perf_counter()
    import main as api
    import_seconds = time.perf_counter() - started
    from fastapi.testclient import TestClient
    from unittest.mock import patch

    # A process that has not loaded data must not advertise readiness.
    client = TestClient(api.app)
    assert client.get("/health/ready").status_code == 503
    assert client.get("/health/live").status_code == 200

    started = time.perf_counter()
    with TestClient(api.app) as client:
        report = {"import_seconds": round(import_seconds, 3),
                  "startup_seconds": round(time.perf_counter() - started, 3),
                  "startup_working_set_mb": working_set_mb(),
                  "startup_heavy_modules": [name for name in
                      ("xgboost", "sklearn", "torch", "transformers", "shap") if name in sys.modules]}
        health = client.get("/")
        assert health.status_code == 200, health.text
        assert health.json()["tickers"] == 44
        ready = client.get("/health/ready")
        assert ready.json()["ready"] is True
        assert ready.json()["external_services"] == "not_checked"
        assert ready.headers["cache-control"] == "no-store"
        hashes = {}
        import copilot_engine as copilot
        import pandas as pd
        original_reader = pd.read_csv

        def serving_reader(path, *args, **kwargs):
            from pathlib import Path
            if Path(path) == copilot.FEATURE_FILES["sentiment"] and kwargs.get("usecols") is None:
                raise AssertionError("Serving must use the compact latest sentiment snapshot")
            return original_reader(path, *args, **kwargs)

        with patch.object(copilot.pd, "read_csv", side_effect=serving_reader):
            for path in ("/dashboard", "/overview", "/stock/AAPL", "/explain/AAPL",
                         "/compare/AAPL", "/history/AAPL", "/accuracy-history/AAPL",
                         "/research/development-models", "/research/development-confidence",
                         "/research/xai-comparison", "/backtest/summary"):
                response = client.get(path)
                assert response.status_code == 200, (path, response.text)
                payload = json.dumps(response.json(), sort_keys=True, separators=(",", ":"))
                hashes[path] = hashlib.sha256(payload.encode()).hexdigest()
        for variant in ("xgb_finance", "xgb_sentiment"):
            response = client.get("/feature-attribution/AAPL", params={"model": variant})
            assert response.status_code == 200, response.text
            assert response.json()["verified"] is True
        assert not any(name in sys.modules for name in ("xgboost", "sklearn")), \
            "Prepared attribution serving must not import model libraries"
        assert not any(name in sys.modules for name in ("torch", "transformers", "shap", "datasets"))
        report.update(response_hashes=hashes, final_working_set_mb=working_set_mb(),
                      saved_xai_verified=True)
        # Compare every compact latest row with the original full CSV lookup.
        for sentiment, path in ((False, copilot.FEATURE_FILES["finance"]),
                                (True, copilot.FEATURE_FILES["sentiment"])):
            reference = pd.read_csv(path)
            for ticker in api._cache["tickers"]:
                expected = reference[reference.ticker == ticker].sort_values("date").iloc[-1]
                pd.testing.assert_series_equal(copilot.get_latest_features(ticker, sentiment),
                                               expected, check_names=False, check_exact=True)
    # Corrupt optional news does not stop serving required assets.
    predictions, features = api._cache["predictions"], api._cache["features"]
    with patch.object(api, "load_predictions", return_value=predictions), \
         patch.object(api, "load_features", return_value=features), \
         patch.object(api, "load_news", side_effect=ValueError("invalid parquet")):
        with TestClient(api.app) as client:
            assert client.get("/health/ready").status_code == 200
            assert client.get("/health/ready").json()["datasets"]["news"]["available"] is False
    with patch.object(api, "load_predictions", return_value={}), \
         patch.object(api, "load_features", return_value=features), \
         patch.object(api, "load_news", return_value=features.iloc[:0]):
        try:
            with TestClient(api.app):
                raise AssertionError("Missing predictions must fail startup")
        except RuntimeError as exc:
            assert str(exc) == "Required saved data failed readiness checks"
        assert TestClient(api.app).get("/health/ready").status_code == 503
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
