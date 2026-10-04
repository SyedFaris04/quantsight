/**
 * frontend/src/hooks/useApi.js
 * ─────────────────────────────────────────────────────────────────
 * Central API hook — all calls to the FastAPI backend go through here.
 *
 * Usage in any component:
 *   const { data, loading, error } = useApi("/overview");
 *
 * One-time fetch (no auto-refetch):
 *   const { data, loading, error } = useApi("/explain/AAPL");
 *
 * The base URL is read from the .env file:
 *   VITE_API_URL=https://nuroquant-api.onrender.com
 *
 * During local development with the Vite proxy, just set:
 *   VITE_API_URL=/api
 * ─────────────────────────────────────────────────────────────────
 */

import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import axios from "axios";
import { createLatestRequest } from "./latestRequest";

const BASE_URL = import.meta.env.VITE_API_URL || "/api";

// Shared axios instance — base URL + default timeout
const api = axios.create({
  baseURL: BASE_URL,
  timeout: 30000,   // 30s — Render free tier can be slow on cold start
});

/**
 * useApi — fetches data from a FastAPI endpoint on mount.
 *
 * @param {string|null} endpoint  - e.g. "/overview" or "/explain/AAPL"
 *                                  pass null to skip fetching
 * @param {any[]}       deps      - extra dependencies that trigger a refetch
 * @returns {{ data, loading, error, isSlow, refetch }}
 */
export function useApi(endpoint, deps = []) {
  const [state, setState] = useState({ endpoint, data: null, loading: !!endpoint, error: null, isSlow: false });
  const activeFetcher = useRef(null);
  const runner = useMemo(() => createLatestRequest(
    (url, options) => api.get(url, options),
    patch => setState(previous => ({ ...previous, ...patch })),
  ), []);
  // Effect changes, unmounts and manual retries all invalidate earlier requests.
  // A sequence guard also protects against transports that finish after abort.
  const refetch = useCallback(function fetchCurrent() {
    // An async callback retained by an old page must not restart its request.
    if (activeFetcher.current === fetchCurrent) return runner.run(endpoint);
  }, [runner, endpoint, ...deps]);
  useEffect(() => {
    activeFetcher.current = refetch;
    refetch();
    return () => { activeFetcher.current = null; runner.cancel(); };
  }, [refetch, runner]);
  // Do not briefly display a previous ticker's data before the new effect runs.
  const visible = state.endpoint === endpoint ? state
    : { data: null, error: null, loading: !!endpoint, isSlow: false };
  return { data: visible.data, loading: visible.loading, error: visible.error, isSlow: visible.isSlow, refetch };
}

/**
 * postApi — sends a POST request to the FastAPI backend.
 * Used by the game answer submission.
 *
 * @param {string} endpoint
 * @param {object} body
 * @returns {Promise<any>}
 */
export async function postApi(endpoint, body) {
  const res = await api.post(endpoint, body);
  return res.data;
}

export default api;
