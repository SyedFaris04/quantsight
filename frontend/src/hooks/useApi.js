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

import { useState, useEffect, useCallback } from "react";
import axios from "axios";

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
 * @returns {{ data, loading, error, refetch }}
 */
export function useApi(endpoint, deps = []) {
  const [data,    setData]    = useState(null);
  const [loading, setLoading] = useState(!!endpoint);
  const [error,   setError]   = useState(null);

  const fetchData = useCallback(async () => {
    if (!endpoint) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.get(endpoint);
      setData(res.data);
    } catch (err) {
      const msg =
        err.response?.data?.detail ||
        err.message ||
        "Unknown error";
      setError(msg);
    } finally {
      setLoading(false);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [endpoint, ...deps]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { data, loading, error, refetch: fetchData };
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
