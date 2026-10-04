/** One active GET per hook, including retries. Cancelled work cannot publish state. */
export function describeApiError(error) {
  const status = error.response?.status;
  if (status === 429) return "The service is receiving too many requests. Please wait before trying again.";
  if ([502, 503, 504].includes(status)) return "The service is temporarily unavailable. Please try again shortly.";
  if (["ECONNABORTED", "ETIMEDOUT"].includes(error.code)) {
    return "The server did not respond within 30 seconds. It may still be starting or busy. Please try again.";
  }
  if (error.code === "ERR_NETWORK") return "Could not reach the server. Check your connection and try again.";
  const detail = error.response?.data?.detail;
  if (typeof detail === "string" && detail.trim()) return detail;
  if (Array.isArray(detail)) return "Some request details were not accepted. Check your selection and try again.";
  return "The request could not be completed. Please try again.";
}

export function createLatestRequest(request, publish, slowAfter = 8000) {
  let generation = 0;
  let controller;
  let slowTimer;

  function cancel() {
    generation += 1;
    controller?.abort();
    clearTimeout(slowTimer);
  }

  async function run(endpoint) {
    cancel();
    const id = generation;
    if (!endpoint) {
      publish({ endpoint, data: null, error: null, loading: false, isSlow: false });
      return;
    }
    const current = new AbortController();
    controller = current;
    const active = () => id === generation && !current.signal.aborted;
    publish({ endpoint, data: null, error: null, loading: true, isSlow: false });
    const timer = setTimeout(() => { if (active()) publish({ isSlow: true }); }, slowAfter);
    slowTimer = timer;
    try {
      const response = await request(endpoint, { signal: current.signal });
      if (active()) publish({ data: response.data });
    } catch (error) {
      if (active() && error.code !== "ERR_CANCELED") publish({ error: describeApiError(error) });
    } finally {
      clearTimeout(timer);
      if (active()) publish({ loading: false, isSlow: false });
    }
  }

  return { run, cancel };
}
