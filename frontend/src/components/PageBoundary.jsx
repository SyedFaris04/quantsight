import { Component } from "react";

export function PageLoading() {
  return <div className="card py-12 text-center" role="status" aria-live="polite">
    <div aria-hidden="true" className="mx-auto mb-4 h-6 w-6 rounded-full border-2 border-gray-700 border-t-indigo-400 animate-spin" />
    <p className="text-sm text-gray-300">Loading page…</p>
    <p className="mt-2 text-xs text-gray-500">You can still use the navigation menu.</p>
  </div>;
}

export default class PageBoundary extends Component {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  render() {
    if (this.state.failed) return <div className="card py-10" role="alert">
      <h1 className="text-lg font-semibold text-white">This page could not be displayed</h1>
      <p className="mt-3 max-w-xl text-sm leading-relaxed text-gray-400">
        A page download or display error interrupted loading. Check your connection and reload to get the latest version,
        or use the menu to open another page.
      </p>
      <button onClick={() => window.location.reload()} className="btn-primary mt-5">Reload page</button>
    </div>;
    return this.props.children;
  }
}
