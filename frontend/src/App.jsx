/**
 * frontend/src/App.jsx
 * Root component — sidebar layout + routing between 7 pages
 *
 * Sidebar is a fixed off-canvas drawer below the `lg` breakpoint (opened via
 * the hamburger button in the mobile header bar below), and a static column
 * at `lg` and above — see Sidebar.jsx for the responsive classes.
 */

import { lazy, Suspense, useState } from "react";
import { Routes, Route, Navigate, useLocation } from "react-router-dom";
import { AuthProvider } from "./context/AuthContext";
import Sidebar     from "./components/Sidebar";
import Dashboard   from "./pages/Dashboard";
import ChatWidget  from "./components/ChatWidget";
import PageBoundary, { PageLoading } from "./components/PageBoundary";

// Keep the landing page and persistent shell ready; defer other pages and charts.
const Overview = lazy(() => import("./pages/Overview"));
const Detail = lazy(() => import("./pages/Detail"));
const Compare = lazy(() => import("./pages/Compare"));
const Leaderboard = lazy(() => import("./pages/Leaderboard"));
const Game = lazy(() => import("./pages/Game"));
const Portfolio = lazy(() => import("./pages/Portfolio"));
const Login = lazy(() => import("./pages/Login"));
const Settings = lazy(() => import("./pages/Settings"));
const TrackRecord = lazy(() => import("./pages/TrackRecord"));
const Backtesting = lazy(() => import("./pages/Backtesting"));

export default function App() {
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const { pathname } = useLocation();

  return (
    <AuthProvider>
    <div className="min-h-screen bg-gray-950 text-gray-100 flex">
      <Sidebar isOpen={mobileNavOpen} onClose={() => setMobileNavOpen(false)} />

      <div className="flex-1 min-w-0 flex flex-col">
        {/* Mobile-only top bar — hidden at lg and above, where the sidebar is static */}
        <header className="lg:hidden flex items-center gap-3 h-14 px-4 border-b border-gray-800
                            bg-gray-900/60 backdrop-blur-sm sticky top-0 z-30">
          <button
            onClick={() => setMobileNavOpen(true)}
            aria-label="Open navigation menu"
            className="w-9 h-9 flex items-center justify-center rounded-lg text-gray-300
                       hover:bg-gray-800 hover:text-white transition-colors"
          >
            <span className="text-xl leading-none">☰</span>
          </button>
          <span className="font-bold text-base tracking-tight text-white">
            Quant<span className="text-indigo-400">Sight</span>
          </span>
        </header>

        <main className="flex-1 min-w-0 px-4 sm:px-6 lg:px-8 py-6 lg:py-8">
          <PageBoundary key={pathname}>
          <Suspense fallback={<PageLoading />}>
          <Routes>
            <Route path="/"               element={<Dashboard />} />
            <Route path="/market"         element={<Overview />}  />
            <Route path="/detail/:ticker" element={<Detail />}    />
            <Route path="/compare"        element={<Compare />}   />
            <Route path="/leaderboard"    element={<Leaderboard />} />
            <Route path="/portfolio"      element={<Portfolio />} />
            <Route path="/game"           element={<Game />}      />
            <Route path="/login"          element={<Login />}     />
            <Route path="/settings"       element={<Settings />}  />
            <Route path="/track-record"   element={<TrackRecord />} />
            <Route path="/backtesting"    element={<Backtesting />} />
            <Route path="*"               element={<Navigate to="/" replace />} />
          </Routes>
          </Suspense>
          </PageBoundary>
        </main>
      </div>

      <ChatWidget />
    </div>
    </AuthProvider>
  );
}
