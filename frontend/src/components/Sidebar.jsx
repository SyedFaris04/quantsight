/**
 * frontend/src/components/Sidebar.jsx
 * Left sidebar navigation. No Help link since it has no real content
 * behind it — Settings does (see pages/Settings.jsx) and is included.
 *
 * Responsive: below the `lg` breakpoint this is a fixed off-canvas drawer,
 * toggled by App.jsx's mobile header hamburger button (isOpen/onClose) and
 * dismissible via its own backdrop or close button. At `lg` and above it's
 * back to a static, always-visible column and isOpen/onClose have no
 * visible effect — this is the fix for the mobile bug where the sidebar
 * used to permanently occupy ~1/3 of a phone-width viewport, squeezing
 * every page's content into an unreadably narrow column.
 */

import { NavLink, useLocation, useNavigate } from "react-router-dom";
import { useEffect } from "react";
import { useAuth } from "../context/AuthContext";

const NAV_LINKS = [
  { to: "/",          label: "Dashboard",  icon: "⬛" },
  { to: "/market",    label: "Market",     icon: "📈" },
  { to: "/compare",   label: "AI Compare", icon: "📊" },
  { to: "/leaderboard", label: "Leaderboard", icon: "🏆" },
  { to: "/portfolio", label: "Portfolio",  icon: "💼" },
  { to: "/game",       label: "Game",       icon: "🎮" },
  { to: "/settings",   label: "Settings",   icon: "⚙️" },
];

function AccountSection() {
  const { user, signOut } = useAuth();
  const navigate = useNavigate();

  if (!user) {
    return (
      <button
        onClick={() => navigate("/login")}
        className="w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm font-medium
                   text-gray-400 hover:text-gray-200 hover:bg-gray-800 border border-gray-800
                   transition-colors"
      >
        <span className="w-6 h-6 rounded-full bg-gray-800 flex items-center justify-center text-xs">
          👤
        </span>
        Sign In
      </button>
    );
  }

  const initial = (user.email?.[0] || "?").toUpperCase();
  return (
    <div className="flex items-center gap-2.5 px-1">
      <span className="w-7 h-7 rounded-full bg-indigo-500/20 border border-indigo-600/40
                       flex items-center justify-center text-indigo-400 font-semibold text-xs flex-shrink-0">
        {initial}
      </span>
      <span className="flex-1 min-w-0 text-xs text-gray-400 truncate" title={user.email}>
        {user.email}
      </span>
      <button
        onClick={() => signOut()}
        className="text-xs text-gray-500 hover:text-gray-300 flex-shrink-0"
      >
        Sign Out
      </button>
    </div>
  );
}

export default function Sidebar({ isOpen = false, onClose = () => {} }) {
  const location = useLocation();

  // Auto-close the mobile drawer whenever the route changes (e.g. after
  // tapping a nav link) — no effect on desktop, where it's already static.
  useEffect(() => {
    onClose();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [location.pathname]);

  return (
    <>
      {/* Backdrop — mobile only, closes the drawer on tap */}
      {isOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/60 lg:hidden"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      <aside
        className={`w-60 flex-shrink-0 h-screen fixed inset-y-0 left-0 z-50 border-r border-gray-800
                    bg-gray-950 lg:bg-gray-900/60 backdrop-blur-sm flex flex-col
                    transform transition-transform duration-200 ease-in-out
                    lg:translate-x-0 lg:sticky lg:top-0 lg:z-auto
                    ${isOpen ? "translate-x-0" : "-translate-x-full"}`}
      >

        {/* Brand */}
        <div className="flex items-center justify-between gap-2 px-5 h-16 border-b border-gray-800 flex-shrink-0">
          <div className="flex items-center gap-2">
            <span className="w-7 h-7 rounded-lg bg-indigo-500/20 border border-indigo-600/40
                             flex items-center justify-center text-indigo-400 font-bold text-sm">
              Q
            </span>
            <span className="font-bold text-lg tracking-tight text-white">
              Quant<span className="text-indigo-400">Sight</span>
            </span>
          </div>
          <button
            onClick={onClose}
            aria-label="Close navigation menu"
            className="lg:hidden w-8 h-8 flex items-center justify-center rounded-lg text-gray-400
                       hover:bg-gray-800 hover:text-white transition-colors"
          >
            ✕
          </button>
        </div>

        {/* Nav links */}
        <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
          {NAV_LINKS.map(({ to, label, icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) =>
                [
                  "flex items-center gap-2.5 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors duration-150",
                  isActive
                    ? "bg-indigo-600/20 text-indigo-400 border border-indigo-600/30"
                    : "text-gray-400 hover:text-gray-200 hover:bg-gray-800 border border-transparent",
                ].join(" ")
              }
            >
              <span className="text-base leading-none w-5 text-center">{icon}</span>
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>

        {/* Account */}
        <div className="px-3 py-3 border-t border-gray-800 flex-shrink-0">
          <AccountSection />
        </div>

        {/* Live status */}
        <div className="px-5 py-4 border-t border-gray-800 flex-shrink-0">
          <div className="flex items-center gap-2 text-xs text-gray-500">
            <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
            <span>Live</span>
          </div>
          <p className="text-[10px] text-gray-600 mt-1">AI Decision Support</p>
        </div>
      </aside>
    </>
  );
}
