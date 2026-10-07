import { NavLink, useLocation, useNavigate } from "react-router-dom";
import { useEffect, useRef } from "react";
import { useAuth } from "../context/AuthContext";
import Icon from "./Icon";
const GROUPS = [
  [
    "Workspace",
    [
      ["/", "Dashboard", "dashboard"],
      ["/market", "Market explorer", "market"],
      ["/portfolio", "Portfolio", "portfolio"],
    ],
  ],
  [
    "Evaluation",
    [
      ["/backtesting", "Backtesting", "backtest"],
      ["/track-record", "Forward results", "track"],
      ["/model-health", "Model health", "track"],
      ["/compare", "Model comparison", "compare"],
      ["/leaderboard", "Model results", "results"],
    ],
  ],
  [
    "Learning & account",
    [
      ["/game", "Prediction quiz", "learn"],
      ["/settings", "Settings", "settings"],
    ],
  ],
];
function AccountSection() {
  const { user, signOut } = useAuth();
  const navigate = useNavigate();
  if (!user)
    return (
      <button
        onClick={() => navigate("/login")}
        className="btn-secondary w-full flex items-center gap-3"
      >
        <Icon name="user" />
        Sign in
      </button>
    );
  return (
    <div className="flex items-center gap-3 text-xs">
      <span className="w-7 h-7 rounded-md bg-gray-800 flex items-center justify-center">
        {(user.email?.[0] || "?").toUpperCase()}
      </span>
      <span className="flex-1 truncate" title={user.email}>
        {user.email}
      </span>
      <button onClick={() => signOut()} className="text-gray-400">
        Sign out
      </button>
    </div>
  );
}
export default function Sidebar({ isOpen = false, onClose = () => {} }) {
  const { pathname } = useLocation();
  const aside = useRef(null);
  useEffect(() => {
    onClose();
  }, [pathname]);
  useEffect(() => {
    if (!isOpen) return;
    const previous = document.activeElement;
    aside.current.querySelector("button")?.focus();
    function keydown(e) {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      }
      if (e.key === "Tab" && window.innerWidth < 1024) {
        const targets = [...aside.current.querySelectorAll("a, button")].filter(
          (el) => !el.disabled,
        );
        const first = targets[0],
          last = targets.at(-1);
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    }
    document.addEventListener("keydown", keydown);
    return () => {
      document.removeEventListener("keydown", keydown);
      if (previous?.isConnected) previous.focus();
    };
  }, [isOpen]);
  return (
    <>
      {isOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/60 lg:hidden"
          onClick={onClose}
          aria-hidden="true"
        />
      )}
      <aside
        ref={aside}
        id="primary-navigation"
        className={`w-60 flex-shrink-0 h-screen fixed inset-y-0 left-0 z-50 border-r border-gray-800 bg-gray-950 flex flex-col transform transition-transform duration-200 lg:visible lg:translate-x-0 lg:sticky lg:top-0 lg:z-auto ${isOpen ? "visible translate-x-0" : "invisible -translate-x-full"}`}
      >
        <div className="flex items-center justify-between gap-2 px-5 h-20 border-b border-gray-800 flex-shrink-0">
          <div className="flex items-center gap-3">
            <span className="w-8 h-8 rounded-md border border-indigo-500/40 bg-indigo-500/10 flex items-center justify-center text-indigo-300 font-semibold">
              Q
            </span>
            <div>
              <p className="font-semibold text-lg tracking-tight">QuantSight</p>
              <p className="text-[11px] text-gray-500">Financial research</p>
            </div>
          </div>
          <button
            onClick={onClose}
            aria-label="Close navigation menu"
            className="lg:hidden p-2 text-gray-400"
          >
            <Icon name="close" />
          </button>
        </div>
        <nav
          aria-label="Primary navigation"
          className="flex-1 px-3 py-5 overflow-y-auto space-y-6"
        >
          {GROUPS.map(([group, links]) => (
            <div key={group}>
              <p className="px-3 mb-2 text-[10px] uppercase tracking-widest text-gray-500 font-medium">
                {group}
              </p>
              <div className="space-y-1">
                {links.map(([to, label, icon]) => (
                  <NavLink
                    key={to}
                    to={to}
                    end={to === "/"}
                    className={({ isActive }) =>
                      `flex items-center gap-3 px-3 py-2.5 rounded-md text-sm transition-colors ${isActive ? "bg-indigo-500/10 text-indigo-300" : "text-gray-400 hover:text-gray-200 hover:bg-gray-900"}`
                    }
                  >
                    <Icon name={icon} />
                    <span>{label}</span>
                  </NavLink>
                ))}
              </div>
            </div>
          ))}
        </nav>
        <div className="px-4 py-4 border-t border-gray-800">
          <AccountSection />
        </div>
        <p className="px-5 py-4 border-t border-gray-800 text-[11px] text-gray-500">
          Research & learning · Five-session horizon
        </p>
      </aside>
    </>
  );
}
