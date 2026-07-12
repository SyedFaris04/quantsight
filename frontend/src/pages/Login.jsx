/**
 * frontend/src/pages/Login.jsx
 * ─────────────────────────────────────────────────────────────────
 * Single page for both sign-in and sign-up via email/password.
 *
 * Signing in is entirely optional — every feature works in guest mode
 * (localStorage) without an account. Logging in only upgrades Portfolio
 * and Game progress to sync across devices via Supabase.
 * ─────────────────────────────────────────────────────────────────
 */

import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

export default function Login() {
  const { user, signInWithEmail, signUpWithEmail } = useAuth();
  const navigate = useNavigate();

  const [mode, setMode] = useState("signin"); // "signin" | "signup"
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [info, setInfo] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (user) navigate("/", { replace: true });
  }, [user, navigate]);

  if (user) return null;

  async function handleEmailSubmit(e) {
    e.preventDefault();
    setError("");
    setInfo("");
    if (!email || !password) {
      setError("Enter both an email and a password.");
      return;
    }
    setBusy(true);
    const action = mode === "signin" ? signInWithEmail : signUpWithEmail;
    const { data, error } = await action(email, password);
    setBusy(false);

    if (error) {
      setError(error.message);
      return;
    }
    if (mode === "signup" && !data.session) {
      setInfo("Check your email to confirm your account, then sign in.");
      return;
    }
    navigate("/");
  }

  return (
    <div className="max-w-sm mx-auto mt-8 sm:mt-16">
      <div className="card">
        <div className="text-center mb-6">
          <h1 className="text-xl font-semibold text-white">
            {mode === "signin" ? "Sign In" : "Create Account"}
          </h1>
          <p className="text-sm text-gray-500 mt-1">
            Optional — sync your Portfolio and Game progress across devices.
            Everything still works without an account.
          </p>
        </div>

        <form onSubmit={handleEmailSubmit} className="space-y-3">
          <input
            type="email"
            placeholder="Email"
            value={email}
            onChange={e => setEmail(e.target.value)}
            className="input w-full"
            autoComplete="email"
          />
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={e => setPassword(e.target.value)}
            className="input w-full"
            autoComplete={mode === "signin" ? "current-password" : "new-password"}
          />

          {error && <p className="text-xs text-red-400">{error}</p>}
          {info && <p className="text-xs text-green-400">{info}</p>}

          <button type="submit" disabled={busy} className="w-full btn-primary disabled:opacity-50">
            {mode === "signin" ? "Sign In" : "Create Account"}
          </button>
        </form>

        <p className="text-xs text-gray-500 text-center mt-4">
          {mode === "signin" ? "Don't have an account?" : "Already have an account?"}{" "}
          <button
            onClick={() => { setMode(mode === "signin" ? "signup" : "signin"); setError(""); setInfo(""); }}
            className="text-indigo-400 hover:text-indigo-300 font-medium"
          >
            {mode === "signin" ? "Create one" : "Sign in"}
          </button>
        </p>
      </div>
    </div>
  );
}
