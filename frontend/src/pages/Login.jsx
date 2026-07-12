/**
 * frontend/src/pages/Login.jsx
 * ─────────────────────────────────────────────────────────────────
 * Single page for both sign-in and sign-up — Google OAuth or
 * email/password. Google sign-up/sign-in are the same call in
 * Supabase (a new Google user gets an account automatically), so only
 * the email/password path needs an explicit sign-in-vs-signup toggle.
 *
 * Signing in is entirely optional — every feature works in guest mode
 * (localStorage) without an account. Logging in only upgrades Portfolio
 * and Game progress to sync across devices via Supabase.
 * ─────────────────────────────────────────────────────────────────
 */

import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

function GoogleIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
      <path fill="#4285F4" d="M17.64 9.2c0-.64-.06-1.25-.16-1.84H9v3.48h4.84a4.14 4.14 0 0 1-1.8 2.72v2.26h2.9c1.7-1.57 2.7-3.88 2.7-6.62z"/>
      <path fill="#34A853" d="M9 18c2.43 0 4.47-.8 5.96-2.18l-2.9-2.26c-.8.54-1.84.86-3.06.86-2.35 0-4.34-1.59-5.05-3.72H.96v2.33A9 9 0 0 0 9 18z"/>
      <path fill="#FBBC05" d="M3.95 10.7A5.4 5.4 0 0 1 3.67 9c0-.59.1-1.17.28-1.7V4.97H.96A9 9 0 0 0 0 9c0 1.45.35 2.83.96 4.03l2.99-2.33z"/>
      <path fill="#EA4335" d="M9 3.58c1.32 0 2.51.46 3.44 1.35l2.58-2.58C13.46.89 11.43 0 9 0A9 9 0 0 0 .96 4.97l2.99 2.33C4.66 5.17 6.65 3.58 9 3.58z"/>
    </svg>
  );
}

export default function Login() {
  const { user, signInWithGoogle, signInWithEmail, signUpWithEmail } = useAuth();
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

  async function handleGoogle() {
    setError("");
    setBusy(true);
    const { error } = await signInWithGoogle();
    if (error) setError(error.message);
    setBusy(false);
  }

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

        <button
          onClick={handleGoogle}
          disabled={busy}
          className="w-full btn-secondary flex items-center justify-center gap-2.5 disabled:opacity-50"
        >
          <GoogleIcon />
          Continue with Google
        </button>

        <div className="flex items-center gap-3 my-5">
          <div className="flex-1 h-px bg-gray-800" />
          <span className="text-xs text-gray-600">or</span>
          <div className="flex-1 h-px bg-gray-800" />
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
