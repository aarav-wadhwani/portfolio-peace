// src/Auth.js
import { useState } from "react";
import { supabase } from "./supabaseClient";

export default function Auth() {
  const [isSignUp, setIsSignUp] = useState(false);
  const [form, setForm] = useState({ email: "", password: "" });
  const [error, setError] = useState("");

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");

    const { email, password } = form;
    try {
      const { error: authError } = isSignUp
        ? await supabase.auth.signUp({ email, password })
        : await supabase.auth.signInWithPassword({ email, password });

      if (authError) throw authError;
    } catch (err) {
      setError(err.message);
    }
  }

  return (
    <div className="auth-box">
      <h2>{isSignUp ? "Create an account" : "Welcome back"}</h2>

      <form onSubmit={handleSubmit}>
        <input
          type="email"
          placeholder="Email"
          value={form.email}
          onChange={(e) => setForm({ ...form, email: e.target.value })}
          required
        />
        <input
          type="password"
          placeholder="Password"
          value={form.password}
          onChange={(e) => setForm({ ...form, password: e.target.value })}
          required
        />

        {error && <p className="auth-error">{error}</p>}

        <button type="submit">
          {isSignUp ? "Sign up" : "Sign in"}
        </button>
      </form>

      <p className="auth-toggle">
        {isSignUp ? "Already have an account?" : "No account?"}{" "}
        <span onClick={() => setIsSignUp(!isSignUp)}>
          {isSignUp ? "Sign in" : "Sign up"}
        </span>
      </p>
    </div>
  );
}
