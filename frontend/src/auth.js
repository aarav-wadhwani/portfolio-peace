// src/Auth.js
import { useState, useEffect } from "react";
import { supabase } from "./supabaseClient";
import { Mail, Lock, Moon, Sun, LogIn, UserPlus, Eye, EyeOff } from "lucide-react";

export default function Auth() {
  const [isSignUp, setIsSignUp] = useState(false);
  const [form, setForm] = useState({ email: "", password: "" });
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [theme, setTheme] = useState(
    localStorage.getItem("theme") || 
    (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light")
  );

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
  }, [theme]);

  const toggleTheme = () => setTheme(theme === "light" ? "dark" : "light");

  async function handleSubmit() {
    if (loading) return;
    
    setError("");
    setMessage("");
    setLoading(true);

    const { email, password } = form;

    try {
      if (isSignUp) {
        // ── Sign-up flow ─────────────────────────
        const { error: signUpError } = await supabase.auth.signUp({
          email,
          password,
        });

        if (signUpError) throw signUpError;

        // ✅ Show confirmation
        setMessage(
          "A verification email has been sent. Please check your inbox."
        );
      } else {
        // ── Sign-in flow ─────────────────────────
        const { error: signInError } =
          await supabase.auth.signInWithPassword({ email, password });

        if (signInError) throw signInError;
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSubmit();
    }
  };

  return (
    <div className="App">
      {/* Header */}
      <header className="App-header">
        <div className="header-content">
          <div className="header-left">
            <h1>OneGlance</h1>
            <p>Your investments, simplified</p>
          </div>
          <button className="theme-toggle" onClick={toggleTheme} title="Toggle theme">
            {theme === "light" ? <Moon size={20} /> : <Sun size={20} />}
          </button>
        </div>
      </header>

      {/* Auth Container */}
      <div className="auth-container">
        <div className="auth-box">
          <div className="auth-icon">
            {isSignUp ? <UserPlus size={32} /> : <LogIn size={32} />}
          </div>
          
          <h2>{isSignUp ? "Create an account" : "Welcome back"}</h2>
          <p className="auth-subtitle">
            {isSignUp 
              ? "Start tracking your investment portfolio" 
              : "Sign in to access your portfolio"}
          </p>

          <div className="auth-form">
            <div className="form-group">
              <label className="form-label">
                <Mail size={16} />
                Email
              </label>
              <input
                type="email"
                placeholder="Enter your email"
                value={form.email}
                onChange={(e) => setForm({ ...form, email: e.target.value })}
                onKeyDown={handleKeyPress}
                autoFocus
                required
              />
            </div>

            <div className="form-group">
              <label className="form-label">
                <Lock size={16} />
                Password
              </label>
              <div style={{ position: "relative" }}>
                <input
                  type={showPassword ? "text" : "password"}
                  placeholder="Enter your password (min. 6 characters)"
                  value={form.password}
                  onChange={(e) => setForm({ ...form, password: e.target.value })}
                  onKeyDown={handleKeyPress}
                  minLength={6}
                  required
                  style={{ paddingRight: "3rem" }}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="password-toggle"
                  tabIndex={-1}
                  title={showPassword ? "Hide password" : "Show password"}
                >
                  {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                </button>
              </div>
            </div>

            {/* Success or error feedback */}
            {message && (
              <div className="auth-message auth-success">
                <span>✓</span> {message}
              </div>
            )}
            {error && (
              <div className="auth-message auth-error">
                <span>!</span> {error}
              </div>
            )}

            <button type="button" onClick={handleSubmit} className="btn btn-primary auth-submit" disabled={loading}>
              {loading ? "Loading..." : (isSignUp ? "Sign up" : "Sign in")}
            </button>
          </div>

          <div className="auth-divider">
            <span>or</span>
          </div>

          <p className="auth-toggle">
            {isSignUp ? "Already have an account?" : "No account?"}{" "}
            <span onClick={() => {
              setIsSignUp(!isSignUp);
              setError("");
              setMessage("");
              setForm({ email: "", password: "" });
              setShowPassword(false);
            }}>
              {isSignUp ? "Sign in" : "Sign up"}
            </span>
          </p>
        </div>

        {/* Footer */}
        <div className="auth-footer">
          <p>Secure authentication powered by Supabase</p>
        </div>
      </div>
    </div>
  );
}