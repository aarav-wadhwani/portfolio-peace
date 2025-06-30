// App.js
import "./App.css";
import { useState, useEffect } from "react";
import {
  TrendingUp,
  TrendingDown,
  Plus,
  X as CloseIcon,
} from "lucide-react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";

// ✅ 1.  Set this to your Render URL in .env
// CRA  : REACT_APP_API_URL=https://your-api.onrender.com
// Vite : VITE_API_URL=https://your-api.onrender.com
const API_BASE = import.meta?.env?.VITE_API_URL || process.env.REACT_APP_API_URL;

const emptyForm = { ticker: "", shares: "", purchasePrice: "" };

export default function App() {
  // ───────── State ─────────
  const [holdings, setHoldings] = useState([]);
  const [showForm, setShowForm] = useState(false);
  const [formData, setFormData] = useState(emptyForm);
  const [chartData, setChartData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");

  // ───────── Helpers ─────────
  async function fetchLivePrice(ticker) {
    try {
      const res = await fetch(`${API_BASE}/api/price/${ticker}`);
      if (!res.ok) throw new Error("Ticker not found on server");
      const data = await res.json();
      return data.price; // number
    } catch (err) {
      throw err;
    }
  }

  // ───────── Handle Add Holding ─────────
  async function handleSubmit(e) {
    e.preventDefault();
    setErrorMsg("");
    setLoading(true);

    const ticker = formData.ticker.trim().toUpperCase();
    const shares = parseFloat(formData.shares);
    const purchasePrice = parseFloat(formData.purchasePrice);

    if (!ticker || !shares || !purchasePrice) {
      setErrorMsg("Please fill in all fields correctly.");
      setLoading(false);
      return;
    }

    try {
      const livePrice = await fetchLivePrice(ticker);

      const newHolding = {
        id: Date.now(),
        ticker,
        shares,
        purchasePrice,
        currentPrice: livePrice,
      };

      setHoldings((prev) => [...prev, newHolding]);
      setFormData(emptyForm);
      setShowForm(false);
    } catch (err) {
      setErrorMsg(err.message || "Could not fetch live price.");
    } finally {
      setLoading(false);
    }
  }

  // ───────── Delete Holding ─────────
  const deleteHolding = (id) =>
    setHoldings((prev) => prev.filter((h) => h.id !== id));

  // ───────── Portfolio Summary ─────────
  const totalInvested = holdings.reduce(
    (sum, h) => sum + h.purchasePrice * h.shares,
    0
  );
  const totalValue = holdings.reduce(
    (sum, h) => sum + h.currentPrice * h.shares,
    0
  );
  const totalProfit = totalValue - totalInvested;
  const profitPercent =
    totalInvested > 0 ? (totalProfit / totalInvested) * 100 : 0;

  // ───────── Generate mock 30-day chart based on live holdings ─────────
  useEffect(() => {
    if (!holdings.length) {
      setChartData([]);
      return;
    }

    const days = 30;
    const totalInvestedNow = totalInvested;

    const series = Array.from({ length: days }, (_, i) => {
      const date = new Date();
      date.setDate(date.getDate() - (days - i));
      const randomFactor = 0.9 + Math.random() * 0.3;
      const value = totalInvestedNow * randomFactor;

      return {
        date: date.toLocaleDateString("en-US", { month: "short", day: "numeric" }),
        value: Number(value.toFixed(2)),
      };
    });

    series.push({
      date: "Today",
      value: Number(totalValue.toFixed(2)),
    });

    setChartData(series);
  }, [holdings, totalInvested, totalValue]);

  // ───────── JSX ─────────
  return (
    <div className="App">
      <header className="App-header">
        <h1>ClearTrack</h1>
        <p>Your calm space for investment tracking</p>
      </header>

      <main>
        {/* ── Summary ── */}
        <div className="portfolio-summary">
          <h2>Portfolio Value</h2>
          <div className="value">${totalValue.toFixed(2)}</div>
          {totalInvested > 0 && (
            <div
              className={`profit ${totalProfit >= 0 ? "positive" : "negative"}`}
            >
              {totalProfit >= 0 ? <TrendingUp size={20} /> : <TrendingDown size={20} />}
              ${Math.abs(totalProfit).toFixed(2)} (
              {totalProfit >= 0 ? "+" : ""}
              {profitPercent.toFixed(1)}%)
            </div>
          )}
        </div>

        {/* ── Chart ── */}
        {chartData.length > 0 && (
          <div className="chart-container">
            <h3>Portfolio Performance&nbsp;(Last 30 Days)</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="date" stroke="#666" tick={{ fontSize: 12 }} />
                <YAxis
                  stroke="#666"
                  tick={{ fontSize: 12 }}
                  tickFormatter={(v) => `$${v}`}
                />
                <Tooltip
                  formatter={(v) => `$${v}`}
                  contentStyle={{
                    backgroundColor: "#fff",
                    border: "1px solid #e0e0e0",
                    borderRadius: "8px",
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="value"
                  stroke="#4F46E5"
                  strokeWidth={2}
                  dot={{ fill: "#4F46E5", r: 3 }}
                  activeDot={{ r: 5 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* ── Add Button or Form ── */}
        {!showForm && (
          <button className="add-button" onClick={() => setShowForm(true)}>
            <Plus size={20} /> Add Holding
          </button>
        )}

        {showForm && (
          <form className="add-form" onSubmit={handleSubmit}>
            <h3>Add New Holding</h3>

            <input
              type="text"
              placeholder="Stock ticker (e.g., RELIANCE)"
              value={formData.ticker}
              onChange={(e) =>
                setFormData({ ...formData, ticker: e.target.value })
              }
              required
            />

            <input
              type="number"
              placeholder="Number of shares"
              value={formData.shares}
              onChange={(e) =>
                setFormData({ ...formData, shares: e.target.value })
              }
              step="0.001"
              required
            />

            <input
              type="number"
              placeholder="Purchase price per share"
              value={formData.purchasePrice}
              onChange={(e) =>
                setFormData({ ...formData, purchasePrice: e.target.value })
              }
              step="0.01"
              required
            />

            {errorMsg && <p className="error">{errorMsg}</p>}

            <div className="form-buttons">
              <button
                type="button"
                onClick={() => {
                  setShowForm(false);
                  setFormData(emptyForm);
                  setErrorMsg("");
                }}
              >
                Cancel
              </button>
              <button type="submit" disabled={loading}>
                {loading ? "Adding…" : "Add Holding"}
              </button>
            </div>
          </form>
        )}

        {/* ── Holdings List ── */}
        <div className="holdings">
          {holdings.map((h) => {
            const value = h.currentPrice * h.shares;
            const cost = h.purchasePrice * h.shares;
            const profit = value - cost;
            const profitPct = (profit / cost) * 100;

            return (
              <div key={h.id} className="holding-card">
                <div className="holding-header">
                  <h3>{h.ticker}</h3>
                  <button
                    className="delete-btn"
                    onClick={() => deleteHolding(h.id)}
                  >
                    <CloseIcon size={18} />
                  </button>
                </div>

                <p className="shares">
                  {h.shares} share{h.shares !== 1 ? "s" : ""} @ ${h.purchasePrice}
                </p>

                <div className="holding-value">
                  <span>Current Value: ${value.toFixed(2)}</span>
                  <span className={profit >= 0 ? "positive" : "negative"}>
                    {profit >= 0 ? "+" : ""}
                    {profitPct.toFixed(1)}%
                  </span>
                </div>
              </div>
            );
          })}
        </div>

        {holdings.length === 0 && !showForm && (
          <p className="empty-state">No holdings yet. Add your first investment!</p>
        )}
      </main>
    </div>
  );
}
