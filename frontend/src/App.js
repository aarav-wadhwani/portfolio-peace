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
import { supabase } from "./supabaseClient";
import { tickerList } from "./tickers";
import Auth from "./auth";

const API_BASE = process.env.REACT_APP_API_URL;
const emptyForm = { ticker: "", shares: "", purchasePrice: "", purchaseDate: "" };

export default function App() {
  // ─── State ───────────────────────────────────────────────────────
  const [holdings, setHoldings] = useState([]);
  const [showForm, setShowForm] = useState(false);
  const [formData, setFormData] = useState(emptyForm);
  const [chartData, setChartData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [filteredTickers, setFilteredTickers] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [user, setUser] = useState(null);
  const [authLoading, setAuthLoading] = useState(true);

  // ─── Supabase: fetch holdings on first load ───────────────────────
  useEffect(() => {
    const loadHoldings = async () => {
      const { data, error } = await supabase
        .from("holdings")
        .select("id,ticker,shares,purchase_price,current_price,purchase_date,created_at")
        .order("created_at", { ascending: false });

      if (error) {
        console.error(error);
        setErrorMsg("Failed to load saved holdings.");
        return;
      }

      const formatted = data.map((d) => ({
        id: d.id,
        ticker: d.ticker,
        shares: Number(d.shares),
        purchasePrice: Number(d.purchase_price),
        currentPrice: Number(d.current_price),
      }));
      setHoldings(formatted);
    };

    loadHoldings();
  }, []);

  // ─── Helper: live price from FastAPI backend ──────────────────────
  async function fetchLivePrice(ticker) {
    const res = await fetch(`${API_BASE}/api/price/${ticker}`);
    if (!res.ok) throw new Error("Ticker not found on server");
    const data = await res.json();
    return data.price;
  }

  // ─── Add Holding ─────────────────────────────────────────────────
  async function handleSubmit(e) {
    e.preventDefault();
    setErrorMsg("");
    setLoading(true);

    const ticker = formData.ticker.trim().toUpperCase();
    const shares = parseFloat(formData.shares);
    const purchasePrice = parseFloat(formData.purchasePrice);
    const purchaseDate   = formData.purchaseDate;

    if (!ticker || !shares || !purchasePrice) {
      setErrorMsg("Please fill in all fields correctly.");
      setLoading(false);
      return;
    }

    try {
      const livePrice = await fetchLivePrice(ticker);

      // Save to Supabase
      const { data, error } = await supabase
        .from("holdings")
        .insert({
          ticker,
          shares,
          purchase_price: purchasePrice,
          current_price: livePrice,
          purchase_date: purchaseDate,
        })
        .select()
        .single();

      if (error) throw error;

      const newHolding = {
        id: data.id,
        ticker: data.ticker,
        shares: Number(data.shares),
        purchasePrice: Number(data.purchase_price),
        currentPrice: Number(data.current_price),
        purchaseDate: data.purchase_date,
      };

      setHoldings((prev) => [newHolding, ...prev]);
      setFormData(emptyForm);
      setShowForm(false);
    } catch (err) {
      setErrorMsg(err.message || "Could not save holding.");
    } finally {
      setLoading(false);
    }
  }

  // ─── Delete Holding  (local + Supabase) ───────────────────────────
  const deleteHolding = async (id) => {
    // Remove locally first for snappy UI
    setHoldings((prev) => prev.filter((h) => h.id !== id));
    const { error } = await supabase.from("holdings").delete().eq("id", id);
    if (error) console.error("Delete failed:", error);
  };

  // ─── Portfolio Summary Calculations ───────────────────────────────
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

  // ─── Real profit/loss chart based on purchase dates ───────────────
  useEffect(() => {
    if (!holdings.length) {
      setChartData([]);
      return;
    }

    const buildChart = async () => {
      try {
        const validHoldings = holdings.filter((h) => h.purchaseDate);
        const histories = await Promise.all(
          validHoldings.map(async (h) => {
            const formattedStart = new Date(h.purchaseDate).toISOString().split("T")[0];
            const res = await fetch(
              `${API_BASE}/api/history/${h.ticker}?start=${formattedStart}`
            );
            if (!res.ok) {
              throw new Error(`Failed to fetch history for ${h.ticker}`);
            }
            return res.json();
          })
        );

        const dateMap = {};
        histories.forEach((hist, idx) => {
          const holding = validHoldings[idx];
          hist?.series?.forEach?.((pt) => {
            const val = pt.close * holding.shares;
            dateMap[pt.date] = (dateMap[pt.date] || 0) + val;
          });
        });

        const totalInvestedConst = validHoldings.reduce(
          (sum, h) => sum + h.purchasePrice * h.shares,
          0
        );

        const sortedDates = Object.keys(dateMap).sort();
        const downsampleRate = Math.ceil(sortedDates.length / 100); // Max 100 points

        const series = sortedDates
          .filter((_, idx) => idx % downsampleRate === 0) // keep every nth point
          .map((d) => ({
            date: new Date(d).toLocaleDateString("en-US", {
              month: "short",
              day: "numeric",
            }),
            value: Number(dateMap[d].toFixed(2)),
            profit: Number((dateMap[d] - totalInvestedConst).toFixed(2)),
          }));

        setChartData(series);
      } catch (err) {
        console.error("Chart build failed:", err);
        setChartData([]);
      }
    };

    buildChart();
  }, [holdings]);

  // ── Track auth session ──
  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      setUser(session?.user ?? null);
      setAuthLoading(false);
    });

    const { data: listener } = supabase.auth.onAuthStateChange(
      (_event, session) => {
        setUser(session?.user ?? null);
      }
    );
    return () => listener.subscription.unsubscribe();
  }, []);

  // ── Modify loadHoldings to fetch only after user is set ──
  useEffect(() => {
    if (!user) return;

    const loadHoldings = async () => {
      const { data, error } = await supabase
        .from("holdings")
        .select("*")
        .eq("user_id", user.id) // ✅ Only fetch rows belonging to the logged-in user
        .order("created_at", { ascending: false });

      if (!error) {
        setHoldings(
          data.map((d) => ({
            id: d.id,
            ticker: d.ticker,
            shares: Number(d.shares),
            purchasePrice: Number(d.purchase_price),
            currentPrice: Number(d.current_price),
            purchaseDate: d.purchase_date,
          }))
        );
      } else {
        console.error("Error loading holdings:", error.message);
      }
    };

    loadHoldings();
  }, [user]);

  if (authLoading) return null;           // wait for Supabase to init
  if (!user) return <Auth />;             // show sign-in / sign-up UI


  // ─── JSX UI ───────────────────────────────────────────────────────
  return (
    <div className="App">
      <header className="App-header">
        <h1>ClearTrack</h1>
        <p>Your calm space for investment tracking</p>
      </header>

      <main>
        {/* Summary */}
        <div className="portfolio-summary">
          <h2>Portfolio Value</h2>
          <div className="value">Rs.{totalValue.toFixed(2)}</div>
          {totalInvested > 0 && (
            <div className={`profit ${totalProfit >= 0 ? "positive" : "negative"}`}>
              {totalProfit >= 0 ? <TrendingUp size={20} /> : <TrendingDown size={20} />}
              Rs.{Math.abs(totalProfit).toFixed(2)} (
              {totalProfit >= 0 ? "+" : ""}
              {profitPercent.toFixed(1)}%)
            </div>
          )}
        </div>

        {/* Chart */}
        {chartData.length > 0 && (
          <div className="chart-container">
            <h3>Portfolio Profit / Loss</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="date" stroke="#666" tick={{ fontSize: 12 }} />
                <YAxis
                  stroke="#666"
                  tick={{ fontSize: 12 }}
                  tickFormatter={(v) => `Rs.${v}`}
                />
                <Tooltip
                  formatter={(v) => `Rs.${v}`}
                  contentStyle={{
                    backgroundColor: "#fff",
                    border: "1px solid #e0e0e0",
                    borderRadius: "8px",
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="profit"
                  stroke="#4F46E5"
                  strokeWidth={2}
                  dot={{ fill: "#4F46E5", r: 3 }}
                  activeDot={{ r: 5 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Add Button / Form */}
        {!showForm && (
          <button className="add-button" onClick={() => setShowForm(true)}>
            <Plus size={20} /> Add Holding
          </button>
        )}

        {showForm && (
          <form className="add-form" onSubmit={handleSubmit}>
            <h3>Add New Holding</h3>
              <div className="autocomplete-wrapper">
                <input
                  type="text"
                  placeholder="Stock ticker (e.g., RELIANCE)"
                  value={formData.ticker}
                  onChange={(e) => {
                    const input = e.target.value.toUpperCase();
                    setFormData({ ...formData, ticker: input });

                    if (input.length > 1) {
                      const filtered = tickerList.filter((t) =>
                        t.startsWith(input)
                      );
                      setFilteredTickers(filtered.slice(0, 5)); // show top 5 suggestions
                      setShowSuggestions(true);
                    } else {
                      setShowSuggestions(false);
                    }
                  }}
                  required
                />

                {showSuggestions && filteredTickers.length > 0 && (
                  <ul className="suggestions-list">
                    {filteredTickers.map((ticker, idx) => (
                      <li
                        key={idx}
                        onClick={() => {
                          setFormData({ ...formData, ticker });
                          setShowSuggestions(false);
                        }}
                      >
                        {ticker}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            
            <input
              type="date"
              value={formData.purchaseDate}
              onChange={(e) =>
                setFormData({ ...formData, purchaseDate: e.target.value })
              }
              required
            />

            <input
              type="number"
              placeholder="Number of shares"
              value={formData.shares}
              onChange={(e) => setFormData({ ...formData, shares: e.target.value })}
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

        {/* Holdings List */}
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
                  <button className="delete-btn" onClick={() => deleteHolding(h.id)}>
                    <CloseIcon size={18} />
                  </button>
                </div>

                <p className="shares">
                  {h.shares} share{h.shares !== 1 ? "s" : ""} @ Rs.{h.purchasePrice} on {h.purchaseDate}
                </p>

                <div className="holding-value">
                  <span>Current Value: Rs.{value.toFixed(2)}</span>
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
