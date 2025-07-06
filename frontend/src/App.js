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
  const [editingId, setEditingId] = useState(null);
  const [editForm, setEditForm] = useState({ shares: "", purchasePrice: "" });
  const [selectedIds, setSelectedIds] = useState([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [sortKey, setSortKey] = useState("");     // "pl", "plPct", "chgPct"
  const [sortDir, setSortDir] = useState("desc"); // "asc" | "desc"
  const [timeline, setTimeline] = useState("all"); // "1d" | "5d" | "1m" | "1y" | "5y" | "all"
  const [theme, setTheme] = useState(
    localStorage.getItem("theme") || 
    (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light")
  );

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
  }, [theme]);

  const toggleTheme = () => setTheme(theme === "light" ? "dark" : "light");



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

      // (quick fetch of yesterday close)
      const histRes = await fetch(
        `${API_BASE}/api/history/${ticker}?start=${new Date(Date.now() - 2*864e5)
          .toISOString()
          .split("T")[0]}`
      );
      let yesterdayClose = 0;
      if (histRes.ok) {
        const tmp = await histRes.json();
        const len = tmp.series.length;
        if (len >= 2) yesterdayClose = tmp.series[len - 2].close;
      }

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

  

  // ─── Edit Holding ───────────────────────────────
  const startEdit = (h) => {
    setEditingId(h.id);
    setEditForm({
      shares: h.shares.toString(),
      purchasePrice: h.purchasePrice.toString(),
    });
  };

  // Handler to cancel editing
  const cancelEdit = () => {
    setEditingId(null);
    setEditForm({ shares: "", purchasePrice: "" });
  };

  // Handler to save changes
  const saveEdit = async (id) => {
    const shares = parseFloat(editForm.shares);
    const purchase_price = parseFloat(editForm.purchasePrice);
    if (!shares || !purchase_price) return;

    // 1) Update Supabase
    const { error, data } = await supabase
      .from("holdings")
      .update({ shares, purchase_price })
      .eq("id", id)
      .select()
      .single();

    if (error) {
      console.error("Update failed:", error);
      return;
    }

    // 2) Update local state
    setHoldings((prev) =>
      prev.map((h) =>
        h.id === id
          ? {
              ...h,
              shares,
              purchasePrice: purchase_price,
              // optionally re-fetch current price:
              currentPrice: h.currentPrice,
            }
          : h
      )
    );
    cancelEdit();
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
  
  // helper to compute daily % change (needs yesterdayClose)
  const calcChangePct = (h) =>
    h.yesterdayClose ? ((h.currentPrice - h.yesterdayClose) / h.yesterdayClose) * 100 : 0;


  // initialise selection whenever holdings change
  useEffect(() => {
    setSelectedIds(holdings.map((h) => h.id));   // default = all selected
  }, [holdings]);

  // toggle entire list
  const toggleSelectAll = () => {
    if (selectedIds.length === holdings.length) {
      setSelectedIds([]);           // unselect all
    } else {
      setSelectedIds(holdings.map((h) => h.id)); // select all
    }
  };

  // toggle single holding
  const toggleSelectOne = (id) => {
    setSelectedIds((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    );
  };
  
  // ─── Real profit/loss chart based on purchase dates ───────────────
  useEffect(() => {
    const filteredHoldings = holdings.filter(
      (h) => h.purchaseDate && selectedIds.includes(h.id)
    );
    if (!filteredHoldings.length) {
      setChartData([]);
      return;
    }

    const buildChart = async () => {
      try {
        const histories = await Promise.all(
          filteredHoldings.map(async (h) => {
            const start = new Date(h.purchaseDate).toISOString().split("T")[0];
            const res = await fetch(`${API_BASE}/api/history/${h.ticker}?start=${start}`);
            if (!res.ok) throw new Error(`Failed to fetch history for ${h.ticker}`);
            return res.json();
          })
        );

        const valueMap = {};
        histories.forEach((hist, i) => {
          const { shares } = filteredHoldings[i];
          hist.series.forEach((pt) => {
            valueMap[pt.date] = (valueMap[pt.date] || 0) + pt.close * shares;
          });
        });

        const sortedDates = Object.keys(valueMap).sort();

        const investMap = {};
        let cumulative = 0;
        sortedDates.forEach((date) => {
          filteredHoldings.forEach((h) => {
            if (h.purchaseDate === date) {
              cumulative += h.purchasePrice * h.shares;
            }
          });
          investMap[date] = cumulative;
        });

        const daysMap = { "1d": 1, "5d": 5, "1m": 30, "6m": 183, "1y": 365, "5y": 1825, all: Infinity };
        const maxDays = daysMap[timeline] ?? Infinity;

        const keptDates =
          maxDays === Infinity
          ? sortedDates
          : sortedDates.filter(
              (d) => d >= new Date(Date.now() - maxDays * 864e5).toISOString().split("T")[0]
            );
        
        const downsampleRate = Math.ceil(keptDates.length / 100);

        const series = keptDates
          .filter((_, idx) => idx % downsampleRate === 0)
          .map((date) => {
            const value = valueMap[date];
            const invested = investMap[date];
            const profit = value - invested;
            const profitPercent = invested > 0 ? (profit / invested) * 100 : 0;

            return {
              date: new Date(date).toLocaleDateString("en-US", {
                month: "short",
                day: "numeric",
              }),
              value: Number(value.toFixed(2)),
              profit: Number(profit.toFixed(2)),
              profitPercent: Number(profitPercent.toFixed(2)),
            };
          });

        setChartData(series);
      } catch (err) {
        console.error("Chart build failed:", err);
        setChartData([]);
      }
    };

    buildChart();
  }, [holdings, selectedIds, timeline]);


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

  // ─── Derived list after search & sort ──────────────────────────────
  const displayed = holdings
    .filter((h) => h.ticker.toLowerCase().includes(searchQuery.toLowerCase()))
    .sort((a, b) => {
      const dir = sortDir === "asc" ? 1 : -1;
      if (!sortKey) {
        return dir * a.ticker.localeCompare(b.ticker);  // ✅ Alphabetical sort;
      }
      if (sortKey === "ticker") {
        return dir * a.ticker.localeCompare(b.ticker);  // ✅ Alphabetical sort
      }
      let va, vb;
      if (sortKey === "pl") {
        va = (a.currentPrice - a.purchasePrice) * a.shares;
        vb = (b.currentPrice - b.purchasePrice) * b.shares;
      } else if (sortKey === "plPct") {
        va = ((a.currentPrice - a.purchasePrice) / a.purchasePrice) * 100;
        vb = ((b.currentPrice - b.purchasePrice) / b.purchasePrice) * 100;
      } else if (sortKey === "chgPct") {
        va = calcChangePct(a);
        vb = calcChangePct(b);
      }
      return dir * (va - vb);
    });

  // ─── JSX UI ───────────────────────────────────────────────────────
  return (
    <div className="App">
      <header className="App-header">
        <h1>OneGlance Finance</h1>
        <p>All You Need, Nothing You Don’t.</p>
        <button
          onClick={toggleTheme}
          style={{
            position: "absolute",
            right: "1rem",
            top: "1rem",
            background: "none",
            border: "none",
            fontSize: "1.2rem",
            cursor: "pointer",
            color: "var(--text-sub)",
          }}
          title="Toggle dark mode"
        >
          {theme === "light" ? "🌙" : "☀️"}
        </button>
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

        {/* top controls  */}
        {holdings.length > 0 && (
          <div
            style={{
              marginBottom: "1rem",
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              flexWrap: "wrap", // ensures it wraps nicely on small screens
              gap: ".5rem",
            }}
          >
            {/* Select-All Control */}
            <label style={{ display: "flex", alignItems: "center", gap: ".5rem" }}>
              <input
                type="checkbox"
                checked={selectedIds.length === holdings.length}
                onChange={toggleSelectAll}
              />
              <span style={{ fontSize: ".9rem" }}>
                {selectedIds.length === holdings.length ? "Deselect all" : "Select all"}
              </span>
            </label>

            {/* Chart Timeline Dropdown */}
            {chartData.length > 0 && (
              <select
                value={timeline}
                onChange={(e) => setTimeline(e.target.value)}
                style={{
                  padding: "6px 10px",
                  fontSize: ".9rem",
                  border: "1px solid #ccc",
                  borderRadius: "6px",
                }}
              >
                <option value="1d">1 Day</option>
                <option value="5d">5 Days</option>
                <option value="1m">1 Month</option>
                <option value="6m">6 Months</option>
                <option value="1y">1 Year</option>
                <option value="5y">5 Years</option>
                <option value="all">All-time</option>
              </select>
            )}
          </div>
        )}


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
                  content={({ payload, label }) => {
                    if (!payload || !payload.length) return null;
                    const point = payload[0]?.payload;
                    return (
                      <div
                        style={{
                          backgroundColor: "#fff",
                          border: "1px solid #e0e0e0",
                          borderRadius: "8px",
                          padding: "8px",
                        }}
                      >
                        <strong>{label}</strong>
                        <div>Profit: Rs.{point.profit}</div>
                        <div style={{ color: point.profitPercent >= 0 ? "green" : "red" }}>
                          ({point.profitPercent >= 0 ? "+" : ""}
                          {point.profitPercent}%)
                        </div>
                      </div>
                    );
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
              step="1"
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

        {/* Search & Sort controls */}
        <div className="search-sort-controls">
          <input
            type="text"
            className="search-input"
            placeholder="🔍 Search ticker…"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />

          <select
            className="sort-select"
            value={sortKey}
            onChange={(e) => setSortKey(e.target.value)}
          >
            <option value="">Sort by…</option>
            <option value="ticker">Alphabetical (A–Z)</option>
            <option value="pl">Profit/Loss ₹</option>
            <option value="plPct">Profit/Loss %</option>
            <option value="chgPct">Price Change % (1d)</option>
          </select>

          <button className="sort-dir-button" onClick={() => setSortDir(sortDir === "asc" ? "desc" : "asc")}>
            {sortDir === "asc" ? "↑" : "↓"}
          </button>
        </div>
        
        {/* Holdings List */}
        <div className="holdings">
          {displayed.map((h) => {
            const value = h.currentPrice * h.shares;
            const cost = h.purchasePrice * h.shares;
            const profit = value - cost;
            const profitPct = (profit / cost) * 100;

            return (
              <div key={h.id} className="holding-card">
                <div className="holding-header">
                  <input
                    type="checkbox"
                    checked={selectedIds.includes(h.id)}
                    onChange={() => toggleSelectOne(h.id)}
                    style={{ marginRight: ".6rem" }}
                  />
                  <h3>{h.ticker}</h3>
                  <div className="holding-actions">
                    {editingId === h.id ? (
                      <button className="delete-btn" onClick={cancelEdit}>✕</button>
                    ) : (
                      <>
                        <button className="edit-btn" onClick={() => startEdit(h)}>✎</button>
                        <button className="delete-btn" onClick={() => deleteHolding(h.id)}>
                          <CloseIcon size={18} />
                        </button>
                      </>
                    )}
                  </div>
                </div>

                {editingId === h.id ? (
                  <div className="edit-form">
                    <input
                      type="number"
                      value={editForm.shares}
                      onChange={(e) => setEditForm({...editForm, shares: e.target.value})}
                      step="0.001"
                    />
                    <input
                      type="number"
                      value={editForm.purchasePrice}
                      onChange={(e) => setEditForm({...editForm, purchasePrice: e.target.value})}
                      step="0.01"
                    />
                    <button onClick={() => saveEdit(h.id)}>Save</button>
                    <button onClick={cancelEdit}>Cancel</button>
                  </div>
                ) : (
                  <>
                    <p className="shares">
                      {h.shares} share{h.shares !== 1 ? "s" : ""} @ Rs.{h.purchasePrice} on {h.purchaseDate}
                    </p>
                    <div className="holding-value">
                      <span>Current Value: Rs.{(h.currentPrice * h.shares).toFixed(2)}</span>
                      <span className={h.currentPrice * h.shares - h.purchasePrice * h.shares >= 0 ? "positive" : "negative"}>
                        {((h.currentPrice * h.shares - h.purchasePrice * h.shares) / (h.purchasePrice * h.shares) * 100).toFixed(1)}%
                      </span>
                      <span style={{ fontSize: ".85rem", color: "#888", marginLeft: "8px" }}>
                        1d: {calcChangePct(h).toFixed(1)}%
                      </span>
                    </div>
                  </>
                )}
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
