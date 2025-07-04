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
    const validHoldings = holdings.filter((h) => h.purchaseDate && selectedIds.includes(h.id));
    if (!validHoldings.length) {
      setChartData([]);
      return;
    }

    const buildChart = async () => {
      try {
        const validHoldings = holdings.filter((h) => h.purchaseDate);
        const histories = await Promise.all(
          validHoldings.map(async (h) => {
            const start = new Date(h.purchaseDate).toISOString().split("T")[0];
            const res = await fetch(`${API_BASE}/api/history/${h.ticker}?start=${start}`);
            if (!res.ok) throw new Error(`Failed to fetch history for ${h.ticker}`);
            return res.json();
          })
        );

        // 1) Build a map of date → total portfolio value on that date
        const valueMap = {};
        histories.forEach((hist, i) => {
          const { shares } = validHoldings[i];
          hist.series.forEach((pt) => {
            valueMap[pt.date] = (valueMap[pt.date] || 0) + pt.close * shares;
          });
        });

        // 2) Build a sorted list of all dates
        const sortedDates = Object.keys(valueMap).sort();

        // 3) Build investMap: date → cumulative invested up to that date
        const investMap = {};
        let cumulative = 0;
        sortedDates.forEach((date) => {
          // add any holdings purchased on this date
          validHoldings.forEach((h) => {
            if (h.purchaseDate === date) {
              cumulative += h.purchasePrice * h.shares;
            }
          });
          investMap[date] = cumulative;
        });

        // 4) Downsample if needed
        const downsampleRate = Math.ceil(sortedDates.length / 100);

        // 5) Build final series with profit = value − invested to date
        const series = sortedDates
          .filter((_, idx) => idx % downsampleRate === 0)
          .map((date) => ({
            date: new Date(date).toLocaleDateString("en-US", {
              month: "short",
              day: "numeric",
            }),
            value: Number(valueMap[date].toFixed(2)),
            profit: Number((valueMap[date] - investMap[date]).toFixed(2)),
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

        {/* Select-All control */}
        {holdings.length > 0 && (
          <div style={{ margin: "0 0 1rem 0", display: "flex", alignItems: "center", gap: ".5rem" }}>
            <input
              type="checkbox"
              checked={selectedIds.length === holdings.length}
              onChange={toggleSelectAll}
            />
            <span style={{ fontSize: ".9rem" }}>
              {selectedIds.length === holdings.length ? "Deselect all" : "Select all"}
            </span>
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
                      <span>Current Value: Rs.{(h.currentPrice*h.shares).toFixed(2)}</span>
                      <span className={h.currentPrice*h.shares - h.purchasePrice*h.shares >= 0 ? "positive":"negative"}>
                        {((h.currentPrice*h.shares - h.purchasePrice*h.shares)/ (h.purchasePrice*h.shares)*100).toFixed(1)}%
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
