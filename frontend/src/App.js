import "./App.css";
import { useState, useEffect } from "react";
import {
  TrendingUp,
  TrendingDown,
  Plus,
  X,
  Edit2,
  Trash2,
  Moon,
  Sun,
  ArrowUp,
  ArrowDown,
} from "lucide-react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  PieChart,
  Pie,
  Cell,
  ReferenceLine,
} from "recharts";
import { supabase } from "./supabaseClient";
import { tickerList } from "./tickers";
import Auth from "./auth";

const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:8000";
const emptyForm = { ticker: "", shares: "", purchasePrice: "", purchaseDate: "" };

// Color palette for pie chart
const COLORS = ["#6366f1", "#22c55e", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4"];

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
  const [searchQuery, setSearchQuery] = useState("");
  const [sortBy, setSortBy] = useState("ticker"); // Default sort by ticker
  const [sortOrder, setSortOrder] = useState("asc"); // Default ascending
  const [timeline, setTimeline] = useState("1y");
  const [selectedIds, setSelectedIds] = useState([]);
  const [chartType, setChartType] = useState("value"); // "value" or "profit"
  const [refreshing, setRefreshing] = useState(false);
  const [theme, setTheme] = useState(
    localStorage.getItem("theme") || 
    (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light")
  );

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
  }, [theme]);

  const toggleTheme = () => setTheme(theme === "light" ? "dark" : "light");

  // ─── Auth ───────────────────────────────────────────────────────
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

  // ─── Load Holdings ───────────────────────────────────────────────
  useEffect(() => {
    if (!user) return;

    const loadHoldings = async () => {
      const { data, error } = await supabase
        .from("holdings")
        .select("*")
        .eq("user_id", user.id)
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
            dailyChangePct: Number(d.daily_change_pct ?? 0),
            prediction: d.prediction,
            confidence: d.confidence,
            expectedReturn: d.expected_return,
            recommendation: d.recommendation,
            riskLevel: d.risk_level,
          }))
        );
      }
    };

    loadHoldings();
  }, [user]);

  // ─── Helper Functions ───────────────────────────────────────────
  async function fetchLivePrice(ticker) {
    const res = await fetch(`${API_BASE}/api/price/${ticker}`);
    if (!res.ok) throw new Error("Ticker not found on server");
    const data = await res.json();
    return data.price;
  }

  async function getCachedPrediction(ticker) {
    const { data, error } = await supabase
      .from("prediction_cache")
      .select("*")
      .eq("ticker", ticker)
      .maybeSingle();

    if (error || !data) return null;

    const isFresh = new Date(data.updated_at) > new Date(Date.now() - 24 * 60 * 60 * 1000);
    return isFresh ? data : null;
  }

  async function fetchPrediction(ticker) {
    const res = await fetch(`${API_BASE}/api/predict/${ticker}`);
    if (!res.ok) throw new Error("Prediction failed for " + ticker);
    return await res.json();
  }

  async function updatePredictionCache(ticker, prediction) {
    await supabase
      .from("prediction_cache")
      .upsert({
        ticker,
        updated_at: new Date().toISOString(),
        prediction: prediction.prediction,
        confidence: prediction.confidence,
        expected_return: prediction.expected_return,
        recommendation: prediction.recommendation,
        risk_level: prediction.risk_level,
        daily_change_pct: prediction.daily_change_pct ?? null
      });
  }

  async function refreshAllHoldings(holdings, userId) {
    const updated = await Promise.all(
      holdings.map(async (h) => {
        try {
          const res = await fetch(`${API_BASE}/api/price/${h.ticker}`);
          const data = await res.json();

          let prediction = await getCachedPrediction(h.ticker);

          if (!prediction) {
            prediction = await fetchPrediction(h.ticker);
            await updatePredictionCache(h.ticker, prediction);
          }

          const updatedHolding = {
            ...h,
            currentPrice: Number(data.price || h.currentPrice),
            dailyChangePct: Number(data.daily_change_pct || 0),
            prediction: prediction.prediction,
            confidence: prediction.confidence,
            expectedReturn: prediction.expected_return,
            recommendation: prediction.recommendation,
            riskLevel: prediction.risk_level,
          };

          await supabase
            .from("holdings")
            .update({
              current_price: updatedHolding.currentPrice,
              daily_change_pct: updatedHolding.dailyChangePct,
              prediction: updatedHolding.prediction,
              confidence: updatedHolding.confidence,
              expected_return: updatedHolding.expectedReturn,
              recommendation: updatedHolding.recommendation,
              risk_level: updatedHolding.riskLevel,
            })
            .eq("id", h.id);

          return updatedHolding;
        } catch (err) {
          console.error(`Error refreshing ${h.ticker}:`, err);
          return h;
        }
      })
    );

    return updated;
  }


  // ─── Portfolio Calculations ─────────────────────────────────────
  const totalInvested = holdings.reduce(
    (sum, h) => sum + h.purchasePrice * h.shares,
    0
  );
  const totalValue = holdings.reduce(
    (sum, h) => sum + h.currentPrice * h.shares,
    0
  );
  const totalProfit = totalValue - totalInvested;
  const profitPercent = totalInvested > 0 ? (totalProfit / totalInvested) * 100 : 0;

  // Calculate today's profit/loss
  const todaysProfit = holdings.reduce((sum, h) => {
    const holdingValue = h.currentPrice * h.shares;
    const dailyChangePct = h.dailyChangePct || 0;
    // Calculate the previous day's value
    const previousValue = holdingValue / (1 + dailyChangePct / 100);
    const todaysGain = holdingValue - previousValue;
    return sum + todaysGain;
  }, 0);
  
  const yesterdayValue = totalValue - todaysProfit;
  const todaysProfitPercent = yesterdayValue > 0 ? (todaysProfit / yesterdayValue) * 100 : 0;

  // Get earliest investment date
  const earliestDate = holdings.length > 0
    ? holdings.reduce((earliest, h) => {
        if (!h.purchaseDate) return earliest;
        return !earliest || h.purchaseDate < earliest ? h.purchaseDate : earliest;
      }, null)
    : null;

  // ─── Chart Data ─────────────────────────────────────────────────
  useEffect(() => {
    const filteredHoldings = holdings.filter(h => selectedIds.includes(h.id));
    
    if (!filteredHoldings.length) {
      setChartData([]);
      return;
    }

    const buildChart = async () => {
      try {
        const histories = await Promise.all(
          filteredHoldings.map(async (h) => {
            const start = h.purchaseDate || new Date(Date.now() - 365 * 864e5).toISOString().split("T")[0];
            const res = await fetch(`${API_BASE}/api/history/${h.ticker}?start=${start}`);
            if (!res.ok) throw new Error(`Failed to fetch history for ${h.ticker}`);
            return { holding: h, data: await res.json() };
          })
        );

        const valueMap = {};
        
        // Build value map
        histories.forEach(({ holding, data }) => {
          const { shares } = holding;
          data.series.forEach((pt) => {
            valueMap[pt.date] = (valueMap[pt.date] || 0) + pt.close * shares;
          });
        });

        // Build invested map (cumulative investment over time)
        const sortedDates = Object.keys(valueMap).sort();
        const investedByDate = {};
        
        // For each date, calculate total invested up to that point
        sortedDates.forEach((date) => {
          let totalInvestedToDate = 0;
          filteredHoldings.forEach((h) => {
            if (h.purchaseDate && h.purchaseDate <= date) {
              totalInvestedToDate += h.purchasePrice * h.shares;
            }
          });
          investedByDate[date] = totalInvestedToDate;
        });

        const daysMap = { "1d": 1, "5d": 5, "1m": 30, "6m": 183, "1y": 365, "5y": 1825, all: Infinity };
        const maxDays = daysMap[timeline] ?? Infinity;

        const keptDates = maxDays === Infinity
          ? sortedDates
          : sortedDates.filter(
              (d) => d >= new Date(Date.now() - maxDays * 864e5).toISOString().split("T")[0]
            );

        const downsampleRate = Math.ceil(keptDates.length / 50);
        const series = keptDates
          .filter((_, idx) => idx % downsampleRate === 0)
          .map((date) => {
            const value = valueMap[date];
            const invested = investedByDate[date] || 0;
            const profit = value - invested;
            
            return {
              date: new Date(date).toLocaleDateString("en-US", {
                month: "short",
                day: "numeric",
              }),
              value: Number(value.toFixed(2)),
              profit: Number(profit.toFixed(2)),
            };
          });

        setChartData(series);
      } catch (err) {
        console.error("Chart build failed:", err);
        setChartData([]);
      }
    };

    buildChart();
  }, [holdings, timeline, selectedIds]);

  // ─── Asset Allocation Data ──────────────────────────────────────
  const selectedHoldings = holdings.filter(h => selectedIds.includes(h.id));
  const selectedTotalValue = selectedHoldings.reduce(
    (sum, h) => sum + h.currentPrice * h.shares,
    0
  );
  
  const allocationData = selectedHoldings
    .map((h, index) => ({
      name: h.ticker,
      value: h.currentPrice * h.shares,
      percentage: selectedTotalValue > 0 
        ? ((h.currentPrice * h.shares) / selectedTotalValue * 100).toFixed(1)
        : "0",
    }))
    .sort((a, b) => b.value - a.value); // Sort from largest to smallest

  // ─── Add Holding ────────────────────────────────────────────────
  async function handleSubmit() {
    setErrorMsg("");
    setLoading(true);

    const ticker = formData.ticker.trim().toUpperCase();
    const shares = parseFloat(formData.shares);
    const purchasePrice = parseFloat(formData.purchasePrice);
    const purchaseDate = formData.purchaseDate;

    if (!ticker || !shares || !purchasePrice) {
      setErrorMsg("Please fill in all fields correctly.");
      setLoading(false);
      return;
    }

    try {
      const livePrice = await fetchLivePrice(ticker);

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
      setSelectedIds((prev) => [data.id, ...prev]); // Add new holding to selected
      setFormData(emptyForm);
      setShowForm(false);
    } catch (err) {
      setErrorMsg(err.message || "Could not save holding.");
    } finally {
      setLoading(false);
    }
  }

  // ─── Delete Holding ─────────────────────────────────────────────
  const deleteHolding = async (id) => {
    setHoldings((prev) => prev.filter((h) => h.id !== id));
    setSelectedIds((prev) => prev.filter((x) => x !== id)); // Remove from selected
    const { error } = await supabase.from("holdings").delete().eq("id", id);
    if (error) console.error("Delete failed:", error);
  };

  // ─── Edit Holding ───────────────────────────────────────────────
  const startEdit = (h) => {
    setEditingId(h.id);
    setEditForm({
      shares: h.shares.toString(),
      purchasePrice: h.purchasePrice.toString(),
    });
  };

  const cancelEdit = () => {
    setEditingId(null);
    setEditForm({ shares: "", purchasePrice: "" });
  };

  const saveEdit = async (id) => {
    const shares = parseFloat(editForm.shares);
    const purchase_price = parseFloat(editForm.purchasePrice);
    if (!shares || !purchase_price) return;

    const { error } = await supabase
      .from("holdings")
      .update({ shares, purchase_price })
      .eq("id", id)
      .select()
      .single();

    if (error) {
      console.error("Update failed:", error);
      return;
    }

    setHoldings((prev) =>
      prev.map((h) =>
        h.id === id
          ? { ...h, shares, purchasePrice: purchase_price }
          : h
      )
    );
    cancelEdit();
  };

  // ─── Filtered and Sorted Holdings ──────────────────────────────
  const displayedHoldings = holdings
    .filter((h) => h.ticker.toLowerCase().includes(searchQuery.toLowerCase()))
    .sort((a, b) => {
      let aVal, bVal;
      
      switch (sortBy) {
        case "ticker":
          aVal = a.ticker;
          bVal = b.ticker;
          return sortOrder === "asc" 
            ? aVal.localeCompare(bVal)
            : bVal.localeCompare(aVal);
        
        case "ltp":
          aVal = a.currentPrice;
          bVal = b.currentPrice;
          break;
        
        case "shares":
          aVal = a.shares;
          bVal = b.shares;
          break;
        
        case "profit":
          aVal = (a.currentPrice - a.purchasePrice) * a.shares;
          bVal = (b.currentPrice - b.purchasePrice) * b.shares;
          break;
        
        default:
          return 0;
      }
      
      if (sortBy !== "ticker") {
        return sortOrder === "asc" ? aVal - bVal : bVal - aVal;
      }
    });

  // Helper to handle column click for sorting
  const handleSort = (column) => {
    if (sortBy === column) {
      setSortOrder(sortOrder === "asc" ? "desc" : "asc");
    } else {
      setSortBy(column);
      setSortOrder("asc");
    }
  };

  // Helper to toggle selection
  const toggleSelectAll = () => {
    if (selectedIds.length === holdings.length) {
      setSelectedIds([]);
    } else {
      setSelectedIds(holdings.map(h => h.id));
    }
  };

  const toggleSelectOne = (id) => {
    setSelectedIds(prev =>
      prev.includes(id) 
        ? prev.filter(x => x !== id)
        : [...prev, id]
    );
  };

  // Helper to calculate daily change percentage (mock for now)
  const getDailyChange = (holding) => {
    // This would ideally come from API data comparing with yesterday's close
    // For now, using a random mock value between -5% and +5%
    const seed = holding.ticker.charCodeAt(0) + holding.ticker.charCodeAt(1);
    const mockChange = ((seed % 100) - 50) / 10;
    return mockChange;
  };

  if (authLoading) return null;
  if (!user) return <Auth />;

  return (
    <div className="App">
      <header className="App-header">
        <div className="header-content">
          <div className="header-left">
            <h1>OneGlance</h1>
            <p>Your investments, simplified</p>
            {refreshing && (
              <span style={{ fontSize: "0.75rem", color: "var(--text-sub)", marginLeft: "0.5rem" }}>
                Updating prices...
              </span>
            )}
          </div>
          <div style={{ display: "flex", gap: "1rem", alignItems: "center" }}>
            <button
              onClick={async () => {
                if (refreshing || holdings.length === 0) return;
                setRefreshing(true);

                try {
                  const updated = await refreshAllHoldings(holdings, user.id);
                  setHoldings(updated);
                  console.log("All holdings refreshed:", updated);
                } catch (err) {
                  console.error("Refresh error:", err);
                } finally {
                  setRefreshing(false);
                }
              }}
              style={{
                background: "none",
                border: "none",
                cursor: refreshing ? "wait" : "pointer",
                padding: "0.5rem",
                borderRadius: "8px",
                color: "var(--text-sub)",
                transition: "all 0.2s ease",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
              title="Refresh prices"
              disabled={refreshing || holdings.length === 0}
            >
              <TrendingUp size={20} className={refreshing ? "spin" : ""} />
            </button>
            <button className="theme-toggle" onClick={toggleTheme} title="Toggle theme">
              {theme === "light" ? <Moon size={20} /> : <Sun size={20} />}
            </button>
          </div>
        </div>
      </header>

      <main className="dashboard">
        {/* Debug Info - Remove in production */}
        {process.env.NODE_ENV === 'development' && (
          <div style={{ 
            background: "var(--card-bg)", 
            padding: "1rem", 
            marginBottom: "1rem", 
            borderRadius: "8px",
            fontSize: "0.75rem",
            color: "var(--text-sub)"
          }}>
            <p>API URL: {API_BASE || 'Not set'}</p>
            <p>Holdings: {holdings.length} | Selected: {selectedIds.length}</p>
            <p>Today's Profit: ₹{todaysProfit.toFixed(2)} ({todaysProfitPercent.toFixed(2)}%)</p>
          </div>
        )}
        
        {/* Summary Cards */}
        <div className="summary-grid">
          <div className="summary-card">
            <p className="summary-label">Total Value</p>
            <h2 className="summary-value">₹{totalValue.toFixed(0)}</h2>
            <div className={`summary-change ${totalProfit >= 0 ? "positive" : "negative"}`}>
              <span style={{ fontSize: "0.875rem" }}>
                {totalProfit >= 0 ? "+" : ""}₹{Math.abs(totalProfit).toFixed(0)} ({profitPercent >= 0 ? "+" : ""}{profitPercent.toFixed(1)}%)
              </span>
            </div>
          </div>
          
          <div className="summary-card">
            <p className="summary-label">Today's Profit/Loss</p>
            <h2 className="summary-value">
              <span className={todaysProfit >= 0 ? "positive" : "negative"}>
                {todaysProfit >= 0 ? "+" : ""}₹{Math.abs(todaysProfit).toFixed(0)}
              </span>
            </h2>
            <div className={`summary-change ${todaysProfit >= 0 ? "positive" : "negative"}`}>
              {todaysProfit >= 0 ? <TrendingUp size={16} /> : <TrendingDown size={16} />}
              <span>({todaysProfitPercent >= 0 ? "+" : ""}{todaysProfitPercent.toFixed(1)}%)</span>
            </div>
          </div>
          
          <div className="summary-card">
            <p className="summary-label">Invested Since</p>
            <h2 className="summary-value">
              {earliestDate
                ? new Date(earliestDate).toLocaleDateString("en-US", { month: "short", year: "numeric" })
                : "—"}
            </h2>
          </div>
          
          <div className="summary-card">
            <p className="summary-label">Holdings</p>
            <h2 className="summary-value">{holdings.length}</h2>
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="content-grid">
          {/* Chart Section */}
          <div className="chart-section">
            <div className="chart-header">
              <h3>{chartType === "value" ? "Value Over Time" : "Profit/Loss Over Time"}</h3>
              <div style={{ display: "flex", gap: "1rem", alignItems: "center" }}>
                <button
                  onClick={() => setChartType(chartType === "value" ? "profit" : "value")}
                  style={{
                    padding: "0.5rem 1rem",
                    fontSize: "0.875rem",
                    border: "1px solid var(--border)",
                    borderRadius: "8px",
                    background: "var(--bg)",
                    color: "var(--text-main)",
                    cursor: "pointer",
                    transition: "all 0.2s ease",
                  }}
                  onMouseEnter={(e) => e.target.style.borderColor = "var(--accent)"}
                  onMouseLeave={(e) => e.target.style.borderColor = "var(--border)"}
                  title={`Switch to ${chartType === "value" ? "Profit/Loss" : "Value"} view`}
                >
                  {chartType === "value" ? "📊 P/L" : "💰 Value"}
                </button>
                <select
                  className="timeline-select"
                  value={timeline}
                  onChange={(e) => setTimeline(e.target.value)}
                >
                  <option value="1d">1 Day</option>
                  <option value="5d">5 Days</option>
                  <option value="1m">1 Month</option>
                  <option value="6m">6 Months</option>
                  <option value="1y">1 Year</option>
                  <option value="5y">5 Years</option>
                  <option value="all">All-time</option>
                </select>
              </div>
            </div>
            
            {selectedIds.length > 0 && (
              <div style={{ marginBottom: "1rem", fontSize: "0.875rem", color: "var(--text-sub)" }}>
                {selectedIds.length === holdings.length 
                  ? "Showing all holdings" 
                  : `Showing ${selectedIds.length} of ${holdings.length} holdings`}
              </div>
            )}
            
            {chartData.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="date" stroke="var(--text-sub)" tick={{ fontSize: 12 }} />
                  <YAxis
                    stroke="var(--text-sub)"
                    tick={{ fontSize: 12 }}
                    tickFormatter={(v) => {
                      const absValue = Math.abs(v);
                      if (absValue >= 1000) {
                        return `${v < 0 ? '-' : ''}₹${(absValue/1000).toFixed(0)}k`;
                      }
                      return `${v < 0 ? '-' : ''}₹${absValue.toFixed(0)}`;
                    }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "var(--card-bg)",
                      border: "1px solid var(--border)",
                      borderRadius: "8px",
                    }}
                    formatter={(value) => [`₹${value.toFixed(2)}`, chartType === "value" ? "Value" : "Profit/Loss"]}
                  />
                  {chartType === "profit" && (
                    <ReferenceLine y={0} stroke="var(--text-sub)" strokeDasharray="3 3" />
                  )}
                  <Line
                    type="monotone"
                    dataKey={chartType === "value" ? "value" : "profit"}
                    stroke={chartType === "value" ? "var(--accent)" : 
                            (chartData.length > 0 && chartData[chartData.length - 1].profit >= 0) ? "var(--positive)" : "var(--negative)"}
                    strokeWidth={2}
                    dot={false}
                    activeDot={{ r: 5 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="empty-state">
                <p>{selectedIds.length === 0 ? "No holdings selected" : "No chart data available"}</p>
              </div>
            )}
          </div>

          {/* Holdings Table */}
          <div className="holdings-section">
            <div className="holdings-header">
              <h3>Holdings</h3>
              <div style={{ display: "flex", gap: "1rem", alignItems: "center" }}>
                <label style={{ display: "flex", alignItems: "center", gap: "0.5rem", fontSize: "0.875rem" }}>
                  <input
                    type="checkbox"
                    checked={selectedIds.length === holdings.length && holdings.length > 0}
                    onChange={toggleSelectAll}
                  />
                  <span>Select all</span>
                </label>
                <input
                  type="text"
                  className="search-input"
                  placeholder="Search ticker..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  style={{ width: "200px" }}
                />
              </div>
            </div>
            
            <div className="holdings-table-wrapper">
              {displayedHoldings.length > 0 ? (
                <table className="holdings-table">
                  <thead>
                    <tr>
                      <th style={{ width: "40px", textAlign: "center" }}></th>
                      <th onClick={() => handleSort("ticker")}>
                        Ticker {sortBy === "ticker" && <span style={{ fontSize: "0.7rem", marginLeft: "4px" }}>{sortOrder === "asc" ? "↑" : "↓"}</span>}
                      </th>
                      <th onClick={() => handleSort("ltp")} style={{ textAlign: "center" }}>
                        LTP {sortBy === "ltp" && <span style={{ fontSize: "0.7rem", marginLeft: "4px" }}>{sortOrder === "asc" ? "↑" : "↓"}</span>}
                      </th>
                      <th onClick={() => handleSort("shares")} style={{ textAlign: "center" }}>
                        Shares {sortBy === "shares" && <span style={{ fontSize: "0.7rem", marginLeft: "4px" }}>{sortOrder === "asc" ? "↑" : "↓"}</span>}
                      </th>
                      <th style={{ textAlign: "center" }}>
                        Prediction
                      </th>
                      <th onClick={() => handleSort("profit")} style={{ textAlign: "right" }}>
                        Profit/Loss {sortBy === "profit" && <span style={{ fontSize: "0.7rem", marginLeft: "4px" }}>{sortOrder === "asc" ? "↑" : "↓"}</span>}
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {displayedHoldings.map((h) => {
                      const profit = (h.currentPrice - h.purchasePrice) * h.shares;
                      const profitPct = ((h.currentPrice - h.purchasePrice) / h.purchasePrice) * 100;
                      const dailyChange = getDailyChange(h);
                      
                      if (editingId === h.id) {
                        return (
                          <tr key={h.id} className="edit-row">
                            <td colSpan="5">
                              <div className="edit-inputs">
                                <input
                                  type="number"
                                  placeholder="Shares"
                                  value={editForm.shares}
                                  onChange={(e) => setEditForm({ ...editForm, shares: e.target.value })}
                                />
                                <input
                                  type="number"
                                  placeholder="Purchase Price"
                                  value={editForm.purchasePrice}
                                  onChange={(e) => setEditForm({ ...editForm, purchasePrice: e.target.value })}
                                />
                                <div className="edit-actions">
                                  <button className="save-btn" onClick={() => saveEdit(h.id)}>Save</button>
                                  <button className="cancel-btn" onClick={cancelEdit}>Cancel</button>
                                </div>
                              </div>
                            </td>
                          </tr>
                        );
                      }
                      
                      return (
                        <tr key={h.id}>
                          <td style={{ width: "40px", textAlign: "center" }}>
                            <input
                              type="checkbox"
                              checked={selectedIds.includes(h.id)}
                              onChange={() => toggleSelectOne(h.id)}
                            />
                          </td>
                          <td className="ticker-cell">{h.ticker}</td>
                          <td className="ltp-cell">
                            <div className="ltp-value">₹{h.currentPrice.toFixed(2)}</div>
                            <div className={`ltp-change ${(h.dailyChangePct || 0) >= 0 ? "positive" : "negative"}`}>
                              {(h.dailyChangePct || 0) >= 0 ? "+" : ""}{(h.dailyChangePct || 0).toFixed(1)}%
                            </div>
                          </td>
                          <td className="shares-cell">{h.shares}</td>
                          <td className="prediction-cell">
                            {h.prediction ? (
                              <div className="prediction-tooltip-wrapper">
                                <span className={`prediction-badge ${h.prediction.toLowerCase()}`}>
                                  {h.prediction.charAt(0) + h.prediction.slice(1).toLowerCase()}
                                </span>
                                <div className="prediction-tooltip">
                                  <p><strong>Confidence:</strong> {(h.confidence * 100).toFixed(1)}%</p>
                                  <p><strong>Expected Return:</strong> {h.expectedReturn?.toFixed(2)}%</p>
                                  <p><strong>Recommendation:</strong> {h.recommendation}</p>
                                  <p><strong>Risk:</strong> {h.riskLevel}</p>
                                </div>
                              </div>
                            ) : (
                              <span className="prediction-badge neutral">—</span>
                            )}
                          </td>
                          <td className={`profit-cell ${profit >= 0 ? "positive" : "negative"}`}>
                            {profit >= 0 ? "+" : ""}₹{Math.abs(profit).toFixed(0)}
                            <span className="arrow">
                              {profit >= 0 ? <ArrowUp size={12} /> : <ArrowDown size={12} />}
                            </span>
                            <div style={{ fontSize: "0.75rem", opacity: 0.8 }}>
                              ({profitPct >= 0 ? "+" : ""}{profitPct.toFixed(1)}%)
                            </div>
                            <div className="action-buttons">
                              <button onClick={() => startEdit(h)}>
                                <Edit2 size={14} />
                              </button>
                              <button onClick={() => deleteHolding(h.id)}>
                                <Trash2 size={14} />
                              </button>
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              ) : (
                <div className="empty-state">
                  <p>No holdings found</p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Asset Allocation */}
        {selectedHoldings.length > 0 && (
          <div className="allocation-section">
            <div className="allocation-header">
              <h3>Asset Allocation</h3>
              {selectedIds.length < holdings.length && (
                <p style={{ fontSize: "0.875rem", color: "var(--text-sub)", margin: "0.5rem 0 0 0" }}>
                  Showing {selectedIds.length} of {holdings.length} holdings
                </p>
              )}
            </div>
            <div className="allocation-content">
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={allocationData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={100}
                    fill="#8884d8"
                    paddingAngle={2}
                    dataKey="value"
                  >
                    {allocationData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const data = payload[0];
                        return (
                          <div className="custom-tooltip">
                            <p className="label">{data.name}</p>
                            <p className="value">₹{data.value.toFixed(2)}</p>
                            <p className="value">{data.payload.percentage}%</p>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
              
              {/* Legend */}
              <div style={{ display: "flex", flexWrap: "wrap", gap: "1rem", justifyContent: "center", marginTop: "1rem" }}>
                {allocationData.map((entry, index) => (
                  <div key={entry.name} style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                    <div
                      style={{
                        width: "12px",
                        height: "12px",
                        backgroundColor: COLORS[index % COLORS.length],
                        borderRadius: "2px",
                      }}
                    />
                    <span style={{ fontSize: "0.875rem", color: "var(--text-sub)" }}>
                      {entry.name} ({entry.percentage}%)
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Floating Add Button */}
        <button className="add-button" onClick={() => setShowForm(true)} title="Add holding">
          <Plus size={24} />
        </button>

        {/* Add Form Modal */}
        {showForm && (
          <div className="modal-overlay" onClick={() => setShowForm(false)}>
            <div className="modal-content" onClick={(e) => e.stopPropagation()}>
              <div className="modal-header">
                <h3>Add New Holding</h3>
                <button className="close-button" onClick={() => setShowForm(false)}>
                  <X size={20} />
                </button>
              </div>
              
              <div>
                <div className="form-group">
                  <label>Stock Ticker</label>
                  <div className="autocomplete-wrapper">
                    <input
                      type="text"
                      placeholder="e.g., RELIANCE"
                      value={formData.ticker}
                      onChange={(e) => {
                        const input = e.target.value.toUpperCase();
                        setFormData({ ...formData, ticker: input });

                        if (input.length > 1) {
                          const filtered = tickerList.filter((t) =>
                            t.startsWith(input)
                          );
                          setFilteredTickers(filtered.slice(0, 5));
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
                </div>
                
                <div className="form-group">
                  <label>Purchase Date</label>
                  <input
                    type="date"
                    value={formData.purchaseDate}
                    onChange={(e) => setFormData({ ...formData, purchaseDate: e.target.value })}
                    required
                  />
                </div>
                
                <div className="form-group">
                  <label>Number of Shares</label>
                  <input
                    type="number"
                    placeholder="0"
                    value={formData.shares}
                    onChange={(e) => setFormData({ ...formData, shares: e.target.value })}
                    step="1"
                    required
                  />
                </div>
                
                <div className="form-group">
                  <label>Purchase Price per Share</label>
                  <input
                    type="number"
                    placeholder="0.00"
                    value={formData.purchasePrice}
                    onChange={(e) => setFormData({ ...formData, purchasePrice: e.target.value })}
                    step="0.01"
                    required
                  />
                </div>
                
                {errorMsg && <p className="auth-error">{errorMsg}</p>}
                
                <div className="form-actions">
                  <button type="button" className="btn btn-secondary" onClick={() => setShowForm(false)}>
                    Cancel
                  </button>
                  <button type="button" className="btn btn-primary" onClick={handleSubmit} disabled={loading}>
                    {loading ? "Adding..." : "Add Holding"}
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}