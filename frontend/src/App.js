import './App.css';
import { useState } from 'react';
import { TrendingUp, TrendingDown, Plus, X } from 'lucide-react';

function App() {
  const [holdings, setHoldings] = useState([]);
  const [showForm, setShowForm] = useState(false);
  const [formData, setFormData] = useState({
    ticker: '',
    shares: '',
    purchasePrice: ''
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    
    const newHolding = {
      id: Date.now(),
      ticker: formData.ticker.toUpperCase(),
      shares: parseFloat(formData.shares),
      purchasePrice: parseFloat(formData.purchasePrice),
      currentPrice: parseFloat(formData.purchasePrice) * 1.1, // Mock 10% gain
    };
    
    setHoldings([...holdings, newHolding]);
    setFormData({ ticker: '', shares: '', purchasePrice: '' });
    setShowForm(false);
  };

  const deleteHolding = (id) => {
    setHoldings(holdings.filter(h => h.id !== id));
  };

  const totalValue = holdings.reduce((sum, h) => sum + (h.currentPrice * h.shares), 0);
  const totalInvested = holdings.reduce((sum, h) => sum + (h.purchasePrice * h.shares), 0);
  const totalProfit = totalValue - totalInvested;
  const profitPercent = totalInvested > 0 ? (totalProfit / totalInvested) * 100 : 0;

  return (
    <div className="App">
      <header className="App-header">
        <h1>ClearTrack</h1>
        <p>Your calm space for investment tracking</p>
      </header>
      
      <main>
        <div className="portfolio-summary">
          <h2>Portfolio Value</h2>
          <div className="value">${totalValue.toFixed(2)}</div>
          {totalInvested > 0 && (
            <div className={`profit ${totalProfit >= 0 ? 'positive' : 'negative'}`}>
              {totalProfit >= 0 ? <TrendingUp size={20} /> : <TrendingDown size={20} />}
              ${Math.abs(totalProfit).toFixed(2)} ({profitPercent.toFixed(1)}%)
            </div>
          )}
        </div>

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
              placeholder="Stock ticker (e.g., AAPL)"
              value={formData.ticker}
              onChange={(e) => setFormData({...formData, ticker: e.target.value})}
              required
            />
            
            <input
              type="number"
              placeholder="Number of shares"
              value={formData.shares}
              onChange={(e) => setFormData({...formData, shares: e.target.value})}
              step="0.001"
              required
            />
            
            <input
              type="number"
              placeholder="Purchase price per share"
              value={formData.purchasePrice}
              onChange={(e) => setFormData({...formData, purchasePrice: e.target.value})}
              step="0.01"
              required
            />
            
            <div className="form-buttons">
              <button type="button" onClick={() => setShowForm(false)}>Cancel</button>
              <button type="submit">Add Holding</button>
            </div>
          </form>
        )}

        <div className="holdings">
          {holdings.map(holding => {
            const value = holding.currentPrice * holding.shares;
            const cost = holding.purchasePrice * holding.shares;
            const profit = value - cost;
            const profitPercent = ((value - cost) / cost) * 100;
            
            return (
              <div key={holding.id} className="holding-card">
                <div className="holding-header">
                  <h3>{holding.ticker}</h3>
                  <button className="delete-btn" onClick={() => deleteHolding(holding.id)}>
                    <X size={18} />
                  </button>
                </div>
                <p className="shares">{holding.shares} shares @ ${holding.purchasePrice}</p>
                <div className="holding-value">
                  <span>Current Value: ${value.toFixed(2)}</span>
                  <span className={profit >= 0 ? 'positive' : 'negative'}>
                    {profit >= 0 ? '+' : ''}{profitPercent.toFixed(1)}%
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

export default App;