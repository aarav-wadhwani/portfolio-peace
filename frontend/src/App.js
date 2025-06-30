import './App.css';
import { useState } from 'react';
import { TrendingUp, Plus } from 'lucide-react';

function App() {
  const [holdings, setHoldings] = useState([]);
  const [showForm, setShowForm] = useState(false);
  
  return (
    <div className="App">
      <header className="App-header">
        <h1>ClearTrack</h1>
        <p>Your calm space for investment tracking</p>
      </header>
      
      <main>
        <div className="portfolio-summary">
          <h2>Portfolio Value</h2>
          <div className="value">$0.00</div>
        </div>
        
        <button className="add-button" onClick={() => setShowForm(true)}>
          <Plus size={20} /> Add Holding
        </button>
        
        {holdings.length === 0 && (
          <p className="empty-state">No holdings yet. Add your first investment!</p>
        )}
      </main>
    </div>
  );
}

export default App;