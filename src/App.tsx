// @ts-nocheck
import { useState } from 'react';
import Panel1 from './components/Panel1';
import Panel2 from './components/Panel2';
import Panel3 from './components/Panel3';
import Panel4 from './components/Panel4';
import Panel5 from './components/Panel5';

export default function App() {
  const [activeTab, setActiveTab] = useState(0);
  const tabs = [
    { title: "1. Entropy Collapse", comp: <Panel1 /> },
    { title: "2. Positional Breakdown", comp: <Panel2 /> },
    { title: "3. Head Redundancy", comp: <Panel3 /> },
    { title: "4. Residual Saturation", comp: <Panel4 /> },
    { title: "5. Induction Heads", comp: <Panel5 /> }
  ];

  return (
    <div className="container">
      <h1 className="title">Transformer Failure Mode Visualizer</h1>
      <p className="subtitle">
        An interactive visualizer designed to systematically study the failure modes of the Transformer architecture through raw mathematics. 
        Most educational resources show when transformers work; this visualizer demonstrates when and why they break down.
      </p>
      
      <div className="glass-panel">
        <div className="tabs-header">
          {tabs.map((tab, idx) => (
            <button 
              key={idx} 
              className={`tab-btn ${activeTab === idx ? 'active' : ''}`}
              onClick={() => setActiveTab(idx)}
            >
              {tab.title}
            </button>
          ))}
        </div>
        
        {tabs[activeTab].comp}
      </div>
    </div>
  );
}
