// @ts-nocheck
import { useState, useMemo } from 'react';
import Plot from 'react-plotly.js';
import { norm, randomMatrix, normalizeRows, matMul, transpose, softmaxRows, entropyRows, getStats } from '../utils/math';

export default function Panel5() {
  const [coupling, setCoupling] = useState(0.9);

  const data = useMemo(() => {
    // Sequence: [Harry, Potter, and, Harry, _] -> expecting 'Potter'
    // Let's abstract this to 5 tokens, vocab size 4.
    // T1: 0 (Harry), T2: 1 (Potter), T3: 2 (and), T4: 0 (Harry), T5: ? (We look at row 4 attention)
    
    let seq = [0, 1, 2, 0, 3];
    let n = seq.length;
    let dModel = 16;
    
    // Create distinct embeddings for the 4 vocab items
    let E_vocab = randomMatrix(4, dModel);
    let E = seq.map(idx => E_vocab[idx]);
    
    // Layer 1: Previous Token Head
    // Hardcode an attention matrix that strictly looks at the previous token
    let A1 = Array.from({length: n}, (_, i) => Array.from({length: n}, (_, j) => (i - 1 === j && i > 0) ? 1 : 0));
    // Let's pretend W_V and W_O in layer 1 are Identity for simplicity, so O1 = A1 * E
    let O1 = matMul(A1, E); 
    
    // Layer 2: Induction Head
    // Q depends on current token (E)
    // K depends on context (O1, which holds the PREVIOUS token's embedding)
    // If we want K to match Q when the previous token of j matches current token of i, 
    // we need K to pull from O1, and Q to pull from E, and W_Q * W_K^T ≈ Identity.
    
    let W_Q2 = randomMatrix(dModel, dModel);
    let W_K2_base = W_Q2; // Perfect matching
    let W_K2_noise = randomMatrix(dModel, dModel);
    
    // Blend the perfect K projection with noise based on coupling
    let W_K2 = W_K2_base.map((row, i) => row.map((val, j) => coupling * val + Math.sqrt(1 - coupling*coupling) * W_K2_noise[i][j]));
    
    // Q is derived from current embedding E
    let Q = matMul(E, W_Q2);
    // K is derived from Layer 1 output O1
    let K = matMul(O1, W_K2);
    
    let logits = matMul(Q, transpose(K));
    // Apply causal mask and softmax
    let maskedLogits = logits.map((row, i) => row.map((val, j) => j <= i ? val * 2.0 : -1e9)); // Scale for sharpness
    let A2 = softmaxRows(maskedLogits);
    
    // Text labels for axes
    let labels = ["Harry (0)", "Potter (1)", "and (2)", "Harry (3)", "[NEXT] (4)"];
    
    return { A2, labels };
  }, [coupling]);

  return (
    <div className="panel-grid">
      <div className="controls-col">
        <div className="control-group">
          <div className="slider-container">
            <div className="control-label"><span>Induction Coupling Strength</span><span className="control-val">{coupling.toFixed(2)}</span></div>
            <input type="range" min="0" max="1" step="0.05" value={coupling} onChange={e => setCoupling(parseFloat(e.target.value))} />
          </div>
          <div style={{color: 'var(--text-muted)', fontSize: '0.9rem', lineHeight: '1.6', marginTop: '20px'}}>
            <strong>Sequence:</strong> <code>[Harry, Potter, and, Harry, ?]</code><br/><br/>
            <strong>Layer 1</strong> extracts the previous token. <br/>
            <strong>Layer 2</strong> compares its current token to Layer 1's output. <br/><br/>
            When coupling is high, the <strong>OV-K</strong> circuit operates perfectly. Look at the last row (predicting after the second 'Harry'). The attention spikes precisely on 'Potter' — this is the mathematical origin of <strong>in-context learning</strong>.
          </div>
        </div>
      </div>
      <div className="plots-col">
        <div className="plot-container">
          <div className="math-equation" style={{marginBottom: 0, borderLeftColor: '#f43f5e'}}>{"Q^{(2)} = E W_Q^{(2)}, \\quad K^{(2)} = (A^{(1)} E W_{OV}^{(1)}) W_K^{(2)}"}</div>
          <Plot
            data={[{ z: data.A2, x: data.labels, y: data.labels, type: 'heatmap', colorscale: 'Blues' }] as any}
            layout={{ 
              title: 'Layer 2 Attention (Induction Head)', 
              paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', font: {color: '#e2e8f0'},
              xaxis: {title: 'Key (Context)', side: 'bottom'},
              yaxis: {title: 'Query (Current Token)', autorange: 'reversed'}
            } as any}
            useResizeHandler={true} style={{width: '100%', height: '550px'}}
          />
        </div>
      </div>
    </div>
  );
}
