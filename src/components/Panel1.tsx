// @ts-nocheck
import { useState, useMemo } from 'react';
import Plot from 'react-plotly.js';
import { norm, randomMatrix, normalizeRows, matMul, transpose, softmaxRows, entropyRows, getStats } from '../utils/math';

export default function Panel1() {
  const [qNorm, setQNorm] = useState(1.0);
  const [kNorm, setKNorm] = useState(1.0);
  const [dk, setDk] = useState(64);
  const [tau, setTau] = useState(1.0);
  const [seqLen, setSeqLen] = useState(50);

  const data = useMemo(() => {
    let Q = normalizeRows(randomMatrix(seqLen, dk), qNorm);
    let K = normalizeRows(randomMatrix(seqLen, dk), kNorm);
    let logits = matMul(Q, transpose(K));
    let scaledLogits = logits.map(row => row.map(x => x / (Math.sqrt(dk) * tau)));
    let A = softmaxRows(scaledLogits);
    let entropy = entropyRows(A);
    let gradMag = A.map(row => row.map(x => x * (1 - x)));
    return { A, entropy, gradMag };
  }, [qNorm, kNorm, dk, tau, seqLen]);

  return (
    <div className="panel-grid">
      <div className="controls-col">
        <div className="control-group">
          <div className="slider-container">
            <div className="control-label"><span>Query Norm ||q||</span><span className="control-val">{qNorm.toFixed(1)}</span></div>
            <input type="range" min="0.1" max="50.0" step="0.1" value={qNorm} onChange={e => setQNorm(parseFloat(e.target.value))} />
          </div>
          <div className="slider-container">
            <div className="control-label"><span>Key Norm ||k||</span><span className="control-val">{kNorm.toFixed(1)}</span></div>
            <input type="range" min="0.1" max="50.0" step="0.1" value={kNorm} onChange={e => setKNorm(parseFloat(e.target.value))} />
          </div>
          <div className="slider-container">
            <div className="control-label"><span>Dimension (d_k)</span><span className="control-val">{dk}</span></div>
            <input type="range" min="1" max="512" step="1" value={dk} onChange={e => setDk(parseInt(e.target.value))} />
          </div>
          <div className="slider-container">
            <div className="control-label"><span>Temperature (τ)</span><span className="control-val">{tau.toFixed(1)}</span></div>
            <input type="range" min="0.1" max="5.0" step="0.1" value={tau} onChange={e => setTau(parseFloat(e.target.value))} />
          </div>
          <div className="slider-container">
            <div className="control-label"><span>Sequence Length</span><span className="control-val">{seqLen}</span></div>
            <input type="range" min="10" max="200" step="10" value={seqLen} onChange={e => setSeqLen(parseInt(e.target.value))} />
          </div>
        </div>
      </div>
      <div className="plots-col">
        <div className="plots-row">
          <div className="plot-container">
            <div className="math-equation" style={{marginBottom: 0, borderLeftColor: '#10b981'}}>{"a_ij = softmax(q_i · k_j / (\\tau \\sqrt{d_k}))"}</div>
            <Plot
              data={[{ z: data.A, type: 'heatmap', colorscale: 'Viridis' }] as any}
              layout={{ title: 'Attention Weights (A)', paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', font: {color: '#e2e8f0'}, margin: {t:40,l:30,r:30,b:30} } as any}
              useResizeHandler={true} style={{width: '100%', height: '260px'}}
            />
          </div>
          <div className="plot-container">
            <div className="math-equation" style={{marginBottom: 0, borderLeftColor: '#ef4444'}}>{"\\nabla_{z_i} a_i \\approx a_i(1 - a_i)"}</div>
            <Plot
              data={[{ z: data.gradMag, type: 'heatmap', colorscale: 'Reds', zmin: 0, zmax: 0.25 }] as any}
              layout={{ title: 'Gradient Vanishing Heatmap', paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', font: {color: '#e2e8f0'}, margin: {t:40,l:30,r:30,b:30} } as any}
              useResizeHandler={true} style={{width: '100%', height: '260px'}}
            />
          </div>
        </div>
        <div className="plot-container">
          <div className="math-equation" style={{marginBottom: 0, borderLeftColor: '#3b82f6'}}>{"H(a_i) = -\\sum a_{ij} \\log a_{ij}"}</div>
          <Plot
            data={[{ y: data.entropy, type: 'scatter', mode: 'lines' }] as any}
            layout={{ 
              title: 'Attention Entropy per Position', paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', font: {color: '#e2e8f0'}, margin: {t:40,l:40,r:30,b:40},
              shapes: [{type: 'line', y0: Math.log(seqLen), y1: Math.log(seqLen), x0: 0, x1: seqLen, line: {color: 'rgba(255,255,255,0.3)', dash: 'dash'}}]
            } as any}
            useResizeHandler={true} style={{width: '100%', height: '260px'}}
          />
        </div>
      </div>
    </div>
  );
}
