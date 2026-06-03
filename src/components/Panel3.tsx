// @ts-nocheck
import { useState, useMemo } from 'react';
import Plot from 'react-plotly.js';
import { norm, randomMatrix, normalizeRows, matMul, transpose, softmaxRows, entropyRows, getStats } from '../utils/math';

export default function Panel3() {
  const [numHeads, setNumHeads] = useState(8);
  const [weightCorr, setWeightCorr] = useState(0.8);
  const seqLen = 50;

  const jsMatrix = useMemo(() => {
    // Math principled redundancy:
    // Generate a shared sequence X
    let dModel = 64;
    let dK = 32;
    let X = randomMatrix(seqLen, dModel);
    
    // Generate a base projection matrices that heads will correlate with
    let WQ_base = randomMatrix(dModel, dK);
    let WK_base = randomMatrix(dModel, dK);
    
    let heads: number[][][] = [];
    for(let h=0; h<numHeads; h++) {
      let WQ_noise = randomMatrix(dModel, dK);
      let WK_noise = randomMatrix(dModel, dK);
      
      // The projection matrices for this head are a mix of the base pattern and noise
      let WQ_h = WQ_base.map((row, i) => row.map((val, j) => weightCorr * val + Math.sqrt(1 - weightCorr*weightCorr) * WQ_noise[i][j]));
      let WK_h = WK_base.map((row, i) => row.map((val, j) => weightCorr * val + Math.sqrt(1 - weightCorr*weightCorr) * WK_noise[i][j]));
      
      let Q = matMul(X, WQ_h);
      let K = matMul(X, WK_h);
      let logits = matMul(Q, transpose(K)).map(row => row.map(v => v / Math.sqrt(dK)));
      heads.push(softmaxRows(logits));
    }
    
    const jsDiv = (p: number[], q: number[]) => {
      let m = p.map((x,i) => 0.5 * (x + q[i]));
      let kld1 = p.reduce((sum, x, i) => sum + (x>0 ? x * Math.log(x/m[i]) : 0), 0);
      let kld2 = q.reduce((sum, x, i) => sum + (x>0 ? x * Math.log(x/m[i]) : 0), 0);
      return 0.5 * kld1 + 0.5 * kld2;
    };

    let matrix = Array.from({length: numHeads}, () => new Array(numHeads).fill(0));
    for(let i=0; i<numHeads; i++){
      for(let j=0; j<numHeads; j++){
        if (i === j) continue;
        let sum = 0;
        for(let p=0; p<seqLen; p++){
          sum += Math.pow(jsDiv(heads[i][p], heads[j][p]), 2);
        }
        matrix[i][j] = sum / seqLen;
      }
    }
    return matrix;
  }, [numHeads, weightCorr]);

  return (
    <div className="panel-grid">
      <div className="controls-col">
        <div className="control-group">
          <div className="slider-container">
            <div className="control-label"><span>Number of Heads</span><span className="control-val">{numHeads}</span></div>
            <input type="range" min="2" max="32" step="1" value={numHeads} onChange={e => setNumHeads(parseInt(e.target.value))} />
          </div>
          <div className="slider-container">
            <div className="control-label"><span>Projection Weight Correlation</span><span className="control-val">{weightCorr.toFixed(2)}</span></div>
            <input type="range" min="0" max="0.99" step="0.05" value={weightCorr} onChange={e => setWeightCorr(parseFloat(e.target.value))} />
          </div>
          <p style={{fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: 12}}>
            We simulate the sequence <strong>X</strong> passing through parameterized projection matrices <strong>W_Q</strong> and <strong>W_K</strong>. By increasing correlation, heads share parameter subspaces, creating true identical attention patterns (a mathematically robust view of redundancy).
          </p>
        </div>
      </div>
      <div className="plots-col">
        <div className="plot-container">
          <div className="math-equation" style={{marginBottom: 0, borderLeftColor: '#f59e0b'}}>
            {"W_Q^{(h)} = \\gamma W_Q^{base} + \\sqrt{1-\\gamma^2} \\epsilon \\quad\\rightarrow\\quad \\rho(l, l') = \\frac{1}{n} \\sum JS(a_i^{(l)} \\| a_i^{(l')})"}
          </div>
          <Plot
            data={[{ z: jsMatrix, type: 'heatmap', colorscale: 'Plasma' }] as any}
            layout={{ title: 'Head JS-Divergence Matrix (Lower = More Redundant)', paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', font: {color: '#e2e8f0'}, margin: {t:40} } as any}
            useResizeHandler={true} style={{width: '100%', height: '550px'}}
          />
        </div>
      </div>
    </div>
  );
}
