// @ts-nocheck
import { useState, useMemo } from 'react';
import Plot from 'react-plotly.js';
import { norm, randomMatrix, normalizeRows, matMul, transpose, softmaxRows, entropyRows, getStats } from '../utils/math';

export default function Panel4() {
  const [numLayers, setNumLayers] = useState(24);
  const [lnType, setLnType] = useState('Pre-LN');
  const [fScale, setFScale] = useState(1.0);

  const data = useMemo(() => {
    let dim = 128;
    let x = Array.from({length: dim}, () => Math.random() * 2 - 1);
    let normsX = [];
    let normsF = [];
    let xHistory = [];
    
    const getStats = (v: number[]) => {
      let mean = v.reduce((a,b)=>a+b,0)/v.length;
      let std = Math.sqrt(v.reduce((a,b)=>a+Math.pow(b-mean,2),0)/v.length);
      return {mean, std};
    };

    for(let l=0; l<numLayers; l++) {
      xHistory.push([...x]);
      normsX.push(norm(x));
      let nX = norm(x);
      
      let f: number[] = [];
      if (lnType === 'Pre-LN') {
        // Pre-LN: F(LN(x))
        let {mean, std} = getStats(x);
        let ln_x = x.map(val => (val - mean) / (std + 1e-5));
        
        let f_raw = Array.from({length: dim}, () => Math.random() * 2 - 1);
        let nF = norm(f_raw);
        f = f_raw.map(val => (val / nF) * fScale * Math.sqrt(dim)); // F output variance scales with fScale
        x = x.map((val, i) => val + f[i]);
      } else if (lnType === 'Post-LN') {
        // Post-LN: LN(x + F(x))
        let f_raw = Array.from({length: dim}, () => Math.random() * 2 - 1);
        let nF = norm(f_raw);
        let x_norm = norm(x);
        f = f_raw.map(val => (val / nF) * fScale * x_norm); 
        let unnorm_x = x.map((val, i) => val + f[i]);
        let {mean, std} = getStats(unnorm_x);
        x = unnorm_x.map(val => (val - mean) / (std + 1e-5));
      } else {
        let f_raw = Array.from({length: dim}, () => Math.random() * 2 - 1);
        let nF = norm(f_raw);
        let x_norm = norm(x);
        f = f_raw.map(val => (val / nF) * fScale * x_norm);
        x = x.map((val, i) => val + f[i]);
      }
      normsF.push(norm(f));
    }
    xHistory.push([...x]);

    let W_U = randomMatrix(dim, 500).map(row => row.map(v => v / Math.sqrt(dim))); 
    let logitsHistory = xHistory.map(xl => matMul([xl], W_U)[0]);
    let finalLogits = logitsHistory[logitsHistory.length - 1];
    let finalPred = finalLogits.indexOf(Math.max(...finalLogits));
    
    let probsOfFinal = logitsHistory.map(logits => {
      let max = Math.max(...logits);
      let exps = logits.map(v => Math.exp(v - max));
      let sum = exps.reduce((a,b)=>a+b,0);
      return exps[finalPred] / sum;
    });

    return { normsX, normsF, probsOfFinal };
  }, [numLayers, lnType, fScale]);

  return (
    <div className="panel-grid">
      <div className="controls-col">
        <div className="control-group">
          <div className="slider-container">
            <div className="control-label"><span>Number of Layers</span><span className="control-val">{numLayers}</span></div>
            <input type="range" min="1" max="100" step="1" value={numLayers} onChange={e => setNumLayers(parseInt(e.target.value))} />
          </div>
          <div className="slider-container">
            <div className="control-label"><span>Contribution ||F(x)||</span><span className="control-val">{fScale.toFixed(1)}</span></div>
            <input type="range" min="0.1" max="5.0" step="0.1" value={fScale} onChange={e => setFScale(parseFloat(e.target.value))} />
          </div>
          <div className="radio-group" style={{marginTop: '20px'}}>
            <label className="radio-label"><input type="radio" checked={lnType==='Pre-LN'} onChange={()=>setLnType('Pre-LN')} /> Pre-LN (Modern)</label>
            <label className="radio-label"><input type="radio" checked={lnType==='Post-LN'} onChange={()=>setLnType('Post-LN')} /> Post-LN (Original)</label>
            <label className="radio-label"><input type="radio" checked={lnType==='No-LN'} onChange={()=>setLnType('No-LN')} /> No LayerNorm</label>
          </div>
        </div>
      </div>
      <div className="plots-col">
        <div className="plot-container">
          <div className="math-equation" style={{marginBottom: 0, borderLeftColor: '#06b6d4'}}>{"x^{(l+1)} = \\text{LayerNorm}(x^{(l)} + F^{(l)}(x^{(l)}))"}</div>
          <Plot
            data={[
              { y: data.normsX, type: 'scatter', mode: 'lines+markers', name: '||x|| (Stream)' },
              { y: data.normsF, type: 'scatter', mode: 'lines+markers', name: '||F(x)|| (Layer)' }
            ] as any}
            layout={{ title: 'Residual Norm Growth', paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', font: {color: '#e2e8f0'} } as any}
            useResizeHandler={true} style={{width: '100%', height: '300px'}}
          />
        </div>
        <div className="plot-container">
          <div className="math-equation" style={{marginBottom: 0, borderLeftColor: '#ec4899'}}>{"P(\\text{token}) = \\text{softmax}(x^{(l)} W_U)"}</div>
          <Plot
            data={[{ y: data.probsOfFinal, type: 'scatter', mode: 'lines+markers', name: 'P(Final Token)', marker: {color: '#ec4899'} }] as any}
            layout={{ title: 'Logit Lens: When does prediction stabilize?', paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', font: {color: '#e2e8f0'} } as any}
            useResizeHandler={true} style={{width: '100%', height: '300px'}}
          />
        </div>
      </div>
    </div>
  );
}
