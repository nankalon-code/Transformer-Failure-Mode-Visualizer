// @ts-nocheck
import { useState, useMemo } from 'react';
import Plot from 'react-plotly.js';
import { norm, randomMatrix, normalizeRows, matMul, transpose, softmaxRows, entropyRows, getStats } from '../utils/math';

export default function Panel2() {
  const [seqLen, setSeqLen] = useState(200);
  const [trainLen, setTrainLen] = useState(100);
  const [encType, setEncType] = useState('Sinusoidal');
  const [dModel, setDModel] = useState(64);

  const simMatrix = useMemo(() => {
    let sim = Array.from({length: seqLen}, () => new Array(seqLen).fill(0));
    if (encType === 'Sinusoidal') {
      let pe = Array.from({length: seqLen}, () => new Array(dModel).fill(0));
      for(let p=0; p<seqLen; p++){
        for(let i=0; i<dModel/2; i++){
          let div = Math.pow(10000, (2*i)/dModel);
          pe[p][2*i] = Math.sin(p / div);
          if (2*i+1 < dModel) pe[p][2*i+1] = Math.cos(p / div);
        }
      }
      sim = matMul(pe, transpose(pe));
    } else if (encType === 'RoPE') {
      for(let m=0; m<seqLen; m++){
        for(let n=0; n<seqLen; n++){
          let sum = 0;
          for(let i=0; i<dModel/2; i++){
            let theta = Math.pow(10000, -2*i/dModel);
            sum += Math.cos((m-n) * theta);
          }
          sim[m][n] = sum;
        }
      }
    } else if (encType === 'ALiBi') {
      let mSlope = 0.5;
      // ALiBi adds bias to the base QK similarity. We simulate a base identity-like QK similarity.
      for(let m=0; m<seqLen; m++){
        for(let n=0; n<seqLen; n++) {
           let baseQK = (m === n) ? 5.0 : Math.random();
           sim[m][n] = baseQK - mSlope * Math.abs(m-n);
        }
      }
    }
    return sim;
  }, [seqLen, encType, dModel]);
  
  const getEquation = () => {
    if (encType === 'Sinusoidal') return "PE(p, 2i) = \\sin(p / 10000^{2i/d})";
    if (encType === 'RoPE') return "q_m^T k_n = (R_m q)^T (R_n k) \\propto \\cos((m-n)\\theta)";
    return "\\text{logit}_{ij} = q_i \\cdot k_j - m \\cdot |i - j|";
  };

  return (
    <div className="panel-grid">
      <div className="controls-col">
        <div className="control-group">
          <div className="slider-container">
            <div className="control-label"><span>Test Sequence Length</span><span className="control-val">{seqLen}</span></div>
            <input type="range" min="10" max="500" step="10" value={seqLen} onChange={e => setSeqLen(parseInt(e.target.value))} />
          </div>
          <div className="slider-container">
            <div className="control-label"><span>Train Sequence Length Boundary</span><span className="control-val">{trainLen}</span></div>
            <input type="range" min="10" max="300" step="10" value={trainLen} onChange={e => setTrainLen(parseInt(e.target.value))} />
          </div>
          <div className="slider-container">
            <div className="control-label"><span>d_model</span><span className="control-val">{dModel}</span></div>
            <input type="range" min="16" max="256" step="16" value={dModel} onChange={e => setDModel(parseInt(e.target.value))} />
          </div>
          <div className="radio-group" style={{marginTop: '20px'}}>
            <label className="radio-label"><input type="radio" checked={encType==='Sinusoidal'} onChange={()=>setEncType('Sinusoidal')} /> Sinusoidal (Vaswani)</label>
            <label className="radio-label"><input type="radio" checked={encType==='RoPE'} onChange={()=>setEncType('RoPE')} /> RoPE (Rotary)</label>
            <label className="radio-label"><input type="radio" checked={encType==='ALiBi'} onChange={()=>setEncType('ALiBi')} /> ALiBi (Linear Penalty)</label>
          </div>
        </div>
      </div>
      <div className="plots-col">
        <div className="plot-container">
          <div className="math-equation" style={{marginBottom: 0, borderLeftColor: '#8b5cf6'}}>{getEquation()}</div>
          <Plot
            data={[{ z: simMatrix, type: 'heatmap', colorscale: 'RdBu' }] as any}
            layout={{ 
              title: encType === 'ALiBi' ? 'ALiBi Attention Logits (Base QK + Bias)' : `${encType} Positional Similarity Matrix`, paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', font: {color: '#e2e8f0'},
              shapes: [
                {type: 'line', x0: trainLen, x1: trainLen, y0: 0, y1: seqLen, line: {color: 'rgba(239,68,68,0.8)', dash: 'dash', width: 2}},
                {type: 'line', y0: trainLen, y1: trainLen, x0: 0, x1: seqLen, line: {color: 'rgba(239,68,68,0.8)', dash: 'dash', width: 2}}
              ]
            } as any}
            useResizeHandler={true} style={{width: '100%', height: '550px'}}
          />
        </div>
      </div>
    </div>
  );
}
