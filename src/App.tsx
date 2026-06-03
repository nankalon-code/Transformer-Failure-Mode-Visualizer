import { useState, useMemo } from 'react';
import Plot from 'react-plotly.js';

// --- MATH HELPERS ---
const norm = (v: number[]) => Math.sqrt(v.reduce((s, x) => s + x * x, 0));
const randomMatrix = (rows: number, cols: number) => 
  Array.from({length: rows}, () => Array.from({length: cols}, () => Math.random() * 2 - 1));
const normalizeRows = (A: number[][], targetNorm = 1) => A.map(row => {
  let n = norm(row);
  return row.map(x => (x / (n || 1)) * targetNorm);
});
const matMul = (A: number[][], B: number[][]) => 
  A.map(row => B[0].map((_, j) => row.reduce((sum, val, i) => sum + val * B[i][j], 0)));
const transpose = (A: number[][]) => A[0].map((_, c) => A.map(r => r[c]));
const softmaxRows = (A: number[][]) => A.map(row => {
  let max = Math.max(...row);
  let exps = row.map(x => Math.exp(x - max));
  let sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
});
const entropyRows = (A: number[][]) => A.map(row => -row.reduce((sum, p) => sum + p * Math.log(p + 1e-12), 0));

// --- PANELS ---
function Panel1() {
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
        <div className="math-equation">{"a_ij = softmax(q_i · k_j / (\\tau \\sqrt{d_k}))"}</div>
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
            <Plot
              data={[{ z: data.A, type: 'heatmap', colorscale: 'Viridis' }] as any}
              layout={{ title: 'Attention Weights (A)', paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', font: {color: '#e2e8f0'}, margin: {t:40,l:30,r:30,b:30} } as any}
              useResizeHandler={true} style={{width: '100%', height: '300px'}}
            />
          </div>
          <div className="plot-container">
            <Plot
              data={[{ z: data.gradMag, type: 'heatmap', colorscale: 'Reds', zmin: 0, zmax: 0.25 }] as any}
              layout={{ title: 'Gradient Vanishing Heatmap', paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', font: {color: '#e2e8f0'}, margin: {t:40,l:30,r:30,b:30} } as any}
              useResizeHandler={true} style={{width: '100%', height: '300px'}}
            />
          </div>
        </div>
        <div className="plot-container">
          <Plot
            data={[{ y: data.entropy, type: 'scatter', mode: 'lines' }] as any}
            layout={{ 
              title: 'Attention Entropy per Position', paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', font: {color: '#e2e8f0'}, margin: {t:40,l:40,r:30,b:40},
              shapes: [{type: 'line', y0: Math.log(seqLen), y1: Math.log(seqLen), x0: 0, x1: seqLen, line: {color: 'rgba(255,255,255,0.3)', dash: 'dash'}}]
            } as any}
            useResizeHandler={true} style={{width: '100%', height: '300px'}}
          />
        </div>
      </div>
    </div>
  );
}

function Panel2() {
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
      for(let m=0; m<seqLen; m++){
        for(let n=0; n<seqLen; n++) sim[m][n] = -mSlope * Math.abs(m-n);
      }
    }
    return sim;
  }, [seqLen, encType, dModel]);

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
          <Plot
            data={[{ z: simMatrix, type: 'heatmap', colorscale: 'RdBu' }] as any}
            layout={{ 
              title: `${encType} Positional Similarity`, paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', font: {color: '#e2e8f0'},
              shapes: [
                {type: 'line', x0: trainLen, x1: trainLen, y0: 0, y1: seqLen, line: {color: 'rgba(239,68,68,0.8)', dash: 'dash', width: 2}},
                {type: 'line', y0: trainLen, y1: trainLen, x0: 0, x1: seqLen, line: {color: 'rgba(239,68,68,0.8)', dash: 'dash', width: 2}}
              ]
            } as any}
            useResizeHandler={true} style={{width: '100%', height: '600px'}}
          />
        </div>
      </div>
    </div>
  );
}

function Panel3() {
  const [numHeads, setNumHeads] = useState(8);
  const [redundancy, setRedundancy] = useState(0.5);
  const seqLen = 50;

  const jsMatrix = useMemo(() => {
    let baseLogits = randomMatrix(seqLen, seqLen);
    let heads: number[][][] = [];
    for(let h=0; h<numHeads; h++) {
      let noise = randomMatrix(seqLen, seqLen);
      let logits = baseLogits.map((row, i) => row.map((x, j) => redundancy * x + (1 - redundancy) * noise[i][j]));
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
  }, [numHeads, redundancy]);

  return (
    <div className="panel-grid">
      <div className="controls-col">
        <div className="math-equation">ρ(l, l') = 1/n ∑ JS(a_i^(l) || a_i^(l'))</div>
        <div className="control-group">
          <div className="slider-container">
            <div className="control-label"><span>Number of Heads</span><span className="control-val">{numHeads}</span></div>
            <input type="range" min="2" max="32" step="1" value={numHeads} onChange={e => setNumHeads(parseInt(e.target.value))} />
          </div>
          <div className="slider-container">
            <div className="control-label"><span>Redundancy Factor</span><span className="control-val">{redundancy.toFixed(2)}</span></div>
            <input type="range" min="0" max="1" step="0.05" value={redundancy} onChange={e => setRedundancy(parseFloat(e.target.value))} />
          </div>
        </div>
      </div>
      <div className="plots-col">
        <div className="plot-container">
          <Plot
            data={[{ z: jsMatrix, type: 'heatmap', colorscale: 'Plasma' }] as any}
            layout={{ title: 'Head JS-Divergence (Lower = More Redundant)', paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', font: {color: '#e2e8f0'} } as any}
            useResizeHandler={true} style={{width: '100%', height: '600px'}}
          />
        </div>
      </div>
    </div>
  );
}

function Panel4() {
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
      
      let f = Array.from({length: dim}, () => Math.random() * 2 - 1);
      let nF = norm(f);
      
      if (lnType === 'Pre-LN') {
        f = f.map(val => (val / nF) * fScale);
        x = x.map((val, i) => val + f[i]);
      } else if (lnType === 'Post-LN') {
        f = f.map(val => (val / nF) * fScale * (l > 0 ? nX : 1.0));
        x = x.map((val, i) => val + f[i]);
        let {mean, std} = getStats(x);
        x = x.map(val => (val - mean) / (std + 1e-5));
      } else {
        f = f.map(val => (val / nF) * fScale);
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
        <div className="math-equation">{"x^(l+1) = x^(l) + F^(l)(x^(l))"}</div>
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
          <Plot
            data={[{ y: data.probsOfFinal, type: 'scatter', mode: 'lines+markers', name: 'P(Final Token)', marker: {color: '#a855f7'} }] as any}
            layout={{ title: 'Logit Lens: When does prediction stabilize?', paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', font: {color: '#e2e8f0'} } as any}
            useResizeHandler={true} style={{width: '100%', height: '300px'}}
          />
        </div>
      </div>
    </div>
  );
}

function Panel5() {
  const [coupling, setCoupling] = useState(0.8);

  const WOVK = useMemo(() => {
    let dHead = 32;
    let identity = Array.from({length: dHead}, (_, i) => Array.from({length: dHead}, (_, j) => i===j ? 1 : 0));
    let noise = randomMatrix(dHead, dHead).map(row => row.map(v => v / Math.sqrt(dHead)));
    
    return identity.map((row, i) => row.map((idVal, j) => coupling * idVal + (1 - coupling) * noise[i][j]));
  }, [coupling]);

  return (
    <div className="panel-grid">
      <div className="controls-col">
        <div className="control-group">
          <div className="slider-container">
            <div className="control-label"><span>Circuit Coupling Strength</span><span className="control-val">{coupling.toFixed(2)}</span></div>
            <input type="range" min="0" max="1" step="0.05" value={coupling} onChange={e => setCoupling(parseFloat(e.target.value))} />
          </div>
          <div style={{color: 'var(--text-muted)', fontSize: '0.9rem', lineHeight: '1.6', marginTop: '20px'}}>
            <strong>1.0</strong> = Perfect copying. <br/><strong>0.0</strong> = Random communication. <br/><br/>
            When the diagonal is strong, Layer 2's Keys directly read the Values extracted by Layer 1. 
            Since Layer 1 attends to the previous token, Layer 2 effectively asks: "Have I seen the current token before? If so, what came after it?" — solving in-context learning mathematically.
          </div>
        </div>
      </div>
      <div className="plots-col">
        <div className="plot-container">
          <Plot
            data={[{ z: WOVK, type: 'heatmap', colorscale: 'RdBu' }] as any}
            layout={{ title: 'OV-K Composition Matrix (Layer 1 -> 2)', paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', font: {color: '#e2e8f0'} } as any}
            useResizeHandler={true} style={{width: '100%', height: '600px'}}
          />
        </div>
      </div>
    </div>
  );
}

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
