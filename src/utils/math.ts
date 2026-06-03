export const norm = (v: number[]) => Math.sqrt(v.reduce((s, x) => s + x * x, 0));

export const randomMatrix = (rows: number, cols: number) => 
  Array.from({length: rows}, () => Array.from({length: cols}, () => (Math.random() * 2 - 1) / Math.sqrt(cols)));

export const normalizeRows = (A: number[][], targetNorm = 1) => A.map(row => {
  let n = norm(row);
  return row.map(x => (x / (n || 1)) * targetNorm);
});

export const matMul = (A: number[][], B: number[][]) => 
  A.map(row => B[0].map((_, j) => row.reduce((sum, val, i) => sum + val * B[i][j], 0)));

export const transpose = (A: number[][]) => A[0].map((_, c) => A.map(r => r[c]));

export const softmaxRows = (A: number[][]) => A.map(row => {
  let max = Math.max(...row);
  let exps = row.map(x => Math.exp(x - max));
  let sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
});

export const entropyRows = (A: number[][]) => A.map(row => -row.reduce((sum, p) => sum + p * Math.log(p + 1e-12), 0));

export const getStats = (v: number[]) => {
  let mean = v.reduce((a,b)=>a+b,0)/v.length;
  let std = Math.sqrt(v.reduce((a,b)=>a+Math.pow(b-mean,2),0)/v.length);
  return {mean, std};
};
