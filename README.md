# Transformer Failure Mode Visualizer

An interactive visualizer designed to systematically study the failure modes of the Transformer architecture through raw mathematics. Most educational resources show when transformers work; this visualizer demonstrates when and why they break down.

## Five Primary Mechanisms & Failure Regimes
1. **Attention Entropy Collapse & Gradient Vanishing**: Drag $d_k$ and the temperature $\tau$ to watch the distribution collapse from a smooth surface to a spike, seeing exactly the moment the gradient (plotted locally) vanishes.
2. **Positional Encoding Breakdown**: Manipulate sequence lengths past training boundaries to observe extrapolation failure across Sinusoidal, RoPE, and ALiBi encodings.
3. **Attention Head Redundancy**: Compute the JS-divergence of attention distributions to build intuition on head similarity.
4. **Residual Stream Saturation & Logit Lens**: Simulate residual norm growth across layers to find the exact crossover point where Layer Normalization discards early signal, and watch the Logit Lens projection stabilize.
5. **Induction Heads**: Analyze the OV-K composition circuit mathematically to see how 2-layer transformers perform in-context learning.

## Tech Stack
- **React / Next.js ready** (Vite build)
- **TypeScript**
- **Plotly.js** for high-performance interactive visualizations
- **Raw Math**: Custom JS implementations of dot products, JS-divergence, and softmax gradients without heavy ML frameworks.

## Running Locally

1. Install dependencies:
```bash
npm install
```

2. Run the application:
```bash
npm run dev
```

## Deployment (Vercel)

This repository is built as a standard React single-page application using Vite. It is instantly deployable on Vercel:
1. Connect this repository to your Vercel account.
2. The framework preset should automatically be detected as **Vite**.
3. Deploy! The Build command will automatically be `npm run build` and output directory `dist`.
