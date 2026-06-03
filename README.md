# Transformer Failure Mode Visualizer
**Live Demo:** [Deploy with Vercel](https://vercel.com/new) (Zero-config Vite preset)

An interactive visualizer designed to systematically study the failure modes of the Transformer architecture through raw mathematics. Most educational resources show when transformers work; this visualizer demonstrates when and why they break down.

## Mathematical Regimes Explored

### 1. Attention Entropy Collapse
Examines how scaling dot-product attention can cause the Softmax function to collapse to a single point, resulting in vanishing gradients:
- $\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\tau\sqrt{d_k}}\right)V$
- **Failure State**: When the variance of $QK^T$ grows, the distribution becomes sparse, leading to a Jacobian $\nabla_{z_i} a_i \approx a_i(1 - a_i)$ approaching zero.

### 2. Positional Encoding Breakdown
Studies extrapolation failures for common positional encoding schemes beyond training sequence lengths:
- **Sinusoidal**: Evaluates period collision and high-frequency decay.
- **RoPE (Rotary Position Embedding)**: $q_m^T k_n \propto \cos((m-n)\theta)$. Demonstrates the decay of relative positional information at unobserved distances.
- **ALiBi (Attention with Linear Biases)**: $\text{logit}_{ij} = q_i \cdot k_j - m \cdot |i - j|$.

### 3. Attention Head Redundancy
Analyzes structural collapse in multi-head attention where heads degenerate into identical subspaces.
- Computed using Jensen-Shannon Divergence: $JS(P || Q) = \frac{1}{2} KL(P || M) + \frac{1}{2} KL(Q || M)$ where $M = \frac{1}{2}(P+Q)$.
- **Failure State**: High parameter correlation forces multiple $W_Q^{(h)}, W_K^{(h)}$ matrices to attend to identical tokens, wasting representational capacity.

### 4. Residual Stream Saturation
Tracks the $L_2$ norm evolution of the residual stream under different LayerNorm implementations.
- **Pre-LN**: $x_{l+1} = x_l + F(LN(x_l))$. Norm grows at $\mathcal{O}(\sqrt{L})$, protecting early signal but causing saturation.
- **Post-LN**: $x_{l+1} = LN(x_l + F(x_l))$. Norm remains constant $\mathcal{O}(\sqrt{d})$, but backpropagation becomes unstable.

### 5. Induction Heads (In-Context Learning)
Visualizes the $W_K^{(2)} W_{OV}^{(1)}$ composition circuit responsible for copying previous patterns.
- Simulates how a previous-token head in Layer 1 is composed with an induction head in Layer 2 to produce true in-context learning, and how parameter uncoupling breaks this capacity.

## Tech Stack & Architecture
- **Frontend Engine**: React, Vite, TypeScript
- **Visualization**: High-performance Plotly.js renders
- **Core Math Engine**: Pure, vectorized JS & Python implementations of transformer algebra (no heavy ML frameworks required).

## Deployment (Vercel)
This project is engineered strictly for serverless web deployment. **It is NOT a Streamlit app.**
To deploy on Vercel:
1. Import this repository into Vercel.
2. Vercel will automatically detect the **Vite** preset.
3. Click **Deploy**. The build command is automatically set to `npm run build` with the output directory `dist`.

## Running Locally

1. Install dependencies:
```bash
npm install
```

2. Run the application:
```bash
npm run dev
```
