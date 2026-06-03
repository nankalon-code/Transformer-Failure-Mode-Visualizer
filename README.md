# Transformer Failure Mode Visualizer

An interactive visualizer designed to systematically study the failure modes of the Transformer architecture through raw mathematics. Most educational resources show when transformers work; this visualizer demonstrates when and why they break down.

## Four Primary Failure Regimes
1. **Attention Entropy Collapse**: Drag $d_k$ and watch the distribution collapse from a smooth surface to a spike, seeing exactly the moment the gradient vanishes.
2. **Positional Encoding Breakdown**: Manipulate sequence lengths past training boundaries to observe extrapolation failure across Sinusoidal, RoPE, and ALiBi encodings.
3. **Attention Head Redundancy**: Compute the JS-divergence of attention distributions to build intuition on head similarity.
4. **Residual Stream Saturation**: Simulate residual norm growth across layers to find the exact crossover point where Layer Normalization discards early signal.

## Stack
- Python
- Streamlit
- NumPy & SciPy
- Plotly

## Running Locally

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run app.py
```
