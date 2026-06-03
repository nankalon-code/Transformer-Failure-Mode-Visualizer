import streamlit as st
import numpy as np
from scipy.spatial.distance import jensenshannon
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Transformer Failure Mode Visualizer", layout="wide")

st.title("Transformer Failure Mode Visualizer")
st.markdown("""
This interactive visualizer explores four primary failure regimes of the transformer architecture:
1. **Attention Entropy Collapse**
2. **Positional Encoding Breakdown**
3. **Attention Head Redundancy**
4. **Residual Stream Saturation**
""")

tabs = st.tabs([
    "1. Entropy Collapse",
    "2. Positional Breakdown",
    "3. Head Redundancy",
    "4. Residual Saturation"
])

# Panel 1: Attention Entropy Collapse
with tabs[0]:
    st.header("Attention Entropy Collapse")
    st.markdown(r"**Equation:** $a_{ij} = \text{softmax}(q_i \cdot k_j / \sqrt{d_k})$")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        q_norm = st.slider("Query Norm ||q||", min_value=0.1, max_value=50.0, value=1.0, step=0.1)
        k_norm = st.slider("Key Norm ||k||", min_value=0.1, max_value=50.0, value=1.0, step=0.1)
        d_k = st.slider("Dimension (d_k)", min_value=1, max_value=512, value=64, step=1)
        seq_len_1 = st.slider("Sequence Length", min_value=10, max_value=200, value=50, step=10)
        
    with col2:
        # Simulate query and key matrices
        np.random.seed(42)
        Q = np.random.randn(seq_len_1, d_k)
        Q = Q / np.linalg.norm(Q, axis=1, keepdims=True) * q_norm
        
        K = np.random.randn(seq_len_1, d_k)
        K = K / np.linalg.norm(K, axis=1, keepdims=True) * k_norm
        
        logits = Q @ K.T
        scaled_logits = logits / np.sqrt(d_k)
        
        # Softmax
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        A = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Entropy
        entropy = -np.sum(A * np.log(A + 1e-12), axis=1)
        
        st.subheader("Attention Matrix (Softmax Surface)")
        fig1 = px.imshow(A, color_continuous_scale='Viridis', title="Attention Weights")
        st.plotly_chart(fig1, use_container_width=True)
        
        st.subheader("Attention Entropy per Position")
        fig2 = px.line(y=entropy, labels={'x': 'Position', 'y': 'Entropy'}, title="Entropy")
        fig2.add_hline(y=np.log(seq_len_1), line_dash="dash", annotation_text="Max Entropy (Uniform)")
        st.plotly_chart(fig2, use_container_width=True)

# Panel 2: Positional Encoding Breakdown
with tabs[1]:
    st.header("Positional Encoding Breakdown")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        seq_len_2 = st.slider("Sequence Length (Testing)", min_value=10, max_value=2000, value=200, step=10)
        train_len = st.slider("Training Sequence Length Boundary", min_value=10, max_value=1000, value=100, step=10)
        enc_type = st.radio("Encoding Type", ["Sinusoidal", "RoPE", "ALiBi"])
        d_model = st.slider("Model Dimension (d_model)", min_value=16, max_value=256, value=64, step=16)
        
    with col2:
        positions = np.arange(seq_len_2)
        sim_matrix = np.zeros((seq_len_2, seq_len_2))
        
        if enc_type == "Sinusoidal":
            st.markdown(r"**Equation:** $PE(p, 2i) = \sin(p / 10000^{2i/d_{\text{model}}})$")
            pe = np.zeros((seq_len_2, d_model))
            div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
            pe[:, 0::2] = np.sin(positions[:, None] * div_term)
            if d_model % 2 == 1:
                pe[:, 1::2] = np.cos(positions[:, None] * div_term[:-1])
            else:
                pe[:, 1::2] = np.cos(positions[:, None] * div_term)
            sim_matrix = pe @ pe.T
            
        elif enc_type == "RoPE":
            st.markdown(r"**Equation:** $q_m^T k_n = (R_m q)^T (R_n k)$ (Focusing on relative decay)")
            theta = 10000.0 ** (-2 * np.arange(0, d_model//2) / d_model)
            m_minus_n = positions[:, None] - positions[None, :]
            sim_matrix = np.sum(np.cos(m_minus_n[:, :, None] * theta), axis=2)
            
        elif enc_type == "ALiBi":
            st.markdown(r"**Equation:** $\text{logit}_{ij} = q_i \cdot k_j - m \cdot |i - j|$")
            m_slope = 0.5
            dist = np.abs(positions[:, None] - positions[None, :])
            sim_matrix = -m_slope * dist
            
        fig3 = px.imshow(sim_matrix, color_continuous_scale='RdBu', title=f"{enc_type} Positional Similarity Matrix")
        fig3.add_vline(x=train_len, line_dash="dash", line_color="red", annotation_text="Train Boundary")
        fig3.add_hline(y=train_len, line_dash="dash", line_color="red")
        st.plotly_chart(fig3, use_container_width=True)

# Panel 3: Attention Head Redundancy
with tabs[2]:
    st.header("Attention Head Redundancy")
    st.markdown(r"**Equation:** $\rho(l, l') = \frac{1}{n} \sum_{i=1}^n \text{JS}(a_i^{(l)} \| a_i^{(l')})$")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        num_heads = st.slider("Number of Heads", min_value=2, max_value=32, value=8, step=1)
        seq_len_3 = 50
        redundancy_factor = st.slider("Redundancy Factor", min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                                     help="1.0 means all heads are identical. 0.0 means completely independent.")
                                     
    with col2:
        np.random.seed(42)
        base_logits = np.random.randn(seq_len_3, seq_len_3)
        heads = []
        for h in range(num_heads):
            noise = np.random.randn(seq_len_3, seq_len_3)
            h_logits = redundancy_factor * base_logits + (1 - redundancy_factor) * noise
            h_exp = np.exp(h_logits - np.max(h_logits, axis=1, keepdims=True))
            h_attn = h_exp / np.sum(h_exp, axis=1, keepdims=True)
            heads.append(h_attn)
            
        js_matrix = np.zeros((num_heads, num_heads))
        for i in range(num_heads):
            for j in range(num_heads):
                if i == j:
                    continue
                js_sum = 0
                for pos in range(seq_len_3):
                    js_sum += jensenshannon(heads[i][pos], heads[j][pos]) ** 2
                js_matrix[i, j] = js_sum / seq_len_3
                
        fig4 = px.imshow(js_matrix, color_continuous_scale='Plasma', title="Head JS-Divergence (Low = Redundant)")
        st.plotly_chart(fig4, use_container_width=True)

# Panel 4: Residual Stream Saturation
with tabs[3]:
    st.header("Residual Stream Saturation")
    st.markdown(r"**Equation:** $x^{(l+1)} = x^{(l)} + F^{(l)}(x^{(l)})$")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        num_layers = st.slider("Number of Layers (Depth)", min_value=1, max_value=100, value=24, step=1)
        ln_type = st.radio("Layer Norm Configuration", ["Pre-LN", "Post-LN", "No-LN"])
        f_scale = st.slider("Layer Contribution Magnitude ||F(x)||", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        
    with col2:
        dim = 128
        np.random.seed(42)
        x = np.random.randn(dim)
        
        norms_x = []
        norms_f = []
        
        for l in range(num_layers):
            norms_x.append(np.linalg.norm(x))
            
            if ln_type == "Pre-LN":
                f = np.random.randn(dim)
                f = f / np.linalg.norm(f) * f_scale
                x = x + f
            elif ln_type == "Post-LN":
                f = np.random.randn(dim)
                f = f / np.linalg.norm(f) * f_scale * (np.linalg.norm(x) if l > 0 else 1.0)
                x = x + f
                x = (x - np.mean(x)) / (np.std(x) + 1e-5) 
            else: 
                f = np.random.randn(dim)
                f = f / np.linalg.norm(f) * f_scale
                x = x + f
                
            norms_f.append(np.linalg.norm(f))
            
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(y=norms_x, mode='lines+markers', name='Residual Stream Norm ||x||'))
        fig5.add_trace(go.Scatter(y=norms_f, mode='lines+markers', name='Layer Contribution Norm ||F(x)||'))
        fig5.update_layout(title="Residual Norm Growth Across Layers", xaxis_title="Layer", yaxis_title="Norm")
        st.plotly_chart(fig5, use_container_width=True)
