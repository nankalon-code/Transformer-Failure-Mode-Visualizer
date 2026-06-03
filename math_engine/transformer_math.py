import numpy as np

def normalize_rows(A, target_norm=1.0):
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (A / norms) * target_norm

def softmax_rows(A):
    max_A = np.max(A, axis=1, keepdims=True)
    exps = np.exp(A - max_A)
    return exps / np.sum(exps, axis=1, keepdims=True)

def entropy_rows(A):
    # A is assumed to be a probability distribution
    return -np.sum(A * np.log(A + 1e-12), axis=1)

# --- PANEL 1: Entropy Collapse ---
def attention_entropy_collapse(seq_len=50, d_k=64, q_norm=1.0, k_norm=1.0, tau=1.0):
    base_signal = np.eye(seq_len)[:, :d_k] * 2.0
    Q = normalize_rows(np.random.randn(seq_len, d_k) + base_signal, q_norm)
    K = normalize_rows(np.random.randn(seq_len, d_k) + base_signal, k_norm)
    logits = Q @ K.T
    scaled_logits = logits / (np.sqrt(d_k) * tau)
    A = softmax_rows(scaled_logits)
    entropy = entropy_rows(A)
    grad_mag = A * (1 - A)
    return A, entropy, grad_mag

# --- PANEL 2: Positional Breakdown ---
def sinusoidal_pe_sim(seq_len=200, d_model=64):
    pe = np.zeros((seq_len, d_model))
    position = np.arange(0, seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe @ pe.T

def rope_sim(seq_len=200, d_model=64):
    sim = np.zeros((seq_len, seq_len))
    for m in range(seq_len):
        for n in range(seq_len):
            theta = 10000.0 ** (-2 * np.arange(d_model//2) / d_model)
            sim[m, n] = np.sum(np.cos((m - n) * theta))
    return sim

def alibi_sim(seq_len=200, m_slope=0.5):
    m = np.arange(seq_len)[:, np.newaxis]
    n = np.arange(seq_len)
    bias = -m_slope * np.abs(m - n)
    base_qk = np.random.randn(seq_len, seq_len) * 0.1 + np.eye(seq_len) * 5.0
    return base_qk + bias

# --- PANEL 3: Head Redundancy ---
def js_divergence(p, q):
    m = 0.5 * (p + q)
    kld1 = np.sum(np.where(p > 0, p * np.log(p / m), 0))
    kld2 = np.sum(np.where(q > 0, q * np.log(q / m), 0))
    return 0.5 * kld1 + 0.5 * kld2

def head_redundancy_sim(seq_len=50, d_model=64, d_k=32, num_heads=8, weight_corr=0.8):
    X = np.random.randn(seq_len, d_model)
    WQ_base = np.random.randn(d_model, d_k)
    WK_base = np.random.randn(d_model, d_k)
    
    heads = []
    for _ in range(num_heads):
        WQ_noise = np.random.randn(d_model, d_k)
        WK_noise = np.random.randn(d_model, d_k)
        
        WQ_h = weight_corr * WQ_base + np.sqrt(1 - weight_corr**2) * WQ_noise
        WK_h = weight_corr * WK_base + np.sqrt(1 - weight_corr**2) * WK_noise
        
        Q = X @ WQ_h
        K = X @ WK_h
        logits = (Q @ K.T) / np.sqrt(d_k)
        heads.append(softmax_rows(logits))
        
    matrix = np.zeros((num_heads, num_heads))
    for i in range(num_heads):
        for j in range(num_heads):
            if i == j: continue
            dist = sum(js_divergence(heads[i][p], heads[j][p])**2 for p in range(seq_len))
            matrix[i, j] = dist / seq_len
    return matrix

# --- PANEL 4: Residual Saturation ---
def residual_saturation_sim(num_layers=24, dim=128, ln_type='Pre-LN', f_scale=1.0):
    x = np.random.randn(dim) # Initial embedding
    norms_X = []
    norms_F = []
    
    for l in range(num_layers):
        norms_X.append(np.linalg.norm(x))
        
        if ln_type == 'Pre-LN':
            # Pre-LN: x_{l+1} = x_l + F(LN(x_l))
            # 1. LN(x)
            ln_x = (x - np.mean(x)) / (np.std(x) + 1e-5)
            # 2. F(LN(x))
            f = np.random.randn(dim) * f_scale 
            x = x + f
        elif ln_type == 'Post-LN':
            # Post-LN: x_{l+1} = LN(x_l + F(x_l))
            # 1. F(x) scales with x's magnitude roughly
            f = np.random.randn(dim) * f_scale * (np.linalg.norm(x) / np.sqrt(dim))
            unnorm_x = x + f
            # 2. LN(x + F(x))
            x = (unnorm_x - np.mean(unnorm_x)) / (np.std(unnorm_x) + 1e-5)
        else: # No-LN
            # x_{l+1} = x_l + F(x_l)
            f = np.random.randn(dim) * f_scale * (np.linalg.norm(x) / np.sqrt(dim))
            x = x + f
            
        norms_F.append(np.linalg.norm(f))
        
    return norms_X, norms_F

# --- PANEL 5: Induction Heads ---
def induction_heads_sim(coupling=0.9, n=5, d_model=16):
    # Sequence: [Token A, Token B, Token C, Token A, _]
    seq = np.array([0, 1, 2, 0, 3]) 
    E_vocab = np.random.randn(4, d_model)
    E = E_vocab[seq]
    
    # Layer 1: Extract previous token
    A1 = np.zeros((n, n))
    for i in range(1, n):
        A1[i, i-1] = 1
    O1 = A1 @ E
    
    # Layer 2: Induction Head (matches current query to previous key)
    W_Q2 = np.random.randn(d_model, d_model)
    W_K2_base = W_Q2 # Perfect induction match
    W_K2_noise = np.random.randn(d_model, d_model)
    
    W_K2 = coupling * W_K2_base + np.sqrt(1 - coupling**2) * W_K2_noise
    
    Q = E @ W_Q2
    K = O1 @ W_K2
    
    logits = Q @ K.T
    
    # Apply Causal Mask
    mask = np.triu(np.ones((n, n)), k=1)
    logits[mask == 1] = -1e9
    logits = logits * 2.0 # Scale for sharpness
    
    A2 = softmax_rows(logits)
    return A2

if __name__ == "__main__":
    print("Transformer Math Engine functions loaded successfully.")
    print("Run individual functions to retrieve the pure mathematical simulation data.")
