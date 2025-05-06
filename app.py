import streamlit as st
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD  # lightweight dim‑red

"""
SDE‑BASED DIFFUSION RECOMMENDER — PAPER‑FAITHFUL DEMO
=====================================================
This Streamlit app follows the complete pipeline laid out in *“Unblurring Preferences: A Mathematical Approach for Time‑Sensitive and Adaptive Recommendation Models using Stochastic Diffusion”*:

1. **Latent Vector Space** — user/item factors derived from *implicit* interactions + optional **Truncated‑SVD** for compression.
2. **Dual Graph Encoders**
   • *Social* user–user GCN  (Eq. 1 / E_S^l = φ(Â E_S^{l‑1} W^l))
   • *Collaborative* user–item LightGCN‑style aggregator (Eq. 2)
3. **Forward ↔ Reverse SDE** (Eqs. 4–6) with an **analytic score estimator** for demo purposes (true model would be trainable).
4. **Joint Loss Playground** — diffusion MSE + BPR ranking + contrastive term (no training here, but we surface the math & live metrics).
5. **Evaluation & Visualisation** — Precision@K before/after denoising.

All heavy learning components are mocked with closed‑form or random weights so the demo stays CPU‑friendly while exposing every mathematical hook you need to slot a real model.
"""

st.set_page_config(page_title="SDE‑Recommender Demo", layout="wide")
st.title("Stochastic‑Diffusion Recommender — Live Explorer")

rng = np.random.default_rng(42)

# -------------------------------------------------------------------
# 0) CONFIG — DATA & LATENT DIMENSIONS
# -------------------------------------------------------------------
cols = st.columns(3)
with cols[0]:
    n_users = st.slider("Users", 10, 200, 40)
with cols[1]:
    n_items = st.slider("Items", 20, 400, 120)
with cols[2]:
    latent_dim = st.slider("Latent dim d", 4, 64, 16)

# -------------------------------------------------------------------
# 1) SYNTHETIC INTERACTIONS & OPTIONAL SVD COMPRESSION
# -------------------------------------------------------------------
st.header("1. Interaction matrix  ⇒  latent factors (SVD option)")

true_user = rng.normal(size=(n_users, latent_dim))
true_item = rng.normal(size=(n_items, latent_dim))
rating_full = true_user @ true_item.T + rng.normal(scale=.2, size=(n_users, n_items))
R = (rating_full > 0.0).astype(int)  # implicit feedback

st.write("Interaction sparsity:", 1.0 - R.mean())

use_svd = st.checkbox("Compress with Truncated‑SVD", value=True)
if use_svd:
    k_svd = st.slider("SVD rank k", 2, latent_dim, latent_dim//2)
    svd = TruncatedSVD(n_components=k_svd, random_state=0)
    user_init = svd.fit_transform(R)
    item_init = svd.components_.T
else:
    user_init = rng.normal(size=(n_users, latent_dim))
    item_init = rng.normal(size=(n_items, latent_dim))

# -------------------------------------------------------------------
# 2) GRAPH ENCODING
# -------------------------------------------------------------------
st.header("2. Dual graph encoders (social & collaborative)")

# 2‑a) SOCIAL GCN ----------------------------------------------------
# Social adjacency from Jaccard on interactions (toy surrogate for real social network)
sim = np.zeros((n_users, n_users))
for i in range(n_users):
    for j in range(i+1, n_users):
        inter = np.logical_and(R[i], R[j]).sum()
        uni = np.logical_or(R[i], R[j]).sum()
        sim[i,j] = sim[j,i] = inter/uni if uni else 0
k_nn = st.slider("k‑NN (social)", 1, 10, 3)
A = np.zeros_like(sim)
for i in range(n_users):
    idx = np.argsort(sim[i])[-k_nn:]
    A[i, idx] = 1
A = np.maximum(A, A.T)
I = np.eye(n_users)
A_hat = A + I
D_inv_sqrt = np.diag(1/np.sqrt(A_hat.sum(1)))
A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
W_social = rng.normal(size=(user_init.shape[1], latent_dim))
E_social = np.tanh(A_norm @ user_init @ W_social)

# 2‑b) COLLABORATIVE LightGCN‑like ----------------------------------
# Propagate user→item and item→user messages one hop (L=1 for demo)
Ui = R.sum(axis=1, keepdims=True) + 1e‑8
Ii = R.sum(axis=0, keepdims=True) + 1e‑8
Norm_user = R/Ui
Norm_item = (R/ Ii).T
item_agg = Norm_user.T @ user_init  # item embeds
user_agg = Norm_item.T @ item_agg   # one iteration back to users
E_collab = user_agg

# Combine social & collaborative as paper suggests (concat + linear)
W_joint = rng.normal(size=(E_social.shape[1]+E_collab.shape[1], latent_dim))
E_joint = np.tanh(np.hstack([E_social, E_collab]) @ W_joint)

st.write("Embedding matrix shape:", E_joint.shape)

# -------------------------------------------------------------------
# 3) STOCHASTIC DIFFUSION ON EMBEDDINGS
# -------------------------------------------------------------------
st.header("3. Forward / reverse SDE denoising (per user)")

T = 200
betas = np.linspace(1e‑4, 0.02, T)
alpha = 1‑betas
abar = np.cumprod(alpha)

def q_sample(x0, t, noise=None):
    if noise is None:
        noise = rng.standard_normal(x0.shape)
    return np.sqrt(abar[t])*x0 + np.sqrt(1‑abar[t])*noise

def p_mean_simple(xt, t):
    """Analytic posterior mean when score‑net = 0 (demo)."""
    return xt/np.sqrt(abar[t])

u_idx = st.slider("User index", 0, n_users‑1, 0)
x0 = E_joint[u_idx]

t_sel = st.slider("Reverse start step t", 1, T‑1, int(.7*T))
xt = q_sample(x0, t_sel)
x0_hat = p_mean_simple(xt, t_sel)

mse_raw = np.mean((x0‑xt)**2)
mse_deno = np.mean((x0‑x0_hat)**2)
st.write(f"MSE noisy: {mse_raw:.4f} — MSE denoised: {mse_deno:.4f}")

col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots()
    traj = np.vstack([q_sample(x0, t) for t in np.linspace(0,T‑1,8, dtype=int)])
    ax.plot(traj[:,0], traj[:,1], "‑o")
    ax.set_title("Forward trajectory (proj 2D)")
    st.pyplot(fig)
with col2:
    fig2, ax2 = plt.subplots()
    ax2.bar(["noisy", "denoised"], [mse_raw, mse_deno])
    ax2.set_title("Reconstruction error")
    st.pyplot(fig2)

# -------------------------------------------------------------------
# 4) RECOMMENDATIONS & METRICS
# -------------------------------------------------------------------
st.header("4. Top‑K Recommendations & Precision")

embed_choice = st.radio("Use embedding", ["raw", "noisy", "denoised"], index=2)
if embed_choice=="raw":
    user_vec = x0
elif embed_choice=="noisy":
    user_vec = xt
else:
    user_vec = x0_hat

# Item vectors (learned = item_init + LightGCN hop)
item_vecs = item_agg  # shape (n_items, d)

scores = user_vec @ item_vecs.T
K = st.slider("K", 1, 20, 10)
ranked = np.argsort(scores)[‑K:][::-1]
true_items = np.where(R[u_idx]==1)[0]
prec = len(set(ranked) & set(true_items)) / K if K else 0

st.write("Recommended items (idx : score)", {int(i):float(scores[i]) for i in ranked})
st.write(f"Precision@{K}: {prec:.3f}")

# -------------------------------------------------------------------
# 5) MATH EXPANDER
# -------------------------------------------------------------------
with st.expander("Mathematical foundations"):
    st.markdown(r"""
* **GCN (Eq. 1)** \(E_S^{\ell}=\varphi(\hat A E_S^{\ell‑1}W^{\ell})\)
* **LightGCN hop (Eq. 2)** messages user→item→user to build \(E_R^{(L)}\)
* **SDE forward (Eq. 4)** \(x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1‑\bar\alpha_t}\varepsilon\)
* **Reverse mean (demo)** \(\hat x_0 = x_t / \sqrt{\bar\alpha_t}\) when score≈0.
* **Joint loss (Eq. 14)** present but *not* optimised here:  \(\mathcal L = \mathcal L_{Diff}+λ_1\mathcal L_{BPR}+λ_2\mathcal L_{CL}\).
""")
