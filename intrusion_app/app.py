# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="ML IDS Dashboard", layout="wide", initial_sidebar_state="expanded")

# -----------------------
# Small CSS polish (white bg during spinner & dark optional)
# -----------------------
st.markdown("""
<style>
/* keep spinner background white */
[data-testid="stSpinner"] div { background-color: white !important; }

/* optional: compact layout for tables */
.streamlit-expanderHeader { font-weight:600; }

/* dark theme toggle (class applied via checkbox) */
body.dark-theme { background-color: #0e1117; color: #e6edf3; }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Paths & config
# -----------------------
DATA_DIR = os.getenv("DATA_DIR", "data")  # allows override via env var or docker -v mount
DATA_PATH = os.path.join(DATA_DIR, "X_ensemble_latent_labeled.parquet")
ENSEMBLE_NPY = os.path.join(DATA_DIR, "X_ensemble_latent.npy")
ANOMALIES_NPY = os.path.join(DATA_DIR, "anomalies.npy")
LABELS_NPY = os.path.join(DATA_DIR, "labels.npy")

# -----------------------
# Cached data loader
# -----------------------
@st.cache_data(show_spinner=False)
def load_data(parquet_path: str):
    df = pd.read_parquet(parquet_path)
    latent_cols = [c for c in df.columns if c.startswith("latent")]
    return df, latent_cols

# fail early with friendly message
if not os.path.exists(DATA_PATH):
    st.error(f"Dataset non trovato: `{DATA_PATH}`. Monta la cartella data o controlla percorso.")
    st.stop()

df, latent_cols = load_data(DATA_PATH)

# UI header
st.title("🔎 ML Intrusion Detection — Interactive Report")
st.markdown(f"Dataset: **{df.shape[0]} samples**, **{len(latent_cols)} latent features**")

# -----------------------
# Sidebar: controls & UX
# -----------------------
st.sidebar.header("Controls")
dr_method = st.sidebar.selectbox("Dimensionality Reduction", ["UMAP 2D", "UMAP 3D", "t-SNE"], index=0)
sample_size = st.sidebar.slider("Projection sample size", min_value=1000, max_value=10000, value=5000, step=500)
label_choice = st.sidebar.selectbox("Filter attack label", ["ALL"] + sorted(df["label_technique"].unique()))
show_anom = st.sidebar.checkbox("Show anomalies only", False)
model_mode = st.sidebar.selectbox("Model Mode",
                                  ["Supervised (RandomForest)", "Unsupervised (LOF + OCSVM)", "Semi-supervised (IsolationForest precomputed)"])
dark_theme = st.sidebar.checkbox("Dark theme (UI only)", False)

if dark_theme:
    st.markdown("<script>document.body.classList.add('dark-theme')</script>", unsafe_allow_html=True)

# -----------------------
# Safe projection functions (cached)
# -----------------------
@st.cache_data(ttl=60*60, show_spinner=False)
def compute_umap(data_np: np.ndarray, n_components=2, n_neighbors=15, min_dist=0.1, random_state=42):
    import umap
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                        min_dist=min_dist, random_state=random_state)
    return reducer.fit_transform(data_np)

@st.cache_data(ttl=60*60, show_spinner=False)
def compute_tsne(data_np: np.ndarray, perplexity: int, n_iter: int = 800):
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate='auto',
                init='pca', random_state=42, n_iter=n_iter, verbose=0)
    return tsne.fit_transform(data_np)

# Helper: make perplexity safe for tsne
def safe_perplexity(n_samples: int):
    p = max(5, min(50, (n_samples // 3) - 1))
    return int(max(5, min(50, p)))

# -----------------------
# Prepare sample
# -----------------------
sample_df = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)
X_sample = sample_df[latent_cols].to_numpy()

# Compute projection with spinner UX (non-blocking cached)
proj_placeholder = st.empty()
proj_placeholder.info(f"Computing {dr_method} on {len(sample_df)} samples — this may take a moment...")

t0 = time.time()
if dr_method == "UMAP 2D":
    X_proj = compute_umap(X_sample, n_components=2)
elif dr_method == "UMAP 3D":
    X_proj = compute_umap(X_sample, n_components=3)
else:  # t-SNE
    p = safe_perplexity(len(X_sample))
    X_proj = compute_tsne(X_sample, perplexity=p, n_iter=600)
t_proj = time.time() - t0
proj_placeholder.empty()  # remove info

# attach dims
if dr_method == "UMAP 3D":
    sample_df["Dim1"], sample_df["Dim2"], sample_df["Dim3"] = X_proj[:,0], X_proj[:,1], X_proj[:,2]
else:
    sample_df["Dim1"], sample_df["Dim2"] = X_proj[:,0], X_proj[:,1]

# -----------------------
# Top visualization area
# -----------------------
with st.container():
    c1, c2 = st.columns([3,1])
    with c1:
        st.subheader(f"Latent Space — {dr_method} (projected in {t_proj:.1f}s)")
        plot_df = sample_df.copy()
        if show_anom:
            plot_df = plot_df[plot_df.anomaly_flag == 1]
        if label_choice != "ALL":
            plot_df = plot_df[plot_df.label_technique == label_choice]

        if dr_method == "UMAP 3D":
            fig = px.scatter_3d(plot_df, x="Dim1", y="Dim2", z="Dim3",
                                color="label_technique", symbol="anomaly_flag",
                                hover_data=["label_technique","anomaly_flag"],
                                width=900, height=700)
        else:
            fig = px.scatter(plot_df, x="Dim1", y="Dim2",
                             color="label_technique", symbol="anomaly_flag",
                             hover_data=["label_technique","anomaly_flag"],
                             width=900, height=700)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("How to read this plot")
        st.markdown("""
- **Each point** = one Zeek session embedding
- **Color** = MITRE technique (label_technique)
- **Symbol** = anomaly_flag (1 = anomaly)
- **UMAP 2D/3D**: preserves global structure and is fast (good for large samples)
- **t-SNE**: emphasizes local clusters (use smaller samples ~3k-5k)
- **If anomalous points cluster separately** → embeddings capture distinct malicious behaviours.
""")

# -----------------------
# Models & Metrics area (below visuals)
# -----------------------
st.markdown("---")
st.subheader("Models & Metrics (below visualization)")

# helper to compute metrics and return dict
def compute_binary_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "specificity": spec,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

# cached trainer for RF
@st.cache_resource
def train_rf_cached(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    return rf

# Always prepare full arrays
X_all = df[latent_cols]
y_present = "anomaly_flag" in df.columns
y_all = df["anomaly_flag"].values if y_present else None

# MODE: Supervised
if model_mode == "Supervised (RandomForest)":
    st.markdown("### Supervised — Random Forest")
    if not y_present:
        st.warning("Ground-truth labels (anomaly_flag) not present; supervised mode unavailable.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_all, df["anomaly_flag"], test_size=0.3, random_state=42)
        with st.spinner("Training RandomForest (cached)..."):
            rf = train_rf_cached(X_train, y_train)

        y_pred = rf.predict(X_test)
        y_prob = rf.predict_proba(X_test)[:,1]

        # metrics table (accuracy, precision, recall, f1, specificity)
        metrics = compute_binary_metrics(y_test.values, y_pred)
        st.table(pd.DataFrame(metrics, index=["value"]).T)

        # confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig_cm)

        # feature importance
        feat_imp = pd.Series(rf.feature_importances_, index=latent_cols).sort_values(ascending=False)
        st.subheader("Top latent features (RF importance)")
        st.bar_chart(feat_imp.head(15))

        # ROC and PR curves
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        fig_roc, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0,1],[0,1],'--'); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()
        st.pyplot(fig_roc)

        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
        fig_pr, ax = plt.subplots()
        ax.plot(recall_vals, precision_vals); ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        st.pyplot(fig_pr)

        # SHAP explainability (summary + single instance)
        st.subheader("SHAP explainability (RandomForest)")
        with st.spinner("Computing SHAP values (sample)..."):
            try:
                explainer = shap.TreeExplainer(rf)
                shap_sample = X_test.sample(min(300, len(X_test)), random_state=42)
                shap_values = explainer.shap_values(shap_sample)
                fig_shap = plt.figure(figsize=(8,5))
                shap.summary_plot(shap_values[1], shap_sample, show=False)
                st.pyplot(fig_shap)
            except Exception as e:
                st.warning(f"SHAP plotting failed: {e}")

        # single instance waterfall/force (matplotlib fallback)
        st.markdown("Explain a single test sample (SHAP)")
        idx_local = st.number_input("Pick index in sampled test set (0..n-1)", min_value=0, max_value=max(0,len(X_test)-1), value=0)
        if st.button("Show SHAP for selected sample"):
            try:
                sample_row = X_test.reset_index(drop=True).iloc[idx_local:idx_local+1]
                shap_vals_single = explainer.shap_values(sample_row)
                fig_force = shap.force_plot(explainer.expected_value[1], shap_vals_single[1][0], sample_row.iloc[0], matplotlib=True)
                st.pyplot(fig_force)
            except Exception as e:
                st.warning(f"SHAP single explain failed: {e}")

# MODE: Unsupervised
elif model_mode == "Unsupervised (LOF + OCSVM)":
    st.markdown("### Unsupervised suite — LOF & One-Class SVM")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    with st.spinner("Fitting LOF & OCSVM (these may take time)..."):
        t0 = time.time()
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True)
        lof.fit(X_scaled)
        lof_pred = lof.predict(X_scaled)
        lof_flag = np.where(lof_pred == -1, 1, 0)
        t_lof = time.time() - t0

        t0 = time.time()
        ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05)
        ocsvm.fit(X_scaled)
        oc_pred = ocsvm.predict(X_scaled)
        oc_flag = np.where(oc_pred == -1, 1, 0)
        t_oc = time.time() - t0

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**LOF** — time {t_lof:.1f}s — anomalies {int(lof_flag.sum())} ({lof_flag.sum()/len(lof_flag)*100:.2f}%)")
        if y_present:
            st.table(pd.DataFrame(compute_binary_metrics(y_all, lof_flag), index=["LOF"]).T)

    with c2:
        st.markdown(f"**One-Class SVM** — time {t_oc:.1f}s — anomalies {int(oc_flag.sum())} ({oc_flag.sum()/len(oc_flag)*100:.2f}%)")
        if y_present:
            st.table(pd.DataFrame(compute_binary_metrics(y_all, oc_flag), index=["OCSVM"]).T)

    # Surrogate Decision Tree to get interpretable feature importances
    st.markdown("Surrogate interpretability (Decision Tree approximating LOF)")
    surrogate = DecisionTreeClassifier(max_depth=3)
    surrogate.fit(X_all, lof_flag)
    importances = pd.Series(surrogate.feature_importances_, index=latent_cols).sort_values(ascending=False)
    st.bar_chart(importances.head(12))

    if y_present:
        cm = confusion_matrix(y_all, lof_flag)
        fig_cm, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig_cm)

# MODE: Semi-supervised (load precomputed IsolationForest results)
else:
    st.markdown("### Semi-supervised — IsolationForest (precomputed)")
    try:
        X_ensemble = np.load(ENSEMBLE_NPY)
        anomalies = np.load(ANOMALIES_NPY)
        flag = np.where(anomalies == -1, 1, 0)
        st.write(f"Loaded ensemble embeddings: {X_ensemble.shape}, anomalies: {len(flag)}")
        if y_present:
            st.table(pd.DataFrame(compute_binary_metrics(y_all, flag), index=["IsolationForest"]).T)
            cm = confusion_matrix(y_all, flag)
            fig_cm, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig_cm)
    except Exception as e:
        st.error(f"Cannot load precomputed artifacts (expected at {ENSEMBLE_NPY} & {ANOMALIES_NPY}): {e}")

# -----------------------
# MITRE heatmap + SOC KPIs + Modal explainer
# -----------------------
st.markdown("---")
st.subheader("MITRE ATT&CK — Technique Heatmap & SOC KPIs")

mitre_counts = df.groupby("label_technique")["anomaly_flag"].sum().reset_index().sort_values("anomaly_flag", ascending=False)
if not mitre_counts.empty:
    fig_m, ax = plt.subplots(figsize=(6, max(3, len(mitre_counts)*0.35)))
    sns.barplot(x="anomaly_flag", y="label_technique", data=mitre_counts, palette="Reds", ax=ax)
    ax.set_xlabel("Detected anomalies"); ax.set_ylabel("Technique")
    st.pyplot(fig_m)

# SOC KPIs (TPR, FPR, MTTR, TTD placeholders)
st.markdown("**SOC KPI Panel (example)**")
k1, k2, k3, k4 = st.columns(4)
last_pred = None
try:
    last_pred = y_pred  # if defined by supervised branch
except:
    try:
        last_pred = lof_flag
    except:
        last_pred = np.zeros_like(y_all) if y_present else np.array([0])

if y_present:
    tn, fp, fn, tp = confusion_matrix(y_all, last_pred).ravel()
    tpr = tp / (tp + fn) if (tp+fn) else 0
    fpr = fp / (fp + tn) if (fp+tn) else 0
else:
    tpr = fpr = 0.0

k1.metric("TPR (Recall)", f"{tpr:.3f}")
k2.metric("FPR", f"{fpr:.3f}")
k3.metric("MTTR (simulated)", "—")
k4.metric("TTD (simulated)", "—")

# Modal-like explainer for selected sample (using expander)
st.markdown("---")
st.subheader("Modal explainer for single event")
sel_idx = st.number_input("Pick index from the current sample (0..n-1)", min_value=0, max_value=max(0,len(sample_df)-1), value=10)
if st.button("Open detail (modal)"):
    with st.expander("Detailed view (modal) — close to dismiss", expanded=True):
        ev = sample_df.iloc[sel_idx]
        st.write("### Selected sample latent features")
        st.dataframe(ev[latent_cols].to_frame().T)
        st.write(f"Label: **{ev['label_technique']}**, anomaly_flag: **{int(ev['anomaly_flag'])}**")
        # nearest neighbours
        nbrs = NearestNeighbors(n_neighbors=6).fit(df[latent_cols].to_numpy())
        dist, ind = nbrs.kneighbors(ev[latent_cols].to_numpy().reshape(1,-1))
        neighbors = pd.DataFrame({"index": ind[0][1:], "distance": dist[0][1:], "label": df.iloc[ind[0][1:]]['label_technique'].values})
        st.table(neighbors)

# Live stream placeholder
st.markdown("---")
st.subheader("Live stream Zeek logs + detection feed (placeholder)")
st.info("To enable live: push Zeek JSON rows to a file/socket; this app can tail and display new rows regularly.")
if st.button("Show last 10 raw events (if file exists)"):
    raw_path = os.path.join(DATA_DIR, "last_raw_events.parquet")
    if os.path.exists(raw_path):
        raw = pd.read_parquet(raw_path)
        st.dataframe(raw.tail(10))
    else:
        st.warning(f"No raw events file found at {raw_path}")

st.markdown("### End of dashboard")
