# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             accuracy_score, confusion_matrix, roc_curve, auc,
                             precision_recall_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import shap
import time

st.set_page_config(page_title="ML IDS Dashboard", layout="wide")

# -----------------------
# Helper: UI polish to avoid grey loading background
# -----------------------
st.markdown("""
<style>
/* keep background white while loading widgets */
[data-testid="stSpinner"] div { background-color: white !important; }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Cached loading routine
# -----------------------
@st.cache_data
def load_data(parquet_path: str):
    df = pd.read_parquet(parquet_path)
    latent_cols = [c for c in df.columns if c.startswith("latent")]
    return df, latent_cols

DATA_PATH = r"C:/Users/maria/Desktop/Zeek_ML/processed_zeekdata22/X_ensemble_latent_labeled.parquet"
df, latent_cols = load_data(DATA_PATH)
st.header("🔎 ML Intrusion Detection — Interactive Report")
st.markdown(f"Dataset: **{df.shape[0]} samples**, **{len(latent_cols)} latent features**")

# -----------------------
# Sidebar controls
# -----------------------
st.sidebar.header("Controls")
dr_method = st.sidebar.selectbox("Dimensionality Reduction", ["UMAP", "t-SNE"], index=0)
sample_size = st.sidebar.slider("Sample size for projection", min_value=1000, max_value=10000, value=5000, step=500)
label_choice = st.sidebar.selectbox("Filter attack label", ["ALL"] + sorted(df["label_technique"].unique()))
show_anom = st.sidebar.checkbox("Show anomalies only", False)
model_mode = st.sidebar.selectbox("Model Mode", ["Unsupervised (LOF, OCSVM)", "Semi-supervised (IsolationForest)","Supervised (RandomForest)"])

# -----------------------
# Safe projection function (adapt perplexity automatically)
# -----------------------
@st.cache_data
def compute_projection(method: str, data_np: np.ndarray):
    n = data_np.shape[0]
    if method == "UMAP":
        import umap
        reducer = umap.UMAP(n_neighbors=min(15, max(2, n//10)), min_dist=0.1, random_state=42)
        return reducer.fit_transform(data_np)
    else:
        from sklearn.manifold import TSNE
        # make perplexity safe: must be < n_samples/3
        safe_perp = max(5, min(50, max(5, (n // 3) - 1)))
        tsne = TSNE(n_components=2, perplexity=safe_perp, learning_rate='auto', random_state=42, n_iter=1000, init='pca', verbose=0)
        return tsne.fit_transform(data_np)

# -----------------------
# Prepare sample for projection (cached)
# -----------------------
@st.cache_data
def get_sample(df_local, n):
    n = min(n, len(df_local))
    return df_local.sample(n, random_state=42).reset_index(drop=True)

sample_df = get_sample(df, sample_size)
X_sample = sample_df[latent_cols].to_numpy()

# show loading message then compute projection (and remove)
proj_status = st.empty()
proj_status.info(f"Computing {dr_method} projection on {len(sample_df)} samples...")
X_proj = compute_projection(dr_method, X_sample)
proj_status.empty()

sample_df["Dim1"], sample_df["Dim2"] = X_proj[:,0], X_proj[:,1]

# -----------------------
# Visualization panel (top)
# -----------------------
with st.container():
    col1, col2 = st.columns([3,1])
    with col1:
        st.subheader(f"Latent Space Visualization — {dr_method}")
        df_plot = sample_df.copy()
        if show_anom:
            df_plot = df_plot[df_plot.anomaly_flag == 1]
        if label_choice != "ALL":
            df_plot = df_plot[df_plot.label_technique == label_choice]

        fig = px.scatter(df_plot, x="Dim1", y="Dim2",
                         color="label_technique", symbol="anomaly_flag",
                         hover_data=["label_technique","anomaly_flag"],
                         title=f"{dr_method} — colored by label_technique")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Interpretation Guide")
        st.markdown("""
- **Ogni punto** = un flusso / sessione Zeek (embedding latente).
- **Colore** = tecnica (label_technique).
- **Forma** = 0 normale / 1 anomalia.
- **UMAP**: mantiene struttura globale, veloce su dataset grandi.
- **t-SNE**: migliora separazione locale ma necessita sample maggiore.
- **Suggerimento**: usa un sample di 3k–5k per t-SNE, UMAP può arrivare più in alto.
""")

# -----------------------
# MODELS & METRICS (always below visualization)
# -----------------------
st.markdown("---")
st.subheader("Models & Metrics")

# Helper metrics function (binary)
def compute_binary_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    # specificity = TN / (TN + FP)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {"accuracy":acc, "precision":prec, "recall":rec, "f1":f1, "specificity":spec, "tn":tn, "fp":fp, "fn":fn, "tp":tp}

# prepare arrays
X_all = df[latent_cols].to_numpy()
y_true_present = "anomaly_flag" in df.columns
y_true_all = df["anomaly_flag"].values if y_true_present else None

# -----------------------
# Unsupervised suite: LOF + One-Class SVM + surrogate explainability
# -----------------------
if model_mode == "Unsupervised (LOF, OCSVM)":
    st.markdown("### Unsupervised models: Local Outlier Factor (LOF) & One-Class SVM")
    st.info("These models produce anomaly scores/predictions. If dataset has labels, we compute semi-supervised metrics.")

    # scale once
    scaler = StandardScaler()
    X_scaled_all = scaler.fit_transform(X_all)

    # LOF
    t0 = time.time()
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True)
    lof.fit(X_scaled_all)
    lof_pred = lof.predict(X_scaled_all)
    lof_flag = np.where(lof_pred == -1, 1, 0)
    t_lof = time.time() - t0

    # One-Class SVM
    t0 = time.time()
    ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05)
    ocsvm.fit(X_scaled_all)
    oc_pred = ocsvm.predict(X_scaled_all)
    oc_flag = np.where(oc_pred == -1, 1, 0)
    t_oc = time.time() - t0

    col_lof, col_oc = st.columns(2)
    with col_lof:
        st.markdown("**LOF results**")
        st.write(f"Time: {t_lof:.1f}s — Anomalies: {int(lof_flag.sum())} ({lof_flag.sum()/len(lof_flag)*100:.2f}%)")
        if y_true_present:
            m = compute_binary_metrics(y_true_all, lof_flag)
            st.table(pd.DataFrame([m], index=["LOF_metrics"]).T)

    with col_oc:
        st.markdown("**One-Class SVM results**")
        st.write(f"Time: {t_oc:.1f}s — Anomalies: {int(oc_flag.sum())} ({oc_flag.sum()/len(oc_flag)*100:.2f}%)")
        if y_true_present:
            m = compute_binary_metrics(y_true_all, oc_flag)
            st.table(pd.DataFrame([m], index=["OCSVM_metrics"]).T)

    # Surrogate explainability: train a small decision tree to approximate LOF decision -> get top features
    st.markdown("**Surrogate explainability (Decision Tree approximating LOF)**")
    surrogate = DecisionTreeClassifier(max_depth=3)
    surrogate.fit(X_all, lof_flag)
    importances = pd.Series(surrogate.feature_importances_, index=latent_cols).sort_values(ascending=False)
    st.bar_chart(importances.head(10))

    # Confusion matrix LOF (if labels)
    if y_true_present:
        cm = confusion_matrix(y_true_all, lof_flag)
        fig_cm, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig_cm)

# -----------------------
# Supervised RandomForest (optional)
# -----------------------
elif model_mode == "Supervised (RandomForest)":
    st.markdown("### Supervised: RandomForest on embeddings")
    if not y_true_present:
        st.warning("No ground-truth labels present in dataset. Supervised training not available.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(df[latent_cols], df["anomaly_flag"], test_size=0.3, random_state=42)
        rf = RandomForestClassifier(n_estimators=200, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        y_prob = rf.predict_proba(X_test)[:,1]

        metrics = compute_binary_metrics(y_test.values, y_pred)
        st.table(pd.DataFrame(metrics, index=["RF"]).T)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig_cm)

        # Feature importance
        feat_imp = pd.Series(rf.feature_importances_, index=latent_cols).sort_values(ascending=False)
        st.subheader("Top latent features (RF importance)")
        st.bar_chart(feat_imp.head(15))

        # ROC & PR
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig_roc, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC={auc(fpr,tpr):.3f}")
        ax.plot([0,1],[0,1],'--'); ax.legend()
        st.pyplot(fig_roc)

        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        fig_pr, ax = plt.subplots()
        ax.plot(recall, precision); ax.set_title("Precision-Recall"); st.pyplot(fig_pr)

        # SHAP explainability (sample)
        st.subheader("SHAP explainability (RF)")
        explainer = shap.TreeExplainer(rf)
        shap_sample = X_test.sample(min(300, len(X_test)), random_state=42)
        shap_values = explainer.shap_values(shap_sample)
        fig_shap = plt.figure(figsize=(8,6))
        shap.summary_plot(shap_values[1], shap_sample, show=False)
        st.pyplot(fig_shap)

# -----------------------
# Semi-supervised placeholder (we used IsolationForest earlier in pipeline)
# -----------------------
else:
    st.markdown("### Semi-supervised: IsolationForest (precomputed earlier in pipeline)")
    st.info("You previously ran an IsolationForest on the ensemble embeddings and saved anomalies.npy. Here we load and evaluate.")
    try:
        X_ensemble = np.load(r"C:/Users/maria/Desktop/Zeek_ML/processed_zeekdata22/X_ensemble_latent.npy")
        anomalies = np.load(r"C:/Users/maria/Desktop/Zeek_ML/processed_zeekdata22/anomalies.npy")
        flag = np.where(anomalies == -1, 1, 0)
        st.write(f"Loaded ensemble embeddings: {X_ensemble.shape}, anomalies: {len(flag)}")
        if y_true_present:
            m = compute_binary_metrics(y_true_all, flag)
            st.table(pd.DataFrame(m, index=["IsolationForest"]).T)
            cm = confusion_matrix(y_true_all, flag)
            fig_cm, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig_cm)
    except Exception as e:
        st.error(f"Cannot load precomputed results: {e}")

# -----------------------
# MITRE ATT&CK Heatmap + Modal explainer + KPI SOC
# -----------------------
st.markdown("---")
st.subheader("MITRE ATT&CK — Technique Heatmap & SOC KPIs")

# MITRE heatmap: number detected per technique
mitre_counts = df.groupby("label_technique")["anomaly_flag"].sum().reset_index()
if len(mitre_counts):
    mitre_counts = mitre_counts.sort_values("anomaly_flag", ascending=False)
    fig_m, ax = plt.subplots(figsize=(6, max(3, len(mitre_counts)*0.3)))
    sns.barplot(x="anomaly_flag", y="label_technique", data=mitre_counts, palette="Reds", ax=ax)
    ax.set_xlabel("Detected anomalies")
    ax.set_ylabel("MITRE Technique")
    st.pyplot(fig_m)

# SOC KPI panel (simple computed KPIs)
st.markdown("**SOC KPI Dashboard (example metrics)**")
# example KPIs computed from confusion matrix of last selected model if available
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
# compute TPR = recall, FPR = FP/(FP+TN), TTD/MTTR placeholders
if y_true_present:
    # choose last computed predictions if exist (prefer rf y_pred, else lof_flag)
    try:
        last_y_pred = y_pred
    except:
        try:
            last_y_pred = lof_flag
        except:
            last_y_pred = np.zeros_like(y_true_all)
    cm = confusion_matrix(y_true_all, last_y_pred)
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp+fn)>0 else 0
    fpr = fp / (fp + tn) if (fp+tn)>0 else 0
else:
    tpr, fpr = 0.0, 0.0

kpi_col1.metric("TPR (Recall)", f"{tpr:.3f}")
kpi_col2.metric("FPR", f"{fpr:.3f}")
kpi_col3.metric("MTTR (sim)", "—")  # placeholder
kpi_col4.metric("TTD (sim)", "—")   # placeholder

# Modal / Explainer for single event
st.markdown("---")
st.subheader("Modal explainer for a single event")
st.markdown("Pick an index from the dataset sample to inspect a single flow. A modal-style detailed view will show embeddings + original labels + nearest neighbors.")

idx = st.number_input("Pick sample index (0..n-1 from sample view)", min_value=0, max_value=len(sample_df)-1, value=10)
if st.button("Show detail"):
    ev = sample_df.iloc[idx]
    st.write("### Detail for selected sample")
    st.write(ev[latent_cols].to_frame().T)
    st.write(f"Label: {ev['label_technique']}, anomaly_flag: {ev['anomaly_flag']}")
    # nearest neighbors in latent space (euclidean)
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=6).fit(df[latent_cols].to_numpy())
    dist, ind = nbrs.kneighbors(ev[latent_cols].to_numpy().reshape(1,-1))
    st.write("Nearest neighbors (index, distance):")
    neighbors = pd.DataFrame({"index": ind[0][1:], "distance": dist[0][1:]})
    st.dataframe(neighbors)

# Live Zeek stream placeholder
st.markdown("---")
st.subheader("Live stream Zeek logs + detection feed (placeholder)")
st.info("To enable live stream: pipe Zeek logs to a local socket / file and use a small tailer that updates this UI. Example: use WebSocket or SSE server producing new JSON rows that this app reads periodically.")
if st.button("Show last 10 saved raw events (file placeholder)"):
    try:
        raw_path = r"C:/Users/maria/Desktop/Zeek_ML/processed_zeekdata22/last_raw_events.parquet"
        raw = pd.read_parquet(raw_path)
        st.dataframe(raw.tail(10))
    except Exception as e:
        st.warning(f"No raw events found at {raw_path}: {e}")

st.markdown("### End of dashboard")
