# app.py — ML Intrusion Detection Interactive Report (full)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
import tensorflow as tf
import shap

# Page config
st.set_page_config(page_title="ML IDS Dashboard", layout="wide")
st.title("🔎 ML Intrusion Detection Interactive Report")

# ---------------------------
# CONFIG PATHS (adatta se vuoi)
DATA_DIR = r"C:/Users/maria/Desktop/Zeek_ML/processed_zeekdata22"
PARQUET_PATH = f"{DATA_DIR}/X_ensemble_latent_labeled.parquet"
BALANCED_PATH = f"{DATA_DIR}/X_ensemble_latent_balanced.parquet"
NP_ENSEMBLE = f"{DATA_DIR}/X_ensemble_latent.npy"
NP_ANOM = f"{DATA_DIR}/anomalies.npy"
NP_LABELS = f"{DATA_DIR}/labels.npy"
# Zeek logs (per live feed) — imposta il path ai tuoi zeek logs
ZEEK_LOG_PATH = r"C:/Users/maria/Desktop/Zeek_ML/zeek_logs/conn.log"  # esempio

# ---------------------------
# Helper: install warnings if shap missing (try/except)
try:
    import shap
except Exception:
    st.warning("shap non installato: explainability SHAP non disponibile. Esegui `pip install shap` se vuoi usarlo.")

# ---------------------------
# LOAD DATA (cached)
@st.cache_data(show_spinner=False)
def load_data(parquet_path):
    df = pd.read_parquet(parquet_path)
    latent_cols = [c for c in df.columns if c.startswith("latent")]
    return df, latent_cols

df, latent_cols = load_data(PARQUET_PATH)
st.sidebar.markdown(f"**Dataset:** {df.shape[0]} samples · {len(latent_cols)} latent features")
st.markdown(f"Dataset caricato: **{df.shape[0]}** samples · **{len(latent_cols)}** latent features")

# ---------------------------
# SAMPLE for DR (cached)
@st.cache_data
def sample_df_fn(df, n=5000):
    if df.shape[0] <= n:
        return df.copy()
    return df.sample(n, random_state=42)

sample_df = sample_df_fn(df)

# ---------------------------
# Sidebar controls
st.sidebar.header("Controls")
dr_method = st.sidebar.selectbox("Dimensionality Reduction", ["UMAP", "t-SNE"])
show_anom = st.sidebar.checkbox("Show anomalies only", value=False)
label_choice = st.sidebar.selectbox("Attack label filter", ["ALL"] + sorted(df["label_technique"].unique()))
model_choice = st.sidebar.selectbox("Model", ["Local Outlier Factor (unsupervised)",
                                              "One-Class SVM (unsupervised)",
                                              "Autoencoder (unsupervised)",
                                              "Random Forest (supervised — optional)"])
refresh_logs = st.sidebar.button("Refresh live logs")

# ---------------------------
# Compute projection (cached per method + sample)
@st.cache_data(show_spinner=False)
def compute_projection(method, X):
    if method == "UMAP":
        import umap
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        proj = reducer.fit_transform(X)
    else:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, perplexity=50, learning_rate='auto', random_state=42, n_iter=1000)
        proj = tsne.fit_transform(X)
    return proj

# show spinner overlay info (and remove background when done)
loading = st.empty()
loading.info(f"Computing {dr_method} projection (sample={len(sample_df)})...")

# we pass only latent cols as 2D matrix
X_latent_sample = sample_df[latent_cols].to_numpy()
X_proj = compute_projection(dr_method, X_latent_sample)

loading.empty()

sample_df["Dim1"], sample_df["Dim2"] = X_proj[:, 0], X_proj[:, 1]
df_plot = sample_df.copy()
if show_anom:
    df_plot = df_plot[df_plot["anomaly_flag"] == 1]
if label_choice != "ALL":
    df_plot = df_plot[df_plot["label_technique"] == label_choice]

# ---------------------------
# Left / Right layout for visuals and model
left_col, right_col = st.columns((2,1))

with left_col:
    st.subheader(f"Latent Space ({dr_method})")
    fig = px.scatter(df_plot, x="Dim1", y="Dim2", color="label_technique",
                     symbol="anomaly_flag", hover_data=["label_technique","anomaly_flag"],
                     width=900, height=600)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ️ Interpretation Guide (UMAP vs t-SNE)"):
        st.markdown("""
**Che cosa mostra il grafico**  
- Ogni punto è un flow/session (rappresentato dagli embeddings latenti).  
- Colore = tecnica (label_technique). Forma = anomaly_flag.  
- Se attacchi/outlier clusterizzano, significa che gli embeddings separano i comportamenti.

**UMAP**
- Mantiene struttura globale, veloce su dataset grandi. Buono per overview.

**t-SNE**
- Eccelle a evidenziare cluster locali; più lento e sensibile a parametri (perplexity).
""")

with right_col:
    st.subheader("Model & Metrics")

    # Prepare dataset for models
    X = df[latent_cols].to_numpy()
    y = df["anomaly_flag"].to_numpy()  # 1 = anomaly

    # Utility: compute metrics and specificity
    def metrics_table(y_true, y_pred):
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        tp = int(((y_true==1) & (y_pred==1)).sum())
        tn = int(((y_true==0) & (y_pred==0)).sum())
        fp = int(((y_true==0) & (y_pred==1)).sum())
        fn = int(((y_true==1) & (y_pred==0)).sum())
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        table = pd.DataFrame({
            "value": [acc, precision, recall, f1, specificity],
        }, index=["accuracy","precision","recall","f1-score","specificity"])
        cm = np.array([[tn, fp],[fn, tp]])
        return table, cm

    # Train / evaluate depending on model choice
    if model_choice == "Local Outlier Factor (unsupervised)":
        st.markdown("**Local Outlier Factor (LOF)** — unsupervised anomaly detection")
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        # LOF supports novelty=True but requires fit on training data; we'll use fit_predict on whole set
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=False)
        y_pred_lof = lof.fit_predict(Xs)  # returns -1 for outliers
        y_pred_flag = np.where(y_pred_lof == -1, 1, 0)
        table, cm = metrics_table(y, y_pred_flag)
        st.table(table.style.format("{:.3f}"))
        fig_cm, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig_cm)

        # distribution of LOF negative outlier scores
        if hasattr(lof, "negative_outlier_factor_"):
            st.subheader("LOF score distribution")
            scores = -lof.negative_outlier_factor_
            fig_hist, ax = plt.subplots()
            ax.hist(scores, bins=50)
            ax.set_title("LOF anomaly score (higher -> more anomalous)")
            st.pyplot(fig_hist)

    elif model_choice == "One-Class SVM (unsupervised)":
        st.markdown("**One-Class SVM** — unsupervised")
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        oc = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
        oc.fit(Xs)
        y_pred_oc = oc.predict(Xs)
        y_pred_flag = np.where(y_pred_oc == -1, 1, 0)
        table, cm = metrics_table(y, y_pred_flag)
        st.table(table.style.format("{:.3f}"))
        fig_cm, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig_cm)

        # SVM decision function histogram
        dec = oc.decision_function(Xs)
        fig_hist, ax = plt.subplots()
        ax.hist(dec, bins=50)
        ax.set_title("OCSVM decision function (lower -> more anomalous)")
        st.pyplot(fig_hist)

    elif model_choice == "Autoencoder (unsupervised)":
        st.markdown("**Autoencoder** — unsupervised (reconstruction error)")

        # cached small autoencoder training on latent features (fast)
        @st.cache_resource(show_spinner=False)
        def train_small_ae(X):
            tf.keras.backend.clear_session()
            input_dim = X.shape[1]
            inp = Input(shape=(input_dim,))
            e = Dense(32, activation='relu')(inp)
            e = Dense(16, activation='relu')(e)
            bottleneck = Dense(8, activation='relu', name='latent_vec')(e)
            d = Dense(16, activation='relu')(bottleneck)
            d = Dense(32, activation='relu')(d)
            out = Dense(input_dim, activation='linear')(d)
            ae = Model(inp, out)
            ae.compile(optimizer='adam', loss='mse')
            ae.fit(X, X, epochs=30, batch_size=256, validation_split=0.1, verbose=0)
            encoder = Model(inp, ae.get_layer('latent_vec').output)
            return ae, encoder

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        with st.spinner("Training Autoencoder (cached) ..."):
            ae, encoder = train_small_ae(Xs)

        recon = ae.predict(Xs, verbose=0)
        rec_err = np.mean((Xs - recon)**2, axis=1)
        # choose threshold: e.g., 95th percentile
        thresh = np.percentile(rec_err, 95)
        y_pred_flag = (rec_err > thresh).astype(int)

        table, cm = metrics_table(y, y_pred_flag)
        st.table(table.style.format("{:.3f}"))

        # show reconstruction error distribution and threshold
        fig_err, ax = plt.subplots()
        ax.hist(rec_err, bins=100)
        ax.axvline(thresh, color='r', linestyle='--', label=f"thr ({thresh:.3e})")
        ax.set_title("Reconstruction error distribution")
        ax.legend()
        st.pyplot(fig_err)

    else:  # Random Forest optional
        st.markdown("**Random Forest (supervised)** — opzionale: utilizza anomaly_flag come target")
        Xdf = pd.DataFrame(X, columns=latent_cols)
        X_train, X_test, y_train, y_test = train_test_split(Xdf, y, test_size=0.3, random_state=42)
        @st.cache_resource
        def train_rf_cached(Xt, yt):
            rf = RandomForestClassifier(n_estimators=200, random_state=42)
            rf.fit(Xt, yt)
            return rf
        with st.spinner("Training Random Forest (cached)..."):
            rf = train_rf_cached(X_train, y_train)

        y_pred = rf.predict(X_test)
        y_prob = rf.predict_proba(X_test)[:,1]
        table, cm = metrics_table(y_test, y_pred)
        st.table(table.style.format("{:.3f}"))

        fig_cm, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig_cm)

        # ROC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        fig_roc, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0,1],[0,1],'--')
        ax.set_title("ROC Curve")
        ax.legend()
        st.pyplot(fig_roc)

        # PR
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        fig_pr, ax = plt.subplots()
        ax.plot(rec, prec)
        ax.set_title("Precision-Recall Curve")
        st.pyplot(fig_pr)

        # SHAP explainability (global)
        try:
            explainer = shap.TreeExplainer(rf)
            # sample test for shape
            shap_sample = X_test.sample(min(300, len(X_test)), random_state=42)
            shap_values = explainer.shap_values(shap_sample)
            st.subheader("SHAP: global importance (summary)")
            fig_sh, ax = plt.subplots(figsize=(6,6))
            shap.summary_plot(shap_values[1], shap_sample, show=False)
            st.pyplot(fig_sh)
        except Exception as e:
            st.warning(f"SHAP failed: {e}")

# ---------------------------
# Modal explainer / single event investigation
st.header("🎬 Modal Explainer — Single Event Investigation")
st.markdown("Scegli un indice (row) del dataset per investigare: mostra feature latenti, anomaly score, SHAP/ recon explanation.")

idx = st.number_input("Select sample index (0..N-1)", min_value=0, max_value=len(df)-1, value=0, step=1)
btn = st.button("Show Explainer for selected sample")
if btn:
    row = df.iloc[idx]
    st.write("### Selected sample summary")
    st.write(row[latent_cols + ["label_technique", "anomaly_flag"]])

    # If RF available and trained, show SHAP force for that sample
    if 'rf' in locals() and model_choice == "Random Forest (supervised)":
        try:
            st.subheader("SHAP force (Random Forest)")
            sample_X = pd.DataFrame([row[latent_cols].values], columns=latent_cols)
            expl = shap.TreeExplainer(rf)
            shap_vals = expl.shap_values(sample_X)
            fig_force = shap.force_plot(expl.expected_value[1], shap_vals[1][0,:], sample_X.iloc[0,:], matplotlib=True)
            st.pyplot(fig_force)
        except Exception as e:
            st.warning(f"SHAP explain failed: {e}")
    else:
        # Provide alternative explain: nearest-centroid difference / reconstruction error components
        st.subheader("Alternative explain (reconstruction or component delta)")
        # show top latent dims that differ from normal mean
        mean_norm = df[df["anomaly_flag"]==0][latent_cols].mean()
        diff = (row[latent_cols] - mean_norm).abs().sort_values(ascending=False)
        st.table(diff.head(10).to_frame("abs_diff_from_normal_mean"))

# ---------------------------
# KPI SOC Dashboard (TPR, FPR, MTTR, TTD)
st.header("📊 KPI SOC Dashboard")
# If you have timestamps in original data you can compute TTD / MTTR — otherwise placeholders
if "ts" in df.columns or "timestamp" in df.columns:
    # user-defined logic: compute detection times etc.
    st.markdown("TTD/MTTR computed from timestamp fields (not implemented: customize per your logs).")
else:
    st.info("TTD / MTTR require timestamped alerts + incident response logs. Currently placeholders shown.")
    # compute TPR/FPR from latest model_choice prediction if available (we used y_pred_flag earlier)
    # For demo, show TPR/FPR from the chosen algorithm (we computed table above)
    try:
        display_table = table.copy()
        st.table(display_table.rename(columns={"value":"metric_value"}))
    except Exception:
        st.info("Run a model first to compute TPR/FPR.")

# ---------------------------
# Live stream Zeek logs + detection feed (simple tail)
st.header("📡 Live Zeek Logs (tail)")
st.markdown("Click 'Refresh' to tail latest lines from a Zeek conn.log (local file). This is not a persistent socket; for production use, wire a websocket or Kafka.")

if refresh_logs:
    st.success("Manual refresh requested.")

def tail_file(path, n_lines=20):
    try:
        with open(path, "rb") as f:
            f.seek(0,2)
            filesize = f.tell()
            block_size = 1024
            data = b""
            while n_lines>0 and filesize>0:
                read_size = min(block_size, filesize)
                f.seek(filesize-read_size)
                chunk = f.read(read_size)
                data = chunk + data
                n_lines = data.count(b"\n")
                filesize -= read_size
            lines = data.splitlines()[-20:]
            return [l.decode(errors="ignore") for l in lines]
    except FileNotFoundError:
        return [f"Zeek log not found: {path}"]

log_lines = tail_file(ZEEK_LOG_PATH, n_lines=50)
st.text_area("Zeek conn.log (last lines)", "\n".join(log_lines), height=300)

# ---------------------------
# Threat Intel (MISP) placeholder
st.header("🌐 Threat Intel (MISP feed)")
st.markdown("""
**Integration notes:** To enable MISP fetching you need:
- `pymisp` installed (`pip install pymisp`)
- MISP URL and API key (kept secret)
- Example: use `from pymisp import ExpandedPyMISP` and call `misphttp.search_index` or `misphttp.get_event`.

Below show top techniques detected from our dataset (heatmap).
""")
mitre_counts = df.groupby("label_technique")["anomaly_flag"].sum().reset_index()
mitre_counts.columns = ["Technique", "Detected"]
fig_mitre, ax = plt.subplots(figsize=(6,8))
sns.barplot(x="Detected", y="Technique", data=mitre_counts.sort_values("Detected", ascending=False), palette="Reds", ax=ax)
ax.set_title("Detected anomalies per MITRE technique")
st.pyplot(fig_mitre)

# ---------------------------
st.sidebar.header("About / Run")
st.sidebar.markdown("Run this app with:\n```\nstreamlit run app.py\n```")
st.sidebar.markdown("Make sure the dataset parquet path is correct and dependencies installed.")
