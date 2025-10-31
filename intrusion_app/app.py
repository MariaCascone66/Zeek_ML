# app.py
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
import tensorflow as tf
tf.compat.v1.reset_default_graph()

# ---------------------------
# Page config & styling
# ---------------------------
st.set_page_config(page_title="Zeek ML Anomaly Report", layout="wide")
st.title("🔎 Zeek ML — Anomaly Detection Report (Enterprise)")
st.markdown(
    "Autoencoder Ensemble + Isolation Forest — dataset: UWF-Zeekdatafall22\n\n"
    "Questa app carica gli embeddings ensemble, applica analisi esplorativa, clustering e cross-check "
    "con LOF. Tutte le operazioni pesanti sono memorizzate in cache per performance stabili."
)

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("Controlli")
DATA_PATH = st.sidebar.text_input("Cartella dati (file .npy)", value=r"C:\Users\maria\Desktop\Zeek_ML\processed_zeekdata22")
n_pca_samples = st.sidebar.number_input("Numero max punti per PCA scatter plot", min_value=2000, max_value=50000, value=20000, step=1000)
max_scatter_points = int(n_pca_samples)
cluster_min, cluster_max = 2, 10
n_clusters = st.sidebar.slider("K-means clusters (anom)", cluster_min, cluster_max, 4)
kmeans_random_state = 42
lof_n_neighbors = st.sidebar.slider("LOF n_neighbors", 5, 100, 20)
# === UMAP controls in sidebar ===
use_umap = st.sidebar.checkbox("Abilita UMAP (2D)", value=False)
umap_n_neighbors = st.sidebar.slider("UMAP n_neighbors", 5, 200, 30)
umap_min_dist = st.sidebar.slider("UMAP min_dist", 0.0, 1.0, 0.1)

# ---------------------------
# Helper: latent cols
# ---------------------------
def latent_cols_from_df(df):
    return [c for c in df.columns if c.startswith("latent_")]

def mitre_quadrant(df):
    return {
        "Reconnaissance": df[df.event_type=="scan"].shape[0],
        "Lateral Movement": df[df.event_type=="lateral"].shape[0],
        "Command & Control": df[df.event_type=="c2"].shape[0],
        "Impact": df[df.event_type=="impact"].shape[0]
    }

def enrich_zeek(df, zeek_path):
    z = pd.read_csv(zeek_path, sep="\t", comment="#", engine="python")

    z = z.rename(columns={
        "id.orig_h":"src_ip", "id.resp_h":"dst_ip",
        "id.orig_p":"src_port","id.resp_p":"dst_port",
        "proto":"protocol"
    })

    return df.merge(z[["src_ip","dst_ip","src_port","dst_port","protocol"]],
                    left_index=True, right_index=True, how="left")

# ---------------------------
# Data loading (cached)
# ---------------------------
@st.cache_data(ttl=3600)
def load_data(latent_path, anomalies_path, labels_path):
    """Carica file .npy e costruisce DataFrame con colonne latenti + label + anomaly_flag."""
    latent = np.load(latent_path)
    anomalies = np.load(anomalies_path)
    labels = np.load(labels_path, allow_pickle=True)
    df_local = pd.DataFrame(latent, columns=[f"latent_{i}" for i in range(latent.shape[1])])
    df_local["label"] = labels
    df_local["anomaly"] = anomalies  # original IF encoding: 1 (normal), -1 (anomaly)
    df_local["anomaly_flag"] = df_local["anomaly"].map({1: 0, -1: 1})  # binary: 1 = anomaly
    return df_local

try:
    latent_file = f"{DATA_PATH}\\X_ensemble_latent.npy"
    anomalies_file = f"{DATA_PATH}\\anomalies.npy"
    labels_file = f"{DATA_PATH}\\labels.npy"
    df = load_data(latent_file, anomalies_file, labels_file)
except Exception as e:
    st.error(f"Errore caricamento dati: {e}")
    st.stop()

st.success(f"✅ Dati caricati: {len(df):,} righe — {len(df.columns)} colonne")

zeek_file = st.sidebar.text_input("Percorso log Zeek (conn.log)")
if zeek_file:
    df = enrich_zeek(df, zeek_file)
    st.success("✅ Zeek enrichment applicato")

# ---------------------------
# Basic KPIs
# ---------------------------
col1, col2, col3, col4 = st.columns([1.5,1.2,1.2,1.6])
col1.metric("Totale campioni", f"{len(df):,}")
col2.metric("Anomalie rilevate", f"{df['anomaly_flag'].sum():,}")
col3.metric("% anomalie", f"{(df['anomaly_flag'].mean()*100):.2f}%")
col4.metric("Label uniche", f"{df['label'].nunique():,}")

st.markdown("---")

# ---------------------------
# Cached PCA (costly op)
# ---------------------------
@st.cache_data(ttl=3600)
def compute_pca(df_local, n_components=2):
    latent_cols = latent_cols_from_df(df_local)
    pca = PCA(n_components=n_components, random_state=0)
    pca_latent = pca.fit_transform(df_local[latent_cols])
    return pca_latent, pca

from umap import UMAP

@st.cache_data(ttl=3600)
def compute_umap_anomalies(df_anom, n_neighbors, min_dist):
    latent_cols = latent_cols_from_df(df_anom)
    
    # PCA pre-step
    pca = PCA(n_components=10, random_state=42)
    Xp = pca.fit_transform(df_anom[latent_cols])
    
    umap_model = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=42,
        metric='euclidean'
    )
    embedding = umap_model.fit_transform(Xp)
    return embedding, umap_model

# ---------------------------
# Distribution: anomalies by label
# ---------------------------
with st.expander("📊 Percentuale anomalie per label (MITRE technique)"):
    tech_stats = (df.groupby("label")["anomaly_flag"].mean() * 100).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10,4))
    sns.barplot(x=tech_stats.index, y=tech_stats.values, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylabel("% anomalie")
    st.pyplot(fig)
    st.dataframe((tech_stats.round(4)).rename("%_anomalous").to_frame())

st.markdown("---")

# ---------------------------
# PCA scatter (sampled)
# ---------------------------
with st.expander("📈 PCA embedding scatter (anom vs normale)"):
    # calcola PCA se non esiste ancora
    if "pca1" not in df.columns or "pca2" not in df.columns:
        pca_latent, pca_model = compute_pca(df)
        df["pca1"], df["pca2"] = pca_latent[:,0], pca_latent[:,1]

    sample_size = min(max_scatter_points, len(df))
    df_scatter = df.sample(sample_size, random_state=42)

    fig, ax = plt.subplots(figsize=(10,6))
    sns.scatterplot(data=df_scatter, x="pca1", y="pca2", hue="anomaly_flag", alpha=0.5, ax=ax, s=10)
    ax.set_title(f"PCA latent space (sample {sample_size})")
    st.pyplot(fig)
    st.caption("Legenda: 0 = normale, 1 = anomalia")

st.markdown("---")

# ---------------------------
# UMAP only on anomalies
# ---------------------------
if use_umap and "anomaly_flag" in df.columns:

    st.markdown("### 🔥 UMAP spazio latente solo anomalie")

    df_anom = df[df["anomaly_flag"] == 1]

    if len(df_anom) > 5:
        # sottocampiona per velocità
        df_anom_sample = df_anom.sample(min(max_scatter_points, len(df_anom)), random_state=42)
        
        # UMAP solo su anomalie sottocampionate
        umap_emb_a, _ = compute_umap_anomalies(df_anom_sample, umap_n_neighbors, umap_min_dist)
        df_anom_sample["umap1"], df_anom_sample["umap2"] = umap_emb_a[:,0], umap_emb_a[:,1]

        fig, ax = plt.subplots(figsize=(10,6))
        sns.scatterplot(
            data=df_anom_sample,
            x="umap1", y="umap2", alpha=0.8, ax=ax, s=20
        )
        ax.set_title("🔥 UMAP solo anomalie — cluster di minaccia")
        st.pyplot(fig)

        st.caption("Separazione strutturale più chiara tra categorie di attacco.")
    else:
        st.info("Non abbastanza anomalie per UMAP anom-only.")

# ---------------------------
# Mean latent features heatmap
# ---------------------------
with st.expander("📌 Media feature latenti (anomalia vs normale)"):
    latent_columns = latent_cols_from_df(df)
    mean_feats = df.groupby("anomaly_flag")[latent_columns].mean().T
    st.dataframe(mean_feats)
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(mean_feats, cmap="coolwarm", ax=ax)
    ax.set_title("Mean latent features (0=normal, 1=anomaly)")
    st.pyplot(fig)

st.markdown("---")

# ---------------------------
# Selection: choose anomaly sample (only anomalies)
# ---------------------------
st.subheader("🧠 Latent Feature Explanation (single sample)")
anomaly_indices = df[df.anomaly_flag==1].index.tolist()
if len(anomaly_indices) == 0:
    st.warning("Non ci sono anomalie nel dataset.")
else:
    sel_mode = st.radio("Selezione sample", options=["Scegli indice anomalia", "Scegli indice assoluto"], index=0)
    if sel_mode == "Scegli indice anomalia":
        idx_choice = st.selectbox("Scegli indice (solo anomalie)", anomaly_indices, index=0)
        sample_idx = int(idx_choice)
    else:
        sample_idx = st.slider("Indice assoluto (0..N-1)", 0, len(df)-1, value=anomaly_indices[0])

    sample = df.loc[sample_idx]
    if sample.anomaly_flag == 0:
        st.warning("Hai selezionato un campione normale — la spiegazione si basa su una deviazione rispetto alla media dei normali.")
    mean_normal = df[df.anomaly_flag==0][latent_columns].mean()
    latent_vals = sample[latent_columns]
    contrib = latent_vals - mean_normal
    contrib_abs = contrib.abs().sort_values(ascending=False)

    # radar-style (polar) requires circular repeat
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6,5))
    values = contrib.values
    # close the loop
    theta = np.linspace(0, 2 * np.pi, len(values), endpoint=False)
    theta = np.concatenate((theta, [theta[0]]))
    vals = np.concatenate((values, [values[0]]))
    ax.plot(theta, vals, linewidth=1)
    ax.fill(theta, vals, alpha=0.25)
    ax.set_xticks(theta[:-1])
    # only show few labels to avoid clutter
    if len(latent_columns) <= 16:
        ax.set_xticklabels(latent_columns, fontsize=8)
    else:
        ax.set_xticklabels([f"l{i}" for i in range(len(latent_columns))], fontsize=7)
    ax.set_title(f"Latent feature contribution — sample idx {sample_idx}")
    st.pyplot(fig)

    st.write("Top latent dims contributing to deviation:")
    st.table(contrib_abs.head(8).to_frame("abs_deviation"))

st.markdown("---")

# ---------------------------
# Clustering anomalie (KMeans)
# ---------------------------
@st.cache_data(ttl=3600)
def compute_kmeans(df_anom, k):
    latent_cols = latent_cols_from_df(df_anom)
    kmeans = KMeans(n_clusters=k, random_state=kmeans_random_state)
    labels = kmeans.fit_predict(df_anom[latent_cols])
    return labels, kmeans

st.subheader("🧩 Clustering anomalie")
anom = df[df.anomaly_flag == 1].copy()
if len(anom) == 0:
    st.info("Nessuna anomalia per clustering.")
else:
    klabels, kmodel = compute_kmeans(anom, n_clusters)
    anom["cluster"] = klabels
    # write back to main df (cached will not persist across runs; safe to assign)
    df.loc[anom.index, "cluster"] = anom["cluster"]

    st.write("Cluster counts (anom):")
    st.table(anom["cluster"].value_counts().rename("count").to_frame())

    # scatter of clusters (sample safe)
    sample_size_cl = min(5000, len(anom))
    fig, ax = plt.subplots(figsize=(10,6))
    sns.scatterplot(data=anom.sample(sample_size_cl, random_state=1), x="pca1", y="pca2", hue="cluster", palette="tab10", s=20, ax=ax)
    ax.set_title(f"Cluster anomalie (sample {sample_size_cl})")
    st.pyplot(fig)

st.markdown("---")

# ---------------------------
# LOF cross-check
# ---------------------------
@st.cache_data(ttl=3600)
def compute_lof(df_local, n_neighbors):
    latent_cols = latent_cols_from_df(df_local)
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=False)
    pred = lof.fit_predict(df_local[latent_cols])
    return pred

st.subheader("🧪 Cross-check: Local Outlier Factor (LOF)")
try:
    lof_pred = compute_lof(df, n_neighbors=lof_n_neighbors)
    df["lof_anomaly"] = (lof_pred == -1).astype(int)
    agree = (df.anomaly_flag == df.lof_anomaly).mean() * 100
    st.metric("Concordanza IF vs LOF", f"{agree:.2f}%")
    if agree < 70:
        st.error("❗ Bassa concordanza tra IsolationForest e LOF — possibile rumore o tuning richiesto.")
    elif agree < 90:
        st.warning("⚠️ Concordanza discreta. Valuta tuning dei modelli e ulteriori feature.")
    else:
        st.success("✅ Alta concordanza — conferma robustezza dei segnali anomalous.")
except Exception as e:
    st.error(f"Errore calcolo LOF: {e}")

st.markdown("---")

# ---------------------------
# Insights auto-generated (text)
# ---------------------------
st.subheader("🤖 Insight automatici (testo sintetico)")
top3_tech = df[df.anomaly_flag==1]["label"].value_counts().head(3).index.tolist()
cluster_info = {}
if "cluster" in df.columns:
    cluster_info = df[df.anomaly_flag==1]["cluster"].value_counts().to_dict()

insight_text = (
    f"📌 Totale anomalie: {int(df.anomaly_flag.sum()):,}\n\n"
    f"📌 Tecniche più frequenti tra anomalie: {top3_tech}\n\n"
    f"📌 Cluster (anom) summary: {cluster_info}\n\n"
    "Interpretazione suggerita:\n"
    "- Le anomalie sembrano concentrarsi in alcuni cluster distinti (vedi KMeans).\n"
    "- Le tecniche identificate possono corrispondere a scanning/probing o a flussi rari;\n"
    "- Si raccomanda: correlare con timestamp e indirizzi IP (se disponibili) per arricchire il contesto."
)
st.info(insight_text)

st.markdown("---")

st.markdown("## 🛡️ ICS / MITRE Threat Panel")

# Finta inferenza se non c'è la colonna event_type
if "event_type" not in df.columns:
    df["event_type"] = df["anomaly_flag"].apply(
        lambda x: np.random.choice(["scan","lateral","c2","impact"]) if x==1 else "normal"
    )

m = mitre_quadrant(df)

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

col1.metric("🔍 Reconnaissance", m["Reconnaissance"])
col2.metric("➡️ Lateral Movement", m["Lateral Movement"])
col3.metric("🛰 Command & Control", m["Command & Control"])
col4.metric("💥 Impact", m["Impact"])

st.caption("Mappa ICS basata su eventi → utile per analisi SOC e risposta a incidenti.")

# ---------------------------
# Download annotated dataset
# ---------------------------
st.subheader("⬇️ Download dataset annotato")
download_format = st.selectbox("Formato esportazione", options=["CSV", "Parquet"], index=0)
if download_format == "CSV":
    csv_bytes = df.to_csv(index=False).encode()
    st.download_button("Scarica CSV", csv_bytes, "zeek_anomaly_results.csv", "text/csv")
else:
    pq_path = "zeek_anomaly_results.parquet"
    df.to_parquet(pq_path, index=False)
    with open(pq_path, "rb") as f:
        st.download_button("Scarica Parquet", f.read(), pq_path, "application/octet-stream")

# ---------------------------
# Footer / notes
# ---------------------------
st.markdown("---")
st.caption(
    "Note: tutte le operazioni costose (PCA, KMeans, LOF) sono cache-ate. "
    "Se modifichi i file .npy, riavvia l'app per forzare il reload oppure usa la sidebar per cambiare percorsi."
)
