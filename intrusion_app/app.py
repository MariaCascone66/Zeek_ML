import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="ML IDS Dashboard", layout="wide")
st.title("🔎 ML Intrusion Detection Interactive Report")

# -------------------------------
# ✅ Cache Data Loading
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_parquet("data/X_ensemble_latent_labeled.parquet")
    latent_cols = [c for c in df.columns if c.startswith("latent")]
    return df, latent_cols

df, latent_cols = load_data()
st.markdown(f"Dataset: **{df.shape[0]} samples**, **{len(latent_cols)} latent features**")

# -------------------------------
# ✅ Cached Sample
# -------------------------------
@st.cache_data
def get_sample(df, n=5000):
    return df.sample(n, random_state=42)

sample_df = get_sample(df)

# -------------------------------
# 🎯 Sidebar Controls
# -------------------------------
st.sidebar.header("Controls")
dr_method = st.sidebar.selectbox("Dimensionality Reduction", ["UMAP", "t-SNE"])
label_choice = st.sidebar.selectbox("Attack Label Filter", ["ALL"] + sorted(df["label_technique"].unique()))
show_anom = st.sidebar.checkbox("Show anomalies only", False)

# -------------------------------
# ✅ Cached UMAP / t-SNE
# -------------------------------
@st.cache_data
def compute_projection(method, data):
    if method == "UMAP":
        import umap
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    else:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, perplexity=30, learning_rate='auto', random_state=42)

    return reducer.fit_transform(data)

with st.spinner(f"Computing {dr_method} projection..."):
    X_proj = compute_projection(dr_method, sample_df[latent_cols])

sample_df["Dim1"], sample_df["Dim2"] = X_proj[:,0], X_proj[:,1]

df_plot = sample_df.copy()
if show_anom: df_plot = df_plot[df_plot.anomaly_flag == 1]
if label_choice != "ALL": df_plot = df_plot[df_plot.label_technique == label_choice]

fig = px.scatter(
    df_plot, x="Dim1", y="Dim2",
    color="label_technique", symbol="anomaly_flag",
    title=f"{dr_method} Latent Space"
)
st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# ✅ Cached Random Forest Model
# -------------------------------
@st.cache_resource
def train_rf(X, y):
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X, y)
    return rf

# -------------------------------
# 🧠 Model Selection
# -------------------------------
model_choice = st.radio("Choose Model", ["Random Forest (Supervised)", "Local Outlier Factor (Unsupervised)"])

if model_choice == "Random Forest (Supervised)":
    st.subheader("📌 Random Forest Results")

    X, y = df[latent_cols], df["anomaly_flag"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    with st.spinner("Training RF... (cached)"):
        rf = train_rf(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    # Metrics table
    report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T
    st.dataframe(report[['precision','recall','f1-score']])

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    st.pyplot(fig_cm)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig_roc, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc(fpr,tpr):.2f}")
    ax.plot([0,1],[0,1],'--')
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig_roc)

else:
    st.subheader("📌 Local Outlier Factor (Unsupervised)")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[latent_cols])
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True)
    lof.fit(X_scaled)
    y_pred_flag = np.where(lof.predict(X_scaled) == -1, 1, 0)

    st.write(pd.DataFrame({"LOF Pred": y_pred_flag}).head(10))

    # If labels exist, evaluate
    y_true = df["anomaly_flag"].values
    precision = (y_pred_flag & y_true).sum() / max(y_pred_flag.sum(),1)
    recall = (y_pred_flag & y_true).sum() / y_true.sum()
    f1 = 2 * precision * recall / max(precision + recall, 1)
    
    st.markdown(f"**Precision:** {precision:.3f}")
    st.markdown(f"**Recall:** {recall:.3f}")
    st.markdown(f"**F1:** {f1:.3f}")
