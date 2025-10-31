import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import time
import hashlib

# ================================
# CONFIGURAZIONE APP
# ================================
st.set_page_config(page_title="ZeekML Intrusion Analyzer", layout="wide")
st.title("🔍 Zeek ML Intrusion Analyzer (Supervised + Semi-Supervised)")

DATA_PATH = "./data/X_ensemble_latent_labeled.parquet"

# ================================
# LOAD DATA
# ================================
@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_PATH)
    return df

df = load_data()

# ================================
# MODE SELECTION
# ================================
mode = st.sidebar.selectbox(
    "Select Task",
    [
        "🔴 Binary Intrusion Detection (Anomaly Flag)",
        "🟦 MITRE Technique Classification"
    ]
)

if mode == "🔴 Binary Intrusion Detection (Anomaly Flag)":
    target_col = "anomaly_flag"
    st.markdown("📌 **Task:** Detect attacks (1) vs normal (0)")
else:
    target_col = "label_technique"
    st.markdown("📌 **Task:** Classify MITRE attack technique")

# ================================
# PREPARE DATA
# ================================
feature_cols = [c for c in df.columns if c.startswith("latent_")]

X = df[feature_cols].values
y = df[target_col].values

# ================================
# TRAIN/TEST SPLIT
# ================================
@st.cache_data
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train, X_test, y_train, y_test = split_data(X, y)

# ================================
# MODEL HASH KEY (avoid retrain)
# ================================
def make_hash():
    s = (str(X_train.sum()) + str(y_train.sum())).encode()
    return hashlib.md5(s).hexdigest()

model_key = make_hash()

# ================================
# TRAIN MODEL (CACHED)
# ================================
@st.cache_resource
def train_model_cached(key, X_train, y_train):
    with st.spinner("Training Random Forest..."):
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        )
        model.fit(X_train, y_train)
        return model

model = train_model_cached(model_key, X_train, y_train)

# ================================
# PREDICTION
# ================================
@st.cache_data
def predict_cached(model, X_test):
    return model.predict(X_test)

y_pred = predict_cached(model, X_test)

# ================================
# METRICS
# ================================
st.subheader("📊 Model Performance")

if mode == "🔴 Binary Intrusion Detection (Anomaly Flag)":
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    st.write(f"**ROC-AUC:** {auc:.4f}")

st.text("Classification Report")
st.code(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
st.write("Confusion Matrix")
st.write(pd.DataFrame(cm,
    index=["True Normal","True Attack"] if target_col=="anomaly_flag" else np.unique(y),
    columns=["Pred Normal","Pred Attack"] if target_col=="anomaly_flag" else np.unique(y)
))

if mode == "🔴 Binary Intrusion Detection (Anomaly Flag)":
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr})
    fig_roc = px.line(roc_df, x="FPR", y="TPR", title="ROC Curve")
    st.plotly_chart(fig_roc, use_container_width=True)

# ================================
# VISUALIZATION (PCA + TSNE)
# ================================
st.subheader("🔎 Projection Visualization")

proj_type = st.selectbox("Projection Type", ["PCA", "t-SNE"])

@st.cache_data
def compute_projection(X, method):
    if method == "PCA":
        reducer = PCA(n_components=2)
    else:
        reducer = TSNE(n_components=2, learning_rate="auto", init="random", perplexity=30)

    return reducer.fit_transform(X)

with st.spinner("Computing projection…"):
    proj = compute_projection(X, proj_type)

proj_df = pd.DataFrame({
    "x": proj[:,0], "y": proj[:,1],
    "label": y.astype(str)
})

fig_proj = px.scatter(proj_df, x="x", y="y", color="label", title=f"{proj_type} Projection")
st.plotly_chart(fig_proj, use_container_width=True)

# ================================
# SHAP EXPLAINABILITY
# ================================
st.subheader("🧠 SHAP Explainability")

try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[:500])

    st.write("Feature Importance")
    shap.summary_plot(shap_values, X_test[:500], feature_names=feature_cols, show=False)
    st.pyplot(bbox_inches='tight', dpi=120)

except Exception as e:
    st.error(f"SHAP plotting failed: {e}")

# ================================
# RERUN BUTTON
# ================================
if st.button("🔁 Retrain / Refresh"):
    st.cache_resource.clear()
    st.cache_data.clear()
    st.success("Cache cleared. Refresh page (CTRL+R).")
