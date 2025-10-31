# app.py (CORRETTO)
import os, warnings, time
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc, precision_recall_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.tree import DecisionTreeClassifier

import shap
import umap

st.set_page_config(page_title="ML Intrusion Dashboard", layout="wide")

# -----------------------
# Load data (cached)
# -----------------------
@st.cache_data
def load_data(path):
    df = pd.read_parquet(path)
    latent_cols = [c for c in df.columns if c.startswith("latent")]
    return df, latent_cols

DATA = os.path.join("data", "X_ensemble_latent_labeled.parquet")
df, latent_cols = load_data(DATA)

st.header("🔎 ML Intrusion Detection Dashboard")
st.write(f"Samples: **{df.shape[0]}** — Latent features: **{len(latent_cols)}**")

# -----------------------
# Sample for projection
# -----------------------
sample_size = st.sidebar.slider("Projection sample size", 1000, min(50000, len(df)), 5000, step=500)
df_s = df.sample(min(sample_size, len(df)), random_state=42).reset_index(drop=True)
X_sample = df_s[latent_cols].to_numpy()

# -----------------------
# Projection selection
# -----------------------
proj_method = st.sidebar.selectbox("Projection method", ["UMAP", "t-SNE"])
st.subheader(f"Latent space projection — {proj_method}")

@st.cache_data
def compute_umap(X, n_neighbors=15, min_dist=0.1):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    return reducer.fit_transform(X)

@st.cache_data
def compute_tsne(X, perplexity=30):
    # use n_iter_without_progress to be compatible across sklearn versions
    from sklearn.manifold import TSNE as SKTSNE
    tsne = SKTSNE(n_components=2, perplexity=perplexity, learning_rate="auto",
                  n_iter_without_progress=300, init="pca", random_state=42)
    return tsne.fit_transform(X)

with st.spinner(f"Computing {proj_method}..."):
    if proj_method == "UMAP":
        X_proj = compute_umap(X_sample, n_neighbors=15, min_dist=0.1)
    else:
        # safe perplexity based on sample size
        safe_p = min(50, max(5, X_sample.shape[0] // 3 - 1))
        X_proj = compute_tsne(X_sample, perplexity=safe_p)

df_vis = pd.DataFrame(X_proj, columns=["Dim1", "Dim2"])
# choose a sensible label column present in your parquet:
# we use 'label_technique' for color and 'anomaly_flag' for symbol
label_col = "label_technique" if "label_technique" in df.columns else ( "label" if "label" in df.columns else None )
if label_col is None:
    df_vis["label"] = "unknown"
else:
    df_vis["label"] = df_s[label_col].values
df_vis["anomaly_flag"] = df_s["anomaly_flag"].values if "anomaly_flag" in df_s.columns else 0

fig = sns.scatterplot(x="Dim1", y="Dim2", hue="label", style="anomaly_flag", data=df_vis, legend="brief", s=30, palette="tab10")
plt.title(f"{proj_method} projection (sample {len(df_vis)})")
plt.tight_layout()
st.pyplot(plt.gcf())
plt.clf()

# -----------------------
# Models & Metrics (below visualization)
# -----------------------
st.markdown("---")
st.subheader("Models & Metrics")

# We'll use the anomaly_flag as supervised target (0/1). This exists in your dataset.
if "anomaly_flag" not in df.columns:
    st.error("Column 'anomaly_flag' not found in dataset — supervised metrics unavailable.")
else:
    # use full dataset for training evaluation (you can change to train/test split)
    X_all = df[latent_cols]
    y_all = df["anomaly_flag"].astype(int)

    model_choice = st.selectbox("Choose model", ["Random Forest (supervised)", "Local Outlier Factor (unsupervised)", "One-Class SVM (unsupervised)"])

    if model_choice == "Random Forest (supervised)":
        st.markdown("### Random Forest (supervised)")

        # train/test split to show realistic metrics
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.3, random_state=42, stratify=y_all)

        with st.spinner("Training Random Forest..."):
            rf = RandomForestClassifier(n_estimators=200, random_state=42)
            rf.fit(X_tr, y_tr)

        y_pred = rf.predict(X_te)
        y_prob = rf.predict_proba(X_te)[:, 1] if hasattr(rf, "predict_proba") else None

        # metrics
        report = classification_report(y_te, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.loc[["0","1","micro avg","macro avg","weighted avg"], ["precision","recall","f1-score","support"]], width=700)

        # confusion matrix (binary safe)
        cm = confusion_matrix(y_te, y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            st.write(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")
        fig_cm, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig_cm); plt.clf()

        # ROC & PR curves if probabilities exist
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_te, y_prob)
            roc_auc = auc(fpr, tpr)
            fig_roc, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
            ax.plot([0,1],[0,1],'--', color='gray')
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()
            st.pyplot(fig_roc); plt.clf()

            prec, rec, _ = precision_recall_curve(y_te, y_prob)
            fig_pr, ax = plt.subplots()
            ax.plot(rec, prec)
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
            st.pyplot(fig_pr); plt.clf()

        # Feature importance + SHAP (sampled)
        feat_imp = pd.Series(rf.feature_importances_, index=latent_cols).sort_values(ascending=False)
        st.subheader("Top latent features (RF importance)")
        st.bar_chart(feat_imp.head(15))

        # SHAP: use sample to avoid heavy computation
        st.subheader("SHAP explainability (sample)")
        try:
            X_shap = X_te.sample(min(300, len(X_te)), random_state=42)
            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(X_shap)
            # shap_values might be list (binary) or array
            if isinstance(shap_values, list):
                vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                vals = shap_values
            fig_shap = plt.figure(figsize=(8,5))
            shap.summary_plot(vals, X_shap, show=False)
            st.pyplot(fig_shap); plt.clf()
        except Exception as e:
            st.warning(f"SHAP plotting failed: {e}")

    elif model_choice == "Local Outlier Factor (unsupervised)":
        st.markdown("### Local Outlier Factor (LOF)")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_all)
        with st.spinner("Fitting LOF..."):
            lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True)
            lof.fit(X_scaled)
            lof_pred = lof.predict(X_scaled)
            lof_flag = np.where(lof_pred == -1, 1, 0)

        st.write(f"LOF flagged {int(lof_flag.sum())} anomalies ({lof_flag.sum()/len(lof_flag)*100:.2f}%)")
        # if labels exist, compute metrics
        if y_all is not None:
            m = {
                "accuracy": accuracy_score(y_all, lof_flag),
                "precision": precision_score(y_all, lof_flag, zero_division=0),
                "recall": recall_score(y_all, lof_flag, zero_division=0),
                "f1": f1_score(y_all, lof_flag, zero_division=0)
            }
            st.table(pd.DataFrame([m]).T)

        # surrogate tree for explainability
        st.markdown("Surrogate DecisionTree (approx LOF)")
        sur = DecisionTreeClassifier(max_depth=3)
        sur.fit(X_all, lof_flag)
        imp = pd.Series(sur.feature_importances_, index=latent_cols).sort_values(ascending=False)
        st.bar_chart(imp.head(12))

    else:
        st.markdown("### One-Class SVM (unsupervised)")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_all)
        with st.spinner("Fitting OCSVM..."):
            oc = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05)
            oc.fit(X_scaled)
            oc_pred = oc.predict(X_scaled)
            oc_flag = np.where(oc_pred == -1, 1, 0)
        st.write(f"OCSVM flagged {int(oc_flag.sum())} anomalies ({oc_flag.sum()/len(oc_flag)*100:.2f}%)")
        if y_all is not None:
            m = {
                "accuracy": accuracy_score(y_all, oc_flag),
                "precision": precision_score(y_all, oc_flag, zero_division=0),
                "recall": recall_score(y_all, oc_flag, zero_division=0),
                "f1": f1_score(y_all, oc_flag, zero_division=0)
            }
            st.table(pd.DataFrame([m]).T)

# -----------------------
# MITRE heatmap + KPI placeholders
# -----------------------
st.markdown("---")
st.subheader("MITRE ATT&CK Heatmap & SOC KPIs")
if "label_technique" in df.columns:
    mitre_counts = df.groupby("label_technique")["anomaly_flag"].sum().reset_index().sort_values("anomaly_flag", ascending=False)
    fig_m, ax = plt.subplots(figsize=(6, max(3, len(mitre_counts)*0.25)))
    sns.barplot(x="anomaly_flag", y="label_technique", data=mitre_counts, palette="Reds", ax=ax)
    ax.set_xlabel("Detected anomalies")
    ax.set_ylabel("Technique")
    st.pyplot(fig_m); plt.clf()

# KPI simple
col1, col2, col3, col4 = st.columns(4)
# use last computed prediction if available
try:
    last_pred = lof_flag if 'lof_flag' in locals() else (oc_flag if 'oc_flag' in locals() else None)
    if last_pred is None and 'rf' in locals():
        # use rf trained on X_te if present
        last_pred = rf.predict(X_all) if 'rf' in locals() else np.zeros(len(df))
    if last_pred is None:
        last_pred = np.zeros(len(df))
    if "anomaly_flag" in df.columns:
        y_true_all = df["anomaly_flag"].astype(int).values
        cm = confusion_matrix(y_true_all, last_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        else:
            tpr = recall_score(y_true_all, last_pred, zero_division=0)
            fpr = 0.0
    else:
        tpr, fpr = 0.0, 0.0
except Exception:
    tpr, fpr = 0.0, 0.0

col1.metric("TPR (Recall)", f"{tpr:.3f}")
col2.metric("FPR", f"{fpr:.3f}")
col3.metric("MTTR (sim)", "—")
col4.metric("TTD (sim)", "—")

st.markdown("### Modal explainer / nearest neighbors")
idx = st.number_input("Index in projection sample (0..n-1)", min_value=0, max_value=len(df_s)-1, value=5)
if st.button("Show detail"):
    ev = df_s.iloc[idx]
    st.write("Selected event:")
    st.write(ev[latent_cols + (['label_technique'] if 'label_technique' in df_s.columns else []) + (['anomaly_flag'] if 'anomaly_flag' in df_s.columns else [])])
    # nearest neighbors
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=6).fit(df[latent_cols].to_numpy())
    dist, ind = nbrs.kneighbors(ev[latent_cols].to_numpy().reshape(1,-1))
    neigh = pd.DataFrame({"index": ind[0][1:], "distance": dist[0][1:]})
    st.dataframe(neigh)

st.markdown("### End of dashboard")
