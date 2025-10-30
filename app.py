import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import umap
import seaborn as sns
import matplotlib.pyplot as plt

st.title("🔎 ML Intrusion Detection Interactive Report")

# -------------------------------
# Caricamento dataset
# -------------------------------
df = pd.read_parquet("C:/Users/maria/Desktop/Zeek_ML/processed_zeekdata22/X_ensemble_latent_labeled.parquet")
latent_cols = [c for c in df.columns if c.startswith("latent")]
st.markdown(f"Dataset caricato: **{df.shape[0]} campioni**, **{len(latent_cols)} feature latenti**")

# -------------------------------
# UMAP riduzione dimensionale
# -------------------------------
st.subheader("📊 UMAP Attack Visualization")
sample_df = df.sample(5000, random_state=42)
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(sample_df[latent_cols])
sample_df["UMAP1"] = X_umap[:, 0]
sample_df["UMAP2"] = X_umap[:, 1]

label_choice = st.selectbox("Filtro label attacco:", ["ALL"] + sorted(df["label_technique"].unique()))
show_anom = st.checkbox("Mostra solo anomalie", False)

df_plot = sample_df.copy()
if show_anom:
    df_plot = df_plot[df_plot.anomaly_flag == 1]
if label_choice != "ALL":
    df_plot = df_plot[df_plot.label_technique == label_choice]

fig = px.scatter(
    df_plot, x="UMAP1", y="UMAP2",
    color="label_technique",
    symbol="anomaly_flag",
    hover_data=["label_technique", "anomaly_flag"],
    title="UMAP Attack Visualization"
)
st.plotly_chart(fig)

# -------------------------------
# Selezione modello
# -------------------------------
model_choice = st.radio("Scegli il modello:", ["Random Forest (supervised)", "Local Outlier Factor (unsupervised)"])

if model_choice == "Random Forest (supervised)":
    st.subheader("📌 Random Forest Performance")
    
    X = df[latent_cols]
    y = df["anomaly_flag"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:,1]
    
    # Report tabellare
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df[['precision','recall','f1-score']])
    
    # Matrice di confusione
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predetti")
    ax.set_ylabel("Reali")
    st.pyplot(fig_cm)
    
    # Feature importance
    feat_imp = pd.Series(rf.feature_importances_, index=latent_cols).sort_values(ascending=False)
    st.subheader("🔑 Feature Importance (Top 15)")
    st.bar_chart(feat_imp.head(15))
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    fig_roc, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    ax.plot([0,1],[0,1],'--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig_roc)
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    fig_pr, ax = plt.subplots()
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    st.pyplot(fig_pr)

else:  # LOF / One-Class SVM
    st.subheader("📌 Local Outlier Factor (unsupervised)")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[latent_cols])
    
    # Modello LOF
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True)
    lof.fit(X_scaled)
    y_pred = lof.predict(X_scaled)
    # mappa -1 -> anomalia, 1 -> normale
    y_pred_flag = np.where(y_pred==-1, 1, 0)
    
    st.markdown("Esempio di predizioni unsupervised (1=anomalia, 0=normale)")
    st.dataframe(pd.DataFrame({"Predizione LOF": y_pred_flag}).head(20))
    
    # Possiamo calcolare metriche semi-supervised se abbiamo labels
    if 'anomaly_flag' in df.columns:
        y_true = df['anomaly_flag'].values
        precision = (y_pred_flag & y_true).sum() / y_pred_flag.sum()
        recall = (y_pred_flag & y_true).sum() / y_true.sum()
        f1 = 2*precision*recall/(precision+recall)
        st.markdown(f"- Precision: {precision:.3f}")
        st.markdown(f"- Recall: {recall:.3f}")
        st.markdown(f"- F1-score: {f1:.3f}")
