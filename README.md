# 🔍 Network Threat Intelligence & Anomaly Detection Dashboard  
### Machine Learning & Explainable AI for Zeek Network Logs

Questo progetto propone una pipeline completa per l’analisi di traffico di rete e il rilevamento di attacchi tramite tecniche di **Machine Learning** e **Explainable AI (XAI)**, partendo dai log `conn.log` di **Zeek (ex Bro)**.

L'applicazione fornisce un ambiente per:

- ✅ Pre–processing e feature engineering da log Zeek
- ✅ Modelli ML per rilevamento anomalie e tecniche MITRE
- ✅ Modellazione latente, clustering e riduzione dimensionale
- ✅ Dashboard interattiva (Streamlit)
- ✅ Heatmap, grafici e spiegazioni evento–per–evento
- ✅ Confronto anomalie vs baseline normale (deviazioni latenti)
- ✅ Generazione report investigativi

---

## 🎯 Obiettivo del progetto

I moderni sistemi di rilevamento e risposta richiedono capacità automatiche per riconoscere tattiche e tecniche d'attacco da grandi volumi di log di rete.

Zeek genera log `conn` con informazioni sulle sessioni TCP/UDP/ICMP che rappresentano una fonte preziosa per individuare comportamenti malevoli.

> **Obiettivo:** trasformare questi log in rappresentazioni utili per modelli ML, analizzare connessioni anomale e fornire strumenti interattivi per investigazione e threat hunting.

---

## 🚀 Funzionalità principali

| Categoria | Funzionalità |
|---|---|
📥 **Ingest & Preprocessing** | Parsing Zeek, normalizzazione feature, flag anomalie |
🧠 **Machine Learning** | Modelli supervisionati & semi-supervisionati (AE, clustering) |
🎛 **Dashboard Streamlit** | Filtri, tabelle, visualizzazioni interattive |
🔥 **Explainable AI** | Feature latenti, confronto anomalie vs baseline |
📈 **Visual Analytics** | PCA / UMAP, heatmap, scatter 2D |
📑 **Reportistica** | Esportazione e analisi degli eventi |

---

## 🧠 Modelli & Approccio

- Estrazione feature da `conn.log`
- Encoding numerico/categorico
- Training su traffico normale + anomalie annotate
- Riduzione dimensionale (Autoencoder / PCA / UMAP)
- Analisi latente per capire *perché* un evento è anomalo
- Dashboard XAI per investigazione manuale

---

## 🖥️ Dashboard — Moduli Principali

- Media delle feature latenti (anomalia vs normale)
- Heatmap latenti
- Metriche del modello
- Cluster 2D UMAP/PCA
- Drill-down anomalie con spiegazione

---

## 🗂 Struttura del progetto
├── intrusion_app/
│ ├── app.py # Streamlit dashboard
│
├── Dataset-Preparation.ipynb
├── Model_SemiSupervised.ipynb
├── Model-Training.ipynb
├── Model-Training-Imbalanced.ipynb
│
├── processed_zeekdata22/ # File temporanei e dataset processati
├── UWF-ZeekDataFall22/ # Log Zeek originali
└── README.md

---

## ▶️ Avvio della dashboard

```bash
streamlit run intrusion_app/app.py

📂 Dataset

Dataset contenente sessioni Zeek annotate (traffico normale + malevolo), incluse tattiche reali.

📎 Fonte dataset:
https://datasets.uwf.edu/

🔗 Codice e risorse

📁 Repository GitHub:
https://github.com/MariaCascone66/Zeek_ML.git

📊 Tutti i notebook e grafici sono presenti nel repository.
(Alcuni grafici non sono inclusi nella relazione per evitare ridondanza)

📜 Riferimenti

Zeek Network Security Monitor — https://zeek.org

Linee guida Explainable AI per cybersecurity

Tecniche MITRE ATT&CK per classificazione tattiche