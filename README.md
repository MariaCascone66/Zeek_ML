🔍 Network Threat Intelligence & Anomaly Detection Dashboard
Machine Learning & Explainable AI for Zeek Network Logs

Questo progetto propone una pipeline completa per l’analisi di traffico di rete e il rilevamento di attacchi tramite tecniche di Machine Learning e Explainable AI (XAI), partendo dai log conn.log di Zeek (ex Bro).

L’applicazione include:

✅ Pre–processing e feature engineering da Zeek

✅ Modelli ML per classificazione anomalie / tattiche di attacco

✅ Latent feature modeling e clustering avanzato

✅ Dashboard interattiva (Streamlit) per analisi, explainability e reportistica

✅ Visualizzazioni avanzate: heatmap latenti, clustering figurativo, spiegazioni evento–per–evento

✅ Modulo per confrontare anomalie con la baseline normale (deviazione latente)

✅ Export report e supporto all’analisi investigativa

🎯 Obiettivo del progetto

I moderni sistemi di rilevamento e risposta richiedono capacità automatiche per riconoscere tattiche e tecniche d’attacco da grandi volumi di log di rete.

Zeek genera log conn che includono dettagli sulle sessioni TCP/UDP/ICMP e rappresentano una fonte informativa ricca per identificare comportamenti malevoli.

L’obiettivo è trasformare questi log in rappresentazioni significative per modelli ML, analizzare il comportamento di connessioni anomale e fornire strumenti interattivi e interpretativi per l’investigazione.

🚀 Funzionalità principali
Categoria	Funzionalità
📥 Ingest & Preprocessing	Parsing Zeek logs, normalizzazione feature, definizione flag anomalia
🧠 Machine Learning	Modelli supervisionati & unsupervisionati (AE/latent, clustering)
🎛 Dashboard Streamlit	Navigazione dataset, filtri, visualizzazioni tecniche
🔥 Explainable AI	Analisi dimensioni latenti, heatmap, confronto anomalie vs baseline
📈 Visual Analytics	UMAP/T-SNE plot, heatmap medie latenti, grafici interattivi
📑 Reportistica	Generazione report, analisi esempi, confronti categorie
🧠 Modelli & Metodologia

Estrazione feature da conn.log

Encoding feature numeriche/categoriche

Training modelli ML per anomalie e tecniche sospette

Riduzione dimensionale per interpretazione (UMAP / autoencoder)

Analisi latente e confronto anomalie con baseline normale

Dashboard per investigazione interattiva

🖥️ Screenshot / UI (placeholder)

Dashboard con moduli per:

Media feature latenti (anomalia vs normale)

Heatmap

Metriche modello

Cluster view

Drill-down anomalie

🗂 Struttura progetto
├── intrusion_app/  
|   ├── app.py                   
├── Dataset-Preparation.ipynb
├── Model_SemiSupervised.ipynb
├── Model-Training.ipynb
├── Model-Training-Imbalanced.ipynb
├── .gitignore
├── processed_zeekdata22/              #file intermedi di salvataggio presenti nel codice
├── UWF-ZeekDataFall22/                # Log Zeek e dataset
└── README.md

▶️ Avvio dashboard
streamlit run src/app.py           #src= percorso cartella di appartenenza

📂 Dataset

Dataset contenente sessioni Zeek annotate (normali vs malevole), con esempi di tattiche di attacco.

📎 Pagina ufficiale dataset:
👉 https://datasets.uwf.edu/

🔗 Codice & Risorse

📁 Repository GitHub:
👉 https://github.com/MariaCascone66/Zeek_ML.git

📊 Grafici completi e notebook di analisi sono disponibili nel repo.
(Alcune visualizzazioni non sono incluse nella relazione per evitare ridondanza.)