# --- Base image
FROM python:3.11-slim

# --- Set working directory
WORKDIR /intrusion_app

# --- Aggiorna pip e installa dipendenze di sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# --- Copia requirements.txt (da creare) e installa python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Copia app e dati
COPY app.py .
COPY data .
# --- Espone porta Streamlit
EXPOSE 8501

# --- Comando per lanciare Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=localhost", "--browser.serverAddress=localhost"]
