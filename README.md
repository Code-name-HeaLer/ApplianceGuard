# ApplianceGuard: Intelligent Appliance Monitoring System

**ApplianceGuard** is a robust MLOps pipeline designed to detect anomalies in household appliances using a hybrid **Graph Neural Network (GNN)** and **Transformer** architecture.

## 🚀 Key Features

- **Hybrid Architecture:** Combines GCNs for state embedding (spatial/structural) and Transformers for temporal reconstruction.
- **Modular Design:** Professional `src` package structure separating concerns (Data, Features, Modeling).
- **Reproducible:** Config-driven experiments (`Hydra`/`YAML`) and strict dependency management.
- **Interactive Dashboard:** Streamlit-based UI for real-time inference and visualization.

## 📂 Project Structure

```text
ApplianceGuard/
├── configs/        # Hyperparameters and paths
├── dashboard/      # Streamlit Web App
├── scripts/        # Execution scripts (train, inference)
├── src/            # Core source code
│   ├── features/   # Signal processing
│   ├── graph/      # Graph topology construction
│   └── models/     # PyTorch GNN & Transformer models
└── models/saved/   # Trained artifacts
```

## 🛠️ Installation

```bash
# Clone and enter directory
git clone https://github.com/yourusername/appliance-guard.git
cd ApplianceGuard

# Create env and install
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
pip install -e .
```

## ⚡ Usage

### 1. Build the Graph

Processes raw `.npz` data and constructs the state graph.

```bash
python scripts/build_graph.py
```

### 2. Train the Model

Trains the Autoencoder using the static graph embeddings.

```bash
python scripts/train.py
```

### 3. Run Inference

Evaluates the model on Healthy vs Unhealthy datasets.

```bash
python scripts/inference.py
```

### 4. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

## 🧪 Testing

Run the automated unit test suite:

```bash
pytest tests/
```

## 🏗️ Architecture

1.  **Preprocessing:** Sliding window segmentation (480 steps).
2.  **Graph Construction:** KMeans clustering on signal features -> Transition Probability Matrix.
3.  **GNN Encoder:** Generates static embeddings for each operational state.
4.  **Transformer Autoencoder:** Reconstructs signals conditioned on the GNN state embeddings.
