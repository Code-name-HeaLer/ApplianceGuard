import streamlit as st
import torch
import numpy as np
import yaml
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import from our source package
from src.models.transformer import GCNTransformerAutoencoder
from src.features.extraction import extract_features
from src.visualization.plots import plot_reconstruction
from src.data.make_dataset import load_and_create_windows  # <--- Added this import

# Page Config
st.set_page_config(page_title="ApplianceGuard Dashboard", layout="wide")

@st.cache_resource
def load_resources():
    """Loads config, artifacts, and model (cached for performance)."""
    # 1. Load Config
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cpu") # Use CPU for dashboard usually
    
    # 2. Load Artifacts
    artifact_path = "models/artifacts/graph_structure.pkl"
    with open(artifact_path, "rb") as f:
        artifacts = pickle.load(f)
        
    # 3. Handle Legacy Artifacts (Missing 'all_signals')
    # If the pickle doesn't have the signals, we load them from the raw file
    if 'all_signals' in artifacts:
        all_signals = artifacts['all_signals']
    else:
        # Construct path to healthy data
        raw_path = Path(config['data']['raw_path'])
        healthy_file = raw_path / config['data']['files']['healthy']
        
        if not healthy_file.exists():
            st.error(f"❌ Data file not found: {healthy_file}")
            st.warning("Please ensure your .npz files are in 'data/raw/'")
            st.stop()
            
        # Load the data dynamically
        all_signals, _ = load_and_create_windows(
            filepath=str(healthy_file),
            column_index=config['data']['signal_column_index'],
            window_size=config['data']['window_size'],
            stride=config['data']['stride'],
            padding_value=config['data']['padding_value']
        )
        
    # 4. Load GCN Embeddings
    # Handle both cases (in pickle vs separate file)
    if 'all_state_embeddings' in artifacts:
        embeddings = torch.tensor(artifacts['all_state_embeddings']).float().to(device)
    else:
        embeddings_path = Path("models/saved/gcn_embeddings.pt")
        if embeddings_path.exists():
            embeddings = torch.load(embeddings_path, map_location=device, weights_only=True)
        else:
            st.error("❌ GCN Embeddings not found in artifacts or saved folder.")
            st.stop()

    # 5. Load Model
    model = GCNTransformerAutoencoder(
        input_dim=config['model']['input_dim'],
        seq_len=config['data']['window_size'],
        n_clusters=config['graph']['n_clusters'],
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_encoder_layers=config['model']['num_encoder_layers'],
        num_decoder_layers=config['model']['num_decoder_layers'],
        dim_feedforward=config['model']['dim_feedforward'],
        gcn_out_dim=config['model']['gcn_out_dim'],
        activation=config['model']['activation']
    ).to(device)
    
    checkpoint = torch.load("models/saved/best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return config, artifacts, embeddings, model, all_signals

def main():
    st.title("🛡️ ApplianceGuard: Intelligent Monitoring")
    st.markdown("Real-time anomaly detection using **GNNs + Transformers**.")
    
    try:
        # Unpack the new return value (all_signals)
        config, artifacts, embeddings, model, all_signals = load_resources()
        st.success("System Ready: Model and Artifacts Loaded.")
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        st.stop()

    # Sidebar
    st.sidebar.header("Control Panel")
    data_source = st.sidebar.selectbox("Select Data Source", ["Healthy Sample", "Anomaly Sample (Simulated)", "Upload .npy"])
    
    threshold = st.sidebar.slider("Anomaly Threshold", 0.0, 0.1, 0.04, format="%.4f")
    
    st.subheader("Live Signal Analysis")
    
    # Logic to get a sample window
    window_size = config['data']['window_size']
    input_signal = None
    
    if data_source == "Healthy Sample":
        # USE THE LOADED SIGNALS HERE
        idx = np.random.randint(0, len(all_signals))
        input_signal = all_signals[idx]
        st.info(f"Loaded Healthy Window Index: {idx}")
        
    elif data_source == "Upload .npy":
        uploaded_file = st.sidebar.file_uploader("Upload a signal window (.npy)", type="npy")
        if uploaded_file:
            input_signal = np.load(uploaded_file)
            if len(input_signal) != window_size:
                st.error(f"Input shape {len(input_signal)} does not match model window size {window_size}.")
                input_signal = None
                
    elif data_source == "Anomaly Sample (Simulated)":
        # Simulate an anomaly by adding noise to a healthy signal
        idx = np.random.randint(0, len(all_signals))
        base = all_signals[idx]
        noise = np.random.normal(0, 1.5, size=window_size).astype(np.float32)
        input_signal = base + noise
        st.warning("Generated Simulated Anomaly (Added Noise)")

    # Processing & Inference
    if input_signal is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 1. Scale
            # Note: We need to reshape for the scaler
            scaler = artifacts['signal_scaler']
            sig_reshaped = input_signal.reshape(-1, 1)
            sig_scaled = scaler.transform(sig_reshaped).flatten()
            
            # 2. Extract Features & Assign Cluster
            feats = extract_features(input_signal)
            # Reshape feature for scaler [1, n_features]
            feats_scaled = artifacts['feature_scaler'].transform(feats.reshape(1, -1))
            
            from sklearn.metrics import pairwise_distances_argmin_min
            cluster_label, _ = pairwise_distances_argmin_min(feats_scaled, artifacts['feature_centroids'])
            cluster = cluster_label[0]
            
            # 3. Model Inference
            tensor_sig = torch.tensor(sig_scaled).float().unsqueeze(0).unsqueeze(2) # [1, 480, 1]
            tensor_idx = torch.tensor([cluster]).long()
            
            with torch.no_grad():
                recon = model(tensor_sig, tensor_idx, embeddings)
                recon_np = recon.squeeze().numpy()
                
            # 4. Error Calc
            # Inverse scale for plotting "Real World" values
            recon_real = scaler.inverse_transform(recon_np.reshape(-1, 1)).flatten()
            
            mae = np.mean(np.abs(input_signal - recon_real))
            
            # Visualization
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(input_signal, label='Input', color='blue', alpha=0.7)
            ax.plot(recon_real, label='Reconstruction', color='red', linestyle='--')
            ax.set_title(f"Reconstruction (Cluster {cluster})")
            ax.legend()
            st.pyplot(fig)
            
        with col2:
            st.metric("Reconstruction Error (MAE)", f"{mae:.4f}")
            
            if mae > threshold:
                st.error("🚨 ANOMALY DETECTED")
                st.markdown("**Status:** Critical\n\nSignal deviation exceeds safety threshold.")
            else:
                st.success("✅ SYSTEM NORMAL")
                st.markdown("**Status:** Healthy\n\nSignal matches learned patterns.")
            
            st.write("---")
            st.write(f"**Assigned Node:** {cluster}")
            st.write(f"**Active Features:** {feats[5]:.0f}") # e.g. active count

if __name__ == "__main__":
    main()