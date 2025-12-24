import sys
import yaml
import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.transformer import GCNTransformerAutoencoder
from src.data.make_dataset import load_and_create_windows
from src.data.dataset import ApplianceWindowDataset
from src.features.extraction import extract_features
from src.visualization.plots import plot_error_distribution, plot_confusion_matrix, plot_reconstruction

def main():
    # 1. Load Config
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config['project']['device'] if torch.cuda.is_available() else "cpu")
    
    # 2. Load Artifacts (Scalers, Centroids, Embeddings)
    print("Loading artifacts...")
    with open("models/artifacts/graph_structure.pkl", "rb") as f:
        artifacts = pickle.load(f)
        
    signal_scaler = artifacts['signal_scaler']
    feature_scaler = artifacts['feature_scaler']
    feature_centroids = artifacts['feature_centroids']
    
    # Load GCN Embeddings from artifacts instead of separate file
    if 'all_state_embeddings' in artifacts:
        static_embeddings = torch.tensor(artifacts['all_state_embeddings']).float().to(device)
    else:
        # Fallback to loading from file if not in pickle
        static_embeddings = torch.load("models/saved/gcn_embeddings.pt", map_location=device)

    # 3. Load Trained Model
    print("Loading model...")
    model_path = "models/saved/best_model.pt"
    checkpoint = torch.load(model_path, map_location=device)
    
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
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 4. Load & Process Inference Data
    # We load ALL files defined in config (Healthy + Unhealthy)
    raw_path = Path(config['data']['raw_path'])
    all_signals_list = []
    all_meta_list = []
    
    # Map for easy processing
    file_map = {
        'Healthy': config['data']['files']['healthy'],
        'UH_HighEnergy': config['data']['files']['unhealthy']['high_energy'],
        'UH_LowEnergy': config['data']['files']['unhealthy']['low_energy'],
        'UH_Noisy': config['data']['files']['unhealthy']['noisy'],
    }
    
    print("Processing inference data...")
    for label_name, filename in file_map.items():
        filepath = raw_path / filename
        if not filepath.exists():
            print(f"Skipping {filename} (not found)")
            continue
            
        sigs, metas = load_and_create_windows(
            str(filepath), 
            config['data']['signal_column_index'], 
            config['data']['window_size'], 
            config['data']['stride'], 
            config['data']['padding_value'],
            label=label_name
        )
        all_signals_list.append(sigs)
        all_meta_list.extend(metas)
        
    inf_signals = np.concatenate(all_signals_list)
    
    # 5. Pipeline: Scale Signal -> Extract Features -> Scale Features -> Assign Cluster
    # A. Scale Signals
    # Note: We must replicate 'fit_and_scale' logic but using transform only
    inf_signals_scaled = np.copy(inf_signals)
    # Simplified application of scaler (handling padding row-by-row)
    for i in range(len(inf_signals_scaled)):
        mask = inf_signals_scaled[i] != config['data']['padding_value']
        if np.any(mask):
            vals = inf_signals_scaled[i][mask].reshape(-1, 1)
            inf_signals_scaled[i][mask] = signal_scaler.transform(vals).flatten()
            
    # B. Extract & Scale Features
    print("Extracting features for cluster assignment...")
    features = np.array([extract_features(w) for w in inf_signals])
    features = np.nan_to_num(features)
    features_scaled = feature_scaler.transform(features)
    
    # C. Assign Cluster (Nearest Centroid)
    print("Assigning clusters...")
    from sklearn.metrics import pairwise_distances_argmin_min
    cluster_labels, _ = pairwise_distances_argmin_min(features_scaled, feature_centroids)
    
    # 6. Run Inference
    print("Running model inference...")
    dataset = ApplianceWindowDataset(inf_signals_scaled, all_meta_list, list(range(len(inf_signals))), {})
    loader = DataLoader(dataset, batch_size=config['inference']['batch_size'], shuffle=False)
    
    criterion = nn.L1Loss(reduction='none')
    errors = []
    
    with torch.no_grad():
        for signals, _, indices in loader:
            signals = signals.to(device)
            batch_labels = cluster_labels[indices]
            state_indices = torch.tensor(batch_labels).long().to(device)
            
            output = model(signals, state_indices, static_embeddings)
            
            # Calculate MAE per window
            batch_loss = torch.mean(torch.abs(output - signals), dim=(1, 2))
            errors.extend(batch_loss.cpu().numpy())
            
    errors = np.array(errors)
    
    # 7. Thresholding & Evaluation
    # Identify healthy indices from metadata to calculate threshold
    healthy_indices = [i for i, m in enumerate(all_meta_list) if m['label'] == 'Healthy']
    healthy_errors = errors[healthy_indices]
    
    threshold = np.quantile(healthy_errors, config['inference']['quantile'])
    print(f"Calculated Threshold (Quantile {config['inference']['quantile']}): {threshold:.6f}")
    
    # Predictions
    predictions = (errors > threshold).astype(int)
    
    # Ground Truth (Healthy=0, Everything else=1)
    true_labels = np.array([0 if m['label'] == 'Healthy' else 1 for m in all_meta_list])
    
    # Metrics
    acc = accuracy_score(true_labels, predictions)
    prec, rec, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
    cm = confusion_matrix(true_labels, predictions)
    
    print("\n--- Results ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    # 8. Visualization
    save_dir = Path(config['inference']['results_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    plot_error_distribution(errors, true_labels, threshold, save_path=save_dir/"error_dist.png")
    plot_confusion_matrix(cm, ['Normal', 'Anomaly'], save_path=save_dir/"confusion_matrix.png")
    
    print(f"Plots saved to {save_dir}")

if __name__ == "__main__":
    main()