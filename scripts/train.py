import sys
import yaml
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.utils import from_networkx
from pathlib import Path

# Ensure imports work
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.gcn import GCNEncoder
from src.models.transformer import GCNTransformerAutoencoder
from src.data.dataset import ApplianceWindowDataset
from src.data.make_dataset import fit_and_scale
from src.training.trainer import Trainer

def main():
    # 1. Load Config & Artifacts
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    artifact_path = "models/artifacts/graph_structure.pkl"
    if not Path(artifact_path).exists():
        print("❌ Error: Graph artifacts not found. Run 'python scripts/build_graph.py' first.")
        return

    print(f"Loading artifacts from {artifact_path}...")
    with open(artifact_path, "rb") as f:
        artifacts = pickle.load(f)

    device = torch.device(config['project']['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Prepare Data (Scaling)
    # We fit the scaler on the signals identified as 'train' in the build step
    print("Scaling signal data...")
    signals = artifacts['all_signals']
    train_indices = artifacts['train_indices']
    val_indices = artifacts['val_indices']
    
    scaled_signals, signal_scaler = fit_and_scale(
        signals, 
        train_indices, 
        padding_value=config['data']['padding_value']
    )
    
    # Save the signal scaler for inference later
    artifacts['signal_scaler'] = signal_scaler
    with open(artifact_path, "wb") as f:
        pickle.dump(artifacts, f)

    # 3. Create Datasets & Loaders
    # We need a label map (though for training everything is Healthy=0)
    label_map = {'Healthy': 0}
    
    train_dataset = ApplianceWindowDataset(scaled_signals, artifacts['all_metadata'], train_indices, label_map)
    val_dataset = ApplianceWindowDataset(scaled_signals, artifacts['all_metadata'], val_indices, label_map)

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    # 4. GCN Step (Offline Encoding)
    print("Running GCN to generate state embeddings...")
    G_nx = artifacts['graph']
    node_features = torch.tensor(artifacts['node_features'], dtype=torch.float).to(device)
    
    # Convert NetworkX to PyTorch Geometric
    pyg_data = from_networkx(G_nx).to(device)
    
    gcn = GCNEncoder(
        node_feature_dim=config['data']['window_size'],
        hidden_dim=config['model']['gcn_out_dim'],
        output_dim=config['model']['gcn_out_dim']
    ).to(device)
    
    # Run one pass to get H (static embeddings)
    gcn.eval()
    with torch.no_grad():
        static_embeddings = gcn(node_features, pyg_data.edge_index, pyg_data.weight)
    
    print(f"Generated static embeddings shape: {static_embeddings.shape}")

    # 5. Initialize Transformer Autoencoder
    print("Initializing Transformer Autoencoder...")
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
        dropout=config['model']['dropout'],
        activation=config['model']['activation']
    ).to(device)

    # 6. Setup Training
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.L1Loss() # MAE Loss
    
    trainer = Trainer(model, optimizer, criterion, device, config)

    # 7. Start Training
    # We pass the pre-calculated cluster labels (artifacts['all_labels']) to the trainer
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        all_state_embeddings=static_embeddings,
        all_cluster_labels=artifacts['all_labels']
    )
    
    # Save the static embeddings with the model artifact for inference
    # (We save them separately or re-inject into artifacts pickle)
    emb_path = Path(config['training']['save_dir']) / "gcn_embeddings.pt"
    torch.save(static_embeddings, emb_path)
    print(f"Saved static GCN embeddings to {emb_path}")

if __name__ == "__main__":
    main()