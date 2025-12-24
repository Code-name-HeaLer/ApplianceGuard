import sys
import os
import yaml
import pickle
import numpy as np
import torch
from pathlib import Path

# Ensure we can import from src
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.make_dataset import load_and_create_windows
from src.graph.builder import build_graph_pipeline

def main():
    # 1. Load Config
    print("Loading configuration...")
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 2. Load Healthy Data
    raw_path = Path(config['data']['raw_path'])
    healthy_file = raw_path / config['data']['files']['healthy']
    
    print(f"Loading healthy data from: {healthy_file}")
    signals, metadata = load_and_create_windows(
        filepath=str(healthy_file),
        column_index=config['data']['signal_column_index'],
        window_size=config['data']['window_size'],
        stride=config['data']['stride'],
        padding_value=config['data']['padding_value']
    )
    print(f"Generated {len(signals)} windows.")

    # 3. Create Train/Val Split Indices
    # We need strictly separated indices to ensure the graph is built ONLY on training data
    n_total = len(signals)
    indices = list(range(n_total))
    np.random.seed(config['project']['seed'])
    np.random.shuffle(indices)

    n_train = config['data']['split_counts']['train']
    # Ensure we don't exceed data length
    if n_train > n_total:
        print(f"Warning: Requested {n_train} training samples, but only {n_total} available.")
        n_train = int(n_total * 0.8)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    print(f"Split: {len(train_indices)} Train, {len(val_indices)} Val")

    # 4. Build Graph Pipeline
    # This runs Feature Extraction -> Scaling -> Clustering -> Transition Matrix
    print("Building Graph Structure...")
    graph_artifacts = build_graph_pipeline(
        all_signals=signals,
        train_indices=train_indices,
        n_clusters=config['graph']['n_clusters'],
        seed=config['project']['seed']
    )

    # Add data to artifacts so training script doesn't have to reload everything
    graph_artifacts['all_signals'] = signals
    graph_artifacts['all_metadata'] = metadata
    graph_artifacts['train_indices'] = train_indices
    graph_artifacts['val_indices'] = val_indices

    # 5. Save Artifacts
    save_dir = Path("models/artifacts")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "graph_structure.pkl"
    
    print(f"Saving graph artifacts to {save_path}...")
    with open(save_path, "wb") as f:
        pickle.dump(graph_artifacts, f)
    
    print("✅ Graph Build Complete.")

if __name__ == "__main__":
    main()