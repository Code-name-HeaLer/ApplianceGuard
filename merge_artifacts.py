import pickle
import sys
from pathlib import Path

def merge():
    # Paths
    old_path = Path("old_artifacts.pkl")
    new_path = Path("models/artifacts/graph_structure.pkl")

    if not old_path.exists():
        print(f"❌ Could not find 'old_artifacts.pkl' in the root folder.")
        print("Please copy your ORIGINAL 'graph_structure_final.pkl' here and rename it.")
        return

    print("Loading artifacts...")
    with open(old_path, "rb") as f:
        old_data = pickle.load(f)
    
    with open(new_path, "rb") as f:
        new_data = pickle.load(f)

    # Check for embeddings in old data
    # The key might be 'all_state_embeddings' or 'state_embeddings' depending on your old code
    emb_key = None
    if 'all_state_embeddings' in old_data:
        emb_key = 'all_state_embeddings'
    elif 'state_embeddings' in old_data:
        emb_key = 'state_embeddings'
    
    if emb_key:
        print(f"✅ Found embeddings in old file.")
        new_data['all_state_embeddings'] = old_data[emb_key]
        
        # Also copy centroids/scalers to ensure the model matches the data perfectly
        if 'feature_centroids' in old_data:
            new_data['feature_centroids'] = old_data['feature_centroids']
        if 'feature_scaler' in old_data:
            new_data['feature_scaler'] = old_data['feature_scaler']
        if 'signal_scaler' in old_data:
            new_data['signal_scaler'] = old_data['signal_scaler']

        # Save back to new path
        with open(new_path, "wb") as f:
            pickle.dump(new_data, f)
        
        print(f"🎉 Success! Embeddings merged into {new_path}")
        print("You can now run the dashboard.")
    else:
        print("❌ Could not find 'all_state_embeddings' in your old artifact file.")

if __name__ == "__main__":
    merge()