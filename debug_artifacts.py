import pickle
import os
import torch
from pathlib import Path

def inspect():
    print("🔍 Inspecting Artifacts...\n")
    
    # 1. Check File Locations
    pkl_path = Path("models/artifacts/graph_structure.pkl")
    pt_path = Path("models/saved/gcn_embeddings.pt")
    
    if pkl_path.exists():
        print(f"✅ Found Pickle: {pkl_path}")
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            
            print(f"   Keys found inside pickle:")
            found_embeddings = False
            for key, value in data.items():
                info = str(type(value))
                if hasattr(value, 'shape'):
                    info += f" shape={value.shape}"
                print(f"    - '{key}': {info}")
                
                # Check if this might be the embeddings
                if "embedding" in key.lower() or (hasattr(value, 'shape') and value.shape == (9, 64)):
                    print(f"      👀 POSSIBLE MATCH for Embeddings ^^^")
                    found_embeddings = True
            
            if not found_embeddings:
                print("\n   ⚠️ No obvious embedding keys found in pickle.")
                
        except Exception as e:
            print(f"   ❌ Error reading pickle: {e}")
    else:
        print(f"❌ Pickle file missing at: {pkl_path}")

    print("-" * 30)

    # 2. Check PT File
    if pt_path.exists():
        print(f"✅ Found GCN PT file: {pt_path}")
    else:
        print(f"❌ GCN PT file missing at: {pt_path}")

if __name__ == "__main__":
    inspect()