import pytest
import torch
import numpy as np
from src.models.transformer import GCNTransformerAutoencoder
from src.features.extraction import extract_features

@pytest.fixture
def model_config():
    return {
        'input_dim': 1,
        'seq_len': 480,
        'n_clusters': 9,
        'd_model': 32, # Small for testing
        'nhead': 4,
        'num_encoder_layers': 1,
        'num_decoder_layers': 1,
        'dim_feedforward': 64,
        'gcn_out_dim': 16
    }

def test_feature_extraction():
    """Test if feature extractor returns correct shape."""
    dummy_window = np.random.rand(480).astype(np.float32)
    feats = extract_features(dummy_window)
    assert feats.shape == (15,), "Feature vector should be size 15"
    assert not np.isnan(feats).any(), "Features should not contain NaNs"

def test_model_forward_pass(model_config):
    """Test if the model accepts input and produces output of same shape."""
    model = GCNTransformerAutoencoder(**model_config)
    
    batch_size = 2
    seq_len = model_config['seq_len']
    
    # Dummy Inputs
    src = torch.randn(batch_size, seq_len, 1)
    state_indices = torch.tensor([0, 1]).long()
    embeddings = torch.randn(model_config['n_clusters'], model_config['gcn_out_dim'])
    
    output = model(src, state_indices, embeddings)
    
    assert output.shape == src.shape, f"Output shape {output.shape} mismatch with input {src.shape}"