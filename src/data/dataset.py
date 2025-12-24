import torch
from torch.utils.data import Dataset
import numpy as np

class ApplianceWindowDataset(Dataset):
    """PyTorch Dataset for appliance energy signal windows."""
    def __init__(self, signals, metadata, indices, label_map):
        """
        Args:
            signals (np.ndarray): Padded and potentially scaled windows [N_total, seq_len].
            metadata (list): List of metadata dictionaries for all windows.
            indices (list): List of global window indices belonging to this split.
            label_map (dict): Dictionary mapping string labels to numerical labels.
        """
        self.signals = signals[indices] # Select only windows for this split
        self.metadata_split = [metadata[i] for i in indices]
        self.indices = indices # Store the original global window indices
        self.label_map = label_map

        # Add channel dimension: [num_samples_split, seq_len, 1]
        if self.signals.ndim == 2:
            self.signals = np.expand_dims(self.signals, axis=-1)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Returns:
            tuple: (signal_tensor, label_tensor, global_index)
        """
        signal = self.signals[idx] # Shape [seq_len, 1]
        meta = self.metadata_split[idx]
        original_label = meta.get('label', 'Unknown')
        global_index = meta.get('global_index', -1)

        # Map original label to numerical label
        numerical_label = self.label_map.get(original_label, -1)

        # Convert to PyTorch tensors
        signal_tensor = torch.from_numpy(signal).float()
        label_tensor = torch.tensor(numerical_label, dtype=torch.long)

        return signal_tensor, label_tensor, global_index