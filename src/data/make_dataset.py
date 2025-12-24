import os
import numpy as np
import random
from sklearn.preprocessing import StandardScaler

def load_and_create_windows(filepath, column_index, window_size, stride, padding_value, label='Healthy'):
    """
    Loads data from a single NPZ file, concatenates, and creates sliding windows.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    all_series_list = []
    
    with np.load(filepath, allow_pickle=True) as data:
        keys = list(data.keys())
        for key in keys:
            array = data[key]
            # Handle different array shapes (1D or 2D)
            if isinstance(array, np.ndarray):
                if array.ndim == 2 and array.shape[1] > column_index:
                    signal = array[:, column_index].astype(np.float32)
                elif array.ndim == 1 and column_index == 0:
                    signal = array.astype(np.float32)
                else:
                    continue
                
                if signal.size > 0:
                    all_series_list.append(signal)

    if not all_series_list:
        raise ValueError(f"No valid signals found in {filepath}")

    # Concatenate all series
    concatenated_series = np.concatenate(all_series_list)
    total_length = len(concatenated_series)

    # Sliding Window
    all_windows_list = []
    start_indices = range(0, total_length, stride)
    
    for start_idx in start_indices:
        end_idx = start_idx + window_size
        window = concatenated_series[start_idx:end_idx]

        if len(window) < window_size:
            # Only pad if it's the very end
            if end_idx >= total_length:
                padding_len = window_size - len(window)
                padded_window = np.pad(window, (0, padding_len), 'constant', constant_values=padding_value)
                all_windows_list.append(padded_window)
        else:
            all_windows_list.append(window)

    # Create Metadata
    all_metadata = []
    for i in range(len(all_windows_list)):
         metadata = {
             'source_file': os.path.basename(filepath),
             'original_window_index': i,
             'label': label,
             'global_index': i # This should be adjusted if merging multiple datasets
         }
         all_metadata.append(metadata)

    return np.stack(all_windows_list).astype(np.float32), all_metadata

def fit_and_scale(windows, train_indices, padding_value=0.0):
    """
    Fits a StandardScaler on training data and scales all windows.
    """
    if not train_indices:
        return windows, None

    train_windows = windows[train_indices]
    
    # Fit only on non-padded values
    non_padded_values = train_windows[train_windows != padding_value].reshape(-1, 1)
    
    scaler = StandardScaler()
    scaler.fit(non_padded_values)

    # Apply transform
    scaled_windows = np.copy(windows)
    
    # Scale window by window to handle padding correctly
    # (Vectorized approach possible but complex with padding)
    for i in range(scaled_windows.shape[0]):
        window = scaled_windows[i]
        mask = (window != padding_value)
        if np.any(mask):
            scaled_vals = scaler.transform(window[mask].reshape(-1, 1))
            window[mask] = scaled_vals.flatten()
            
    # Clean up any NaNs that might have slipped in
    scaled_windows = np.nan_to_num(scaled_windows, nan=padding_value)
    
    return scaled_windows, scaler