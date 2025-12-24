import numpy as np
from scipy.stats import skew, kurtosis

def extract_features(window: np.ndarray, threshold: float = 10.0) -> np.ndarray:
    """
    Extracts a 15-dimensional feature vector from a single time-series window.

    Args:
        window (np.ndarray): 1D array representing the time-series window.
        threshold (float): Threshold to determine 'active' points.

    Returns:
        np.ndarray: 15-dimensional feature vector. Returns None if input is invalid.
    """
    if not isinstance(window, np.ndarray) or window.ndim != 1 or window.size == 0:
        # Return a zero-vector or handle gracefully if strict validation is needed
        return None 

    features = []
    epsilon = 1e-9 # To avoid division by zero or issues with zero std dev

    # Basic Stats
    mean_val = np.mean(window)
    std_val = np.std(window)
    features.append(mean_val)
    features.append(std_val)
    features.append(np.max(window))
    features.append(np.min(window))
    features.append(np.sqrt(np.mean(window**2))) # RMS

    # Shape Stats (handle low std dev)
    features.append(skew(window) if std_val > epsilon else 0)
    features.append(kurtosis(window) if std_val > epsilon else 0)

    # Active Power Stats
    active_indices = np.where(window > threshold)[0]
    n_active = len(active_indices)
    features.append(n_active)

    if n_active > 0:
        active_signal = window[active_indices]
        features.append(np.mean(active_signal))
        features.append(np.sum(active_signal)) # Energy approximation
        features.append(np.max(active_signal))
        # Peak count in active signal (simple difference method)
        if len(active_signal) > 2:
            diffs = np.diff(active_signal)
            # Count points where slope changes from positive to negative
            peaks = np.sum((diffs[:-1] > 0) & (diffs[1:] < 0))
            features.append(peaks)
        else:
            features.append(0) # Not enough points for peaks
    else:
        # Append zeros if no active power
        features.extend([0, 0, 0, 0])

    # FFT Features (handle potential errors)
    try:
        n_fft = len(window)
        if n_fft > 1: # Need at least 2 points for FFT
             fft_vals = np.fft.rfft(window) # Real FFT for real signal
             fft_mag_sq = np.abs(fft_vals)**2 # Power spectrum
             n_rfft = len(fft_mag_sq)

             # Define frequency bins (simple split into 3)
             bin1_end = max(1, n_rfft // 3) # Ensure at least one element
             bin2_end = max(bin1_end + 1, 2 * n_rfft // 3) # Ensure progression

             energy_bin1 = np.sum(fft_mag_sq[0:bin1_end])
             energy_bin2 = np.sum(fft_mag_sq[bin1_end:bin2_end])
             energy_bin3 = np.sum(fft_mag_sq[bin2_end:])
             total_energy = energy_bin1 + energy_bin2 + energy_bin3 + epsilon # Avoid division by zero

             features.extend([energy_bin1 / total_energy,
                              energy_bin2 / total_energy,
                              energy_bin3 / total_energy])
        else:
             features.extend([0, 0, 0]) # Not enough points for FFT
    except Exception as fft_e:
        print(f"Warning: FFT error - {fft_e}. Appending zeros for FFT features.")
        features.extend([0, 0, 0])

    # Ensure features are finite and numeric, handle NaNs/Infs
    final_features = np.nan_to_num(np.array(features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    # Double check after nan_to_num
    if np.any(np.isinf(final_features)) or np.any(np.isnan(final_features)):
        final_features = np.nan_to_num(final_features, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure correct dimension (should be 15)
    if len(final_features) != 15:
         if len(final_features) < 15:
              final_features = np.pad(final_features, (0, 15 - len(final_features)), 'constant', constant_values=0.0)
         elif len(final_features) > 15:
              final_features = final_features[:15]

    return final_features