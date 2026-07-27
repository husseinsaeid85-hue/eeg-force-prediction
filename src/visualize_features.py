"""Sanity-check plots for the extracted EEG feature matrix.

Standalone diagnostic script (nothing in the pipeline imports it). It loads the
EEG-only feature cache written by ``main.py --features eeg``, un-scales it with
the saved ``StandardScaler``, and plots:

  1. the distribution of each of the five feature types (alpha, beta, gamma, RMS,
     variance), to confirm scaling behaved sensibly;
  2. a short time-series of a few C3 features against the D1 flexion force, to
     eyeball whether the features track the target at all.

Run from the repository root (the cache path below is relative to the cwd)::

    python src/visualize_features.py

This was previously named ``de.py``; the name carried no meaning.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# This script needs to import get_best_channels to know the feature names
# Add the 'src' directory (this file's own directory) to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from features import get_best_channels


def plot_feature_distributions(X_scaled, feature_names):
    """
    Plots histograms for each of the 5 different feature types.
    """
    print("\n--- Plotting feature distributions... ---")

    num_features = X_scaled.shape[1]
    if num_features != 80:
        print(f"Warning: Expected 80 features, but found {num_features}. Plots may be incorrect.")
        return

    # We know the features were concatenated in this order: Alpha, Beta, Gamma, RMS, VAR
    # Each block has 16 channels.
    feature_blocks = {
        "Alpha Power": X_scaled[:, 0:16].flatten(),
        "Beta Power": X_scaled[:, 16:32].flatten(),
        "Gamma Power": X_scaled[:, 32:48].flatten(),
        "RMS": X_scaled[:, 48:64].flatten(),
        "Variance": X_scaled[:, 64:80].flatten()
    }

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle("Distributions of Scaled Feature Types", fontsize=16)

    for i, (name, data) in enumerate(feature_blocks.items()):
        axes[i].hist(data, bins=50, density=True)
        axes[i].set_title(name)
        axes[i].set_xlabel("Standardized Value")
    axes[0].set_ylabel("Density")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_feature_timeseries(X_unscaled, y_unscaled, feature_names):
    """
    Plots a few sample features and a force signal over time for one trial.
    NOTE: This requires loading un-sequenced data to be meaningful.
    This example will just plot the first 500 epochs as a proxy for time.
    """
    print("\n--- Plotting feature time-series for first 500 epochs... ---")

    # Let's pick a few interesting features to inspect
    # e.g., Alpha and Gamma from C3 (a motor channel) and a force signal
    try:
        alpha_c3_index = feature_names.index("Alpha_C3")
        gamma_c3_index = feature_names.index("Gamma_C3")
        rms_c3_index = feature_names.index("RMS_C3")
    except ValueError:
        print("Could not find C3 in feature names. Skipping time-series plot.")
        return

    # We'll plot against the first force signal (D1_Flex)
    force_signal_to_plot = y_unscaled[:, 0]

    # Get the data for the first 500 epochs
    num_samples = 500

    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    fig.suptitle("Sample Feature and Force Time-Series (First 500 epochs)", fontsize=16)

    # Plot 1: Force
    axes[0].plot(force_signal_to_plot[:num_samples], color='black', label='Force (D1_Flex)')
    axes[0].set_ylabel("Force Value")
    axes[0].legend()
    axes[0].grid(True)

    # Plot 2: Power Features
    axes[1].plot(X_unscaled[:num_samples, alpha_c3_index], color='green', label='Alpha Power (C3)')
    axes[1].plot(X_unscaled[:num_samples, gamma_c3_index], color='purple', label='Gamma Power (C3)')
    axes[1].set_ylabel("Power Value")
    axes[1].legend()
    axes[1].grid(True)

    # Plot 3: Time-Domain Features
    axes[2].plot(X_unscaled[:num_samples, rms_c3_index], color='red', label='RMS (C3)')
    axes[2].set_ylabel("Amplitude Value")
    axes[2].legend()
    axes[2].grid(True)

    axes[2].set_xlabel("Epoch Index")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    FEATURES_CACHE_PATH = 'extracted_features_cache.pkl'

    if not os.path.exists(FEATURES_CACHE_PATH):
        sys.exit(
            f"ERROR: Feature cache not found at '{FEATURES_CACHE_PATH}'.\nPlease run main.py to generate it first.")

    print(f"--- Loading features from cache: {FEATURES_CACHE_PATH} ---")
    with open(FEATURES_CACHE_PATH, 'rb') as f:
        # We need to load the unscaled data to plot the time-series meaningfully
        # This assumes your cache saves (X_scaled, y, scaler). We need to un-scale X.
        X_scaled, y, scaler = pickle.load(f)

    # Un-scale the features to see their original values
    X_unscaled = scaler.inverse_transform(X_scaled)

    # Generate the full list of our 80 feature names
    best_channels = get_best_channels()
    full_feature_names = []
    for band in ["Alpha", "Beta", "Gamma"]:
        full_feature_names.extend([f"{band}_{ch}" for ch in best_channels])
    for time_feat in ["RMS", "VAR"]:
        full_feature_names.extend([f"{time_feat}_{ch}" for ch in best_channels])

    # --- Run the visualizations ---
    plot_feature_distributions(X_scaled, full_feature_names)
    plot_feature_timeseries(X_unscaled, y, full_feature_names)
