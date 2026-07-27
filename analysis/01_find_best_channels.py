# In file: analysis/01_find_best_channels.py

# --- THE FIX IS HERE ---
# These two lines must be the VERY FIRST thing that happens.
import matplotlib

matplotlib.use('Qt5Agg')  # Force matplotlib to use a more stable backend

import numpy as np
import mne
import matplotlib.pyplot as plt

# We need to tell Python where to find our source code
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now we can import our functions
from src.data_loader import load_and_align_data
from src.preprocessing import preprocess_data


def find_best_channels_and_visualize():
    """
    This script runs the full preprocessing pipeline and then performs an
    analysis to find and visualize the EEG channels that are most correlated
    with the force data.
    """
    print("--- STEP 1: LOADING AND ALIGNING DATA ---")
    aligned_data = load_and_align_data()

    print("\n--- STEP 2: PREPROCESSING DATA ---")
    epoched_data = preprocess_data(aligned_data)

    print("\n--- STEP 3: FEATURE EXTRACTION (FOR ALL CHANNELS) ---")
    all_beta_power = []
    all_avg_force = []

    for trial_name, data in epoched_data.items():
        eeg_epochs = data['eeg_epochs']
        force_epochs = data['force_epochs']
        sfreq = data['sfreq']

        BETA_FREQ_RANGE = (13.0, 30.0)
        psd, freqs = mne.time_frequency.psd_array_multitaper(
            eeg_epochs, sfreq=sfreq, fmin=BETA_FREQ_RANGE[0], fmax=BETA_FREQ_RANGE[1], verbose=False
        )

        all_beta_power.append(np.mean(psd, axis=2))
        all_avg_force.append(np.mean(force_epochs, axis=(1, 2)))

    X_features = np.concatenate(all_beta_power)
    y_target = np.concatenate(all_avg_force)

    print(f"Created feature matrix X with shape: {X_features.shape}")
    print(f"Created target vector y with shape: {y_target.shape}")

    print("\n--- STEP 4: CALCULATING R-SQUARED FOR EACH CHANNEL ---")
    r_squared_values = []

    for i in range(X_features.shape[1]):
        channel_feature = X_features[:, i]
        correlation_matrix = np.corrcoef(channel_feature, y_target)
        correlation = correlation_matrix[0, 1]
        r_squared_values.append(np.nan_to_num(correlation ** 2))

    print(f"Calculated {len(r_squared_values)} R-squared values.")

    print("\n--- STEP 5: VISUALIZING THE RESULTS ---")
    first_trial_name = list(epoched_data.keys())[0]
    ch_names = epoched_data[first_trial_name]['ch_names']
    sfreq = epoched_data[first_trial_name]['sfreq']
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')

    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)

    fig, ax = plt.subplots(figsize=(8, 8))
    im, cn = mne.viz.plot_topomap(r_squared_values, info, axes=ax, show=True, cmap='viridis')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('R-squared (predictive power)')
    ax.set_title('Channel Importance: Beta Power vs. Average Finger Force', fontsize=16)

    plt.show()

    print("\n--- TOP 10 MOST IMPORTANT CHANNELS ---")
    channel_scores = list(zip(ch_names, r_squared_values))
    channel_scores.sort(key=lambda x: x[1], reverse=True)

    for ch, score in channel_scores[:10]:
        print(f"  {ch.ljust(5)}: R-squared = {score:.4f}")


if __name__ == '__main__':
    find_best_channels_and_visualize()