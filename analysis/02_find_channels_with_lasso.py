# In file: analysis/02_find_channels_with_lasso.py

# --- THE FIX IS HERE: All imports are now at the top of the file ---
import pickle
import os
import sys
import numpy as np
import mne

# Set the backend before importing pyplot
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# We need to tell Python where to find our source code
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def find_best_channels_with_lasso():
    """
    This script uses Lasso regression to identify the most important EEG channels
    for predicting finger force.
    """

    # --- STEP 1: LOADING CLEAN, PREPROCESSED DATA ---
    clean_data_path = os.path.join(
        os.path.dirname(__file__),
        '..',
        'data',
        'preprocessed_epoched_data.pkl'
    )

    print(f"--- STEP 1: LOADING CLEAN DATA from {clean_data_path} ---")
    try:
        with open(clean_data_path, 'rb') as f:
            epoched_data = pickle.load(f)
    except FileNotFoundError:
        print("\n!!! ERROR: Clean data file not found. !!!")
        print("Please run 'analysis/00_preprocess_and_save.py' first.")
        return

    print(f"Successfully loaded clean data for {len(epoched_data)} trials.")

    # --- STEP 2: FEATURE EXTRACTION (FOR ALL CHANNELS) ---
    print("\n--- STEP 2: EXTRACTING BETA POWER FEATURES ---")
    all_beta_power = []
    all_avg_force = []

    for trial_name, data in epoched_data.items():
        eeg_epochs = data['eeg_epochs']
        force_epochs = data['force_epochs']
        sfreq = data['sfreq']

        BETA_FREQ_RANGE = (13.0, 30.0)
        # Now this line will work because 'mne' has been imported
        psd, freqs = mne.time_frequency.psd_array_multitaper(
            eeg_epochs, sfreq=sfreq, fmin=BETA_FREQ_RANGE[0], fmax=BETA_FREQ_RANGE[1], verbose=False
        )
        all_beta_power.append(np.mean(psd, axis=2))
        all_avg_force.append(np.mean(force_epochs, axis=(1, 2)))

    X_features = np.concatenate(all_beta_power)
    y_target = np.concatenate(all_avg_force)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)

    # --- STEP 3: TRAINING THE LASSO MODEL ---
    print("\n--- STEP 3: TRAINING LASSO REGRESSION MODEL ---")
    lasso = Lasso(alpha=0.001)
    lasso.fit(X_scaled, y_target)

    # --- STEP 4: IDENTIFYING THE SELECTED CHANNELS ---
    print("\n--- STEP 4: IDENTIFYING IMPORTANT CHANNELS ---")

    first_trial_name = list(epoched_data.keys())[0]
    ch_names = epoched_data[first_trial_name]['ch_names']

    model_coefficients = lasso.coef_

    important_channels = []
    for i, coef in enumerate(model_coefficients):
        if abs(coef) > 1e-6:
            important_channels.append((ch_names[i], coef))

    important_channels.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"\nLasso selected {len(important_channels)} channels as important.")
    print("\n--- MOST IMPORTANT CHANNELS (AND THEIR WEIGHTS) ---")
    for ch, coef in important_channels:
        print(f"  {ch.ljust(5)}: Weight = {coef:.4f}")

    # --- BONUS: Visualize the coefficient weights on a topomap ---
    sfreq = epoched_data[first_trial_name]['sfreq']
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)

    fig, ax = plt.subplots(figsize=(8, 8))
    im, cn = mne.viz.plot_topomap(model_coefficients, info, axes=ax, show=True, cmap='RdBu_r')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Lasso Coefficient Weight')
    ax.set_title('Channel Importance According to Lasso', fontsize=16)
    plt.show()


if __name__ == '__main__':
    find_best_channels_with_lasso()