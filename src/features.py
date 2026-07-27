# In file: src/features.py

import numpy as np
from mne.time_frequency import psd_array_multitaper
from sklearn.preprocessing import StandardScaler


def get_best_channels():
    """
    Returns the final, definitive list of the most important channels
    that we discovered during our R-squared and Lasso analysis.
    This is now the single source of truth for our channel selection.
    """
    best_channels = [
        'CP6', 'C4', 'F7', 'FT9', 'FT10', 'C3', 'F3', 'F8', 'O1', 'FC2',
        'FC6', 'P4', 'F4', 'FC5', 'P3', 'T8'
    ]
    print(f"\n--- Using {len(best_channels)} pre-selected channels for feature extraction. ---")
    return best_channels

# --- FUNCTION 1: The original EEG-only feature extraction ---
def extract_features(epoched_data):
    """
    Takes the fully preprocessed (and epoched) data and extracts the final
    feature matrix (X) and target matrix (y).

    Features: Alpha, Beta, and Gamma band power, plus RMS and Variance from the pre-selected best channels.
    Target: The average force for each of the 10 finger channels for each epoch.
    """
    print("--- STEP 3: EXTRACTING FEATURES (FREQUENCY & TIME DOMAIN) ---")

    best_channels = get_best_channels()

    X_list = []  # To hold feature sets from each trial
    y_list = []  # To hold target sets from each trial

    for trial_name, data in epoched_data.items():
        eeg_epochs_array = data['eeg_epochs']
        force_epochs = data['force_epochs']
        sfreq = data['sfreq']
        ch_names = data['ch_names']

        # Get the numerical indices of our best channels in the data array
        channel_indices = [ch_names.index(ch) for ch in best_channels if ch in ch_names]

        # Use these indices to select only the data from our best channels
        eeg_epochs_selected_data = eeg_epochs_array[:, channel_indices, :]

        # --- NEW FEATURE 1: TIME-DOMAIN FEATURES (RMS & VARIANCE) ---
        print(f"    Calculating Time-Domain features for trial: {trial_name}...")
        rms_features = np.sqrt(np.mean(eeg_epochs_selected_data**2, axis=2))
        var_features = np.var(eeg_epochs_selected_data, axis=2)

        # --- NEW FEATURE 2: FREQUENCY-DOMAIN FEATURES (Alpha, Beta, AND GAMMA) ---
        # Define frequency bands, including Gamma
        ALPHA_BAND = (8.0, 12.0)
        BETA_BAND = (13.0, 30.0)
        GAMMA_BAND = (30.0, 48.0) # Using up to the 48Hz filter limit

        print(f"    Calculating Frequency-Domain features for trial: {trial_name} using Multitaper...")

        # Calculate PSD across the entire range of interest at once for efficiency
        psds_combined, freqs = psd_array_multitaper(
            eeg_epochs_selected_data,
            sfreq=sfreq,
            fmin=ALPHA_BAND[0],
            fmax=GAMMA_BAND[1],  # Calculate up to the max frequency we need
            verbose=False,
            normalization='length',
            n_jobs=-1 # Use all available cores to speed up
        )

        # Find frequency bins for each band
        alpha_freq_indices = np.where((freqs >= ALPHA_BAND[0]) & (freqs <= ALPHA_BAND[1]))[0]
        beta_freq_indices = np.where((freqs >= BETA_BAND[0]) & (freqs <= BETA_BAND[1]))[0]
        gamma_freq_indices = np.where((freqs >= GAMMA_BAND[0]) & (freqs <= GAMMA_BAND[1]))[0]

        # Extract average power for each band
        alpha_power = np.mean(psds_combined[:, :, alpha_freq_indices], axis=2)
        beta_power = np.mean(psds_combined[:, :, beta_freq_indices], axis=2)
        gamma_power = np.mean(psds_combined[:, :, gamma_freq_indices], axis=2)

        # --- COMBINE ALL FEATURES ---
        # Combine the frequency and time-domain features for this trial
        trial_features = np.concatenate([
            alpha_power,
            beta_power,
            gamma_power,
            rms_features,
            var_features
        ], axis=1) # axis=1 stacks them as columns (new features)
        X_list.append(trial_features)

        # For the target 'y', we want the average force across each epoch for all 10 finger channels
        trial_targets = np.mean(force_epochs, axis=2)
        y_list.append(trial_targets)

    # Combine the data from all trials into two big matrices
    X = np.concatenate(X_list)
    y = np.concatenate(y_list)

    # Finally, scale the features. This is a crucial step for most ML models.
    print("    Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"--- Feature extraction complete. Total features shape: {X_scaled.shape}, Target shape: {y.shape} ---")
    return X_scaled, y, scaler


# --- FUNCTION 2: The new EEG + EMG feature extraction ---
def extract_features_with_emg(epoched_data):
    """
    Takes the epoched data (EEG, EMG, Force) and extracts a combined
    neuro-muscular feature matrix (X) and target matrix (y).
    """
    print("--- STEP 3: EXTRACTING FEATURES (EEG + EMG) ---")

    best_eeg_channels = get_best_channels()

    X_list = []  # To hold the combined feature sets from each trial
    y_list = []  # To hold the target sets from each trial

    for trial_name, data in epoched_data.items():
        # Unpack all data streams for this trial
        eeg_epochs = data['eeg_epochs']
        emg_feature_epochs = data['emg_feature_epochs']
        force_epochs = data['force_epochs']
        sfreq = data['sfreq']
        ch_names = data['ch_names']

        # --- Part 1: EEG Feature Extraction (same as before) ---
        print(f"    Extracting EEG features for trial: {trial_name}...")
        channel_indices = [ch_names.index(ch) for ch in best_eeg_channels if ch in ch_names]
        eeg_epochs_selected = eeg_epochs[:, channel_indices, :]

        eeg_rms = np.sqrt(np.mean(eeg_epochs_selected ** 2, axis=2))
        eeg_var = np.var(eeg_epochs_selected, axis=2)

        ALPHA_BAND = (8.0, 12.0)
        BETA_BAND = (13.0, 30.0)
        GAMMA_BAND = (30.0, 48.0)
        psds, freqs = psd_array_multitaper(eeg_epochs_selected, sfreq, ALPHA_BAND[0], GAMMA_BAND[1], verbose=False,
                                           normalization='length', n_jobs=-1)

        alpha_idx = np.where((freqs >= ALPHA_BAND[0]) & (freqs <= ALPHA_BAND[1]))[0]
        beta_idx = np.where((freqs >= BETA_BAND[0]) & (freqs <= BETA_BAND[1]))[0]
        gamma_idx = np.where((freqs >= GAMMA_BAND[0]) & (freqs <= GAMMA_BAND[1]))[0]

        eeg_alpha_power = np.mean(psds[:, :, alpha_idx], axis=2)
        eeg_beta_power = np.mean(psds[:, :, beta_idx], axis=2)
        eeg_gamma_power = np.mean(psds[:, :, gamma_idx], axis=2)

        eeg_features = np.concatenate([eeg_alpha_power, eeg_beta_power, eeg_gamma_power, eeg_rms, eeg_var], axis=1)

        # --- Part 2: EMG Feature Extraction ---
        print(f"    Extracting EMG features for trial: {trial_name}...")
        # The EMG data is already pre-processed RMS. The best feature to extract
        # for a given epoch is the mean of these RMS values.
        # emg_feature_epochs shape is (n_epochs, n_emg_channels, n_samples_per_epoch)
        emg_features = np.mean(emg_feature_epochs, axis=2)  # Shape becomes (n_epochs, n_emg_channels)

        # --- Part 3: Combine Features and Targets ---
        # Concatenate the EEG and EMG features side-by-side
        combined_trial_features = np.concatenate([eeg_features, emg_features], axis=1)
        X_list.append(combined_trial_features)

        # Target 'y' is the same as before
        trial_targets = np.mean(force_epochs, axis=2)
        y_list.append(trial_targets)

    # Combine data from all trials
    X = np.concatenate(X_list)
    y = np.concatenate(y_list)

    print("    Scaling all features using StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"--- Feature extraction with EMG complete. ---")
    print(f"    Final EEG features shape: {eeg_features.shape}")
    print(f"    Final EMG features shape: {emg_features.shape}")
    print(f"    Combined features shape (X_scaled): {X_scaled.shape}")
    print(f"    Final targets shape (y): {y.shape}")

    return X_scaled, y, scaler