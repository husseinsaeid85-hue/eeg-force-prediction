# src/visualize.py

import mne
import numpy as np
import matplotlib.pyplot as plt
import optuna
import torch
from data_sequencer import create_sequences
from scipy.signal import savgol_filter

def plot_eeg_raw(raw_data, sfreq, ch_names, title="Raw EEG Data"):
    """Plots a short segment of raw EEG data."""
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(raw_data, info, verbose=False)
    raw.plot(duration=5, scalings={'eeg': 100e-6}, title=title, show_scrollbars=False, block=True)
    plt.show()


def plot_eeg_epochs_average(epoched_eeg_data, sfreq, ch_names, title="Average EEG Epoch"):
    """
    Plots the average of a specific channel across all epochs.
    """
    if epoched_eeg_data.shape[0] == 0:
        print(f"No epochs to plot for {title}")
        return

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    epochs = mne.EpochsArray(epoched_eeg_data, info, verbose=False)
    evoked = epochs.average()
    evoked.plot(titles=title, picks='eeg', window_title=title, time_unit='s', gfp=True, spatial_colors=True)
    plt.show()


# In src/visualize.py, replace the plot_channel_power_spectrum function

def plot_channel_power_spectrum(eeg_epochs_data, sfreq, ch_name, band_alpha, band_beta, band_gamma, title="Power Spectrum"):
    """
    Plots the power spectrum for a specific channel, highlighting alpha, beta, and gamma bands.
    Uses Multitaper method to be consistent with feature extraction.
    """
    if eeg_epochs_data.shape[0] == 0:
        print(f"No epochs to plot for {title}")
        return

    info = mne.create_info(ch_names=ch_name, sfreq=sfreq, ch_types='eeg')
    epochs_single_ch = mne.EpochsArray(eeg_epochs_data, info, verbose=False)

    fig = epochs_single_ch.plot_psd(
        fmin=1.0, fmax=48.0,
        average=True,
        picks='eeg',
        show=False,
        verbose=False
    )

    # --- CORRECTED LINE BELOW ---
    # fig.axes is a list of axes. We need to select the first one.
    ax = fig.axes[0]
    ax.set_title(f"{title} - Channel: {ch_name[0]}")

    # Highlight all three bands
    ax.axvspan(band_alpha[0], band_alpha[1], color='green', alpha=0.15, label='Alpha')
    ax.axvspan(band_beta[0], band_beta[1], color='red', alpha=0.15, label='Beta')
    ax.axvspan(band_gamma[0], band_gamma[1], color='purple', alpha=0.15, label='Gamma')
    ax.legend()
    plt.tight_layout()
    plt.show()


# In src/visualize.py, replace the plot_topomap_power function

# In src/visualize.py, replace the plot_topomap_power function

# In src/visualize.py, replace the plot_topomap_power function

def plot_topomap_power(epoched_eeg_data, sfreq, ch_names, band_min, band_max, title="Power Topomap"):
    """
    Plots a topomap of average power across the scalp for a given frequency band.
    """
    if epoched_eeg_data.shape[0] == 0:
        print(f"No epochs to plot for {title}")
        return

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')

    try:
        montage = mne.channels.make_standard_montage('standard_1020')
        info.set_montage(montage)
    except Exception as e:
        print(f"WARNING: Could not set standard 10-20 montage for topomap. Error: {e}")
        print("Skipping topomap plot.")
        return

    epochs = mne.EpochsArray(epoched_eeg_data, info, verbose=False)

    # Compute PSD for a broad range that includes our band of interest
    spectrum = epochs.compute_psd(
        method='multitaper',
        fmin=1.0, fmax=48.0,
        picks='eeg', verbose=False
    )
    psds, freqs = spectrum.get_data(return_freqs=True)  # psds shape: (n_epochs, n_channels, n_freq_bins)

    # --- CORRECTED DATA AVERAGING ---
    # Find frequency bins for the specific band
    band_freq_indices = np.where((freqs >= band_min) & (freqs <= band_max))[0]

    # Select the power data for the desired band
    psds_band = psds[:, :, band_freq_indices]

    # Now, average power over both epochs AND frequency bins for each channel
    # This will result in a 1D array of shape (n_channels,), which is what plot_topomap expects.
    avg_power_per_channel = psds_band.mean(axis=(0, 2))  # Average across axis 0 (epochs) and axis 2 (freqs)

    # --- PLOTTING ---
    fig, ax = plt.subplots(figsize=(6, 6))
    im, _ = mne.viz.plot_topomap(
        avg_power_per_channel,
        info,
        axes=ax,
        show=False,
        outlines='head',
        sensors=True,
        cmap='viridis'
    )

    fig.colorbar(im, ax=ax, orientation='vertical', label='Power (dB)')
    fig.suptitle(f"{title} ({band_min}-{band_max} Hz)", fontsize=16)
    plt.show()

def plot_force_timeseries(force_data, sfreq, title="Force Time Series (First 5 seconds)"):
    """Plots the time series of force data for the first few seconds."""
    if force_data.shape[1] == 0:
        print(f"No force data to plot for {title}")
        return

    n_samples_to_plot = min(force_data.shape[1], int(sfreq * 5))

    plt.figure(figsize=(12, 6))
    for i in range(force_data.shape[0]):
        plt.plot(np.arange(n_samples_to_plot) / sfreq, force_data[i, :n_samples_to_plot].T,
                 label=f'Force Signal {i + 1}')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Force Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_predicted_vs_actual(y_actual, y_predicted, title="Predicted vs. Actual Force"):
    """
    Plots predicted versus actual force values.
    """
    if y_actual.shape[0] == 0 or y_predicted.shape[0] == 0:
        print(f"No data to plot for {title}")
        return

    n_signals = y_actual.shape[1]
    fig, axes = plt.subplots(n_signals, 1, figsize=(15, 4 * n_signals), sharex=True)

    if n_signals == 1:
        axes = [axes]

    for i in range(n_signals):
        axes[i].plot(y_actual[:, i], label='Actual', alpha=0.7)
        axes[i].plot(y_predicted[:, i], label='Predicted', alpha=0.7, linestyle='--')
        axes[i].set_ylabel(f'Signal {i + 1} Force')
        axes[i].legend()
        axes[i].grid(True)

    axes[0].set_title(title)
    axes[-1].set_xlabel('Epoch Index')
    plt.tight_layout()
    plt.show()


def plot_xgboost_feature_importance(model, features_list, top_n=20):
    """
    Plots the top N most important features from a trained XGBoost model.

    Args:
        model: A trained XGBoost model instance.
        features_list (list): A list of strings with the names of all features.
        top_n (int): The number of top features to display.
    """
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have 'feature_importances_'. Cannot plot.")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]  # Get indices of top N features

    # Create a mapping from feature names for readability
    feature_names_map = []
    bands = ["Alpha", "Beta", "Gamma", "RMS", "VAR"]
    channels = [ch_name for ch_name in features_list if
                ch_name in ['CP6', 'C4', 'F7', 'FT9', 'FT10', 'C3', 'F3', 'F8', 'O1', 'FC2', 'FC6', 'P4', 'F4', 'FC5',
                            'P3', 'T8']]

    # This is a simplified way to create feature names.
    # A more robust way would be to generate and save the exact names during feature extraction.
    num_channels = 16  # Based on your get_best_channels()
    num_bands = 3
    num_time_features = 2

    # Generate the full list of feature names in the same order as concatenation
    full_feature_names = []
    for band in ["Alpha", "Beta", "Gamma"]:
        full_feature_names.extend([f"{band}_{ch}" for ch in channels])
    for time_feat in ["RMS", "VAR"]:
        full_feature_names.extend([f"{time_feat}_{ch}" for ch in channels])

    plt.figure(figsize=(10, 8))
    plt.title(f'Top {top_n} Feature Importances (XGBoost)')
    plt.barh(range(top_n), importances[indices], align='center')
    plt.yticks(range(top_n), [full_feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.tight_layout()
    plt.show()


def plot_optuna_history(study):
    """
    Plots the optimization history from an Optuna study.

    Args:
        study: A completed Optuna study object.
    """
    if study is None or len(study.trials) == 0:
        print("No Optuna study to plot.")
        return

    # Plot optimization history
    fig = optuna.visualization.plot_optimization_history(study)
    fig.update_layout(title="Optuna Optimization History")
    fig.show()

    # Plot parameter importance
    try:
        fig = optuna.visualization.plot_param_importances(study)
        fig.update_layout(title="Optuna Hyperparameter Importances")
        fig.show()
    except Exception as e:
        print(f"Could not plot parameter importances: {e}")



def plot_loss_curves(history, title="Training and Validation Loss"):
    """
    Plots the training and validation loss curves from the model's history.
    """
    if not history or 'train_loss' not in history or 'val_loss' not in history:
        print("Invalid history object for plotting loss. Skipping plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss', color='blue')
    plt.plot(history['val_loss'], label='Validation Loss', color='orange')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_continuous_prediction_for_trial(model, single_trial_data, scaler, sequence_length=5):
    """
    Takes a trained model and data for a single trial and plots the
    continuous predicted vs. actual force. This is the "unshuffled" plot.
    """
    print(f"\n--- Generating continuous prediction plot for a single trial... ---")

    # This function needs to recreate the feature extraction pipeline for one trial
    ch_names = single_trial_data['ch_names']
    best_channels = ['CP6', 'C4', 'F7', 'FT9', 'FT10', 'C3', 'F3', 'F8', 'O1', 'FC2', 'FC6', 'P4', 'F4', 'FC5', 'P3',
                     'T8']
    channel_indices = [ch_names.index(ch) for ch in best_channels if ch in ch_names]

    eeg_epochs = single_trial_data['eeg_epochs'][:, channel_indices, :]
    force_epochs = single_trial_data['force_epochs']
    sfreq = single_trial_data['sfreq']

    # Extract all 5 types of features
    alpha_power = np.mean(mne.time_frequency.psd_array_multitaper(eeg_epochs, sfreq, 8, 12, verbose=False)[0], axis=2)
    beta_power = np.mean(mne.time_frequency.psd_array_multitaper(eeg_epochs, sfreq, 13, 30, verbose=False)[0], axis=2)
    gamma_power = np.mean(mne.time_frequency.psd_array_multitaper(eeg_epochs, sfreq, 30, 48, verbose=False)[0], axis=2)
    rms_features = np.sqrt(np.mean(eeg_epochs ** 2, axis=2))
    var_features = np.var(eeg_epochs, axis=2)

    X_trial = np.concatenate([alpha_power, beta_power, gamma_power, rms_features, var_features], axis=1)
    y_trial = np.mean(force_epochs, axis=2)

    # Use the pre-fitted scaler to transform the trial's features
    X_trial_scaled = scaler.transform(X_trial)

    # Create sequences from the trial's data
    X_trial_seq, y_trial_seq = create_sequences(X_trial_scaled, y_trial, sequence_length)

    # Convert to a PyTorch tensor
    X_tensor = torch.from_numpy(X_trial_seq).float()

    # Get model predictions
    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(X_tensor)

    y_pred_trial = y_pred_tensor.numpy()

    # Create a proper time axis (step is 50ms)
    time_step = 0.050
    time_axis = np.arange(len(y_trial_seq)) * time_step

    # Plotting logic
    n_signals = y_trial_seq.shape[1]
    signal_names = ["D1_Flex", "D1_Abd", "D2_Flex", "D2_Abd", "D3_Flex", "D3_Abd", "D4_Flex", "D4_Abd", "D5_Flex",
                    "D5_Abd"]
    fig, axes = plt.subplots(n_signals, 1, figsize=(18, 5 * n_signals), sharex=True)
    if n_signals == 1: axes = [axes]

    for i in range(n_signals):
        axes[i].plot(time_axis, y_trial_seq[:, i], label='Actual', alpha=0.9, color='blue')
        axes[i].plot(time_axis, y_pred_trial[:, i], label='Predicted', alpha=0.8, color='red', linestyle='--')
        axes[i].set_ylabel(f'{signal_names[i]} Force')
        axes[i].legend()
        axes[i].grid(True)

    axes[0].set_title(f"Continuous (Unshuffled) Prediction on a Single Trial")
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()


def plot_individual_signal_predictions(y_actual, y_predicted, title_prefix="Prediction for Signal"):
    """
    Generates a separate plot for each force signal, showing actual, raw predicted,
    and smoothed predicted values for a SLICE of the test set for clarity.
    """
    if y_actual.shape[0] == 0 or y_predicted.shape[0] == 0:
        print(f"No data to plot for individual signals.")
        return

    # --- NEW: Define a slice of the data to plot for clarity ---
    # Let's plot the first 300 samples of the test set to zoom in.
    num_samples_to_plot = 300
    y_actual_slice = y_actual[:num_samples_to_plot]
    y_predicted_slice = y_predicted[:num_samples_to_plot]


    # Apply the Savgol Filter for smoothing on the sliced data
    window_length = 7
    polyorder = 2
    # Important: Ensure window_length is not greater than the number of samples
    if len(y_predicted_slice) < window_length:
        print("Warning: Not enough data points to apply smoothing. Skipping smoothed plot.")
        y_smoothed_slice = y_predicted_slice # Just use raw if not enough points
    else:
        y_smoothed_slice = savgol_filter(y_predicted_slice, window_length, polyorder, axis=0)

    n_signals = y_actual_slice.shape[1]
    signal_names = [
        "D1_Flex (Thumb)", "D1_Abd (Thumb)",
        "D2_Flex (Index)", "D2_Abd (Index)",
        "D3_Flex (Middle)", "D3_Abd (Middle)",
        "D4_Flex (Ring)", "D4_Abd (Ring)",
        "D5_Flex (Pinky)", "D5_Abd (Pinky)"
    ]

    # Loop to create one plot per signal
    for i in range(n_signals):
        plt.figure(figsize=(15, 5))

        # Plot the actual data from the slice
        plt.plot(y_actual_slice[:, i], label='Actual', alpha=0.9, color='blue', linewidth=1.5)

        # Plot the raw prediction from the slice
        plt.plot(y_predicted_slice[:, i], label='Raw Predicted', alpha=0.5, color='orange', linestyle=':')

        # Plot the smoothed prediction from the slice
        plt.plot(y_smoothed_slice[:, i], label='Smoothed Predicted', alpha=0.9, color='red', linestyle='--')

        plt.title(f"{title_prefix}: {signal_names[i]} (First {num_samples_to_plot} Test Samples)")
        plt.xlabel(f'Epoch Index (Slice of Shuffled Test Set)')
        plt.ylabel('Force Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()