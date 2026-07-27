# In real_time_inference.py

import torch
import numpy as np
import pickle
import time
from pylsl import StreamInlet, resolve_streams
import mne
import sys
import os

# Add the 'src' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model import LSTMRegressor
from features import get_best_channels
from data_loader import load_and_align_data

# --- CONFIGURATION ---
MODEL_PATH = 'trained_lstm_optuna_eeg_emg_model.pth'
SCALER_PATH = 'extracted_features_with_emg_cache.pkl'
PREPROCESSING_TOOLS_PATH = 'real_time_preprocessing_tools.pkl'
SEQUENCE_LENGTH = 5
NUM_EEG_CHANNELS = 31
NUM_EMG_CHANNELS = 40

# --- 1. Load Preprocessing Tools, Model, and Scaler ---
# ... (The "Load or Generate" block for preprocessing tools is correct as is)

# --- 2. Load the Trained Model and Scaler ---
print("\n--- Loading trained model and scaler... ---")
with open(SCALER_PATH, 'rb') as f:
    # The cache saves (X_scaled, y, scaler). We only need the scaler.
    _, _, scaler = pickle.load(f)

# --- CORRECTED: Use the exact best parameters from your Optuna run ---
best_params = {
    'hidden_units': 128,
    'num_layers': 2,
    'learning_rate': 0.002383691006993713,
    'dropout': 0.29937005637666414,
}

model = LSTMRegressor(
    input_features=120,  # 80 EEG + 40 EMG
    output_features=10,
    hidden_units=best_params['hidden_units'],
    num_layers=best_params['num_layers'],
    dropout=best_params['dropout'],

)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
print("--- Model and scaler loaded successfully. ---")

# --- 3. Connect to the LSL Stream ---
print("\nLooking for LSL data stream...")
# Resolve all streams and then filter for the one you want
streams = resolve_streams(wait_time=5.0)
if not streams:
    sys.exit("Error: No LSL streams found. Is the BrainVision Connector streaming?")

eeg_streams = [s for s in streams if s.type() == 'EEG']  # Or filter by name
if not eeg_streams:
    sys.exit("Error: No stream with type 'EEG' found.")

inlet = StreamInlet(eeg_streams[0])
info = inlet.info()
sfreq = info.nominal_srate()
ch_names_from_stream = [info.desc().child("channels").child("channel").child_value("label") for i in
                        range(info.channel_count())]

# ... (The rest of the channel setup is correct as is)
expected_total_channels = NUM_EEG_CHANNELS + NUM_EMG_CHANNELS
if info.channel_count() < expected_total_channels:
    sys.exit(f"Error: Stream has {info.channel_count()} channels, but expected at least {expected_total_channels}.")
eeg_indices = list(range(NUM_EEG_CHANNELS))
emg_indices = list(range(NUM_EEG_CHANNELS, expected_total_channels))
eeg_channel_names = [ch_names_from_stream[i] for i in eeg_indices]
best_eeg_channels = get_best_channels()
best_eeg_indices_in_eeg_subset = [eeg_channel_names.index(ch) for ch in best_eeg_channels if ch in eeg_channel_names]

# --- 4. The Real-Time Inference Loop ---
feature_buffer = np.zeros((SEQUENCE_LENGTH, 120))  # 120 features
# Use a buffer for raw EEG data to ensure continuous filtering
eeg_buffer = np.zeros((len(eeg_indices), int(sfreq * 1.0)))  # 1-second buffer

print("\n--- Starting real-time prediction loop (press Ctrl+C to stop) ---")
try:
    while True:
        # Pull a small chunk of data (e.g., 50ms)
        chunk_size = int(sfreq * 0.050)
        chunk, timestamps = inlet.pull_chunk(timeout=1.0, max_samples=chunk_size)

        if chunk:
            live_data = np.array(chunk).T

            # 1. Separate live data
            live_eeg = live_data[eeg_indices, :]
            live_emg = live_data[emg_indices, :]

            # 2. Preprocess the live EEG using the same steps as offline
            eeg_buffer = np.roll(eeg_buffer, -live_eeg.shape[1], axis=1)
            eeg_buffer[:, -live_eeg.shape[1]:] = live_eeg

            mne_info = mne.create_info(ch_names=eeg_channel_names, sfreq=sfreq, ch_types='eeg')
            raw_obj = mne.io.RawArray(eeg_buffer.copy(), mne_info, verbose=False)
            raw_obj.set_montage(montage, on_missing='warn')
            raw_obj.filter(0.1, 48.0, method='iir', iir_params=dict(order=4, ftype='butter'), verbose=False)
            raw_csd = mne.preprocessing.compute_current_source_density(raw_obj, verbose=False)
            ica.apply(raw_csd, verbose=False)  # Apply the pre-fitted ICA

            window_samples = int(sfreq * 0.200)  # Our feature window is 200ms
            preprocessed_eeg_window = raw_csd.get_data()[:, -window_samples:]

            # 3. Extract features from the processed windows
            eeg_window_selected = preprocessed_eeg_window[best_eeg_indices_in_eeg_subset, :]
            eeg_rms = np.sqrt(np.mean(eeg_window_selected ** 2, axis=1))
            eeg_var = np.var(eeg_window_selected, axis=1)
            psds, freqs = mne.time_frequency.psd_array_multitaper(eeg_window_selected[np.newaxis, :, :], sfreq, 8, 48,
                                                                  verbose=False)
            eeg_alpha = np.mean(psds[0, :, (freqs >= 8) & (freqs <= 12)], axis=1)
            eeg_beta = np.mean(psds[0, :, (freqs >= 13) & (freqs <= 30)], axis=1)
            eeg_gamma = np.mean(psds[0, :, (freqs >= 30) & (freqs <= 48)], axis=1)
            eeg_features = np.concatenate([eeg_alpha, eeg_beta, eeg_gamma, eeg_rms, eeg_var])

            # EMG features - Assuming streamed EMG is raw signal, so calculate RMS
            emg_features = np.sqrt(np.mean(live_emg ** 2, axis=1))

            # 4. Combine and scale
            current_feature_vector = np.concatenate([eeg_features, emg_features])
            scaled_features = scaler.transform(current_feature_vector.reshape(1, -1))

            # 5. Update feature buffer and predict
            feature_buffer = np.roll(feature_buffer, -1, axis=0)
            feature_buffer[-1, :] = scaled_features

            input_for_lstm = torch.from_numpy(feature_buffer).float().unsqueeze(0)

            with torch.no_grad():
                prediction = model(input_for_lstm)

            pred_str = " ".join([f"{p:.2f}" for p in prediction.numpy().flatten()])
            print(f"Predicted Force: [ {pred_str} ]", end='\r')

except KeyboardInterrupt:
    print("\n\n--- Stopping real-time loop. ---")
except Exception as e:
    print(f"\n\nAn error occurred in the real-time loop: {e}")