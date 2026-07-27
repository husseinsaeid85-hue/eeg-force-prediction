# In test_inference_speed.py
import torch
import time
import numpy as np
import sys
import os

# Add the 'src' directory to the Python path to import the model class
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from model import LSTMRegressor

# --- 1. Define Model Path and Parameters ---
# Use the exact parameters and path you provided.
MODEL_PATH = os.environ.get(
    'EEG_MODEL_PATH',
    os.path.join(os.path.dirname(__file__), 'trained_lstm_optuna_eeg_emg_model.pth'),
)
BEST_PARAMS = {
    'input_features': 120,  # 80 EEG + 40 EMG
    'output_features': 10,
    'hidden_units': 128,
    'num_layers': 2,
    'dropout': 0.29937005637666414,

}


def measure_inference_speed():
    """
    Loads the trained LSTM model and measures its inference speed.
    """
    # --- 2. Initialize Model and Load Trained Weights ---
    # Check if a GPU is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device: {device} ---")

    if not os.path.exists(MODEL_PATH):
        sys.exit(f"ERROR: Trained model not found at '{MODEL_PATH}'. Please ensure the file is in the project root.")

    # Instantiate the model architecture with the correct parameters
    model = LSTMRegressor(
        input_features=BEST_PARAMS['input_features'],
        output_features=BEST_PARAMS['output_features'],
        hidden_units=BEST_PARAMS['hidden_units'],
        num_layers=BEST_PARAMS['num_layers'],
        dropout=BEST_PARAMS['dropout'],

    )

    # Load the saved weights (state dictionary) into the model architecture
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    # Move the model to the selected device and set it to evaluation mode
    model.to(device)
    model.eval()
    print(f"--- Successfully loaded trained model from: {MODEL_PATH} ---")

    # --- 3. Create Dummy Input ---
    # Create a single dummy input sequence on the same device
    # Shape is (batch_size, sequence_length, num_features)
    sequence_length = 5
    input_tensor = torch.randn(1, sequence_length, BEST_PARAMS['input_features']).to(device)

    # --- 4. Run Inference Speed Test ---
    n_runs = 1000
    latencies = []

    print(f"\n--- Running inference speed test ({n_runs} runs)... ---")

    # Warm-up run (very important for GPU to initialize kernels)
    with torch.no_grad():
        _ = model(input_tensor)

    # Main measurement loop
    for _ in range(n_runs):
        start_time = time.time()
        with torch.no_grad():  # Disable gradient calculation for maximum speed
            _ = model(input_tensor)
        end_time = time.time()
        latencies.append((end_time - start_time) * 1000)  # Convert to milliseconds

    # --- 5. Report Results ---
    average_latency = np.mean(latencies)
    std_latency = np.std(latencies)

    print("\n--- Model Inference Speed Results ---")
    print(f"    Average latency per prediction: {average_latency:.4f} ms")
    print(f"    Standard Deviation: {std_latency:.4f} ms")
    print(f"    Predictions per second (Throughput): {1000 / average_latency:.2f} FPS")
    print("-----------------------------------")


if __name__ == "__main__":
    measure_inference_speed()