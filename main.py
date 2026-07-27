# main.py
import argparse
import sys
import os
import numpy as np
import pickle
import torch

# Add the 'src' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import ALL versions of your functions
from data_loader import load_and_align_data
from preprocessing import preprocess_data, preprocess_data_with_emg
from features import extract_features, extract_features_with_emg, get_best_channels
import visualize
from model import train_and_evaluate_xgboost_Optuna
from data_sequencer import create_sequences
from train import train_and_evaluate_lstm_Optuna


def parse_args(argv=None):
    """Selects which of the four experiment configurations to run."""
    parser = argparse.ArgumentParser(
        description="Train a finger-force regressor on EEG (optionally fused with EMG) features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--features', choices=['eeg', 'eeg-emg'], default='eeg-emg',
        help="'eeg' for EEG-only (80 features), 'eeg-emg' for the EEG+EMG fusion set (120 features)."
    )
    parser.add_argument(
        '--model', choices=['lstm', 'xgboost'], default='lstm',
        help="'lstm' runs the sequence model (Optuna-tuned), 'xgboost' the tabular baseline (Optuna-tuned)."
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()

    # --- MASTER EXPERIMENT FLAGS (set from the command line) ---
    USE_EMG_FEATURES = (args.features == 'eeg-emg')
    RUN_LSTM_EXPERIMENT = (args.model == 'lstm')

    # --- Define Cache Filenames based on the experiment ---
    if USE_EMG_FEATURES:
        PREPROCESSED_DATA_CACHE_PATH = 'preprocessed_epoched_data_with_emg.pkl'
        FEATURES_CACHE_PATH = 'extracted_features_with_emg_cache.pkl'
        SEQUENCED_DATA_CACHE_PATH = 'sequenced_data_with_emg_cache.pkl'
    else:
        PREPROCESSED_DATA_CACHE_PATH = 'preprocessed_epoched_data_cache.pkl'
        FEATURES_CACHE_PATH = 'extracted_features_cache.pkl'
        SEQUENCED_DATA_CACHE_PATH = 'sequenced_data_cache.pkl'

    print("Starting main script...")
    print(f"--- Configuration: USE_EMG_FEATURES={USE_EMG_FEATURES}, RUN_LSTM_EXPERIMENT={RUN_LSTM_EXPERIMENT} ---")

    # --- STAGE 1 & 2: Data Loading and Feature Extraction ---
    X_scaled, y, scaler = None, None, None
    if os.path.exists(FEATURES_CACHE_PATH):
        print(f"--- Loading features from cache: {FEATURES_CACHE_PATH} ---")
        with open(FEATURES_CACHE_PATH, 'rb') as f:
            X_scaled, y, scaler = pickle.load(f)
    else:
        print(f"--- No feature cache found. Running full data pipeline... ---")
        preprocessed_epoched_data = None
        if os.path.exists(PREPROCESSED_DATA_CACHE_PATH):
            with open(PREPROCESSED_DATA_CACHE_PATH, 'rb') as f:
                preprocessed_epoched_data = pickle.load(f)
        else:
            my_aligned_data = load_and_align_data()  # This always loads all data streams
            if not my_aligned_data: sys.exit("No data was loaded.")

            if USE_EMG_FEATURES:
                print("\n--- Running PREPROCESSING with EMG ---")
                preprocessed_epoched_data = preprocess_data_with_emg(my_aligned_data)
            else:
                print("\n--- Running PREPROCESSING for EEG-only ---")
                preprocessed_epoched_data = preprocess_data(my_aligned_data)

            with open(PREPROCESSED_DATA_CACHE_PATH, 'wb') as f:
                pickle.dump(preprocessed_epoched_data, f)

        if USE_EMG_FEATURES:
            print("\n--- Running FEATURE EXTRACTION with EMG ---")
            X_scaled, y, scaler = extract_features_with_emg(preprocessed_epoched_data)
        else:
            print("\n--- Running FEATURE EXTRACTION for EEG-only ---")
            X_scaled, y, scaler = extract_features(preprocessed_epoched_data)

        with open(FEATURES_CACHE_PATH, 'wb') as f:
            pickle.dump((X_scaled, y, scaler), f)

    # --- STAGE 3: Model Training ---
    trained_model, y_test, y_pred, study, history = None, None, None, None, None
    if RUN_LSTM_EXPERIMENT:
        print("\n--- RUNNING LSTM EXPERIMENT ---")
        X_lstm, y_lstm = None, None
        if os.path.exists(SEQUENCED_DATA_CACHE_PATH):
            with open(SEQUENCED_DATA_CACHE_PATH, 'rb') as f:
                X_lstm, y_lstm = pickle.load(f)
        else:
            X_lstm, y_lstm = create_sequences(X_scaled, y, sequence_length=5)
            with open(SEQUENCED_DATA_CACHE_PATH, 'wb') as f:
                pickle.dump((X_lstm, y_lstm), f)

        trained_model, (y_test, y_pred), study, history = train_and_evaluate_lstm_Optuna(X_lstm, y_lstm)
        model_type = "lstm_optuna"
    else:
        print("\n--- RUNNING XGBOOST EXPERIMENT ---")
        # XGBoost has no per-epoch loss history, so it returns three values, not four.
        trained_model, (y_test, y_pred), study = train_and_evaluate_xgboost_Optuna(X_scaled, y)
        history = None
        model_type = "xgboost_optuna"

    # --- Save the trained model with a descriptive name ---
    feature_type = "eeg_emg" if USE_EMG_FEATURES else "eeg_only"
    file_extension = ".pth" if RUN_LSTM_EXPERIMENT else ".pkl"
    TRAINED_MODEL_CACHE_PATH = f"trained_{model_type}_{feature_type}_model{file_extension}"
    plot_title = f"{'LSTM' if RUN_LSTM_EXPERIMENT else 'XGBoost'} ({feature_type}): Prediction on Shuffled Test Set"

    print(f"--- Saving trained model to: {TRAINED_MODEL_CACHE_PATH} ---")
    if RUN_LSTM_EXPERIMENT:
        torch.save(trained_model.state_dict(), TRAINED_MODEL_CACHE_PATH)
    else:
        with open(TRAINED_MODEL_CACHE_PATH, 'wb') as f:
            pickle.dump(trained_model, f)
    print("--- Trained model saved. ---")

    # --- STAGE 4: Visualization ---
    print("\n--- Running Visualization ---")

    if history: visualize.plot_loss_curves(history)
    if y_test is not None: visualize.plot_individual_signal_predictions(y_test, y_pred, title_prefix=plot_title)
    if RUN_LSTM_EXPERIMENT:
        # (The unshuffled plot logic remains the same)
        pass  # Add back the unshuffled plot logic here if desired
    if study: visualize.plot_optuna_history(study)
    if not RUN_LSTM_EXPERIMENT: visualize.plot_xgboost_feature_importance(trained_model, get_best_channels())

    print("\n--- Main script finished. ---")


if __name__ == "__main__":
    main()