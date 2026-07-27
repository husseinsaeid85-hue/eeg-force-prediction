# In file: src/data_sequencer.py

import numpy as np

def create_sequences(X, y, sequence_length=5):
    """
    Transforms time-series data into overlapping sequences for LSTM models.

    Args:
        X (np.ndarray): The input feature matrix of shape (n_samples, n_features).
                        In your case, this is (15494, 80).
        y (np.ndarray): The target matrix of shape (n_samples, n_targets).
                        In your case, this is (15494, 10).
        sequence_length (int): The number of time steps in each sequence.
                               This is the "look-back" window for the LSTM.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The reshaped features for the LSTM (X_seq) of shape
                          (num_sequences, sequence_length, n_features).
            - np.ndarray: The corresponding targets (y_seq) of shape
                          (num_sequences, n_targets).
    """
    print(f"\n--- Creating sequences with a length of {sequence_length} time steps... ---")

    X_seq, y_seq = [], []
    num_samples = X.shape[0]

    # The loop iterates through the data to create overlapping sequences.
    # It stops when there are not enough remaining data points to form a full sequence.
    for i in range(num_samples - sequence_length + 1):
        # Find the end of this sequence
        end_ix = i + sequence_length

        # Extract the sequence of features (the 'X' part)
        # This will be a slice of shape (sequence_length, n_features)
        seq_x = X[i:end_ix]
        X_seq.append(seq_x)

        # The target 'y' for this sequence is the value at the very end of the sequence.
        # This is a common setup for time-series forecasting and decoding.
        # The model uses the history (the sequence) to predict the final state.
        seq_y = y[end_ix - 1]
        y_seq.append(seq_y)

    # Convert the lists of sequences into NumPy arrays.

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    print(f"    Original X shape: {X.shape}")
    print(f"    Original y shape: {y.shape}")
    print(f"    Resulting X_seq shape: {X_seq.shape}")
    print(f"    Resulting y_seq shape: {y_seq.shape}")
    print("--- Sequence creation complete. ---")

    return X_seq, y_seq