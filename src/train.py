
import torch
import torch.nn as nn
#from torch.optim import AdamW
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import copy
import optuna

from model import LSTMRegressor


def train_and_evaluate_lstm(X_seq, y_seq, test_size=0.2, random_state=42):
    """
    Manages the full training and evaluation pipeline for the LSTM model.
    """
    print("\n--- STEP 4: MODEL TRAINING AND EVALUATION (LSTM REGRESSION) ---")

    # 1. Convert NumPy arrays to PyTorch Tensors
    X_tensor = torch.from_numpy(X_seq).float()
    y_tensor = torch.from_numpy(y_seq).float()

    # 2. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor, y_tensor, test_size=test_size, random_state=random_state
    )
    print(f"    Data split: Train samples={X_train.shape[0]}, Test samples={X_test.shape[0]}")

    # 3. Create PyTorch DataLoaders
    batch_size = 64
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    print(f"    Using batch size: {batch_size}")

    # 4. Initialize the model with specific hyperparameters.
    #    You correctly identified these values from the data shapes!
    input_features = X_train.shape[2]  # Get 80 from (samples, seq_len, 80)
    output_features = y_train.shape[1]  # Get 10 from (samples, 10)
    hidden_units = 50
    num_layers = 3  # Changed back to 2 as a simpler starting point

    model = LSTMRegressor(
        input_features=input_features,
        hidden_units=hidden_units,
        num_layers=num_layers,
        output_features=output_features
    )
    print(f"    LSTM model initialized with {hidden_units} hidden units and {num_layers} layers.")

    # 5. Define the Loss Function and Optimizer
    loss_function = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    # 6. The Training Loop
    num_epochs = 25
    print(f"\n--- Starting training for {num_epochs} epochs... ---")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            # --- CORRECTED LINE ---
            # a. Make a prediction with the model INSTANCE
            y_pred = model(X_batch)

            # b. Calculate the loss between prediction and actual
            loss = loss_function(y_pred, y_batch)

            # c. Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Training Loss: {avg_loss:.4f}")

    # 7. The Evaluation Loop
    model.eval()
    all_y_test = []
    all_y_pred = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_pred = model(X_batch)  # Also need to use the model instance here
            all_y_test.append(y_batch.numpy())
            all_y_pred.append(y_pred.numpy())

    # Concatenate all batches to get the full test set predictions
    y_test_full = np.concatenate(all_y_test)
    y_pred_full = np.concatenate(all_y_pred)

    # 8. Calculate final metrics
    mse = np.mean((y_pred_full - y_test_full) ** 2)
    r2 = r2_score(y_test_full, y_pred_full)

    print("\n--- LSTM Model Evaluation Results ---")
    print(f"    Mean Squared Error (MSE): {mse:.4f}")
    print(f"    R-squared (R2 Score): {r2:.4f}")
    print("-----------------------------------")

    return model, (y_test_full, y_pred_full)


def train_and_evaluate_lstm_Optuna(X_seq, y_seq, test_size=0.2, random_state=42):
    """
    Finds the best hyperparameters for the LSTM model using Optuna, trains a final
    model with those params using early stopping, and evaluates it.
    """
    print("\n--- STEP 4: HYPERPARAMETER TUNING & EVALUATION (LSTM with Optuna) ---")

    # 1. Convert to Tensors and create Train/Validation/Test splits
    X_tensor = torch.from_numpy(X_seq).float()
    y_tensor = torch.from_numpy(y_seq).float()

    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor, y_tensor, test_size=test_size, random_state=random_state
    )
    X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
        X_train, y_train, test_size=0.25, random_state=random_state
    )
    print(f"    Data split: Train={X_train_opt.shape[0]}, Validation={X_val_opt.shape[0]}, Test={X_test.shape[0]}")

    # --- 2. THE OPTUNA SEARCH (This part was missing) ---
    def objective(trial):
        # (The objective function is the same as before, it finds the best params)
        params = {
            'input_features': X_train.shape[2], 'output_features': y_train.shape[1],
            'hidden_units': trial.suggest_int('hidden_units', 32, 128, step=16),
            'num_layers': trial.suggest_int('num_layers', 1, 3),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        }
        model = LSTMRegressor(
            input_features=params['input_features'], hidden_units=params['hidden_units'],
            num_layers=params['num_layers'], output_features=params['output_features'],
            dropout=params['dropout']
        )
        batch_size = 128
        train_loader = DataLoader(TensorDataset(X_train_opt, y_train_opt), shuffle=True, batch_size=batch_size)
        val_loader = DataLoader(TensorDataset(X_val_opt, y_val_opt), shuffle=False, batch_size=batch_size)
        loss_function = nn.MSELoss()
        optimizer = Adam(model.parameters(), lr=params['learning_rate'])
        num_epochs = 25
        for epoch in range(num_epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                y_pred = model(X_batch);
                loss = loss_function(y_pred, y_batch)
                optimizer.zero_grad();
                loss.backward();
                optimizer.step()
        model.eval()
        val_preds = []
        with torch.no_grad():
            for X_batch_val, _ in val_loader:
                y_pred_val = model(X_batch_val)
                val_preds.append(y_pred_val.numpy())
        y_pred_full_val = np.concatenate(val_preds)
        r2 = r2_score(y_val_opt.numpy(), y_pred_full_val)
        return r2

    print("\n--- Starting Optuna search... ---")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30, timeout=1800)
    print("--- Optuna search complete. ---")

    print("\n--- Best LSTM Model Parameters Found (Optuna) ---")
    best_params = study.best_params
    print(best_params)
    print("-------------------------------------------------")

    # --- 3. Train the FINAL model with the best params and EARLY STOPPING ---
    print("\n--- Training final LSTM model on full training data with best params and early stopping... ---")
    final_model = LSTMRegressor(
        input_features=X_train.shape[2],
        output_features=y_train.shape[1],
        hidden_units=best_params['hidden_units'],
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout']
    )
    final_loss_function = nn.MSELoss()
    final_optimizer = Adam(final_model.parameters(), lr=best_params['learning_rate'])

    final_train_loader = DataLoader(TensorDataset(X_train, y_train), shuffle=True, batch_size=128)
    final_val_loader = DataLoader(TensorDataset(X_test, y_test), shuffle=False, batch_size=128)

    train_loss_history, val_loss_history = [], []
    best_val_loss, best_model_state = float('inf'), None
    epochs_no_improve, patience = 0, 5

    num_final_epochs = 50  # Set a higher max, early stopping will find the best
    print(f"--- Starting final training for up to {num_final_epochs} epochs (patience={patience})... ---")

    for epoch in range(num_final_epochs):
        # Training Phase
        final_model.train()
        batch_train_loss = 0
        for X_batch, y_batch in final_train_loader:
            y_pred = final_model(X_batch);
            loss = final_loss_function(y_pred, y_batch)
            final_optimizer.zero_grad();
            loss.backward();
            final_optimizer.step()
            batch_train_loss += loss.item()
        epoch_train_loss = batch_train_loss / len(final_train_loader)
        train_loss_history.append(epoch_train_loss)

        # Validation Phase
        final_model.eval()
        batch_val_loss = 0
        with torch.no_grad():
            for X_batch_val, y_batch_val in final_val_loader:
                y_pred_val = final_model(X_batch_val)
                val_loss = final_loss_function(y_pred_val, y_batch_val)
                batch_val_loss += val_loss.item()
        epoch_val_loss = batch_val_loss / len(final_val_loader)
        val_loss_history.append(epoch_val_loss)

        print(
            f"Epoch {epoch + 1}/{num_final_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

        # Early Stopping Logic
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = copy.deepcopy(final_model.state_dict())
            epochs_no_improve = 0
            print(f"    -> New best validation loss: {best_val_loss:.4f}. Saving model state.")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"\n--- Early stopping triggered after {patience} epochs with no improvement. ---")
            break

    # --- 4. Final Evaluation using the BEST model state ---
    print("\n--- Loading best model state for final evaluation... ---")
    final_model.load_state_dict(best_model_state)

    final_model.eval()
    all_y_test, all_y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in final_val_loader:
            y_pred = final_model(X_batch)
            all_y_test.append(y_batch.numpy());
            all_y_pred.append(y_pred.numpy())

    y_test_full = np.concatenate(all_y_test)
    y_pred_full = np.concatenate(all_y_pred)

    mse = mean_squared_error(y_test_full, y_pred_full)
    r2 = r2_score(y_test_full, y_pred_full)

    print("\n--- Final LSTM Model Evaluation Results (from best epoch) ---")
    print(f"    Mean Squared Error (MSE): {mse:.4f}")
    print(f"    R-squared (R2 Score): {r2:.4f}")
    print("---------------------------------------------------------")

    history = {'train_loss': train_loss_history, 'val_loss': val_loss_history}

    return final_model, (y_test_full, y_pred_full), study, history