# src/model.py

import numpy as np
import optuna
import torch
import torch.nn as nn
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import uniform, randint # For defining parameter distributions
from optuna.integration import XGBoostPruningCallback


def train_and_evaluate_xgboost(X_scaled, y, test_size=0.2, random_state=42):
    """
    Trains an XGBoost Regression model with Randomized Search for hyperparameter tuning,
    evaluates its performance, and prints results.

    Args:
        X_scaled (np.ndarray): Scaled feature matrix.
        y (np.ndarray): Target matrix (force data).
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Controls the shuffling applied to the data before splitting.

    Returns:
        xgboost.XGBRegressor: The best trained XGBoost Regression model.
        tuple: (y_test, y_pred) for further analysis/visualization.
    """
    print("\n--- STEP 4: MODEL TRAINING AND EVALUATION (XGBOOST REGRESSION with RandomizedSearchCV) ---")

    # 1. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )
    print(f"    Data split: Train samples={X_train.shape[0]}, Test samples={X_test.shape[0]}")
    print(f"    X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"    X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # 2. Define the base XGBoost Regressor model
    base_model = XGBRegressor(
        objective='reg:squarederror',
        random_state=random_state,
        n_jobs=-1, # Use all available CPU cores
        tree_method='hist' # Faster algorithm for larger datasets
    )

    # 3. Define the hyperparameter search space for Randomized Search
    # We use scipy.stats distributions for continuous parameters
    # and lists for discrete parameters.
    param_distributions = {
        'n_estimators': randint(200, 400), # Number of boosting rounds (trees)
        'learning_rate': uniform(0.01, 0.2), # Step size shrinkage
        'max_depth': randint(7, 12),       # Maximum depth of a tree
        'subsample': uniform(0.6, 0.4),    # Subsample ratio of the training instance
        'colsample_bytree': uniform(0.6, 0.4), # Subsample ratio of columns when constructing each tree
        'gamma': uniform(0, 0.5),          # Minimum loss reduction required to make a further partition
        'lambda': uniform(0.5, 2),         # L2 regularization term on weights (reg_lambda in XGBoost)
        'alpha': uniform(0, 0.5),          # L1 regularization term on weights (reg_alpha in XGBoost)
    }
    print("    XGBoost Regressor base model initialized.")
    print("    Defined hyperparameter search space for Randomized Search.")

    # 4. Initialize RandomizedSearchCV
    # n_iter: Number of parameter settings that are sampled.
    #         More iterations mean more thorough search, but take longer.
    # cv: Number of cross-validation folds.
    # scoring: Metric to optimize (e.g., 'r2' for R-squared, 'neg_mean_squared_error' for MSE)
    # verbose: How much output to print (2 is good for progress)
    # n_jobs: Number of CPU cores to use for parallel processing during search (-1 means all available)
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=200, # Number of different parameter combinations to try (adjust as needed)
        scoring='r2', # Optimize for R-squared
        cv=5,         # 5-fold cross-validation
        verbose=1,    # Print progress
        random_state=random_state,
        n_jobs=-1     # Use all available CPU cores for the search
    )
    print(f"    RandomizedSearchCV initialized with {random_search.n_iter} iterations and {random_search.cv}-fold CV, optimizing for R-squared.")

    # 5. Train the model (perform the randomized search)
    print("    Starting Randomized Search to find best model hyperparameters...")
    random_search.fit(X_train, y_train)
    print("    Randomized Search complete.")

    # Get the best model found
    best_model = random_search.best_estimator_
    print("\n--- Best XGBoost Model Parameters Found ---")
    print(random_search.best_params_)
    print("------------------------------------------")

    # 6. Make predictions on the test set using the best model
    y_pred = best_model.predict(X_test)
    print("    Predictions made on the test set using the best model.")

    # 7. Evaluate the best model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Model Evaluation Results (Best XGBoost) ---")
    print(f"    Mean Squared Error (MSE): {mse:.4f}")
    print(f"    R-squared (R2 Score): {r2:.4f}")
    print("----------------------------------------------")

    return best_model, (y_test, y_pred)


def train_and_evaluate_xgboost_GridSearch(X_scaled, y, test_size=0.2, random_state=42):

    print("\n--- STEP 4: MODEL TRAINING AND EVALUATION (XGBOOST REGRESSION with GridSearchCV) ---")

    # 1. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )
    print(f"    Data split: Train samples={X_train.shape[0]}, Test samples={X_test.shape[0]}")


    # 2. Define the base XGBoost Regressor model
    base_model = XGBRegressor(
        objective='reg:squarederror',
        random_state=random_state,
        n_jobs=-1, # Use all available CPU cores
        tree_method='hist' # Faster algorithm for larger datasets
    )
    print("    XGBoost Regressor base model initialized.")

    # 3. Define the hyperparameter GRID for Grid Search

    param_grid = {
        'n_estimators': [250, 300, 350],  # Values around the previous best of 262
        'learning_rate': [0.08, 0.1, 0.12],  # Values around the previous best of 0.1197
        'max_depth': [7, 8, 9],  # Values around the previous best of 8
        'subsample': [0.8, 0.9],  # Common good values
        'colsample_bytree': [0.8, 0.9]  # Common good values
    }
    print("    Defined hyperparameter grid for Grid Search.")

    # 4. Initialize GridSearchCV

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid = param_grid,

        scoring='r2', # Optimize for R-squared
        cv=5,         # 5-fold cross-validation
        verbose=2,    # Print progress

        n_jobs=-1     # Use all available CPU cores for the search
    )
    total_fits = (
            len(param_grid['n_estimators']) *
            len(param_grid['learning_rate']) *
            len(param_grid['max_depth']) *
            len(param_grid['subsample']) *
            len(param_grid['colsample_bytree']) *
            grid_search.cv
    )
    print(f"    GridSearchCV initialized. It will perform a total of {total_fits} fits.")

    # 5. Train the model (perform the Grid search)
    print("    Starting Grid Search to find best model hyperparameters...")
    grid_search.fit(X_train, y_train)
    print("    Grid Search complete.")

    # Get the best model found
    best_model = grid_search.best_estimator_
    print("\n--- Best XGBoost Model Parameters Found ---")
    print(grid_search.best_params_)
    print("------------------------------------------")

    # 6. Make predictions on the test set using the best model
    y_pred = best_model.predict(X_test)
    print("    Predictions made on the test set using the best model.")

    # 7. Evaluate the best model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Model Evaluation Results (Best XGBoost) ---")
    print(f"    Mean Squared Error (MSE): {mse:.4f}")
    print(f"    R-squared (R2 Score): {r2:.4f}")
    print("----------------------------------------------")

    return best_model, (y_test, y_pred)


def train_and_evaluate_xgboost_Optuna(X_scaled, y, test_size=0.2, random_state=42):
    """
    Trains an XGBoost model using Optuna.
    NOTE: This version is a diagnostic test with NO early stopping.
    """
    print("\n--- STEP 4: MODEL TRAINING AND EVALUATION (XGBOOST REGRESSION with Optuna - DIAGNOSTIC MODE) ---")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )
    # We don't need a separate validation set if we're not doing early stopping
    print(f"    Data split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")

    def objective(trial):
        param = {
            'objective': 'reg:squarederror',
            'n_jobs': -1,
            'tree_method': 'hist',
            'verbosity': 0,
            'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.2, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        }
        model = XGBRegressor(**param)

        # --- DIAGNOSTIC: Train on the full training set without early stopping ---
        # We split the data inside the objective to get a validation score
        X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.25, random_state=random_state)
        model.fit(X_t, y_t)  # Simplest possible fit call

        preds = model.predict(X_v)
        r2 = r2_score(y_v, preds)
        return r2

    print("    Starting Optuna search (NO early stopping)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, timeout=600)  # Reduced n_trials because it will be slower
    print("    Optuna search complete.")

    print("\n--- Best XGBoost Model Parameters Found (Optuna) ---")
    print(study.best_params)
    print("----------------------------------------------------")

    print("    Training final model on full training data with best params...")
    best_params = study.best_params
    final_model = XGBRegressor(objective='reg:squarederror', n_jobs=-1, tree_method='hist', verbosity=0, **best_params)
    final_model.fit(X_train, y_train)
    print("    Final model training complete.")

    y_pred = final_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Model Evaluation Results (Best from Optuna) ---")
    print(f"    Mean Squared Error (MSE): {mse:.4f}")
    print(f"    R-squared (R2 Score): {r2:.4f}")
    print("---------------------------------------------------")

    return final_model, (y_test, y_pred), study


class LSTMRegressor(nn.Module):

    def __init__(self, input_features, hidden_units, num_layers, output_features, dropout=0.2):
        super().__init__()
        self.hidden_units = hidden_units
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_units,
            num_layers=num_layers,
            batch_first=True,
            dropout= dropout if num_layers > 1 else 0
        )

        self.linear = nn.Linear(
            in_features=hidden_units,
            out_features=output_features
        )

    def forward(self, x):
        """
        Defines the forward pass of the model.
        This is where data flows through the layers.
        """
        # Initialize the hidden state and cell state for the LSTM
        # Shape: (num_layers, batch_size, hidden_units)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_units).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_units).to(x.device)

        # Pass the input sequence through the LSTM layer
        # `lstm_out` will contain the outputs for each time step in the sequence.
        # `_` will contain the final hidden and cell states, which we don't need here.
        lstm_out, _ = self.lstm(x, (h0, c0))

        # We only need the output from the VERY LAST time step of the sequence
        # to make our final prediction. lstm_out is (batch, seq, hidden), so we
        # take all batches, the last time step (-1), and all hidden units.
        last_time_step_out = lstm_out[:, -1, :]

        # Pass this final output through our linear layer to get the prediction
        predictions = self.linear(last_time_step_out)

        return predictions


