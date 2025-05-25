import numpy as np
import pandas as pd
import shap
import optuna
import lightgbm as lgb
from sklearn.metrics import r2_score
import logging

# Shouldve used RMSE isntead of R^2 kinda...

class SHAPFeatureSelector:
    def __init__(self, X, y, param_space, cv, scoring=r2_score, step=0.1, min_features=10, early_stopping_rounds=50, n_trials=100):
        """
        :param X: Feature dataframe
        :param y: Target array
        :param param_space: Dictionary defining Optuna hyperparameter search space
        :param cv: Cross-validation strategy (KFold instance)
        :param scoring: Scikit-learn scoring function (e.g., r2_score)
        :param step: Fraction of features to remove per iteration (0.0-1.0)
        :param min_features: Minimum number of features to retain
        :param early_stopping_rounds: Number of rounds for early stopping in LightGBM
        :param n_trials: Number of trials for Optuna optimization
        """
        self.X = X
        self.y = y
        self.param_space = param_space
        self.cv = cv
        self.scoring = scoring
        self.step = step
        self.min_features = min_features
        self.early_stopping_rounds = early_stopping_rounds
        self.n_trials = n_trials
        self.history = []
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    def objective(self, trial, X_current):
        """Objective function for Optuna hyperparameter tuning using reduced features."""
        params = {}
        for key, value in self.param_space.items():
            if isinstance(value, tuple):
                if len(value) == 4:  # For float parameters with log scale
                    params[key] = trial.suggest_float(key, value[0], value[1], log=True)
                elif len(value) == 2:
                    params[key] = trial.suggest_categorical(key, value[0])
                elif value[2] == 'int':
                    params[key] = trial.suggest_int(key, value[0], value[1])
                elif value[2] == 'float':
                    params[key] = trial.suggest_float(key, value[0], value[1])
            else:
                params[key] = value  # Handle constant parameters

        scores = []
        for train_idx, val_idx in self.cv.split(X_current, self.y):
            train_data = lgb.Dataset(X_current.iloc[train_idx], label=self.y.iloc[train_idx])
            val_data = lgb.Dataset(X_current.iloc[val_idx], label=self.y.iloc[val_idx], reference=train_data)
            model = lgb.train(params, train_data, valid_sets=[val_data], 
                            callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)])
            y_pred = model.predict(X_current.iloc[val_idx])
            scores.append(self.scoring(self.y.iloc[val_idx], y_pred))
        return np.mean(scores)
    
    def optimize_hyperparameters(self, X_current):
        """Runs Optuna hyperparameter tuning using the reduced feature set."""
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: self.objective(trial, X_current), n_trials=self.n_trials, timeout=3600 * 2) # 2 hours timeout
        return study.best_params, study.best_value
    
    def compute_shap_importance(self, best_params, X_current, iteration):
        """Computes SHAP feature importances for the current feature set."""
        shap_values_list = []
        for train_idx, val_idx in self.cv.split(X_current, self.y):
            train_data = lgb.Dataset(X_current.iloc[train_idx], label=self.y.iloc[train_idx])
            val_data = lgb.Dataset(X_current.iloc[val_idx], label=self.y.iloc[val_idx], reference=train_data)
            model = lgb.train(best_params, train_data, valid_sets=[val_data], 
                              callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)])
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_current.iloc[val_idx])
            shap_values_list.append(np.abs(shap_values.values).mean(axis=0))
        
        importances = np.mean(shap_values_list, axis=0)
        feature_importance_df = pd.DataFrame({'feature': X_current.columns, 'importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False).reset_index(drop=True)
        
        # Save feature importance for this iteration
        feature_importance_df.to_csv(f"feature_importance_iter_{iteration}.csv", index=False)
        
        return feature_importance_df
    
    def select_features(self, output_file="feature_selection_history.csv"):
        """Performs iterative feature selection using SHAP importances."""
        X_current = self.X.copy()
        iteration = 0
        while X_current.shape[1] > self.min_features:
            iteration += 1
            logging.info(f"Starting iteration {iteration} with {X_current.shape[1]} features.")
            
            # Hyperparameter tuning with reduced feature set
            best_params, best_score = self.optimize_hyperparameters(X_current)
            
            # Compute SHAP importance with reduced feature set
            shap_importance = self.compute_shap_importance(best_params, X_current, iteration)
            
            self.history.append({
                'iteration': iteration,
                'num_features': X_current.shape[1],
                'features': X_current.columns.tolist(),
                'score': best_score,
                'best_params': best_params
            })
            
            # Determine which features to keep
            num_to_remove = max(1, int(self.step * X_current.shape[1]))
            features_to_keep = shap_importance['feature'].iloc[:-num_to_remove].tolist()
            X_current = X_current[features_to_keep]
            
            logging.info(f"Iteration {iteration} completed. Best score: {best_score}. Features remaining: {X_current.shape[1]}")

            # Save selection history
            history_df = pd.DataFrame(self.history)
            history_df.to_csv(output_file, index=False)
            
            if X_current.shape[1] <= self.min_features:
                logging.info("Minimum number of features reached. Stopping feature selection.")
                break
        
        return X_current, self.history

if __name__ == "__main__":
    from sklearn.model_selection import KFold
    import random
    import os

    # Set Random Seeds
    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)

    train = pd.read_parquet("./DATA/DATA_CLEAN/all_train_features.parquet")
    X = train.drop(columns=["UHI_Index"])
    y = train["UHI_Index"]

    # Example usage
    param_space = {
        "objective": (["regression"], 'str'),
        "metric": (["rmse"], 'str'),
        "device": (["cpu"], 'str'),
        "verbosity": (-1, -1, 'int'),
        "random_state": (SEED, SEED, 'int'),
        'n_estimators': (3000, 3000, 'int'),
        'num_leaves': (20, 256, 'int'),
        'learning_rate': (0.005, 0.3, 'float', 'log'),
        'max_depth': (5, 16, 'int'),
        'min_child_samples': (10, 100, 'int'),
        'min_child_weight': (1e-3, 10, 'float', 'log'),
        'subsample': (0.5, 1.0, 'float'),
        'colsample_bytree': (0.5, 1.0, 'float'),
        'reg_alpha': (1e-8, 10.0, 'float', 'log'),
        'reg_lambda': (1e-8, 10.0, 'float', 'log'),
    }
    
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    selector = SHAPFeatureSelector(X, y, param_space, cv, step=0.2, min_features=10, early_stopping_rounds=50, n_trials=100)
    selected_X, history = selector.select_features()

    print("Selected features:", selected_X.columns)
    print("Feature selection history saved to feature_selection_history.csv")
