from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import ExtraTreesRegressor
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import random
import ast

def main():
    # Load data
    train = pd.read_parquet("./DATA/DATA_CLEAN/all_train_features.parquet")
    X = train.drop(columns=["UHI_Index"])
    y = train["UHI_Index"]
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Load selected features from previous feature selection
    selected_features = pd.read_csv("./feature_selection/lgbm/feature_selection_history.csv")
    selected_features = ast.literal_eval(selected_features.loc[selected_features["num_features"] == 12]["features"].item())

    # Ensure selected features are not sampled again
    features = set(X.columns) - set(selected_features)
    
    history = []
    max_features_to_evaluate = 100
    patience = 3
    no_improvement_count = 0
    failed_features = set()
    best_score = -np.inf

    tqdm.write(f"Starting Forward Feature Selection from scratch.")
    pbar = tqdm(desc="Feature Selection", total=len(features))

    # Use a local copy of selected features to modify dynamically
    current_selected_features = selected_features.copy()

    try:
        while features:
            best_feature = None
            best_new_score = best_score

            # Exclude previously failed features from sampling
            available_features = list(features - failed_features)
            if not available_features:
                tqdm.write("No more new features to evaluate. Stopping because all features have been exhausted.")
                break

            sampled_features = np.random.choice(available_features, 
                                                min(max_features_to_evaluate, len(available_features)), 
                                                replace=False)

            # Evaluate sampled features
            for i, feature in enumerate(sampled_features):
                pbar.set_description(f"Evaluating: {feature} | {i+1}/{len(sampled_features)}")

                X_temp = train[current_selected_features + [feature]]
                model = ExtraTreesRegressor(n_estimators=1_000, n_jobs=-1, random_state=42)
                score = cross_val_score(model, X_temp, y, cv=cv, scoring="r2").mean()

                if score > best_new_score:
                    best_new_score = score
                    best_feature = feature

            # If we find a better feature, reset failed features
            if best_feature is not None:
                current_selected_features.append(best_feature)
                features.remove(best_feature)
                best_score = best_new_score
                history.append({"features": list(current_selected_features), "score": best_score})

                tqdm.write(f"Added {best_feature} | New R^2 Score: {best_score:.4f}")
                pbar.update(1)
                no_improvement_count = 0  # Reset patience counter
                failed_features.clear()  # Reset failed features since the context has changed
            else:
                no_improvement_count += 1
                # Only add features that did not improve to failed_features
                failed_features.update(set(sampled_features) - {best_feature} if best_feature else sampled_features)
                tqdm.write(f"No improvement found in this round. (Patience {no_improvement_count}/{patience})")

            # Stop if no improvement for `patience` consecutive rounds
            if no_improvement_count >= patience:
                tqdm.write(f"Stopping due to lack of improvements after {patience} consecutive rounds.")
                failed_features.clear()  # Reset and allow new attempts
                no_improvement_count = 0  # Reset patience counter

            # Save history after each iteration
            if history:
                history_df = pd.DataFrame(history)
                history_df.to_csv("./forward_feature_selection/selection_history5.csv", index=False)

    finally:
        # Ensure history is saved before exiting
        if history:
            history_df = pd.DataFrame(history)
            history_df.to_csv("./forward_feature_selection/selection_history5.csv", index=False)
        tqdm.write("Final history saved before exiting.")

    pbar.close()

if __name__ == "__main__":
    # Set Random Seeds for Reproducibility
    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)

    # Run Main Function
    main()
