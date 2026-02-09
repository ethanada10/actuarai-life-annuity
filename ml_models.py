import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


FEATURES = ["x", "m", "mprime", "n", "i", "A", "generation"]

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def evaluate(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred))
    }

def train_two_models(df: pd.DataFrame, target: str = "Pi1", test_size: float = 0.2, seed: int = 42):
    """
    Entraîne 2 modèles :
      1) Ridge (avec scaling)
      2) RandomForest
    Retourne modèles + scores + params retenus.
    """
    X = df[FEATURES].copy()
    y = df[target].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # ---- Modèle 1 : Ridge (Pipeline scaler + ridge) ----
    ridge_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge())
    ])

    ridge_grid = {
        "ridge__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]
    }

    ridge_cv = GridSearchCV(
        ridge_pipe, ridge_grid, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1
    )
    ridge_cv.fit(X_train, y_train)
    ridge_best = ridge_cv.best_estimator_

    y_pred_ridge = ridge_best.predict(X_test)
    ridge_scores = evaluate(y_test, y_pred_ridge)

    # ---- Modèle 2 : RandomForest ----
    rf = RandomForestRegressor(random_state=seed, n_jobs=-1)


    rf_grid = {
        "n_estimators": [200, 500],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    rf_cv = GridSearchCV(
        rf, rf_grid, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1
    )
    rf_cv.fit(X_train, y_train)
    rf_best = rf_cv.best_estimator_

    y_pred_rf = rf_best.predict(X_test)
    rf_scores = evaluate(y_test, y_pred_rf)

    results = {
        "target": target,
        "features": FEATURES,
        "ridge": {
            "model": ridge_best,
            "best_params": ridge_cv.best_params_,
            "scores_test": ridge_scores
        },
        "random_forest": {
            "model": rf_best,
            "best_params": rf_cv.best_params_,
            "scores_test": rf_scores
        }
    }

    return results
