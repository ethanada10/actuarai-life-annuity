import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES = ["x", "m", "mprime", "n", "i", "A", "generation"]

def evaluate(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse,
        "R2": float(r2_score(y_true, y_pred)),
    }

def train_fast(df: pd.DataFrame, target: str, seed: int = 42):
    X = df[FEATURES].copy()
    y = df[target].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # Ridge (alphas fixés selon tes meilleurs résultats)
    ridge_alpha = 1.0 if target == "Pi1" else 100.0
    ridge = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=ridge_alpha))])
    ridge.fit(X_train, y_train)
    pred_r = ridge.predict(X_test)
    scores_r = evaluate(y_test, pred_r)

    # RandomForest (params fixés + plus léger que 500 arbres)
    rf_params = dict(
        n_estimators=200,
        max_depth=10 if target == "Pi1" else None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=seed,
        n_jobs=-1,
    )
    rf = RandomForestRegressor(**rf_params)
    rf.fit(X_train, y_train)
    pred_f = rf.predict(X_test)
    scores_f = evaluate(y_test, pred_f)

    return {
        "target": target,
        "features": FEATURES,
        "ridge": {"model": ridge, "best_params": {"ridge__alpha": ridge_alpha}, "scores_test": scores_r},
        "random_forest": {"model": rf, "best_params": rf_params, "scores_test": scores_f},
    }
