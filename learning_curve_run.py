import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mortality import load_tgf05_lx
from data_gen import generate_dataset
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES = ["x", "m", "mprime", "n", "i", "A", "generation"]

def eval_scores(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse,
        "R2": float(r2_score(y_true, y_pred)),
    }

def fit_and_score_fixed(X_train, y_train, X_test, y_test, ridge_alpha=1.0, rf_params=None, seed=42):
    if rf_params is None:
        rf_params = dict(n_estimators=200, max_depth=10, min_samples_split=2, min_samples_leaf=1)

    ridge = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=ridge_alpha))])
    ridge.fit(X_train, y_train)
    pred_r = ridge.predict(X_test)
    sc_r = eval_scores(y_test, pred_r)

    rf = RandomForestRegressor(random_state=seed, n_jobs=-1, **rf_params)
    rf.fit(X_train, y_train)
    pred_f = rf.predict(X_test)
    sc_f = eval_scores(y_test, pred_f)

    return sc_r, sc_f

def run_learning_curve_fixed_test(
    target="Pi1",
    N_list=(50,100,200,500,1000),
    big_N=1000,
    ridge_alpha=1.0,
    rf_params=None,
    seed_data=123,
    seed_split=42
):
    lx = load_tgf05_lx()
    df_big = generate_dataset(lx, N=big_N, tariff_year=2025, seed=seed_data)

    X = df_big[FEATURES].copy()
    y = df_big[target].copy()

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed_split
    )

    rng = np.random.default_rng(seed_split)
    rows = []

    for N in N_list:
        idx = rng.choice(len(X_train_full), size=N, replace=False)
        X_train = X_train_full.iloc[idx]
        y_train = y_train_full.iloc[idx]

        sc_r, sc_f = fit_and_score_fixed(
            X_train, y_train, X_test, y_test,
            ridge_alpha=ridge_alpha,
            rf_params=rf_params,
            seed=seed_split
        )

        rows.append({"N": N, "model": "Ridge", **sc_r})
        rows.append({"N": N, "model": "RandomForest", **sc_f})

    return pd.DataFrame(rows)

def plot_metric(df, target, metric="RMSE"):
    plt.figure()
    for model in df["model"].unique():
        s = df[df["model"] == model].sort_values("N")
        plt.plot(s["N"], s[metric], marker="o", label=model)
    plt.xlabel("Taille du dataset (N)")
    plt.ylabel(metric)
    plt.title(f"Learning curve (test fixe) - {target} ({metric})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    rf_params = dict(n_estimators=200, max_depth=10, min_samples_split=2, min_samples_leaf=1)

    # Pi1
    df_pi1 = run_learning_curve_fixed_test(target="Pi1", ridge_alpha=1.0, rf_params=rf_params)
    print(df_pi1)
    plot_metric(df_pi1, "Pi1", "RMSE")
    plot_metric(df_pi1, "Pi1", "R2")

    # P (si tu veux)
    df_p = run_learning_curve_fixed_test(target="P", ridge_alpha=100.0, rf_params=rf_params)
    print(df_p)
    plot_metric(df_p, "P", "RMSE")
    plot_metric(df_p, "P", "R2")
