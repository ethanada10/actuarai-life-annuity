import numpy as np
import pandas as pd
from actuarial import single_premium, annual_premium

X_SET = [20, 30, 40, 50, 60]
M_SET = [1, 5, 10, 20, 30, 40]
MPRIME_SET = [0, 1, 5, 10, 20, 30, 40]
N_SET = [1, 5, 10, 20, 30, 40, 50, 60]
I_SET = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025]
A_SET = [50, 100, 200, 400, 800, 1000, 2000]

def generate_dataset(lx_table, N: int, tariff_year: int = 2025, seed: int = 42) -> pd.DataFrame:
    if not (1 <= N <= 1000):
        raise ValueError("N doit être entre 1 et 1000")

    rng = np.random.default_rng(seed)

    age_max = int(lx_table.index.max())  # normalement 121
    rows = []

    # On boucle jusqu'à obtenir N lignes valides
    tries = 0
    max_tries = 100000  # sécurité pour éviter boucle infinie

    while len(rows) < N and tries < max_tries:
        tries += 1

        x = int(rng.choice(X_SET))
        m = int(rng.choice(M_SET))
        mprime = int(rng.choice(MPRIME_SET))
        n = int(rng.choice(N_SET))
        i = float(rng.choice(I_SET))
        A = float(rng.choice(A_SET))

        generation = int(tariff_year - x)

        # ✅ contrainte table: besoin de lx(x+k) jusqu'à k = mprime+n
        if x + mprime + n > age_max:
            continue  # tirage invalide -> on rejette

        # Calculs actuariels
        pi1 = single_premium(lx_table, x, generation, mprime, n, i, A)
        P = annual_premium(lx_table, x, generation, m, mprime, n, i, A)

        rows.append({
            "x": x,
            "m": m,
            "mprime": mprime,
            "n": n,
            "i": i,
            "A": A,
            "generation": generation,
            "Pi1": pi1,
            "P": P
        })

    if len(rows) < N:
        raise RuntimeError(f"Impossible de générer {N} lignes valides (obtenu {len(rows)}).")

    return pd.DataFrame(rows)
