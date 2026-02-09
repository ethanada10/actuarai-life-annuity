import numpy as np
import pandas as pd
from actuarial import single_premium, annual_premium

X_SET = [20, 30, 40, 50, 60]
M_SET = [1, 5, 10, 20, 30, 40]
MPRIME_SET = [0, 1, 5, 10, 20, 30, 40]
N_SET = [1, 5, 10, 20, 30, 40, 50, 60]
I_SET = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025]
A_SET = [50, 100, 200, 400, 800, 1000, 2000]

def _valid_base_combinations(age_max: int):
    """
    Pré-calcule toutes les combinaisons (x,m,mprime,n,i,A) valides
    au sens x+mprime+n <= age_max.
    """
    combos = []
    for x in X_SET:
        for m in M_SET:
            for mprime in MPRIME_SET:
                for n in N_SET:
                    if x + mprime + n <= age_max:
                        for i in I_SET:
                            for A in A_SET:
                                combos.append((x, m, mprime, n, float(i), float(A)))
    return combos

def generate_dataset(lx_table, N: int, tariff_year: int = 2025, seed: int = 42) -> pd.DataFrame:
    if not (1 <= N <= 1000):
        raise ValueError("N doit être entre 1 et 1000")

    rng = np.random.default_rng(seed)
    age_max = int(lx_table.index.max())

    combos = _valid_base_combinations(age_max)
    if len(combos) == 0:
        raise RuntimeError("Aucune combinaison valide avec les contraintes de table.")

    # Échantillonnage direct (avec remise) => pas de boucle de rejet
    idx = rng.integers(0, len(combos), size=N)

    rows = []
    for j in idx:
        x, m, mprime, n, i, A = combos[j]
        generation = int(tariff_year - x)

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

    return pd.DataFrame(rows)
