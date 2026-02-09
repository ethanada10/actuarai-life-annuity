from mortality import load_tgf05_lx, lx_value, kpx
from actuarial import single_premium, annual_premium
from data_gen import generate_dataset
from ml_models import train_two_models


def test_mortality(lx_table):
    print("=== TEST MORTALITÉ (Étape 1) ===")
    print("lx(0,1996) =", lx_value(lx_table, 0, 1996))

    tariff_year = 2025
    x = 30
    gen = tariff_year - x  # 1995

    print(f"generation (tarif {tariff_year}, âge {x}) =", gen)
    print("kpx(30, gen, 1)  =", kpx(lx_table, 30, gen, 1))
    print("kpx(60, gen, 10) =", kpx(lx_table, 60, gen, 10))
    print()


def test_pricing(lx_table):
    print("=== TEST PRIX ACTUARIELS (Étape 2) ===")
    tariff_year = 2025
    x = 30
    generation = tariff_year - x  # 1995

    # Paramètres (dans les sets de l'énoncé)
    m = 20
    mprime = 10
    n = 20
    i = 0.02
    A = 1000

    pi1 = single_premium(lx_table, x, generation, mprime, n, i, A)
    P = annual_premium(lx_table, x, generation, m, mprime, n, i, A)

    print("x =", x, "generation =", generation)
    print("m =", m, "mprime =", mprime, "n =", n, "i =", i, "A =", A)
    print("Prime unique (Pi^(1)) =", pi1)
    print("Prime annuelle (P)    =", P)
    print()


def test_dataset(lx_table):
    print("=== TEST DATASET (Étape 3) ===")
    df = generate_dataset(lx_table, N=10, tariff_year=2025, seed=1)
    print(df.head(10))
    print("\nRésumé:")
    print(df[["Pi1", "P"]].describe())
    print()
    return df


def test_ml(lx_table):
    print("=== TEST ML (Étape 4) ===")
    # dataset plus grand pour le ML
    df_big = generate_dataset(lx_table, N=500, tariff_year=2025, seed=123)

    # ---- ML sur Pi1 ----
    res_pi1 = train_two_models(df_big, target="Pi1", seed=42)
    print("Target:", res_pi1["target"])
    print("Ridge  params:", res_pi1["ridge"]["best_params"])
    print("Ridge  scores:", res_pi1["ridge"]["scores_test"])
    print("RF     params:", res_pi1["random_forest"]["best_params"])
    print("RF     scores:", res_pi1["random_forest"]["scores_test"])
    print()

    # ---- ML sur P ----
    res_p = train_two_models(df_big, target="P", seed=42)
    print("Target:", res_p["target"])
    print("Ridge  params:", res_p["ridge"]["best_params"])
    print("Ridge  scores:", res_p["ridge"]["scores_test"])
    print("RF     params:", res_p["random_forest"]["best_params"])
    print("RF     scores:", res_p["random_forest"]["scores_test"])
    print()


def main():
    lx_table = load_tgf05_lx()

    # Étape 1
    test_mortality(lx_table)

    # Étape 2
    test_pricing(lx_table)

    # Étape 3
    test_dataset(lx_table)

    # Étape 4
    test_ml(lx_table)


if __name__ == "__main__":
    main()
