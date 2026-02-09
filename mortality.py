from pathlib import Path
import pandas as pd
import numpy as np

# ==========
# Chemins
# ==========
BASE_DIR = Path(__file__).resolve().parent
MORTALITY_PATH = BASE_DIR / "TGF05-TGH05.xls"


# ==========
# Lecture TGF05 -> table lx
# ==========
def load_tgf05_lx(path=MORTALITY_PATH, sheet_name="TGF05"):
    """
    Lit la feuille TGF05 du fichier Excel et retourne une DataFrame lx :
      - index = ages (0..121)
      - colonnes = generation (année de naissance)
      - valeurs = lx (nb de survivants)
    Hypothèse structure (comme ton fichier) :
      - ligne 1 : années (à partir de la colonne 1)
      - colonne 0 : âge (à partir de la ligne 2)
    """
    xl = pd.ExcelFile(path)
    df = xl.parse(sheet_name, header=None)

    # Années de naissance sur la ligne 1, colonnes à partir de 1
    years = df.iloc[1, 1:].dropna().astype(int).values

    # Âges sur la colonne 0, lignes à partir de 2
    ages = df.iloc[2:, 0].dropna().astype(int).values

    # lx = bloc de données
    lx = df.iloc[2:, 1:].copy()
    lx.index = ages
    lx.columns = years

    # Convertir en numeric
    lx = lx.apply(pd.to_numeric, errors="coerce")

    return lx

# ==========
# Accès lx(age, generation)
# ==========
def lx_value(lx_table, age, generation):
    """
    Retourne lx pour un âge et une génération.
    """
    if generation not in lx_table.columns:
        raise ValueError(
            f"generation {generation} absente. "
            f"Dispo: {int(lx_table.columns.min())}..{int(lx_table.columns.max())}"
        )
    if age not in lx_table.index:
        raise ValueError(
            f"age {age} absent. "
            f"Dispo: {int(lx_table.index.min())}..{int(lx_table.index.max())}"
        )

    val = lx_table.loc[age, generation]
    if pd.isna(val):
        raise ValueError(f"lx manquant (NA) pour age={age}, gen={generation}")
    return float(val)

# ==========
# Survie {}_k p_x
# ==========
def kpx(lx_table, x, generation, k):
    """
    {}_k p_x = lx(x+k)/lx(x)
    """
    if k < 0:
        raise ValueError("k doit être >= 0")

    l_x = lx_value(lx_table, x, generation)
    l_xk = lx_value(lx_table, x + k, generation)

    if l_x <= 0:
        raise ValueError(f"lx(x) <= 0 pour x={x}, gen={generation}")

    return l_xk / l_x

# ==========
# Utilitaire : lister les feuilles Excel (si besoin)
# ==========
def list_sheets(path=MORTALITY_PATH):
    xl = pd.ExcelFile(path)
    return xl.sheet_names
