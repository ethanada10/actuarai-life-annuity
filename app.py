import streamlit as st
import pandas as pd

from mortality import load_tgf05_lx
from actuarial import single_premium, annual_premium
from data_gen import generate_dataset
from ml_models import train_two_models

# Sets de l'énoncé (pour UI)
X_SET = [20, 30, 40, 50, 60]
M_SET = [1, 5, 10, 20, 30, 40]
MPRIME_SET = [0, 1, 5, 10, 20, 30, 40]
N_SET = [1, 5, 10, 20, 30, 40, 50, 60]
I_SET = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025]
A_SET = [50, 100, 200, 400, 800, 1000, 2000]


st.set_page_config(page_title="ActuarAI - Life Annuity", layout="wide")

st.title("ActuarAI — Life Annuity (Temporary & Deferred)")
st.caption("Actuarial pricing (TGF05) vs ML models (Ridge & RandomForest). Dataset generated on-the-fly.")

@st.cache_resource
def get_lx_table():
    return load_tgf05_lx()

@st.cache_resource
def train_models_cached(N, seed, tariff_year):
    """
    Entraîne 2 modèles sur Pi1 et 2 modèles sur P.
    Cache pour éviter de retrainer à chaque slider move.
    """
    lx_table = get_lx_table()
    df = generate_dataset(lx_table, N=N, tariff_year=tariff_year, seed=seed)

    res_pi1 = train_two_models(df, target="Pi1", seed=42)
    res_p = train_two_models(df, target="P", seed=42)

    return df, res_pi1, res_p


with st.sidebar:
    st.header("1) Dataset (ML training)")
    N = st.slider("Dataset size N (1–1000)", min_value=50, max_value=1000, value=500, step=50)
    seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=123, step=1)
    tariff_year = st.selectbox("Tariff year (for generation = year - x)", options=[2025], index=0)

    st.divider()
    st.header("2) Contract parameters (query)")
    x = st.selectbox("Age x", options=X_SET, index=1)
    m = st.selectbox("Premium payments m (beginning of each year)", options=M_SET, index=3)
    mprime = st.selectbox("Deferral m' (years)", options=MPRIME_SET, index=3)
    n = st.selectbox("Term n (years of annuity payments)", options=N_SET, index=3)
    i = st.selectbox("Technical rate i", options=I_SET, index=4, format_func=lambda z: f"{100*z:.1f}%")
    A = st.selectbox("Annuity amount A (end of year)", options=A_SET, index=5)

    st.divider()
    train_button = st.button("🚀 Generate dataset & Train models", type="primary")


# Guardrail table limits (avoid x+m'+n beyond age max)
lx_table = get_lx_table()
age_max = int(lx_table.index.max())
if x + mprime + n > age_max:
    st.warning(f"Invalid combination for mortality table: x+m'+n = {x+mprime+n} > {age_max}. "
               "Please reduce n or m' or choose smaller x.")

# Training section
if train_button:
    with st.spinner("Generating dataset & training models..."):
        df, res_pi1, res_p = train_models_cached(N=N, seed=seed, tariff_year=tariff_year)

    st.success("Models trained ✅ (cached). You can now change contract inputs without retraining.")

    # Store in session_state so we can use after reruns
    st.session_state["df"] = df
    st.session_state["res_pi1"] = res_pi1
    st.session_state["res_p"] = res_p

# If already trained in session
df = st.session_state.get("df")
res_pi1 = st.session_state.get("res_pi1")
res_p = st.session_state.get("res_p")

if df is None or res_pi1 is None or res_p is None:
    st.info("Click **Generate dataset & Train models** in the sidebar to start.")
    st.stop()

# Query row
generation = tariff_year - x
query = pd.DataFrame([{
    "x": x, "m": m, "mprime": mprime, "n": n, "i": float(i), "A": float(A), "generation": int(generation)
}])

# Actuarial computations
pi1_act = single_premium(lx_table, x, generation, mprime, n, float(i), float(A))
p_act = annual_premium(lx_table, x, generation, m, mprime, n, float(i), float(A))

# ML predictions
ridge_pi1 = res_pi1["ridge"]["model"].predict(query[res_pi1["features"]])[0]
rf_pi1 = res_pi1["random_forest"]["model"].predict(query[res_pi1["features"]])[0]

ridge_p = res_p["ridge"]["model"].predict(query[res_p["features"]])[0]
rf_p = res_p["random_forest"]["model"].predict(query[res_p["features"]])[0]


# ===== Layout results =====
col1, col2 = st.columns(2)

with col1:
    st.subheader("Actuarial vs ML — Single Premium (Pi1)")
    out_pi1 = pd.DataFrame({
        "Method": ["Actuarial", "Ridge", "RandomForest"],
        "Pi1": [pi1_act, ridge_pi1, rf_pi1],
        "Abs error vs Actuarial": [0.0, abs(ridge_pi1 - pi1_act), abs(rf_pi1 - pi1_act)],
        "Rel error vs Actuarial": [0.0, (ridge_pi1 - pi1_act) / pi1_act if pi1_act != 0 else None,
                                   (rf_pi1 - pi1_act) / pi1_act if pi1_act != 0 else None],
    })
    st.dataframe(out_pi1, use_container_width=True)

    st.caption("Pi1 is the net single premium: actuarial present value of deferred temporary life annuity payments.")

with col2:
    st.subheader("Actuarial vs ML — Annual Premium (P)")
    out_p = pd.DataFrame({
        "Method": ["Actuarial", "Ridge", "RandomForest"],
        "P": [p_act, ridge_p, rf_p],
        "Abs error vs Actuarial": [0.0, abs(ridge_p - p_act), abs(rf_p - p_act)],
        "Rel error vs Actuarial": [0.0, (ridge_p - p_act) / p_act if p_act != 0 else None,
                                   (rf_p - p_act) / p_act if p_act != 0 else None],
    })
    st.dataframe(out_p, use_container_width=True)

    st.caption("P is the net annual premium: Pi1 divided by ä_{x:m} (premiums in advance).")


st.divider()

col3, col4 = st.columns(2)

with col3:
    st.subheader("Model scores (test set)")
    scores = pd.DataFrame([
        {"Target": "Pi1", "Model": "Ridge", **res_pi1["ridge"]["scores_test"]},
        {"Target": "Pi1", "Model": "RandomForest", **res_pi1["random_forest"]["scores_test"]},
        {"Target": "P", "Model": "Ridge", **res_p["ridge"]["scores_test"]},
        {"Target": "P", "Model": "RandomForest", **res_p["random_forest"]["scores_test"]},
    ])
    st.dataframe(scores, use_container_width=True)

with col4:
    st.subheader("Best hyperparameters")
    hp = pd.DataFrame([
        {"Target": "Pi1", "Model": "Ridge", "Best params": str(res_pi1["ridge"]["best_params"])},
        {"Target": "Pi1", "Model": "RandomForest", "Best params": str(res_pi1["random_forest"]["best_params"])},
        {"Target": "P", "Model": "Ridge", "Best params": str(res_p["ridge"]["best_params"])},
        {"Target": "P", "Model": "RandomForest", "Best params": str(res_p["random_forest"]["best_params"])},
    ])
    st.dataframe(hp, use_container_width=True)

st.divider()

st.subheader("Generated dataset preview")
st.dataframe(df.head(25), use_container_width=True)

st.caption("Dataset columns: x, m, mprime, n, i, A, generation + actuarial targets Pi1 and P.")
