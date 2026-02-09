import numpy as np

def v(i: float) -> float:
    """Facteur d'actualisation v = 1/(1+i)."""
    if i < -1.0:
        raise ValueError("i doit être > -100%")
    return 1.0 / (1.0 + i)

def annuity_immediate_deferred_temp(lx_table, x: int, generation: int, mprime: int, n: int, i: float) -> float:
    """
    Valeur actuelle d'une rente viagère temporaire n, différée de mprime,
    payée en fin d'année (annuity-immediate).

    a = sum_{k=mprime+1}^{mprime+n} v^k * {}_k p_x
    avec {}_k p_x = lx(x+k)/lx(x)
    """
    if n <= 0:
        return 0.0
    if mprime < 0:
        raise ValueError("mprime doit être >= 0")

    disc = v(i)

    # lx(x) pour le dénominateur des kpx
    lx_x = lx_table.loc[x, generation]
    if lx_x <= 0:
        raise ValueError(f"lx(x) <= 0 pour x={x}, gen={generation}")

    total = 0.0
    for k in range(mprime + 1, mprime + n + 1):
        # {}_k p_x
        lx_xk = lx_table.loc[x + k, generation]
        pk = lx_xk / lx_x
        total += (disc ** k) * pk

    return float(total)

def a_double_dot_temp(lx_table, x: int, generation: int, m: int, i: float) -> float:
    """
    Valeur actuelle d'une annuité à échoir temporaire m (primes en début d'année):
    ä_{x: m} = sum_{k=0}^{m-1} v^k * {}_k p_x
    """
    if m <= 0:
        return 0.0

    disc = v(i)
    lx_x = lx_table.loc[x, generation]
    if lx_x <= 0:
        raise ValueError(f"lx(x) <= 0 pour x={x}, gen={generation}")

    total = 0.0
    for k in range(0, m):
        lx_xk = lx_table.loc[x + k, generation]
        pk = lx_xk / lx_x
        total += (disc ** k) * pk

    return float(total)

def single_premium(lx_table, x: int, generation: int, mprime: int, n: int, i: float, A: float) -> float:
    """
    Prime unique nette: Pi^(1) = A * sum_{k=mprime+1}^{mprime+n} v^k * {}_k p_x
    """
    a = annuity_immediate_deferred_temp(lx_table, x, generation, mprime, n, i)
    return float(A * a)

def annual_premium(lx_table, x: int, generation: int, m: int, mprime: int, n: int, i: float, A: float) -> float:
    """
    Prime annuelle nette (m paiements en début d'année):
      P = Pi^(1) / ä_{x:m}
    """
    pi1 = single_premium(lx_table, x, generation, mprime, n, i, A)
    denom = a_double_dot_temp(lx_table, x, generation, m, i)

    if denom <= 0:
        raise ValueError("Denominateur ä_{x:m} <= 0 (m trop petit ou table invalide)")

    return float(pi1 / denom)
