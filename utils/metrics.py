"""
Métricas de desempeño para modelos hidrológicos.
Referencias:
- Nash & Sutcliffe (1970) — NSE
- Gupta et al. (2009)    — KGE
- Moriasi et al. (2007)  — guías de PBIAS
"""
import numpy as np


def _clean(obs, sim):
    """Elimina NaN pareados de las dos series."""
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    mask = ~(np.isnan(obs) | np.isnan(sim))
    return obs[mask], sim[mask]


def nse(obs, sim):
    """Nash-Sutcliffe Efficiency. 1 = perfecto, 0 = igual a la media, < 0 = peor."""
    obs, sim = _clean(obs, sim)
    if len(obs) == 0:
        return np.nan
    denom = np.sum((obs - obs.mean()) ** 2)
    if denom == 0:
        return np.nan
    return 1 - np.sum((obs - sim) ** 2) / denom


def kge(obs, sim):
    """Kling-Gupta Efficiency. 1 = perfecto. Descompone en correlación, sesgo y variabilidad."""
    obs, sim = _clean(obs, sim)
    if len(obs) == 0 or obs.std() == 0 or obs.mean() == 0:
        return np.nan
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = sim.std() / obs.std()
    beta = sim.mean() / obs.mean()
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def rmse(obs, sim):
    """Root Mean Square Error en las mismas unidades del caudal (m³/s)."""
    obs, sim = _clean(obs, sim)
    if len(obs) == 0:
        return np.nan
    return float(np.sqrt(np.mean((obs - sim) ** 2)))


def mae(obs, sim):
    """Mean Absolute Error."""
    obs, sim = _clean(obs, sim)
    if len(obs) == 0:
        return np.nan
    return float(np.mean(np.abs(obs - sim)))


def pbias(obs, sim):
    """Percent Bias. Positivo = subestimación, negativo = sobreestimación. 0 = ideal."""
    obs, sim = _clean(obs, sim)
    if len(obs) == 0 or obs.sum() == 0:
        return np.nan
    return 100.0 * (obs - sim).sum() / obs.sum()


def r2(obs, sim):
    """Coeficiente de determinación R²."""
    obs, sim = _clean(obs, sim)
    if len(obs) == 0:
        return np.nan
    return float(np.corrcoef(obs, sim)[0, 1] ** 2)


def all_metrics(obs, sim):
    """Devuelve un diccionario con todas las métricas listas para mostrar."""
    return {
        "NSE":   round(nse(obs, sim), 3),
        "KGE":   round(kge(obs, sim), 3),
        "R²":    round(r2(obs, sim), 3),
        "RMSE":  round(rmse(obs, sim), 3),
        "MAE":   round(mae(obs, sim), 3),
        "PBIAS": round(pbias(obs, sim), 2),
    }


def classify_nse(value):
    """Clasificación cualitativa de NSE según Moriasi et al. (2007)."""
    if np.isnan(value):
        return "—"
    if value > 0.75:
        return "Muy bueno"
    if value > 0.65:
        return "Bueno"
    if value > 0.50:
        return "Satisfactorio"
    return "Insatisfactorio"
