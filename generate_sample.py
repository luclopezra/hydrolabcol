"""Genera un CSV de muestra realista para probar la aplicación."""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "sample_data" / "cuenca_ejemplo.csv"
OUT.parent.mkdir(exist_ok=True)

rng = np.random.default_rng(42)
dates = pd.date_range("2010-01-01", "2023-12-31", freq="D")
n = len(dates)
doy = dates.dayofyear.values

# Régimen bimodal típico andino
P_seasonal = 4 + 4 * np.sin(2 * np.pi * (doy - 80) / 365) + \
                 3 * np.sin(2 * np.pi * (doy - 260) / 365)
P = np.clip(P_seasonal + rng.gamma(0.5, 4, n) - 2, 0, None)

# Temperatura
T = 16 + 2 * np.sin(2 * np.pi * (doy - 30) / 365) + rng.normal(0, 1.2, n)

# Evapotranspiración
ET = np.clip(2.5 + 0.15 * (T - 16) + rng.normal(0, 0.4, n), 0, None)

# ONI (índice ENSO mensual repetido a diario)
months = pd.date_range("2010-01-01", "2023-12-31", freq="MS")
oni_monthly = np.cumsum(rng.normal(0, 0.25, len(months)))
oni_monthly = np.clip(oni_monthly - oni_monthly.mean(), -2.5, 2.5)
oni_daily = pd.Series(oni_monthly, index=months).reindex(dates, method="ffill").values

# Caudal observado (modelo bucket simple para que sea coherente con P/ET)
storage = 80.0
Q_obs = np.zeros(n)
for i in range(n):
    recharge = max(P[i] - ET[i] * 0.5, 0)
    storage += recharge
    baseflow = 0.04 * storage
    enso = 1 - 0.18 * oni_daily[i]
    direct = 0.5 * recharge * enso
    Q_obs[i] = max(direct + baseflow + rng.normal(0, 0.5), 0.2)
    storage = max(storage - baseflow - direct * 0.1, 10)

df = pd.DataFrame({
    "date": dates,
    "P": np.round(P, 2),
    "T": np.round(T, 2),
    "ET": np.round(ET, 2),
    "ONI": np.round(oni_daily, 2),
    "Q_obs": np.round(Q_obs, 2),
})
df.to_csv(OUT, index=False)
print(f"OK -> {OUT}  ({len(df)} filas)")
