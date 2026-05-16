"""
Validación y parseo del CSV que sube el usuario.

Formato esperado:
- Una columna de fecha (date, fecha, time, datetime)
- Columnas: P (mm), T (°C), ET (mm), ONI (adim.)
- Opcional: Q_obs (m³/s) para comparar con la simulación
"""

import pandas as pd

REQUIRED = ["P", "T", "ET", "ONI"]
DATE_CANDIDATES = ["date", "fecha", "time", "datetime", "Fecha", "Date"]
OBS_CANDIDATES = ["Q_obs", "Q", "caudal", "caudal_obs", "Qobs"]


class ValidationError(Exception):
    pass


def parse_csv(file) -> pd.DataFrame:
    """Lee un CSV detectando separador automáticamente."""
    try:
        df = pd.read_csv(file, sep=None, engine="python")
    except Exception as e:
        raise ValidationError(f"No se pudo leer el archivo: {e}")
    return df


def find_date_column(df: pd.DataFrame):
    for c in df.columns:
        if c in DATE_CANDIDATES or c.lower() in [x.lower() for x in DATE_CANDIDATES]:
            return c
    return None


def find_obs_column(df: pd.DataFrame):
    for c in df.columns:
        if c in OBS_CANDIDATES or c.lower() in [x.lower() for x in OBS_CANDIDATES]:
            return c
    return None


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Valida y normaliza el dataframe. Devuelve un df limpio con índice temporal."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValidationError(
            f"Faltan columnas obligatorias: {', '.join(missing)}. "
            f"El archivo debe incluir P, T, ET, ONI."
        )

    date_col = find_date_column(df)
    if date_col is None:
        raise ValidationError(
            "No se encontró columna de fecha. Usa un nombre como 'date', 'fecha', 'time' o 'datetime'."
        )

    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except Exception:
        raise ValidationError(f"No se pudo interpretar la columna '{date_col}' como fecha.")

    df = df.sort_values(date_col).set_index(date_col)

    # Tipos numéricos
    for c in REQUIRED:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    obs_col = find_obs_column(df)
    if obs_col is not None:
        df = df.rename(columns={obs_col: "Q_obs"})
        df["Q_obs"] = pd.to_numeric(df["Q_obs"], errors="coerce")

    # Chequeos básicos
    if df[REQUIRED].isna().all().any():
        raise ValidationError("Una o más columnas obligatorias están completamente vacías.")

    if len(df) < 30:
        raise ValidationError(
            f"La serie es muy corta ({len(df)} registros). Sube al menos 30 puntos."
        )

    return df


def summary(df: pd.DataFrame) -> dict:
    return {
        "n_records": len(df),
        "start": df.index.min().strftime("%Y-%m-%d"),
        "end":   df.index.max().strftime("%Y-%m-%d"),
        "has_obs": "Q_obs" in df.columns and df["Q_obs"].notna().any(),
    }
