"""
Carga e inferencia de los 4 modelos regionales de IA.

INSTRUCCIONES PARA REEMPLAZAR CON TUS MODELOS REALES:
====================================================
1. Coloca tus archivos .pt o .pth en la carpeta `models/`.
2. Ajusta la clase `ModelArchitecture` abajo para que coincida con la
   arquitectura con la que entrenaste (LSTM, Transformer, MLP, etc.).
3. En `MODEL_REGISTRY` apunta cada modelo a su archivo y describe brevemente
   qué región de Colombia cubre.
4. Si los modelos esperan una normalización específica, ajústala en
   `preprocess_features` con los `mean`/`std` que guardaste durante el
   entrenamiento.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


# ---------------------------------------------------------------------------
# 1. Arquitectura  — REEMPLAZAR con la arquitectura real de tus modelos
# ---------------------------------------------------------------------------
class ModelArchitecture(nn.Module):
    """
    Placeholder LSTM. Sustituye por la arquitectura exacta con la que
    entrenaste tus modelos. Si entrenaste con `torch.save(model, ...)` en
    lugar de `torch.save(model.state_dict(), ...)`, puedes cargar el modelo
    completo con `torch.load(...)` directamente y omitir esta clase.
    """

    def __init__(self, input_size: int = 4, hidden_size: int = 64,
                 num_layers: int = 2, output_size: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out).squeeze(-1)


# ---------------------------------------------------------------------------
# 2. Registro de los 4 modelos regionales
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "Andes Norte": {
        "file":  "modelo_andes_norte.pt",
        "desc":  "Cuencas de la cordillera Oriental al norte de Bogotá.",
        "color": "#4FB3BF",
    },
    "Andes Centro-Sur": {
        "file":  "modelo_andes_centro_sur.pt",
        "desc":  "Cuencas del Macizo Colombiano y el Valle del Magdalena medio-alto.",
        "color": "#E89B5A",
    },
    "Caribe": {
        "file":  "modelo_caribe.pt",
        "desc":  "Cuencas costeras y de la Sierra Nevada de Santa Marta.",
        "color": "#C76B6B",
    },
    "Orinoquía-Amazonía": {
        "file":  "modelo_llanos.pt",
        "desc":  "Cuencas de los llanos orientales y piedemonte amazónico.",
        "color": "#7FA66B",
    },
}

REQUIRED_FEATURES = ["P", "T", "ET", "ONI"]


# ---------------------------------------------------------------------------
# 3. Carga (con caché en memoria)
# ---------------------------------------------------------------------------
_loaded = {}


def load_model(region_name: str):
    """Carga un modelo desde disco, con caché. Devuelve None si no existe el archivo."""
    if region_name in _loaded:
        return _loaded[region_name]

    cfg = MODEL_REGISTRY[region_name]
    path = MODELS_DIR / cfg["file"]

    if not path.exists():
        _loaded[region_name] = None
        return None

    model = ModelArchitecture(input_size=len(REQUIRED_FEATURES))
    state = torch.load(path, map_location="cpu")
    # Soporta tanto state_dict como modelo completo
    if isinstance(state, dict):
        model.load_state_dict(state)
    else:
        model = state
    model.eval()
    _loaded[region_name] = model
    return model


# ---------------------------------------------------------------------------
# 4. Preprocesado de features  — AJUSTAR con tu normalización real
# ---------------------------------------------------------------------------
def preprocess_features(df: pd.DataFrame) -> torch.Tensor:
    """
    Convierte el dataframe del usuario en un tensor (1, T, F) listo para PyTorch.
    Reemplaza la normalización con los mean/std que guardaste al entrenar.
    """
    arr = df[REQUIRED_FEATURES].to_numpy(dtype=np.float32)

    # Normalización por feature (placeholder — sustituye con valores reales)
    mu = arr.mean(axis=0, keepdims=True)
    sd = arr.std(axis=0, keepdims=True) + 1e-6
    arr = (arr - mu) / sd

    return torch.from_numpy(arr).unsqueeze(0)  # (1, T, 4)


# ---------------------------------------------------------------------------
# 5. Inferencia
# ---------------------------------------------------------------------------
def predict(region_name: str, df: pd.DataFrame) -> np.ndarray:
    """
    Devuelve un arreglo de caudales simulados (m³/s) del mismo largo que df.
    Si el modelo no está disponible, usa una simulación sintética para
    permitir probar la interfaz sin los archivos .pt cargados todavía.
    """
    model = load_model(region_name)

    if model is None:
        return _synthetic_prediction(df, region_name)

    x = preprocess_features(df)
    with torch.no_grad():
        y = model(x).numpy().squeeze()
    return np.clip(y, 0, None)


def _synthetic_prediction(df: pd.DataFrame, region_name: str) -> np.ndarray:
    """
    Simulación con un balance hídrico simplificado tipo bucket.
    Solo se activa cuando los modelos reales no están en disco; sirve para
    que la UI sea explorable inmediatamente.
    """
    rng = np.random.default_rng(abs(hash(region_name)) % (2**32))
    P  = df["P"].to_numpy()
    ET = df["ET"].to_numpy()
    T  = df["T"].to_numpy()
    ONI = df["ONI"].to_numpy()

    # Coeficientes "regionales" deterministas por nombre
    runoff_coef = {
        "Andes Norte":         0.55,
        "Andes Centro-Sur":    0.48,
        "Caribe":              0.32,
        "Orinoquía-Amazonía":  0.62,
    }.get(region_name, 0.45)

    # Modelo bucket muy simple con almacenamiento y memoria de 30 días
    storage = 50.0
    Q = np.zeros_like(P)
    for i, (p, et, t, oni) in enumerate(zip(P, ET, T, ONI)):
        recharge = max(p - et * 0.6, 0)
        storage += recharge
        baseflow = 0.05 * storage
        # ENSO modula salida: La Niña (ONI < 0) aumenta caudal
        enso_factor = 1 - 0.15 * oni
        direct = runoff_coef * recharge * enso_factor
        Q[i] = max(direct + baseflow, 0.1)
        storage = max(storage - baseflow - direct * 0.1, 5.0)

    # Algo de ruido para realismo visual
    Q *= 1 + rng.normal(0, 0.05, len(Q))
    return np.clip(Q, 0, None)
