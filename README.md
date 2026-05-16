# HydroLabCol

Modelado hidrológico con IA para cuencas de Colombia. Cuatro modelos PyTorch entrenados sobre regiones del país estiman el caudal diario a partir de precipitación, temperatura, evapotranspiración y el índice ONI.

> Proyecto de tesis · interfaz interactiva tipo dashboard.



---

## Formato del CSV de entrada

| date       | P    | T    | ET   | ONI  | Q_obs (opcional) |
|------------|------|------|------|------|------------------|
| 2010-01-01 | 0.00 | 16.2 | 2.40 | 0.85 | 1.20             |
| 2010-01-02 | 5.40 | 15.8 | 2.10 | 0.85 | 1.40             |

- **P** · precipitación diaria en mm
- **T** · temperatura media diaria en °C
- **ET** · evapotranspiración diaria en mm
- **ONI** · Oceanic Niño Index
- **Q_obs** · caudal observado en m³/s, opcional. 


---

## Estructura del repo

```
hydrolabcol/
├── app.py                    ← interfaz Streamlit
├── requirements.txt
├── generate_sample.py
├── .streamlit/config.toml
├── models/                   ← tus .pt aquí
├── sample_data/cuenca_ejemplo.csv
└── utils/
    ├── data_loader.py
    ├── model_loader.py
    └── metrics.py
```

---

## Métricas implementadas

| Métrica | Rango      | Ideal | Referencia                |
|---------|------------|-------|---------------------------|
| NSE     | (-∞, 1]    | 1     | Nash & Sutcliffe (1970)   |
| KGE     | (-∞, 1]    | 1     | Gupta et al. (2009)       |
| R²      | [0, 1]     | 1     | Coef. de determinación    |
| RMSE    | [0, ∞)     | 0     | m³/s                      |
| MAE     | [0, ∞)     | 0     | m³/s                      |
| PBIAS   | (-∞, ∞)    | 0     | %, Moriasi et al. (2007)  |

Clasificación NSE (Moriasi 2007): Muy bueno (>0.75), Bueno (>0.65), Satisfactorio (>0.50), Insatisfactorio (≤0.50).

---

## Citación

```
López, 2026. HydroLabCol: Modelado hidrológico
con IA para cuencas de Colombia. Tesis, Universidad Nacional de Colombia.
```
