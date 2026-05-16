"""
HydroLabCol — Modelado hidrológico regional con IA para cuencas de Colombia
Tesis · Streamlit Community Cloud
"""
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from streamlit_folium import st_folium

from utils import data_loader, metrics
from utils.model_loader import MODEL_REGISTRY, REQUIRED_FEATURES, predict

# ---------------------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="HydroLabCol · Modelado de Caudales con IA",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
SAMPLE_CSV = ROOT / "sample_data" / "cuenca_ejemplo.csv"

# ---------------------------------------------------------------------------
# CSS — identidad visual editorial científica con paleta andina
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@400;500;600;700;900&family=JetBrains+Mono:wght@400;500&family=Inter:wght@400;500;600&display=swap');

    :root {
        --bg-deep:    #0E1E2B;
        --bg-card:    #162A3A;
        --bg-elev:    #1E364A;
        --accent-1:   #4FB3BF;   /* azul caudal */
        --accent-2:   #E89B5A;   /* ocre cuenca */
        --accent-3:   #7FA66B;   /* verde andino */
        --accent-4:   #C76B6B;   /* terracota Caribe */
        --text-1:     #E8EDF2;
        --text-2:     #9AAEC0;
        --line:       #2B4258;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    h1, h2, h3, h4 {
        font-family: 'Fraunces', serif !important;
        letter-spacing: -0.02em;
    }

    h1 { font-weight: 900 !important; }
    h2 { font-weight: 600 !important; }

    .hero-title {
        font-family: 'Fraunces', serif;
        font-size: 3.4rem;
        font-weight: 900;
        line-height: 1.0;
        letter-spacing: -0.035em;
        background: linear-gradient(135deg, var(--text-1) 0%, var(--accent-1) 70%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.25em;
        color: var(--accent-1);
        margin-bottom: 0.4rem;
    }
    .hero-lead {
        font-family: 'Fraunces', serif;
        font-size: 1.15rem;
        color: var(--text-2);
        font-style: italic;
        max-width: 720px;
        line-height: 1.5;
    }

    /* Sidebar: pasos numerados con barra lateral */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0A1620 0%, #0E1E2B 100%);
        border-right: 1px solid var(--line);
    }
    .step-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.2em;
        color: var(--accent-1);
        text-transform: uppercase;
        margin-top: 1.4rem;
        margin-bottom: 0.3rem;
        padding-left: 0.5rem;
        border-left: 2px solid var(--accent-1);
    }

    /* Cards de métricas */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin: 1rem 0 1.5rem 0;
    }
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--line);
        border-radius: 4px;
        padding: 14px 18px;
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 3px;
        background: var(--accent-1);
    }
    .metric-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        color: var(--text-2);
        margin-bottom: 4px;
    }
    .metric-value {
        font-family: 'Fraunces', serif;
        font-size: 1.85rem;
        font-weight: 600;
        color: var(--text-1);
        line-height: 1;
    }
    .metric-unit {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: var(--text-2);
        margin-left: 4px;
    }
    .metric-card.good::before    { background: var(--accent-3); }
    .metric-card.warn::before    { background: var(--accent-2); }
    .metric-card.bad::before     { background: var(--accent-4); }

    /* Etiqueta de región/modelo */
    .region-pill {
        display: inline-block;
        padding: 4px 10px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        border: 1px solid var(--accent-1);
        color: var(--accent-1);
        border-radius: 2px;
        margin-bottom: 0.6rem;
    }

    /* Section divider */
    .section-rule {
        border: 0;
        height: 1px;
        background: var(--line);
        margin: 2rem 0 1.2rem 0;
    }
    .section-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: var(--accent-2);
    }

    /* Botones */
    .stButton > button {
        font-family: 'JetBrains Mono', monospace !important;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.75rem !important;
        font-weight: 500 !important;
        border-radius: 2px !important;
        border: 1px solid var(--accent-1) !important;
        background: transparent !important;
        color: var(--accent-1) !important;
        padding: 0.55rem 1.2rem !important;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: var(--accent-1) !important;
        color: var(--bg-deep) !important;
    }
    .stDownloadButton > button {
        font-family: 'JetBrains Mono', monospace !important;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.7rem !important;
        background: var(--accent-3) !important;
        color: var(--bg-deep) !important;
        border: none !important;
    }

    /* Hide default streamlit chrome */
    #MainMenu  { visibility: hidden; }
    footer     { visibility: hidden; }
    header     { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Estado de sesión
# ---------------------------------------------------------------------------
for key, default in [
    ("data", None),
    ("results", None),
    ("region", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<div style='font-family:Fraunces,serif;font-size:1.6rem;font-weight:700;"
        "letter-spacing:-0.02em;color:var(--text-1);'>"
        "Hydro<span style='color:var(--accent-1)'>Lab</span>Col</div>"
        "<div style='font-family:JetBrains Mono,monospace;font-size:0.65rem;"
        "letter-spacing:0.2em;color:var(--text-2);text-transform:uppercase;"
        "margin-bottom:1.5rem;'>Tesis · IA Regional</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="step-label">01 · Modelo Regional</div>', unsafe_allow_html=True)
    region = st.selectbox(
        " ",
        options=list(MODEL_REGISTRY.keys()),
        label_visibility="collapsed",
        key="region_select",
    )
    st.caption(MODEL_REGISTRY[region]["desc"])

    st.markdown('<div class="step-label">02 · Datos de Entrada</div>', unsafe_allow_html=True)
    upload_mode = st.radio(
        " ",
        ["Subir CSV propio", "Usar cuenca de ejemplo"],
        label_visibility="collapsed",
    )

    uploaded = None
    if upload_mode == "Subir CSV propio":
        uploaded = st.file_uploader(
            "Archivo CSV",
            type=["csv"],
            help=f"Debe contener columnas: fecha, {', '.join(REQUIRED_FEATURES)}, y opcionalmente Q_obs.",
        )
    else:
        st.info("Cuenca andina sintética · 2010 – 2023 · datos diarios.")

    st.markdown('<div class="step-label">03 · Ejecutar</div>', unsafe_allow_html=True)
    run = st.button("▸ Simular Caudal", use_container_width=True)

    st.markdown('<div class="step-label">04 · Descargar</div>', unsafe_allow_html=True)
    download_placeholder = st.empty()


# ---------------------------------------------------------------------------
# Cabecera
# ---------------------------------------------------------------------------
st.markdown('<div class="hero-sub">Modelado Hidrológico · Inteligencia Artificial Regional</div>',
            unsafe_allow_html=True)
st.markdown('<div class="hero-title">Caudales para cuencas de Colombia,<br/>predichos con IA.</div>',
            unsafe_allow_html=True)
st.markdown(
    '<p class="hero-lead">Cuatro modelos regionales entrenados sobre cuencas '
    'colombianas estiman el caudal diario a partir de precipitación, temperatura, '
    'evapotranspiración y el índice ONI. Sube los datos de tu cuenca y obtén '
    'series simuladas con métricas de desempeño.</p>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cargar datos al activar Run
# ---------------------------------------------------------------------------
def _load_data():
    if upload_mode == "Subir CSV propio":
        if uploaded is None:
            st.error("Sube un archivo CSV o cambia a la cuenca de ejemplo.")
            return None
        df_raw = data_loader.parse_csv(uploaded)
    else:
        df_raw = pd.read_csv(SAMPLE_CSV)

    try:
        return data_loader.validate(df_raw)
    except data_loader.ValidationError as e:
        st.error(f"Datos inválidos: {e}")
        return None


if run:
    with st.spinner("Procesando series y ejecutando modelo…"):
        df = _load_data()
        if df is not None:
            sim = predict(region, df)
            df_out = df.copy()
            df_out["Q_sim"] = sim
            st.session_state.data = df_out
            st.session_state.region = region
            st.session_state.results = True


# ---------------------------------------------------------------------------
# Vista principal
# ---------------------------------------------------------------------------
left, right = st.columns([1.05, 1.0], gap="large")

# -------- Mapa (siempre visible) ---------------------------------------------
with left:
    st.markdown('<hr class="section-rule"/>', unsafe_allow_html=True)
    st.markdown('<span class="section-tag">▸ Localización · Colombia</span>', unsafe_allow_html=True)
    st.markdown("### Cuencas de referencia")

    m = folium.Map(
        location=[4.5, -74.0],
        zoom_start=5,
        tiles="CartoDB dark_matter",
    )
    region_centers = {
        "Andes Norte":         (7.0, -72.5),
        "Andes Centro-Sur":    (3.0, -76.0),
        "Caribe":             (10.5, -74.0),
        "Orinoquía-Amazonía":  (3.5, -71.5),
    }
    for name, (lat, lon) in region_centers.items():
        is_selected = (st.session_state.region or region) == name
        folium.CircleMarker(
            location=[lat, lon],
            radius=14 if is_selected else 8,
            color=MODEL_REGISTRY[name]["color"],
            fill=True,
            fill_opacity=0.85 if is_selected else 0.4,
            weight=3 if is_selected else 1,
            popup=folium.Popup(f"<b>{name}</b><br/>{MODEL_REGISTRY[name]['desc']}", max_width=240),
            tooltip=name,
        ).add_to(m)

    st_folium(m, height=420, use_container_width=True, returned_objects=[])


# -------- Panel de resultados ------------------------------------------------
with right:
    st.markdown('<hr class="section-rule"/>', unsafe_allow_html=True)

    if not st.session_state.results:
        st.markdown('<span class="section-tag">▸ Resultados</span>', unsafe_allow_html=True)
        st.markdown("### Esperando ejecución del modelo")
        st.markdown(
            "<div style='color:var(--text-2);font-family:Fraunces,serif;font-style:italic;"
            "font-size:1.05rem;line-height:1.6;margin-top:0.8rem;'>"
            "Selecciona un modelo regional, carga los datos meteorológicos de la cuenca "
            "y pulsa <strong style='color:var(--accent-1);font-style:normal;'>Simular Caudal</strong> "
            "en el panel lateral.</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='margin-top:1.2rem;padding:14px 18px;border:1px dashed var(--line);"
            "border-radius:4px;font-family:JetBrains Mono,monospace;font-size:0.78rem;"
            "color:var(--text-2);'>"
            f"<strong style='color:var(--accent-1);'>Formato esperado del CSV</strong><br/><br/>"
            "date · P · T · ET · ONI · [Q_obs]<br/><br/>"
            "<span style='color:var(--text-2);font-family:Inter,sans-serif;'>"
            "P en mm, T en °C, ET en mm, ONI adimensional. La columna Q_obs es opcional "
            "y, si está, se usa para calcular métricas de desempeño."
            "</span></div>",
            unsafe_allow_html=True,
        )
    else:
        df = st.session_state.data
        sel_region = st.session_state.region
        info = data_loader.summary(df)

        st.markdown(
            f'<span class="section-tag">▸ Resultados</span>',
            unsafe_allow_html=True,
        )
        st.markdown(f'<div class="region-pill">Modelo · {sel_region}</div>',
                    unsafe_allow_html=True)
        st.markdown(f"### {info['n_records']:,} días simulados")
        st.caption(f"Período: {info['start']} → {info['end']}")

        # Métricas si hay observado
        if info["has_obs"]:
            m_dict = metrics.all_metrics(df["Q_obs"].values, df["Q_sim"].values)
            cls_nse = metrics.classify_nse(m_dict["NSE"])

            def _q(metric, value, unit=""):
                # color del borde según calidad
                cls = ""
                if metric == "NSE":
                    if value > 0.65: cls = "good"
                    elif value > 0.5: cls = "warn"
                    else: cls = "bad"
                if metric == "KGE":
                    if value > 0.7: cls = "good"
                    elif value > 0.5: cls = "warn"
                    else: cls = "bad"
                return (
                    f'<div class="metric-card {cls}">'
                    f'<div class="metric-label">{metric}</div>'
                    f'<div class="metric-value">{value}<span class="metric-unit">{unit}</span></div>'
                    f'</div>'
                )

            st.markdown(
                '<div class="metric-grid">'
                + _q("NSE",   m_dict["NSE"])
                + _q("KGE",   m_dict["KGE"])
                + _q("R²",    m_dict["R²"])
                + _q("RMSE",  m_dict["RMSE"], "m³/s")
                + _q("MAE",   m_dict["MAE"],  "m³/s")
                + _q("PBIAS", m_dict["PBIAS"], "%")
                + '</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='font-family:JetBrains Mono,monospace;font-size:0.72rem;"
                f"color:var(--text-2);text-transform:uppercase;letter-spacing:0.15em;'>"
                f"Calidad NSE → <span style='color:var(--accent-1);'>{cls_nse}</span></div>",
                unsafe_allow_html=True,
            )
        else:
            # Sin obs: mostrar estadísticas del simulado
            q = df["Q_sim"]
            stats = {
                "Q medio":  round(q.mean(), 2),
                "Q máx":    round(q.max(), 2),
                "Q mín":    round(q.min(), 2),
                "Q5":       round(np.percentile(q, 95), 2),
                "Q95":      round(np.percentile(q, 5), 2),
                "Volumen":  round(q.sum() * 86400 / 1e6, 1),
            }
            grid = '<div class="metric-grid">'
            for k, v in stats.items():
                unit = "Mm³" if k == "Volumen" else "m³/s"
                grid += (f'<div class="metric-card">'
                         f'<div class="metric-label">{k}</div>'
                         f'<div class="metric-value">{v}<span class="metric-unit">{unit}</span></div>'
                         f'</div>')
            grid += '</div>'
            st.markdown(grid, unsafe_allow_html=True)
            st.caption("Sube un CSV con columna Q_obs para obtener métricas de desempeño.")


# ---------------------------------------------------------------------------
# Gráficos (debajo, todo el ancho)
# ---------------------------------------------------------------------------
if st.session_state.results:
    df = st.session_state.data
    sel_region = st.session_state.region
    has_obs = "Q_obs" in df.columns and df["Q_obs"].notna().any()
    color = MODEL_REGISTRY[sel_region]["color"]

    st.markdown('<hr class="section-rule"/>', unsafe_allow_html=True)
    st.markdown('<span class="section-tag">▸ Series temporales</span>', unsafe_allow_html=True)
    st.markdown("### Hidrograma simulado")

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.3, 0.7], vertical_spacing=0.04,
        subplot_titles=("", ""),
    )
    # Precipitación arriba (invertida)
    fig.add_trace(
        go.Bar(x=df.index, y=df["P"], marker_color="#4FB3BF",
               opacity=0.55, name="Precipitación (mm)"),
        row=1, col=1,
    )
    fig.update_yaxes(autorange="reversed", title_text="P (mm)", row=1, col=1)

    # Caudales abajo
    if has_obs:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["Q_obs"],
                       line=dict(color="#9AAEC0", width=1.2),
                       name="Q observado"),
            row=2, col=1,
        )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["Q_sim"],
                   line=dict(color=color, width=1.6),
                   name="Q simulado"),
        row=2, col=1,
    )
    fig.update_yaxes(title_text="Q (m³/s)", row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=11, color="#E8EDF2"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=460,
        legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center",
                    bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#2B4258", zerolinecolor="#2B4258")
    fig.update_yaxes(gridcolor="#2B4258", zerolinecolor="#2B4258")
    st.plotly_chart(fig, use_container_width=True)

    # Curva de duración + scatter
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<span class="section-tag">▸ Curva de duración</span>', unsafe_allow_html=True)
        fdc_fig = go.Figure()
        sim_sorted = np.sort(df["Q_sim"].dropna())[::-1]
        prob = 100 * np.arange(1, len(sim_sorted) + 1) / (len(sim_sorted) + 1)
        fdc_fig.add_trace(go.Scatter(x=prob, y=sim_sorted, line=dict(color=color, width=2),
                                     name="Simulado", fill="tozeroy",
                                     fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.18)"))
        if has_obs:
            obs_sorted = np.sort(df["Q_obs"].dropna())[::-1]
            prob_obs = 100 * np.arange(1, len(obs_sorted) + 1) / (len(obs_sorted) + 1)
            fdc_fig.add_trace(go.Scatter(x=prob_obs, y=obs_sorted,
                                         line=dict(color="#9AAEC0", width=1.5, dash="dot"),
                                         name="Observado"))
        fdc_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", size=11, color="#E8EDF2"),
            xaxis_title="Probabilidad de excedencia (%)",
            yaxis_title="Q (m³/s)", yaxis_type="log",
            margin=dict(l=10, r=10, t=20, b=10),
            height=320,
            legend=dict(orientation="h", y=-0.18),
        )
        fdc_fig.update_xaxes(gridcolor="#2B4258")
        fdc_fig.update_yaxes(gridcolor="#2B4258")
        st.plotly_chart(fdc_fig, use_container_width=True)

    with col2:
        if has_obs:
            st.markdown('<span class="section-tag">▸ Observado vs simulado</span>',
                        unsafe_allow_html=True)
            sc = go.Figure()
            sc.add_trace(go.Scatter(
                x=df["Q_obs"], y=df["Q_sim"],
                mode="markers",
                marker=dict(color=color, size=4, opacity=0.5,
                            line=dict(width=0)),
                name="Pares (obs, sim)",
            ))
            mx = max(df["Q_obs"].max(), df["Q_sim"].max())
            sc.add_trace(go.Scatter(x=[0, mx], y=[0, mx],
                                    line=dict(color="#9AAEC0", dash="dash", width=1),
                                    name="1:1"))
            sc.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", size=11, color="#E8EDF2"),
                xaxis_title="Q observado (m³/s)",
                yaxis_title="Q simulado (m³/s)",
                margin=dict(l=10, r=10, t=20, b=10),
                height=320,
                legend=dict(orientation="h", y=-0.18),
            )
            sc.update_xaxes(gridcolor="#2B4258")
            sc.update_yaxes(gridcolor="#2B4258")
            st.plotly_chart(sc, use_container_width=True)
        else:
            st.markdown('<span class="section-tag">▸ Régimen mensual</span>',
                        unsafe_allow_html=True)
            monthly = df["Q_sim"].groupby(df.index.month).mean()
            mfig = go.Figure(go.Bar(
                x=["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"],
                y=monthly.values, marker_color=color,
            ))
            mfig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", size=11, color="#E8EDF2"),
                yaxis_title="Q medio (m³/s)",
                margin=dict(l=10, r=10, t=20, b=10),
                height=320,
            )
            mfig.update_xaxes(gridcolor="#2B4258")
            mfig.update_yaxes(gridcolor="#2B4258")
            st.plotly_chart(mfig, use_container_width=True)

    # Botón de descarga en sidebar
    csv = df.to_csv().encode("utf-8")
    download_placeholder.download_button(
        "▾ Descargar CSV",
        data=csv,
        file_name=f"hydrolabcol_{sel_region.replace(' ','_')}.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown('<hr class="section-rule"/>', unsafe_allow_html=True)
st.markdown(
    "<div style='font-family:JetBrains Mono,monospace;font-size:0.7rem;"
    "color:var(--text-2);letter-spacing:0.15em;text-transform:uppercase;"
    "padding:1rem 0;'>"
    "HydroLabCol · Tesis · Modelado regional con IA para cuencas colombianas"
    "</div>",
    unsafe_allow_html=True,
)
