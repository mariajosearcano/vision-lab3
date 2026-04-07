"""
Resultados - Vision por Computador
====================================
Ejecutar con:  streamlit run resultados.py
"""

import streamlit as st
import cv2
import numpy as np
import os
import glob
import re
from pathlib import Path
from PIL import Image

# ─────────────────────────────────────────────
# CONFIGURACION GLOBAL
# ─────────────────────────────────────────────
Image.MAX_IMAGE_PIXELS = None 

BASE = Path(__file__).parent / 'imgs'

DIRS = {
    'scaled':      BASE / 'img_scaled',
    'filtered':    BASE / 'img_scaled_filtered',
    'classified':  BASE / 'img_classified',
    'water':       BASE / 'img_water',
    'registered':  BASE / 'img_registered',
    'dataset':     BASE / 'img_dataset',
    'noisy':       BASE / 'img_dataset' / 'Noisy',
    'gtruth':      BASE / 'img_dataset' / 'Gtruth',
    'gt_full':     BASE / 'GroundTruth.png',
    'area':        BASE / 'area.png',
}

# ─────────────────────────────────────────────
# ESTILOS
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Resultados SAR",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background-color: #0d1117;
    color: #e6edf3;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #21262d;
}
section[data-testid="stSidebar"] * {
    color: #e6edf3 !important;
}

/* Cards */
.card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
}
.card-accent {
    border-left: 4px solid #58a6ff;
}
.card-green {
    border-left: 4px solid #3fb950;
}
.card-yellow {
    border-left: 4px solid #d29922;
}
.card-red {
    border-left: 4px solid #f85149;
}

/* Title */
.page-title {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #58a6ff;
    margin-bottom: 0.2rem;
    letter-spacing: -0.5px;
}
.page-subtitle {
    font-size: 1rem;
    color: #8b949e;
    margin-bottom: 2rem;
    font-weight: 300;
}

/* Section header */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #58a6ff;
    border-bottom: 1px solid #21262d;
    padding-bottom: 0.4rem;
    margin: 1.5rem 0 1rem 0;
}

/* Metric chips */
.metric-chip {
    display: inline-block;
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 20px;
    padding: 0.3rem 0.8rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: #58a6ff;
    margin: 0.2rem;
}

/* Conclusion box */
.conclusion {
    background: linear-gradient(135deg, #0d2137 0%, #0d1117 100%);
    border: 1px solid #58a6ff44;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-top: 1rem;
}
.conclusion h4 {
    color: #58a6ff;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.6rem;
}
.conclusion p {
    color: #c9d1d9;
    font-size: 0.95rem;
    line-height: 1.6;
    margin: 0;
}

/* Explanation box */
.explanation {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    margin-bottom: 1rem;
}
.explanation p {
    color: #8b949e;
    font-size: 0.92rem;
    line-height: 1.65;
    margin: 0;
}

/* Image caption */
.img-caption {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #8b949e;
    text-align: center;
    margin-top: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Step badge */
.step-badge {
    display: inline-block;
    background: #58a6ff22;
    border: 1px solid #58a6ff44;
    color: #58a6ff;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    margin-right: 0.5rem;
    text-transform: uppercase;
}

/* Warning / info */
.info-box {
    background: #1c2128;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    color: #8b949e;
    font-size: 0.88rem;
}

/* Hide streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Streamlit image border */
img {
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def load_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return img

def load_rgb(path):
    img = cv2.imread(str(path))
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def show_img(img, caption="", use_container_width=True):
    if img is not None:
        # Replaced use_column_width with use_container_width
        st.image(img, use_container_width=use_container_width)
        if caption:
            st.markdown(f'<div class="img-caption">{caption}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">⚠️ Imagen no encontrada</div>', unsafe_allow_html=True)

def extract_base_id(filename):
    name = Path(filename).stem
    name = name.replace('_scaled_filtered', '').replace('_scaled', '')
    return name

def extract_leading_number(text):
    match = re.match(r'^(\d+)', text)
    return int(match.group(1)) if match else None

def card(content_fn, accent='blue'):
    cls = {'blue': 'card-accent', 'green': 'card-green', 'yellow': 'card-yellow', 'red': 'card-red'}.get(accent, 'card-accent')
    st.markdown(f'<div class="card {cls}">', unsafe_allow_html=True)
    content_fn()
    st.markdown('</div>', unsafe_allow_html=True)

def get_scaled_images():
    paths = sorted(glob.glob(str(DIRS['scaled'] / '*.png')))
    return paths

def get_image_ids():
    paths = get_scaled_images()
    ids = [extract_base_id(Path(p).name) for p in paths]
    return sorted(set(ids), key=lambda x: extract_leading_number(x) or 0)

def calculate_water_pct(mask):
    return (np.count_nonzero(mask) / mask.size) * 100

def calculate_enl(image, region_size=20):
    h, w = image.shape[:2]
    best_enl = 0.0
    for y in range(0, h - region_size, region_size):
        for x in range(0, w - region_size, region_size):
            patch = image[y:y+region_size, x:x+region_size].astype(np.float64)
            mu, sigma = np.mean(patch), np.std(patch)
            if sigma > 0 and mu > 0:
                enl = (mu**2) / (sigma**2)
                if enl > best_enl:
                    best_enl = enl
    return best_enl

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem 0 1.5rem 0;">
        <div style="font-family: 'Space Mono', monospace; font-size: 1.1rem; font-weight: 700; color: #58a6ff;">
            🛰️ SAR Vision Lab
        </div>
        <div style="font-size: 0.78rem; color: #8b949e; margin-top: 0.3rem;">
            Procesamiento de imágenes radar
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navegación",
        [
            "🏠 Resumen General",
            "1️⃣ Rescalado y Filtrado",
            "2️⃣ Clasificación K-Means",
            "3️⃣ Clasificación Agua",
            "4️⃣ Creación del Dataset",
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown('<div style="font-size:0.75rem; color:#8b949e;">Pipeline SAR · Laboratorio 3</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PÁGINA: RESUMEN GENERAL
# ─────────────────────────────────────────────
if page == "🏠 Resumen General":
    st.markdown('<div class="page-title">Resultados del Laboratorio</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Procesamiento y análisis de imágenes SAR Sentinel-1</div>', unsafe_allow_html=True)

    # Cargar y mostrar la imagen del área de estudio
    img_area = load_rgb(DIRS['area'])
    show_img(img_area, "Esta es el área de estudio utilizada para todo el ejercicio.")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    scaled_count  = len(glob.glob(str(DIRS['scaled'] / '*.png')))
    water_folders = len(glob.glob(str(DIRS['water'] / '*/')))
    noisy_count   = len(glob.glob(str(DIRS['noisy'] / '*.png')))
    gtruth_count  = len(glob.glob(str(DIRS['gtruth'] / '*.png')))

    with col1:
        st.markdown(f"""
        <div class="card card-accent" style="text-align:center;">
            <div style="font-family:'Space Mono',monospace; font-size:2rem; color:#58a6ff;">{scaled_count}</div>
            <div style="font-size:0.82rem; color:#8b949e; margin-top:0.3rem;">Imágenes rescaladas</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="card card-green" style="text-align:center;">
            <div style="font-family:'Space Mono',monospace; font-size:2rem; color:#3fb950;">{water_folders}</div>
            <div style="font-size:0.82rem; color:#8b949e; margin-top:0.3rem;">Máscaras de agua</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="card card-yellow" style="text-align:center;">
            <div style="font-family:'Space Mono',monospace; font-size:2rem; color:#d29922;">{noisy_count}</div>
            <div style="font-size:0.82rem; color:#8b949e; margin-top:0.3rem;">Parches Noisy</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="card card-red" style="text-align:center;">
            <div style="font-family:'Space Mono',monospace; font-size:2rem; color:#f85149;">{gtruth_count}</div>
            <div style="font-size:0.82rem; color:#8b949e; margin-top:0.3rem;">Parches Ground Truth</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Pipeline de procesamiento</div>', unsafe_allow_html=True)

    steps = [
        ("1️⃣", "Rescalado y Filtrado", "#58a6ff",
         "Las imágenes TIFF de 16 bits (0–65535) se normalizan a uint8 (0–255) usando la media×3 como límite superior. Se aplica un filtro de mediana 5×5 para reducir el speckle."),
        ("2️⃣", "Clasificación K-Means", "#3fb950",
         "Se agrupa cada imagen en K clases de intensidad (K=2,3,4) usando K-Means sobre los valores de gris. Se compara la versión filtrada vs no filtrada."),
        ("3️⃣", "Clasificación Agua/No Agua", "#d29922",
         "Se toma la clase de menor intensidad del K-Means (agua aparece oscura en SAR por reflexión especular) y se genera una máscara binaria con el porcentaje de cobertura hídrica."),
        ("4️⃣", "Creación del Dataset", "#f85149",
         "Se registran todas las imágenes respecto a una referencia con ORB+RANSAC, se promedian para obtener el Ground Truth (menor speckle), y se recortan en parches 512×512."),
    ]

    for icon, title, color, desc in steps:
        st.markdown(f"""
        <div class="card" style="border-left: 4px solid {color}; display:flex; gap:1rem; align-items:flex-start;">
            <div style="font-size:1.6rem; line-height:1;">{icon}</div>
            <div>
                <div style="font-family:'Space Mono',monospace; font-size:0.9rem; color:{color}; font-weight:700; margin-bottom:0.4rem;">{title}</div>
                <div style="color:#8b949e; font-size:0.88rem; line-height:1.6;">{desc}</div>
            </div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PÁGINA: EJERCICIO 1
# ─────────────────────────────────────────────
elif page == "1️⃣ Rescalado y Filtrado":
    st.markdown('<div class="page-title">Ejercicio 1 · Rescalado y Filtrado</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Normalización de imágenes SAR de 16 bits + filtro de mediana</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="explanation"><p>
    Las imágenes descargadas de Sentinel-1 son TIFF de 16 bits con valores entre 0 y 65535.
    Al visualizarlas directamente aparecen completamente negras porque la mayor parte de los píxeles
    tienen valores muy bajos. El rescalado recorta todos los valores mayores a <b style="color:#58a6ff">media × 3</b>
    y normaliza al rango 0–255. Luego se aplica un <b style="color:#58a6ff">filtro de mediana 5×5</b>
    que reduce el ruido speckle preservando los bordes.
    </p></div>
    """, unsafe_allow_html=True)

    scaled_paths = sorted(glob.glob(str(DIRS['scaled'] / '*.png')))
    filtered_paths = sorted(glob.glob(str(DIRS['filtered'] / '*.png')))

    if not scaled_paths:
        st.warning("No se encontraron imágenes en img_scaled/. Ejecuta ejercicio1.py primero.")
        st.stop()

    ids = [extract_base_id(Path(p).name) for p in scaled_paths]
    selected = st.selectbox("Seleccionar imagen", ids, format_func=lambda x: x[:60])
    idx = ids.index(selected)

    img_scaled   = load_gray(scaled_paths[idx])
    img_filtered = load_gray(filtered_paths[idx]) if idx < len(filtered_paths) else None

    st.markdown('<div class="section-header">Comparación visual</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        show_img(img_scaled, "Escalada — Sin filtro")
        if img_scaled is not None:
            st.markdown(f"""
            <div style="margin-top:0.5rem;">
                <span class="metric-chip">Media: {np.mean(img_scaled):.1f}</span>
                <span class="metric-chip">Std: {np.std(img_scaled):.1f}</span>
                <span class="metric-chip">Min: {np.min(img_scaled)}</span>
                <span class="metric-chip">Max: {np.max(img_scaled)}</span>
            </div>""", unsafe_allow_html=True)
    with col2:
        show_img(img_filtered, "Escalada — Filtro mediana 5×5")
        if img_filtered is not None:
            st.markdown(f"""
            <div style="margin-top:0.5rem;">
                <span class="metric-chip">Media: {np.mean(img_filtered):.1f}</span>
                <span class="metric-chip">Std: {np.std(img_filtered):.1f}</span>
                <span class="metric-chip">Min: {np.min(img_filtered)}</span>
                <span class="metric-chip">Max: {np.max(img_filtered)}</span>
            </div>""", unsafe_allow_html=True)

    if img_scaled is not None and img_filtered is not None:
        st.markdown('<div class="section-header">Diferencia absoluta</div>', unsafe_allow_html=True)
        diff = cv2.absdiff(img_scaled, img_filtered)
        col1, col2 = st.columns([1, 2])
        with col1:
            show_img(diff, "Diferencia (escalada × 3 para visibilidad)")
        with col2:
            st.markdown(f"""
            <div class="card card-accent" style="margin-top:0;">
                <div style="font-family:'Space Mono',monospace; font-size:0.75rem; color:#58a6ff; margin-bottom:0.8rem;">ANÁLISIS DEL FILTRADO</div>
                <p style="color:#c9d1d9; font-size:0.9rem; line-height:1.7;">
                El filtro de mediana 5×5 modifica en promedio <b style="color:#58a6ff">{np.mean(diff):.2f} niveles</b> de intensidad por píxel.
                La desviación estándar baja de <b>{np.std(img_scaled):.1f}</b> a <b>{np.std(img_filtered):.1f}</b>,
                lo que indica que el filtro <b>reduce la variabilidad local</b> (speckle) manteniendo la estructura general de la imagen.
                </p>
            </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="conclusion">
        <h4>🔍 Conclusión</h4>
        <p>
        El rescalado es indispensable para poder visualizar y procesar estas imágenes SAR.
        El umbral de media×3 conserva la mayor parte del contenido informativo mientras recorta
        brillos extremos causados por estructuras metálicas o superficies muy reflectivas.
        El filtro de mediana atenúa eficazmente el speckle (ruido granular característico del SAR)
        sin borronear los contornos, a diferencia del filtro gaussiano. Esto se evidencia en la
        reducción de la desviación estándar de la imagen y en el mapa de diferencias donde los
        cambios se concentran en zonas heterogéneas (vegetación, agua, zonas urbanas ruidosas).
        </p>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PÁGINA: EJERCICIO 2
# ─────────────────────────────────────────────
elif page == "2️⃣ Clasificación K-Means":
    st.markdown('<div class="page-title">Ejercicio 2 · Clasificación K-Means</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Segmentación no supervisada por intensidad con K=2, 3 y 4 clases</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="explanation"><p>
    K-Means agrupa los píxeles de la imagen en <b style="color:#58a6ff">K clústeres</b> según su valor de intensidad.
    Para evitar problemas de memoria con imágenes grandes, el entrenamiento se realiza sobre una versión
    reducida (máx. 1M píxeles) y luego los centros aprendidos se aplican a la imagen completa mediante
    una <b style="color:#58a6ff">tabla de consulta (LUT)</b>. Se compara el resultado sobre la imagen
    filtrada vs no filtrada para evaluar el efecto del filtro en la segmentación.
    </p></div>
    """, unsafe_allow_html=True)

    classified_dir = DIRS['classified']
    image_folders  = sorted(glob.glob(str(classified_dir / '*/')))

    if not image_folders:
        st.warning("No se encontraron resultados en img_classified/. Ejecuta ejercicio2.py primero.")
        st.stop()

    folder_names = [Path(f).name for f in image_folders]
    selected_folder = st.selectbox("Seleccionar imagen", folder_names)
    folder_path = classified_dir / selected_folder

    k_tabs = st.tabs(["K = 2 clases", "K = 3 clases", "K = 4 clases"])

    for tab, k in zip(k_tabs, [2, 3, 4]):
        with tab:
            # Find image ID in folder
            all_files = list(folder_path.glob('*.png'))
            unf_clustered = next((f for f in all_files if f'_unfiltered_k{k}.png' in f.name and 'class' not in f.name), None)
            fil_clustered = next((f for f in all_files if f'_filtered_k{k}.png' in f.name and 'class' not in f.name), None)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f'<div style="font-size:0.8rem; color:#8b949e; margin-bottom:0.5rem;">SIN FILTRO — K={k}</div>', unsafe_allow_html=True)
                show_img(load_gray(unf_clustered) if unf_clustered else None, f"K={k} sin filtro")
            with col2:
                st.markdown(f'<div style="font-size:0.8rem; color:#8b949e; margin-bottom:0.5rem;">CON FILTRO — K={k}</div>', unsafe_allow_html=True)
                show_img(load_gray(fil_clustered) if fil_clustered else None, f"K={k} con filtro")

            # Show individual class masks
            st.markdown('<div class="section-header">Máscaras por clase</div>', unsafe_allow_html=True)
            mask_cols = st.columns(k)
            for cls in range(k):
                mask_unf = next((f for f in all_files if f'_unfiltered_k{k}_class_{cls}.png' in f.name), None)
                with mask_cols[cls]:
                    show_img(load_gray(mask_unf) if mask_unf else None, f"Clase {cls}")

            color_labels = {2: ["Oscuro (agua/sombra)", "Brillante (suelo/urbano)"],
                            3: ["Oscuro (agua)", "Medio (vegetación/suelo)", "Brillante (urbano/metal)"],
                            4: ["Muy oscuro (agua profunda)", "Oscuro (sombra/vegetación)", "Medio (suelo)", "Brillante (urbano)"]}
            labels = color_labels.get(k, [f"Clase {i}" for i in range(k)])

            st.markdown(f"""
            <div class="card card-green" style="margin-top:1rem;">
                <div style="font-family:'Space Mono',monospace; font-size:0.75rem; color:#3fb950; margin-bottom:0.7rem;">INTERPRETACIÓN DE CLASES — K={k}</div>
                {''.join(f'<div style="color:#c9d1d9; font-size:0.87rem; padding:0.2rem 0;">• <b style="color:#58a6ff">Clase {i}</b>: {lbl}</div>' for i, lbl in enumerate(labels))}
            </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="conclusion">
        <h4>🔍 Conclusión</h4>
        <p>
        K=2 ofrece la separación más clara entre zonas oscuras (agua) y zonas iluminadas (tierra),
        pero pierde detalle en las áreas de intensidad media. K=3 es el balance más útil para
        esta aplicación: separa agua, vegetación/suelo y zonas urbanas/metálicas. K=4 empieza
        a fragmentar clases de manera poco interpretable dado el nivel de ruido speckle presente.
        La imagen filtrada produce segmentaciones más compactas y con menos "sal y pimienta" dentro
        de cada clase, confirmando que el filtro de mediana facilita el agrupamiento. Sin embargo,
        en zonas de agua el resultado es muy similar porque el filtro preserva bien esas regiones
        homogéneas.
        </p>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PÁGINA: EJERCICIO 3
# ─────────────────────────────────────────────
elif page == "3️⃣ Clasificación Agua":
    st.markdown('<div class="page-title">Ejercicio 3 · Clasificación Agua / No Agua</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Máscara binaria de cobertura hídrica a partir de K-Means</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="explanation"><p>
    En imágenes SAR, el agua produce <b style="color:#58a6ff">reflexión especular</b>: la señal se aleja
    del sensor y prácticamente nada vuelve, resultando en píxeles muy oscuros (casi negros).
    Esto permite identificar el agua como la <b style="color:#58a6ff">clase de menor intensidad</b>
    en el resultado de K-Means. La máscara binaria resultante (blanco=agua, negro=tierra)
    permite medir el porcentaje de cobertura hídrica en cada imagen.
    </p></div>
    """, unsafe_allow_html=True)

    water_dir = DIRS['water']
    image_folders = sorted(glob.glob(str(water_dir / '*/')),
                           key=lambda x: extract_leading_number(Path(x).name) or 0)

    if not image_folders:
        st.warning("No se encontraron resultados en img_water/. Ejecuta ejercicio3.py primero.")
        st.stop()

    folder_names = [Path(f).name for f in image_folders]
    selected_folder = st.selectbox("Seleccionar imagen", folder_names)
    folder_path = water_dir / selected_folder

    all_files = list(folder_path.glob('*.png'))

    original_unf = next((f for f in all_files if 'original_unfiltered' in f.name), None)
    original_fil = next((f for f in all_files if 'original_filtered' in f.name), None)
    mask_unf     = next((f for f in all_files if 'water_mask_unfiltered' in f.name), None)
    mask_fil     = next((f for f in all_files if 'water_mask_filtered' in f.name), None)
    kmeans_unf   = next((f for f in all_files if 'kmeans' in f.name and 'unfiltered' in f.name), None)
    kmeans_fil   = next((f for f in all_files if 'kmeans' in f.name and 'filtered' in f.name), None)

    st.markdown('<div class="section-header">Original → K-Means → Máscara de agua</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        show_img(load_gray(original_unf) if original_unf else None, "Original sin filtro")
    with col2:
        show_img(load_gray(kmeans_unf) if kmeans_unf else None, "K-Means (K=3)")
    with col3:
        show_img(load_gray(mask_unf) if mask_unf else None, "Máscara agua")

    st.markdown('<div class="section-header">Comparación filtrada vs no filtrada</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    pct_unf = pct_fil = None
    with col1:
        img_mask_unf = load_gray(mask_unf) if mask_unf else None
        show_img(img_mask_unf, "Máscara — Sin filtro")
        if img_mask_unf is not None:
            pct_unf = calculate_water_pct(img_mask_unf)
            st.markdown(f'<span class="metric-chip">💧 Agua: {pct_unf:.2f}%</span>', unsafe_allow_html=True)
    with col2:
        img_mask_fil = load_gray(mask_fil) if mask_fil else None
        show_img(img_mask_fil, "Máscara — Con filtro mediana")
        if img_mask_fil is not None:
            pct_fil = calculate_water_pct(img_mask_fil)
            st.markdown(f'<span class="metric-chip">💧 Agua: {pct_fil:.2f}%</span>', unsafe_allow_html=True)

    # Summary table across all images
    st.markdown('<div class="section-header">Resumen — todas las imágenes</div>', unsafe_allow_html=True)

    rows = []
    for folder in image_folders:
        fpath = Path(folder)
        files = list(fpath.glob('*.png'))
        m_unf = next((f for f in files if 'water_mask_unfiltered' in f.name), None)
        m_fil = next((f for f in files if 'water_mask_filtered' in f.name), None)
        if m_unf and m_fil:
            iu = load_gray(m_unf)
            if_ = load_gray(m_fil)
            rows.append({
                "Imagen": fpath.name,
                "% Agua (sin filtro)": f"{calculate_water_pct(iu):.2f}%",
                "% Agua (filtrada)":   f"{calculate_water_pct(if_):.2f}%",
            })

    if rows:
        import pandas as pd
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="conclusion">
        <h4>🔍 Conclusión</h4>
        <p>
        La clasificación agua/no agua demuestra la utilidad directa de la segmentación K-Means
        en teledetección SAR. La reflexión especular del agua es un fenómeno físico consistente
        que hace que esta clase sea siempre la de menor intensidad, lo que permite una identificación
        automática y robusta. La imagen filtrada produce máscaras ligeramente más limpias con menos
        píxeles aislados clasificados como agua en zonas terrestres (ruido speckle),
        aunque el porcentaje total de cobertura hídrica varía poco entre ambas versiones.
        Este resultado podría usarse como insumo para monitoreo de cuerpos de agua,
        detección de inundaciones o análisis de cambio temporal.
        </p>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PÁGINA: EJERCICIO 4
# ─────────────────────────────────────────────
elif page == "4️⃣ Creación del Dataset":
    st.markdown('<div class="page-title">Ejercicio 4 · Creación del Dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Dataset de pares Noisy/Ground Truth para entrenamiento de filtros de speckle</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="explanation"><p>
    Siguiendo la metodología del artículo <i>"Labeled dataset for training despeckling filters for SAR imagery"</i>,
    se construye un dataset de pares de imágenes: una imagen ruidosa (speckle) y su correspondiente
    Ground Truth (imagen limpia). El GT se obtiene promediando múltiples imágenes registradas de la misma
    zona en diferentes fechas — el speckle es aleatorio, por lo que al promediar N imágenes
    su potencia se reduce en un factor <b style="color:#58a6ff">1/N</b>.
    </p></div>
    """, unsafe_allow_html=True)

    # Pipeline steps
    st.markdown('<div class="section-header">Paso 2 · Registro con ORB</div>', unsafe_allow_html=True)

    reg_paths = sorted(glob.glob(str(DIRS['registered'] / '*.png')))
    if reg_paths:
        st.markdown(f'<div style="color:#8b949e; font-size:0.85rem; margin-bottom:0.8rem;">{len(reg_paths)} imágenes registradas encontradas</div>', unsafe_allow_html=True)
        cols = st.columns(min(len(reg_paths), 4))
        for i, (col, path) in enumerate(zip(cols, reg_paths[:4])):
            with col:
                show_img(load_gray(path), f"{'Referencia' if i == 0 else f'Registrada {i}'}")
    else:
        st.markdown('<div class="info-box">⚠️ No se encontraron imágenes en img_registered/</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Paso 3 · Ground Truth (promedio multitemporal)</div>', unsafe_allow_html=True)

    gt_path = DIRS['gt_full']
    noisy_ref = reg_paths[0] if reg_paths else None

    col1, col2 = st.columns(2)
    with col1:
        img_noisy_ref = load_gray(noisy_ref) if noisy_ref else None
        show_img(img_noisy_ref, "Imagen de referencia (Noisy)")
        if img_noisy_ref is not None:
            enl_n = calculate_enl(img_noisy_ref)
            st.markdown(f'<span class="metric-chip">ENL: {enl_n:.2f}</span>', unsafe_allow_html=True)
    with col2:
        img_gt = load_gray(gt_path) if Path(gt_path).exists() else None
        show_img(img_gt, "Ground Truth (promedio)")
        if img_gt is not None:
            enl_g = calculate_enl(img_gt)
            st.markdown(f'<span class="metric-chip">ENL: {enl_g:.2f}</span>', unsafe_allow_html=True)

    if img_noisy_ref is not None and img_gt is not None:
        enl_n = calculate_enl(img_noisy_ref)
        enl_g = calculate_enl(img_gt)
        improvement = ((enl_g - enl_n) / enl_n * 100) if enl_n > 0 else 0
        st.markdown(f"""
        <div class="card card-green" style="margin-top:0.5rem;">
            <div style="font-family:'Space Mono',monospace; font-size:0.75rem; color:#3fb950; margin-bottom:0.6rem;">MÉTRICA ENL (Equivalent Number of Looks)</div>
            <p style="color:#c9d1d9; font-size:0.9rem; line-height:1.7;">
            El ENL del Ground Truth (<b style="color:#3fb950">{enl_g:.2f}</b>) es
            <b style="color:#3fb950">{improvement:.0f}% mayor</b> que el de la imagen original (<b>{enl_n:.2f}</b>).
            Un ENL más alto indica <b>menor ruido speckle</b>. El ENL ideal de una imagen completamente homogénea
            tiende a infinito; en la práctica, valores ≥ 50 indican muy buen promediado.
            </p>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Paso 4 · Parches 512×512 del dataset</div>', unsafe_allow_html=True)

    noisy_imgs  = sorted(glob.glob(str(DIRS['noisy']  / '*.png')))
    gtruth_imgs = sorted(glob.glob(str(DIRS['gtruth'] / '*.png')))

    if noisy_imgs and gtruth_imgs:
        st.markdown(f"""
        <div style="margin-bottom:1rem;">
            <span class="metric-chip">📦 Total pares: {min(len(noisy_imgs), len(gtruth_imgs))}</span>
            <span class="metric-chip">🔊 Noisy: {len(noisy_imgs)}</span>
            <span class="metric-chip">✅ Gtruth: {len(gtruth_imgs)}</span>
        </div>""", unsafe_allow_html=True)

        n_show = st.slider("Pares a mostrar", 2, min(8, len(noisy_imgs)), 4)
        step   = max(1, len(noisy_imgs) // n_show)
        sample_noisy  = [noisy_imgs[i * step]  for i in range(n_show)]
        sample_gtruth = [gtruth_imgs[i * step] for i in range(n_show)]

        st.markdown('<div style="color:#8b949e; font-size:0.78rem; margin-bottom:0.4rem; font-family:Space Mono,monospace; text-transform:uppercase; letter-spacing:1px;">NOISY (entrada)</div>', unsafe_allow_html=True)
        cols = st.columns(n_show)
        for col, path in zip(cols, sample_noisy):
            with col:
                show_img(load_gray(path), Path(path).stem[:12])

        st.markdown('<div style="color:#8b949e; font-size:0.78rem; margin-bottom:0.4rem; margin-top:0.8rem; font-family:Space Mono,monospace; text-transform:uppercase; letter-spacing:1px;">GROUND TRUTH (salida deseada)</div>', unsafe_allow_html=True)
        cols = st.columns(n_show)
        for col, path in zip(cols, sample_gtruth):
            with col:
                show_img(load_gray(path), Path(path).stem[:12])
    else:
        st.markdown('<div class="info-box">⚠️ No se encontraron parches. Ejecuta ejercicio4.py primero.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="conclusion">
        <h4>🔍 Conclusión</h4>
        <p>
        El dataset generado replica la metodología del artículo de Vásquez-Salazar et al. (2024)
        usando imágenes propias de Sentinel-1. El registro ORB con K=500 puntos alinea correctamente
        las imágenes capturadas en distintas fechas, como se verifica con la reducción del MSE
        antes y después del registro. El promediado multitemporal produce un Ground Truth con
        un ENL significativamente mayor que la imagen original, validando que el speckle fue
        efectivamente reducido. Los parches 512×512 sin solapamiento son el formato estándar
        para entrenar redes neuronales de denoising (autoencoders, U-Net, DnCNN), donde cada
        par Noisy/Gtruth sirve como ejemplo de entrenamiento supervisado.
        Con 5 imágenes el GT tiene menos calidad que con 10 (artículo original), pero el
        incremento del ENL confirma que el procedimiento funciona correctamente.
        </p>
    </div>""", unsafe_allow_html=True)