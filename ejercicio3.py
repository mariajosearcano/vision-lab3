import cv2
import numpy as np
import os
import glob
import re


# =========================
# CONFIGURACION
# =========================
base_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'imgs')

unfiltered_dir = os.path.join(base_root, 'img_scaled')
filtered_dir = os.path.join(base_root, 'img_scaled_filtered')
output_dir = os.path.join(base_root, 'img_water')
os.makedirs(output_dir, exist_ok=True)

# Numero de clases K a usar (elige el que mejor separe agua del resto)
K = 3

# Limites para K-Means (igual que ejercicio2)
max_training_pixels = 1_000_000
max_training_dimension = 2048


# =========================
# FUNCIONES AUXILIARES
# (mismas que ejercicio2)
# =========================
def extract_base_id(filename: str) -> str:
    name = os.path.splitext(filename)[0]
    name = name.replace('_scaled_filtered', '')
    name = name.replace('_scaled', '')
    return name


def extract_leading_number(text: str):
    match = re.match(r'^(\d+)', text)
    if match is None:
        return None
    return int(match.group(1))


def sort_image_id(image_id: str):
    leading_number = extract_leading_number(image_id)
    if leading_number is None:
        return (1, image_id)
    return (0, leading_number, image_id)


def prepare_image_for_kmeans(image, max_pixels=max_training_pixels, max_dimension=max_training_dimension):
    height, width = image.shape[:2]
    total_pixels = height * width
    scale_by_pixels = np.sqrt(max_pixels / float(total_pixels))
    scale_by_dimension = min(max_dimension / float(height), max_dimension / float(width))
    scale = min(1.0, scale_by_pixels, scale_by_dimension)
    if scale >= 1.0:
        return image, 1.0
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, scale


def kmeans_grayscale(image, training_image, k, attempts=3):
    """Entrena K-Means y clasifica la imagen completa. Devuelve labels 2D y centros ordenados."""
    data = training_image.reshape((-1, 1)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _compactness, _labels, centers = cv2.kmeans(
        data, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS
    )

    # Ordenar centros de menor a mayor intensidad
    ordered_centers = np.sort(centers.flatten())

    # Asignar cada pixel al centro mas cercano
    intensity_values = np.arange(256, dtype=np.float32).reshape(-1, 1)
    lut_labels = np.argmin(
        np.abs(intensity_values - ordered_centers.reshape(1, -1)),
        axis=1
    ).astype(np.uint8)

    labels_2d = lut_labels[image]
    return labels_2d, ordered_centers


# =========================
# FUNCION PRINCIPAL: MASCARA DE AGUA
# =========================
def extract_water_mask(image: np.ndarray, labels_2d: np.ndarray, ordered_centers: np.ndarray):
    """
    En imagenes SAR/radar, el agua aparece oscura (baja reflectancia).
    La clase con el centro de menor intensidad se asigna como 'agua'.
    Devuelve: mascara binaria (255=agua, 0=no agua) y el indice de clase agua.
    """
    water_class_index = 0  # Clase con centro mas bajo = agua
    water_mask = np.where(labels_2d == water_class_index, 255, 0).astype(np.uint8)
    return water_mask, water_class_index


def calculate_water_percentage(water_mask: np.ndarray) -> float:
    total_pixels = water_mask.size
    water_pixels = np.count_nonzero(water_mask)
    return (water_pixels / total_pixels) * 100.0


# =========================
# BUSCAR PARES DE IMAGENES
# =========================
unfiltered_paths = sorted(glob.glob(os.path.join(unfiltered_dir, '*.png')))
filtered_paths = sorted(glob.glob(os.path.join(filtered_dir, '*.png')))

unfiltered_map = {extract_base_id(os.path.basename(p)): p for p in unfiltered_paths}
filtered_map = {extract_base_id(os.path.basename(p)): p for p in filtered_paths}

common_ids = sorted(
    set(unfiltered_map.keys()) & set(filtered_map.keys()),
    key=sort_image_id
)

if not common_ids:
    raise FileNotFoundError('No se encontraron pares de imagenes.')

print(f'Total de pares encontrados: {len(common_ids)}')
print(f'Usando K = {K} clases\n')

# Acumular resultados para resumen final
summary = []

for pair_index, selected_id in enumerate(common_ids, start=1):
    unfiltered_path = unfiltered_map[selected_id]
    filtered_path = filtered_map[selected_id]

    leading = extract_leading_number(selected_id)
    folder_name = str(leading) if leading is not None else str(pair_index)
    image_output_dir = os.path.join(output_dir, folder_name)
    os.makedirs(image_output_dir, exist_ok=True)

    print('=' * 60)
    print(f'Imagen {folder_name}/{len(common_ids)} | ID: {selected_id}')

    # --- Carga ---
    img_unf = cv2.imread(unfiltered_path, cv2.IMREAD_GRAYSCALE)
    img_fil = cv2.imread(filtered_path, cv2.IMREAD_GRAYSCALE)

    if img_unf is None or img_fil is None:
        print(f'  ERROR: No se pudo leer la imagen. Saltando...')
        continue

    train_unf, _ = prepare_image_for_kmeans(img_unf)
    train_fil, _ = prepare_image_for_kmeans(img_fil)

    # --- K-Means ---
    labels_unf, centers_unf = kmeans_grayscale(img_unf, train_unf, K)
    labels_fil, centers_fil = kmeans_grayscale(img_fil, train_fil, K)

    # --- Mascaras de agua ---
    mask_unf, water_idx_unf = extract_water_mask(img_unf, labels_unf, centers_unf)
    mask_fil, water_idx_fil = extract_water_mask(img_fil, labels_fil, centers_fil)

    # --- Porcentaje de agua ---
    pct_unf = calculate_water_percentage(mask_unf)
    pct_fil = calculate_water_percentage(mask_fil)

    print(f'  Centros no filtrada : {np.round(centers_unf, 2)}  -> clase agua: {water_idx_unf} (centro={centers_unf[water_idx_unf]:.2f})')
    print(f'  Centros filtrada    : {np.round(centers_fil, 2)}  -> clase agua: {water_idx_fil} (centro={centers_fil[water_idx_fil]:.2f})')
    print(f'  Porcentaje agua (sin filtro) : {pct_unf:.2f}%')
    print(f'  Porcentaje agua (filtrada)   : {pct_fil:.2f}%')

    # --- Guardar resultados ---
    # Imagen original
    cv2.imwrite(os.path.join(image_output_dir, f'{selected_id}_original_unfiltered.png'), img_unf)
    cv2.imwrite(os.path.join(image_output_dir, f'{selected_id}_original_filtered.png'), img_fil)

    # Imagen K-Means coloreada (para referencia visual de las clases)
    gray_levels = np.linspace(0, 255, K).astype(np.uint8)
    clustered_unf = gray_levels[labels_unf]
    clustered_fil = gray_levels[labels_fil]
    cv2.imwrite(os.path.join(image_output_dir, f'{selected_id}_kmeans_k{K}_unfiltered.png'), clustered_unf)
    cv2.imwrite(os.path.join(image_output_dir, f'{selected_id}_kmeans_k{K}_filtered.png'), clustered_fil)

    # Mascara binaria agua / no agua
    cv2.imwrite(os.path.join(image_output_dir, f'{selected_id}_water_mask_unfiltered.png'), mask_unf)
    cv2.imwrite(os.path.join(image_output_dir, f'{selected_id}_water_mask_filtered.png'), mask_fil)

    summary.append({
        'id': selected_id,
        'folder': folder_name,
        'centers_unf': centers_unf,
        'centers_fil': centers_fil,
        'pct_agua_unf': pct_unf,
        'pct_agua_fil': pct_fil,
    })

# =========================
# RESUMEN FINAL
# =========================
print('\n' + '=' * 60)
print(f'RESUMEN - K={K}')
print('=' * 60)
print(f'{"ID":<20} {"% Agua sin filtro":>20} {"% Agua filtrada":>18}')
print('-' * 60)
for row in summary:
    print(f'{row["id"]:<20} {row["pct_agua_unf"]:>19.2f}% {row["pct_agua_fil"]:>17.2f}%')

print('\nProceso terminado.')
print('Resultados guardados en:', output_dir)