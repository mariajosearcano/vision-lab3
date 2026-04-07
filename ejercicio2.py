import cv2
import numpy as np
import os
import glob
import re


# =========================
# CONFIGURACION
# =========================
base_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'imgs')

# Carpeta con imagenes reescaladas SIN filtro
unfiltered_dir = os.path.join(base_root, 'img_scaled')

# Carpeta con imagenes reescaladas Y filtradas
filtered_dir = os.path.join(base_root, 'img_scaled_filtered')

# Carpeta de salida
output_dir = os.path.join(base_root, 'img_classified')
os.makedirs(output_dir, exist_ok=True)

# Cantidades de clases a generar
k_values = [2, 3, 4]

# Limites para evitar errores de memoria durante el entrenamiento de K-Means
max_training_pixels = 1_000_000
max_training_dimension = 2048


# =========================
# FUNCIONES AUXILIARES
# =========================
def extract_base_id(filename: str) -> str:
    """
    Convierte:
      xxx_scaled.png -> xxx
      xxx_scaled_filtered.png -> xxx
    """
    name = os.path.splitext(filename)[0]
    name = name.replace('_scaled_filtered', '')
    name = name.replace('_scaled', '')
    return name


def extract_leading_number(text: str):
    """
    Extrae el numero inicial del nombre de la imagen, por ejemplo:
      10-algo -> 10
    """
    match = re.match(r'^(\d+)', text)
    if match is None:
        return None
    return int(match.group(1))


def sort_image_id(image_id: str):
    """
    Ordena primero por el numero inicial para evitar el orden:
    1, 10, 2, 3...
    """
    leading_number = extract_leading_number(image_id)
    if leading_number is None:
        return (1, image_id)
    return (0, leading_number, image_id)


def build_image_output_dir(base_output_dir: str, image_id: str, fallback_index: int):
    """
    Crea la carpeta de salida para cada imagen:
      .../1, .../2, .../3, etc.
    """
    leading_number = extract_leading_number(image_id)
    folder_name = str(leading_number) if leading_number is not None else str(fallback_index)
    image_output_dir = os.path.join(base_output_dir, folder_name)
    os.makedirs(image_output_dir, exist_ok=True)
    return image_output_dir, folder_name


def prepare_image_for_kmeans(
    image: np.ndarray,
    max_pixels: int = max_training_pixels,
    max_dimension: int = max_training_dimension
):
    """
    Reduce la imagen solo para entrenar K-Means cuando la resolucion es muy grande.
    """
    height, width = image.shape[:2]
    total_pixels = height * width

    scale_by_pixels = np.sqrt(max_pixels / float(total_pixels))
    scale_by_dimension = min(max_dimension / float(height), max_dimension / float(width))
    scale = min(1.0, scale_by_pixels, scale_by_dimension)

    if scale >= 1.0:
        return image, 1.0

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    return resized, scale


def apply_centers_to_full_image(image: np.ndarray, centers: np.ndarray):
    """
    Clasifica la imagen completa usando centros ya entrenados.
    Se usa una tabla de consulta 0-255 para evitar crear arreglos gigantes.
    """
    ordered_centers = np.sort(centers.astype(np.float32).flatten())

    intensity_values = np.arange(256, dtype=np.float32).reshape(-1, 1)
    lut_labels = np.argmin(
        np.abs(intensity_values - ordered_centers.reshape(1, -1)),
        axis=1
    ).astype(np.uint8)

    labels_2d = lut_labels[image]

    if len(ordered_centers) == 1:
        gray_levels = np.array([0], dtype=np.uint8)
    else:
        gray_levels = np.linspace(0, 255, len(ordered_centers)).astype(np.uint8)

    clustered = gray_levels[labels_2d]

    return clustered, labels_2d, ordered_centers


def kmeans_grayscale(
    image: np.ndarray,
    training_image: np.ndarray,
    k: int,
    attempts: int = 3
):
    """
    Entrena K-Means con una imagen reducida y luego clasifica la imagen completa.
    Devuelve:
      - imagen clasificada reescalada a 0-255
      - labels 2D
      - centros
    """
    data = training_image.reshape((-1, 1)).astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.2
    )

    _compactness, _labels, centers = cv2.kmeans(
        data,
        k,
        None,
        criteria,
        attempts,
        cv2.KMEANS_PP_CENTERS
    )

    return apply_centers_to_full_image(image, centers)


def save_labels_as_individual_masks(labels_2d: np.ndarray, k: int, save_prefix: str):
    """
    Guarda una mascara por clase.
    """
    for cls in range(k):
        mask = np.where(labels_2d == cls, 255, 0).astype(np.uint8)
        cv2.imwrite(f'{save_prefix}_class_{cls}.png', mask)


# =========================
# BUSCAR PARES DE IMAGENES
# =========================
unfiltered_paths = sorted(glob.glob(os.path.join(unfiltered_dir, '*.png')))
filtered_paths = sorted(glob.glob(os.path.join(filtered_dir, '*.png')))

unfiltered_map = {
    extract_base_id(os.path.basename(p)): p
    for p in unfiltered_paths
}

filtered_map = {
    extract_base_id(os.path.basename(p)): p
    for p in filtered_paths
}

common_ids = sorted(set(unfiltered_map.keys()) & set(filtered_map.keys()), key=sort_image_id)

if not common_ids:
    raise FileNotFoundError(
        'No se encontraron pares de imagenes entre img_scaled e img_scaled_filtered.'
    )

print('Total de pares encontrados:', len(common_ids))

for pair_index, selected_id in enumerate(common_ids, start=1):
    unfiltered_path = unfiltered_map[selected_id]
    filtered_path = filtered_map[selected_id]
    image_output_dir, folder_name = build_image_output_dir(output_dir, selected_id, pair_index)

    print('\n' + '=' * 60)
    print(f'Procesando imagen {folder_name} de {len(common_ids)}')
    print('ID:', selected_id)
    print('No filtrada:', unfiltered_path)
    print('Filtrada:', filtered_path)
    print('Carpeta de salida:', image_output_dir)

    # =========================
    # CARGA DE IMAGENES
    # =========================
    img_unfiltered = cv2.imread(unfiltered_path, cv2.IMREAD_GRAYSCALE)
    img_filtered = cv2.imread(filtered_path, cv2.IMREAD_GRAYSCALE)

    if img_unfiltered is None:
        raise ValueError(f'No se pudo leer la imagen no filtrada: {unfiltered_path}')

    if img_filtered is None:
        raise ValueError(f'No se pudo leer la imagen filtrada: {filtered_path}')

    if img_unfiltered.shape != img_filtered.shape:
        raise ValueError('La imagen filtrada y la no filtrada no tienen el mismo tamano.')

    train_unfiltered, scale_unfiltered = prepare_image_for_kmeans(img_unfiltered)
    train_filtered, scale_filtered = prepare_image_for_kmeans(img_filtered)

    print('Tamano original no filtrada:', img_unfiltered.shape)
    print('Tamano usado en K-Means no filtrada:', train_unfiltered.shape)
    print('Factor de escala no filtrada:', f'{scale_unfiltered:.4f}')
    print('Tamano original filtrada:', img_filtered.shape)
    print('Tamano usado en K-Means filtrada:', train_filtered.shape)
    print('Factor de escala filtrada:', f'{scale_filtered:.4f}')

    # =========================
    # GUARDAR COPIAS BASE
    # =========================
    cv2.imwrite(
        os.path.join(image_output_dir, f'{selected_id}_original_unfiltered.png'),
        img_unfiltered
    )
    cv2.imwrite(
        os.path.join(image_output_dir, f'{selected_id}_original_filtered.png'),
        img_filtered
    )

    # =========================
    # CLUSTERING PARA CADA K
    # =========================
    for k in k_values:
        # No filtrada
        clustered_unf, labels_unf, centers_unf = kmeans_grayscale(
            img_unfiltered,
            train_unfiltered,
            k
        )
        cv2.imwrite(
            os.path.join(image_output_dir, f'{selected_id}_unfiltered_k{k}.png'),
            clustered_unf
        )
        save_labels_as_individual_masks(
            labels_unf,
            k,
            os.path.join(image_output_dir, f'{selected_id}_unfiltered_k{k}')
        )

        # Filtrada
        clustered_fil, labels_fil, centers_fil = kmeans_grayscale(
            img_filtered,
            train_filtered,
            k
        )
        cv2.imwrite(
            os.path.join(image_output_dir, f'{selected_id}_filtered_k{k}.png'),
            clustered_fil
        )
        save_labels_as_individual_masks(
            labels_fil,
            k,
            os.path.join(image_output_dir, f'{selected_id}_filtered_k{k}')
        )

        print(f'\n=== K = {k} ===')
        print('Centros no filtrada:', centers_unf)
        print('Centros filtrada   :', centers_fil)

print('\nProceso terminado.')
print('Resultados guardados en:', output_dir)