"""
Ejercicio 4 - Creacion del Dataset
===================================
Basado en el articulo: "Labeled dataset for training despeckling filters for SAR imagery"
y el repositorio: https://github.com/rubenchov/SAR_despeckling_dataset

Pipeline completo:
  1. Toma las imagenes NO filtradas (img_scaled/)
  2. Selecciona la primera imagen como referencia (base para Noisy)
  3. Registra todas las demas respecto a la referencia usando ORB
  4. Fusiona todas por promedio para generar el Ground Truth
  5. Recorta ambas (referencia y GT) en parches de 512x512

Requiere: pip install opencv-python numpy imutils
"""

import cv2
import numpy as np
import os
import glob

# =========================
# CONFIGURACION
# =========================
base_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'imgs')

# Carpeta de entrada: imagenes escaladas SIN filtro
input_dir = os.path.join(base_root, 'img_scaled')

# Carpetas de salida intermedias y finales
registered_dir = os.path.join(base_root, 'img_registered')
dataset_dir    = os.path.join(base_root, 'img_dataset')
noisy_dir      = os.path.join(dataset_dir, 'Noisy')
gtruth_dir     = os.path.join(dataset_dir, 'Gtruth')

for d in [registered_dir, noisy_dir, gtruth_dir]:
    os.makedirs(d, exist_ok=True)

# Parametros de registro ORB (K=500 segun el articulo)
MAX_FEATURES  = 500
KEEP_PERCENT  = 0.2

# Parametros de recorte
CLIP_SIZE = 512
CLIP_STEP = 512   # sin solapamiento (igual al tamanio)

# Dimension maxima de trabajo para evitar errores de memoria.
# Las imagenes Sentinel-1 originales son ~16k x 25k px.
# Se reducen a esta dimension para registro y fusion.
# Bajar a 2048 si sigue habiendo problemas de memoria.
MAX_WORKING_DIM = 4096


def resize_to_max(image: np.ndarray, max_dim: int = MAX_WORKING_DIM) -> np.ndarray:
    """Redimensiona la imagen para que su lado mayor no supere max_dim."""
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


# =========================
# FUNCIONES
# =========================

def align_image(image: np.ndarray, template: np.ndarray,
                max_features: int = MAX_FEATURES,
                keep_percent: float = KEEP_PERCENT) -> np.ndarray:
    """
    Registra 'image' respecto a 'template' usando descriptores ORB + homografia RANSAC.
    Adaptado de: https://pyimagesearch.com/2020/08/31/image-alignment-and-registration-with-opencv/
    """
    orb = cv2.ORB_create(max_features)
    kps_a, desc_a = orb.detectAndCompute(image,   None)
    kps_b, desc_b = orb.detectAndCompute(template, None)

    if desc_a is None or desc_b is None:
        print('  ADVERTENCIA: No se encontraron descriptores. Devolviendo imagen sin registrar.')
        return cv2.resize(image, (template.shape[1], template.shape[0]),
                          interpolation=cv2.INTER_AREA)

    matcher  = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches  = matcher.match(desc_a, desc_b, None)
    matches  = sorted(matches, key=lambda x: x.distance)
    keep     = max(4, int(len(matches) * keep_percent))
    matches  = matches[:keep]

    pts_a = np.zeros((len(matches), 2), dtype='float')
    pts_b = np.zeros((len(matches), 2), dtype='float')
    for i, m in enumerate(matches):
        pts_a[i] = kps_a[m.queryIdx].pt
        pts_b[i] = kps_b[m.trainIdx].pt

    H, _ = cv2.findHomography(pts_a, pts_b, cv2.RANSAC)

    if H is None:
        print('  ADVERTENCIA: Homografia no encontrada. Devolviendo imagen sin registrar.')
        return cv2.resize(image, (template.shape[1], template.shape[0]),
                          interpolation=cv2.INTER_AREA)

    h, w = template.shape[:2]
    aligned = cv2.warpPerspective(image, H, (w, h))
    return aligned


def calculate_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    a = img1.astype(np.float64)
    b = img2.astype(np.float64)
    return float(np.mean((a - b) ** 2))


def calculate_enl(image: np.ndarray, region_size: int = 20) -> float:
    """
    Calcula el Equivalent Number of Looks (ENL) en la region mas homogenea
    encontrada con ventana deslizante de region_size x region_size.
    ENL = mean^2 / variance  (ecuacion 2 del articulo)
    """
    h, w = image.shape[:2]
    best_enl = 0.0
    step = region_size

    for y in range(0, h - region_size, step):
        for x in range(0, w - region_size, step):
            patch = image[y:y + region_size, x:x + region_size].astype(np.float64)
            mu    = np.mean(patch)
            sigma = np.std(patch)
            if sigma > 0 and mu > 0:
                enl = (mu ** 2) / (sigma ** 2)
                if enl > best_enl:
                    best_enl = enl
    return best_enl


def clip_image(image: np.ndarray, output_dir: str, prefix: str,
               size: int = CLIP_SIZE, step: int = CLIP_STEP) -> int:
    """
    Recorta la imagen en parches de size x size con stride step.
    Nombre de cada parche: y_x.png  (coordenadas en la imagen original)
    """
    h, w   = image.shape[:2]
    count  = 0
    for y in range(0, h - size + 1, step):
        for x in range(0, w - size + 1, step):
            patch = image[y:y + size, x:x + size]
            name  = os.path.join(output_dir, f'{prefix}{y}_{x}.png')
            cv2.imwrite(name, patch)
            count += 1
    return count


# =========================
# PASO 1: CARGAR IMAGENES
# =========================
all_paths = sorted(glob.glob(os.path.join(input_dir, '*.png')))

if len(all_paths) < 2:
    raise FileNotFoundError(
        f'Se necesitan al menos 2 imagenes en {input_dir}.\n'
        'Ejecuta ejercicio1.py primero para generar img_scaled/.'
    )

print(f'Imagenes encontradas en img_scaled/: {len(all_paths)}')
for p in all_paths:
    print(f'  {os.path.basename(p)}')

# La primera imagen es la referencia (Noisy)
reference_path = all_paths[0]
other_paths    = all_paths[1:]

print(f'\nImagen de referencia (Noisy): {os.path.basename(reference_path)}')
template_raw = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)

if template_raw is None:
    raise ValueError(f'No se pudo leer: {reference_path}')

print(f'Tamano original de referencia: {template_raw.shape[0]} x {template_raw.shape[1]}')
template = resize_to_max(template_raw)
h_ref, w_ref = template.shape
print(f'Tamano de trabajo (reducido) : {h_ref} x {w_ref}')


# =========================
# PASO 2: REGISTRO
# =========================
print('\n' + '=' * 60)
print('PASO 2 - Registro de imagenes con ORB')
print('=' * 60)

# La referencia se copia sin registrar (es la base)
ref_registered_path = os.path.join(registered_dir, os.path.basename(reference_path))
cv2.imwrite(ref_registered_path, template)
registered_images = [template.copy()]

for imgpath in other_paths:
    name  = os.path.basename(imgpath)
    image = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f'  ERROR leyendo {name}. Saltando.')
        continue

    # Reducir al tamano de trabajo antes de cualquier operacion
    image = resize_to_max(image)

    # Redimensionar al tamano de referencia antes de registrar
    image_resized = cv2.resize(image, (w_ref, h_ref), interpolation=cv2.INTER_AREA)

    print(f'\n  Registrando: {name}')
    mse_before = calculate_mse(template, image_resized)

    aligned = align_image(image_resized, template)

    mse_after  = calculate_mse(template, aligned)
    print(f'  MSE antes del registro : {mse_before:.2f}')
    print(f'  MSE despues del registro: {mse_after:.2f}')

    # Guardar imagen registrada
    save_name = name.replace('.png', '_registered.png')
    save_path = os.path.join(registered_dir, save_name)
    cv2.imwrite(save_path, aligned)
    registered_images.append(aligned)

print(f'\nTotal de imagenes registradas (incluyendo referencia): {len(registered_images)}')


# =========================
# PASO 3: FUSION (GROUND TRUTH)
# =========================
print('\n' + '=' * 60)
print('PASO 3 - Fusion multitemporal (Ground Truth por promedio)')
print('=' * 60)

accumulator = np.zeros((h_ref, w_ref), dtype=np.float32)

for img in registered_images:
    accumulator = cv2.add(accumulator, img.astype(np.float32))

n_images = float(len(registered_images))
avg_gt   = (accumulator / n_images).astype(np.uint8)

gt_path = os.path.join(base_root, 'GroundTruth.png')
cv2.imwrite(gt_path, avg_gt)

# Calcular ENL (ecuacion 2 del articulo)
enl_noisy = calculate_enl(template)
enl_gt    = calculate_enl(avg_gt)

print(f'ENL imagen original (Noisy) : {enl_noisy:.3f}')
print(f'ENL Ground Truth (promedio) : {enl_gt:.3f}')
print(f'(Un ENL mayor indica menos ruido speckle)')
print(f'Ground Truth guardado en: {gt_path}')


# =========================
# PASO 4: RECORTE 512x512
# =========================
print('\n' + '=' * 60)
print(f'PASO 4 - Recorte en parches de {CLIP_SIZE}x{CLIP_SIZE} px')
print('=' * 60)

n_noisy  = clip_image(template, noisy_dir,  prefix='', size=CLIP_SIZE, step=CLIP_STEP)
n_gtruth = clip_image(avg_gt,   gtruth_dir, prefix='', size=CLIP_SIZE, step=CLIP_STEP)

print(f'Parches Noisy  guardados en: {noisy_dir}  ({n_noisy} imagenes)')
print(f'Parches Gtruth guardados en: {gtruth_dir}  ({n_gtruth} imagenes)')

if n_noisy != n_gtruth:
    print('ADVERTENCIA: el numero de parches no coincide entre Noisy y Gtruth.')

# =========================
# RESUMEN FINAL
# =========================
print('\n' + '=' * 60)
print('RESUMEN DEL DATASET GENERADO')
print('=' * 60)
print(f'Imagenes de entrada      : {len(all_paths)}')
print(f'Imagen de referencia     : {os.path.basename(reference_path)}')
print(f'Tamano de parches        : {CLIP_SIZE} x {CLIP_SIZE} px')
print(f'Stride de recorte        : {CLIP_STEP} px (sin solapamiento)')
print(f'Total de pares generados : {n_noisy}')
print(f'ENL Noisy / GT           : {enl_noisy:.3f} / {enl_gt:.3f}')
print(f'\nEstructura de salida:')
print(f'  imgs/')
print(f'  ├── img_registered/   <- imagenes alineadas (intermedias)')
print(f'  ├── GroundTruth.png   <- imagen GT completa (promedio)')
print(f'  └── img_dataset/')
print(f'      ├── Noisy/        <- {n_noisy} parches ruidosos')
print(f'      └── Gtruth/       <- {n_gtruth} parches ground truth')
print('\nProceso terminado.')