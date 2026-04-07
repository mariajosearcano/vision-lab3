import cv2
import numpy as np
import os
import glob

basepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'imgs', 'img') + os.sep  # Path of folder where the images are located
imgpaths = sorted(glob.glob(os.path.join(basepath, '*.tiff')))  # Image names and extension

# Crear carpetas de salida si no existen
imgs_root = os.path.dirname(basepath.rstrip(os.sep))  # sube un nivel: imgs/img -> imgs/
scaled_dir = os.path.join(imgs_root, 'img_scaled')
filtered_dir = os.path.join(imgs_root, 'img_scaled_filtered')

os.makedirs(scaled_dir, exist_ok=True)
os.makedirs(filtered_dir, exist_ok=True)

for imgpath in imgpaths:

    img = cv2.imread(imgpath, cv2.IMREAD_UNCHANGED)  # Load image

    # Información de la imagen
    print('Processing:', os.path.basename(imgpath))
    print('Image shape: ', img.shape)
    print('Image type: ', img.dtype)
    print('Image max pixel value: ', np.max(img))
    print('Image mean pixel value: ', np.mean(img))
    print('Image min pixel value: ', np.min(img))

    img2 = img.astype(np.single)  # Change datatype to real values
    escala_display = np.mean(img2) * 3.0  # Mean value times 3
    min_val = np.min(img2)

    # Escalamiento
    img2[img2 > escala_display] = escala_display
    img2[img2 < min_val] = 0
    img3 = 255.0 * (img2 / escala_display)
    img4 = img3.astype(np.uint8)  # Imagen escalada (SIN filtro)

    # ---------- GUARDAR IMAGEN ESCALADA SIN FILTRO ----------
    filename = os.path.basename(imgpath)
    name = filename.replace('.tiff', '')
    imgpath_scaled = os.path.join(scaled_dir, name + '_scaled.png')
    cv2.imwrite(imgpath_scaled, img4)

    # ---------- FILTRADO ----------
    img5 = cv2.medianBlur(img4, 5)

    # ---------- GUARDAR IMAGEN FILTRADA ----------
    imgpath_filtered = os.path.join(filtered_dir, name + '_scaled_filtered.png')
    cv2.imwrite(imgpath_filtered, img5)

    print('Shape of final image: ', img5.shape)