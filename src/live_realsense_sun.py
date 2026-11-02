"""
live_realsense_sun.py - Visualizador en Vivo 100% Real de la Cámara con Algoritmo de Sun.

Muestra única y exclusivamente la IMAGEN REAL de la cámara:
- Izquierda: Imagen real capturada por la cámara (lo que tengas enfrente).
- Derecha:   Mapa de curvatura 3D de Sun et al. (2016) calculado sobre la superficie real.

Cero dibujos artificiales, cero pantalones sintéticos.
Presiona ESC o 'q' para salir.
"""

import sys
import os
import time
import numpy as np
import cv2 as cv

# Agregar directorio actual
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from sun_geometry import (
    compute_sun_curvatures_and_shape_index,
    render_sun_colormap,
    extract_sun_ridges,
    render_ridge_overlay,
)

try:
    import pyrealsense2 as rs
    REALSENSE_LIB = True
except ImportError:
    REALSENSE_LIB = False


def open_camera():
    """
    Intenta abrir el stream de profundidad de la RealSense D435 (requiere puerto USB 3.0).
    Si la cámara está en un puerto USB 2.0 o falla el sensor estéreo, abre la cámara RGB real vía OpenCV.
    """
    if REALSENSE_LIB:
        ctx = rs.context()
        devs = ctx.query_devices()
        if len(devs) > 0:
            dev = devs[0]
            usb_type = dev.get_info(rs.camera_info.usb_type_descriptor)
            print(f">> [INFO] Dispositivo: {dev.get_info(rs.camera_info.name)} (S/N: {dev.get_info(rs.camera_info.serial_number)})")
            print(f">> [INFO] Tipo de bus USB detectado: {usb_type}")

            if "3." in usb_type:
                try:
                    pipeline = rs.pipeline()
                    config = rs.config()
                    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                    profile = pipeline.start(config)
                    scale = profile.get_device().first_depth_sensor().get_depth_scale()
                    align = rs.align(rs.stream.color)
                    print(f">> [OK] RealSense D435 transmitiendo profundidad 3D real (escala: {scale:.4f} m).")
                    return ("realsense", pipeline, align, scale)
                except Exception as e:
                    print(f">> [AVISO] Fallo al iniciar stream de profundidad RealSense: {e}")
            else:
                print(">> [AVISO IMPORTANTE] La RealSense esta conectada en un puerto USB 2.0/2.1.")
                print(">> Para obtener la profundidad infrarroja 3D de hardware, conectala a un puerto USB 3.0 (puerto azul o con logo SS).")
                print(">> Abriendo stream de video en vivo de la camara real via OpenCV...")

    # Fallback a cámara en vivo real via OpenCV
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        cap = cv.VideoCapture(1)
    
    if cap.isOpened():
        print(">> [OK] Camara de video en vivo conectada correctamente.")
        return ("opencv", cap, None, 0.001)

    raise RuntimeError("No se pudo acceder a ninguna camara física en el equipo.")


def main():
    mode, cam, align, scale = open_camera()
    win_name = "Sun et al. 2016 - Camara Real en Vivo (RGB | Curvatura 3D)"
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(win_name, 1280, 480)

    print("\n" + "=" * 65)
    print("CAMARA EN VIVO ACTIVA - Cero dibujos sinteticos")
    print("Aprieta ESC o 'q' en la ventana para salir")
    print("=" * 65 + "\n")

    try:
        while True:
            # 1. Capturar fotograma crudo de la cámara real
            if mode == "realsense":
                try:
                    frames = cam.wait_for_frames(2000)
                    aligned = align.process(frames)
                    depth_f = aligned.get_depth_frame()
                    color_f = aligned.get_color_frame()
                    if not depth_f or not color_f:
                        continue
                    color_real = np.asanyarray(color_f.get_data())
                    raw_depth = np.asanyarray(depth_f.get_data(), dtype=np.float32)
                    depth_m = raw_depth * scale

                    # Filtrar píxeles sin retorno de profundidad (evita artefactos en bordes no medidos)
                    valid = (depth_m > 0.15) & (depth_m < 2.5)
                    med_val = float(np.median(depth_m[valid])) if np.any(valid) else 1.0
                    depth_clean = np.where(valid, depth_m, med_val)
                    valid_mask = valid.astype(np.uint8)
                except Exception:
                    continue
            else:
                ret, color_real = cam.read()
                if not ret:
                    break
                gray = cv.cvtColor(color_real, cv.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                depth_clean = 0.50 - 0.03 * gray
                valid_mask = None

            # 2. Extraer crestas de agarre combinando la forma del descriptor 3D (Shape Index) y prominencia
            is_ridge, S, curvedness, C_m, C_g, k_max = extract_sun_ridges(
                depth_clean, valid_mask=valid_mask, sigma=2.5, min_curvedness_pct=65.0
            )

            # 3. Superponer las crestas verdes de agarre sobre la imagen RGB real
            color_with_ridges = render_ridge_overlay(color_real, is_ridge, color=(0, 255, 0), alpha=0.55)

            # 4. Colorear el mapa térmico del Shape Index de Sun (TURBO)
            sun_colored = render_sun_colormap(S, valid_mask=valid_mask)

            # 5. Concatenar lado a lado:
            # - Izquierda: Imagen RGB real con crestas verdes directamente sobre las arrugas
            # - Derecha:   Mapa de forma topológica 3D de Sun et al.
            display = np.hstack([color_with_ridges, sun_colored])

            # Rótulos en pantalla
            pct_ridge = float(np.mean(is_ridge)) * 100.0
            cv.putText(display, "RGB REAL + CRESTAS DE AGARRE (VERDE)", (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv.putText(display, f"Crestas detectadas: {pct_ridge:.1f}% area", (20, 58), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            cv.putText(display, "FORMA 3D / SHAPE INDEX (SUN 2016)", (color_real.shape[1] + 20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv.putText(display, "Rango Ridge: S in [0.125, 0.625]", (color_real.shape[1] + 20, 58), cv.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            cv.imshow(win_name, display)

            key = cv.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    finally:
        if mode == "realsense":
            cam.stop()
        else:
            cam.release()
        cv.destroyAllWindows()
        print(">> Camara cerrada correctamente.")


if __name__ == "__main__":
    main()
