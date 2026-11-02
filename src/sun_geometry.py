"""
sun_geometry.py - Geometría Diferencial 3D y Shape Index (Sun et al. 2016).

Implementación fiel y vectorizada de las Ecuaciones 1-4 de:
"Robot Vision Architecture for Autonomous Clothes Manipulation" (Sun et al., 2016).

Calcula la curvatura de superficie de Monge Z = f(x, y) a partir del mapa de profundidad
de la cámara Intel RealSense D435 y cuantiza las regiones en crestas (ridges) y valles (ruts).
"""

from typing import Tuple, Optional
import numpy as np
from scipy.ndimage import gaussian_filter
import cv2 as cv

EPSILON: float = 1e-7


def compute_sun_curvatures_and_shape_index(
    depth_m: np.ndarray,
    sigma: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula curvaturas locales y Shape Index para un mapa de profundidad.
    
    Args:
        depth_m: Mapa de profundidad en metros # shape: (H, W), dtype float32
        sigma: Desviación típica del filtro gaussiano para regularizar ruido de sensor
        
    Returns:
        C_m: Curvatura media (H, W) en m^-1
        C_g: Curvatura gaussiana (H, W) en m^-2
        k_max: Curvatura principal máxima (H, W) en m^-1
        k_min: Curvatura principal mínima (H, W) en m^-1
        S: Koenderink Shape Index adimensional en [-1.0, 1.0] # shape: (H, W)
    """
    assert depth_m.ndim == 2, f"Se esperaba tensor 2D de profundidad, recibido {depth_m.shape}"
    depth = depth_m.astype(np.float32)

    # Centrado respecto a la mediana para eliminar fuga numérica de DC del kernel gaussiano discreto
    z_baseline = float(np.median(depth))
    depth_zero_mean = depth - z_baseline

    # 1. Derivadas parciales de primer y segundo orden mediante kernel gaussiano
    fx = gaussian_filter(depth_zero_mean, sigma=sigma, order=(0, 1))   # dZ / dx # shape: (H, W)
    fy = gaussian_filter(depth_zero_mean, sigma=sigma, order=(1, 0))   # dZ / dy # shape: (H, W)
    fxx = gaussian_filter(depth_zero_mean, sigma=sigma, order=(0, 2))  # d²Z / dx² # shape: (H, W)
    fyy = gaussian_filter(depth_zero_mean, sigma=sigma, order=(2, 0))  # d²Z / dy² # shape: (H, W)
    fxy = gaussian_filter(depth_zero_mean, sigma=sigma, order=(1, 1))  # d²Z / dxdy # shape: (H, W)

    # 2. Curvatura Media C_m (Ecuación 1 Sun et al. 2016)
    c_m_top = ((1.0 + fy**2) * fxx) + ((1.0 + fx**2) * fyy) - (2.0 * fx * fy * fxy)
    c_m_bot = 2.0 * (1.0 + fx**2 + fy**2)**1.5 + EPSILON
    C_m = c_m_top / c_m_bot

    # 3. Curvatura Gaussiana C_g (Ecuación 2 Sun et al. 2016)
    c_g_top = (fxx * fyy - fxy**2)
    c_g_bot = (1.0 + fx**2 + fy**2)**2 + EPSILON
    C_g = c_g_top / c_g_bot

    # 4. Curvaturas Principales k_max y k_min (Ecuación 3 Sun et al. 2016)
    discriminante = np.clip(C_m**2 - C_g, 0.0, None)
    raiz = np.sqrt(discriminante)
    k_max = C_m + raiz
    k_min = C_m - raiz

    # 5. Koenderink Shape Index S in [-1.0, 1.0] (Ecuación 4 Sun et al. 2016)
    num = k_min + k_max
    den = k_max - k_min
    S = (2.0 / np.pi) * np.arctan2(num, den + EPSILON)

    # Guardia de estabilidad numérica: regiones planas no deben amplificar residuo a +-1.0
    flat_regions = (np.abs(num) < 1e-6) & (np.abs(den) < 1e-6)
    S = np.where(flat_regions, 0.0, S)

    return C_m, C_g, k_max, k_min, S


def get_sun_ridge_mask(
    S: np.ndarray,
    cloth_mask: Optional[np.ndarray] = None,
    s_min: float = 0.125,
    s_max: float = 0.625,
) -> np.ndarray:
    """
    Aísla las crestas convexas (saddle ridge [1/8, 3/8] y ridge cilíndrico [3/8, 5/8]).
    
    Args:
        S: Shape Index # shape: (H, W)
        cloth_mask: Máscara binaria opcional de la tela (uint8 o bool)
        s_min: Cota inferior de cresta (0.125 = 1/8)
        s_max: Cota superior de cresta (0.625 = 5/8)
        
    Returns:
        is_ridge: Máscara booleana # shape: (H, W)
    """
    is_ridge = (S >= s_min) & (S <= s_max)
    if cloth_mask is not None:
        is_ridge &= (cloth_mask > 0)
    return is_ridge


def extract_sun_ridges(
    depth_m: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    sigma: float = 2.5,
    min_curvedness_pct: float = 65.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Pipeline completo de extracción de crestas de Sun et al. acoplado a la forma del descriptor 3D.
    
    Combina:
      1. Forma topológica: Koenderink Shape Index S in [0.125, 0.625] (crestas convexas).
      2. Magnitud de curvatura física: Curvedness C > percentil adaptativo (descarta tela plana).
      3. Regularización morfológica: apertura para eliminar ruido espurio del sensor.
      
    Returns:
        clean_ridges: Máscara booleana de crestas limpias # shape: (H, W)
        S: Shape Index # shape: (H, W)
        curvedness: Curvatura escalar C # shape: (H, W)
        C_m: Curvatura media
        C_g: Curvatura gaussiana
        k_max: Curvatura principal máxima
    """
    C_m, C_g, k_max, k_min, S = compute_sun_curvatures_and_shape_index(depth_m, sigma=sigma)
    curvedness = np.sqrt((k_max**2 + k_min**2) / 2.0)

    # 1. Filtro de forma del descriptor 3D (Cresta = Saddle Ridge + Cylindrical Ridge)
    is_shape_ridge = (S >= 0.125) & (S <= 0.625)

    # 2. Filtro de prominencia física (Curvedness)
    if valid_mask is not None and np.any(valid_mask > 0):
        valid_curv = curvedness[valid_mask > 0]
        c_thresh = np.percentile(valid_curv, min_curvedness_pct) if len(valid_curv) > 0 else 1e-4
        is_prominent = curvedness > c_thresh
        ridge_mask = is_shape_ridge & is_prominent & (valid_mask > 0)
    else:
        c_thresh = np.percentile(curvedness, min_curvedness_pct)
        ridge_mask = is_shape_ridge & (curvedness > c_thresh)

    # 3. Limpieza morfológica para unir filamentos y eliminar píxeles aislados
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    clean_ridges = cv.morphologyEx(ridge_mask.astype(np.uint8), cv.MORPH_OPEN, kernel).astype(bool)

    return clean_ridges, S, curvedness, C_m, C_g, k_max


def render_sun_colormap(
    S: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Renderiza el Shape Index S en una imagen BGR térmica con mapa TURBO.
    
    Escala cromática:
      - Azul oscuro / Púrpura: Valles cóncavos (Ruts, tela pegada a la mesa)
      - Verde / Amarillo: Crestas convexas (Ridges, agarrables)
      - Rojo oscuro: Domos / Picos
    """
    # Mapear [-1.0, 1.0] a [0, 255]
    s_norm = np.clip((S + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)
    color_map = cv.applyColorMap(s_norm, cv.COLORMAP_TURBO)

    if valid_mask is not None:
        # Poner fondo inválido o mesa plana en gris oscuro
        color_map[valid_mask == 0] = [40, 40, 40]

    return color_map


def render_ridge_overlay(
    bgr_img: np.ndarray,
    is_ridge: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),  # Verde fosforito BGR
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Superpone las crestas detectadas como un resalte semitransparente sobre la imagen RGB.
    """
    overlay = bgr_img.copy()
    colored_ridge = np.zeros_like(bgr_img)
    colored_ridge[is_ridge] = color

    mask_3c = np.repeat(is_ridge[:, :, np.newaxis], 3, axis=2)
    blended = np.where(mask_3c, cv.addWeighted(bgr_img, 1.0 - alpha, colored_ridge, alpha, 0.0), bgr_img)
    return blended


# ==============================================================================
# PRUEBA UNITARIA Y ASERSIONES DE NO DIVERGENCIA (FEYNMAN PROTOCOL)
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("EJECUTANDO TEST NUMÉRICO DE GEOMETRÍA DIFERENCIAL (SUN ET AL. 2016)")
    print("=" * 70)

    # 1. Crear un cilindro convexo analítico (Ridge perfecto horizontal)
    # Z(x, y) = Z_base - h * cos(pi * y / width), con curvatura principal k_y > 0, k_x = 0
    h, w = 240, 320
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    
    # Mesa plana a 0.50 m
    synth_depth = np.full((h, w), 0.50, dtype=np.float32)
    # Arruga convexa en y in [100, 140] de altura 15 mm
    crest_mask = (y_coords >= 100) & (y_coords <= 140)
    rel_y = (y_coords[crest_mask] - 120) / 20.0  # en [-1, 1]
    synth_depth[crest_mask] -= 0.015 * np.cos(rel_y * (np.pi / 2.0))

    # 2. Calcular derivadas y Shape Index
    C_m, C_g, k_max, k_min, S = compute_sun_curvatures_and_shape_index(synth_depth, sigma=3.0)

    # 3. Aserciones de invariantes matemáticos
    print(f"[TEST 1] Rango Shape Index S: min={S.min():.4f}, max={S.max():.4f} (Debe estar en [-1, 1])")
    assert S.min() >= -1.0 - 1e-4 and S.max() <= 1.0 + 1e-4, "Shape Index fuera de [-1, 1]"
    assert np.isfinite(S).all(), "Shape Index contiene NaNs o Infs"

    # En la mesa plana (lejos de la arruga), S debe ser exactamente 0.0
    mesa_s = S[30:50, 30:50]
    print(f"[TEST 2] Shape Index en mesa plana: max={np.abs(mesa_s).max():.6f} (Esperado ~0.0)")
    assert np.allclose(mesa_s, 0.0, atol=1e-3), "La mesa plana no tiene curvatura nula"

    # En el lomo de la arruga (y=120), S debe caer en el rango de Ridge [0.125, 0.625]
    ridge_s = S[120, 100:220]
    mean_ridge_s = float(np.mean(ridge_s))
    print(f"[TEST 3] Shape Index en lomo de cresta y=120: media={mean_ridge_s:.4f} (Esperado en [0.125, 0.625])")
    assert 0.125 <= mean_ridge_s <= 0.625, f"El lomo de cresta no cayó en rango de Ridge: {mean_ridge_s}"

    # 4. Probar renderizado de mapas
    is_ridge = get_sun_ridge_mask(S)
    colormap = render_sun_colormap(S)
    print(f"[TEST 4] Renderizado de mapa térmico: shape={colormap.shape}, dtype={colormap.dtype}")
    assert colormap.shape == (h, w, 3), "Dimensiones incorrectas de colormap"

    print("\nTODOS LOS TESTS DE GEOMETRÍA DE SUN SUPERADOS CON ÉXITO [PASS].")
