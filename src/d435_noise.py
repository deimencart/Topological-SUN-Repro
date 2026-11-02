import numpy as np


def apply_d435_depth_noise(
    depth_image: np.ndarray,
    baseline_m: float = 0.05,
    focal_length_px: float = 600.0,
    subpixel_factor: float = 0.08,
    quantization_step_m: float = 0.001,
    min_depth: float = 0.15,
    max_depth: float = 3.0,
) -> np.ndarray:
    """
    Aplica el modelo de ruido experimental de la cámara Intel RealSense D435 a un mapa de profundidad.
    
    Formula física de dispersión estéreo:
        sigma_z(z) = (z^2 / (focal_length * baseline)) * subpixel_disp_error
    
    Args:
        depth_image: Imagen de profundidad en metros # shape: (H, W)
        baseline_m: Distancia entre lentes del par estéreo (D435: 0.05m = 50mm).
        focal_length_px: Distancia focal en píxeles.
        subpixel_factor: Factor empírico de error subpíxel del ASIC estéreo (~0.08 px).
        quantization_step_m: Cuantización métrica de la salida (D435 usa entero de 1mm = 0.001m).
        min_depth: Rango de corte inferior (Z < 0.15m da 0 / dropout).
        max_depth: Rango de corte superior.

    Returns:
        depth_noisy: Mapa de profundidad con ruido realista en metros # shape: (H, W)
    """
    depth_clean = depth_image.copy()
    valid_mask = (depth_clean >= min_depth) & (depth_clean <= max_depth)
    
    # 1. Ruido Gaussiano dependiente del cuadrado de la distancia
    # sigma_z proporcional a z^2
    sigma_z = (depth_clean ** 2 / (focal_length_px * baseline_m + 1e-7)) * subpixel_factor
    noise = np.random.normal(0.0, sigma_z, size=depth_clean.shape)
    
    noisy_depth = depth_clean + noise
    
    # 2. Cuantización a 1 mm (resolución nativa D435)
    noisy_depth = np.round(noisy_depth / quantization_step_m) * quantization_step_m
    
    # 3. Aplicar máscara de validez (fuera de rango = 0)
    noisy_depth[~valid_mask] = 0.0
    noisy_depth[noisy_depth < min_depth] = 0.0
    noisy_depth[noisy_depth > max_depth] = 0.0
    
    return noisy_depth.astype(np.float32)


if __name__ == "__main__":
    # Test con parche plano a 0.8 metros
    clean_patch = np.ones((50, 50), dtype=np.float32) * 0.80
    noisy_patch = apply_d435_depth_noise(clean_patch)
    
    std_dev = float(np.std(noisy_patch))
    mean_val = float(np.mean(noisy_patch))
    assert abs(mean_val - 0.80) < 0.01, f"Sesgo en media: {mean_val}"
    assert std_dev > 0.0, "El ruido no produjo variación"
    print(f"d435_noise: Mini-test completado con éxito (sigma empírico a 0.8m: {std_dev*1000:.2f} mm).")
