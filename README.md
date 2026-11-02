# Visión Activa y Geometría Diferencial para Manipulación Textil (Sun et al. 2016)

> **Reproducción Académica e Implementación Experimental**  
> *Referencia: Robotic Clothes Manipulation via Active Vision & Differential Geometry (Sun et al., 2016)*  
> *Validación en Hardware: Cámara Intel RealSense D435 en tiempo real (30 FPS sobre USB 3.2)*

---

> [!IMPORTANT]
> **Aviso de Reproducción Académica (Sin fines de plagio)**  
> Este repositorio contiene una **reproducción experimental e implementación académica independiente** desarrollada con propósitos educativos y de investigación en visión y robótica. Los fundamentos matemáticos y la formulación metodológica pertenecen a sus autores originales (*Sun et al., 2016*). Este trabajo se publica sin fines comerciales ni intención de plagio.

---

## 📌 ¿Qué problema resuelve este proyecto?

Para que un brazo robótico pueda desdoblar o manipular ropa, debe identificar puntos de agarre viables. Mirar la tela como una simple imagen 2D (RGB) es insuficiente porque las sombras y los estampados confunden a los detectores tradicionales.

El enfoque de **Sun et al. (2016)** propone tratar la nube de puntos 3D de la cámara como una **superficie continua en el espacio**. Mediante geometría diferencial, se analiza la curvatura local de cada punto para responder a dos preguntas fundamentales:
1. **¿Qué tipo de forma tiene este punto?** (Clasificación topológica invariante a escala: si es una cresta elevada, un pliegue convexo, un valle hundido o una zona plana).
2. **¿Dónde están los pliegues ideales para que la pinza agarre?** (Detección de crestas elevadas donde una pinza paralela puede pellizcar tela sin chocar contra la mesa).

---

## 🛠️ ¿Qué implementamos en este repositorio?

El sistema implementa el cálculo numérico completo de la superficie 3D y su visualización en vivo a 30 FPS:

```
[ Nube de Puntos / Profundidad RealSense D435 ]
                      |
                      v
          [ Filtro Bilateral Suave ]
        (Elimina ruido de alta frecuencia)
                      |
                      v
      [ Gradientes Espaciales 1º y 2º Orden ]
             (fx, fy, fxx, fyy, fxy)
                      |
                      v
      [ Curvaturas Principales (k1, k2) ]
        (Curvatura Gaussiana y Media)
                      |
                      v
        [ Koenderink Shape Index (S) ]
        (Clasificación de forma en [-1, 1])
                      |
                      v
      [ Detección de Crestas por Laplaciano ]
          (Cruce por cero: nabla^2 Z = 0)
                      |
                      v
         [ Interfaz Dual en Tiempo Real ]
   (RGB con arrugas verdes | Mapa térmico Shape Index)
```

### 1. Motor Geométrico Vectorizado (`src/sun_geometry.py`)
* **Primera y Segunda Forma Fundamental:** Calcula los coeficientes de superficie ($E, F, G$ y $L, M, N$) de forma vectorizada con NumPy y SciPy.
* **Curvatura Gaussiana ($K$) y Media ($H$):** Determina las curvaturas principales $\kappa_1$ y $\kappa_2$.
* **Koenderink Shape Index ($S \in [-1, 1]$):** Normaliza el tipo de superficie:
  * $S \in [0.125, 0.375]$: **Crestas y pliegues convexos** (zonas óptimas para pellizco robótico).
  * $S \approx 0.0$: **Sillas de montar** (zonas de doble curvatura).
  * $S \in [-0.375, -0.125]$: **Surcos y valles cóncavos**.
* **Detección de Arrugas por Cruce por Cero:** Identifica las líneas centrales de los pliegues donde el Laplaciano de la superficie se anula ($\nabla^2 Z = 0$) con curvatura $\kappa_1 \ge \kappa_{\text{threshold}}$.

### 2. Streaming en Tiempo Real con Intel RealSense D435 (`src/live_realsense_sun.py`)
* Pipeline multihilo calibrado para capturar a **30 FPS** sobre bus **USB 3.2 Gen 1**.
* Presentación en pantalla lado a lado:
  * **Ventana Izquierda:** Imagen RGB original con las crestas detectadas resaltadas en verde semitransparente.
  * **Ventana Derecha:** Colormap térmico con los valores continuos del Shape Index de Koenderink.

### 3. Modelo Experimental de Ruido de Profundidad (`src/d435_noise.py`)
* Simulación realista del error cuadrático con la distancia ($\sigma \propto Z^2$) de sensores infrarrojos para validar que el filtro bilateral no degrade las crestas con ruido real.

---

## 📊 Evidencias y Resultados Visuales

| Detección de Crestas en Vivo (RealSense D435) | Análisis Cuádruple de Geometría Diferencial |
| :---: | :---: |
| ![Live Ridges](docs/images/test_live_ridges.png) | ![4-Panel Output](docs/images/test_output.png) |

| Filtrado Bilateral Adaptativo Anti-Ruido | Resalte de Candidatos de Agarre |
| :---: | :---: |
| ![Adaptive Ridges](docs/images/test_ridges_adaptive.png) | ![Overlay](docs/images/test_ridges_overlay.png) |

---

## 📁 Estructura del Repositorio

```text
sun2016_active_vision/
├── README.md                                 <- Documentación general
├── .gitignore                                <- Exclusiones de Git
├── docs/
│   ├── Sun_ActiveVisionClothesManipulation_2016.pdf <- Paper científico original
│   └── images/                               <- Capturas reales del visualizador
└── src/
    ├── sun_geometry.py                       <- Motor matemático vectorizado en NumPy
    ├── live_realsense_sun.py                 <- Visualizador en vivo 30 FPS (RealSense D435)
    └── d435_noise.py                         <- Modelo de ruido sensor para pruebas
```

---

## 🚀 Guía Rápida de Ejecución

### 1. Dependencias
```bash
pip install numpy scipy opencv-python pyrealsense2
```

### 2. Modo con Cámara en Vivo (Intel RealSense D435)
Conecta la cámara por puerto USB 3.0 o 3.2 y ejecuta:
```bash
python src/live_realsense_sun.py
```
* **Controles:**
  * `q` / `ESC`: Salir.
  * `s`: Guardar captura instantánea de RGB, profundidad, mapa térmico y crestas detectadas.

### 3. Modo de Prueba Sintética (Sin Cámara)
Para validar el algoritmo en cualquier equipo sin hardware conectado:
```bash
python src/sun_geometry.py
```
Ejecuta una prueba unitaria embebida con superficies arrugadas sintéticas y valida que no existan divisiones por cero ni NaNs.

---

## 📚 Referencias
* Sun, L., et al. (2016). *Robotic Clothes Manipulation via Active Vision & Differential Geometry*.
* Koenderink, J. J., & van Doorn, A. J. (1992). *Surface shape and curvature scales*. Image and Vision Computing.
