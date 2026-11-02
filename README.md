# Active Vision & Differential Geometry for Robotic Clothes Manipulation (Sun et al. 2016)

> **Independent Academic Reproduction & Experimental Implementation**  
> *Reference: Robotic Clothes Manipulation via Active Vision & Differential Geometry (Sun et al., 2016)*  
> *Hardware Validation: Real-Time Intel RealSense D435 Streaming (30 FPS over USB 3.2)*

---

> [!IMPORTANT]
> **Academic Reproduction & Non-Plagiarism Disclaimer**  
> This repository contains an **independent academic reproduction and experimental implementation** developed for educational and research purposes in computer vision and robotics. The mathematical surface models and methodology belong to their original authors (*Sun et al., 2016*). This work is non-commercial and carries no intent of plagiarism.

---

## 📌 Problem Statement

To enable a robot to grasp and manipulate textiles, the system must locate viable grasp points. Relying solely on flat 2D RGB images is fragile: fabric textures, shadows, and printed patterns easily confuse classical contour detectors.

The formulation by **Sun et al. (2016)** treats depth point clouds as a **continuous 2-manifold surface in 3D space**. By calculating local surface curvatures, the method answers two key questions:
1. **What geometric shape does each point have?** (Scale-invariant topological classification: convex fold, elevated ridge, trough, or flat surface).
2. **Where are the optimal grasp candidates?** (Detecting elevated ridge lines where a parallel-jaw gripper can pinch cloth without colliding with the support table).

---

## 🛠️ Implementation Overview

The codebase provides full 3D surface geometry computation and real-time 30 FPS visualization:

```
[ Intel RealSense D435 Point Cloud / Depth ]
                      |
                      v
          [ Bilateral Smooth Filter ]
         (Removes high-frequency noise)
                      |
                      v
      [ 1st & 2nd Order Spatial Gradients ]
             (fx, fy, fxx, fyy, fxy)
                      |
                      v
      [ Principal Curvatures (k1, k2) ]
        (Gaussian & Mean Curvatures)
                      |
                      v
        [ Koenderink Shape Index (S) ]
       (Shape classification in [-1, 1])
                      |
                      v
     [ Laplacian Zero-Crossing Detector ]
          (Ridge detection: nabla^2 Z = 0)
                      |
                      v
         [ Real-Time Dual-Window GUI ]
   (RGB with green ridges | Thermal Shape Index map)
```

### 1. Vectorized Geometry Engine (`src/sun_geometry.py`)
* **First & Second Fundamental Forms:** Vectorized computation of surface coefficients ($E, F, G$ and $L, M, N$) using NumPy and SciPy.
* **Gaussian ($K$) and Mean ($H$) Curvature:** Resolves principal curvatures $\kappa_1$ and $\kappa_2$.
* **Koenderink Shape Index ($S \in [-1, 1]$):** Normalizes topological shape types:
  * $S \in [0.125, 0.375]$: **Convex ridges and folds** (primary pinch targets for robot grippers).
  * $S \approx 0.0$: **Saddle points** (regions of mixed curvature).
  * $S \in [-0.375, -0.125]$: **Concave valleys and troughs**.
* **Ridge Extraction via Laplacian Zero-Crossings:** Delimits fold centerlines where the surface Laplacian vanishes ($\nabla^2 Z = 0$) subject to principal curvature threshold $\kappa_1 \ge \kappa_{\text{threshold}}$.

### 2. Real-Time Streaming with Intel RealSense D435 (`src/live_realsense_sun.py`)
* Multithreaded pipeline delivering **30 FPS** over **USB 3.2 Gen 1**.
* Side-by-side interactive display:
  * **Left Window:** Raw RGB frame with detected grasp ridges highlighted in semi-transparent green.
  * **Right Window:** Thermal colormap displaying continuous Koenderink Shape Index values.

### 3. Depth Sensor Noise Model (`src/d435_noise.py`)
* Quadratic error simulation ($\sigma \propto Z^2$) based on physical stereo infrared camera characteristics to ensure bilateral filtering preserves real folds under realistic sensor noise.

---

## 📊 Visual Evidence & Results

| Live Ridge Detection (RealSense D435) | Full Differential Geometry Pipeline |
| :---: | :---: |
| ![Live Ridges](docs/images/test_live_ridges.png) | ![4-Panel Output](docs/images/test_output.png) |

| Adaptive Noise-Resistant Bilateral Filter | Grasp Ridge Overlay on Real Garment |
| :---: | :---: |
| ![Adaptive Ridges](docs/images/test_ridges_adaptive.png) | ![Overlay](docs/images/test_ridges_overlay.png) |

---

## 📁 Repository Structure

```text
sun2016_active_vision/
├── README.md                                 <- Master documentation
├── .gitignore                                <- Git ignore rules
├── docs/
│   ├── Sun_ActiveVisionClothesManipulation_2016.pdf <- Original paper
│   └── images/                               <- Experimental capture screenshots
└── src/
    ├── sun_geometry.py                       <- Vectorized surface geometry engine
    ├── live_realsense_sun.py                 <- 30 FPS RealSense live visualizer
    └── d435_noise.py                         <- Realistic depth noise model
```

---

## 🚀 Quickstart Guide

### 1. Requirements & Dependencies
```bash
pip install numpy scipy opencv-python pyrealsense2
```

### 2. Live Camera Streaming Mode (Intel RealSense D435)
Connect the camera to a USB 3.0/3.2 port and run:
```bash
python src/live_realsense_sun.py
```
* **Hotkeys:**
  * `q` / `ESC`: Exit viewer.
  * `s`: Save a snapshot of RGB, depth, thermal Shape Index, and detected ridges.

### 3. Standalone Synthetic Demo Mode (No Camera Required)
To verify the math without connected camera hardware:
```bash
python src/sun_geometry.py
```
Executes an embedded unit test on synthetic wrinkled surfaces and verifies zero division guards and numerical stability.

---

## 📚 References
* Sun, L., et al. (2016). *Robotic Clothes Manipulation via Active Vision & Differential Geometry*.
* Koenderink, J. J., & van Doorn, A. J. (1992). *Surface shape and curvature scales*. Image and Vision Computing.
