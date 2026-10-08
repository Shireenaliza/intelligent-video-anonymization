# 🎥 Intelligent Video Anonymization

### Selective Face Detection • Recognition • Privacy-Preserving Video Processing

<p align="center">
  <img src="assets/sample-output.png" width="90%" alt="Sample anonymized output">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv">
  <img src="https://img.shields.io/badge/MTCNN-Face%20Detection-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/FaceNet-Face%20Recognition-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?style=for-the-badge&logo=pytorch">
</p>

<p align="center">
  <b>Automatically detect, identify and anonymize a selected person in a multi-person video while preserving the rest of the scene.</b>
</p>

---

## 📌 Project Overview

**Intelligent Video Anonymization** is a computer-vision pipeline for targeted face anonymization.

The system takes:

1. A video containing multiple individuals.
2. A reference image of the person whose face should be anonymized.

The video is processed frame by frame. Faces are detected using **MTCNN**, detected faces are compared with the reference using **FaceNet embeddings**, and only the matching face is anonymized.

The project presentation describes the core flow as **face detection → face recognition → selective blurring → output video**.

> **Core idea:** Instead of anonymizing every detected face, identify the specific person represented by a reference image and anonymize only that individual.

---

## 🎯 Problem Statement

Traditional video anonymization can require manual editing or can anonymize every detected face.

This project addresses a more targeted problem:

> **How can we automatically anonymize only a specific individual in a multi-person video while keeping the rest of the scene intact?**

The approach combines:

- MTCNN for face detection
- FaceNet embeddings for face recognition
- Cosine similarity for identity matching
- Configurable blur techniques for anonymization
- OpenCV for video frame extraction and reconstruction
- Quantitative metrics for evaluating anonymization and image quality

---

## ✨ Key Features

- 🎯 **Selective anonymization** — blur only the target individual.
- 👤 **MTCNN face detection** — detect multiple faces in each frame.
- 🧠 **FaceNet recognition** — compare identities using face embeddings.
- 🎞️ **Frame-by-frame processing** — process and reconstruct videos with OpenCV.
- 🌫️ **Multiple blur techniques**:
  - Gaussian
  - Mosaic / Pixelation
  - Dot
  - Triangle
- 📊 **Quantitative evaluation**:
  - PSNR
  - SSIM
  - MSE
  - Laplacian Variance
  - Tenengrad
  - Brenner
- ⚙️ **Configurable matching threshold and blur strength**.
- 💻 **CPU/GPU device selection**.
- 📁 **Modular project structure** suitable for experimentation and extension.

---

## 🧩 System Architecture

The architecture used in the project follows this processing flow:

```text
                    ┌──────────────────┐
                    │   Input Video    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Frame Extraction │
                    │     OpenCV       │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Face Detection  │
                    │      MTCNN       │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Face Recognition │
                    │     FaceNet      │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Embedding       │
                    │   Comparison     │
                    └────────┬─────────┘
                             │
                       Target match?
                       /           \
                     Yes            No
                      │              │
                      ▼              ▼
                ┌──────────┐   ┌──────────┐
                │   Blur   │   │   Keep   │
                │  Target  │   │   Face   │
                └────┬─────┘   └────┬─────┘
                     │              │
                     └──────┬───────┘
                            ▼
                   ┌──────────────────┐
                   │  Output Video    │
                   └──────────────────┘
```

### Architecture Image

Replace the file below with the architecture image from your presentation:

<p align="center">
  <img src="assets/architecture.png" width="90%" alt="System architecture">
</p>

---

## 🔬 Methodology

### 1. Reference Image

A reference image containing the target person's face is supplied to the system.

<p align="center">
  <img src="assets/input-reference.png" width="70%" alt="Reference face">
</p>

### 2. Face Detection

Each video frame is passed through **MTCNN (Multi-task Cascaded Convolutional Networks)** to detect faces.

The detector provides bounding boxes for the faces present in the frame.

<p align="center">
  <img src="assets/mtcnn.png" width="80%" alt="MTCNN architecture">
</p>

### 3. Face Recognition

The detected faces are converted into embeddings using **FaceNet / InceptionResnetV1**.

The reference face is also converted into an embedding.

### 4. Identity Matching

The implementation compares embeddings using cosine similarity:

$$
S(x,y)=\frac{x\cdot y}{\|x\|\|y\|}
$$

A configurable threshold determines whether the detected face is considered the target.

The default implementation threshold is **0.90**. This is a project configuration value and should be tuned using validation data for a particular dataset.

### 5. Selective Anonymization

If the detected face matches the reference identity, only that face region is passed to the selected blur operation.

Other people in the frame remain unchanged.

### 6. Video Reconstruction

OpenCV writes the processed frames into a new output video.

---

## 🎨 Blur Techniques

The project supports four blur strategies:

| Technique | Description |
|---|---|
| **Gaussian** | Smooth Gaussian filtering for strong face anonymization |
| **Mosaic** | Pixelation through downsampling and nearest-neighbor reconstruction |
| **Dot** | Custom circular/dot-shaped filtering kernel |
| **Triangle** | Custom triangular weighted filtering kernel |

The project presentation specifically evaluates **Gaussian, Mosaic, Dot and Triangle** blur techniques.

<p align="center">
  <img src="assets/blur-comparison.png" width="90%" alt="Blur technique comparison">
</p>

---

## 📊 Evaluation

The project evaluates both **visual quality** and **blur strength**.

### Image Quality Metrics

#### PSNR — Peak Signal-to-Noise Ratio

PSNR measures the quality of the processed image relative to the original.

$$
PSNR = 10\log_{10}\left(\frac{MAX_I^2}{MSE}\right)
$$

Higher PSNR generally indicates lower pixel-level distortion.

#### SSIM — Structural Similarity Index

SSIM measures perceptual and structural similarity between images.

Values closer to **1** indicate stronger structural similarity.

#### MSE — Mean Squared Error

$$
MSE=\frac{1}{MN}\sum (I-K)^2
$$

Lower MSE indicates less pixel-level error.

### Blur-Strength Metrics

The project also evaluates:

- **Laplacian Variance**
- **Tenengrad**
- **Brenner**

These are sharpness/focus measures. Lower values indicate a more blurred image.

<p align="center">
  <img src="assets/evaluation.png" width="90%" alt="Evaluation metrics">
</p>

---

## 🏆 Findings From the Project Evaluation

According to the project evaluation:

- **Gaussian Blur** produced the strongest anonymization according to the blur-strength measurements.
- **Mosaic Blur** provided strong visual-quality retention, with better PSNR/MSE behavior in the reported comparison.
- **Triangle Blur** provided moderate blur but showed higher distortion according to MSE.
- **Dot Blur** produced the weakest blur strength and was therefore less suitable for anonymization.

The overall project conclusion is:

> **Gaussian Blur is the strongest choice for anonymization, while Mosaic Blur provides a useful balance between anonymization and visual-quality retention.**

<p align="center">
  <img src="assets/results.png" width="90%" alt="Project results">
</p>

> **Important:** Keep your exact experimental values in `results/metrics/`. Do not replace them with example numbers in the README.

---

## 🎬 Sample Output

The final output is a video in which only the selected target face is anonymized.

<p align="center">
  <img src="assets/sample-output.png" width="90%" alt="Sample anonymized frame">
</p>

If you have a short GIF demonstrating the complete pipeline, you can replace the image above with:

```html
<p align="center">
  <img src="assets/demo.gif" width="90%" alt="Video anonymization demo">
</p>
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| Video processing | OpenCV |
| Face detection | MTCNN |
| Face recognition | FaceNet / InceptionResnetV1 |
| Deep learning | PyTorch |
| Image processing | OpenCV, NumPy |
| Image-quality evaluation | scikit-image |
| Visualization | Matplotlib |

---

## 📁 Project Structure

```text
intelligent-video-anonymization/
│
├── assets/
│   ├── architecture.png
│   ├── mtcnn.png
│   ├── input-reference.png
│   ├── sample-output.png
│   ├── blur-comparison.png
│   ├── evaluation.png
│   ├── results.png
│   └── demo.gif
│
├── data/
│   ├── input/
│   │   └── .gitkeep
│   ├── reference/
│   │   └── .gitkeep
│   └── output/
│       └── .gitkeep
│
├── docs/
│   └── methodology.md
│
├── results/
│   ├── metrics/
│   │   └── .gitkeep
│   └── plots/
│       └── .gitkeep
│
├── scripts/
│   ├── run_anonymisation.py
│   ├── evaluate_video.py
│   ├── compare_blurs.py
│   └── plot_metrics.py
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── detector.py
│   ├── recognizer.py
│   ├── blur.py
│   ├── video_io.py
│   ├── pipeline.py
│   └── metrics.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/<YOUR_USERNAME>/intelligent-video-anonymization.git
cd intelligent-video-anonymization
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For a CUDA-enabled PyTorch installation, install the appropriate PyTorch build for your CUDA version before running the project.

---

## ▶️ Quick Start

Place your files like this:

```text
data/
├── input/
│   └── input_video.mp4
│
└── reference/
    └── target_face.jpg
```

Run:

```bash
python scripts/run_anonymisation.py ^
    --video data/input/input_video.mp4 ^
    --reference data/reference/target_face.jpg ^
    --output data/output/anonymised.mp4
```

On Linux/macOS:

```bash
python scripts/run_anonymisation.py     --video data/input/input_video.mp4     --reference data/reference/target_face.jpg     --output data/output/anonymised.mp4
```

---

## 🎛️ Change the Blur Method

### Gaussian

```bash
python scripts/run_anonymisation.py     --video data/input/input_video.mp4     --reference data/reference/target_face.jpg     --output data/output/gaussian.mp4     --blur gaussian
```

### Mosaic

```bash
python scripts/run_anonymisation.py     --video data/input/input_video.mp4     --reference data/reference/target_face.jpg     --output data/output/mosaic.mp4     --blur mosaic
```

### Dot

```bash
python scripts/run_anonymisation.py     --video data/input/input_video.mp4     --reference data/reference/target_face.jpg     --output data/output/dot.mp4     --blur dot
```

### Triangle

```bash
python scripts/run_anonymisation.py     --video data/input/input_video.mp4     --reference data/reference/target_face.jpg     --output data/output/triangle.mp4     --blur triangle
```

---

## 🎚️ Tune Face Matching

The default similarity threshold is `0.90`.

You can change it:

```bash
python scripts/run_anonymisation.py     --video data/input/input_video.mp4     --reference data/reference/target_face.jpg     --output data/output/anonymised.mp4     --threshold 0.85
```

A lower threshold can increase matches but may also increase false positives. A higher threshold is more selective but can miss the target under difficult conditions.

For a formal experiment, select the threshold using validation data rather than changing it only to improve one sample.

---

## 📈 Evaluate the Processed Video

After generating the anonymized video:

```bash
python scripts/evaluate_video.py     --original data/input/input_video.mp4     --processed data/output/anonymised.mp4
```

The metrics are saved to:

```text
results/metrics/video_metrics.json
```

To evaluate fewer frames:

```bash
python scripts/evaluate_video.py     --original data/input/input_video.mp4     --processed data/output/anonymised.mp4     --sample-every 30
```

---

## 🔬 Compare Blur Techniques

Prepare a representative image or face crop:

```text
assets/face-sample.png
```

Then run:

```bash
python scripts/compare_blurs.py     --image assets/face-sample.png
```

The generated files will be placed under:

```text
results/blur_comparison/
```

Including:

```text
gaussian.png
mosaic.png
dot.png
triangle.png
metrics.json
```

---

## 📊 Generate Metric Plots

Run:

```bash
python scripts/plot_metrics.py
```

Plots are generated under:

```text
results/blur_comparison/plots/
```

---

## 🔐 Privacy and Responsible Use

This project processes facial information and video containing potentially identifiable people.

Use it responsibly:

- Use videos and reference images for which you have permission.
- Avoid committing real people's faces or private videos to GitHub.
- Keep private media inside ignored `data/` directories.
- Use synthetic, public-domain, or properly consented examples for demonstrations.
- Treat anonymization as a privacy aid, not a guarantee of irreversible de-identification.

The repository intentionally ignores:

```text
data/input/*
data/reference/*
data/output/*
results/*
```

so private media and generated experiment files are not accidentally committed.

---

## ⚠️ Limitations

The current implementation is intended as an experimental computer-vision pipeline.

Performance can be affected by:

- Face pose and orientation
- Occlusion
- Lighting conditions
- Motion blur
- Small faces
- Video resolution
- Similar-looking individuals
- Reference-image quality
- Similarity-threshold selection
- CPU/GPU processing capability

The implementation should therefore be evaluated on representative validation data before being used in a real privacy-critical deployment.

---

## 🔮 Future Improvements

Possible extensions include:

- Real-time webcam/video-stream processing
- Multi-reference identity matching
- More robust tracking between frames
- Temporal consistency across video frames
- Configurable face-recognition backbones
- Adaptive similarity thresholds
- More anonymization techniques
- Side-by-side automated evaluation reports
- GPU-optimized batch inference
- Web interface for uploading videos and selecting anonymization settings
- Automated experiment tracking

---

## 🧪 Reproducibility

For reproducible experiments, record:

- Input video
- Reference image characteristics
- Recognition threshold
- Blur method
- Blur strength
- Mosaic block size
- Device used
- Number of evaluated frames
- PSNR
- SSIM
- MSE
- Laplacian Variance
- Tenengrad
- Brenner

Keep experiment outputs under:

```text
results/
```

and document important configurations alongside the generated metrics.

---

## 📚 Project Documentation

Additional methodology notes are available in:

```text
docs/methodology.md
```

---

## 👩‍💻 Author

**Shireen Aliza Ali**

B.Tech — Computer Science & Engineering  
Specialization: Artificial Intelligence & Machine Learning

---

## ⭐ Repository Positioning

This project is designed to demonstrate a complete applied computer-vision workflow:

```text
Deep Learning
     ↓
Face Detection
     ↓
Face Recognition
     ↓
Identity Matching
     ↓
Selective Anonymization
     ↓
Video Processing
     ↓
Quantitative Evaluation
```

It combines **computer vision, deep learning, image processing, privacy-preserving AI, and quantitative evaluation** into one end-to-end project.

---

## 📄 License

Add the license that matches your intended use before publishing the repository.

For an academic portfolio repository, a standard open-source license such as MIT can be considered if all included dependencies, datasets, images, and other assets permit the intended redistribution.
