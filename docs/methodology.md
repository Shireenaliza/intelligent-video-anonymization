# Methodology

## 1. Input

The pipeline accepts:

- A video containing multiple individuals.
- A reference image containing the face to anonymize.

## 2. Frame Processing

OpenCV reads the video frame by frame.

## 3. Face Detection

MTCNN detects the faces present in each frame.

## 4. Face Recognition

FaceNet/InceptionResnetV1 generates an embedding for each detected face.

## 5. Identity Matching

The reference embedding is compared with detected-face embeddings using cosine similarity.

For embeddings x and y:

S(x, y) = (x · y) / (||x|| ||y||)

A configurable similarity threshold determines whether a detected face is considered the target.

## 6. Selective Anonymization

Only the matching face region is passed to the selected blur method.

Supported methods:

- Gaussian
- Mosaic / Pixelation
- Dot
- Triangle

## 7. Video Reconstruction

Processed frames are written back to a new video file.

## 8. Evaluation

The project provides:

- PSNR
- SSIM
- MSE
- Laplacian Variance
- Tenengrad
- Brenner

Use the provided scripts to generate quantitative experiment results.
