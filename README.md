# Multi-Target Motion Tracker

Real-time motion tracking system using OpenCV and Pygame, capable of
tracking up to 5 moving objects simultaneously using background
subtraction, localized centroid tracking, and adaptive tracking windows.

## Features

-   Tracks up to 5 independent targets
-   Adaptive tracking window per target
-   Background subtraction & noise filtering
-   Median-blur preprocessing
-   Automatic motion segmentation
-   Unique color for each target
-   Real-time visualization using Pygame

## Requirements
```bach
    pip install opencv-python pygame numpy
  ```

## How to Run
```bach
    python motion_tracker.py
  ```

## Controls

  Key         Action
  ----------- -------------------------------------
  Q           Quit the program

## How It Works

### 1. Background Sampling

The program keeps an updated grayscale background using median filtering
and computes difference with the live frame to detect motion.

### 2. Motion Segmentation

Pixels with moderate intensity change are selected and used as motion
candidates.

### 3. Multi-Target Tracking

Each tracker searches within a local window around its last known
position. If no local motion is found, the tracker falls back to global
motion centroid.

### 4. Visualization

Pygame displays a real‑time view with bounding boxes and markers for
each tracked target.

## Notes

-   Works best with stable lighting.
-   Real-time performance depends on webcam and CPU speed.
