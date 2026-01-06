# Handwritten Digit Recognition (Mobile App + Model)

A **PyTorch-based Convolutional Neural Network (CNN)** for recognizing handwritten digits (0–9), now integrated into a **Flutter Mobile Application**.

## 📌 Project Overview

The objective of this project is to build a deep learning model that can accurately classify handwritten digit images and deploy it to a mobile device.

This project consists of:
1.  **ML Model**: A PyTorch CNN trained on MNIST (Accuracy ~99.45%).
2.  **Mobile App**: A cross-platform Flutter app that runs the model locally on Android using ONNX Runtime.

---

## 📱 Mobile Application (Flutter)

The `mnist_onnx_app` allows users to:
*   **Draw** digits directly on the screen.
*   **Upload** photos of handwritten digits from the gallery/camera.
*   Get instant offline predictions.

### Getting Started with the App

1.  **Prerequisites**:
    *   Flutter SDK installed.
    *   Android Studio / VS Code.

2.  **Installation**:
    ```bash
    flutter pub get
    flutter run
    ```

---

## 🧠 Tech Stack

*   **Mobile**: Flutter, Dart, ONNX Runtime
*   **ML Core**: Python, PyTorch, torchvision, NumPy
*   **Data**: MNIST Dataset

---

## 🏗️ Model Architecture (CNN)

*   **layers**: 3 Convolutional Blocks + MaxPool
*   **Input**: 28x28 Grayscale images
*   **Export**: Converted to ONNX format for mobile deployment.

---

## 📂 Repository Structure

```
├── lib/                 # Flutter UI Code (main.dart)
├── assets/              # ONNX Model file
├── train_model.py       # Python script to train & export model
├── Handwritten-digit-recognition.ipynb # Original Colab Notebook
└── android/             # Native Android code
```

---

## 👩‍💻 Author

**Riddhima Patra**
B.Tech in Mathematics & Computing | BS in Data Science
