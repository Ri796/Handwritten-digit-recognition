# Handwritten-digit-recognition

A **PyTorch-based Convolutional Neural Network (CNN)** for recognizing handwritten digits (0–9) using the **MNIST dataset**. This project was developed and trained in **Google Colab** and later version-controlled on **GitHub**.

---

## 📌 Project Overview

The objective of this project is to build a deep learning model that can accurately classify handwritten digit images. The model learns visual patterns such as strokes and shapes from labeled data and predicts the correct digit class.

This project demonstrates:

* End-to-end ML workflow (data loading → training → evaluation → visualization)
* Practical use of PyTorch for image classification
* Clean experimentation and result analysis

---

## 🧠 Tech Stack

* Python
* PyTorch
* torchvision
* NumPy
* Matplotlib
* Google Colab

---

## 📊 Dataset

**MNIST Handwritten Digits Dataset**

* 60,000 training images
* 10,000 test images
* Image size: 28×28 (grayscale)
* Classes: Digits 0–9

Dataset is loaded using `torchvision.datasets.MNIST`.

---

## 🏗️ Model Architecture

* Convolutional Neural Network (CNN)
* Convolution + ReLU + MaxPooling layers
* Fully connected layers for classification
* Optimizer: Adam
* Loss function: CrossEntropyLoss

---

## 🚀 How to Run

### Option 1: Run on Google Colab

Click below to open the notebook directly in Colab:

[![Open In Colab](

---

### Option 2: Run Locally

```bash
git clone https://github.com/your-username/handwritten-digit-recognition.git
cd handwritten-digit-recognition
pip install -r requirements.txt
```

Run the notebook or training script.

---

## 📈 Results

* Test accuracy: **~98%**
* Model correctly predicts most handwritten digits
* Sample predictions are visualized with true vs predicted labels

---

## 📂 Repository Structure

```
handwritten-digit-recognition/
├── notebooks/
│   └── handwritten_digit_recognition.ipynb
├── model/
│   └── trained_model.pth
├── requirements.txt
└── README.md
```

---

## ✨ Future Improvements

* Add a Streamlit or Tkinter GUI for digit drawing
* Train on custom handwritten digit image

---

## 👩‍💻 Author

**Riddhima Patra**
B.Tech in Mathematics & Computing | BS in Data Science

---

⭐ If you find this project useful, feel free to star the repository!
