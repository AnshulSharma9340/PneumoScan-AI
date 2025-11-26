# 🩺 PneumoScan-AI: Automated Pneumonia Detection from Chest X-rays

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.9-blue?style=for-the-badge\&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-FF6F00?style=for-the-badge\&logo=tensorflow)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.13-D00000?style=for-the-badge\&logo=keras)](https://keras.io/)
[![Flask](https://img.shields.io/badge/Web%20App-Flask-000000?style=for-the-badge\&logo=flask)](https://flask.palletsprojects.com/)



## ⭐ Project Overview

PneumoScan-AI is a **Deep Learning** web application for **Binary Classification** to detect **Pneumonia** (Viral or Bacterial) from Chest X-ray images.
It includes **Explainable AI (XAI)** using Grad-CAM heatmaps for transparent diagnosis.

### Key Features

* **High Accuracy:** Fine-tuned MobileNetV2 achieving 95%+ accuracy.
* **Explainable AI:** Grad-CAM heatmaps highlight infected regions.
* **Modern Web UI:** Glassmorphism/Neon-themed UI with Flask + Tailwind CSS.
* **Robust Setup:** Tested for dependency and version conflicts.


## 🛠️ Tech Stack & Dependencies

| Category         | Tool / Library | Version Used        | Purpose                                          |
| ---------------- | -------------- | ------------------- | ------------------------------------------------ |
| Deep Learning    | TensorFlow     | 2.13.0              | Core framework for model building & training     |
| Model            | MobileNetV2    | (Inbuilt)           | Pre-trained Transfer Learning model              |
| Web Framework    | Flask          | Latest Stable       | Backend for handling uploads & predictions       |
| Image Processing | OpenCV (cv2)   | Latest Stable       | Image manipulation & Grad-CAM heatmap generation |
| Data Handling    | NumPy, h5py    | Compatible Versions | Numerical operations & model file handling       |

---

## 📂 Project Structure


pneumonia_detection/
├── dataset/        <-- Downloaded dataset goes here
│   ├── train/
│   ├── valid/
│   └── test/
├── model/          <-- Trained model (final_model.h5)
├── static/
│   ├── uploads/
│   └── heatmap.jpg
├── templates/
│   ├── index.html
│   └── result.html
├── app.py          <-- Flask backend
├── train_model.ipynb
├── grad_cam.py
└── README.md       <-- This file


**Dataset Note:** (~1.2 GB) is **not included**. Download from [Kaggle Link] and place folders inside `dataset/`.



## 🚀 Installation & Usage

### 1. Clone Repository


git clone https://github.com/AnshulSharma9340/PneumoScan-AI.git
cd PneumoScan-AI


### 2. Create & Activate Environment


# Using conda
conda create -n pneumonia python=3.10 -y
conda activate pneumonia


### 3. Install Dependencies


pip install tensorflow==2.13
pip install numpy==1.24.3
pip install keras==2.13.1
pip install protobuf==4.25.8
pip install opencv-python matplotlib flask pillow


### 4. Run the Web App


python app.py
# Open browser at http://127.0.0.1:5000/


### 5. Features Demo

* Upload Chest X-ray image (PNG/JPEG)
* Classify as **NORMAL** or **PNEUMONIA** with confidence score
* Grad-CAM heatmap shows infected areas



## 👤 Author

**Anshul Sharma**
[GitHub Profile](https://github.com/AnshulSharma9340)


## 📜 License

This project is licensed under MIT License.
