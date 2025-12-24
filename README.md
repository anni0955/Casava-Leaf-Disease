# 🌿 Cassava Leaf Disease Classification using Deep Learning

![Deep Learning](https://img.shields.io/badge/Deep%20Learning-TensorFlow-blue)
![Keras](https://img.shields.io/badge/Keras-CNN%20%7C%20Transfer%20Learning-red)
![EfficientNet](https://img.shields.io/badge/Model-EfficientNetB0-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

An end-to-end **Deep Learning image classification project** for detecting **Cassava Leaf Diseases** using **TensorFlow & Keras**.  
This project follows a **progressive learning approach**:  
➡️ Custom CNN → Transfer Learning → Fine-Tuning with EfficientNetB0.

---

## 📌 Problem Statement

Cassava is a vital crop, but leaf diseases significantly reduce yield and quality.  
The objective of this project is to **automatically classify cassava leaf images into five categories** using deep learning.

---

## 🧬 Dataset Overview

- **Dataset**: Cassava Leaf Disease Classification  
- **Input Size**: `224 × 224 × 3`
- **Total Classes**: 5 (Multi-class classification)
- **Labels mapped to readable class names**

### 🏷️ Classes
| Label | Disease |
|-----|--------|
| 0 | Cassava Bacterial Blight (CBB) |
| 1 | Cassava Brown Streak Disease (CBSD) |
| 2 | Cassava Green Mottle (CGM) |
| 3 | Cassava Mosaic Disease (CMD) |
| 4 | Healthy |

📊 **Note**: Dataset is **highly imbalanced**, with CMD having the highest samples.

---

## 🔄 Project Workflow

### 1️⃣ Data Preparation
- Loaded image paths & labels from CSV
- Converted numeric labels → class names
- Created optimized `tf.data.Dataset`
- Applied batching, shuffling & prefetching

---

### 2️⃣ Baseline Model – Custom CNN 🧠

- Built a **custom CNN from scratch**
- Used:
  - Convolution layers
  - ReLU activations
  - Dense classifier
- Loss: `Sparse Categorical Crossentropy`
- Optimizer: `Adam`

📉 **Results**
- Training Accuracy: ~74%
- Validation Accuracy: ~70%

🔍 **Observation**: Model learned basic patterns but struggled to generalize.

---

### 3️⃣ Transfer Learning – EfficientNetB0 🚀

Used **EfficientNetB0 pretrained on ImageNet** for better feature extraction.

✔️ Key points:
- `include_top = False`
- Base model frozen initially
- Correct **EfficientNet preprocessing**
- Added:
  - GlobalAveragePooling
  - Dense layers
  - Dropout

📈 **Results**
- Validation Accuracy improved to **~76%**
- Faster convergence
- More stable training

---

### 4️⃣ Fine-Tuning 🔥

To further boost performance:
- Unfroze **last few layers** of EfficientNet
- Added **Batch Normalization**
- Controlled overfitting with Dropout

🏆 **Final Performance**
- Training Accuracy: ~81%
- Validation Accuracy: **~77–78%**
- Reduced train–validation gap

---

## 📊 Training Curves

### 📉 Loss & Accuracy Trends
- Training loss decreases smoothly
- Validation accuracy plateaus due to class imbalance
- Fine-tuning improves generalization

📌 Best models saved automatically using **ModelCheckpoint**

---

## 🛠️ Tech Stack

- **Language**: Python 🐍
- **Frameworks**:
  - TensorFlow
  - Keras
  - NumPy
  - Pandas
  - Matplotlib
- **Models**:
  - Custom CNN
  - EfficientNetB0 (Transfer Learning + Fine-Tuning)

---

## 📂 Repository Structure

📁 cassava-leaf-disease-classification  
├── 📓 notebooks  
│   └── cassava_classification.ipynb  
├── 💾 saved_models  
│   ├── best_model_in_custom_cnn.keras  
│   └── best_model_in_transfer_learning.keras  
├── 📄 README.md  


---

## 🎯 Key Learnings

✅ Importance of correct preprocessing for pretrained models  
✅ Transfer learning significantly boosts accuracy  
✅ Fine-tuning improves feature specialization  
✅ Validation metrics matter more than training accuracy  
✅ ModelCheckpoint is essential for experimentation  

---

## 🚀 Future Improvements

- Handle class imbalance using **class weights**
- Try **EfficientNetB3 / B4**
- Add **Grad-CAM** for interpretability
- Deploy using **Streamlit or FastAPI**

---

## 👨‍💻 Author

**Animesh Porwal**  
Machine Learning & Deep Learning Enthusiast  
Focused on building strong fundamentals through real-world projects 🚀

---

⭐ If you found this project helpful, consider giving it a star!


