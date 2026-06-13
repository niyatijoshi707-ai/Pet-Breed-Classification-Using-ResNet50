# 🐾 Pets Classification — Oxford-IIIT Pet Dataset

<div align="center">

<!-- Cute cat and dog playing together -->
<img src="https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExZ2I4dG9jemNwNncyc3QyOGFvc3g0aW5xZWg4b2ZreHBzMHBvdTJnNCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/MwR4Q2mryREZ2/giphy.gif" width="420" alt="Cat and Dog Playing Together"/>

<br/>

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-ResNet50-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-2ecc71?style=for-the-badge)

**Classifying 37 cat and dog breeds using Transfer Learning on ResNet50**

</div>

---

## 🎯 Objective

The goal of this project is to **automatically identify the breed** of a cat or dog from its photo. Given a picture of a pet, the model returns one of 37 possible breed labels — from Abyssinian cats to Yorkshire Terriers.

This is a **fine-grained image classification** problem: the images look similar at first glance, but the model must learn the subtle visual differences between breeds (ear shape, coat texture, facial structure).

---

## 📂 Dataset

| Property | Detail |
|---|---|
| **Source** | Oxford-IIIT Pet Dataset (via Kaggle) |
| **Total Images** | ~7,400 JPEG images |
| **Classes** | 37 breeds (25 dog + 12 cat) |
| **Image Size** | Resized to 224 × 224 |
| **Split** | 60% Train / 20% Validation / 20% Test |

---

## 🧠 Why ResNet50 (Transfer Learning)?

Training a deep CNN from scratch on 7,400 images would **overfit badly** — there simply isn't enough data. Transfer Learning solves this elegantly:

> ResNet50 was pre-trained on **ImageNet** (1.2 million images, 1000 classes). It already "knows" edges, textures, shapes, and high-level visual patterns. We freeze those weights and attach a new final layer trained only for our 37 pet breeds.

This gives us the power of a massive model at a fraction of the training cost.

---

## 🏗️ Model Architecture

```
Input Image (224 × 224 × 3)
        │
        ▼
 Data Augmentation
 (RandomFlip, RandomRotation)
        │
        ▼
 ResNet50 Backbone (frozen, ImageNet weights)
        │
        ▼
   GlobalAveragePooling2D
        │
        ▼
     Dropout (0.2)
        │
        ▼
Dense(37, activation='softmax')  ← Our custom head
        │
        ▼
  Predicted Breed (0–36)
```

---

## ⚙️ What Each Piece Does & Why

| Component | What it is | Why it's used here |
|---|---|---|
| `ResNet50` | Deep CNN with residual skip-connections, 50 layers | Pre-trained on ImageNet; avoids training from scratch |
| `include_top=False` | Removes ResNet's final classification layer | We replace it with our own 37-class head |
| `pooling='avg'` | Global Average Pooling after the backbone | Reduces feature maps to a 1D vector without flattening manually |
| `Data Augmentation` | Random flips & rotations during training | Artificially increases variety, prevents overfitting |
| `Dropout(0.2)` | Randomly zeroes 20% of neurons during training | Regularization — forces the model not to rely on any single feature |
| `Adam Optimizer` | Adaptive gradient descent | Fast convergence, handles sparse gradients well |
| `CategoricalCrossentropy` | Multi-class log loss | Correct loss function for one-hot encoded labels |
| `pp_i` (preprocess_input) | ResNet50's specific normalization | Ensures pixel values match what ResNet expects (mean subtraction) |

---

## 📊 Training Pipeline

```python
# 1. Load & resize all images to 224×224
# 2. One-hot encode 37 breed labels
# 3. Split: 60 / 20 / 20
# 4. Build model: augmentation → ResNet50 (frozen) → Dropout → Dense(37)
# 5. Compile with Adam + CategoricalCrossentropy
# 6. Train for 10 epochs, monitoring val_accuracy
# 7. Evaluate on unseen test set
```

---

## 📈 Results

| Metric | Value |
|---|---|
| Training Accuracy | Tracked per epoch |
| Validation Accuracy | Tracked per epoch |
| Test Set Evaluation | `model.evaluate(X_test, y_test)` |

Training and validation accuracy/loss curves are plotted side-by-side to detect overfitting.

---

## 🗂️ Project Structure

```
pets_classification/
├── pets_classification_Niyati_Joshi.ipynb   # Main notebook
└── README.md
```

---

## 🛠️ Tech Stack

- **Python 3.10**
- **TensorFlow / Keras** — model building & training
- **ResNet50** — transfer learning backbone
- **NumPy** — array operations
- **Matplotlib** — training curve visualization
- **Scikit-learn** — train/val/test splitting

---

## 🚀 How to Run

```bash
# 1. Open on Kaggle or locally
jupyter notebook pets_classification_Niyati_Joshi.ipynb

# 2. Dataset path (Kaggle)
/kaggle/input/datasets/tanlikesmath/the-oxfordiiit-pet-dataset/images/

# 3. Run all cells top to bottom
```

---

## 👩‍💻 Author

**Niyati Joshi** — ML & Deep Learning enthusiast building real-world computer vision systems.

---

<div align="center"><i>Built with curiosity, trained with patience.</i></div>
