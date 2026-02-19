# Age Prediction using ResNet50 (UTKFace)

This project predicts human age from face images using transfer learning with ResNet50 and TensorFlow/Keras.

## 📌 Project Overview

- Dataset: UTKFace  
- Task: Age regression  
- Backbone: ResNet50 (ImageNet pretrained)  
- Loss: Mean Squared Error (MSE)  
- Metric: Mean Absolute Error (MAE)  

---

## 🏗 Model Architecture

- ResNet50 (include_top=False)
- GlobalAveragePooling2D
- Dense(256) + ReLU
- BatchNormalization
- Dropout(0.3)
- Dense(128) + ReLU
- Dropout(0.2)
- Dense(1) — Linear output (age)

---

## 🔄 Training Strategy

**Phase 1:**  
- Freeze base model  
- Train custom head  
- LR = 1e-3  

**Phase 2 (Fine-tuning):**  
- Unfreeze last 10 layers  
- LR = 1e-5  

EarlyStopping and ReduceLROnPlateau are used.

---

## 🎨 Data Augmentation

- Rotation  
- Shift (width/height)  
- Shear  
- Zoom  
- Horizontal flip  
- Brightness adjustment  

---

## 🧪 Inference Example

```python
img = load_img(img_path, target_size=(224, 224))
img = img_to_array(img)
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)

predicted_age = model.predict(img)[0][0]
print("Predicted Age:", predicted_age)
```

---

## 🚀 Requirements

```
tensorflow
pandas
numpy
matplotlib
scikit-learn
```

---

## 📜 Notes

This project is for educational and research purposes.
