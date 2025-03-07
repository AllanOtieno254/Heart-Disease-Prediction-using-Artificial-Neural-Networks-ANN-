# Heart Disease Prediction using Artificial Neural Networks (ANN)

## 📌 Project Overview
This project aims to develop an **Artificial Neural Network (ANN) model** to predict the likelihood of heart disease based on multiple health-related features. The dataset used contains key health indicators such as age, cholesterol levels, blood pressure, and more. The goal is to build a model that assists healthcare professionals in early diagnosis and risk assessment.

---

## 📊 Dataset Information
- **Name:** Heart Disease Dataset
- **Source:** UCI Machine Learning Repository
- **Features:**
  - Age
  - Sex
  - Chest Pain Type (CP)
  - Resting Blood Pressure (trestbps)
  - Serum Cholesterol (chol)
  - Fasting Blood Sugar (fbs)
  - Resting Electrocardiographic Results (restecg)
  - Maximum Heart Rate Achieved (thalach)
  - Exercise Induced Angina (exang)
  - ST Depression Induced by Exercise (oldpeak)
  - Slope of the Peak Exercise ST Segment (slope)
  - Number of Major Vessels (ca)
  - Thalassemia (thal)
  - **Target Variable:** Presence of heart disease (1 = Yes, 0 = No)

---

## 🚀 Technologies Used
- **Python** (Pandas, NumPy, Matplotlib, Scikit-learn, TensorFlow/Keras)
- **Jupyter Notebook/Google Colab**
- **Artificial Neural Networks (ANN)**
- **Data Preprocessing Techniques**
- **Model Evaluation Metrics**

---

## ⚙️ Installation & Setup
To run this project, follow these steps:

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/your-username/Heart-Disease-Prediction-ANN.git
cd Heart-Disease-Prediction-ANN
```

### **2️⃣ Create a Virtual Environment (Optional but Recommended)**
```bash
python -m venv env
source env/bin/activate  # On Mac/Linux
env\Scripts\activate  # On Windows
```

### **3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## 📖 Project Workflow
1. **Data Preprocessing**
   - Load dataset
   - Handle missing values
   - Feature scaling using StandardScaler
   - Train-test split
2. **Model Training**
   - Define an ANN model using TensorFlow/Keras
   - Compile and train the model
   - Evaluate performance using accuracy and loss
3. **Model Prediction**
   - Save the trained model
   - Load the model for predictions
   - Predict heart disease risk on new data

---

## 🏗️ Model Architecture
```python
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.1),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))
```

---

## 📊 Model Evaluation
- Training Accuracy: **~95%**
- Validation Accuracy: **~90%**
- Loss Function: Binary Crossentropy

---

## 📈 Results
- The model successfully predicts heart disease risk with high accuracy.
- It identifies key risk factors contributing to heart disease.
- Future improvements: Fine-tuning hyperparameters, adding more layers, testing with larger datasets.

---

## 📌 How to Use the Model for Predictions
### **Step 1: Load the Trained Model**
```python
from tensorflow.keras.models import load_model
model = load_model("models/heart_disease_model.h5")
```

### **Step 2: Prepare New Data**
```python
import numpy as np
new_patient = np.array([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]])  # Example input
data_scaled = scaler.transform(new_patient)
```

### **Step 3: Make Predictions**
```python
prediction = model.predict(data_scaled)
predicted_class = (prediction > 0.5).astype(int)
print("Predicted Heart Disease Risk:", predicted_class)
```

---

## 📜 License
This project is licensed under the **MIT License**.

---

## 🤝 Contributing
We welcome contributions! Feel free to fork the repository and submit pull requests.

---

## 📬 Contact
- **Author:** Allan Otieno Akumu
- **GitHub:** [AllanOtieno254](https://github.com/AllanOtieno254)
- **Email:** allanotieno2001@gmail.com

