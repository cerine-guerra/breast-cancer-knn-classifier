# breast-cancer-knn-classifier
Breast Cancer KNN Classifier – A machine learning project that uses a K-Nearest Neighbors algorithm to predict whether a tumor is benign or malignant based on 30 features from the UCI Breast Cancer dataset with 0.93 accurancy .
Includes preprocessing, oversampling, scaling, model training, evaluation, and ready-to-use saved model files. 


# Breast Cancer KNN Classifier

This project implements a **K-Nearest Neighbors (KNN)** machine learning model to classify breast tumors as **benign** or **malignant** based on features from the [UCI Breast Cancer dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)).

The project includes:

- Data preprocessing (handling features, encoding labels)
- Random oversampling to balance classes in the training set
- Feature scaling using `StandardScaler`
- Distance-weighted KNN classification
- Model evaluation using accuracy, precision, recall, F1-score, confusion matrix, and PCA visualization
- Saved trained model and scaler for use in other applications

---

## 🗂 Repository Structure


---

## ⚡ How to Use

1. **Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/breast-cancer-knn.git
cd breast-cancer-knn




Install dependencies

pip install -r requirements.txt

3 - load the model in python
import joblib
import numpy as np

# Load saved model and scaler
model = joblib.load("model/knn_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# Example new patient (replace with real feature values)
new_data = np.array([[value1, value2, ..., value30]])
new_data_scaled = scaler.transform(new_data)

# Predict
prediction = model.predict(new_data_scaled)
print("Prediction:", "Benign" if prediction[0]==0 else "Malignant")
