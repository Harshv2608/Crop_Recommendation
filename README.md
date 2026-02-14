# 🌱 Crop Recommendation System (Machine Learning Project)

A Machine Learning project that predicts the most suitable crop to cultivate based on soil nutrients and climatic conditions.

This project trains and compares multiple classification algorithms to determine the best-performing model for crop prediction.

---

## 📌 Project Objective

To build a Machine Learning model that recommends the optimal crop using the following input parameters:

- Nitrogen (N)
- Phosphorous (P)
- Potassium (K)
- Temperature
- Humidity
- pH
- Rainfall

The goal is to assist in **data-driven agricultural decision-making** and support farmers in selecting the most suitable crop for given environmental conditions.

---

## 🛠️ Technologies Used

- **Programming Language:** Python 3
- **Libraries:**
  - NumPy
  - Pandas
  - Scikit-learn
  - XGBoost
  - Matplotlib
  - Seaborn
  - Pickle (for model serialization)
- **Environment:** Jupyter Notebook

---

## 📊 Dataset Information

- **Dataset File:** `Crop_recommendation.csv`
- Located inside the `/data` directory.

The dataset contains seven agricultural parameters:

| Parameter   | Description                         |
|------------|-------------------------------------|
| N          | Nitrogen content in soil            |
| P          | Phosphorous content in soil         |
| K          | Potassium content in soil           |
| Temperature| Temperature (°C)                    |
| Humidity   | Relative Humidity (%)               |
| pH         | Soil pH value                       |
| Rainfall   | Rainfall (mm)                       |

Each record corresponds to a specific crop recommendation.

---

## 🧠 Machine Learning Models Implemented

The following classification models were trained and evaluated:

- Logistic Regression
- Decision Tree Classifier
- Gaussian Naive Bayes
- Random Forest Classifier
- XGBoost Classifier

All trained models are saved in the `/models` directory as `.pkl` files.

---

## 📈 Model Evaluation

The project includes:

- Data preprocessing
- Train-test split
- Accuracy comparison
- Model performance evaluation

📌 **Result:**  
The **Random Forest Classifier** achieved the highest accuracy among the tested models on this dataset.

---

## 📂 Project Structure

```
Crop_Recommendation/
│
├── data/
│   └── Crop_recommendation.csv
│
├── models/
│   ├── DecisionTree.pkl
│   ├── LogisticRegression.pkl
│   ├── NBClassifier.pkl
│   ├── RandomForest.pkl
│   └── XGBoost.pkl
│
├── notebooks/
│   └── code.ipynb
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Harshv2608/Crop_Recommendation.git
cd Crop_Recommendation
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv .venv
```

Activate the environment:

**Windows**
```bash
.venv\Scripts\activate
```

**Mac/Linux**
```bash
source .venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Run Jupyter Notebook

```bash
jupyter notebook
```

Then open:

```
notebooks/code.ipynb
```

---

## 🔍 How to Use a Saved Model

Example of loading a trained model:

```python
import pickle

model = pickle.load(open("models/RandomForest.pkl", "rb"))

sample = [[90, 40, 40, 20, 80, 6.5, 200]]
prediction = model.predict(sample)

print("Recommended Crop:", prediction[0])
```

---

## 🚀 Future Improvements

- Add feature importance visualization  
- Add confusion matrix analysis  
- Perform hyperparameter tuning  
- Build a Flask/Streamlit web interface  
- Deploy as a cloud-based application  
- Add cross-validation and performance metrics (Precision, Recall, F1-score)

---

## 👨‍💻 Author

**Harsh Vardhan**

---

## 📜 License

This project is open-source and available under the MIT License.
