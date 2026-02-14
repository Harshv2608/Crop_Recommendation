🌱 Crop Recommendation SystemA Machine Learning-powered system designed to help farmers optimize their agricultural yield. By analyzing soil nutrients and climatic conditions, this project recommends the most suitable crop to cultivate.📌 Project ObjectiveThe primary goal of this project is to provide data-driven recommendations for precision agriculture. By inputting environmental and soil data, the system predicts the crop that has the highest probability of success.Key Parameters Analyzed:Nitrogen (N), Phosphorous (P), & Potassium (K): Essential soil nutrients.Temperature & Humidity: Climatic conditions.pH Level: Soil acidity/alkalinity.Rainfall: Water availability.📊 Dataset OverviewThe system is trained on the Crop_recommendation.csv dataset, which includes:ParameterDescriptionUnitNNitrogen content in soilRatioPPhosphorous content in soilRatioKPotassium content in soilRatioTemperatureAmbient Temperature°CHumidityRelative Humidity%pHSoil pH value0 - 14RainfallAnnual Rainfallmm🧠 Machine Learning ModelsWe implemented and compared multiple classification algorithms to ensure the highest accuracy:Logistic RegressionDecision Tree ClassifierGaussian Naive BayesRandom Forest Classifier (Top Performer)XGBoost Classifier📂 Project StructurePlaintext├── data/
│   └── Crop_recommendation.csv
├── models/
│   ├── DecisionTree.pkl
│   ├── LogisticRegression.pkl
│   ├── NBClassifier.pkl
│   ├── RandomForest.pkl
│   └── XGBoost.pkl
├── notebooks/
│   └── code.ipynb
├── requirements.txt
├── .gitignore
└── README.md
⚙️ Installation & Setup1. Clone the RepositoryBashgit clone https://github.com/Harshv2608/Crop_Recommendation.git
cd Crop_Recommendation
2. Create Virtual EnvironmentWindows:Bashpython -m venv .venv
.venv\Scripts\activate
Mac/Linux:Bashpython -m venv .venv
source .venv/bin/activate
3. Install DependenciesBashpip install -r requirements.txt
🔍 UsageYou can load the pre-trained models using pickle to make instant predictions:Pythonimport pickle

# Load the Random Forest model
model = pickle.load(open("models/RandomForest.pkl", "rb"))

# Format: [N, P, K, Temp, Humidity, pH, Rainfall]
sample_input = [[90, 40, 40, 20, 80, 6.5, 200]]
prediction = model.predict(sample_input)

print(f"Recommended Crop: {prediction[0]}")
🚀 Future Roadmap[ ] Develop a Flask or Streamlit web dashboard.[ ] Add Feature Importance visualizations.[ ] Implement Hyperparameter tuning for XGBoost.[ ] Real-time weather API integration.👨‍💻 AuthorHarsh Vardhan Passionate about AI and Sustainable Agriculture.