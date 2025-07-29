# Stroke Prediction Engine 🧠🩺

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Framework](https://img.shields.io/badge/Framework-Streamlit-red.svg)
![Libraries](https://img.shields.io/badge/Libraries-Scikit--learn%20%7C%20Pandas%20%7C%20Imblearn-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A machine learning project designed to predict the likelihood of a patient having a stroke based on various health parameters and demographic data. This repository contains the code for model training, evaluation, and an interactive web application built with Streamlit for real-time predictions.

---

## 🚀 Live Demo

> **Note:** A live version of the app can be deployed on services like Streamlit Community Cloud or Heroku.

**[Link to your deployed Streamlit App]** (<- Replace with your actual deployment URL)

![Streamlit App Screenshot](https://placehold.co/800x500/2B303A/FFFFFF?text=App+Screenshot+Here)
*A preview of the interactive stroke prediction web application.*

---

## ✨ Features

- **Machine Learning Model:** Utilizes a Logistic Regression model within a robust Scikit-learn pipeline to predict stroke risk.
- **Data Imbalance Handling:** Employs SMOTE (Synthetic Minority Over-sampling Technique) to address the severe class imbalance inherent in stroke datasets.
- **Advanced Feature Engineering:** Creates new interaction features (e.g., `age * bmi`) to capture complex relationships between variables.
- **Interactive Web Interface:** A user-friendly web app built with Streamlit that allows users to input their own data and receive an instant risk assessment.
- **Reproducible Pipeline:** The entire preprocessing and modeling workflow is encapsulated in a single `joblib` file, ensuring consistency between training and prediction.
- **Feature Analysis:** The training script provides insights into which factors are most influential in predicting stroke risk.

---

## 🛠️ Technology Stack

- **Language:** Python 3.9+
- **Core Libraries:**
  - **Pandas & NumPy:** For data manipulation and numerical operations.
  - **Scikit-learn:** For building the machine learning pipeline, feature scaling, and model training.
  - **Imbalanced-learn:** For handling class imbalance with SMOTE.
  - **Joblib:** For saving and loading the trained model pipeline.
- **Web Framework:**
  - **Streamlit:** For creating and serving the interactive web application.
- **Plotting:**
  - **Matplotlib:** For visualizing feature importance/coefficients.

---

## 📂 Directory Structure

```
stroke-prediction/
├── 📄 .gitignore
├── 🐍 app.py                  # The Streamlit web application script
├── 📄 cleaned_data.csv        # The pre-processed dataset used for training
├── 🐍 model.py                # The script for training, evaluating, and saving the model
├── 📦 model-lr-feature-eng.joblib # The saved, trained model pipeline
├── 📄 README.md               # You are here!
└── 📄 requirements.txt        # List of Python dependencies
```

---

## ⚙️ Installation & Setup

Follow these steps to set up the project environment and run it on your local machine.

### 1. Prerequisites

- Python 3.9 or higher
- `pip` package manager

### 2. Clone the Repository

```bash
git clone [https://github.com/your-username/stroke-prediction.git](https://github.com/your-username/stroke-prediction.git)
cd stroke-prediction
```

### 3. Create a Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies.

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

Install all the required libraries from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Running the Web Application

To start the interactive Streamlit application, run the following command in your terminal:

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your web browser to use the app.

### Re-training the Model

If you make changes to the modeling process or want to retrain the model on new data, you can run the `model.py` script. This will perform the entire training and evaluation process and save a new `model-lr-feature-eng.joblib` file.

```bash
python model.py
```

---

## 🧠 Modeling Process

The machine learning model was developed through a structured process:

1.  **Data Source:** The model was trained on the [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset), which was pre-cleaned and saved as `cleaned_data.csv`.
2.  **Feature Engineering:** A custom function `create_interaction_features` was used via `FunctionTransformer` to generate new features that capture synergistic effects between variables like age, BMI, and glucose levels.
3.  **Preprocessing Pipeline:** An `ImbPipeline` was constructed to streamline the workflow. It sequentially performs:
    - Custom feature creation.
    - **Scaling:** `StandardScaler` to normalize feature values, which is crucial for Logistic Regression.
    - **Feature Selection:** `SelectKBest` to choose the most relevant features.
    - **Oversampling:** `SMOTE` to generate synthetic samples for the minority class (stroke=1), mitigating bias.
4.  **Model Training:** A **Logistic Regression** classifier was chosen for its interpretability and efficiency.
5.  **Hyperparameter Tuning:** `RandomizedSearchCV` was used to find the optimal combination of hyperparameters for the feature selector, SMOTE, and the classifier, optimizing for the F1-score.
6.  **Evaluation:** The model's performance was evaluated using a detailed `classification_report` and the F1-score, which is a suitable metric for imbalanced datasets.

---

## 🤝 Contributing

Contributions are welcome! If you have ideas for improvements or find any issues, please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/YourAmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/YourAmazingFeature`).
5.  Open a Pull Request.

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

*This README was generated on July 29, 2025.*
