import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

# Import the models and pipeline tools
import xgboost as xgb
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN

# --------------------------------------------------------------------------
# 1. DATA PREPARATION
# --------------------------------------------------------------------------

# Load your cleaned dataset
try:
    df = pd.read_csv("cleaned_data.csv")
except FileNotFoundError:
    print("Error: 'cleaned_data.csv' not found. Make sure the file is in the same directory.")
    exit()

# Drop the old index column if it exists
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Define features (X) and target (y)
X = df.drop('stroke', axis=1)
y = df['stroke']

# Split the data into training and testing sets, ensuring balanced classes in both
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,     # 20% of data will be for testing
    stratify=y,        # Ensures same proportion of stroke/no-stroke in train/test
    random_state=42    # For reproducible results
)

# --------------------------------------------------------------------------
# 2. MODEL PIPELINE AND HYPERPARAMETER TUNING
# --------------------------------------------------------------------------

# Define the steps for the machine learning pipeline
# Step 1: Resample the data using SMOTEENN to handle class imbalance
# Step 2: Train the XGBoost Classifier
pipeline = Pipeline(steps=[
    ('sampler', SMOTEENN(random_state=42)),
    ('classifier', xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))
])

# Define the grid of hyperparameters to search through
# The model will be tested with each combination of these settings
param_grid = {
    'classifier__max_depth': [3, 5, 7],
    'classifier__n_estimators': [100, 200, 300],
    'classifier__learning_rate': [0.01, 0.1, 0.2]
}

# Set up the grid search
# This will automatically handle resampling within each cross-validation fold
# to prevent data leakage and find the best model robustly.
# We are optimizing for the F1-score of the positive class.
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='f1', # Focus on the F1 score for the minority class
    cv=5,         # 5-fold cross-validation
    n_jobs=-1,    # Use all available CPU cores
    verbose=1     # Show progress
)

# Execute the grid search on the training data
print("Starting hyperparameter tuning with GridSearchCV...")
grid_search.fit(X_train, y_train)

# --------------------------------------------------------------------------
# 3. EVALUATION
# --------------------------------------------------------------------------

# Print the best parameters and the best cross-validation F1 score
print("\n--- Grid Search Results ---")
print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Best Cross-Validation F1-Score: {grid_search.best_score_:.4f}")

# Use the best model found by the grid search to make predictions on the unseen test data
print("\n--- Final Evaluation on Test Data ---")
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Print the final classification report
print(classification_report(y_test, y_pred, labels=[0, 1], zero_division=0))

filename = 'model-xgb.joblib'
joblib.dump(best_model, filename)
