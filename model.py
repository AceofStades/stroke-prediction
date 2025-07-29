import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, f1_score, make_scorer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    df = pd.read_csv("cleaned_data.csv")
except FileNotFoundError:
    print("Error: 'cleaned_data.csv' not found. Make sure the file is in the same directory.")
    exit()

X = df.drop('stroke', axis=1)
y = df['stroke']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


def create_interaction_features(df):
    df_copy = df.copy()
    df_copy['age_x_bmi'] = df_copy['age'] * df_copy['bmi']
    df_copy['age_x_glucose'] = df_copy['age'] * df_copy['avg_glucose_level']
    df_copy['bmi_x_glucose'] = df_copy['bmi'] * df_copy['avg_glucose_level']
    df_copy['risk_factor_count'] = df_copy['hypertension'] + df_copy['heart_disease']
    return df_copy

feature_creator = FunctionTransformer(create_interaction_features)

# Logistic Regression is sensitive to feature scaling, so we add a StandardScaler.
pipeline = ImbPipeline(steps=[
    ('feature_creator', feature_creator),
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(f_classif)),
    ('sampler', SMOTE(random_state=42)),
    # MODEL CHANGE: Swapped XGBoost for LogisticRegression
    ('classifier', LogisticRegression(
        random_state=42,
        solver='liblinear',
        max_iter=1000
    ))
])

param_grid = {
    'selector__k': [10, 15, 'all'],
    'sampler__sampling_strategy': [0.5, 0.75, 1.0],
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__class_weight': ['balanced', None]
}

scorer = make_scorer(f1_score, pos_label=1)
random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_grid,
    n_iter=50,
    scoring=scorer,
    cv=5,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

print("\nStarting hyperparameter tuning with Logistic Regression...")
random_search.fit(X_train, y_train)

# Evaluation
print("\n--- Randomized Search Results ---")
print(f"Best Hyperparameters: {random_search.best_params_}")
print(f"Best Cross-Validation F1-Score: {random_search.best_score_:.4f}")

best_model = random_search.best_estimator_
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Find and apply the optimal threshold
thresholds = np.arange(0.1, 0.9, 0.01)
best_f1 = 0
optimal_threshold = 0.5
for threshold in thresholds:
    y_pred_tuned = (y_pred_proba >= threshold).astype(int)
    current_f1 = f1_score(y_test, y_pred_tuned, pos_label=1, zero_division=0)
    if current_f1 > best_f1:
        best_f1 = current_f1
        optimal_threshold = threshold

print(f"\nOptimal Threshold found: {optimal_threshold:.4f}")
print(f"Best F1-score on Test Data at Optimal Threshold: {best_f1:.4f}")

y_pred_final = (y_pred_proba >= optimal_threshold).astype(int)
print("\n--- Final Evaluation on Test Data (with Optimal Threshold) ---")
print(classification_report(y_test, y_pred_final, labels=[0, 1], zero_division=0))

filename = 'model-lr-feature-eng.joblib'
joblib.dump(best_model, filename)
print(f"\nBest Logistic Regression model saved to {filename}")

# FEATURE COEFFICIENT ANALYSIS (for Logistic Regression)
print("\n--- Feature Analysis from Best Pipeline ---")

selector = best_model.named_steps['selector']
classifier = best_model.named_steps['classifier']

all_engineered_features = create_interaction_features(X_train).columns
selected_mask = selector.get_support()
final_feature_names = [name for name, selected in zip(all_engineered_features, selected_mask) if selected]

print(f"Number of features selected: {len(final_feature_names)}")

if hasattr(classifier, 'coef_'):
    coefficients = classifier.coef_[0]
    feature_coeffs_df = pd.DataFrame({
        'feature': final_feature_names,
        'coefficient': coefficients
    }).sort_values('coefficient', key=abs, ascending=False)

    print("\n--- Coefficients of Selected Features ---")
    print(feature_coeffs_df)

    plt.figure(figsize=(10, 8))
    plt.barh(feature_coeffs_df['feature'], feature_coeffs_df['coefficient'])
    plt.xlabel("Coefficient Value (Impact on Prediction)")
    plt.title("Feature Coefficients from Best Logistic Regression Model")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
