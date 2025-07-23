import numpy as np
import pandas as pd

# Models
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier

df = pd.read_csv("cleaned_data.csv")
df = df.drop(columns=['Unnamed: 0'])

X = df.drop('stroke', axis = 1)
y = df['stroke']

np.random.seed(50)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

models = {
	SVC(),
	LogisticRegression(n_jobs=-1),
	KNeighborsClassifier(n_jobs=-1, n_neighbors=3),
	RandomForestClassifier(n_jobs=-1, n_estimators=1000),
	XGBClassifier(Device='cuda')
}

def hyperparamSVC(X_train, y_train):
	param_grid = {'C': [0.1, 1, 10, 100, 1000],
				'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
				'kernel': ['rbf']}

	models = {
		SVC(),
	}

	for model in models:
		grid = GridSearchCV(model, param_grid, refit = True, verbose = 0)

		grid.fit(X_train, y_train)

		print("\n\n")
		print(model)
		print(grid.best_params_)
		print(grid.best_estimator_)

def hyperparamLogisticReg(X_train, y_train):
	param_grid = {
    'penalty':['l1','l2','elasticnet','none'],
    'C' : np.logspace(-4,4,20),
    'solver': ['lbfgs','newton-cg','liblinear','sag','saga'],
    'max_iter'  : [10,100,1000,10000]
	}

	models = {
		LogisticRegression(),
	}

	for model in models:
		grid = GridSearchCV(model, param_grid=param_grid, cv=5, verbose=0)

		grid.fit(X_train, y_train)

		print("\n\n")
		print(model)
		print(grid.best_params_)
		print(grid.best_estimator_)

def hyperparamRF(X_train, y_train):
	param_grid = {
	    'n_estimators': [100, 200, 500, 1000],
	    'max_depth': [None, 10, 20],
	    'min_samples_split': [2, 5],
	    'min_samples_leaf': [1, 2],
	    'bootstrap': [True, False]
	}

	models = {
		RandomForestClassifier(n_jobs=-1, n_estimators=1000)
	}

	for model in models:
		grid = GridSearchCV(model, param_grid=param_grid, cv=5)

		grid.fit(X_train, y_train)

		print("\n\n")
		print(model)
		print(grid.best_params_)
		print(grid.best_estimator_)

def hyperparamknn(X_train, y_train):
	param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

	models = {
		KNeighborsClassifier()
	}

	for model in models:
		grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=3)

		grid.fit(X_train, y_train)

		print("\n\n")
		print(model)
		print(grid.best_params_)
		print(grid.best_estimator_)

def hyperparamxgb(X_train, y_train):
	neg_count = y_train.value_counts()[0]
	pos_count = y_train.value_counts()[1]
	scale_pos_weight_value = neg_count / pos_count
	print(f"Calculated scale_pos_weight: {scale_pos_weight_value:.2f}")

	xgb_model = XGBClassifier(
	    objective='binary:logistic',
	    eval_metric='logloss',
	    use_label_encoder=False,
	    tree_method='gpu_hist',
	    predictor='gpu_predictor',
	    scale_pos_weight=scale_pos_weight_value,
	    random_state=42
	)

	param_grid_xgb = {
	    'n_estimators': [100, 200, 300],
	    'learning_rate': [0.01, 0.05, 0.1],
	    'max_depth': [3, 5, 7],
	    'subsample': [0.7, 0.8, 1.0],
	    'colsample_bytree': [0.7, 0.8, 1.0],
	    'gamma': [0, 0.1, 0.2]
	}

	print("\nStarting GridSearchCV for XGBoost with GPU support...")
	grid = GridSearchCV(
	    estimator=xgb_model,
	    param_grid=param_grid_xgb,
	    cv=3,                       # Number of cross-validation folds
	    scoring='f1',               # Use F1-score due to class imbalance, or 'roc_auc'
	    n_jobs=-1,                  # Use all available CPU cores for parallelism across parameter sets
	    verbose=2                   # Verbosity level
	)

	grid.fit(X_train, y_train)

	print("\n\n")
	print(xgb_model)
	print(grid.best_params_)
	print(grid.best_estimator_)

# hyperparamSVC(X_train=X_train, y_train=y_train)
# hyperparamRF(X_train=X_train, y_train=y_train)
# hyperparamLogisticReg(X_train, y_train)
# hyperparamknn(X_train, y_train)
# hyperparamxgb(X_train, y_train)
