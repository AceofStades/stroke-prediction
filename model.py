import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Models
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

df = pd.read_csv("cleaned_data.csv")
df = df.drop(columns=['Unnamed: 0'])

X = df.drop('stroke', axis = 1)
y = df['stroke']

np.random.seed(24)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

models = {
	XGBClassifier(colsample_bytree = 0.7, gamma= 0.1, learning_rate= 0.01, max_depth= 5, n_estimators= 200, subsample= 0.8),
	RandomForestClassifier(n_estimators=500, n_jobs=-1, bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=5),
	SVC(C=0.1, gamma=1, kernel='rbf'),
	LogisticRegression(C=np.float64(0.08858667904100823), solver='newton-cg', max_iter=1000, penalty='l2'),
	KNeighborsClassifier(metric='euclidean', n_neighbors=5, weights='uniform')
}

def train():
	for model in models:
		model.fit(X_train, y_train)

def test():
	y = dict()
	report = dict()
	for model in models:
		y = model.predict(X_test)
		report[model] = classification_report(y_test, y)
	for model in models:
		print(report[model])


train()
# test()
