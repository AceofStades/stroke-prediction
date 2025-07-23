import joblib

model = joblib.load('model-xgb.joblib')
print(model.predict((1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)))
