import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

data = pd.read_csv('data/health_data.csv')

print("Columns in the dataset:", data.columns)

if 'risk' in data.columns:
    X = data.drop('risk', axis=1)  
    y = data['risk'] 
else:
    raise ValueError("Column 'risk' not found in the dataset.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = RandomForestClassifier()
model.fit(X_train, y_train)


joblib.dump(model, 'model/model.pkl')

print("Model trained and saved successfully.")
