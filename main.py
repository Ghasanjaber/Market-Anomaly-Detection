import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

# Load Dataset
data = pd.read_csv('market_data.csv', header=1)

# Ensure Numeric Conversion
numeric_columns = ['S&P', 'Nasdaq', 'Eurostoxx', 'Gold', 'Brent']
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Fill Missing Values
data[numeric_columns] = data[numeric_columns].ffill().bfill()

# Define Target Column
data['target'] = (data['S&P'].pct_change() < -0.05).astype(int)

# Drop Missing Target Rows
data.dropna(subset=['target'], inplace=True)

# Feature and Target Selection
X = data[numeric_columns]
y = data['target']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle Class Imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Feature Scaling
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Train Model with Balanced Class Weights
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
model.fit(X_train_resampled, y_train_resampled)

# Model Evaluation
y_pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))

# Save Model
joblib.dump(model, 'anomaly_detection_model.pkl')

import numpy as np

# Predict Probabilities
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Adjust Prediction Threshold
threshold = 0.4  # Experiment with thresholds
y_pred_threshold = (y_pred_proba > threshold).astype(int)

print('Accuracy:', accuracy_score(y_test, y_pred_threshold))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred_threshold))
print('Classification Report:\n', classification_report(y_test, y_pred_threshold))



# Step 4: Investment Strategy Backtesting
# Simulated Investment Strategy
def investment_strategy(predictions, returns):
    portfolio = []
    for pred, ret in zip(predictions, returns):
        if pred == 1:
            portfolio.append(0)  # Avoid investment during detected crash
        else:
            portfolio.append(ret)
    return np.sum(portfolio)

# Apply Strategy
strategy_performance = investment_strategy(y_pred_threshold, data['S&P'][-len(y_pred_threshold):])
print('Strategy Performance:', strategy_performance)
from flask import Flask, request, jsonify
import joblib

# Load Model and Scaler
model = joblib.load('anomaly_detection_model.pkl')
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Anomaly Detection API!", "status": "running"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = scaler.transform([data['features']])
        prediction = model.predict(features)
        return jsonify({'prediction': 'crash' if prediction[0] == 1 else 'no crash'})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

