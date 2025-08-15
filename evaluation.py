
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from ucimlrepo import fetch_ucirepo
import joblib
import json
import numpy as np

def evaluate_models(format_type="json"):
    # Load test data and scaler
    concrete_data = fetch_ucirepo(id=165)
    X = concrete_data.data.features
    y = concrete_data.data.targets

    X.columns = [
        'Cement', 'Slag', 'Fly Ash', 'Water', 'Superplasticizer',
        'Coarse Aggregate', 'Fine Aggregate', 'Age'
    ]
    y.columns = ['Strength']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = joblib.load("scaler.pkl")
    X_test_scaled = scaler.transform(X_test)

    # Load models
    lr = joblib.load("linear_regression.pkl")
    rf = joblib.load("random_forest.pkl")
    mlp = joblib.load("neural_network.pkl")

    # Predictions
    lr_pred = lr.predict(X_test_scaled)
    rf_pred = rf.predict(X_test)
    nn_pred = mlp.predict(X_test_scaled)

    # Metrics
    results = {
        "Linear Regression": {
            "R2 Score": round(r2_score(y_test, lr_pred), 2),
            "RMSE": round(np.sqrt(mean_squared_error(y_test, lr_pred)), 2),
            "MAE": round(mean_absolute_error(y_test, lr_pred), 2)
        },
        "Random Forest": {
            "R2 Score": round(r2_score(y_test, rf_pred), 2),
            "RMSE": round(np.sqrt(mean_squared_error(y_test, rf_pred)), 2),
            "MAE": round(mean_absolute_error(y_test, rf_pred), 2)
        },
        "Neural Network": {
            "R2 Score": round(r2_score(y_test, nn_pred), 2),
            "RMSE": round(np.sqrt(mean_squared_error(y_test, nn_pred)), 2),
            "MAE": round(mean_absolute_error(y_test, nn_pred), 2)
        }
    }

    if format_type == "json":
        return json.dumps(results)
    elif format_type == "html":
        df = pd.DataFrame({
            "Model": ["Linear Regression", "Random Forest", "Neural Network"],
            "R² Score": [
                results["Linear Regression"]["R2 Score"],
                results["Random Forest"]["R2 Score"],
                results["Neural Network"]["R2 Score"]
            ],
            "RMSE (MPa)": [
                results["Linear Regression"]["RMSE"],
                results["Random Forest"]["RMSE"],
                results["Neural Network"]["RMSE"]
            ],
            "MAE (MPa)": [
                results["Linear Regression"]["MAE"],
                results["Random Forest"]["MAE"],
                results["Neural Network"]["MAE"]
            ]
        })
        return df.to_html(index=False, classes="table table-striped", border=0)
