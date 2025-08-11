from pathlib import Path
import json
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import numpy as np

def run_pipeline():
    # 1) Datos
    X, y = fetch_california_housing(return_X_y=True, as_frame=True)
    # 2) Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 3) Modelo baseline
    model = LinearRegression()
    model.fit(X_train, y_train)
    # 4) Evaluación
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)
    # 5) Reporte
    metrics = {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}
    Path("reports").mkdir(parents=True, exist_ok=True)
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics:", metrics)

if __name__ == "__main__":
    run_pipeline()
