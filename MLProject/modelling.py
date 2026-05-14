import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import json
from mlflow.models import infer_signature

dagshub_token = "your_dagshub_token_here"
dagshub_username = "45rabbitto"
dagshub_repo = "Eksperimen_SML_Desi-Triana"

mlflow.set_tracking_uri(f"https://{dagshub_username}:{dagshub_token}@dagshub.com/{dagshub_username}/{dagshub_repo}.mlflow")

df = pd.read_csv('data_preprocessed/diamonds_clean.csv')

X = df.drop(['price', 'price_log'], axis=1)
y = df['price_log']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_param_dist = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4]
}

model = RandomForestRegressor(random_state=42, n_jobs=-1, n_estimators=50)
random_search = RandomizedSearchCV(model, rf_param_dist, n_iter=5, cv=3, scoring='r2', random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_

y_pred_test_log = best_model.predict(X_test)
y_test_original = np.expm1(y_test)
y_pred_test_original = np.expm1(y_pred_test_log)

test_r2 = r2_score(y_test_original, y_pred_test_original)
test_rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_test_original))
test_mae = mean_absolute_error(y_test_original, y_pred_test_original)

signature = infer_signature(X_train, best_model.predict(X_train))

with mlflow.start_run(run_name="Random_Forest_Tuned"):
    mlflow.log_params(random_search.best_params_)
    mlflow.log_metrics({
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'best_cv_r2': random_search.best_score_
    })
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(y_test_original, y_pred_test_original, alpha=0.3, s=5)
    ax.plot([y_test_original.min(), y_test_original.max()], [y_test_original.min(), y_test_original.max()], 'r--', lw=2)
    ax.set_xlabel('Actual Price (USD)')
    ax.set_ylabel('Predicted Price (USD)')
    ax.set_title(f'Random Forest - Actual vs Predicted\nR2: {test_r2:.4f}, RMSE: ${test_rmse:,.0f}')
    plt.tight_layout()
    plt.savefig('training_confusion_matrix.png')
    mlflow.log_artifact('training_confusion_matrix.png')
    plt.close()
    os.remove('training_confusion_matrix.png')
    
    estimator_info = f"""
    <html>
    <body>
    <h2>Random Forest Regressor - Diamond Price Prediction</h2>
    <h3>Best Parameters:</h3>
    <ul>
    """
    for key, value in random_search.best_params_.items():
        estimator_info += f"<li>{key}: {value}</li>"
    
    estimator_info += f"""
    </ul>
    <h3>Performance Metrics:</h3>
    <ul>
    <li>Test R2: {test_r2:.4f}</li>
    <li>Test RMSE: ${test_rmse:,.2f}</li>
    <li>Test MAE: ${test_mae:,.2f}</li>
    <li>Best CV R2: {random_search.best_score_:.4f}</li>
    </ul>
    </body>
    </html>
    """
    
    with open('estimator.html', 'w') as f:
        f.write(estimator_info)
    mlflow.log_artifact('estimator.html')
    os.remove('estimator.html')
    
    metric_info = {
        "model": "Random Forest",
        "best_params": random_search.best_params_,
        "best_cv_r2": random_search.best_score_,
        "test_r2": test_r2,
        "test_rmse": test_rmse,
        "test_mae": test_mae
    }
    
    with open('metric_info.json', 'w') as f:
        json.dump(metric_info, f, indent=4)
    mlflow.log_artifact('metric_info.json')
    os.remove('metric_info.json')
    
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        signature=signature,
        input_example=X_train.iloc[:5]
    )

print("Preprocessing completed!")
print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")