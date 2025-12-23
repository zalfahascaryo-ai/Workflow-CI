import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error

# 1. Aktifkan Autolog
mlflow.autolog()

def train_model():
    # 2. Load dataset hasil preprocessing (sesuaikan nama filenya)
    df = pd.read_csv('HealthRisk_preprocessing.csv')
    
    # Pisahkan Fitur dan Target (Contoh: target kolom 'health_risk')
    X = df.drop('health_risk', axis=1)
    y = df['health_risk']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Mulai MLflow Run
    with mlflow.start_run(run_name="Baseline_Model"):
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        print(f"Model trained with MSE: {mse}")

if __name__ == "__main__":
    train_model()

