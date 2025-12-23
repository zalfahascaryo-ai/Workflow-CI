import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import os
import dagshub
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

# --- KONFIGURASI DAGSHUB ---
dagshub.init(repo_owner='zalfahascaryo-ai', repo_name='Eksperimen_SML_Annisa-Zalfa')
# Gunakan link tracking yang benar (.mlflow)
mlflow.set_tracking_uri("https://dagshub.com/zalfahascaryo-ai/Eksperimen_SML_Annisa-Zalfa.mlflow")
mlflow.set_experiment("HealthRisk_Classification")

def train_advanced():
    # Load dataset
    df = pd.read_csv(r'D:\DICODING\MSML\Membangun_model\HealthRisk_preprocessing.csv')
    
    # Pastikan health_risk bertipe integer/kategori agar bisa diklasifikasi
    X = df.drop('health_risk', axis=1)
    y = df['health_risk'].astype(int) 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- START MLFLOW RUN ---
    with mlflow.start_run(run_name="Classifier_Manual_Logging"):
        model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 1. Manual Logging Params & Metrics (Klasifikasi)
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred, average='weighted'))
        mlflow.log_metric("precision", precision_score(y_test, y_pred, average='weighted'))
        mlflow.log_metric("recall", recall_score(y_test, y_pred, average='weighted'))

        # 2. Artefak 1: Confusion Matrix Plot
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, cmap='Greens')
        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

        # 3. Artefak 2: Feature Importance Plot
        plt.figure(figsize=(10, 6))
        feat_importances = pd.Series(model.feature_importances_, index=X.columns)
        feat_importances.nlargest(10).plot(kind='barh')
        plt.title("Top 10 Feature Importances")
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")
        plt.close()

        # 4. Artefak 3: Hasil Prediksi CSV
        results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        results.to_csv("test_predictions.csv", index=False)
        mlflow.log_artifact("test_predictions.csv") 

        # 5. Log Model
        mlflow.sklearn.log_model(model, "model_classifier_advanced")

        print("Eksperimen Klasifikasi selesai dan telah terkirim ke DagsHub!")

if __name__ == "__main__":
    train_advanced()
