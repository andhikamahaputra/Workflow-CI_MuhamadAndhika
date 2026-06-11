import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
print("Tracking URI:", mlflow.get_tracking_uri())
mlflow.set_experiment("Titanic_Experiment")
mlflow.sklearn.autolog()


def train_model():

    BASE_DIR = os.path.dirname(
        os.path.abspath(__file__)
    )

    DATASET_PATH = os.path.join(
        BASE_DIR,
        "cleaned_dataset.csv"
    )

    print("Dataset ditemukan di:")
    print(DATASET_PATH)

    df = pd.read_csv(DATASET_PATH)

    print("\nShape Dataset:")
    print(df.shape)

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n=== HASIL EVALUASI ===")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    mlflow.log_metric(
        "accuracy_manual",
        accuracy
    )

    mlflow.log_metric(
        "precision_manual",
        precision
    )

    mlflow.log_metric(
        "recall_manual",
        recall
    )

    mlflow.log_metric(
        "f1_manual",
        f1
    )

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )

    print("\nModel berhasil tersimpan ke MLflow")


if __name__ == "__main__":
    train_model()