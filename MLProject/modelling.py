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

# Experiment
mlflow.set_experiment("Titanic_Experiment")

# Autolog
mlflow.sklearn.autolog()


def train_model():

    base_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    dataset_path = os.path.join(
        base_dir,
        "cleaned_dataset.csv"
    )

    print("Dataset ditemukan di:")
    print(dataset_path)

    # Load dataset
    df = pd.read_csv(dataset_path)

    print("\nShape Dataset:")
    print(df.shape)

    # Feature dan target
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # Training
    model.fit(X_train, y_train)

    # Evaluasi
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

    print("\nTraining selesai.")
    print("MLflow Autolog berhasil mencatat parameter, metric, model, dan artifact.")


if __name__ == "__main__":
    train_model()