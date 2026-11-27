import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)


def prepare_data(data_path):
    """
    Load and preprocess the customer churn dataset.
    """

    # 1. Load the dataset
    df = pd.read_csv(data_path)

    # 2. Encode the Gender column
    encoder = LabelEncoder()
    df["Gender"] = encoder.fit_transform(df["Gender"])

    # 3. Remove unnecessary columns
    df = df.drop(["Surname", "Geography"], axis=1)

    # 4. Separate features (X) and target (y)
    X = df.drop(["Exited"], axis=1)
    y = df["Exited"]

    # 5. Remove identifier columns
    X = X.drop(["RowNumber", "CustomerId"], axis=1)

    # 6. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Train a Random Forest classification model.
    """

    # 1. Create the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # 2. Train the model
    model.fit(X_train, y_train)

    # 3. Return the trained model
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model using multiple metrics.
    """

    # 1. Make predictions
    y_pred = model.predict(X_test)

    # 2. Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # 3. Calculate ROC-AUC
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    # 4. Calculate confusion matrix
    matrix = confusion_matrix(y_test, y_pred)

    # 5. Display results
    print("\n===== MODEL EVALUATION =====")
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1-score:  {f1 * 100:.2f}%")
    print(f"ROC-AUC:   {auc:.4f}")

    print("\n===== CLASSIFICATION REPORT =====")
    print(classification_report(y_test, y_pred, target_names=["Not Exited", "Exited"]))

    print("\n===== CONFUSION MATRIX =====")
    print(matrix)

    return accuracy, matrix


def save_model(model, model_path):
    """
    Save the trained model using Joblib.
    """

    # Save the model
    joblib.dump(model, model_path)

    print("Model saved successfully at:", model_path)


def load_model(model_path):
    """
    Load a previously saved model using Joblib.
    """

    # Load the saved model
    model = joblib.load(model_path)

    print("Model loaded successfully from:", model_path)

    return model
