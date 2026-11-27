"""Tests unitaires de model_pipeline.py (donnees synthetiques)."""

import pandas as pd
import pytest

from model_pipeline import prepare_data, train_model, save_model, load_model


@pytest.fixture
def csv_path(tmp_path):
    """Cree un petit CSV avec les memes colonnes que Churn_Modelling.csv."""
    n = 50
    df = pd.DataFrame(
        {
            "RowNumber": range(1, n + 1),
            "CustomerId": range(1000, 1000 + n),
            "Surname": ["Test"] * n,
            "CreditScore": [600 + i for i in range(n)],
            "Geography": ["France", "Spain"] * (n // 2),
            "Gender": ["Male", "Female"] * (n // 2),
            "Age": [20 + i % 40 for i in range(n)],
            "Tenure": [i % 10 for i in range(n)],
            "Balance": [1000.0 * i for i in range(n)],
            "NumOfProducts": [1 + i % 3 for i in range(n)],
            "HasCrCard": [i % 2 for i in range(n)],
            "IsActiveMember": [(i + 1) % 2 for i in range(n)],
            "EstimatedSalary": [50000.0 + i for i in range(n)],
            "Exited": [i % 2 for i in range(n)],
        }
    )
    path = tmp_path / "churn.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_prepare_data(csv_path):
    X_train, X_test, y_train, y_test = prepare_data(csv_path)
    assert len(X_train) == 40 and len(X_test) == 10
    assert len(y_train) == 40 and len(y_test) == 10
    for col in ["Surname", "Geography", "RowNumber", "CustomerId", "Exited"]:
        assert col not in X_train.columns


def test_train_save_load(csv_path, tmp_path):
    X_train, X_test, y_train, _ = prepare_data(csv_path)
    model = train_model(X_train, y_train)
    model_path = str(tmp_path / "model.pkl")
    save_model(model, model_path)
    loaded = load_model(model_path)
    assert len(loaded.predict(X_test)) == len(X_test)
