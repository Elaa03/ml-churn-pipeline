"""Pipeline Prefect - Atelier 3 (Customer Churn).

Lancement :
    python pipeline_prefect.py --flow install
    python pipeline_prefect.py --flow code
    python pipeline_prefect.py --flow train        (alias : entrainement)
    python pipeline_prefect.py --flow evaluate
    python pipeline_prefect.py --flow predict
    python pipeline_prefect.py --flow all
"""

import argparse
import subprocess
import sys
from pathlib import Path

from prefect import flow, task

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = str(BASE_DIR / "Churn_Modelling.csv")
MODEL_PATH = str(BASE_DIR / "churn_model.pkl")

# On ne cible que nos fichiers (sinon scan du venv = tres lent)
CODE_FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]


def run_cmd(cmd):
    """Execute une commande et leve une erreur si elle echoue."""
    subprocess.run(cmd, check=True, cwd=BASE_DIR)


# ======================================================================
# TASKS : suivi du code
# ======================================================================
@task(name="Installer les dependances")
def install_dependencies_task():
    run_cmd([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


@task(name="Formatage du code (black)")
def format_code_task():
    run_cmd([sys.executable, "-m", "black", *CODE_FILES])


@task(name="Qualite du code (pylint)")
def quality_task():
    # --exit-zero : une note imparfaite ne bloque pas le pipeline
    run_cmd([sys.executable, "-m", "pylint", "--exit-zero", *CODE_FILES])


@task(name="Securite du code (bandit)")
def security_task():
    # -ll : ne remonte que les severites moyenne et haute
    run_cmd([sys.executable, "-m", "bandit", "-ll", "-q", *CODE_FILES])


@task(name="Tests unitaires (pytest)")
def unit_tests_task():
    run_cmd([sys.executable, "-m", "pytest", "-v", "test_pipeline.py"])


# ======================================================================
# TASKS : data / modele
# ======================================================================
@task(name="Preparation des donnees")
def prepare_data_task(data_path):
    return prepare_data(data_path)


@task(name="Entrainement du modele")
def train_model_task(X_train, y_train):
    return train_model(X_train, y_train)


@task(name="Sauvegarde du modele")
def save_model_task(model, model_path):
    save_model(model, model_path)


@task(name="Chargement du modele")
def load_model_task(model_path):
    return load_model(model_path)


@task(name="Evaluation du modele")
def evaluate_model_task(model, X_test, y_test):
    return evaluate_model(model, X_test, y_test)


@task(name="Prediction")
def predict_task(model, X_test):
    predictions = model.predict(X_test)
    print("First 5 predictions:", predictions[:5])
    return predictions


# ======================================================================
# FLOWS
# ======================================================================
@flow(name="install")
def install():
    install_dependencies_task()


@flow(name="code")
def code():
    install_dependencies_task()
    format_code_task()
    quality_task()
    security_task()
    unit_tests_task()


@flow(name="train")
def train(data_path: str = DATA_PATH, model_path: str = MODEL_PATH):
    X_train, _, y_train, _ = prepare_data_task(data_path)
    model = train_model_task(X_train, y_train)
    save_model_task(model, model_path)


@flow(name="evaluate")
def evaluate(data_path: str = DATA_PATH, model_path: str = MODEL_PATH):
    _, X_test, _, y_test = prepare_data_task(data_path)
    model = load_model_task(model_path)
    evaluate_model_task(model, X_test, y_test)


@flow(name="predict")
def predict(data_path: str = DATA_PATH, model_path: str = MODEL_PATH):
    _, X_test, _, _ = prepare_data_task(data_path)
    model = load_model_task(model_path)
    predict_task(model, X_test)


@flow(name="all")
def all_flow(data_path: str = DATA_PATH, model_path: str = MODEL_PATH):
    # Etapes code (install, format, qualite, securite, tests)
    code()
    # Etapes 2-3-4-6 : preparation, entrainement, sauvegarde, evaluation
    X_train, X_test, y_train, y_test = prepare_data_task(data_path)
    model = train_model_task(X_train, y_train)
    save_model_task(model, model_path)
    evaluate_model_task(model, X_test, y_test)


FLOWS = {
    "install": install,
    "code": code,
    "train": train,
    "entrainement": train,  # alias utilise dans l'enonce
    "evaluate": evaluate,
    "predict": predict,
    "all": all_flow,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Prefect - Churn")
    parser.add_argument("--flow", choices=FLOWS.keys(), default="all")
    args = parser.parse_args()
    FLOWS[args.flow]()
