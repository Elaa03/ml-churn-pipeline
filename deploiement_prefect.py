"""Deploiements Prefect (Etape 05) : planification des flows.

Terminal 1 : prefect server start
Terminal 2 : prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"
             python deploiement_prefect.py
Terminal 3 : prefect deployment ls
             prefect deployment run 'all/ml-pipeline-all'
"""

from prefect import serve

from pipeline_prefect import all_flow, code, evaluate, install, predict, train

if __name__ == "__main__":
    serve(
        # Pipeline complet : tous les jours a 02:00
        all_flow.to_deployment(
            name="ml-pipeline-all",
            cron="0 2 * * *",
            tags=["full-pipeline", "mlops"],
        ),
        # Evaluation du modele : tous les jours a 03:00
        evaluate.to_deployment(
            name="ml-pipeline-evaluate",
            cron="0 3 * * *",
            tags=["evaluation", "mlops"],
        ),
        # Flows lances manuellement
        train.to_deployment(name="ml-pipeline-train", tags=["training", "mlops"]),
        code.to_deployment(name="ml-pipeline-code", tags=["code", "mlops"]),
        install.to_deployment(name="ml-pipeline-install", tags=["setup", "mlops"]),
        predict.to_deployment(name="ml-pipeline-predict", tags=["predict", "mlops"]),
    )
