import argparse

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)


def main():

    # 1. Define command-line arguments
    parser = argparse.ArgumentParser(description="Customer Churn ML Pipeline")

    parser.add_argument(
        "--data", default="Churn_Modelling.csv", help="Path to the dataset"
    )

    parser.add_argument(
        "--model", default="churn_model.pkl", help="Path to save or load the model"
    )

    parser.add_argument(
        "--mode",
        choices=["train", "load"],
        default="train",
        help="Choose train or load mode",
    )

    args = parser.parse_args()

    # 2. Prepare the data
    X_train, X_test, y_train, y_test = prepare_data(args.data)

    if args.mode == "train":

        # 3. Train the model
        model = train_model(X_train, y_train)

        # 4. Evaluate the model
        evaluate_model(model, X_test, y_test)

        # 5. Save the model
        save_model(model, args.model)

    else:

        # Load an existing model
        model = load_model(args.model)

        # Make predictions
        predictions = model.predict(X_test)

        print("First 5 predictions:", predictions[:5])


if __name__ == "__main__":
    main()
