"""Tabular ML experiment: grid-search RandomForest hyperparameters on a small
built-in dataset, logging each configuration as one MLflow run, with everything
the ExperimentLens explainability views need (train/test data, predictions,
model and ROC curve). No LLM or Langfuse involved.

`ExperimentTracker(..., experiment_type="ml")` shows the experiment under
ExperimentLens' ML Experiments area instead of Agents & LLMs, and never
touches Langfuse.

Needs scikit-learn:  pip install "experimentlens[examples]"
Run with:            python examples/ml_experiment.py
"""
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import ParameterGrid, train_test_split

from experimentlens import ExperimentTracker

PARAM_GRID = list(ParameterGrid({
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 10],
    # "criterion": ["gini", "entropy"],
    # "min_samples_leaf": [1, 2],
}))


def main():
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tracker = ExperimentTracker("breast-cancer-rf-grid", experiment_type="ml")

    for i, params in enumerate(PARAM_GRID):
        with tracker.start_run(f"rf_{i}", params=params) as ctx:
            ctx.log_dataset(
                pd.concat([X_train, y_train.rename("target")], axis=1),
                name="train_data",
                context="training",
            )

            clf = RandomForestClassifier(**params, random_state=42)
            clf.fit(X_train, y_train)
            predictions = clf.predict(X_test)

            ctx.log_metrics({
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions),
                "recall": recall_score(y_test, predictions),
                "f1": f1_score(y_test, predictions),
            })

            # Everything ExperimentLens' explainability views read, logged under the
            # run's "explainability/" artifact folder.
            ctx.log_explainability_data(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                y_pred=predictions,
                model=clf,
                y_score=clf.predict_proba(X_test),  # -> roc_data.json
            )

            mlflow.sklearn.log_model(clf, name="model")
            ctx.set_tags({"model_type": "RandomForestClassifier", "dataset": "breast_cancer"})

            print(f"rf_{i} {params} -> accuracy={accuracy_score(y_test, predictions):.3f}")

    tracker.close()


if __name__ == "__main__":
    main()
