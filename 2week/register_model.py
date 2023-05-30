import os
import pickle
import mlflow

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from mlflow.models.signature import infer_signature

HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
EXPERIMENT_NAME = "random-forest-best-models"
RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state', 'n_jobs']

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def train_and_log_model(data_path, params):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    new_dict = {}
    with mlflow.start_run():
        for param in RF_PARAMS:
            new_dict[param] = eval(params[param])

        rf = RandomForestRegressor(**new_dict)
        rf.fit(X_train, y_train)

        # Evaluate model on the validation and test sets
        val_rmse = mean_squared_error(y_val, rf.predict(X_val), squared=False)
        mlflow.log_metric("val_rmse", val_rmse)
        test_rmse = mean_squared_error(y_test, rf.predict(X_test), squared=False)
        mlflow.log_metric("test_rmse", test_rmse)
        signature = infer_signature(X_train, rf.predict(X_train))
        mlflow.sklearn.log_model(rf, "rf", signature=signature)


def run_register_model(data_path: str, top_n: int):

    client = MlflowClient()

    # Retrieve the top_n model runs and log the models
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"]
    )
    for run in runs:
        train_and_log_model(data_path=data_path, params=run.data.params)

    # Select the model with the lowest test RMSE
    # Sort runs by RMSE (assuming RMSE is a metric logged)
    runs.sort(key=lambda r: r.data.metrics.get("test_rmse", float("inf")))
    # Get the run ID with the lowest RMSE
    print(f'The best run id is {runs[0].info.run_id}')
    best_run_path = runs[0].info.artifact_uri
    result = mlflow.register_model(best_run_path, EXPERIMENT_NAME)

if __name__ == '__main__':

    run_register_model(os.path.abspath('2week/output'), top_n=10)