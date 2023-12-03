import os
import warnings
from pathlib import Path

import fire
import gdown
import hydra
import mlflow
import pandas as pd
from hydra.core.config_store import ConfigStore
from sklearn.metrics import f1_score, precision_score, recall_score
from xgboost import XGBClassifier

from config import Params


warnings.simplefilter(
    action="ignore", category=FutureWarning
)  # ignore only FutureWarning

cs = ConfigStore.instance()
cs.store(name="params", node=Params)


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def download_data(cfg: Params):
    gdown.download(cfg["data"]["test_url"], cfg["data"]["test_filename"], quiet=False)

    Path("data").mkdir(exist_ok=True)
    cur_path = Path.cwd() / cfg["data"]["test_filename"]
    dest_path = Path.cwd() / "data" / cfg["data"]["test_filename"]
    dest_path.write_bytes(cur_path.read_bytes())
    os.remove(cur_path)


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def split_data(cfg: Params) -> None:
    df = pd.read_csv(Path.cwd() / "data" / cfg["data"]["test_filename"])
    drop_column = cfg["data"]["drop_column"]
    drop_column.append(cfg["data"]["label_column"])
    X, y = (df.drop(drop_column, axis=1), df[cfg["data"]["label_column"]])
    X.to_csv("./data/X_test.csv", index=False)
    y.to_csv("./data/y_test.csv", index=False)


def test(model, X_test, y_test, path_result):
    mlflow.set_tracking_uri("http://172.21.0.13:5000")
    mlflow.set_experiment(experiment_name="mlops_course")
    y_pred = model.predict(X_test)
    pd.DataFrame({"y_pred": y_pred}).to_csv(path_result)
    print(f"The prediction is saved to a file {path_result}")

    recall = round(recall_score(y_test, y_pred), 3)
    precision = round(precision_score(y_test, y_pred), 3)
    f_1 = round(f1_score(y_test, y_pred), 3)

    with mlflow.start_run():
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("f_1", f_1)

    print(f"Recall: {recall}, Precision: {precision}, F1-score: {f_1}")


def main(
    download_files: bool = True,
    weight_file: str = "model.json",
    pred_file: str = "Class_pred.csv",
):
    if download_files:
        download_data()

    split_data()
    X_test = pd.read_csv("./data/X_test.csv")
    y_test = pd.read_csv("./data/y_test.csv")

    model = XGBClassifier()
    model.load_model(Path.cwd() / "data" / weight_file)

    test(model, X_test, y_test, Path.cwd() / "data" / pred_file)


if __name__ == "__main__":
    fire.Fire(
        command="main --download_files False --weight_file model.json --pred_file Class_pred.csv"
    )
