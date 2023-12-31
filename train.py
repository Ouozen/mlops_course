import os
import warnings
from pathlib import Path

import fire
import gdown
import hydra
import mlflow
import pandas as pd
from hydra.core.config_store import ConfigStore
from xgboost import XGBClassifier

from config import Params


warnings.simplefilter(
    action="ignore", category=FutureWarning
)  # ignore only FutureWarning

cs = ConfigStore.instance()
cs.store(name="params", node=Params)


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def download_data(cfg: Params) -> None:
    gdown.download(cfg["data"]["train_url"], cfg["data"]["train_filename"], quiet=False)

    Path("data").mkdir(exist_ok=True)
    cur_path = Path.cwd() / cfg["data"]["train_filename"]
    dest_path = Path.cwd() / "data" / cfg["data"]["train_filename"]
    dest_path.write_bytes(cur_path.read_bytes())
    os.remove(cur_path)


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def split_data(cfg: Params) -> None:
    df = pd.read_csv(Path.cwd() / "data" / cfg["data"]["train_filename"])
    drop_column = cfg["data"]["drop_column"]
    drop_column.append(cfg["data"]["label_column"])
    X, y = (df.drop(drop_column, axis=1), df[cfg["data"]["label_column"]])
    X.to_csv("./data/X_train.csv", index=False)
    y.to_csv("./data/y_train.csv", index=False)


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def train(cfg: Params) -> None:
    os.environ["AWS_ACCESS_KEY_ID"] = cfg["s3"]["key"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = cfg["s3"]["pass"]
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = cfg["s3"]["uri"]

    X_train = pd.read_csv("./data/X_train.csv")
    y_train = pd.read_csv("./data/y_train.csv")

    mlflow.set_tracking_uri(cfg["mlflow"]["uri"])
    mlflow.set_experiment(experiment_name="mlops_course")
    mlflow.xgboost.autolog()
    with mlflow.start_run():
        model = XGBClassifier()
        model.fit(X_train, y_train)
    model.save_model(Path.cwd() / "data" / cfg["model"]["weight_file"])


def main(download_files: bool = False, weight_file: str = "model.json") -> None:
    if download_files:
        download_data()
    split_data()
    train()
    print("The model has been successfully trained!")


if __name__ == "__main__":
    fire.Fire(command="main --download_files False --weight_file model.json")
