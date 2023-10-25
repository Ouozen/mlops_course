import os
import warnings
from pathlib import Path

import fire
import gdown
import hydra
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


def main(download_files: bool = False, weight_file: str = "model.json") -> None:
    if download_files:
        download_data()

    split_data()
    X_train = pd.read_csv("./data/X_train.csv")
    y_train = pd.read_csv("./data/y_train.csv")

    model = XGBClassifier()
    model.fit(X_train, y_train)
    model.save_model(Path.cwd() / "data" / weight_file)

    print("The model has been successfully trained!")


if __name__ == "__main__":
    fire.Fire(command="main --download_files False --weight_file model.json")
