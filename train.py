from pathlib import Path

import gdown
import pandas as pd
import yaml
from xgboost import XGBClassifier


def get_data(url, output):
    gdown.download(url, output, quiet=False)


def train(model, data, **kwargs):
    return model.fit(data)


def main():
    with open("config.yaml", "r") as file:
        data_load = yaml.safe_load(file)

    url = data_load["train"]["url"]
    filename = data_load["train"]["filename"]
    model_name = data_load["model_name"]
    label_column = data_load["label_column"]
    drop_column = data_load["drop_column"]

    train_path = Path.cwd() / filename
    get_data(url, filename)

    train_data = pd.read_csv(train_path)
    drop_column.append(label_column)
    X_train, y_train = (
        train_data.drop(drop_column, axis=1),
        train_data[label_column],
    )

    model = XGBClassifier()
    model.fit(X_train, y_train)
    model.save_model(model_name)

    print("The model has been successfully trained!")


if __name__ == "__main__":
    main()
