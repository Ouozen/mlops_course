import os
import warnings
from pathlib import Path

import fire
import gdown
import pandas as pd
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score
from xgboost import XGBClassifier


warnings.simplefilter(
    action="ignore", category=FutureWarning
)  # ignore only FutureWarning


def get_data(url, filename, dir):
    gdown.download(url, filename, quiet=False)

    Path(dir).mkdir(exist_ok=True)
    cur_path = Path.cwd() / filename
    dest_path = Path.cwd() / dir / filename
    dest_path.write_bytes(cur_path.read_bytes())
    os.remove(cur_path)


def test(model, X_test, y_test, path_result):
    y_pred = model.predict(X_test)
    pd.DataFrame({"y_pred": y_pred}).to_csv(path_result)
    print(f"The prediction is saved to a file {path_result}")

    recall = round(recall_score(y_test, y_pred), 3)
    precision = round(precision_score(y_test, y_pred), 3)
    f_1 = round(f1_score(y_test, y_pred), 3)
    print(f"Recall: {recall}, Precision: {precision}, F1-score: {f_1}")


def main(
    filename: str = "creditcard_train.csv",
    model_name: str = "model.json",
    result_file: str = "Class_pred.csv",
    download_files: bool = True,
):
    with open("config.yaml", "r") as file:
        data_load = yaml.safe_load(file)

    url = data_load["test"]["url"]
    dir_data = data_load["dir_data"]
    label_column = data_load["label_column"]
    drop_column = data_load["drop_column"]

    test_path = Path.cwd() / dir_data / filename
    path_result = Path.cwd() / dir_data / result_file
    if download_files:
        get_data(url, filename, dir_data)

    test_data = pd.read_csv(test_path)
    drop_column.append(label_column)
    X_test, y_test = (
        test_data.drop(drop_column, axis=1),
        test_data[label_column],
    )

    model = XGBClassifier()
    model.load_model(Path.cwd() / dir_data / model_name)

    test(model, X_test, y_test, path_result)


if __name__ == "__main__":
    fire.Fire(main)
