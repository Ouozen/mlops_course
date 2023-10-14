from pathlib import Path

import gdown
import pandas as pd
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score
from xgboost import XGBClassifier


def get_data(url, output):
    gdown.download(url, output, quiet=False)


def test(model, X_test, y_test, path_result):
    y_pred = model.predict(X_test)
    pd.DataFrame({"y_pred": y_pred}).to_csv(path_result)

    recall = round(recall_score(y_test, y_pred), 3)
    precision = round(precision_score(y_test, y_pred), 3)
    f_1 = round(f1_score(y_test, y_pred), 3)
    print(f"Recall: {recall}, Precision: {precision}, F1-score: {f_1}")


def main():
    with open("config.yaml", "r") as file:
        data_load = yaml.safe_load(file)

    url = data_load["test"]["url"]
    filename = data_load["test"]["filename"]
    model_name = data_load["model_name"]
    result_file = data_load["result_file"]
    label_column = data_load["label_column"]
    drop_column = data_load["drop_column"]

    test_path = Path.cwd() / filename
    path_result = Path.cwd() / result_file
    get_data(url, filename)

    test_data = pd.read_csv(test_path)
    drop_column.append(label_column)
    X_test, y_test = (
        test_data.drop(drop_column, axis=1),
        test_data[label_column],
    )

    model = XGBClassifier()
    model.load_model(model_name)

    test(model, X_test, y_test, path_result)


if __name__ == "__main__":
    main()
