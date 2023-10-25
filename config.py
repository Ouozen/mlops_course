from dataclasses import dataclass


@dataclass
class Data:
    train_url: str
    test_url: str
    train_filename: str
    test_filename: str
    label_column: str
    dir_data: str
    seed: int
    drop_column: list


@dataclass
class Model:
    name: str
    weight_file: str
    pred_file: str


@dataclass
class Params:
    data: Data
    model: Model
