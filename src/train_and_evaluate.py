import os
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import argparse
import joblib
import json
from get_data import read_params


def eval_metric(test_y, predicted_quality):
    rmse = np.sqrt(mean_squared_error(test_y, predicted_quality))
    mae = mean_absolute_error(test_y, predicted_quality)
    r2 = r2_score(test_y, predicted_quality)
    return rmse, mae, r2


def train_and_evaluate(config_path):
    config = read_params(config_path)
    train_data_path = config["split_data"]["train_path"]
    test_data_path = config["split_data"]["test_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]

    alpha = config["estimators"]["ElasticNet"]["params"]["alpha"]
    l1 = config["estimators"]["ElasticNet"]["params"]["l1_ratio"]

    target = [config["base"]["target_col"]]
    train_data = pd.read_csv(train_data_path, sep=",")
    test_data = pd.read_csv(test_data_path, sep=",")

    train_y = train_data[target]
    test_y = test_data[target]

    train_x = train_data.drop(target, axis=1)
    test_x = test_data.drop(target, axis=1)

    lr = ElasticNet(alpha=alpha, l1_ratio=l1, random_state=random_state)
    lr.fit(train_x, train_y)
    predicted_quality = lr.predict(test_x)

    (rmse, mae, r2) = eval_metric(test_y, predicted_quality)

    print("RMSE %s" % rmse)
    print("MAE %s" % mae)
    print("R2 Score %s" % r2)

    ############################################################################
    score_file = config["reports"]["scores"]
    param_file = config["reports"]["params"]

    with open(score_file, 'w') as f:
        scores = {
            "RMSE": rmse,
            "MAE": mae,
            "R2Score": r2

        }
        json.dump(scores, f, indent=4)

    with open(param_file, 'w') as f:
        params = {
            "alpha": alpha,
            "l1_ratio": l1
        }
        json.dump(params, f, indent=4)

    ############################################################################

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(lr, model_path)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)
