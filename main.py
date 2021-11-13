"""Main for Project CS701"""
import importlib
# import numpy as np
import pandas as pd

def create_dataset(training, test, prediction):
    """สร้าง Dataset ด้วยการเอาทุกข้อมูลมารวมกัน"""
    dataset_training = pd.read_csv(training)
    dataset_test = pd.read_csv(test)
    dateset_prediction = pd.read_csv(prediction)
    dataset_test["Risk_Flag"] = dateset_prediction["risk_flag"]
    dataset_test["ID"] = dataset_test["ID"] + 252000
    dataset_test = dataset_test.rename(columns = {"ID" : "Id"})
    dataset_result = pd.concat([dataset_training, dataset_test])
    return dataset_result

if __name__ == "__main__":
    importlib.import_module("elasticnet_logistic_regression")
    importlib.import_module("logistic_regression_smote")
    importlib.import_module("elasticnet_logistic_regression_smote")
