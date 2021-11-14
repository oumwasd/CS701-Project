"""Create Dataset for Project CS701"""
import pandas as pd

def create_dataset(training, test, prediction):
    """สร้าง Dataset.csv ด้วยการเอาทุกข้อมูลมารวมกัน โดย dataset ต้องอยู่ที่เดียวกับไฟล์นี้"""
    dataset_training = pd.read_csv(training)
    dataset_test = pd.read_csv(test)
    dateset_prediction = pd.read_csv(prediction)
    dataset_test["Risk_Flag"] = dateset_prediction["risk_flag"]
    dataset_test["ID"] = dataset_test["ID"] + 252000
    dataset_test = dataset_test.rename(columns = {"ID" : "Id"})
    dataset_result = pd.concat([dataset_training, dataset_test])
    return dataset_result

if __name__ == "__main__":
    create_dataset("Training Data.csv", "Test Data.csv", "Sample Prediction Dataset.csv")