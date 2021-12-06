"""Bayesian Network with SMOTE"""
# %%
# import library
import pathlib
import numpy as np
import pandas as pd
import imblearn as il
import bnlearn as bl
import sklearn.model_selection
import sklearn.metrics
import sklearn.preprocessing
import my_metrics
# %%
# Loading Dataset
parent_path = pathlib.Path(__file__).parent.parent.resolve()
dataset = pd.read_csv(parent_path.joinpath("Dataset.csv"))
dataset = dataset.drop(columns = "Id")
MODEL_NAME = "bay_net"
# Performance
PERF = {"n_jobs":1, "pre_dispatch":1}
# Verbosity
VERBOSE = {"verbose":1}
# %%
# Splitting Dataset
x_old = dataset[dataset.columns.difference(["Risk_Flag"], sort = False)]
y_old = dataset["Risk_Flag"]
x_train, x_test, y_train, y_test = \
    sklearn.model_selection.train_test_split(x_old, y_old, test_size = 0.5)
# %%
# Metrics
metrics_name = ["F1", "AUC", "H-measure", "KS_score", "Brier_score", "Log_loss"]
f1_score = lambda y_true, y_pred : sklearn.metrics.f1_score(y_true, y_pred["Risk_Flag"])
auc_score = lambda y_true, y_pred : sklearn.metrics.roc_auc_score(y_true, y_pred["Risk_Flag"])
h_score = lambda y_true, y_pred : my_metrics.h_socre(y_true, np.asarray(y_pred["p"]))
ks_score = lambda y_true, y_pred : my_metrics.ks_socre(y_true, np.asarray(y_pred["p"]))
brier_score = lambda y_true, y_pred : -1 * sklearn.metrics.brier_score_loss(y_true, y_pred["p"])
log_loss_score = lambda y_true, y_pred : -1 * sklearn.metrics.log_loss(y_true, y_pred["p"])
metrics = {"F1":f1_score, "AUC":auc_score, "H-measure":h_score, \
    "KS_score":ks_score, "Brier_score":brier_score, "Log_loss":log_loss_score}
# %%
# SMOTE
smote = il.over_sampling.SMOTENC(sampling_strategy = "minority", \
    categorical_features = [3, 4, 5, 6, 7, 8], n_jobs = -1)
# %%
# Parameters Grid
parameters_struc = [
    {"methodtype":["cl"], "root_node":["Risk_Flag"],
    "scoretype":["bic", "k2"]},
    {"methodtype":["tan"],"root_node":["Risk_Flag"],
    "class_node":["Income"], "scoretype":["bic", "k2"]}
    ]
parameters_struc = list(sklearn.model_selection.ParameterGrid(parameters_struc))
parameters_para = {"methodtype":["bayes", "ml"]}
parameters_para = list(sklearn.model_selection.ParameterGrid(parameters_para))
# %%
# Grid Search
cv = sklearn.model_selection.StratifiedKFold(n_splits = 5, shuffle = True)
grid_search = []
for ps in parameters_struc:
    for pp in parameters_para:
        score = []
        for idx_train, idx_test in cv.split(x_train, y_train):
            # Preprocessing Data
            in_x_train = x_train.iloc[idx_train]
            in_y_train = y_train.iloc[idx_train]
            # SMOTE with training data
            smote.fit_resample(in_x_train, in_y_train)
            in_x_test = x_train.iloc[idx_test]
            in_y_test = y_train.iloc[idx_test]
            # combine x and y
            in_x_train["Risk_Flag"] = in_y_train
            in_x_test["Risk_Flag"] = in_y_test
            # fit
            struc_learn = bl.structure_learning.fit(in_x_train, **ps, **VERBOSE)
            para_learn = bl.parameter_learning.fit(struc_learn, \
                in_x_test, **pp, **VERBOSE)
            pred_result = bl.predict(para_learn, in_x_test, \
                variables = "Risk_Flag", **VERBOSE)
            # calculate score
            score.append([metric(in_y_test, pred_result) \
                for metric in metrics.values()])
        # calculate mean score
        score = np.matrix(score)
        socre = [
            np.mean(score[:, 0]),
            np.mean(score[:, 1]),
            np.mean(score[:, 2]),
            np.mean(score[:, 3]),
            np.mean(score[:, 4]),
            np.mean(score[:, 5])
            ]
        score_result = dict(zip(metrics_name, socre))
        grid_search.append({"ps":ps, "pp":pp, **score_result})
grid_result = pd.DataFrame(grid_search)
# %%
# Retrieval Parameter
indexs = [grid_result[name].idxmax() for name in metrics_name]
parameters_struc = [grid_result.iloc[idx]["ps"] for idx in indexs]
parameters_para = [grid_result.iloc[idx]["pp"] for idx in indexs]
parameters = list(zip(parameters_struc, parameters_para))
parameters_result = pd.concat([
    pd.DataFrame(dict(zip(metrics_name, parameters_struc))),
    pd.DataFrame(dict(zip(metrics_name, parameters_para)))
    ])
# %%
# Evaluation
scores = []
for i, para in enumerate(parameters):
    score = []
    metric = metrics[metrics_name[i]]
    ps = para[0]
    pp = para[1]
    for idx_train, idx_test in cv.split(x_test, y_test):
        # Preprocessing Data
        in_x_train = x_test.iloc[idx_train]
        in_y_train = y_test.iloc[idx_train]
        # SMOTE with training data
        smote.fit_resample(in_x_train, in_y_train)
        in_x_test = x_test.iloc[idx_test]
        in_y_test = y_test.iloc[idx_test]
        # combine x and y
        in_x_train["Risk_Flag"] = in_y_train
        in_x_test["Risk_Flag"] = in_y_test
        # fit
        struc_learn = bl.structure_learning.fit(in_x_train, **ps, **VERBOSE)
        para_learn = bl.parameter_learning.fit(struc_learn, \
            in_x_test, **pp, **VERBOSE)
        pred_result = bl.predict(para_learn, in_x_test, \
            variables = "Risk_Flag", **VERBOSE)
        # calculate score
        score.append(metric(in_y_test, pred_result))
    scores.append(score)
scores_result = pd.DataFrame(dict(zip(metrics_name, scores)))
# %%
# save to file
FILE_NAME = "Bay Net"
pathlib.Path.mkdir(parent_path.joinpath("result"), exist_ok = True)
# grid_result.to_csv(parent_path.joinpath("result", \
    # f"{FILE_NAME} with SMOTE Grid Result.csv"), index = False)
parameters_result.to_csv(parent_path.joinpath("result", \
    f"{FILE_NAME} with SMOTE Parameters Result.csv"), index = True)
scores_result.to_csv(parent_path.joinpath("result", \
    f"{FILE_NAME} with SMOTE Scores Result.csv"), index = False)
# %%
# Ending
print(f"{FILE_NAME} finish")
