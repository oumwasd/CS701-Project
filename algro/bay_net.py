"""Bayesian Network"""
# %%
# import
import pathlib
import numpy as np
import pandas as pd
import bnlearn as bl
import sklearn.model_selection
import sklearn.metrics
import sklearn.preprocessing
import my_metrics
# %%
# load dataset
parent_path = pathlib.Path(__file__).parent.parent.resolve()
dataset = pd.read_csv(parent_path.joinpath("Dataset.csv"))
dataset = dataset.drop(columns = "Id")
MODEL_NAME = "bay_net"
# performance
PERF = {"n_jobs":1, "pre_dispatch":1}
# Verbosity
VERBOSE = {"verbose":2}
# %%
# split dataset
x_old = dataset[dataset.columns.difference(["Risk_Flag"], sort = False)]
y_old = dataset["Risk_Flag"]
x_train, x_test, y_train, y_test = \
    sklearn.model_selection.train_test_split(x_old, y_old, test_size = 0.5)
# %%
# Metrics
metrics_name = ["F1", "AUC", "H-measure", "KS_score", "Brier_score", "Log_loss"]
f1_score = sklearn.metrics.make_scorer(sklearn.metrics.f1_score)
auc_score = sklearn.metrics.make_scorer(sklearn.metrics.roc_auc_score)
h_score = sklearn.metrics.make_scorer(my_metrics.h_socre, needs_proba = True)
ks_score = sklearn.metrics.make_scorer(my_metrics.ks_socre, needs_proba = True)
brier_score = sklearn.metrics.make_scorer \
    (sklearn.metrics.brier_score_loss, greater_is_better = False, needs_proba = True)
log_loss_score = sklearn.metrics.make_scorer \
    (sklearn.metrics.log_loss, greater_is_better = False, needs_proba = True)
metrics = {"F1":f1_score, "AUC":auc_score, "H-measure":h_score, \
    "KS_score":ks_score, "Brier_score":brier_score, "Log_loss":log_loss_score}
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
# Grid search
cv = sklearn.model_selection.StratifiedKFold(n_splits = 5, shuffle = True)
grid_search = []
for ps in parameters_struc:
    for pp in parameters_para:
        for idx_train, idx_test in cv.split(x_train, y_train):
            # pre data
            in_x_train = x_train.iloc[idx_train]
            in_y_train = y_train.iloc[idx_train]
            in_x_test = x_train.iloc[idx_test]
            in_y_test = y_train.iloc[idx_test]
            in_x_train["Risk_Flag"] = in_y_train
            in_x_test["Risk_Flag"] = in_y_test
            # fit
            struc_learn = bl.structure_learning.fit(in_x_train, **ps)
            para_learn = bl.parameter_learning.fit(struc_learn, in_x_test, **pp)
            pred_result = bl.predict(para_learn, in_x_test, \
                variables = "Risk_Flag")
            scores = [
                sklearn.metrics.f1_score(in_y_test, pred_result["Risk_Flag"]),
                sklearn.metrics.roc_auc_score(in_y_test, pred_result["Risk_Flag"]),
                my_metrics.h_socre(in_y_test, np.asarray(pred_result["p"])),
                my_metrics.ks_socre(in_y_test, np.asarray(pred_result["p"])),
                sklearn.metrics.brier_score_loss(in_y_test, pred_result["p"]),
                sklearn.metrics.log_loss(in_y_test, pred_result["p"])
            ]
            score_result = dict(zip(metrics_name, scores))
            grid_search.append({"ps":ps, "pp":pp, **score_result})
grid_result = pd.DataFrame(grid_search)