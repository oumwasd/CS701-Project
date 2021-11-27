"""XGBoost"""
# %%
# import
import pathlib
import numpy as np
import pandas as pd
import sklearn.model_selection
import sklearn.metrics
import sklearn.preprocessing
import my_metrics
import xgboost
# %%
# load dataset
parent_path = pathlib.Path(__file__).parent.parent.resolve()
dataset = pd.read_csv(parent_path.joinpath("Dataset.csv"))
dataset = dataset.drop(columns = "Id")
MODEL_NAME = "xgboost"
# performance
PERF = {"n_jobs":1, "pre_dispatch":1}
# Verbosity
VERBOSE = {"verbose":2}
# %%
# remove special characters
for col in ["STATE", "CITY"]:
    for ch in ["[", "]"]:
        dataset[col] = dataset[col].str.replace(ch, '', regex = False)
# %%
# Ordinal Encoding
# Married/Single, Car_Ownership
label_encoder = sklearn.preprocessing.LabelEncoder()
cols_orde = ["Married/Single", "Car_Ownership"]
for col in cols_orde:
    dataset[col] = label_encoder.fit_transform(dataset[col])
# %%
# One-Hot Encoding
# House_Ownership, Profession, CITY, STATE
cols_oneh = ["House_Ownership", "Profession", "CITY", "STATE"]
for col in cols_oneh:
    one_hot_vec = pd.get_dummies(dataset[col], prefix = col, dtype = np.int64)
    dataset = dataset.drop(columns = col)
    dataset = pd.concat([dataset, one_hot_vec], axis = 1)
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
# Grid search
model = xgboost.XGBClassifier(use_label_encoder = False, tree_method = "exact")
in_cv = sklearn.model_selection.StratifiedKFold(n_splits = 5, shuffle = True)
space = {"n_estimators":[10, 20, 50, 100], "learning_rate":[0.1, 0.3, 0.5, 0.7, 0.9, 1], \
    "max_depth":[2, 4, 6]}
grid_search = sklearn.model_selection.GridSearchCV \
    (model, space, scoring = metrics, cv = in_cv, refit = False, **PERF, **VERBOSE)
grid_search.fit(x_train, y_train, eval_metric = "logloss")
grid_result = pd.DataFrame(grid_search.cv_results_)
# %%
# Retrieval Parameter
parameters = []
for name in metrics_name:
    row_first = grid_result[grid_result[f"rank_test_{name}"] == 1]
    para = row_first["params"].iloc[0]
    parameters.append(para)
parameters_result = pd.DataFrame(dict(zip(metrics_name, parameters)))
# %%
# Evaluation
scores = []
for i, para in enumerate(parameters):
    metric = metrics[metrics_name[i]]
    out_cv = sklearn.model_selection.StratifiedKFold(n_splits = 5, shuffle = True)
    eval_model = xgboost.XGBClassifier(use_label_encoder = False, tree_method = "hist", **para)
    result = sklearn.model_selection.cross_val_score \
        (eval_model, X = x_test, y = y_test, cv = out_cv, scoring = metric, \
            fit_params = {"eval_metric":"logloss"}, **PERF, **VERBOSE)
    scores.append(result)
scores_result = pd.DataFrame(dict(zip(metrics_name, scores)))
# %%
# save to file
FILE_NAME = "XGB"
pathlib.Path.mkdir(parent_path.joinpath("result"), exist_ok = True)
# grid_result.to_csv(parent_path.joinpath("result", \
    # f"{FILE_NAME} Grid Result.csv"), index = False)
parameters_result.to_csv(parent_path.joinpath("result", \
    f"{FILE_NAME} Parameters Result.csv"), index = True)
scores_result.to_csv(parent_path.joinpath("result", \
    f"{FILE_NAME} Scores Result.csv"), index = False)
# %%
# End
print(f"{FILE_NAME} finish")
