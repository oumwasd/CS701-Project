"""Elastic-net Logistic Regression"""
# %%
# import
import numpy as np
import pandas as pd
import sklearn.linear_model
import sklearn.model_selection
import sklearn.metrics
import sklearn.preprocessing
import my_metrics
# %%
# load dataset
dataset = pd.read_csv("../Dataset.csv")
dataset = dataset.drop(columns = "Id")
MODEL_NAME = "ela_logit"
# %%
# Normalization
# Income Age Experience CURRENT_JOB_YRS CURRENT_HOUSE_YRS
minmax = sklearn.preprocessing.MinMaxScaler(copy = False)
cols_nor = ["Income", "Age", "Experience", "CURRENT_JOB_YRS", "CURRENT_HOUSE_YRS"]
for col in cols_nor:
    dataset[col] = minmax.fit_transform(dataset[col].values.reshape(-1,1))
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
model = sklearn.linear_model.LogisticRegression \
    (penalty = "elasticnet", n_jobs = -1, solver = "saga")
in_cv = sklearn.model_selection.StratifiedKFold(n_splits = 5, shuffle = True)
space = {"max_iter":[100, 500, 1000], "C":[0.1, 0.3, 0.5, 0.7, 1], \
    "l1_ratio":[0.1, 0.3, 0.5, 0.7, 1]}
grid_search = sklearn.model_selection.GridSearchCV \
    (model, space, scoring = metrics, cv = in_cv, n_jobs = 3, refit = False)
grid_search.fit(x_train, y_train)
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
    eval_model = sklearn.linear_model.LogisticRegression \
        (penalty = "elasticnet", n_jobs = -1, solver = "saga", **para)
    result = sklearn.model_selection.cross_val_score \
        (eval_model, X = x_test, y = y_test, cv = out_cv, scoring = metric, n_jobs = 3)
    scores.append(result)
scores_result = pd.DataFrame(dict(zip(metrics_name, scores)))
# %%
# save to file
# grid_result.to_csv("../result/Ela-net Logit Grid Result.csv", index = False)
parameters_result.to_csv("../result/Ela-net Logit Parameters Result.csv", index = True)
scores_result.to_csv("../result/Ela-net Logit Scores Result.csv", index = False)
