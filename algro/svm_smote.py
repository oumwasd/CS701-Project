"""Support Vector Machines with SMOTE"""
# %%
# import
import pathlib
import numpy as np
import pandas as pd
import imblearn as il
import sklearn.svm
import sklearn.model_selection
import sklearn.metrics
import sklearn.preprocessing
import my_metrics
# %%
# load dataset
parent_path = pathlib.Path(__file__).parent.parent.resolve()
dataset = pd.read_csv(parent_path.joinpath("Dataset.csv"))
dataset = dataset.drop(columns = "Id")
MODEL_NAME = "svm"
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
# SMOTE
smote = il.over_sampling.SMOTE(sampling_strategy = "minority", n_jobs = -1)
# %%
# Grid search
model = sklearn.svm.NuSVC(probability = True)
pipl_model = il.pipeline.Pipeline([("smote", smote), (f"{MODEL_NAME}", model)])
in_cv = sklearn.model_selection.StratifiedKFold(n_splits = 5, shuffle = True)
space = {"nu":[0.1, 0.3, 0.5, 0.7], "kernel":["linear", "poly", "rbf"]}
# เพิ่มตัวอักษร model__ เข้าไปในชื่อพารามิเตอร์เพื่อให้สามารถใช้กับ pipeline ได้
new_parameter_names = [f"{MODEL_NAME}__{key}" for key in space]
pipl_space = dict(zip(new_parameter_names, space.values()))
grid_search = sklearn.model_selection.GridSearchCV \
    (pipl_model, pipl_space, scoring = metrics, cv = in_cv, refit = False, \
        n_jobs = -1,  pre_dispatch = 1)
grid_search.fit(x_train, y_train)
grid_result = pd.DataFrame(grid_search.cv_results_)
# %%
# Retrieval Parameter
parameters = []
for name in metrics_name:
    row_first = grid_result[grid_result[f"rank_test_{name}"] == 1]
    para = row_first["params"].iloc[0]
    para_val = [para[f"{MODEL_NAME}__{key}"] for key in space]
    parameters.append(dict(zip(space.keys(), para_val)))
parameters_result = pd.DataFrame(dict(zip(metrics_name, parameters)))
# %%
# Evaluation
scores = []
for i, para in enumerate(parameters):
    metric = metrics[metrics_name[i]]
    out_cv = sklearn.model_selection.StratifiedKFold(n_splits = 5, shuffle = True)
    eval_model = sklearn.svm.NuSVC(probability = True, **para)
    eval_pipl_model = il.pipeline.Pipeline([("smote", smote), (f"{MODEL_NAME}", eval_model)])
    result = sklearn.model_selection.cross_val_score \
        (eval_pipl_model, X = x_test, y = y_test, cv = out_cv, scoring = metric, \
            n_jobs = -1, pre_dispatch = 1)
    scores.append(result)
scores_result = pd.DataFrame(dict(zip(metrics_name, scores)))
# %%
# save to file
FILE_NAME = "SVM"
pathlib.Path.mkdir(parent_path.joinpath("result"), exist_ok = True)
# grid_result.to_csv(parent_path.joinpath("result", \
    # f"{FILE_NAME} with SMOTE Grid Result.csv"), index = False)
parameters_result.to_csv(parent_path.joinpath("result", \
    f"{FILE_NAME} with SMOTE Parameters Result.csv"), index = True)
scores_result.to_csv(parent_path.joinpath("result", \
    f"{FILE_NAME} with SMOTE Scores Result.csv"), index = False)
# %%
# End
print(f"{FILE_NAME} finish")
