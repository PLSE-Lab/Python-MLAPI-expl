import os
import random
import numpy as np
import pandas as pd
import preprocessing_functions as pf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
import optuna
from functools import partial
from heamy.dataset import Dataset
from heamy.estimator import Regressor,Classifier
from heamy.pipeline import ModelsPipeline

# fix the seed
def seed_everything(random_seed=1234):
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    np.random.seed(random_seed)

#split data into train(features/target)/test(features/target), or just into features/target
def data_splitter(df, target, random_seed=1234, method="extract_split", split_ratio=0.8):
    #extract target column
    y = df[target]
    x = df.drop(target, axis=1).values
    if method == "extract_split":
        #train(features/target)/test(features/target)
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=split_ratio, random_state=random_seed, stratify=y_train)
        return x_train, x_test, y_train, y_test
    else:
        #features/target
        return x,y

#create parameters for each trial
def get_params_for_search(trial, param_list):
    params = {}
    for key, value in param_list.items():
        if type(value) != list:
            parameter_value = value
        else:
            suggest_type = value[0]
            if suggest_type == "int":
                parameter_value = trial.suggest_int(key,value[1][0],value[1][1])
            elif suggest_type == "category":
                parameter_value = trial.suggest_categorical(key,value[1])
            elif suggest_type == "uniform":
                parameter_value = trial.suggest_uniform(key,value[1][0],value[1][1])
            elif suggest_type == "loguniform":
                parameter_value = trial.suggest_loguniform(key,value[1][0],value[1][1])
            else: #discrete_uniform
                parameter_value = trial.suggest_discrete_uniform(key,value[1][0],value[1][1],value[1][2])
        params[key] = parameter_value
    return params
    
# hyper parameter search
def objective_function(x_train, y_train , param_list, model, kf_num, random_seed, trial):
    target_params = get_params_for_search(trial, param_list)
    model.set_params(**target_params)
    #cross validation
    kf = StratifiedKFold(n_splits=kf_num, shuffle=True, random_state=random_seed)
    scores = cross_validate(model, X=x_train, y=y_train, cv=kf)
    return 1.0 - scores['test_score'].mean()

def get_final_params(param_list, best_params):
    final_parmas = {}
    for key, value in param_list.items():
        if key in best_params.keys():
            final_parmas[key] = best_params[key]
        else:
            final_parmas[key] = value
    return final_parmas

def parameter_search(target_model_set, kf_num, trial_num, x_train, y_train, x_test, random_seed=1234):
    param_search_results = {}
    params_for_stacking = {}
    for model_name, model_setting in target_model_set.items():
        param_list = model_setting[0]
        model_instance = model_setting[1]
        #get the best parameter for each models by using 'hyperparameter search tool' optuna
        f = partial(objective_function, x_train, y_train, param_list, model_instance, kf_num, random_seed)
        optuna.logging.disable_default_handler()
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=random_seed))
        study.optimize(f, n_trials=trial_num)
        best_params = study.best_params
        best_value = "validation result : " + str(study.best_value)
        final_params = get_final_params(param_list=param_list, best_params=best_params)
        params_for_stacking[model_name] = [model_instance, final_params]
        #predict with best params by each models
        model_instance.set_params(**final_params)
        model_instance.fit(x_train, y_train)
        #if the model is tree-based one, get the feature importance
        if "tree" in model_name or "forest" in model_name:
            print(model_instance.feature_importances_)
        y_pred_tuned = model_instance.predict(x_test)
        param_search_results[model_name] = [y_pred_tuned, best_value]
    return params_for_stacking, param_search_results

def stacking_function(x_train, y_train, x_test, params_for_stacking, random_seed=1234, kf_num=5):
    dataset = Dataset(x_train, y_train, x_test) 
    models = []
    for key, value in params_for_stacking.items():
        best_model = value[0]
        best_param = value[1]
        model_name = key
        models.append(Regressor(dataset=dataset, estimator=best_model.__class__, parameters=best_param, name=model_name))
    #prediction 1
    pipeline = ModelsPipeline(*models)
    stack_ds = pipeline.stack(k=kf_num, seed=random_seed)
    #prediction 2
    stacker = Regressor(dataset=stack_ds, estimator=LinearRegression)
    y_trues, y_preds = stacker.validate(k=kf_num)
    y_pred = stacker.predict()
    return y_pred

if __name__ == "__main__":
    base_dir = "/kaggle/input/"
    ######### preprocessing by using util script #########
    # data load
    data_train = pd.read_csv(base_dir + "titanic/train.csv")
    data_test = pd.read_csv(base_dir + "titanic/test.csv")
    #drop na row / columns
    data_train_dropna = pf.missing_value(data_train,method="drop",_na_ratio=0.3)
    data_test_dropna = pf.missing_value(data_test,method="drop", _na_ratio=0.3)
    # label encoder
    lb_list = ["Sex","Ticket","Embarked"]
    data_train_lb = data_train_dropna
    for lb in lb_list:
        label_encoded = pf.label_enc(data_train_dropna, lb)
        data_train_lb[lb] = label_encoded
    data_test_lb = data_test_dropna
    for lb in lb_list:
        label_encoded = pf.label_enc(data_test_dropna, lb)
        data_test_lb[lb] = label_encoded
    # remove columns that are not used for modeling
    remove_list = ["Name"]
    data_train_lb.drop(remove_list, axis=1,inplace=True)
    data_test_lb.drop(remove_list, axis=1,inplace=True)
    ######### preprocessing by using util script #########
    seed_everything()
    # TEST : data_splitter
    data_train_x, data_train_y = data_splitter(df=data_train_lb, target="Survived", method="extract", split_ratio=0.8)
    data_test_x = np.array(data_test_lb)
    
    # TEST : create_params, objective_function
    #model definition 1
    target_model_set = {}
    param_list_rf = {
        "n_estimators":["int",[150,200]],
        "max_depth":["int",[1,3]],
        "random_state":1234
    }
    model_rf = RandomForestClassifier()
    target_model_set["random_forest"] = [param_list_rf, model_rf]
    #model definition 2
    param_list_lr = {
        "penalty":["category",["l2","none"]],
        "tol":["loguniform",[1e-4,5e-4]],
        "C":["discrete_uniform",[0.1,1.0,0.1]],
        "random_state":1234,
        "solver":["category",["lbfgs","sag"]]
    }
    model_lr = LogisticRegression()
    target_model_set["logistic_regression"] = [param_list_lr, model_lr]
    
    param_list_svm = {
    "C":["discrete_uniform",[0.001,0.01,0.001]],
    "gamma":["discrete_uniform",[0.001,0.01,0.01]],
    "kernel":["category",["rbf"]],
    "tol":["discrete_uniform",[1e-3,5e-3,1e-3]],
    "random_state":1234
    }
    model_svc = svm.SVC()
    target_model_set["svc"] = [param_list_svm, model_svc]

    param_list_abc = {
        "n_estimators":["int",[150,200]],
        "algorithm":["category",["SAMME","SAMME.R"]],
        "random_state":1234
    }
    model_abc = AdaBoostClassifier()
    target_model_set["ada_boost"] = [param_list_abc, model_abc]

    #hyper parameter search
    params_for_stacking, parameter_search_results = parameter_search(target_model_set=target_model_set, kf_num=3, trial_num=3, x_train=data_train_x, y_train=data_train_y, x_test=data_test_x)
    #stacking
    stacking_result = stacking_function(x_train=data_train_x, y_train=data_train_y, x_test=data_test_x, params_for_stacking=params_for_stacking, kf_num=3)
    print(stacking_result)

    