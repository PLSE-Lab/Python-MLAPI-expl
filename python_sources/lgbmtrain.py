import pandas as pd
import lightgbm as lgb

# read data from csv file
heart = pd.read_csv("../input/heart-disease-uci/heart.csv")
X = heart.loc[:, ~heart.columns.isin(["target"])]
y = heart["target"]
# create lgb dataset
train_set = lgb.Dataset(data=X, label=y, categorical_feature=["sex", "cp", "fbs", "restecg", "exang", "thal"])
# build model
cls = lgb.LGBMClassifier(boosting_type='gbdt', 
                         num_leaves=31, 
                         max_depth=-1, 
                         learning_rate=0.1, 
                         n_estimators=100, 
                         subsample_for_bin=200000, 
                         objective=None, 
                         class_weight=None, 
                         min_split_gain=0.0, 
                         min_child_weight=0.001,
                         min_child_samples=20, 
                         subsample=1.0, 
                         subsample_freq=0, 
                         colsample_bytree=1.0, 
                         reg_alpha=0.0, 
                         reg_lambda=0.0, 
                         random_state=None, 
                         n_jobs=-1, 
                         silent=True, 
                         importance_type='split')
# cross validation
print("--Start cross validation--")
lgb_params = cls.get_params()
cv_result = lgb.cv(lgb_params, train_set, num_boost_round=100, folds=None, nfold=5, stratified=True, shuffle=True, metrics="auc", fobj=None, feval=None, init_model=None, feature_name='auto', categorical_feature='auto', early_stopping_rounds=None, fpreproc=None, verbose_eval=None, show_stdv=True, seed=0, callbacks=None)
print("Evaluation result:")
for metric in cv_result.keys():
    print("metric: ",metric, "length: ", len(cv_result[metric]), "result: ", cv_result[metric])


