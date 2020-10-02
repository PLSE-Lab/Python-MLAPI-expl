# -*-coding=utf-8-*-
# Loading Library -----------------------------------------------------------------------------------------------------------
import warnings

import random
import numpy as np 
import pandas as pd 

from datetime import datetime
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score

# model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
# tree 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from rgf.sklearn import RGFClassifier
# from gcforest.gcforest import GCForest

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Data preparation ---------------------------------------------------------------------------------------------------------
# ignore warning
warnings.filterwarnings('ignore')

# constant
DIR_MODEL_SELECT = './model_selection_result'

# result repetition
random.seed(1234)
np.random.seed(1234)

# data split
data = load_iris()
data_x = data['data']
data_y = data['target']


# Define model list-----------------------------------------------------------------------------------------------------------
def model_selection(selection_x, selection_y, evl_metric, cv_num):
    
    model_list = {
        
        # linear model
        'Logistic Regression': LogisticRegression(),
        'SVM': SVC(),
        'LinearSVC': LinearSVC(),
        'K-NN': KNeighborsClassifier(),
        
        # tree_based model
        'Random Forest': RandomForestClassifier(),
        'ExtraTree': ExtraTreesClassifier(),
        'Regularized Greedy Forest': RGFClassifier(),
        
        'AdaBoost': AdaBoostClassifier(),
        'GBDT': GradientBoostingClassifier(),
        'XgBoost': XGBClassifier(),
        'LightGBM': LGBMClassifier(),
        'CatBoost': CatBoostClassifier(verbose=False)
        
    }
    
    for model_name in model_list:
        
        cv_score = cross_val_score(model_list[model_name], selection_x, selection_y, scoring=evl_metric, cv=cv_num)
        print(datetime.now(), model_name, evl_metric, '=', np.mean(cv_score))

        # write out result
        write_str = str(datetime.now()) + '  ' + model_name + '  ' + evl_metric + '=' + str(np.mean(cv_score))
        with open(DIR_MODEL_SELECT, 'a+') as f:
            f.write(write_str + '\n')


# model selection-----------------------------------------------------------------------------------------------------
model_selection(data_x, data_y, evl_metric='accuracy', cv_num=3)


