"""
Author: Malik R. Booker
Created: 23-MAY-2020
Edited: 23-MAY-2020
"""

import pandas as pd
import numpy as np
import gc

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from multiprocessing import cpu_count

from xgboost import XGBClassifier

#############################################

url = 'https://raw.githubusercontent.com/malikrb/HumanResourcesDemonstration/master/data/hr_data.csv'
df = pd.read_csv(url)

##########################
###### Preprocessing #####
##########################

columns = df.select_dtypes(include='object').columns
for col in columns:
    df[col] = LabelEncoder().fit_transform(df[col])

df = pd.get_dummies(df)

del columns
gc.collect()

##########################
##### Model Building #####
##########################

X = df.drop('Attrition', axis=1)
y = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def algorithm_pipeline(X_train_data, X_test_data, y_train_data, y_test_data, 
                       model, param_grid, cv=10, scoring_fit='accuracy',
                       do_probabilities=False):
    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid, 
        cv=cv,  
        scoring=scoring_fit,
        verbose=2,
        n_jobs=cpu_count()//2,
    )
    fitted_model = gs.fit(X_train_data, y_train_data)
    
    if do_probabilities:
      pred = fitted_model.predict_proba(X_test_data)
    else:
      pred = fitted_model.predict(X_test_data)
    
    score = accuracy_score(pred, y_test)
    
    return fitted_model, pred, score

param_grid = {
    'colsample_bytree': [0.7],
    'learning_rate': [0.01],
    'max_depth': [5],
    'n_estimators': [500],
    'reg_alpha': [1.1],
    'reg_lambda': [1.2],
    'subsample': [0.8],
#     'colsample_bytree': [0.7, 0.8],
#     'learning_rate': [0.01, 0.05],
#     'n_estimators': [500, 1000],
#     'max_depth': [5, 10],
#     'reg_alpha': [1.1, 1.2, 1.3],
#     'reg_lambda': [1.1, 1.2, 1.3],
#     'subsample': [0.7, 0.8, 0.9]
}

model = XGBClassifier()

xgb_model, xgb_pred, xgb_score = algorithm_pipeline(X_train, X_test, y_train, y_test, model, 
                                        param_grid, cv=5, scoring_fit='accuracy')

del df
gc.collect()

print("\n" + "Accuracy:", xgb_score)