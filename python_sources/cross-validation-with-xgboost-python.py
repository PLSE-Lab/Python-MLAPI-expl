#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 14:10:01 2017

@author: marcelo
"""

import numpy as np
import pandas as pd
import xgboost as xgb

x_train = pd.read_csv('../input/diabetic_data.csv', na_values='?')

x_train = x_train.drop(['encounter_id', 'patient_nbr'] , axis=1);


x_train.loc[x_train.age== '[0-10)','age'] = 0;
x_train.loc[x_train.age== '[10-20)','age'] = 10;
x_train.loc[x_train.age== '[20-30)','age'] = 20;
x_train.loc[x_train.age== '[30-40)','age'] = 30;
x_train.loc[x_train.age== '[40-50)','age'] = 40;
x_train.loc[x_train.age== '[50-60)','age'] = 50;
x_train.loc[x_train.age== '[60-70)','age'] = 60;
x_train.loc[x_train.age== '[70-80)','age'] = 70;
x_train.loc[x_train.age== '[80-90)','age'] = 80;
x_train.loc[x_train.age== '[90-100)','age'] = 90;
x_train.age = x_train.age.astype(np.int32)


x_train.loc[x_train.weight== '[0-25)','weight'] = 0;
x_train.loc[x_train.weight== '[25-50)','weight'] = 25;
x_train.loc[x_train.weight== '[50-75)','weight'] = 50;
x_train.loc[x_train.weight== '[75-100)','weight'] = 75;
x_train.loc[x_train.weight== '[100-125)','weight'] = 100;
x_train.loc[x_train.weight== '[125-150)','weight'] = 125;
x_train.loc[x_train.weight== '[150-175)','weight'] = 150;
x_train.loc[x_train.weight== '[175-200)','weight'] = 175;
x_train.loc[x_train.weight== '>200','weight'] = -100;
x_train.weight = x_train.weight.astype(np.float32)


x_train.loc[x_train.max_glu_serum== 'None','max_glu_serum'] = 0;
x_train.loc[x_train.max_glu_serum== 'Norm','max_glu_serum'] = 100;
x_train.loc[x_train.max_glu_serum== '>200','max_glu_serum'] = 200;
x_train.loc[x_train.max_glu_serum== '>300','max_glu_serum'] = 300;
x_train.max_glu_serum = x_train.max_glu_serum.astype(np.int32)


x_train.loc[x_train.A1Cresult== 'None','A1Cresult'] = 0;
x_train.loc[x_train.A1Cresult== 'Norm','A1Cresult'] = 5;
x_train.loc[x_train.A1Cresult== '>7','A1Cresult'] = 7;
x_train.loc[x_train.A1Cresult== '>8','A1Cresult'] = 8;
x_train.A1Cresult = x_train.A1Cresult.astype(np.int32)


x_train.loc[x_train.change== 'No','change'] = 0;
x_train.loc[x_train.change== 'Ch','change'] = 1;
x_train.change = x_train.change.astype(np.int8)



x_train.loc[x_train.diabetesMed== 'No','diabetesMed'] = 0;
x_train.loc[x_train.diabetesMed== 'Yes','diabetesMed'] = 1;
x_train.diabetesMed = x_train.diabetesMed.astype(np.int8)


medications = ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"]

for med in medications:
    x_train.loc[x_train[med] == 'No', med] = -20;
    x_train.loc[x_train[med] == 'Down', med] = -10;
    x_train.loc[x_train[med] == 'Steady', med] = 0;
    x_train.loc[x_train[med] == 'Up', med] = 10;
    x_train[med] = x_train[med].astype(np.int32)
    

categoricals = ['race', 'gender', 'payer_code', 'medical_specialty','diag_1', 'diag_2', 'diag_3']



for c in categoricals:
    x_train[c] = pd.Categorical(x_train[c]).codes


x_train.loc[x_train.readmitted != 'NO','readmitted'] = 0;
x_train.loc[x_train.readmitted == 'NO','readmitted'] = 1;

#x_train.loc[x_train.readmitted != '<30','readmitted'] = 0;
#x_train.loc[x_train.readmitted == '<30','readmitted'] = 1;
x_train.readmitted = x_train.readmitted.astype(np.int8)

y_train = x_train.readmitted
x_train = x_train.drop('readmitted', axis=1)


n_folds = 5
early_stopping = 10
params = {'eta': 0.02, 'max_depth': 5, 'subsample': 0.7, 'colsample_bytree': 0.7, 'objective': 'binary:logistic', 'seed': 99, 'silent': 1, 'eval_metric':'auc', 'nthread':4}

xg_train = xgb.DMatrix(x_train, label=y_train);

cv = xgb.cv(params, xg_train, 5000, nfold=n_folds, early_stopping_rounds=early_stopping, verbose_eval=1)

