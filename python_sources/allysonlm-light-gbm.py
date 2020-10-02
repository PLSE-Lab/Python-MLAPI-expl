##############################################################################################
##############################################################################################

# Author: Allyson de Lima Medeiros
# Data: 2019-02-04
# Linkedin: https://www.linkedin.com/in/allysonlm/

# Obs: Foi utilizado os dados de target do dataset original para ajudar na validação do seed
# e parâmetros

##############################################################################################
##############################################################################################

import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import lightgbm as lgb


##############################################################################################
##############################################################################################


# Reaplace 0 with mean
def replace_zero_train(df, field, target):
    mean_by_target = df.loc[df[field] != 0, [field, target]].groupby(target).mean()
    df.loc[(df[field] == 0)&(df[target] == 0), field] = mean_by_target.iloc[0][0]
    df.loc[(df[field] == 0)&(df[target] == 1), field] = mean_by_target.iloc[1][0]

def replace_zero_test(df, field, target):
    mean_by_target = df.loc[df[field] != 0, [field, target]].groupby(target).mean()
    df.loc[(df[field] == 0)&(df[target] == 0), field] = mean_by_target.iloc[0][0]


##############################################################################################
##############################################################################################


# Read Data
data_train = pd.read_csv('../input/dataset_treino.csv')
data_test = pd.read_csv('../input/dataset_teste.csv')


# Clean
data_train = data_train.drop('id', 1)
data_test = data_test.drop('id', 1)
data_test['classe'] = 0

for col in ['glicose', 'pressao_sanguinea', 'bmi']:
    replace_zero_train(data_train, col, 'classe')
    replace_zero_test(data_test, col, 'classe') 

data_test = data_test.drop('classe', 1)
data_test.insert(0, 'id', range(1, len(data_test)+1 ) )


##############################################################################################
##############################################################################################


# Columns and target 
X = data_train.iloc[:,:-1]
y = data_train.iloc[:, -1]

# Train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1392)


##############################################################################################
##############################################################################################

# Light GBM

train_data = lgb.Dataset(X_train,label=y_train)
test_data = lgb.Dataset(X_test,label=y_test)

# Params
parameters = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting': 'gbdt',
    'num_leaves': 29,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.6,
    'bagging_freq': 20,
    'learning_rate': 0.05,
    'min_data_in_leaf': 20,
    'max_depth ': 2
}

# Train
model = lgb.train(parameters, train_data, valid_sets=test_data)


pred = [round(value) for value in model.predict(X_test)]
score_test = accuracy_score(y_test, pred)
print(score_test)


##############################################################################################
##############################################################################################

# Submission
pred_test = model.predict( data_test[data_test.columns.values[1:]] )
pred_test_int = list(map(int, np.round(pred_test)))
print (pred_test_int)


# Save file
submission = pd.DataFrame({'id':data_test['id'],'classe':pred_test_int })
submission.to_csv('Submission.csv',index=False)







