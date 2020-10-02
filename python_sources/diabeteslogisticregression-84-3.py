#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

diabetes_df = pd.read_csv('../input/diabetes.csv')
diabetes_df.head()


# In[2]:


diabetes_df.info()


# In[3]:


percentiles = [0.15, 0.3, 0.75, 0.9]
diabetes_df.describe(percentiles = percentiles)


# In[4]:


def fill_missing_data_byPredict(df, fillColumns, dataColumns):
    needFill_df = df[dataColumns]
    
    known_df = needFill_df[needFill_df[fillColumns] != 0].as_matrix()
    unknown_df = needFill_df[needFill_df[fillColumns] == 0].as_matrix()
    
    y = known_df[:, 0]
    
    X = known_df[:,1:]
    
    
    rfr = RandomForestRegressor(oob_score = True, n_jobs = -1,random_state =50,
                                max_features = "auto", min_samples_leaf = 5)
    rfr.fit(X, y)
    
    predictData = rfr.predict(unknown_df[:, 1:])
    df.loc[(df[fillColumns] == 0), fillColumns] = predictData
    
    return df

print('The Numbers Of AbnormalityGlucoseDatas:', diabetes_df[diabetes_df['Glucose'] == 0].shape[0])
print('The Numbers Of AbnormalityBloodPressureDatas:', diabetes_df[diabetes_df['BloodPressure'] == 0].shape[0])
print('The Numbers Of AbnormalitySkinThicknessDatas:', diabetes_df[diabetes_df['SkinThickness'] == 0].shape[0])
print('The Numbers Of AbnormalityInsulinDatas:', diabetes_df[diabetes_df['Insulin'] == 0].shape[0])
print('The Numbers Of AbnormalityBMIDatas:', diabetes_df[diabetes_df['BMI'] == 0].shape[0])

diabetes_df_mod = diabetes_df[(diabetes_df.Glucose != 0) & (diabetes_df.BloodPressure != 0) & (diabetes_df.BMI != 0)]
print('The Shape Of Normal Data:', diabetes_df_mod.shape)


# In[5]:


SkinThickness_fillColumns = ['SkinThickness', 'Pregnancies', 'Glucose', 'BloodPressure', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
diabetes_df_deal = fill_missing_data_byPredict(diabetes_df_mod, 'SkinThickness', SkinThickness_fillColumns)
diabetes_df_deal.head()


# In[6]:


Insulin_fillColumns = ['Insulin', 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
diabetes_df_deal = fill_missing_data_byPredict(diabetes_df_mod, 'Insulin', Insulin_fillColumns)
diabetes_df_deal.head()


# In[7]:


diabetes_df[['Pregnancies', 'Outcome']].groupby(['Pregnancies'], as_index = False).mean().sort_values(by = 'Pregnancies', ascending=False)


# In[8]:


fig = plt.figure()
fig.set(alpha = 0.2)

Outcome_1 = diabetes_df.Pregnancies[diabetes_df.Outcome == 1].value_counts()
Outcome_0 = diabetes_df.Pregnancies[diabetes_df.Outcome == 0].value_counts()

df = pd.DataFrame({'ill':Outcome_1, 'normal':Outcome_0})
df.plot(kind = 'line', stacked = False)
plt.title('The Relationship Between The Number Of Pregnancies And The Illness.')
plt.xlabel('The Number Of Pregnancies')
plt.ylabel('The Number Of Samples')
plt.show()


# In[9]:


data = diabetes_df_deal.iloc[:, 0:-1]
target = diabetes_df_deal['Outcome']

X_train_data, X_test_data, y_train_target, y_test_target = train_test_split(data, target, test_size = 0.3, random_state = 4)

lr = LogisticRegression()

parameters = [
    {'C' : np.arange(1, 10), 'penalty' : ['l1'], 'solver' : ['liblinear'], 'multi_class' : ['ovr']},
    {'C' : np.arange(1, 10), 'penalty' : ['l2'], 'solver' : ['liblinear', 'newton-cg', 'lbfgs'], 'multi_class' : ['ovr']}
]

grid = GridSearchCV(estimator = lr, param_grid = parameters, cv = 5)
lr.fit(X_train_data, y_train_target)


# In[10]:


roc_auc = np.mean(cross_val_score(grid, X_test_data, y_test_target, cv=5, scoring='roc_auc'))
print('roc_auc:{}', np.around(roc_auc, decimals = 4))

