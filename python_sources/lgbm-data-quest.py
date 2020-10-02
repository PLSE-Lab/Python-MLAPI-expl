#!/usr/bin/env python
# coding: utf-8

# # **Imports**

# In[ ]:


#Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


# # **Loading Dataset**

# In[ ]:


train = pd.read_csv('../input/pasc-data-quest-20-20/doctor_train.csv')
test = pd.read_csv('../input/pasc-data-quest-20-20/doctor_test.csv')


#train.info()
#test.info()

y = train['Y']

train = train.drop(['Y'], axis =1)


# # **Preprocessing dataset**

# In[ ]:


dataset = pd.concat([train, test])
dataset.info()

dataset.isna().sum()


# In[ ]:


dataset.drop(['ID'], axis = 1, inplace = True)

dataset.nunique()


# In[ ]:


dataset.loc[ dataset['age'] <= 25, 'age'] = 0
dataset.loc[(dataset['age'] > 25) & (dataset['age'] <= 32), 'age'] = 1
dataset.loc[(dataset['age'] > 32) & (dataset['age'] <= 40), 'age'] = 2
dataset.loc[(dataset['age'] > 40) & (dataset['age'] <= 48), 'age'] = 3
dataset.loc[(dataset['age'] > 48) & (dataset['age'] <= 60), 'age'] = 4
dataset.loc[ dataset['age'] > 60, 'age'] = 5


# In[ ]:


profession= pd.get_dummies(dataset['Profession'])
profession.sum()

dataset = pd.concat([dataset, profession], axis = 1)
dataset.drop(['Profession'], axis =1, inplace = True)


# In[ ]:


dataset['edu'].replace("unknown", np.NaN, inplace = True)
dataset['edu'].replace('primary', 1, inplace = True)
dataset['edu'].replace('secondary', 2, inplace = True)
dataset['edu'].replace('tertiary', 3, inplace = True)


# In[ ]:


dataset['Status'].replace('single', 0, inplace = True)
dataset['Status'].replace('married', 1, inplace= True)
dataset['Status'].replace('divorced', -1, inplace = True)


# In[ ]:


dataset['Irregular'].replace('no', 0, inplace = True)
dataset['Irregular'].replace('yes', 1, inplace = True)


# In[ ]:


sns.distplot(dataset['Money'])


# In[ ]:


dataset['Money'] = dataset['Money'].fillna(dataset['Money'].sum()/30403)
dataset['Money'] = np.cbrt(dataset['Money'])


# In[ ]:


dataset['residence'].nunique()
dataset['residence'].replace('no', 0, inplace = True)
dataset['residence'].replace('yes', 1, inplace = True)


# In[ ]:


dataset['prev_diagnosed'].replace('no', 0, inplace = True)
dataset['prev_diagnosed'].replace('yes', 1, inplace = True)


# In[ ]:


month = pd.get_dummies(dataset['Month'])
dataset = pd.concat([dataset, month], axis = 1)
dataset.drop(["Month"], axis = 1, inplace = True)


# In[ ]:


dataset['communication'].replace('cellular', 1, inplace = True)
dataset['communication'].replace('telephone', 0, inplace = True)
dataset['communication'].replace('unknown', -1, inplace = True)


# In[ ]:


dataset.loc[dataset['last_visit'] != -1, 'last_visit'] = 1
dataset.loc[dataset['last_visit'] == -1, 'last_visit'] = 0


# In[ ]:


dataset['side_effects'].replace('unknown', 0, inplace = True)
dataset['side_effects'].replace('other', 0, inplace = True)
dataset['side_effects'].replace('success', 1, inplace = True)
dataset['side_effects'].replace('failure', -1, inplace = True)


# In[ ]:


dataset.drop(['Irregular', 'cured_in', 'unknown'], axis = 1, inplace = True)


# In[ ]:


dataset['age'].fillna(dataset['age'].mean(), inplace = True)
dataset['edu'].fillna(2, inplace= True)
dataset['residence'].fillna(0, inplace = True)
dataset['prev_diagnosed'].fillna(0, inplace = True)


# In[ ]:


dataset['profession'] = dataset['admin.'] + dataset['management'] + dataset['retired'] + dataset['student'] + dataset['unemployed']

dataset.drop(['admin.',    'blue-collar',   'entrepreneur',
            'housemaid',     'management',        'retired',  'self-employed',
             'services',        'student',     'technician',     'unemployed',], axis = 1, inplace= True)


# In[ ]:


dataset['Time'] = np.log(dataset['Time'])
dataset.loc[(dataset['Time'] == np.inf), 'Time'] = 0
dataset.loc[(dataset['Time'] == -np.inf), 'Time'] = 0


# In[ ]:


dataset.describe()
dataset.info()
dataset.isna().sum()


# # **Spliting data back to train and test set**

# In[ ]:


train = dataset.iloc[:28196, :]
test = dataset.iloc[28196:, :]


# # **Fitting LightGBM Classifier**

# In[ ]:


from lightgbm import LGBMClassifier
lgc = LGBMClassifier(class_weight = {'no':0.11, 'yes': 0.89}, objective = 'binary', learning_rate = 0.035)
lgc.fit(train, y)
y_pred_lgc = lgc.predict(test)


# In[ ]:


for i in range(dataset.shape[1]):
    print(dataset.columns[i], ':  ', lgc.feature_importances_[i])


# **Checking output**

# In[ ]:


Counter(y_pred_lgc)


# # **Converting output to csv file**

# In[ ]:


y_pred_lgc = pd.DataFrame(y_pred_lgc)
y_pred_lgc.to_csv('lgc.csv')

