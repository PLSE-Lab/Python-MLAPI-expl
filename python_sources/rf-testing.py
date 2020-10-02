#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

data = pd.read_csv("../input/data.csv")

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# def test_it_RF(data_test):
#     clf = RandomForestClassifier(n_jobs=-1, n_estimators=64, max_depth=4, random_state=2016)
#     return cross_val_score(clf, data_test.drop('shot_made_flag', 1), data_test.shot_made_flag,
#                            scoring='roc_auc', cv=10
#                           )

# In[ ]:


# define the sort & enumeration function
def sort_encode(df, field):
    ct = pd.crosstab(df.shot_made_flag, df[field]).apply(lambda x:x/x.sum(), axis=0)
    temp = list(zip(ct.values[1, :], ct.columns))
    temp.sort()
    new_map = {}
    for index, (acc, old_number) in enumerate(temp):
        new_map[old_number] = index
    new_field = field + '_sort_enumerated'
    df[new_field] = df[field].map(new_map)
    return new_field


# In[ ]:


# action_type
new_field = sort_encode(data, 'action_type')
data_test = data[[new_field, 'shot_made_flag']].copy()
data_test = data_test.dropna()
auc = test_it_RF(data_test)
print(auc.mean())


# In[ ]:


# shot_distance
new_field = 'shot_distance'
data_test = data[[new_field, 'shot_made_flag']].copy()
data_test = data_test.dropna()
auc = test_it_RF(data_test)
print(auc.mean())


# In[ ]:


data['away'] = data.matchup.str.contains('@')


# In[ ]:


data['home_away']=99
data.loc[data.away==True, ['home_away']] = 0
data.loc[data.away==False, ['home_away']] = 1


# In[ ]:


data['xy'] = np.abs(data.loc_x)*np.abs(data.loc_y)


# In[ ]:


# Impute
mode = data.action_type_sort_enumerated.mode()[0]
data.action_type_sort_enumerated.fillna(mode, inplace=True)


# In[ ]:


train = data.loc[~data.shot_made_flag.isnull(),                 ['action_type_sort_enumerated', 'shot_distance', 'home_away', 'shot_made_flag']]
test = data.loc[data.shot_made_flag.isnull(),                ['action_type_sort_enumerated', 'shot_distance', 'home_away', 'shot_id']]

min_max_scaler = preprocessing.MinMaxScaler()
train_scaled = min_max_scaler.fit_transform(train.drop('shot_made_flag', axis=1))
test_scaled = min_max_scaler.transform(test.drop('shot_id', axis=1))


# clf = RandomForestClassifier(n_jobs=-1, n_estimators=70, max_depth=4, random_state=2016)
# 
# # Train and predict
# clf.fit(train_scaled, train.shot_made_flag)
# predictions = clf.predict_proba(test_scaled)
# 
# # convert to CSV
# submission = pd.DataFrame({'shot_id': test.shot_id,
#                            'shot_made_flag': predictions[:, 1]})
# submission[['shot_id', 'shot_made_flag']].to_csv('submission_RF.csv', index=False)
