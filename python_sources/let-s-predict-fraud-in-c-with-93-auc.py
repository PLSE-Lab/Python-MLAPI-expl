#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


transaction_data = pd.read_csv("../input/train_transaction.csv")


# In[ ]:


identity_data = pd.read_csv(r"../input/train_identity.csv")


# In[ ]:


final_data = transaction_data.merge(identity_data,how = "left",on  = "TransactionID")

del transaction_data

del identity_data


# In[ ]:


def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
                
    return col_corr


# In[ ]:


final_data = final_data.query("ProductCD == 'C' ")

corr_features_1 = correlation(final_data, 0.8)

len(corr_features_1) 


# In[ ]:


final_data.drop(labels=corr_features_1, axis=1, inplace=True)

s = final_data.isnull().mean()*100

s = s.where(s >15).dropna().index

final_data.drop(labels=list(s), axis=1, inplace=True)

del final_data['TransactionID']

del final_data['ProductCD']

for col in ['card1',
 'card2',
 'card3',
 'card4',
 'card5',
 'card6',
 'P_emaildomain',
 'R_emaildomain',
 'M4',
  'DeviceType',
  'id_12',
 'id_13',
 'id_15',
 'id_17',
 'id_19',
 'id_20',
 'id_28',
 'id_29',
 'id_31',
 'id_35',
 'id_36',
 'id_37',
 'id_38']:
    
    temp_df = pd.Series(final_data[col].value_counts() / final_data[col].value_counts().sum())
    
    grouping_dict = {
        k: ('rare' if k not in temp_df[temp_df >= 0.1].index else k)
        for k in temp_df.index
    }

    final_data[col] = final_data[col].map(grouping_dict)

def impute_na(final_data, variable):
    most_frequent_category = final_data.groupby([variable])[variable].count().sort_values(ascending=False).index[0]
    final_data[variable].fillna(most_frequent_category, inplace=True)

for variable in ['card1',
 'card2',
 'card3',
 'card4',
 'card5',
 'card6',
 'P_emaildomain',
 'R_emaildomain',
 'M4',
  'DeviceType',
  'id_12',
 'id_13',
 'id_15',
 'id_17',
 'id_19',
 'id_20',
 'id_28',
 'id_29',
 'id_31',
 'id_35',
 'id_36',
 'id_37',
 'id_38']:
    
    impute_na(final_data, variable)

def impute_na(final_data, variable):
    
    final_data[variable] = final_data[variable].fillna(final_data[variable].median())

for variable in final_data.columns:
    
    if variable not in ['card1',
 'card2',
 'card3',
 'card4',
 'card5',
 'card6',
 'P_emaildomain',
 'R_emaildomain',
 'M4',
  'DeviceType',
  'id_12',
 'id_13',
 'id_15',
 'id_17',
 'id_19',
 'id_20',
 'id_28',
 'id_29',
 'id_31',
 'id_35',
 'id_36',
 'id_37',
 'id_38']:
        
        impute_na(final_data,variable)
        



    


# In[ ]:


S = pd.get_dummies(final_data)


# In[ ]:


quasi_constant_feat = []

for feature in S.columns:
    
    predominant = (S[feature].value_counts() / np.float(
        len(S))).sort_values(ascending=False).values[0]

    if predominant > 0.998:
        quasi_constant_feat.append(feature)


# In[ ]:


S.drop(labels=quasi_constant_feat, axis=1, inplace=True)


# In[ ]:


from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix,precision_score,f1_score,recall_score

from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler

from sklearn.feature_selection import VarianceThreshold


# In[ ]:


from imblearn.combine import SMOTETomek


# In[ ]:


from imblearn.pipeline import make_pipeline


# In[ ]:


pipe = make_pipeline(
    
    SMOTETomek(random_state=100),
    
    RandomForestClassifier(random_state=100,n_estimators= 200,n_jobs = -1)

)


# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


gsc = GridSearchCV(
    estimator=pipe,
    
    param_grid={

        'smotetomek__ratio': [.8],
        'randomforestclassifier__max_depth':[25],
    },
    
    scoring='f1',
    cv=3
)


# In[ ]:


import warnings

warnings.filterwarnings("ignore")


# In[ ]:


y = S.isFraud

X = S.drop("isFraud",axis = 1)

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state =1, stratify =y)


# In[ ]:


gsc.fit(x_train, y_train)


# In[ ]:


y_pred = gsc.best_estimator_.predict(x_test)

f1_score(y_pred,y_test),precision_score(y_pred,y_test),recall_score(y_pred,y_test)


# In[ ]:


from sklearn.metrics import auc,roc_curve


# In[ ]:


y_probs  = gsc.predict_proba(x_test)[:,1]


# In[ ]:


fp_rate, tp_rate, thresholds = roc_curve(y_test, y_probs)


# In[ ]:


auc(fp_rate, tp_rate)


# In[ ]:




