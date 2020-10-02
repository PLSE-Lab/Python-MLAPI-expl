#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# + Overexposure to certain environmental factors, such as pesticides and other chemicals, and radiation.
# 
# + Cigarette smoking, alcohol, marijuana, anabolic steroids, and taking medications to treat bacterial infections, high blood pressure and depression also can affect fertility.
# 
# + Frequent exposure to heat, such as in saunas or hot tubs, can raise body temperature and may affect sperm production.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,confusion_matrix
from imblearn.over_sampling import RandomOverSampler


# In[ ]:


df = pd.read_csv("/kaggle/input/fertility-data-set/fertility.csv")
df.head()


# In[ ]:


df.columns.values


# In[ ]:


df['Frequency of alcohol consumption'].value_counts()


# In[ ]:


req_cols = ['Age','Number of hours spent sitting per day''Season_fall',
 'Season_spring','Season_summer','Season_winter','Childish diseases_no',
 'Childish diseases_yes']


# In[ ]:


df['Frequency of alcohol consumption'].value_counts()


# In[ ]:


df['Smoking habit'].value_counts()


# In[ ]:


df['High fevers in the last year'].value_counts()


# In[ ]:


def change_target(y):
    y = y.apply(lambda x : 0 if x=='Normal' else 1)
    return y

def preprocessing(df):
    df['Season_risk'] = df['Season'].apply(lambda x : 1 if x=='summer' else 0)
    dummy_cols = ['Season']
    df = pd.get_dummies(df,columns=dummy_cols)
    df['Childish diseases'] = df['Childish diseases'].apply(lambda x : 1 if x=='yes' else 0)
    df['Smoking habit'] = df['Smoking habit'].apply(lambda x : -1 if x=='Never' else 1 if x=='Daily' else 0)
    df['Frequency of alcohol consumption'] = df['Frequency of alcohol consumption'].apply(lambda x : -1 if x=='hardly ever or never'
    else 1 if x=='once a week' else 2 if x=='several times a week' else 3 if x=='every day' else 4)
    df['High fevers in the last year'] = df['High fevers in the last year'].apply(lambda x : -1 if x=='no'
    else 1 if x=='more than 3 months ago' else 2)
    df['Accident or serious trauma'] = df['Accident or serious trauma'].apply(lambda x : 1 if x=='yes' else 0)
    df['Surgical intervention'] = df['Surgical intervention'].apply(lambda x : 1 if 'yes' else 0)
    cols = df.columns.values
    for x in req_cols:
        if x not in cols:
            df[x]=0
    return df

def scaling(df,scaler=None):
    if scaler==None:
        sc=StandardScaler()
        sc.fit(df)
        df = sc.transform(df)
        pkl.dump(sc,open("fertility_scaler.pkl",'wb'))
    else:
        df = scaler.transform(df)
    return df


# In[ ]:


y = df['Diagnosis']
X = df.drop(columns=['Diagnosis'])


# In[ ]:


# performed over-sampling as the cases of infertility are low
ros = RandomOverSampler(random_state=0)
X, y = ros.fit_resample(X, y)
from collections import Counter
print(sorted(Counter(y).items()))
y = pd.Series(y)
X = pd.DataFrame(X,columns=['Season','Age', 'Childish diseases', 'Accident or serious trauma',
       'Surgical intervention', 'High fevers in the last year',
       'Frequency of alcohol consumption', 'Smoking habit',
       'Number of hours spent sitting per day'])


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=0,stratify=y)


# In[ ]:


y_train = change_target(y_train)
X_train = preprocessing(X_train)
X_train = scaling(X_train)


# In[ ]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(penalty='l2')
logreg.fit(X_train,y_train)


# In[ ]:


y_test = change_target(y_test)
X_test = preprocessing(X_test)
X_test = scaling(X_test)


# In[ ]:


y_pred = logreg.predict(X_test)
confusion_matrix(y_test,y_pred)


# In[ ]:


f1_score(y_test,y_pred)


# In[ ]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# In[ ]:


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plot_roc_curve(fpr, tpr)


# In[ ]:


auc = roc_auc_score(y_test, y_pred)
print('AUC: %.2f' % auc)

