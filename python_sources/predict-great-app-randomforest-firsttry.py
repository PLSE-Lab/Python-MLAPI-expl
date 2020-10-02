#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, roc_auc_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Load Data

# In[ ]:


df = pd.read_csv("/kaggle/input/17k-apple-app-store-strategy-games/appstore_games.csv")


# In[ ]:


df.head()


# # Feature Engineering

# In[ ]:


def in_app_p(row):
    
    x=row["In-app Purchases"]
    if isinstance(x,np.float64) or pd.isnull(x):
        row["In_App_Count"]= 0
        row["In_App_Max"]  = 0
    else:
        x_list=[float(x) for x in row["In-app Purchases"].split(",")]
        row["In_App_Count"]=len(x_list)
        row["In_App_Max"]=max(x_list)
    return row 

def languages(row):
    if pd.isnull(row["Languages"]):
        row["Languages"]="EN"
    if "EN" in row["Languages"]:
        row["Language_EN"]=1
    else:
        row["Language_EN"]=1
    row["Languages_Count"]=len(row["Languages"].split(","))
    
    return row

def genres(row):
    row["Genres_Count"] = len(row["Genres"].split())
    return row

###############################################################################################################################

df2 = df.loc[df["User Rating Count"]>=10,:].copy()

df2 = df2.assign(Great_App=lambda x: np.where(x["Average User Rating"]>=4.5,1,0))         .assign(Subtitle_Present=lambda x: np.where(x["Subtitle"].isnull(),0,1))         .assign(Price=lambda x: np.where(x["Price"]>=10,10,x["Price"]))         .assign(Price_Free=lambda x: np.where(x["Price"]==0,1,0))         .assign(Age_Rating=lambda x: x["Age Rating"].str.replace("+","").astype(int))         .assign(Description_Length=lambda x: x["Description"].str.len())         .apply(genres,axis=1)         .apply(languages,axis=1)         .apply(in_app_p,axis=1)         .drop(columns=["URL","ID","Name","Subtitle","Icon URL","Primary Genre","In-app Purchases","Developer","Description","Languages","Average User Rating",
                        "Original Release Date","Current Version Release Date","Genres","Age Rating"])

df2.head()


# # Build Forest

# In[ ]:


X=df2.drop(columns=["Great_App"])
y=df2["Great_App"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)

rf = RandomForestClassifier(n_estimators=100)

param_grid = { 
    'n_estimators': [300, 500, 750],
    'max_features': ['auto', 'sqrt'],
    'max_depth' : [10,15,20],
    'criterion' :['gini']
}

cv_rf = GridSearchCV(estimator=rf, param_grid=param_grid, scoring="roc_auc", cv= 5)

cv_rf.fit(X_train,y_train)

#cross_val_score(rf,X_train,y_train, scoring="accuracy",cv=5)


# In[ ]:


cv_rf.best_params_


# In[ ]:


cv_rf.best_score_


# # Test Score

# In[ ]:


final_model = cv_rf.best_estimator_
final_model.fit(X_train,y_train)
print(confusion_matrix(y_test,final_model.predict(X_test)))
print(roc_auc_score(y_test,final_model.predict(X_test)))


# In[ ]:




