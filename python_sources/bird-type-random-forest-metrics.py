#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv("../input/birds-bones-and-living-habits/bird.csv")


# In[ ]:


df.head()


# In[ ]:


def initial_observation(df):
    if isinstance(df, pd.DataFrame):
        total_na = df.isna().sum().sum()
        print("Dimensions : %d rows, %d columns" % (df.shape[0], df.shape[1]))
        print("Total NA Values : %d " % (total_na))
        print("%38s %10s     %10s %10s" % ("Column Name", "Data Type", "#Distinct", "NA Values"))
        col_name = df.columns
        dtyp = df.dtypes
        uniq = df.nunique()
        na_val = df.isna().sum()
        for i in range(len(df.columns)):
            print("%38s %10s   %10s %10s" % (col_name[i], dtyp[i], uniq[i], na_val[i]))
        
    else:
        print("Expect a DataFrame but got a %15s" % (type(df)))


# In[ ]:


initial_observation(df)


# In[ ]:


df["type"].value_counts()


# In[ ]:


sns.catplot(x="type", y="huml", data= df);


# In[ ]:


sns.catplot(x="type", y="humw", data= df);


# In[ ]:


sns.catplot(x="type", y="ulnal", data= df);


# In[ ]:


sns.catplot(x="type", y="ulnaw", data= df);


# In[ ]:


sns.catplot(x="type", y="feml", data= df);


# In[ ]:


sns.catplot(x="type", y="femw", data= df);


# In[ ]:


sns.catplot(x="type", y="tibl", data= df);


# In[ ]:


sns.catplot(x="type", y="tibw", data= df);


# In[ ]:


sns.catplot(x="type", y="tarl", data= df);


# In[ ]:


sns.catplot(x="type", y="tarw", data= df);


# In[ ]:


df1 = df.copy()


# In[ ]:


df1 = df1.dropna(axis=0, subset=['huml', "humw", "ulnal", "ulnaw", "feml", "femw", "tibl", "tibw", "tarl", "tarw"])


# In[ ]:


initial_observation(df1)


# In[ ]:


df["type"].value_counts()


# In[ ]:


df1['type'].value_counts().plot(kind='bar');


# In[ ]:


df_oversample = df1.copy()
df_undersample = df1.copy()
df_smote = df1.copy()


# In[ ]:


df_oversample['type'].value_counts().plot(kind='bar');


# In[ ]:


from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


# In[ ]:


df2 = df.copy()


# In[ ]:


x = df1.drop(["type"], axis = 1)
y = df1["type"]


# In[ ]:


x.head()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(x,y)


# In[ ]:


X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train_scaled)
x_train_scaled_1 = scaler.transform(X_train_scaled)
x_val_scaled_1 = scaler.transform(X_val_scaled)


# In[ ]:


print("X Train shape:" , X_train.shape)
print("X Validation shape:" ,   X_val.shape)
print("Y Train shape:",     Y_train.shape)
print( "Y Validation Shape:",   Y_val.shape)


# ### Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# In[ ]:


rf_parm = dict(n_estimators = [20, 30, 50, 70, 100, 150], max_features = [0.1, 0.2, 0.6, 0.9], max_depth = [10,20,30],min_samples_leaf=[1,10,100, 400, 500, 600],random_state=[0])


# In[ ]:


rc = RandomForestClassifier()
rf_grid = GridSearchCV(estimator = rc, param_grid = rf_parm)


# In[ ]:


rf_grid.fit(X_train,Y_train)


# In[ ]:


print("RF Best Score:", rf_grid.best_score_)
print("RF Best Parameters:", rf_grid.best_params_)


# In[ ]:


rc_best = RandomForestClassifier(n_estimators = 20,  max_features = 0.9)


# In[ ]:


rc_best.fit(X_train, Y_train)
rc_tr_pred = rc_best.predict(X_train)
rc_val_pred = rc_best.predict(X_val)


# In[ ]:


print(rc_val_pred)


# In[ ]:


from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


# In[ ]:


print("Precision Score : ",precision_score(Y_val, rc_val_pred , 
                                           pos_label='positive',
                                           average='weighted'))
print("Recall Score : ",recall_score(Y_val, rc_val_pred , 
                                           pos_label='positive',
                                           average='weighted'))
print("F1 Score:",  f1_score(Y_val, rc_val_pred , 
                                           pos_label='positive',
                                           average='weighted'))


# In[ ]:


from sklearn.metrics import classification_report

print(classification_report(Y_val, rc_val_pred))

