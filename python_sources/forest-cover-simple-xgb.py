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


import pandas as pd
import numpy as np
from pathlib import Path
#from sklearn. import 


# In[ ]:


PATH=Path('../input/')


# In[ ]:


raw_tr=pd.read_csv(PATH/'train.csv')
raw_te=pd.read_csv(PATH/'test.csv')


# In[ ]:


import seaborn as sns
from matplotlib import pyplot as plt


# In[ ]:


corrmat=raw_tr.corr()


# In[ ]:


fig,ax=plt.subplots(figsize=(16,12))
sns.heatmap(corrmat,ax=ax)


# Soil Type 7 and 15 have may be constant 

# In[ ]:


print(f'Unique elements in SoilType7: {raw_tr.Soil_Type7.unique()} | Expected 2')
print(f'Unique elements in SoilType15: {raw_tr.Soil_Type15.unique()} | Expected 2')


# We can drop these columnts as these are constant for all training data 

# In[ ]:


drop=['Soil_Type7','Soil_Type15']
raw_tr.drop(drop,axis=1,inplace=True)
raw_te.drop(drop,axis=1,inplace=True)


# In[ ]:


n=12
cols=corrmat.nlargest(n,'Cover_Type')['Cover_Type'].index
cm = np.corrcoef(raw_tr[cols].values.T)
sns.set(font_scale=1.25)
fig,ax=plt.subplots(figsize=(16,9))
sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values,ax=ax)
#plt.show()


# In[ ]:


{c:raw_tr[c].dtype.name for c in raw_tr.columns}


# In[ ]:


raw_tr.columns


# In[ ]:


em_cols1=['Hillshade_9am', 'Hillshade_Noon',
       'Hillshade_3pm']
em_cols2=['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3',
       'Wilderness_Area4', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3',
       'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type8', 'Soil_Type9',
       'Soil_Type10', 'Soil_Type11', 'Soil_Type12', 'Soil_Type13',
       'Soil_Type14', 'Soil_Type16', 'Soil_Type17', 'Soil_Type18',
       'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22',
       'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26',
       'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30',
       'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34',
       'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38',
       'Soil_Type39', 'Soil_Type40']
cont_col=['Elevation', 'Aspect', 'Slope',
       'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
       'Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points']


# In[ ]:


#{c:(len(raw_tr[c].unique())) for c in em_cols}


# In[ ]:


from pandas.api.types import CategoricalDtype
def to_cat(col_list):
    for c in col_list:
        if c in ['Hillshade_9am', 'Hillshade_Noon','Hillshade_3pm']:
            cat=CategoricalDtype(categories=[i for i in range(256)])
        else :
            cat=CategoricalDtype(categories=[i for i in range(2)])
        raw_tr[c]=raw_tr[c].astype(cat).cat.as_ordered()
        raw_te[c]=raw_te[c].astype(cat).cat.as_ordered()
        raw_tr[c]=raw_tr[c].cat.codes
        raw_te[c]=raw_te[c].cat.codes


# In[ ]:


#from sklearn.model_selection import train_test_split
to_cat(em_cols1)
to_cat(em_cols2)


# In[ ]:


cont_map={}
for c in cont_col:
    cont_map[c]=raw_tr[c].mean(),raw_tr[c].std()+1e-2


# In[ ]:


def apply_norm(df,mapping):
    for k,v in mapping.items():
        df[k]=(df[k]-v[0])/v[1]
    #return df


# In[ ]:


apply_norm(raw_tr,cont_map)


# In[ ]:


apply_norm(raw_te,cont_map)


# In[ ]:


from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[ ]:


xtrain,xval,ytrain,yval=train_test_split(raw_tr.drop(['Id','Cover_Type'],axis=1),raw_tr.Cover_Type,test_size=.2)


# In[ ]:


xgb=XGBClassifier(n_estimators=1000,max_depth=25)


# In[ ]:


xgb.fit(xtrain,ytrain)
y_pred=xgb.predict(xval)


# In[ ]:


print(accuracy_score(yval,y_pred))
score=cross_val_score(xgb,xtrain,ytrain,n_jobs=-1,cv=5)
print(np.mean(score))


# In[ ]:


yt=xgb.predict(raw_te.drop('Id',axis=1))


# In[ ]:


len(yt)


# In[ ]:


sub=pd.DataFrame({'Id':raw_te.Id,'Cover_Type':yt},columns=['Id','Cover_Type'])


# In[ ]:


sub.to_csv('submission_simple_XGB_1.csv',index=False)


# In[ ]:




