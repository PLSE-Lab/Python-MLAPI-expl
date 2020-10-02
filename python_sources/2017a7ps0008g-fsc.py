#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#THIS CONTAINS THE CODE FOR THE BEST SUBMISSION


# In[ ]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import math
import copy
from scipy import stats as stat
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.model_selection import train_test_split
import random
from sklearn.linear_model import LogisticRegression

random.seed(32)
np.random.seed(36)


# In[ ]:


data_train = pd.read_csv("../input/data-mining-assignment-2/train.csv", sep=',')
data_test = pd.read_csv("../input/data-mining-assignment-2/test.csv", sep=',')

data_train.head()


# # Analysis

# In[ ]:


corr = data_train.corr()
corr
graph ,axis= plt.subplots()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
square=True, ax=axis, annot = False)


# In[ ]:


print(data_train[data_train.columns[0:]].corr()['Class'][:].sort_values(ascending=False)[:15])
print(data_train[data_train.columns[0:]].corr()['Class'][:].sort_values(ascending=True)[:15])


# In[ ]:


data_train.drop(['Class'],axis=1).hist(figsize=(20,20))


# In[ ]:


data_test.hist(figsize=(20,20))


# In[ ]:


data_train.describe()


# In[ ]:


data_test.describe()


# # Pre Processing

# In[ ]:


#Remove outliers
indexNames = data_train[ (data_train['ID'] == 478) | (data_train['ID'] == 594) | (data_train['ID'] == 288) ].index
data_train = data_train.drop(indexNames)


# In[ ]:


data_train=data_train.drop(['col5','col6','col19','col30','col46','col60','col59'],axis=1)
data_test=data_test.drop(['col5','col6','col19','col30','col46','col60','col59'],axis=1)

data_train["col49"] = 15.6 * data_train["col49"]

#drop all float
# data_train=data_train.drop(['col13','col23','col26','col32','col38','col42','col49','col54',],axis=1)
# data_test=data_test.drop(['col13','col23','col26','col32','col38','col42','col49','col54',],axis=1)


# In[ ]:


data_train_0=data_train[(data_train['Class']==0)]
data_train_1=data_train[(data_train['Class']==1)]
data_train_2=data_train[(data_train['Class']==2)]
data_train_3=data_train[(data_train['Class']==3)]


# In[ ]:


data_train_0.hist(figsize=(20,20))


# In[ ]:


data_train_1.hist(figsize=(20,20))


# In[ ]:


data_train_2.hist(figsize=(20,20))


# In[ ]:


data_train_3.hist(figsize=(20,20))


# In[ ]:


data_train=pd.get_dummies(data_train,columns=['col11','col37','col44'])
data_test=pd.get_dummies(data_test,columns=['col11','col37','col44'])

data_train=data_train.drop(['col11_Yes','col37_Female','col44_Yes'],axis=1)
data_test=data_test.drop(['col11_Yes','col37_Female','col44_Yes'],axis=1)

def replace_objs(s):
    map1 = {"Silver":0,"Gold":1,"Diamond":2,"Platinum":3,"Low":0,"Medium":1,"High":2}
    return map1[s]

column_objects=['col2','col56']
for name in column_objects:
    data_train[name] = data_train[name].apply(replace_objs)
for name in column_objects:
    data_test[name] = data_test[name].apply(replace_objs)


# In[ ]:


data_train_copy=data_train.copy()
data_test_copy=data_test.copy()


# Remove on Basis of High Correlation among features and Low Correlation with output Class

# In[ ]:


# ll=data_train[data_train.columns[1:]].corr()['Class'][:]
# corr = data_train.corr()
# columns = np.full((corr.shape[0],), True, dtype=bool)
# for i in range(len(ll)):
#     if(ll[i]<0.09 and ll[i]>-0.09):
#         columns[i+1]=False
# columns[3]=True
# columns=columns.copy()
# selected_columns = data_train.columns[columns]
# data_train = data_train[selected_columns]
# selected_columns =selected_columns[:-1]
# data_test = data_test[selected_columns]


# In[ ]:


data_1=data_train[(data_train['Class']==1)]
data_train=pd.concat([data_train,data_1])
data_1=data_train[(data_train['Class']==2)]
data_train=pd.concat([data_train,data_1])


# In[ ]:


Y_train=data_train['Class']
data_train=data_train.drop(['ID','Class'],axis=1)


# In[ ]:


data_out=pd.DataFrame()
data_out['ID']=data_test['ID']
data_test=data_test.drop(['ID'],axis=1)


# In[ ]:


# corr = data_train.corr()
# columns = np.full((corr.shape[0],), True, dtype=bool)
# for i in range(corr.shape[0]):
#     for j in range(i+1, corr.shape[0]):
#         if corr.iloc[i,j] >= 0.96:
# #             print(i,j)
#             if columns[j]:
# #                 print("--",j)
#                 columns[j] = False
# selected_columns = data_train.columns[columns]
# data_train = data_train[selected_columns]
# data_test = data_test[selected_columns]

# # selected_columns


# Normalisation and PCA

# In[ ]:


scaler=StandardScaler()
sclrfit=scaler.fit(pd.concat([data_train,data_test]))
data_train_norm=sclrfit.transform(data_train)
data_test_norm=sclrfit.transform(data_test)
data_train_n=pd.DataFrame(data_train_norm)
data_test_n=pd.DataFrame(data_test_norm)     


# In[ ]:


pca10 = PCA(n_components=5,random_state=32)
pca10.fit(pd.concat([data_train_n,data_test_n]))
data_train_pca = pca10.transform(data_train_n)
data_test_pca = pca10.transform(data_test_n)  
data_train_pca=pd.DataFrame(data_train_pca)
data_test_pca=pd.DataFrame(data_test_pca)
data_train=pd.concat([data_train.reset_index(drop=True),data_train_pca.reset_index(drop=True)],axis=1)
data_test=pd.concat([data_test.reset_index(drop=True),data_test_pca.reset_index(drop=True)],axis=1)


# In[ ]:


X_train,X_val,y_train,y_val=train_test_split(data_train, Y_train, test_size=0.13,random_state=42,stratify = Y_train)


# In[ ]:


data_train


# # FINAL MODEL
# Methods used for Parameter Tuning shown in ID_PSC file.

# In[ ]:


rf = RandomForestClassifier(n_estimators = 2000,criterion='entropy',max_features=25,random_state=18,max_depth=9,min_samples_split=4,min_samples_leaf=2) 
rf.fit(X_train,y_train)


# In[ ]:


cfm = confusion_matrix(y_val, rf.predict(X_val), labels = [0,1,2,3])
cfm


# In[ ]:


print(classification_report(y_val,rf.predict(X_val)))


# Generate Output

# In[ ]:


rf.fit(data_train,Y_train)
out=rf.predict(data_test)


# In[ ]:


data_out['Class']=pd.DataFrame(out)


# In[ ]:


from IPython.display import HTML
import base64
def create_download_link(df, title = "Download CSV file", filename = "upsample_RF simple_final_last.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)
create_download_link(data_out)


# In[ ]:




