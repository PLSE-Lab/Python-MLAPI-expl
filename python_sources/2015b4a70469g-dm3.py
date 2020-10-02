#!/usr/bin/env python
# coding: utf-8

# # Importing Packages

# In[ ]:


import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score


# # Preprocessing Data

# In[ ]:


mal_data=pd.read_csv('../input/opcode_frequency_malware.csv')


# In[ ]:


benign_data=pd.read_csv('../input/opcode_frequency_benign.csv')


# In[ ]:


data_test = pd.read_csv('../input/Test_data.csv')
indexes = data_test['FileName']


# In[ ]:


data_test = data_test.drop(['Unnamed: 1809','FileName','1808'],axis=1)


# In[ ]:


benign_data['1808']=0
mal_data['1808']=1
datacol=[mal_data,benign_data]
data=pd.concat(datacol)


# In[ ]:


data=data.drop_duplicates()


# In[ ]:


data=data.drop(columns=['FileName'],axis=1)


# In[ ]:


X=data.drop(columns=['1808'],axis=1)
y=data['1808']


# # KNeighborsClassifier

# In[ ]:


knn=KNeighborsClassifier(n_neighbors=3).fit(X,y)
print(roc_auc_score(knn.predict(X),y))


# # Decision Tree Classifier

# In[ ]:


dec=DecisionTreeClassifier(max_depth=12).fit(X,y)
print(roc_auc_score(dec.predict(X),y))


# # RandomForest Classsifier

# In[ ]:


rnd=RandomForestClassifier(max_depth=20,n_estimators=100).fit(X,y)
print(roc_auc_score(rnd.predict(X),y))


# # Gradient boosting Classifier

# In[ ]:


gbc = GradientBoostingClassifier(n_estimators=100,max_depth=20).fit(X,y)
print(roc_auc_score(gbc.predict(X_test),y_test))


# In[ ]:


preds = rnd.predict(data_test)
df = pd.DataFrame(columns=['FileName','Class'])
df['FileName'] = indexes
df['Class'] = preds
df.to_csv('2015B4A70469G.csv',index=False)


# In[ ]:


from IPython.display import HTML
import base64
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)
create_download_link(df)

