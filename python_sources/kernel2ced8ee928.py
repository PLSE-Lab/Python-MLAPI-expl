#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as py
import seaborn as sns
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split 
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score


# In[ ]:


df = pd.read_csv("../input/googleplaystore.csv")


# In[ ]:


df_copy =df.copy()


# In[ ]:


df.drop(columns=["App"],inplace=True,axis=1)
df.drop(columns=["Current Ver" ,"Android Ver"],inplace=True,axis=1)
df.drop(columns=["Last Updated"],inplace=True,axis=1)


# In[ ]:



df['Size'] = df['Size'].apply(lambda x: str(x).replace('Varies with device', 'NaN') if 'Varies with device' in str(x) else x)
df['Size'] = df['Size'].apply(lambda x: str(x).replace('M', '') if 'M' in str(x) else x)
df['Size'] = df['Size'].apply(lambda x: str(x).replace('k', '') if 'k' in str(x) else x)

df.Size = pd.to_numeric(df.Size, errors='coerce')
scaler = MinMaxScaler()
# MinMaxScaler(copy=True, feature_range=(0, 1))
df["Size"]=scaler.fit_transform(df["Size"].values.reshape(-1,1))


# In[ ]:


df = df.dropna(subset=["Installs"])
df['Installs'] = df['Installs'].apply(lambda x: x.replace('+', '') if '+' in str(x) else x)
df['Installs'] = df['Installs'].apply(lambda x: x.replace('Free', "0") if 'Free' in str(x) else x)
df['Installs'] = df['Installs'].apply(lambda x: x.replace(',', '') if ',' in str(x) else x)
df.Installs = pd.to_numeric(df.Installs, errors='coerce')
df['Installs'] = df['Installs'].apply(lambda x: round(np.log(x+10)))
#print(type(df['Installs'].values))


# In[ ]:


scaler = MinMaxScaler()
# MinMaxScaler(copy=True, feature_range=(0, 1))
df["Installs"]=scaler.fit_transform(df["Installs"].values.reshape(-1,1))


# In[ ]:


df["Content Rating"] = df["Content Rating"].apply(lambda x: x.replace("Everyone 10+","Teen") if "Everyone 10+" in str(x) else x)
df["Content Rating"] = df["Content Rating"].apply(lambda x: x.replace("Mature 17+","Teen") if "Mature 17+" in str(x) else x)
df["Content Rating"] = df["Content Rating"].apply(lambda x: x.replace("Unrated","Everyone") if "Unrated" in str(x) else x)


# In[ ]:


df=pd.concat([df,pd.get_dummies(df['Content Rating'],prefix='Content')],axis=1).drop(['Content Rating'],axis=1)
df=pd.concat([df,pd.get_dummies(df['Type'],prefix='Type')],axis=1).drop(['Type'],axis=1)


# In[ ]:


df.drop(columns=["Price"] ,inplace=True, axis=1)


# In[ ]:


try:
    df['Reviews'] = df['Reviews'].apply(lambda x: int(x))
    df['Reviews'] = df['Reviews'].apply(lambda x: round(np.log(x+10)))
except Exception as e:
    a=e


# In[ ]:


df.drop(df.index[df[df["Reviews"]=="3.0M"].index],inplace=True)
scaler = MinMaxScaler()
# MinMaxScaler(copy=True, feature_range=(0, 1))
df["Reviews"]=scaler.fit_transform(df["Reviews"].values.reshape(-1,1))


# In[ ]:


df.drop(df.index[[10472]],inplace=True)


# In[ ]:


df.drop(columns=["Category","Genres"], inplace=True,axis =1)


# In[ ]:


df.dropna(subset=['Rating'],inplace=True)
y = df.Rating

df.drop(columns=["Rating"],axis =1,inplace=True)


# In[ ]:



x = pd.DataFrame(data=df)
x.fillna(method="ffill",inplace=True)
pca = PCA()
pca.fit_transform(x)
x = pd.DataFrame(data=x)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)  


# In[ ]:


rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(X_train, y_train)


# In[ ]:


y_pred=rf.predict(X_test)


# In[ ]:


y_pred


# In[ ]:





# In[ ]:




