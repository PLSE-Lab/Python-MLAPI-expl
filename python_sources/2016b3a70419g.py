#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import AgglomerativeClustering as AC
from sklearn.manifold import TSNE
import os
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


PATH = "/kaggle/input/dmassign1/"


# In[ ]:


data = pd.read_csv(os.path.join(PATH,"data.csv"),index_col=0,na_values='?',usecols = range(199))
y = data['Class']
data.drop(['Class'],inplace=True,axis=1)


# In[ ]:


cols = data.isnull().sum().sort_values(ascending=False)
cols = cols[cols > 0]


# In[ ]:


for column in cols.index:
    if(isinstance(data[column][0],str)):
        data[column].fillna(data[column].value_counts().index[0],inplace=True)
    elif(isinstance(data[column][0],float)):
        data[column].fillna(data[column].median(),inplace=True)# Have to discuss between mean or median


# In[ ]:


def change_values(size):
    if size=='XL':
        return 3
    elif size=='LA' or size=='la':
        return 2
    elif size=='me' or size=='ME' or size=='M.E.':
        return 1
    elif size=='SM' or size=='sm':
        return 0


# In[ ]:


data.loc[:,'Col189'].replace({'yes':1,'no':0},inplace=True)
data.loc[:,'Col190'].replace({p : int(p[4]) for p in data.loc[:,'Col190'].unique()},inplace=True)
data.loc[:,'Col191'].replace({p : int(p[4]) for p in data.loc[:,'Col191'].unique()},inplace=True)
data.loc[:,'Col192'].replace({p : int(p[1]) for p in data.loc[:,'Col192'].unique()},inplace=True)
data.loc[:,'Col193'].replace({'F1':1,'F0':0,'M1':3,'M0':2},inplace=True)
data.loc[:,'Col194'].replace({p : ord(p[1])-ord('b') for p in data.loc[:,'Col194'].unique()},inplace=True)
data.loc[:,'Col195'].replace({p : int(p[2]) for p in data.loc[:,'Col195'].unique()},inplace=True)
data.loc[:,'Col196'].replace({p : int(p[1]) for p in data.loc[:,'Col196'].unique()},inplace=True)
data.loc[:,'Col197'] = data.loc[:,'Col197'].apply(change_values)


# In[ ]:


data = pd.get_dummies(data,columns = data.columns[189:].tolist(),prefix = data.columns[189:].tolist())
new_data = data.iloc[:,:188]


# In[ ]:


data_nodup = data.drop_duplicates()# Drops only based on new columns
new_data_nodup = new_data.reindex(data_nodup.index)


# In[ ]:


scaler = StandardScaler()
scaled_data=scaler.fit_transform(new_data_nodup)
scaled_data=pd.DataFrame(scaled_data,columns=new_data_nodup.columns,index = new_data_nodup.index)
new_data = pd.DataFrame(scaler.transform(new_data),columns=new_data.columns,index = new_data.index)


# In[ ]:


corr = abs(pd.concat([y[:1300],data[:1300]],axis=1).corr())
corr['Class'].sort_values(ascending=False).tail(10)


# In[ ]:


model=TSNE(n_iter=1000,n_components=2,perplexity=100,verbose=2,n_jobs=-1)
model_data=model.fit_transform(new_data)


# In[ ]:


X = pd.DataFrame(model_data,index=new_data.index)

acc_list = []

for i in range(5,70):
    kmean = KMeans(n_clusters = i, random_state = 69)
    kmean.fit(X)
    pred = kmean.predict(X)

    predictions = pd.Series(pred+1,index=data.index,dtype = np.float64)
    classes = (confusion_matrix(y[:1300],predictions[:1300]).argmax(axis=0)+1).astype(np.float64)
    predictions.replace({cluster+1:classes[cluster] for cluster in range(0,len(classes))},inplace=True)

    acc = ((predictions[:1300] != y[:1300])).sum()
    acc_list.append(1 - acc/1300)


# In[ ]:


plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.plot(range(5,70),acc_list)
plt.title("K Means : Hyperparameter Search")
plt.xlabel("Number of Clusters")
plt.ylabel("Accuracy")


# In[ ]:


kmean = KMeans(n_clusters = np.argmax(acc_list)+5, random_state = 69)
kmean.fit(X)
pred = kmean.predict(X)


# In[ ]:


predictions = pd.Series(pred+1,index=data.index,dtype = np.float64)
classes = (confusion_matrix(y[:1300],predictions[:1300]).argmax(axis=0)+1).astype(np.float64)
predictions.replace({cluster+1:classes[cluster] for cluster in range(0,len(classes))},inplace=True)


# In[ ]:


acc = ((predictions[:1300] != y[:1300])).sum()
print(acc)


# In[ ]:


plt.figure(figsize=(8, 8))

plt.scatter(model_data[:, 0], model_data[:, 1], c=predictions)
plt.title("Final Plot")
plt.xlabel("Dimension 1 : tSNE")
plt.ylabel("Dimension 2 : tSNE")


# In[ ]:


predictions = (predictions[1300:].astype(int)).reset_index()
from IPython.display import HTML 
import pandas as pd
import numpy as np
import base64
def create_download_link(df, title = "Download CSV file", filename = "data.csv"): 
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)


# In[ ]:


create_download_link(predictions)


# In[ ]:




