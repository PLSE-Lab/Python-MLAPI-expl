#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#from google.colab import drive

#drive.mount('/content/drive') 


# In[ ]:


#ls


# In[ ]:


#cd drive/


# In[ ]:


#cd My Drive


# In[ ]:


#cd Data mining


# In[ ]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
#import seaborn as sns

#import sklearn.preprocessing as sk
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm


# In[ ]:


df=pd.read_csv('../input/dataset.csv')


# In[ ]:


data=df


# In[ ]:


data.head()


# In[ ]:





# In[ ]:


data.corr()


# In[ ]:


data = data.drop(['Class'], 1)
data.info()


# In[ ]:


data = data.replace({'?': np.nan})


# In[ ]:


data.mode()


# In[ ]:


data.fillna(data.mode().iloc[0], inplace=True)


# In[ ]:





# In[ ]:


data = data.drop(columns=['id']) # Drop Total from domain knowledge


# In[ ]:


data.columns = data.columns.str.replace(' ','')


# In[ ]:



data.columns = data.columns.str.replace('#','')


# In[ ]:


data['Age']=data.Age.astype(int)


# In[ ]:


data['Plotsize']=data.Plotsize.astype(str).str.upper()


# In[ ]:


data.info()


# In[ ]:


xtrain=data


# In[ ]:


xtrain.iloc[0]["YearlyPeriod"]


# In[ ]:


#for i in range(0,len(xtrain)):
#  xtrain.iloc[i]["YearlyPeriod"]=float(xtrain.iloc[i]["YearlyPeriod"])
#  xtrain.iloc[i]["InstallmentCredit"]=float(xtrain.iloc[i]["InstallmentCredit"])
  


# In[ ]:


xtrain[['InstallmentCredit']]=xtrain[['InstallmentCredit']].astype(float,errors='ignore')


# In[ ]:


xtrain[['YearlyPeriod']]=xtrain[['YearlyPeriod']].astype(float,errors='ignore')


# In[ ]:


xtrain.fillna(xtrain.mode().iloc[0], inplace=True)


# In[ ]:


xtrain.info()


# In[ ]:


xtrain.corr()


# In[ ]:


type(xtrain.iloc[0]["YearlyPeriod"])


# In[ ]:


import math


# In[ ]:


#for i in range(0,len(xtrain)):
#  xtrain.iloc[i]["YearlyPeriod"]=(xtrain.iloc[i]["YearlyPeriod"])
#  xtrain.iloc[i]["InstallmentCredit"]=math.log(xtrain.iloc[i]["InstallmentCredit"])
#xtrain['YearlyPeriod']= math.log (xtrain['YearlyPeriod'])


# 

# In[ ]:


valcol=[]
for col in xtrain:
    if(type(xtrain[col][0])==str):
       valcol.append(col)


# In[ ]:


valcol


# In[ ]:


xtrain['MonthlyPeriod']=xtrain.MonthlyPeriod.astype(int)


# In[ ]:


xtrain['TenancyPeriod']=xtrain.TenancyPeriod.astype(int)


# In[ ]:


xtrain.corr()


# In[ ]:


xtrain=xtrain.drop(columns=["Phone",'MonthlyPeriod'])


# In[ ]:


xtrain = pd.get_dummies(xtrain, columns=["Account1","History","Motive","Account2","EmploymentPeriod","Gender&Type","Sponsors","Plotsize","Plan","Housing","Post"])
xtrain.info()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as acc
from mlxtend.feature_selection import SequentialFeatureSelector as sfs


# In[ ]:


#clf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
'''
# Build step forward feature selection
sfs1 = sfs(clf,
           k_features=15,
           forward=True,
           floating=False,
           verbose=2,
           scoring='accuracy',
           cv=5)

# Perform SFFS
sfs1 = sfs1.fit(xtrain[0:175], y[0:175])
'''


# In[ ]:


#feat_cols = list(sfs1.k_feature_idx_)


# In[ ]:


#feat_cols


# In[ ]:


#xtest=xtrain[xtrain.columns[feat_cols]]


# In[ ]:


import seaborn as sns
f, ax = plt.subplots(figsize=(10, 8))
corr = xtrain.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax, annot = True);


# In[ ]:





# In[ ]:


import seaborn as sns
f, ax = plt.subplots(figsize=(10, 8))
corr = data.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax, annot = True);


# In[ ]:


from sklearn import preprocessing
#Performing Min_Max Normalization
min_max_scaler = preprocessing.StandardScaler()
min_max_scaler = preprocessing.MinMaxScaler()
xtrain = min_max_scaler.fit_transform(xtrain)
min_max_scaler = preprocessing.StandardScaler()
np_scaled = min_max_scaler.fit_transform(xtrain)
dataN1 = pd.DataFrame(np_scaled)
dataN1.head()


# In[ ]:


from sklearn.decomposition import PCA
pca1 = PCA(n_components=4)
pca1.fit(dataN1)
T1 = pca1.transform(dataN1)


# In[ ]:


from sklearn.cluster import KMeans


# In[ ]:


from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
preds1=[]
kmean = KMeans(n_clusters = 6, random_state =42)
kmean.fit(T1)
pred = kmean.predict(T1)
preds1.append(pred)
centroids = kmean.cluster_centers_


# In[ ]:


unique, counts=np.unique(preds1[0], return_counts=True)
counts 


# In[ ]:


#import data from csv to dataframe df
df2=pd.read_csv('../input/dataset.csv')#dataframe object


# In[ ]:


y=df2['Class']


# In[ ]:


y_pred1=[]
for i in range(len(preds1[0])):
    if(preds1[0][i]==0):
      y_pred1.append(1)
    elif (preds1[0][i]==1):
      y_pred1.append(1)
    elif (preds1[0][i]==2):
      y_pred1.append(2)
    elif (preds1[0][i]==3):
      y_pred1.append(0)
    elif (preds1[0][i]==4):
      y_pred1.append(0)
    elif (preds1[0][i]==5):
      y_pred1.append(2)
    #y_pred1.append(preds1[0][i])
unique, counts=np.unique(y_pred1, return_counts=True)
counts


# In[ ]:


get_ipython().system('pip install pandas_ml')


# In[ ]:


from pandas_ml import ConfusionMatrix


# In[ ]:


cm = ConfusionMatrix(y[0:175],y_pred1[0:175])


# In[ ]:


cm


# In[ ]:


accuracy_score(y[0:175],y_pred1[0:175])


# In[ ]:


'''
y_pred1=[]
for i in range(len(preds1[0])):
    if(preds1[0][i]==1):
      y_pred1.append(0)
    elif (preds1[0][i]==0):
      y_pred1.append(1)
    elif (preds1[0][i]==2):
      y_pred1.append(2)
    #y_pred1.append(preds1[0][i])
unique, counts=np.unique(y_pred1, return_counts=True)
counts
'''


# In[ ]:


accuracy_score(y[0:175],y_pred1[0:175])


# In[ ]:


'''
from sklearn.cluster import AgglomerativeClustering
preds2=[]
cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')  
cluster.fit_predict(dataN1)
preds2.append(pred)
'''


# In[ ]:


'''
y_pred2=[]
for i in range(len(preds2[0])):
    y_pred2.append(preds2[0][i])
unique, counts=np.unique(y_pred2, return_counts=True)
counts
'''


# In[ ]:


get_ipython().system('pip install pandas_ml')


# In[ ]:


from pandas_ml import ConfusionMatrix


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


cm = ConfusionMatrix(y[0:175],y_pred1[0:175])


# In[ ]:


cm


# In[ ]:


X11=df2[['id']].iloc[175:1031]


# In[ ]:


X11['Class']=y_pred1[175:1031]


# In[ ]:


X11.head()


# In[ ]:


X11.to_csv("ans.csv",index=False)


# In[ ]:


from IPython.display import HTML
import pandas as pd
import numpy as np
import base64

# function that takes in a dataframe and creates a text link to  
# download it (will only work for files < 2MB or so)
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
    csv = X11.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)


# create a link to download the dataframe
create_download_link(X11)

