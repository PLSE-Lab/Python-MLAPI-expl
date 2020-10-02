#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


data=pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/train.csv")
tester=pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/test.csv")


# In[ ]:


tester.isnull().any()


# In[ ]:


data.drop('id',axis=1,inplace=True)


# In[ ]:


data.head()


# In[ ]:


data.fillna(method='ffill',inplace=True)
tester.fillna(method='ffill',inplace =True)


# In[ ]:


data.head()


# In[ ]:


y=data['rating']
x=data.drop(['rating'],axis=1)


# In[ ]:


for i in range(len(x)):
    if(x.loc[i,'type']=='new'):
        x.loc[i,'type']=int(1)
    else:
        x.loc[i,'type']=int(0)


# In[ ]:


for i in range(len(tester)):
    if(tester.loc[i,'type']=='new'):
        tester.loc[i,'type']=int(1)
    else:
        tester.loc[i,'type']=int(0)


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_val = train_test_split(x,y,test_size = 0.2,random_state = 0) 


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1,1))
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_train = pd.DataFrame(x_train,columns=['feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9','type','feature10','feature11'])


# In[ ]:


from sklearn import svm
#clf = svm.SVC(gamma='scale')
#clf.fit(x_train,y_train)
from sklearn.model_selection import GridSearchCV 
  
# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001], 
              'kernel': ['rbf',]}  
  
grid = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 3) 
  
# fitting the model for grid search 
grid.fit(x_train, y_train) 


# In[ ]:


tester.drop('id',axis=1,inplace=True)


# In[ ]:


tester = scaler.transform(tester)
tester = pd.DataFrame(tester,columns=['feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9','type','feature10','feature11'])


# In[ ]:


y_pred=grid.predict(tester)


# In[ ]:


idq=pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/test.csv")
idw=idq[['id']]
data_sub=pd.DataFrame(y_pred)
data_fin=idw.join(data_sub[0],how='left')
data_fin.rename(columns= {0:'rating'},inplace=True)

import math
for i in range(len(data_fin)):
    data_fin.loc[i,'rating']=int(math.floor(data_fin.loc[i,'rating']))
data_fin = data_fin.astype(int)
data_fin.head()


# In[ ]:


data_fin.to_csv('submission.csv',columns=['id','rating'],index=False)


# In[ ]:





# **
# 2nd Submission Starting after this
# **

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


data=pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/train.csv")
tester=pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/test.csv")


# In[ ]:


tester.isnull().any()


# In[ ]:


data.drop('id',axis=1,inplace=True)


# In[ ]:


data.head()


# In[ ]:


data.fillna(method='ffill',inplace=True)
tester.fillna(method='ffill',inplace =True)


# In[ ]:


data.head()


# In[ ]:


y=data['rating']
x=data.drop(['rating'],axis=1)


# In[ ]:


for i in range(len(x)):
    if(x.loc[i,'type']=='new'):
        x.loc[i,'type']=int(1)
    else:
        x.loc[i,'type']=int(0)


# In[ ]:


for i in range(len(tester)):
    if(tester.loc[i,'type']=='new'):
        tester.loc[i,'type']=int(1)
    else:
        tester.loc[i,'type']=int(0)


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_val = train_test_split(x,y,test_size = 0.2,random_state = 0) 


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_train = pd.DataFrame(x_train,columns=['feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9','type','feature10','feature11'])


# In[ ]:


from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(1000,20,100), random_state=1)
clf.fit(x_train, y_train) 


# In[ ]:


tester.drop('id',axis=1,inplace=True)


# In[ ]:


tester = scaler.transform(tester)
tester = pd.DataFrame(tester,columns=['feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9','type','feature10','feature11'])


# In[ ]:


y_pred=clf.predict(tester)


# In[ ]:


idq=pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/test.csv")
idw=idq[['id']]
data_sub=pd.DataFrame(y_pred)
data_fin=idw.join(data_sub[0],how='left')
data_fin.rename(columns= {0:'rating'},inplace=True)
data_fin.head()


# In[ ]:


data_fin.to_csv('submission.csv',columns=['id','rating'],index=False)


# In[ ]:


data_fin.head()


# In[ ]:




