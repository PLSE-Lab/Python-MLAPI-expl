#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


video = pd.read_csv('../input/vgsales.csv')
video.head()


# In[ ]:


video.info()


# In[ ]:


video.isnull().sum()


# In[ ]:


video.Name.unique()


# In[ ]:


print(video[video['Publisher'].isnull()].shape[0])
video[video['Publisher'].isnull()]


# In[ ]:


video[video['Name'].str.contains('Cartoon Network')][['Name','Publisher','Genre','Platform']]


# In[ ]:


video['Publisher'].fillna('Unknown',inplace=True)


# In[ ]:


video.loc[[2236,8368],'Publisher'] = 'Konami Digital Entertainment'


# In[ ]:


video.loc[8896,'Publisher'] = 'Majesco Entertainment'


# In[ ]:


video.loc[6849,'Publisher'] = 'THQ'


# In[ ]:


video.loc[15788,'Publisher'] = 'Activision'


# In[ ]:


video.loc[14698,'Publisher'] = 'Rondomedia'


# In[ ]:


video.loc[12517,'Publisher'] = 'Tecmo Koei'


# In[ ]:


video.loc[8162,'Publisher'] = 'THQ'


# In[ ]:


video.loc[13278,'Publisher'] = 'Capcom'


# In[ ]:


video.loc[12487,'Publisher'] = 'Konami Digital Entertainment'


# In[ ]:


video.loc[15915,'Publisher'] = 'Zoo Games'


# In[ ]:


video.loc[16198,'Publisher'] = 'Namco Bandai Games'


# In[ ]:


video.loc[16229,'Publisher'] = 'Ubisoft'


# In[ ]:


video.loc[6562,'Publisher'] = 'Take-Two Interactive'


# In[ ]:


video.loc[1303,'Publisher'] = 'Electronic Arts'


# In[ ]:


video.loc[[4145,6437],'Publisher'] = 'Sega'


# In[ ]:


video.loc[6272,'Publisher'] = 'Nintendo'


# In[ ]:


video.loc[8503,'Publisher'] = 'Take-Two Interactive'


# In[ ]:


video.loc[16191,'Publisher'] = 'Vivendi Games'


# In[ ]:


video.loc[14942,'Publisher'] = 'Alchemist'


# In[ ]:


video.loc[16494,'Publisher'] = 'D3Publisher'


# In[ ]:


video.loc[7953,'Publisher'] = 'Unknown'


# In[ ]:


video.loc[15261,'Publisher'] = 'Nintendo'


# In[ ]:


video.loc[[3166,3766,7470],'Publisher'] = 'THQ'


# In[ ]:


video.loc[[8330,5302],'Publisher'] = 'Atari'


# In[ ]:


video.loc[470,'Publisher'] = 'THQ'


# In[ ]:


video.loc[8848,'Publisher'] = 'Nintendo'


# In[ ]:


video.loc[16553,'Publisher'] = 'Focus Home Interactive'


# In[ ]:


video.loc[1662,'Publisher'] = 'Activision'


# In[ ]:


video.loc[8341,'Publisher'] = 'Global Star'


# In[ ]:


video.loc[14296,'Publisher'] = 'Namco Bandai Games'


# In[ ]:


video.loc[9749,'Publisher'] = 'Namco Bandai Games'


# In[ ]:


video.loc[16208,'Publisher'] = 'Banpresto'


# In[ ]:


video.loc[10494,'Publisher'] = 'Konami Digital Entertainment'


# In[ ]:


video.loc[10382,'Publisher'] = 'Disney Interactive Studios'


# In[ ]:


video.loc[[15325,7208,6042,3159],'Publisher'] = 'THQ'


# In[ ]:


video.loc[[4526,4635],'Publisher'] = 'THQ'


# In[ ]:


video.loc[9517,'Publisher'] = 'Focus Home Interactive'


# In[ ]:


video.loc[7351,'Publisher'] = 'Atari'


# In[ ]:


video.loc[13962,'Publisher'] = 'Wargaming.net'


# In[ ]:


video.corr()


# In[ ]:


dummydata = pd.get_dummies(video.drop(['Year','Name'],axis=1))
dummydata.head()


# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


scaleddata = pd.DataFrame(StandardScaler().fit_transform(dummydata),columns=dummydata.columns)
scaleddata.head()


# In[ ]:


from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import r2_score,mean_squared_error


# In[ ]:


x = scaleddata.drop('Global_Sales',axis=1)
y = scaleddata['Global_Sales']


# In[ ]:


xtrain,xtest,ytrain,ytest = train_test_split(x,y,random_state=0,test_size=0.30)


# In[ ]:


lr = LinearRegression()


# In[ ]:


lr.fit(xtrain,ytrain)


# In[ ]:


ypred = lr.predict(xtest)


# In[ ]:


r2_score(ytest,ypred)


# In[ ]:


np.sqrt(mean_squared_error(ytest,ypred))


# In[ ]:


r = Ridge(alpha=0.5,random_state=0,solver='lsqr')


# In[ ]:


r.fit(xtrain,ytrain)


# In[ ]:


ypredR = r.predict(xtest)
ypredR


# In[ ]:


r2_score(ytest,ypredR)


# In[ ]:


np.sqrt(mean_squared_error(ytest,ypredR))


# In[ ]:


l = Lasso()


# In[ ]:


l.fit(xtrain,ytrain)
ypredLasso = l.predict(xtest)
ypredLasso


# In[ ]:


r2_score(ytest,ypredLasso)


# In[ ]:


np.sqrt(mean_squared_error(ytest,ypredLasso))


# In[ ]:


enet = ElasticNet()


# In[ ]:


enet.fit(xtrain,ytrain)


# In[ ]:


ypredEnet = enet.predict(xtest)
ypredEnet


# In[ ]:


r2_score(ytest,ypredEnet)


# In[ ]:


np.sqrt(mean_squared_error(ytest,ypredEnet))


# In[ ]:


params = {"alpha":[0.01,0.5,1,2,3,4,0.02,0.03,0.09,10,50],
         "solver":["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
         "random_state":[0,2,1,3,500]}
params_lasso = {"alpha":[0.01,0.5,1,2,3,0.05,0.02,0.03,0.09,0.001,5],
         "random_state":[0,2,1,3,500]}
params_elastic = {"alpha":[0.01,0.5,1,2,3,0.05,0.02,0.03,0.001,5],
         "random_state":[0,2,1,3,500]}


# In[ ]:


grid = GridSearchCV(estimator=r,param_grid=params,cv =3)


# In[ ]:


#grid.fit(xtrain,ytrain)


# In[ ]:


#grid.best_params_


# In[ ]:




