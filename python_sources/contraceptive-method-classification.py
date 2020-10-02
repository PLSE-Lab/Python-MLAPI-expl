#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data=pd.read_csv('../input/cmc.data.txt',names=['Wife Age','Wife Education','Husband Education','Children',
                                                'Wife religion','Wife working','Husband Occupation','SOLI',
                                                'Media Exposure','Contraceptive Method'])
data.head()


# In[ ]:


sns.heatmap(data.isnull(),yticklabels=False,cmap='viridis',cbar=False)


# In[ ]:


data.info()


# In[ ]:


data.describe()


# In[ ]:


sns.set_style('whitegrid')
plt.figure(figsize=(10,6))
sns.distplot(data['Wife Age'],bins=50)


# In[ ]:


sns.jointplot(x='Wife Age',y='Children',data=data)


# In[ ]:


plt.figure(figsize=(10,6))
sns.distplot(data['Children'],bins=30,kde=False,color="red",hist_kws={'edgecolor':'red'})


# In[ ]:


sns.countplot(x='Contraceptive Method',data=data,hue="Wife religion")


# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(x='Contraceptive Method',data=data,hue="SOLI")


# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(x='SOLI',data=data)


# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(x='Contraceptive Method',data=data,hue="Wife working")


# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(x='Contraceptive Method',data=data,hue="Media Exposure")


# In[ ]:


sns.pairplot(data)


# In[ ]:


plt.figure(figsize=(14,6))
sns.countplot(x='Wife Age',data=data,hue="Contraceptive Method")


# In[ ]:


plt.figure(figsize=(14,6))
sns.boxplot(x='Contraceptive Method',y='Wife Age',data=data)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# In[ ]:


X=data.drop('Contraceptive Method',axis=1)
y=data['Contraceptive Method']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


# In[ ]:


rfc=RandomForestClassifier(n_estimators=10)


# In[ ]:


rfc.fit(X_train,y_train)


# In[ ]:


predictions=rfc.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(y_test,predictions))


# In[ ]:


from sklearn.svm import SVC


# In[ ]:


svc = SVC()


# In[ ]:


svc.fit(X_train,y_train)


# In[ ]:


pred=svc.predict(X_test)


# In[ ]:


print(classification_report(y_test,pred))


# In[ ]:


param_grid = {'C': [0.1,1,2,3, 10,20,30,40,50,60,70,80,90, 100,200,300, 1000], 'gamma': [1,0.1,0.01,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.0001], 'kernel': ['rbf']} 


# In[ ]:


from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)


# In[ ]:


grid.fit(X_train,y_train)


# In[ ]:


grid.best_params_


# In[ ]:


grid_predictions = grid.predict(X_test)


# In[ ]:


print(classification_report(y_test,grid_predictions))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




