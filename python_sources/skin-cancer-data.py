#!/usr/bin/env python
# coding: utf-8

# In[4]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[98]:


data = pd.read_csv('../input/HAM10000_metadata.csv')


# In[6]:


data.head()


# In[ ]:


data.tail()


# In[ ]:


data.describe()


# In[ ]:


data.sex = [1 if each == 'female'else 0 for each in data.sex]


# In[ ]:


data.tail()


# In[7]:


image = pd.read_csv('../input/hmnist_28_28_L.csv')


# In[9]:


image.head()


# In[100]:


data.dx = [1 if each == 'bkl' or each == 'nv' or each == 'df' else 0 for each in data.dx]


# In[101]:


data.dx


# In[12]:


data.head()


# In[13]:


#data['localization'] = pd.Categorical(data['localization'])


# In[14]:


#new_categ_df = pd.get_dummies(data['localization'],prefix ='local')


# In[15]:


#new_data_frame = pd.concat([data,new_categ_df],axis=1)


# In[16]:


#new_data_frame.head()


# In[17]:


#new_data_frame['dx_type'] = pd.Categorical(new_data_frame['dx_type'])


# In[18]:


#new_categ_df = pd.get_dummies(new_data_frame['dx_type'],prefix ='dx_type')


# In[19]:


#new_data =  pd.concat([new_data_frame,new_categ_df],axis=1)


# In[20]:


#new_data.head()


# In[21]:


#data = new_data.drop(columns=['image_id', 'lesion_id','dx_type','localization'],axis=1)


# In[22]:


data.head()


# In[23]:


sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[24]:


data['age'].dropna()


# In[25]:


data = data.dropna()


# In[26]:


sns.set_style('whitegrid')


# In[102]:


sns.countplot(x='dx',data=data)


# In[ ]:


data.head()


# In[ ]:


sns.countplot(x='dx',hue='age',data=data)
plt.legend(loc='best')


# In[ ]:


sns.distplot(data['age'].dropna(),kde=False)


# In[ ]:


data['age'].plot.hist(bins=35)


# In[ ]:


import cufflinks as cf


# In[ ]:


cf.go_offline()


# In[ ]:


#data['age'].iplot(kind='hist',bins=10)


# In[ ]:


sns.boxplot(x='dx_type', y='age', data=data)


# In[ ]:


data['localization'] = pd.Categorical(data['localization'])


# In[ ]:


new_categ_df = pd.get_dummies(data['localization'],prefix ='local')


# In[ ]:


new_data_frame = pd.concat([data,new_categ_df],axis=1)


# In[ ]:


new_data_frame.head()


# In[ ]:


data = new_data_frame


# In[ ]:


data.head()


# In[ ]:


new_data_frame['dx_type'] = pd.Categorical(new_data_frame['dx_type'])


# In[ ]:


new_categ_df = pd.get_dummies(new_data_frame['dx_type'],prefix ='dx_type')


# In[ ]:


new_data =  pd.concat([new_data_frame,new_categ_df],axis=1)


# In[ ]:


new_data.head()


# In[ ]:


new_data_frame = pd.concat([data,new_categ_df],axis=1)


# In[ ]:


data = new_data_frame


# In[ ]:


data.head()


# In[ ]:


data = data.drop(['localization','lesion_id','image_id','dx_type'],axis=1)


# In[ ]:


data.head()


# In[ ]:


x = data.drop('dx',axis=1)
y = data['dx']


# In[ ]:


#Logistic Regression
from sklearn.model_selection import train_test_split


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=101)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel = LogisticRegression()


# In[ ]:


logmodel.fit(x_train,y_train)


# In[ ]:


predictions = logmodel.predict(x_test)


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(y_test,predictions))


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


confusion_matrix(y_test,predictions)


# In[ ]:


sns.countplot(x='dx_type_histo',hue='age',data=data)
plt.legend(loc='best')


# In[ ]:


#KNN
from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler = StandardScaler()


# In[ ]:


scaler.fit(data.drop('dx',axis=1))


# In[ ]:


scaled_features = scaler.transform(data.drop('dx',axis=1))


# In[ ]:


scaled_features


# In[ ]:


df_feat = pd.DataFrame(scaled_features,columns=data.columns[:-1])


# In[ ]:


df_feat.head()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x = df_feat
y = data['dx']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=101)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=3)


# In[ ]:


knn.fit(x_train,y_train)


# In[ ]:


pred = knn.predict(x_test)


# In[ ]:


pred


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


print(confusion_matrix(y_test,pred))


# In[ ]:


print(classification_report(y_test,pred))


# In[ ]:


error_rate = []
for i in range(1,50):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    pred_i = knn.predict(x_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[ ]:


plt.figure(figsize=(10,6))
plt.plot(range(1,50),error_rate,color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)
plt.title('error rate vs K value')
plt.xlabel('k')
plt.ylabel('error rate')


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
pred = knn.predict(x_test)
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# In[ ]:


from sklearn.svm import SVC


# In[ ]:


model = SVC()


# In[ ]:


model.fit(x_train,y_train)


# In[ ]:


predictions = model.predict(x_test)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))


# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


param_grid = {'C': [0.1,1,10,100,1000], 'gamma':[1,0.1,0.01,0.001,0.0001]}


# In[ ]:


grid = GridSearchCV(SVC(),param_grid,verbose=3)


# In[ ]:


grid.fit(x_train,y_train)


# In[ ]:


grid.best_params_


# In[ ]:


grid.best_estimator_


# In[ ]:


grid_predictions = grid.predict(x_test)


# In[ ]:


grid_predictions


# In[ ]:


print(confusion_matrix(y_test,grid_predictions))
print('\n')
print(classification_report(y_test,grid_predictions))


# In[ ]:


data.head()


# In[ ]:


sns.scatterplot(x='dx',y='age',data=data)
plt.plot()


# In[ ]:


sns.pairplot(data,hue='dx')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




