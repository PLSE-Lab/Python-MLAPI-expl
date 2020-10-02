#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir("../input"))


# In[32]:


data = pd.read_csv('../input/data.csv')


# In[33]:


data.head()


# In[34]:


data = data.drop(['Unnamed: 32','id'], axis=1)


# In[35]:


data.head()


# In[36]:


data.info()


# In[37]:


data['diagnosis'].unique()


# In[38]:


data['diagnosis_num'] = np.where(data['diagnosis']=='M',1,0)


# In[39]:


from sklearn import preprocessing
df_scaled = pd.DataFrame(preprocessing.scale(data.drop(['diagnosis','diagnosis_num'],axis=1)), columns=data.drop(['diagnosis','diagnosis_num'],axis=1).columns)
df_scaled['diagnosis_num'] = data['diagnosis_num']
df_scaled.head()


# In[40]:


data.head()


# In[41]:


plt.figure(figsize=(20,20))
sns.heatmap(df_scaled.corr(),annot=True,cmap='YlGnBu',fmt='.2f',linewidths=2)


# In[42]:


data.columns


# In[43]:


plt.figure(figsize=(10,10))
sns.countplot(data=data,x='diagnosis')


# In[ ]:





# In[44]:


from sklearn.model_selection import train_test_split

X = df_scaled.drop('diagnosis_num',axis=1)
y = df_scaled['diagnosis_num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[45]:


from sklearn.linear_model import LogisticRegression


# In[46]:


logmodel = LogisticRegression()


# In[47]:


logmodel.fit(X_train,y_train)


# In[48]:


pred = logmodel.predict(X_test)


# In[49]:


from sklearn.metrics import classification_report,confusion_matrix


# In[50]:


print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))


# In[51]:


#lets try KNN!

from sklearn.neighbors import KNeighborsClassifier


# In[52]:


error_rate = []

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[53]:


plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[54]:


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
knn_pred = knn.predict(X_test)


# In[56]:


print(classification_report(y_test,knn_pred))
print(confusion_matrix(y_test,knn_pred))


# In[57]:


#Lets try Random Forest Classifier


# In[58]:


from sklearn.ensemble import RandomForestClassifier


# In[60]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train,y_train)
random_forest_pred = random_forest.predict(X_test)


# In[61]:


print(classification_report(y_test,random_forest_pred))
print(confusion_matrix(y_test,random_forest_pred))


# In[62]:


#Lets try to improve our forest!


# In[63]:


from sklearn.model_selection import GridSearchCV


# In[64]:


RANDOM_SEED = 31416


# In[65]:


random_forest_parameters = {
    'n_jobs': [-1],
    'random_state': [RANDOM_SEED],
    'n_estimators': [10, 50, 100, 150, 200],
    'max_depth': [4, 8, 12, 16],
    'min_samples_split': [3, 5, 7, 12, 16],
    'min_samples_leaf': [1, 3, 5, 7],
}


# In[66]:


random_forest_cv = GridSearchCV(estimator=RandomForestClassifier(),
                         param_grid=random_forest_parameters,
                         cv=10,
                         verbose=0,
                         n_jobs=-1)


# In[67]:


random_forest_cv.fit(X_train,y_train)


# In[68]:


print('Best cross validation score: {}'.format(random_forest_cv.best_score_))
print('Optimal parameters: {}'.format(random_forest_cv.best_params_))


# In[69]:


random_forest_2 = RandomForestClassifier(**random_forest_cv.best_params_)
random_forest_2.fit(X_train,y_train)
random_forest_2_pred = random_forest_2.predict(X_test)


# In[70]:


print(classification_report(y_test,random_forest_2_pred))
print(confusion_matrix(y_test,random_forest_2_pred))


# In[ ]:




