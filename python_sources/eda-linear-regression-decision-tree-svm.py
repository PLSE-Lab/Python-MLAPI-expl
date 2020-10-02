#!/usr/bin/env python
# coding: utf-8

# In[94]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[95]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[96]:


wine=pd.read_csv('../input/winequality-red.csv')


# In[97]:


wine.head()


# In[98]:


wine.info()


# # EDA

# In[99]:


sns.barplot(x = 'quality', y = 'volatile acidity', data = wine)


# In[100]:


sns.barplot(x = 'quality', y = 'fixed acidity', data = wine)


# In[101]:


sns.barplot(x = 'quality', y = 'citric acid', data = wine)


# In[102]:


sns.barplot(x = 'quality', y = 'residual sugar', data = wine)


# In[103]:


sns.barplot(x = 'quality', y = 'chlorides', data = wine)


# In[104]:


sns.barplot(x = 'quality', y = 'free sulfur dioxide', data = wine)


# In[105]:


sns.barplot(x = 'quality', y = 'total sulfur dioxide', data = wine)


# In[106]:


sns.barplot(x = 'quality', y = 'density', data = wine)


# In[107]:


sns.barplot(x = 'quality', y = 'pH', data = wine)


# In[108]:


sns.barplot(x = 'quality', y = 'sulphates', data = wine)


# In[109]:


sns.barplot(x = 'quality', y = 'alcohol', data = wine)


# In[110]:


X = wine.drop('quality', axis = 1)
y = wine['quality']


# In[111]:


from sklearn.model_selection import train_test_split


# In[112]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[113]:


X_train.head()


# In[114]:


X_test.head()


# In[115]:


y_train.head()


# # Linear Regression

# In[116]:


from sklearn.linear_model import LinearRegression


# In[117]:


lm=LinearRegression()


# In[118]:


lm.fit(X_train,y_train)


# In[119]:


cnf=pd.DataFrame(lm.coef_,X_train.columns)
cnf


# In[120]:


predictions=lm.predict(X_test)


# In[121]:


y_test.head()


# In[122]:


predictions


# In[123]:


plt.scatter(y_test,predictions)


# In[124]:


from sklearn import metrics


# In[125]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[126]:


from sklearn.metrics import classification_report,confusion_matrix


# # Decision Tree

# In[127]:


from sklearn.tree import DecisionTreeClassifier


# In[128]:


dtree = DecisionTreeClassifier(max_depth=3)


# In[129]:


dtree.fit(X_train,y_train)


# Predict and Evaluate

# In[130]:


predictions = dtree.predict(X_test)


# In[131]:


print(classification_report(y_test,predictions))


# In[132]:


print(confusion_matrix(y_test,predictions))


# Tree Visualization
# 

# In[133]:


from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydot 

features = list(wine.columns[:-1])
features


# In[134]:


dot_data = StringIO()  
export_graphviz(dtree, out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph[0].create_png(),width=5000000,height=100)  


# 

# # Random Forest

# In[135]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)


# In[136]:


rfc_pred = rfc.predict(X_test)


# In[137]:


print(confusion_matrix(y_test,rfc_pred))


# In[138]:


print(classification_report(y_test,rfc_pred))


# # SVM

# In[139]:


from sklearn.svm import SVC


# In[140]:


model = SVC()


# In[141]:


model.fit(X_train,y_train)


# In[142]:


predictions = model.predict(X_test)


# In[143]:


print(classification_report(y_test,predictions))


# In[144]:


print(confusion_matrix(y_test,predictions))


# **After implementing all, it is safe to say that Random Forest Classifier is best to use for this dataset among all the above methods used.**
