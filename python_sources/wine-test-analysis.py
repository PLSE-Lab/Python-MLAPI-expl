#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing all the necessarry libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#importing the data
data=pd.read_csv('../input/winequality-red.csv')


# In[ ]:


#checking the head,info and description  of our data


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data.describe()


# # Exploratory Data Analysis(EDA)

# In[ ]:


fig,axes=plt.subplots(figsize=(10,6))
sns.distplot(data['fixed acidity'],kde=False,bins=50)


# In[ ]:


#citric acid vs quality
sns.boxplot(x='quality',y='citric acid',data=data)


# In[ ]:


#alcohol  vs quality
sns.boxplot(x='quality',y='alcohol',data=data)


# In[ ]:


#pH  vs quality
sns.boxplot(x='quality',y='pH',data=data)


# In[ ]:


#pairplot 
sns.pairplot(data)


# In[ ]:


#counting the number of varibles in quality column using countplot
sns.countplot(data['quality'])
data['quality'].value_counts()


# In[ ]:


#checking the correlation between different columns
data.corr()


# In[ ]:


#sulphates vs alcohol
sns.jointplot(x='sulphates',y='alcohol',data=data,kind='hex',color='red',size=8)


# In[ ]:


#checking for missing data
fig,axes=plt.subplots(figsize=(10,6))
sns.heatmap(data.isnull(),cmap='viridis',yticklabels=False,cbar=False)


# In[ ]:


#coversion of multivariable column target column into two varible (0 and 1 )


# In[ ]:


#converting the quality column into binary
#i.e bad or good using the pandas cut method
data['quality']=pd.cut(data['quality'],bins=(2,6.5,8),labels=['bad','good'])


# In[ ]:


#converting the categorical feature i.e bad or good into numerical feature 0 and 1
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
new_data=le.fit_transform(data['quality'])


# In[ ]:


#saving the new quality column in our orignal dataframe and checking its head
data['quality']=new_data
data.head()


# Train Test Spilt

# In[ ]:


from sklearn.cross_validation import train_test_split
X=data.drop('quality',axis=1)
y=data['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[ ]:


#training different models and comparing the results


# In[ ]:


# K-nearest Neighbour


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knc=KNeighborsClassifier()


# In[ ]:


knc.fit(X_train,y_train)


# In[ ]:


knc_prediction=knc.predict(X_test)


# In[ ]:


# Decision Tree and Random Forest


# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


dtc=DecisionTreeClassifier()


# In[ ]:


dtc.fit(X_train,y_train)


# In[ ]:


dtc_prediction=dtc.predict(X_test)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rfc=RandomForestClassifier(n_estimators=200)


# In[ ]:


rfc.fit(X_train,y_train)


# In[ ]:


rfc_predictions=rfc.predict(X_test)


# In[ ]:


#Support Vector Machine


# In[ ]:


from sklearn.svm import SVC


# In[ ]:


svc=SVC()


# In[ ]:


svc.fit(X_train,y_train)


# In[ ]:


svc_predictions=svc.predict(X_test)


# In[ ]:


from sklearn.grid_search import GridSearchCV


# In[ ]:


#for getting the best parameters in out model
grid=GridSearchCV(SVC(),param_grid={'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.001,0.0001]},verbose=3)


# In[ ]:


grid.fit(X_train,y_train)


# In[ ]:


#checking the best parameters for our model
grid.best_params_


# In[ ]:


#Using the best parameters in our model
grid.best_estimator_


# In[ ]:


grid_predictions=grid.predict(X_test)


# In[ ]:


#Comparing the Predictions and Accuracy of our models


# In[ ]:


from sklearn.metrics import classification_report,accuracy_score


# In[ ]:


print('\t\t\tK-Nearest Neighbours\n\n',classification_report(y_test,knc_prediction))

print('\n\n\t\t\tDecision Tree\n\n',classification_report(y_test,dtc_prediction))

print('\n\n\t\t\tRandom Forest\n\n',classification_report(y_test,rfc_predictions))

print('\n\n\t\t\tSupport Vector Machine\n\n',classification_report(y_test,svc_predictions))

print('\n\n\t\t\tSVM With GridSearch\n\n',classification_report(y_test,grid_predictions))


# Training Different Model on our dataset we found that the Support Vector Machine with GridSearch was the most accurate in predicting the result

# In[ ]:




