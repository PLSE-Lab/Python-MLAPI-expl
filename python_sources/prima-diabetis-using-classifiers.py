#!/usr/bin/env python
# coding: utf-8

# In[ ]:


############## DIABETIES DATASET #################


# In[ ]:


#####importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

import statsmodels.formula.api as smf
from scipy.stats import shapiro,levene


# ## DIABETIES DATASET

# In[ ]:


########## importing the data ########################
data=pd.read_csv('../input/diabetes.csv')
#checking the head of the data
data.head()


# In[ ]:


#describing the data
data.describe()


# In[ ]:


#getting the information regarding the data
data.info()


# In[ ]:


#checking the shape of the data
data.shape


# ## EDA

# In[ ]:


#checking for null values in the data and performing EDA
data.isnull().sum()


# In[ ]:


#getting the individua count of the outcome yes or no in the dataset
data['Outcome'].value_counts()


# In[ ]:


#dropping the outcome in the x and considering it in y as y is the target variable
x=data.drop('Outcome',axis=1)
x.head()
y=data['Outcome']
y.head()


# ## PLOTS

# In[ ]:


#### plotting a HISTOGRAM on the data
data.hist(figsize=(10,8))
plt.show()


# In[ ]:


#### BOXPLOT for checking the outliers
data.plot(kind= 'box' , subplots=True,layout=(3,3), sharex=False, sharey=False, figsize=(10,8))


# In[ ]:


#### checking the correlation in matrix for variables using HEATMAP
import seaborn as sns
sns.heatmap(data.corr(), annot = True)


# ## SPLITTING THE DATA

# In[ ]:


X=data.iloc[:,:-1]
X.head()
Y=data.iloc[:,-1]
Y.head()


# In[ ]:


#### splitting X and y into training and testing sets 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=100)


# ## DATA SCALING

# In[ ]:


# Scaling the data
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ## BUILDING A LOGISTIC REGRESSION  MODEL

# In[ ]:


#logistic regression model
model=LogisticRegression()
model.fit(X_train,y_train)
ypred=model.predict(X_test)
ypred


# ## CHECKING FOR THE ACCURACY WITH THE METRICS

# In[ ]:


# accuracy score
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,ypred)
print(accuracy)


# In[ ]:


#confusion matrix
cm=metrics.confusion_matrix(y_test,ypred)
print(cm)
plt.imshow(cm, cmap='binary')


# In[ ]:


#sensitivity and specificity check
tpr=cm[1,1]/cm[1,:].sum()
print(tpr*100)
tnr=cm[0,0]/cm[0,:].sum()
print(tnr*100)


# In[ ]:


#checking roc and auc curves
from sklearn.metrics import roc_curve,auc
fpr,tpr,_=roc_curve(y_test,ypred)
roc_auc=auc(fpr,tpr)
print(roc_auc)
plt.figure()
plt.plot(fpr,tpr)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.show()


# ## BUILDING A DECISION TREE CLASSIFIER

# In[ ]:


#### importing the classifier and building the model
from sklearn import tree

model = tree.DecisionTreeClassifier()


# In[ ]:


model.fit(X_train, y_train)


# In[ ]:


y_predict = model.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_predict)


# In[ ]:




