#!/usr/bin/env python
# coding: utf-8

# # Predictions on the Iris Dataset 

# ### Import libraries and data

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import sklearn.datasets


# In[ ]:


from sklearn.datasets import load_iris
iris = load_iris()
iris


# In[ ]:


#Create a concatenated dataframe
df = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])
df


# In[ ]:


X = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
display(X.head())
X.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
display(X.head())


# In[ ]:


target = df['target']
target.head()


# In[ ]:


X.dropna(how='all', inplace=True)
X.head()


# ### Visualise

# In[ ]:


sns.scatterplot(x = X.sepal_length, y = X.sepal_width, style = df.target )


# ### Standardise the Data

# In[ ]:


from sklearn.preprocessing import scale
x_train = scale(X)


# ### Find Eigenvalues and Eigenvectors

# In[ ]:


#Find covariance matrix
covariancematrix = np.cov(x_train.T)
covariancematrix


# In[ ]:


eigenvalues, eigenvectors = np.linalg.eig(covariancematrix)
display(eigenvalues, eigenvectors)


# In[ ]:


#Alternatively, use Singular Value Decomposition (SVD)
eigenvec_svd, s, v = np.linalg.svd(x_train.T)
display(eigenvec_svd)


# ### Principle Components

# In[ ]:


display(eigenvalues)


# In[ ]:


variance_accounted = []
for i in eigenvalues:
    va = (i/(eigenvalues.sum())*100)
    variance_accounted.append(va)
display(variance_accounted)


# In[ ]:


cumulative_variance = np.cumsum(variance_accounted)
cumulative_variance


# In[ ]:


sns.lineplot(x = [1,2,3,4], y = cumulative_variance);
plt.xlabel("No. of components")
plt.ylabel("Cumulative variance")
plt.title("Variance vs No. of components")
plt.show()


# #### Therefore, we select the first two components because they contribute the most to the variance

# In[ ]:


#Project data onto a lower dimensional plane
proj_vector = (eigenvectors.T[:][:])[:2].T
proj_vector


# In[ ]:


x_pca = np.dot(x_train, proj_vector)


# ### Making Predictions using LogisticRegression

# In[ ]:


#split the data set into train and test sets
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x_pca, target, test_size=0.2)


# In[ ]:


xtrain,xtest, ytrain, ytest = train_test_split(x_train, target, test_size=0.2)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


model_pca = LogisticRegression()
model_pca.fit(xTrain, yTrain)


# In[ ]:


y_pred = model_pca.predict(xTest)
y_pred


# In[ ]:


model_original = LogisticRegression()
model_original.fit(xtrain, ytrain)
y_pred_original = model_original.predict(xtest)
y_pred_original


# In[ ]:


from sklearn.metrics import confusion_matrix
cm_pca = confusion_matrix(y_pred, yTest)
cm_pca


# In[ ]:


cm_original = confusion_matrix(y_pred_original, ytest)
cm_original


# In[ ]:


#Confusion Matrix showing percentages
print('Confusion Matrix for values predicted after selecting 2 components using principle component analysis' ,sns.heatmap((cm_pca/np.sum(cm_pca))*100, annot=True, cmap="GnBu"))


# In[ ]:


print('Confusion Matrix for values predicted using all 4 components from original standardised data', sns.heatmap((cm_original/np.sum(cm_original))*100, annot = True, cmap="Blues"))


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print('Classification Report for PCA data')
p = np.asarray(yTest)
p1 = pd.DataFrame(p, columns =['Actual'])
p2 = pd.DataFrame(y_pred, columns = ['Predictions_pca'])
pred = pd.concat([p1,p2], axis = 1)
print(classification_report(pred['Actual'], pred['Predictions_pca']))


# In[ ]:


print('Classification Report for values predicted using all 4 components')
q = np.asarray(ytest)
q1 = pd.DataFrame(q, columns=['Actual'])
q2 = pd.DataFrame(y_pred_original, columns=['Predictions_without_pca'])
pred_2 = pd.concat([q1,q2], axis=1)
print(classification_report(pred_2['Actual'], pred_2['Predictions_without_pca']))


# In[ ]:




