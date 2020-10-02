#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### import the dataset

# https://archive.ics.uci.edu/ml/datasets/banknote+authentication

# In[ ]:


banknote = pd.read_csv('../input/bill_authentication.csv')


# ### explore the dataset

# In[ ]:


banknote.shape


# data points are 1372 and 4 features and target class

# In[ ]:


banknote.head(4)


# In[ ]:


# Seaborn visualization library
import seaborn as sns
# Create the default pairplot
sns.pairplot(banknote,hue='Class')


# ### Data Preprocessing
# 

# ###### 1- split the target class

# In[ ]:


x , y = banknote.drop('Class',axis=1) , banknote['Class']


# ###### 2- split the data to train and test sets

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


get_ipython().run_line_magic('pinfo', 'train_test_split')


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.20)


# In[ ]:


print("x train shape ", x_train.shape)
print("x test shape ", x_test.shape)


# ## algorithms training

# ### 1- linear SVM

# In[ ]:


from sklearn.svm import SVC


# In[ ]:


linear_svc_classifier = SVC(kernel="linear")
linear_svc_classifier.fit(x_train,y_train)


# In[ ]:


# prediction
linear_svc_classifier_prediction = linear_svc_classifier.predict(x_test)


# In[ ]:


# evaluation
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[ ]:


print(confusion_matrix(y_test, linear_svc_classifier_prediction))


# In[ ]:


print(classification_report(y_test, linear_svc_classifier_prediction))


# In[ ]:


print("accuracy of linear svm", accuracy_score(y_test, linear_svc_classifier_prediction)*100,"%")


# ### 2- Polynomial Kernel

# In[ ]:


poly_svm_classifier = SVC(kernel="poly",degree=4)
poly_svm_classifier.fit(x_train,y_train)


# In[ ]:


# prediction
poly_svm_predict = poly_svm_classifier.predict(x_test)


# In[ ]:


# evaluation
print(classification_report(poly_svm_predict,y_test))


# In[ ]:


print(confusion_matrix(poly_svm_predict,y_test))


# In[ ]:


print(accuracy_score(poly_svm_predict,y_test)*100)


# ## 2. Gaussian Kernel

# In[ ]:


rbf_svm_classifier = SVC(kernel="rbf")
rbf_svm_classifier.fit(x_train,y_train)


# In[ ]:


# prediction
rbf_svm_predict = rbf_svm_classifier.predict(x_test)


# In[ ]:


# evaluation
print(classification_report(rbf_svm_predict,y_test))


# In[ ]:


print(confusion_matrix(rbf_svm_predict,y_test))


# In[ ]:


print(accuracy_score(rbf_svm_predict,y_test)*100)


# ## 4. Sigmoid Kernel

# In[ ]:


sigmoid_svm_classifier = SVC(kernel="sigmoid")
sigmoid_svm_classifier.fit(x_train,y_train)


# In[ ]:


sigmoid_svm_classifier_prediction = sigmoid_svm_classifier.predict(x_test)


# In[ ]:


print(accuracy_score(sigmoid_svm_classifier_prediction,y_test)*100)


# # Comparison of Kernel Performance
# 

# If we compare the performance of the different types of kernels we can clearly see that the sigmoid kernel performs the worst. 
# 
# Amongst the Gaussian kernel and polynomial kernel, we can see that Gaussian kernel achieved a perfect 100% prediction rate while polynomial kernel misclassified one instance. Therefore the Gaussian kernel performed slightly better.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




