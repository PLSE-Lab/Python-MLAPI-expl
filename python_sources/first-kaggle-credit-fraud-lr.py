#!/usr/bin/env python
# coding: utf-8

# # This Model Uses Logistic regression for Prdeicting a credit card Fraud

# IMPORTING NECESSARY LIBRARIES FOR DATA MANIPULATION

# In[ ]:


import numpy as np
import pandas as pd


# Reading The downloaded CSV File

# In[ ]:


data=pd.read_csv("../input/creditcard.csv")


# In[ ]:


data.head()


# ALL the Dependent Variables and the independent class variable are numbers 

# In[ ]:


data.describe()


# Checking for Size of Data Set and Null values in Dataset

# In[ ]:


data.Class.value_counts()


# In[ ]:


data.isna().sum()


# Visualizing the dependency of the Class variable using all the inependednt variables.For simplicity as the data load takes so much time taken the sample size of first ten thousand records

# In[ ]:


from matplotlib import pyplot as plt


# In[ ]:


import seaborn as sns


# In[ ]:


sns.stripplot(data.Class[0:10000],data.Time[0:10000])
    

    


# In[ ]:


data1=data[:10000]
data1.head()


# In[ ]:


for i in data1.columns:
    print(i)
    if i!='Class':
        sns.stripplot(data1.Class,data1[i])
        plt.show()
#sns.stripplot(data.Class[0:10000],data.Time[0:10000])


# From the plot its clear that the variables that affect the Class are V17,V16,V14,V12,V11,V10,V9,V4,V3

# In[ ]:


####PREAPARING THE TRAINING DATA USING THE 10000 SAMPLES


# In[ ]:


data_train_x=data1[['V17','V16','V14','V12','V11','V10','V9','V4','V3']].copy()


# In[ ]:


data_train_x.shape


# In[ ]:


data_train_y=data1['Class']
data_train_y.head()


# # Implementing the Logistic Regression algorithm using 10000 samples by dividing it into test data and train data 

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(data_train_x, data_train_y, test_size=0.3, random_state=0)


# In[ ]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[ ]:


y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# For accuracy Matrix the confusion matrix is plotted with this data

# In[ ]:


cnf_matrix = metrics.confusion_matrix(y_test, y_pred)


# In[ ]:


cnf_matrix


# Now running the fitted model to the entire dataset ie from 10000 to the end

# In[ ]:


y_tes=data.Class[10000:]


# In[ ]:


x_tes=data[['V17','V16','V14','V12','V11','V10','V9','V4','V3']].copy()
x_tes=x_tes[10000:]


# In[ ]:


y_pre=logreg.predict(x_tes)


# In[ ]:


cnf_mat = metrics.confusion_matrix(y_tes, y_pre)


# In[ ]:


cnf_mat


# In[ ]:


logreg.score(x_tes, y_tes)


# In[ ]:


from sklearn.metrics import recall_score
print(recall_score(y_tes, y_pre, average='macro'))
print(recall_score(y_tes, y_pre, average='micro'))
print(recall_score(y_tes, y_pre, average='weighted'))
print(recall_score(y_tes, y_pre, average=None))


# In[ ]:


from sklearn.metrics import average_precision_score


# In[ ]:


average_precision = average_precision_score(y_tes, y_pre)


# In[ ]:


print(average_precision)


# In[ ]:




