#!/usr/bin/env python
# coding: utf-8

# ### Data preparation 

# [](http://)$y=1$ means malign tumor and $y=0$ means benign tumor.

# In[ ]:


import pandas as pd
import numpy as np

data = pd.read_csv('../input/data.csv').iloc[:, 1:32]
data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})
data.head(n=10)


# We split the data set in training, validation and test with the proportion (60/20/20)<p>
# 

# In[ ]:


train, validate, test = np.split(data.sample(frac=1, random_state=42),
                                 [int(.6*len(data)), int(.8*len(data))])


# ## Logistic Regression Models

# We apply different logistic regression models depending  on the  regularization parameter C <p>
# ![](http://)We calculate the model accuracy with every data subset and and recall/precision only with the test set.

# In[ ]:


from sklearn import linear_model, metrics

list_C = np.arange(100 , 1000, 1)
score_train = np.zeros(len(list_C))
score_val = np.zeros(len(list_C))
precision_val= np.zeros(len(list_C))
score_test = np.zeros(len(list_C))
recall_test = np.zeros(len(list_C))
precision_test= np.zeros(len(list_C))
count = 0
for C in list_C:
    reg = linear_model.LogisticRegression(C=C)
    reg.fit(train.iloc[:,2:32], train['diagnosis'])
    score_train[count]= metrics.accuracy_score(
        train['diagnosis'], reg.predict(train.iloc[:, 2:32]))
    score_val[count] = metrics.accuracy_score(
        validate['diagnosis'], reg.predict(validate.iloc[:, 2:32]))
    precision_val[count] = metrics.precision_score(
        validate['diagnosis'], reg.predict(validate.iloc[:, 2:32]))
    score_test[count] = metrics.accuracy_score(
        test['diagnosis'], reg.predict(test.iloc[:, 2:32]))
    recall_test[count] = metrics.recall_score(
        test['diagnosis'], reg.predict(test.iloc[:, 2:32]))
    precision_test[count] = metrics.precision_score(
        test['diagnosis'], reg.predict(test.iloc[:, 2:32]))
    count = count + 1


# We create a data frame with every model applied  and their metrics calculated previously. <p>
# Let's see the first data frame rows!

# In[ ]:


matrix = np.matrix(np.c_[list_C, score_train, score_val, precision_val,
                         score_test, recall_test, precision_test])
models = pd.DataFrame(data = matrix, columns = 
             ['C', 'Train Accuracy', 'Validation Accuracy', 'Validation Precision' ,
              'Test Accuracy', 'Test Recall', 'Test Precision'])
models.head(n=10)


# We have trained my models with the training set as usual in the machine learning techniques. <p>
# We select the model with more accuracy. 

# In[ ]:


best_index = models['Validation Accuracy'].idxmax()
models.iloc[best_index, :]


# We calculate again the best logistic regression model.<p>
# The best parameter C has been selected based on the accuracy with the validation set.. Furthermore, If we estimate the generalization accuracy with the validation test,  we probably overestimate it. <p>
# **In order to estimate correctly the generalization accuracy, We will use the test set (absolutely indepent with the model chosed).** <p>

# In[ ]:


reg = linear_model.LogisticRegression(C=list_C[best_index])
reg.fit(train.iloc[:,2:32], train['diagnosis'])


# Confusion matrix. Train set, validate set, test set.

# In[ ]:


print('Train Set')
m_confusion_train = metrics.confusion_matrix(train['diagnosis'],
            reg.predict(train.iloc[:, 2:32]))
pd.DataFrame(data = m_confusion_train, columns = ['Predicted 0', 'Predicted 1'],
            index = ['Actual 0', 'Actual 1'])


# In[ ]:


print('Validation Set')
m_confusion_validate = metrics.confusion_matrix(validate['diagnosis'],
                         reg.predict(validate.iloc[:, 2:32]))
pd.DataFrame(data = m_confusion_validate, columns = ['Predicted 0', 'Predicted 1'],
            index = ['Actual 0', 'Actual 1'])


# In[ ]:


print('Test Set')
m_confusion_test = metrics.confusion_matrix(test['diagnosis'],
                         reg.predict(test.iloc[:, 2:32]))
pd.DataFrame(data = m_confusion_test, columns = ['Predicted 0', 'Predicted 1'],
            index = ['Actual 0', 'Actual 1'])


# The real accuracy of my model is approximately 93% <p>
# (We should evaluate it  with test set because it is independent of the model selected)
# 
# 
