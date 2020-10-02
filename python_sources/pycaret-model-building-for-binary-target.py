#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train=pd.read_csv('/kaggle/input/loan-prediction/train_loan.csv')
test=pd.read_csv('/kaggle/input/loan-prediction/test_loan.csv')
sample_submission=pd.read_csv('/kaggle/input/loan-prediction/sample_submission_loan.csv')


# In[ ]:


print(train.shape)
print(test.shape)
print(sample_submission.shape)


# In[ ]:


get_ipython().system('pip install pycaret')


# In[ ]:


from pycaret import classification
from pycaret.classification import *


# In[ ]:


#Loan_Status is the target variable in the dataset
from pycaret import classification

classification_setup=classification.setup(data=train, target='Loan_Status')


# In[ ]:


#Building all classification models with the single step and comparing them classification metrics like Accuracy, AUC, Recall, Kappa, F1
from pycaret.classification import *
compare_models()


# In[ ]:


#Model 1: Building catboost due to its high accuracy
catboost  = create_model('catboost') 


# In[ ]:


#Model 2: Building Gradient Boosting Classifier due to its high accuracy
gbc  = create_model('gbc') 


# In[ ]:


#Plotting the confusion matrix for Gradient Boosting
plot_model(estimator=gbc, plot = 'confusion_matrix')


# In[ ]:


#Predicting the target
#Scores variable created at the end of prediction is the probability of the Loan Approval Prediction
#Label variable created at the end of prediction is the Label of 0's and 1's defined based on the probability/score

predict_model(gbc, data=test)


# In[ ]:


#Producing the predictions in sample submission file with Probabilities/Scores
predictions=predict_model(gbc, data=test)
predictions.head()



sample_submission['Loan_Status'] = predictions['Score']
sample_submission.to_csv('Gradient_Boosting.csv',index=False)
sample_submission.head()


# In[ ]:


#Producing the predictions in sample submission file with 1/0's

predictions=predict_model(gbc, data=test)
predictions.head()



sample_submission['Loan_Status'] = predictions['Label']
sample_submission.to_csv('Gradient_Boosting.csv',index=False)
sample_submission.head()


# In[ ]:


#Producing the predictions in sample submission file with Y/N's

cleanup_cols={"Loan_Status": {1:"Y", 0:"N"}}
sample_submission.replace(cleanup_cols, inplace=True) 
sample_submission.head()


sample_submission.to_csv('Gradient_Boosting.csv',index=False) 
sample_submission.head()

