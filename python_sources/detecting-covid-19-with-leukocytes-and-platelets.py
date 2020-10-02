#!/usr/bin/env python
# coding: utf-8

# # Binary classifier based on leukocytes and platelets levels
# 
# This notebook runs a simple binary classifier based on tested patients data for COVID-19 and their leukocytes and platelets reported levels in their blood tests.
# 
# ## For this purpose I've used GaussianNB

# In[ ]:


import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split


# In[ ]:


df = pd.read_csv("/kaggle/input/uncover/UNCOVER/einstein/diagnosis-of-covid-19-and-its-clinical-spectrum.csv")


# In[ ]:


df.sars_cov_2_exam_result.replace('negative', 0, inplace=True)
df.sars_cov_2_exam_result.replace('positive', 1, inplace=True)


# ## Selecting the features we want to train our classifier with
# 
# You may change it if you want to try other features that you think might be more correlated and precise

# In[ ]:


selected_columns = ['sars_cov_2_exam_result','platelets','leukocytes']

df = df[selected_columns]


# In[ ]:


df.dropna(inplace=True)
toTrain = df.drop('sars_cov_2_exam_result', axis=1).to_numpy()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(toTrain, df.sars_cov_2_exam_result, test_size=0.33, random_state=42)


# In[ ]:


gnb = GaussianNB()


# In[ ]:


gnb.fit(X_train,y_train)


# In[ ]:


preds = gnb.predict(X_test)
y_test = y_test.to_numpy()


# ## Here we calculate the results:

# In[ ]:


truePositive = 0
trueNegative = 0
falsePositive = 0
falseNegative = 0

for i in range(len(preds)):
    if(preds[i]==y_test[i]):
        if(preds[i]==1):
            truePositive += 1
        else:
            trueNegative += 1
    else:
        if(preds[i]==1):
            falsePositive += 1
        else:
            falseNegative += 1

print("True Positive predictions:",truePositive)
print("True Negative predictions:",trueNegative)
print("False Positive predictions:",falsePositive)
print("False Negative predictions:",falseNegative)
print("Accuracy:", (truePositive+trueNegative)/(truePositive+trueNegative+falsePositive+falseNegative)*100)

