#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# importing train and test data into train_df and test_df dataframes
import pandas as pd
train = pd.read_csv('/kaggle/input/sce-data-science-2020-course/train.csv')
test = pd.read_csv('/kaggle/input/sce-data-science-2020-course/test.csv')


# In[ ]:


# printing training data information 
# (number of non-null observations, datatype)
print(train.info())
print('-'*100)
print(test.info())


# In[ ]:


# taking care of missing values
def missing(data):
    d = data.copy(deep = True)
    for c in data:
        if (data[c].dtype =='int64') or (data[c].dtype =='float64') : 
            if data[c].isnull().values.any():
                m = data[c].dropna().mean()
                d[c].fillna(m, inplace=True)
        else:          
            if data[c].isnull().values.any():
                m = data[c].dropna().mode()[0]
                d[c].fillna(m, inplace=True)
    return d

trm = missing(train)
tsm = missing(test)


# In[ ]:


# printing training data information with missing values treatment
print(trm.info())
print('-'*100)
print(tsm.info())


# In[ ]:


# preparing training data
cols = ['Pclass','Age','SibSp','Parch','Fare']
x_train = trm[cols]
y = trm['Survived']
x_test = tsm[cols]


# In[ ]:


# defining naive bayes model
# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.naive_bayes
from sklearn.naive_bayes import GaussianNB  
m = GaussianNB(priors=None, var_smoothing=1e-09)


# In[ ]:


# scoring decision tree model
from sklearn.model_selection import cross_val_score
scores = cross_val_score(m, x_train, y, cv = 10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[ ]:


# fitting naive bayes model and building predictions
m.fit(x_train, y)
yy = m.predict(x_test) 


# ===================================================================================================================

# In[ ]:


# preparing submission file
submission = pd.DataFrame( { 'PassengerId': test['PassengerId'] , 'Survived': yy } )
submission.to_csv('naive_bayes_model.csv' , index = False )


# ===================================================================================================================
