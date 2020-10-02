#!/usr/bin/env python
# coding: utf-8

# The competition is simple: use machine learning to create a model that predicts which passengers survived the Titanic shipwreck.
# I have Previously Done EDA & Applied Logistic Regression Model on same data.
# Please Refer : [Kaggle](https://www.kaggle.com/adarshsambare) here,
# If you like my work **Do UPVOTE**

# Let's work on our [Titanic](https://www.kaggle.com/c/titanic) Data set
# 

# In[ ]:


# Let's Get Started by importing libraries
import numpy as np # linear algebra
import pandas as pd # data processing

# Visulation libraries
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Reading the data as train and test
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')


# In[ ]:


# Null values are empty data points
# checking the data for Null values
print(train.isnull().sum())
print('\n')
print(test.isnull().sum())
# we do have null values in some columns


# In[ ]:


# for combining both the data we need survived column in test data
test['Survived'] = 0
# need a new colume to differencate between the two
train['istest'] = 0
test['istest'] = 1


# In[ ]:


# Checking if survived was added or not
#train.describe()
test.describe()


# In[ ]:


# combining the data
dataset = pd.concat([train,test], join = 'inner')


# In[ ]:


dataset.describe()


# In[ ]:


dataset.isna().sum()


# In[ ]:


# Writing a function for changing null values as per median
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
        
    else:
        return Age


# In[ ]:


# Applying above impute on age columns
dataset['Age'] = dataset[['Age','Pclass']].apply(impute_age,axis = 1)


# In[ ]:


# Dropping the Cabin as too many null values
dataset.drop('Cabin', axis = 1, inplace = True)


# In[ ]:


# Filling up the null values with logical values
dataset['Fare'].fillna(14.454, inplace = True)
dataset['Embarked'].fillna("S", inplace = True)


# In[ ]:


dataset.describe()


# In[ ]:


dataset.isna().sum()


# In[ ]:


sex_dum = pd.get_dummies(dataset['Sex'],drop_first = True)


# In[ ]:


embark_dum = pd.get_dummies(dataset['Embarked'],drop_first = True)


# In[ ]:


pclass_dum = pd.get_dummies(dataset['Pclass'],drop_first = True)


# In[ ]:


# adding the dummies in the data frame
dataset = pd.concat([dataset,sex_dum,embark_dum,pclass_dum],axis = 1)


# In[ ]:


dataset.head()


# In[ ]:


# Checking the new train data
dataset.drop(['PassengerId','Pclass','Sex','Embarked','Name','Ticket'],axis = 1,inplace =True)


# In[ ]:


dataset.describe()


# In[ ]:


dataset.head()


# # **Splitting the Train Data**
# **NOTE** : We had done work only on the train data here. The test data is not used in this tutorial.
# It will be updated soon, Kindly wait till then.

# In[ ]:


train = dataset[dataset['istest'] == 0]
test = dataset[dataset['istest'] == 1]


# In[ ]:


train.drop('istest', axis = 1,inplace = True)


# In[ ]:


test.drop('istest', axis = 1,inplace = True)


# In[ ]:


train.isna().sum()


# In[ ]:


# splitting the train data for cross validations
x = train.drop('Survived',axis = 1)
y = train['Survived']


# In[ ]:


# Splitting using sklearn
from sklearn.model_selection import train_test_split


# In[ ]:


# Actual splitting
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state = 101)


# # **Decision Tree Model**

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


dt = DecisionTreeClassifier()


# In[ ]:


dt.fit(x,y)


# **Predicting and Accuracy of Decision Tree **

# In[ ]:


# Predicting on the splited train data
prediction = dt.predict(x_test)


# In[ ]:


# As it is classification problem we need confusion matrix
from sklearn.metrics import confusion_matrix, classification_report


# In[ ]:


print(classification_report(y_test, prediction))
# Accuracy = 98%  


# In[ ]:


test.shape


# In[ ]:


Survived = dt.predict(test.drop('Survived',axis = 1)) 


# In[ ]:


type(Survived)


# In[ ]:


PassengerId = (list(range(892,1310)))


# In[ ]:


type(PassengerId)


# In[ ]:


kaggle_submission = pd.DataFrame(PassengerId, columns=['PassengerId'])


# In[ ]:


kaggle_submission['Survived'] = np.array(Survived)


# In[ ]:


kaggle_submission.describe()


# In[ ]:


# saving the dataframe 
kaggle_submission.to_csv('kaggle_submission_dt', index=False)


# # **Random Forest Model**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


# n_estimators are nothing but no of trees
# do not make to complex forest of decision tree
rfc = RandomForestClassifier(n_estimators=500)


# In[ ]:


# Fitting the Random Forest to trainng data
rfc.fit(x_train,y_train)


# **Prediciton and Accuracy of Random Tree**

# In[ ]:


# Predicting on the training splitted test data by Random Forest 
pred_rfc = rfc.predict(x_test)


# In[ ]:


print(classification_report(y_test, pred_rfc))
# Accuracy = 82%


# # **Conclusion**
# Random Forest Performed better on our training data
