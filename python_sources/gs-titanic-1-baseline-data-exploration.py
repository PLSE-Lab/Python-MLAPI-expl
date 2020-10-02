#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Exploring data
# The first step is to import the data and verify it's been imported correctly. We'll do this with a .describe() and a .head() 

# In[ ]:


training_data = pd.read_csv("../input/train.csv")
training_data.describe()


# In[ ]:





# In[ ]:


training_data.head()


# In[ ]:


training_data.isnull().sum()


# Looking at those two results, it's apparent that .describe() does not describe non-numeric values such as Name, Sex, Ticket, Cabin, or Embarked. An easy way to describe these values would be .info(). 

# In[ ]:


training_data.info()


# But that isn't too useful for finding out how many unique values there are. Let's see if there are any dtype==object columns which may be easy to encode. We'll decide what should be encoded by:
# 1. Checking if the dtype is object
# 2. Checking if the number of unique values is under 20
# The first check determines if the value is not already an integer or a float. Basically, if a number exists, it's a number, and doesn't need to be encoded as a number.
# The second check determines the reasonability of encoding a value. For example, there are  (aka, there are under 20 possible categories)

# In[ ]:


n = 0
for col in training_data:
    print (training_data.dtypes.index[n])
    if (training_data[col].dtypes) == object:
        if (len(training_data[col].unique())) <= 20:
            print ("Encodable - len =", len(training_data[col].unique()))
    else:
        print ("Should not be encoded")
    n += 1


# In[ ]:


# Another way to do this is to only print values that should be encoded:
n = 0
for col in training_data:
    if (training_data[col].dtypes) == object:
        if (len(training_data[col].unique())) <= 20:
            print (training_data.dtypes.index[n])
            print ("Encodable - len =", len(training_data[col].unique()))
    n += 1


# In[ ]:


# For reference, here is the len of all dtype==object columns.
n = 0
for col in training_data:
    if (training_data[col].dtypes) == object:
        print (training_data.dtypes.index[n])
        print ("Encodable - len =", len(training_data[col].unique()))
    n += 1


# ## Encoding gender and childhood
# Off the bat, I'm going to encode gender to make it easier for a machine to process. By instinct, it's easy to assume gender will be an important feature, because of the "get the women and children in first" code of conduct associated with the sinking of the RMS Titanic. Since childhood is not explicitly stated in the data frame, I will have to base childhood of an age of 15 years or younger (<= 15) or (<16). Since there are only 714 values for age, we will have to deal with null values.
# ### Encoding gender
# I will encode gender first by creating a new column called female and setting it to True if the Sex column = 'female'

# In[ ]:


training_data2 = training_data.copy()
training_data2['female'] = training_data2.Sex == 'female'
training_data2.head()


# ### Encoding Childhood
# Now I'll explore data where age is notnull and <16 so that I can see if there is anything that may indicate childhood outside of age. I'll do this by adding a column which indicates age under 16

# In[ ]:


training_data3 = training_data2.copy()
training_data3['child'] = training_data2.Age < 16
training_data3.head()


# ## Dealing with missing values in the Age column
# There is a concern is what to do with NaN values.  For now, we are simply going to replace the null Age values with the median. We can return to this after we have an initial model built and move from there.

# In[ ]:


training_data4 = training_data3.copy()
training_data4['Age'] = training_data3.Age.fillna(training_data3['Age'].median())
training_data4.isnull().sum()


# # Establishing a baseline model
# The best "first" thing to do when creating a model is to get a baseline established. Let's do that.

# In[ ]:


# here are the models I'll use for a first-try
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
# here are the metrics I'll check them with
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
# and the code to split the test/train data
from sklearn.model_selection import train_test_split


# In[ ]:


# I'm going to write a function to make the confusion matrix easier on the eyes
def confusionMatrixBeautification(y_true, y_pred):
    rows = ['Actually Died', 'Actually Lived']
    cols = ['Predicted Dead', 'Predicted Lived']
    conf_mat = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(conf_mat, rows, cols)


# In[ ]:


# I also want to split the training_data dataframe into a training and testing portion
train_baseline, test_baseline = train_test_split(training_data4, random_state = 0)
train_baseline.shape, test_baseline.shape


# In[ ]:


# For a first run, I'll try using the following features
features = ['Age', 'female', 'child', 'Fare', 'Pclass']
target = 'Survived'


# In[ ]:


# Now comes fitting the models
model01 = RandomForestClassifier(random_state=0)
model02 = DecisionTreeClassifier(random_state=0)
model03 = LogisticRegression()

model01.fit(train_baseline[features], train_baseline[target]);
model02.fit(train_baseline[features], train_baseline[target]);
model03.fit(train_baseline[features], train_baseline[target]);


# In[ ]:


# Now let's define a pretty function to measure the scores of those models
def printScore(model_number):
    print("Train Accuracy: ", round(accuracy_score(train_baseline[target], model_number.predict(train_baseline[features]))*100,2), "%")
    print("Train Recall: ", round(recall_score(train_baseline[target], model_number.predict(train_baseline[features]))*100,2), "%")
    print("Train Confusion Matrix: \n", confusionMatrixBeautification(train_baseline[target], model_number.predict(train_baseline[features])))
    print("Test Accuracy: ", round(accuracy_score(test_baseline[target], model_number.predict(test_baseline[features]))*100,2), "%")
    print("Test Recall: ", round(recall_score(test_baseline[target], model_number.predict(test_baseline[features]))*100,2), "%")
    print("Test Confusion Matrix: \n", confusionMatrixBeautification(test_baseline[target], model_number.predict(test_baseline[features])))
    
def printTestRecall(model_number):
    print("Test Recall: ", round(recall_score(test_baseline[target], model_number.predict(test_baseline[features]))*100,2), "%")


# In[ ]:


print("RandomForestClassifier()")
printTestRecall(model01)
print("\n\nDecisionTreeClassifier()")
printTestRecall(model02)
print("\n\nLogisticRegression()")
printTestRecall(model03)


# Okay, so now we have a baseline. 
# 71.43% recall with random forest.
# 69.05% with decision tree.
# 69.05% with logistic regression.
# None of those will get me on the leaderboard. Let's explore some more data.

# In[ ]:


# I'll run those same models with all features which are numerical/boolean (except ID)
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'female', 'child']


# In[ ]:


model01.fit(train_baseline[features], train_baseline[target]);
model02.fit(train_baseline[features], train_baseline[target]);
model03.fit(train_baseline[features], train_baseline[target]);


# In[ ]:


print("RandomForestClassifier()")
printTestRecall(model01)
print("\n\nDecisionTreeClassifier()")
printTestRecall(model02)
print("\n\nLogisticRegression()")
printTestRecall(model03)


# Okay, so now we have a baseline. 
# 67.86% recall with random forest.
# 69.05% with decision tree.
# 67.86% with logistic regression.
# None of those will get me on the leaderboard. Let's explore some more data. <br><br>
# Go to next notebook
