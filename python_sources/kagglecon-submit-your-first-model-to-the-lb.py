#!/usr/bin/env python
# coding: utf-8

# ### Run a simple Cell on Kaggle 

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed.
# Let's import few libraries an

import os
print("hello kaggle")
print("Listing in the working directory:" , os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ### Steps to complete this workshop on your own computer:
# The kernel below can be run in the browser. But if you would like to run the code locally on your own computer, you can follow the steps below.
# 
# 1. Create a Kaggle account (https://www.kaggle.com/).
# 
# 2. Download Titanic dataset (https://www.kaggle.com/c/titanic/data).
#     a. Download 'train.csv' and 'test.csv'.
#     b. Place both files in a folder named 'input'.
#     c. Place that folder in the same directory as your notebook.
# 
# 3. Install Jupyter Notebooks (follow my installation tutorial if you are confused)
# 4. Download this kernel as a notebook with empty cells from my GitHub. If you are new to GitHub go the repository folder, click "Clone or Download", then unzip the file and pull out the notebook you want.
# Run every cell in the notebook (except the optional visualization cells).
# 5. Submit CSV containing the predictions.
# 6. Try to improve the prediction by using the challenge prompts which are suitable to your level.
# 
# ### 1. Load the train/test datasets
# #### load data
# 
# 

# In[2]:


import pandas as pd

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#Drop features we are not going to use
train = train.drop(['Name','SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],axis=1)
test = test.drop(['Name','SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],axis=1)

#Look at the first 3 rows of our training data
train.head(3)


# #### Prepare the data to be read by our algorithm
# 

# In[3]:


for df in [train,test]:
    df['Sex_binary']=df['Sex'].map({'male':1,'female':0})
    
#Fill in missing age values with 0 (presuming they are a baby if they do not have a listed age)
train['Age'] = train['Age'].fillna(0)
test['Age'] = test['Age'].fillna(0)

#Select feature column names and target variable we are going to use for training
features = ['Pclass','Age','Sex_binary']
target = 'Survived'

#Look at the first 3 rows (we have over 800 total rows) of our training data.; 
#This is input which our classifier will use as an input.
train[features].head(3)


# #### base model

# In[4]:


from sklearn.tree import DecisionTreeClassifier

#Create classifier object with default hyperparameters
clf = DecisionTreeClassifier()  

#Fit our classifier using the training features and the training target values
clf.fit(train[features],train[target]) 


# #### run model

# In[5]:


#Make predictions using the features from the test data set
predictions = clf.predict(test[features])
predictions


# #### submit to kaggle

# In[6]:


#Create a  DataFrame with the passengers ids and our prediction regarding whether they survived or not
submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':predictions})

#Convert DataFrame to a csv file that can be uploaded
#This is saved in the same directory as your notebook
filename = 'titanic_prediction_model_base.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)


# #### Submit file to Kaggle
# 
# Go to the submission section of the Titanic competition (https://www.kaggle.com/c/titanic/submit). Drag your file from the directory which contains your code and make your submission.
# 
# Congratulations - you're on the leaderboard!

# In[ ]:




