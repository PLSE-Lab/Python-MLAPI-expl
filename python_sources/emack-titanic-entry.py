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


# ### Read the data into their respective dataframes

# In[ ]:


import pandas as pd

test = pd.read_csv("/kaggle/input/titanic/test.csv")
train = pd.read_csv("/kaggle/input/titanic-competion-data/train.csv")


# ### Get the shape of the data

# In[ ]:


test_shape = test.shape
train_shape = train.shape

print('Output is in (row, col)')
print('test.csv: ', test_shape) 
print('train.csv: ', train_shape) 


# ### Visualize our data

# In[ ]:


import matplotlib.pyplot as plt

sex_pivot = train.pivot_table(index="Sex",values="Survived")
sex_pivot.plot.bar()
train_pivot = train.pivot_table(index='Pclass', values='Survived')

train_pivot.plot.bar()
plt.show()


# From above we cab see that more women than men survived and that there were a greater number of survivors from 1st class and with descending numbers of survivors down to third class.

# ### Let's examine the age column

# In[ ]:


print(train["Age"].describe())


# In[ ]:


survived = train[train["Survived"] == 1]
died = train[train["Survived"] == 0]
survived["Age"].plot.hist(alpha=0.5,color='red',bins=50)
died["Age"].plot.hist(alpha=0.5,color='blue',bins=50)
plt.legend(['Survived','Died'])
plt.show()


# At first glance it looks like the highest number of survivors were in the 18-40 year-old range
# 

# ### Prepare the data for machine learning

# ### Seperate the age ranges into survived and not survived in order to use for learning model.
# 
# What ever changes I make to the train set I have to make to the test data set otherwise my learning model will not work. (i.e.) if I add or remove a column in the training set I have to do the same in the test set.
# 
# #### Use Pandas "cut" to do the heavy lifting here
# 
# (From Pandas documentation)
# 
# Use cut when you need to segment and sort data values into bins. This function is also useful for going from a continuous variable to a categorical variable. For example, cut could convert ages to groups of age ranges. Supports binning into an equal number of bins, or a pre-specified array of bins.

# In[ ]:


def process_age(df,cut_points,label_names):
    df["Age"] = df["Age"].fillna(-0.5)
    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)
    return df

# define the age cutoff for age classification
cut_points = [-1,0,18,100] # the ages are in a list
label_names = ["Missing","Child","Adult"] # the age categories are in a list

train = process_age(train,cut_points,label_names)
test = process_age(test,cut_points,label_names)


# In[ ]:


def process_age(df,cut_points,label_names):
    df["Age"] = df["Age"].fillna(-0.5)
    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)
    return df
    
cut_points = [-1,0,5,12,18,35,60,100]
label_names = ["Missing","Infant", "Child", "Teenager", "Young Adult", "Adult", "Senior"] 

train = process_age(train,cut_points,label_names)
test = process_age(test,cut_points,label_names)   

pivot = train.pivot_table(index="Age_categories", values="Survived")
pivot.plot.bar()


# The bar chart shows the rate of survival based on age group, ignoring sex.

# ### Let's examine the Pclass variable

# In[ ]:


train["Pclass"].value_counts()


# In[ ]:


train["Pclass"].head(12)


# ### Prepare the columns Pclass, Age and Sex for machine learning by creating dummy columns to better represent the data we want to learn from

# In[ ]:


def create_dummies(df,column_name):
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    return df

train = create_dummies(train,"Pclass")
test = create_dummies(test,"Pclass")

train.head()


# In[ ]:


def create_dummies(df,column_name):
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    return df

train = create_dummies(train,"Pclass")
test = create_dummies(test,"Pclass")

train = create_dummies(train,"Sex")
test = create_dummies(test,"Sex")

train = create_dummies(train,"Age_categories")
test = create_dummies(test,"Age_categories")


# ## Create the Logistic Regression machine-learning model

# In[ ]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()


# ## Now train the model 

# In[ ]:


columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',
       'Age_categories_Missing','Age_categories_Infant',
       'Age_categories_Child', 'Age_categories_Teenager',
       'Age_categories_Young Adult', 'Age_categories_Adult',
       'Age_categories_Senior'] # The x variables

lr.fit(train[columns], train['Survived']) # The y or target variable we want to predict


# ### First, create a test set from our updated training set
# 
# The convention in machine learning is to call these two parts train and test. This can become confusing, since we already have our test dataframe that we will eventually use to make predictions to submit to Kaggle. To avoid confusion, from here on, we're going to call this Kaggle 'test' data holdout data, which is the technical name given to this type of data used for final predictions.

# In[ ]:


holdout = test # from now on we will refer to this
               # dataframe as the holdout data

from sklearn.model_selection import train_test_split

columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',
       'Age_categories_Missing','Age_categories_Infant',
       'Age_categories_Child', 'Age_categories_Teenager',
       'Age_categories_Young Adult', 'Age_categories_Adult',
       'Age_categories_Senior']

all_X = train[columns]
all_y = train['Survived']

train_X, test_X, train_y, test_y = train_test_split(
    all_X, all_y, test_size=0.2,random_state=0)


# ## Fit the model on the new training set
# 
# ### and check the accuracy of the prediction

# In[ ]:


from sklearn.metrics import accuracy_score
lr = LogisticRegression()
lr.fit(train_X, train_y)
predictions = lr.predict(test_X) # here we make our predictions
accuracy = accuracy_score(test_y, predictions)

print("Our model's accuracy is: ", accuracy)


# Let's check if I'm overfitting with k-fold cross-validation

# In[ ]:


from sklearn.model_selection import cross_val_score
import numpy as np

lr = LogisticRegression()
scores = cross_val_score(lr, all_X, all_y, cv=10)

accuracy = np.mean(scores)
print('Scores = :', scores)
print('Accuracy = :', accuracy)


# #### We are now ready to use the model we have built to train our final model and then make predictions on our unseen holdout data, or what Kaggle calls the 'test' data set.

# In[ ]:


columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',
       'Age_categories_Missing','Age_categories_Infant',
       'Age_categories_Child', 'Age_categories_Teenager',
       'Age_categories_Young Adult', 'Age_categories_Adult',
       'Age_categories_Senior']
lr = LogisticRegression()
lr.fit(all_X,all_y)
holdout_predictions = lr.predict(holdout[columns])


# ### Now create the Kaggle submission file

# In[ ]:


holdout_ids = holdout["PassengerId"]
submission_df = {"PassengerId": holdout_ids,
                 "Survived": holdout_predictions}
submission = pd.DataFrame(submission_df)

submission.to_csv('submission.csv', index=False)


# Let's save the submission.csv 

# In[ ]:


sub = pd.read_csv('submission.csv')
sub.head()


# In[ ]:




