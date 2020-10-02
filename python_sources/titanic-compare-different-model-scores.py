#!/usr/bin/env python
# coding: utf-8

# Simple Data exploration and prediction using different models and comparing the scores.
# Step by step instruction for easy understanding. Finally submitting the best prediction result to Kaggle.

# In[ ]:


#Import all the necessary python packages to perform data exploration and prediction
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.neighbors import KNeighborsClassifier #Simple model for initial classification
from sklearn.model_selection import train_test_split # For splitting train and test data


# In[ ]:


#Read both training dataset and test dataset
train_data = pd.read_csv(r'../input/train.csv') 
test_dataset = pd.read_csv(r'../input/test.csv')


# In[ ]:


#Get information about the training dataset
train_data.info()


# In[ ]:


# Remove the Survived column from train data and put into a target variable test_data. 
# Because this is the feature / target that we need to predict
test_data= train_data.pop('Survived')


# In[ ]:


# We do not need passenger name and Passenger Id for prediction as they do not impact the result at all.
train_data.pop('Name')
train_data.pop('PassengerId')


# In[ ]:


# We can even ignore the Ticket feature as a ticket can not help in predicting the survival.
train_data.pop('Ticket')


# In[ ]:


# Get an overview of the training dataset now.
train_data.head()


# # Lets get deeper insight about each of the feature one by one.

# In[ ]:


# Lets get more information about the Pclass feature
train_data['Pclass'].value_counts()


# In[ ]:


# Now we know that the Pclass has only 3 values i.e 3, 1 and 2. 
# So, it is a categorical feature and hence we should convert to dummy variable.


# In[ ]:


train_data = pd.get_dummies(train_data, columns = ['Pclass'])


# In[ ]:


# Here is how the new dataset looks like
train_data


# Lets explore the Sex feature
# ----------------------------

# In[ ]:


train_data['Sex'].value_counts()


# In[ ]:


# So, there are 577 males and 314 females. Again we need to create dummy variable for this.
# This is required because Sex is also a categorical feature and has only two values - male and female


# In[ ]:


train_data = pd.get_dummies(train_data, columns = ['Sex'])


# In[ ]:


# Lets see the new train dataset
train_data


# Lets explore Age feature
# =====

# In[ ]:


# Since age is a continuous variable and not categorical, we should not create dummy
# Lets check for any missing data in age
train_data['Age'].count()


# In[ ]:


# So, we have only 714 rows with age data. Whereas we should be having 891 rows.
# We need to fill the empty Age data with some value.
# This value should be derived logically. The most common way is to find the mean age 
# And fill the empty values with the mean age.

train_data['Age'][train_data['Age'].isnull()] = train_data['Age'].mean()


# Let's explore SibSp feature
# ---------------------------

# In[ ]:


# Let's see whether its a continuous or categorical variable
train_data['SibSp'].value_counts()


# In[ ]:


# As you can see, it just has 7 values, it is a categorical variable.
# So, we should create dummy variable for this
# I.e create a separate feature for each value


# In[ ]:


train_data = pd.get_dummies(train_data, columns = ['SibSp'])


# In[ ]:


# Lets see the new Training data
train_data


# Let's explore Parch feature
# ---------------------------

# In[ ]:


train_data['Parch'].value_counts()


# In[ ]:


# This is again a categorical variable and hence we need to create dummy variables


# In[ ]:


train_data = pd.get_dummies(train_data, columns=['Parch'])


# Let's explore Fare feature
# --------------------------

# In[ ]:


train_data['Fare'].count()


# In[ ]:


# Since there is no missing data in Fare feature, we need not do much of feature engineering here


# Let's explore Cabin feature
# ---------------------------

# In[ ]:


train_data['Cabin'].value_counts()


# In[ ]:


# Looks like Cabin is not a categorical variable and has alphanumeric values.
# So, we should be good to ignore the data for Cabin
train_data.pop('Cabin')


# Let's explore Embarked feature
# ------------------------------

# In[ ]:


train_data['Embarked'].value_counts()


# In[ ]:


# So, Embarked is a categorical variable and we need to create dummy variables out of it
# Before that, Embarked has only 889 values. We need to fill the two missing rows
# Here we can use the value that appears the max number of times to fill the blank two values
# So, the value to be used will be S. Because S appears 644 times. So, it should be a fair guess 
# to use S as the missing value


# In[ ]:


# FIll the empty two rows with value S
train_data['Embarked'][train_data['Embarked'].isnull()] = 'S'


# In[ ]:


train_data = pd.get_dummies(train_data, columns=['Embarked'])


# In[ ]:


train_data


# In[ ]:


# If you check the test data, there is an additional PARCH_9 value 
# which is not available in the training data. So, adding a new feature Parch_9 with value = 0
train_data['Parch_9'] = 0


# In[ ]:


train_data.info()


# In[ ]:



#Lets split the training and test data for predicting the score
X_train,X_test,y_train,y_test = train_test_split(train_data, test_data, random_state =100)


# ## Prediction using KNeighbors Classifier

# In[ ]:


# Initialize the KNN classifier
knn = KNeighborsClassifier(n_neighbors=1)


# In[ ]:


# Train the model
knn = knn.fit(X_train, y_train)


# In[ ]:


#Calculate the score on training data
knn.score(X_train, y_train)


# In[ ]:


#Calculate the score on test data
knn.score(X_test, y_test)


# ## Lets try using Decision tree

# In[ ]:


from sklearn import tree


# In[ ]:


my_tree = tree.DecisionTreeClassifier(random_state=1)


# In[ ]:


my_tree = my_tree.fit(X_train, y_train)


# In[ ]:


my_tree.score(X_train, y_train)


# In[ ]:


my_tree.score(X_test, y_test)


# ## Lets try Random Forest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


my_forest = RandomForestClassifier(max_depth = 10, min_samples_split=2,
                                   random_state=10, n_estimators=80)


# In[ ]:


my_forest = my_forest.fit(X_train, y_train)


# In[ ]:


my_forest.score(X_train, y_train)


# In[ ]:


my_forest.score(X_test, y_test)


# In[ ]:


#Random forest gave the best prediction, so we should use this model to submit our prediction to Kaggle


# ## Lets create Kaggle submission

# In[ ]:


test_dataset.head()


# In[ ]:


# Need to pop out all unwanted features
PassengerId = test_dataset.pop('PassengerId')
test_dataset.pop('Name')
test_dataset.pop('Ticket')
test_dataset.pop('Cabin')


# In[ ]:


test_dataset.info()


# In[ ]:


# Fill empty rows just like we did for training dataset
test_dataset['Age'][test_dataset['Age'].isnull()] = test_dataset['Age'].mean()


# In[ ]:


test_dataset['Fare'][test_dataset['Fare'].isnull()] = test_dataset['Fare'].mean()


# In[ ]:


test_dataset.info()


# In[ ]:


#Create dummy variables just like we did for training dataset
test_dataset = pd.get_dummies(test_dataset, columns = ['Pclass'])


# In[ ]:


test_dataset = pd.get_dummies(test_dataset, columns = ['Sex'])


# In[ ]:


test_dataset = pd.get_dummies(test_dataset, columns = ['SibSp'])


# In[ ]:


test_dataset = pd.get_dummies(test_dataset, columns=['Parch'])


# In[ ]:


test_dataset = pd.get_dummies(test_dataset, columns=['Embarked'])


# In[ ]:


test_dataset.head()


# In[ ]:


test_dataset.shape


# In[ ]:


train_data.shape


# In[ ]:


# Predict the result on test data
prediction = my_forest.predict(test_dataset)


# In[ ]:


# Save in CSV and submit to Kaggle.

