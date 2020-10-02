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


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
data_frame = [train_df, test_df]


# What are the various features in the dataset? We know PassengerID and Survival, but what else is there?

# In[ ]:


print(train_df.columns.values)


# In[ ]:


print(test_df.columns.values)


# Let us look at the preview of the data to get better idea and visualisation.

# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


train_df.describe()


# In[ ]:


train_df.shape, test_df.shape


# Here is what we can see and deduce about the data from the above:
# 1. About 38% of 891 total survived in the Training Data. 
# 2. Since 50% and 75% correspond to Pclass of 3, we can deduce that there were most passengers in Pclass == 3. 
# 3. The average age of the passengers is approximately 29 and the eldest is 80 years old. The youngest is a baby of about 0.4 years. 
# 
# Now, first lets get an idea of how many survived of the 891 in the training set.

# In[ ]:


train_df['Survived'].value_counts()


# 0 --> Not survived; 1 --> Survived. As stated above from description, it checks out that 38% (=342/891) survived. 
# 
# Now, let us get an idea of how many passengers were in each Pclass.

# In[ ]:


train_df['Pclass'].value_counts()


# How many males and females?

# In[ ]:


train_df['Sex'].value_counts()


# In[ ]:


test_df['Sex'].value_counts()


# Now, we shall see the correlation between survival and the other features individually. Below are the various survival rates for different classes of Pclass, Sex, Embarked port.

# In[ ]:


train_df[['Pclass', 'Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train_df[['Embarked','Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# This same kind of correlation can be done with the remaining features of Age and Fare also but since these are continuous values there would be much too many classes and it is difficult to see correlation between survival and the feature. 
# One way to go about this is to make categories of ranges of ages. For example, 0-15 yrs, 15-35 yrs etc. 

# In[ ]:


train_df['AgeRange'] = pd.cut(train_df['Age'], 5)
train_df[['AgeRange', 'Survived']].groupby(['AgeRange'], as_index=False).mean().sort_values(by='Survived',ascending=False)


# Similarly, with the Fares. 
# However, we must also ensure there are no missing values. If there are, these need to be filled in before we can proceed to make Fare range categories. 

# In[ ]:


test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)


# In[ ]:


train_df['FareRange'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareRange', 'Survived']].groupby(['FareRange'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# We can see some trends and correlations from the above. For example, more people from Pclass 1 and Pclass 2 survived. A lot of passenger from Pclass 3 did not survive. 
# If a passenger is female, then also there is much more likelihood of survival. 
# Passengers who embarked from 'C' had the most number of survivors, followed by port 'Q' and finally 'S' had the lowest number of survivors. 
# The age range of 0-16 had the most survivors. 
# Passengers who spent between 32 and 512 currency units had the most number of survivors. 

# In[ ]:


train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
data_frame = [train_df, test_df]


# In[ ]:


train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
data_frame = [train_df, test_df]


# In[ ]:


train_df.shape, test_df.shape


# In[ ]:


for dataset in data_frame:
    dataset.loc[dataset['Age'] <= 16.336, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16.336) & (dataset['Age'] <= 32.252), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32.252) & (dataset['Age'] <= 48.168), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48.168) & (dataset['Age'] <= 64.084), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 64.084) & (dataset['Age'] <= 80.00), 'Age'] = 4
    
train_df.head()
    


# In[ ]:


train_df = train_df.drop(['AgeRange'], axis=1)
data_frame = [train_df, test_df]
train_df.head()


# In[ ]:


for dataset in data_frame:
    dataset['Embarked'] = dataset['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)
    
train_df.head()


# We get the above error because there are missing values in the Port column of our data. Therefore, these need to be filled in before we can convert the values into numerical form. 
# We need to get an idea of the number of missing values. 

# In[ ]:


train_df['Embarked'].value_counts()


# Adding these, we have 644 + 168 + 77 = 889, out of 891 training examples. Therefore, there are two missing values. Since this is a small number, we can fill in these values with the mode(most commonly occuring) value. 

# In[ ]:


mode_value_embarked = train_df.Embarked.dropna().mode()[0]
mode_value_embarked


# In[ ]:


for dataset in data_frame:
    dataset['Embarked'] = dataset['Embarked'].fillna(mode_value_embarked)


# Now, we can attempt again to convert the values into numerical ones.

# In[ ]:


for dataset in data_frame:
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

train_df.head()


# Voila! It works. Now, same needs to be checked and done for other columns as there may be missing values. 
# Attempting to convert to numerical the fare values, we get a similar error, so we must fill in the missing fare values before going ahead to convert to numerical values for FareRange. 

# In[ ]:


for dataset in data_frame:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31.0), 'Fare'] = 2
    dataset.loc[(dataset['Fare'] > 31.0), 'Fare'] = 3
    
    dataset['Fare'] = dataset['Fare'].astype(int)
    
train_df = train_df.drop(['FareRange'], axis=1)

data_frame=[train_df, test_df]

train_df.head()


# In[ ]:


train_df.shape, test_df.shape


# In[ ]:


test_df.head()


# Now, the only remaining non-numerical values in our features is 'Sex'. This is an important one and we cannot drop it because we have observed earlier that the gender of the passenger has a strong correlation with survival. Therefore, let us first attempt to conver these to numerical values. 

# In[ ]:


for dataset in data_frame:
    dataset['Sex'] = dataset['Sex'].map({'male':0, 'female':1}).astype(int)

train_df.head()


# In[ ]:


test_df.head()


# Now, we can finally move ahead with fitting this data into the ML model and seeing how well the predictions are. 

# In[ ]:


from sklearn.linear_model import LogisticRegression
logit = LogisticRegression()
X_train = train_df.drop('Survived', axis=1)
Y_train = train_df['Survived']
X_test = test_df.drop('PassengerId', axis=1)

X_train.shape, Y_train.shape, X_test.shape


# In[ ]:


logit.fit(X_train, Y_train)
Y_pred = logit.predict(X_test)
acc_log = round(logit.score(X_train, Y_train)*100, 2)
acc_log


# We get this error because we have some missing values in some of our columns. We need to find out which these are. The following are the feature columns we have:
# Pclass, Sex, Age, SibSp, Parch, Fare, Embarked.
# Near the beginning, we had conducted value count for Pclass, Sex and Embarked. Embarked had two missing values that have been taken care of. The other two's values added to 891 which is the total number of training examples. Therefore these three, we know, do not have any missing values. The columns we need to check for missing values are:
# Age, SibSp, Parch, Fare
# 

# In[ ]:


print(train_df)


# We see above that there are numerous cases of NaN values in the Age column. This could be the case for the other three also. One way to ensure how many values are NaN in all the columns is to do value count for the four columns.

# In[ ]:


train_df['Age'].value_counts()


# The total here adds to 714, meaning there are 177 cases of null values in the Age column in training set. Before we do anything about this, we need to know the null values in the other columns and repeat for the test set. 

# In[ ]:


train_df['SibSp'].value_counts(), train_df['Parch'].value_counts(), train_df['Fare'].value_counts()


# All of these add up to 891 thereby showing that there are no null values in these feature columns. Now, for the test_df, we had earlier done completing of the Fare column as there had been 1 null value. For the other three, it still needs to be done. 

# In[ ]:


test_df['Age'].value_counts(), test_df['SibSp'].value_counts(), test_df['Parch'].value_counts()


# The totals in this case should add to 418. 
# In the case of 'Age', there are 86 missing or null values. 
# In the case of the other two, the values add up to 418, therefore no null values. 
# We now have null values in the "Age' column in both the training and testing part of the dataframe. 
# Since there are a large number of these, we cannot fill it up by using the mode as it may not be accurate. A more accurate method would be to use the median instead. 

# In[ ]:


for dataset in data_frame:
    dataset['Age'].fillna(dataset['Age'].median(), inplace=True)
    
print (train_df)


# Now that we have gotten rid of the null values, we can get back to our model and predict. 

# In[ ]:


X_train = train_df.drop('Survived', axis=1)
Y_train = train_df['Survived']
X_test = test_df.drop('PassengerId', axis=1)

logit.fit(X_train, Y_train)
Y_pred = logit.predict(X_test)
logit_score = logit.score(X_train, Y_train)*100
print(str(logit_score)+' %')


# There we have it! Logistic Regression has given us 79.12% accuracy in the test set. 
# We can see how each of the features correlate with survival. 

# In[ ]:


feature_correlation = pd.DataFrame(train_df.columns.delete(0))
feature_correlation.columns = ['Feature']
feature_correlation["Correlation"] = pd.Series(logit.coef_[0])

feature_correlation.sort_values(by='Correlation', ascending=False)


# We can try some other linear models to see if they yield a better score. 

# In[ ]:


from sklearn.svm import SVC, LinearSVC

svc=SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)

svc_score = svc.score(X_train, Y_train)*100
print(str(svc_score)+' %')


# Gives a better score than Logistic Regression!
# 
# Let us now try a nonlinear model: K Nearest Neighbors

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)

knn_score = knn.score(X_train, Y_train)*100
print(str(knn_score)+' %')


# This is the highest score we have achieved! We can play around with different values for n to see how it affects the score. It can be seen that n=5 gives the best result ( approx. 85%).
# We shall stop here. 
# 
# I am glad I have been able to go through the Titanic dataset and build a model to predict its dataset. 
# 
# Thanks very much to:
# * Manav Sehgal's Titanic Data Science Solutions kernel
# * LD Freeman's A Data Science Framework kernel
# * Suhas's notebook_so_titanic kernel
# 
# 
# Cheers. 
