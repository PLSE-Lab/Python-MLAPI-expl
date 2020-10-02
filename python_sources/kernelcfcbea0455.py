#!/usr/bin/env python
# coding: utf-8

# Titanic Kaggle Competition : Which Passengers Survived the Titanic Voyage? A Machine Learning Approach

# I am relatively new to the field of Data Science and Machine Learning. I will, nonetheless, try my best to take you through my solution to this challenge.
# Feel free to leave any comment at the appropriate section. Let's dive in!

# ## Import necessary libraries
# Let's begin by importing the needed machine learning libraries

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ## Load datasets (train and test)

# In[ ]:


training_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

#Let's take a look at the first 10 rows from the training dataset
training_df.head(10)


# An explanation of the 12 columns(features) are defined below
# 
# PassengerId: the unique id of a passenger
# Survived: A binary value representing whether a passenger survived (1) or died (0)
# Pclass: the ticket class; 1 = 1st, 2 = 2nd and 3 = 3rd
# Sex: Gender of the passenger
# Age: Age of the passenger in years
# Sibsp: Number of siblings or spouses on board
# Parch: Number of parents / children on board
# Ticket: The ticket number
# Fare: The passenger fare (amount paid to board the ship)
# Cabin: The cabin number
# Embarked: Port of embarkation; C = Cherbourg, Q = Queenstown, S = Southampton

# In[ ]:


training_df.dtypes ##Examine data types of each column


# # Exploratory data analysis

# In[ ]:


training_df.describe() #get some summary statistics on the numerical variables


# We can see there are 891 rows in the training dataset representing passengers. However, the actual number of passengers who board the titanic ship, according to sources, were more than this figure. 
# From the summary statistics, the mean value of passengers who survived is nearly 40%.
# The Age distribution of passengers were in the range of 1 to 80 years.
# Some passengers paid as high as USD 512 with some paying as low as USD 8
# It is apparent from the summary statistics that no passenger had more than 8 siblings and spouses on board. With this, we can conclude confidently that the maximum family size should be around 14 (including the Parch class)

# In[ ]:


training_df.isnull().sum() ##get sum of null/NaN rows

#There are 177 null or nan rows in the Age column as are a whopping 687 (about 80%) missing rows in the Cabin column. 
#Embarked has only 2 missing or NaN rows


# In[ ]:


test_df.isnull().sum() ##get sum of null/NaN rows for test dataset


# In[ ]:


#Plot histogram on the Age column to show the distribution
training_df.Age.hist()

#The histogram tells that there are more youth (20 - 35 years) in the dataset


# In[ ]:


#The distribution of Fares
training_df.Fare.hist()


# In[ ]:


#A bar plot showing the survival ratio by the category of ticket (pClass)
training_df.groupby('Pclass').mean()["Survived"].plot(kind='bar')

#It appears that passengers with class A or 1st class tickets had higher chances of surving that passengers with 3rd ticket class


# In[ ]:


#Bar plot showing the survival ration by gender or Sex. We can see more female survived than did male
training_df.groupby('Sex').mean()[["Survived"]].plot(kind='bar')


# In[ ]:


sns.violinplot(x='Sex', y='Age', hue='Survived', data=training_df, split=True, scale="count", inner="quartile") 
#Of the females who survived, the violinplot below shows that a greater proportion of them (as shown by the distribution) were
# within the age range of 25 to 30 years


# In[ ]:


sns.violinplot(x='Embarked', y='Age', hue='Survived', data=training_df, split=True, scale="count", inner="quartile") ##distribution of embarked with age

##It is also clear from the plot below that passengers who board the ship at C had higher chance of surviving, followed by S and the Q


# In[ ]:


#Let's also plot a correlation matric to investigate which features correlate with others.
corr_matrix = training_df.corr()
corr_matrix


# In[ ]:


sns.heatmap(corr_matrix) ##correlation matric to visualize the relationship with the target variable


# # Sanitize Datasets - Drop some features and Fill NaNs with appropriate techniques

# In[ ]:


##inpute age with mean since it has missing data
training_df.Age.fillna(training_df.Age.mean(), inplace=True)
test_df.Age.fillna(test_df.Age.mean(), inplace=True)
test_df.Fare.fillna(test_df.Fare.mean(), inplace=True)


# In[ ]:


##drop the Cabin column as it has over 600 missing values
training_df.drop('Cabin', axis=1, inplace=True)
test_df.drop('Cabin', axis=1, inplace=True)


# In[ ]:


training_df.info()
test_df.info()


# # Feature Engineering
# In addition to the features that already exist in the dataframe, we can also create / engineer new features.

# In[ ]:


#Process family
def process_family(dataset):
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['SoloFamily'] = dataset.FamilySize.map(lambda x: 1 if x==1 else 0)
    dataset['SmallFamily'] = dataset.FamilySize.map(lambda x: 1 if 2 <= x <=4 else 0)
    dataset['LargeFamily'] = dataset.FamilySize.map(lambda x: 1 if x >= 5 else 0)
    dataset.drop(['SibSp','Parch'], axis=1, inplace=True)
    return dataset


# In[ ]:


training_df = process_family(training_df)

test_df = process_family(test_df)
training_df.head()


# In[ ]:


#process gender
def process_gender(data):
    gender_map = {'male':0, 'female':1}
    data['Sex'] = data.Sex.map(lambda x: 1 if x=='male' else 0)
    return data


# In[ ]:


training_df = process_gender(training_df)
test_df = process_gender(test_df)
training_df.head()


# In[ ]:


#process embarked
def process_embarked(data):
    data.Embarked.fillna(data['Embarked'].mode(), inplace=True)
    #one hot encode process embarked
    encoded_embarked = pd.get_dummies(data['Embarked'], prefix='Embarked')
    data = pd.concat([data, encoded_embarked], axis=1)
    data.drop(["Embarked"], axis=1, inplace=True)
    return data
    


# In[ ]:


training_df = process_embarked(training_df)

test_df = process_embarked(test_df)
training_df.head()


# In[ ]:


##Generate unique titles from dataset


# In[ ]:


def get_titles(data):
    titles = set()
    for title in data:
        titles.add(title.split(",")[1].split(".")[0].strip())
    return titles


# In[ ]:


titles = get_titles(training_df.Name)
titles


# In[ ]:


#process titles
##Portion of code referenced from https://towardsdatascience.com/kaggle-titanic-machine-learning-model-top-7-fa4523b7c40
def process_names(data):
    title_dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess":"Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Royalty"
    }
    data['Title'] = data['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())
    data['Title'] = data['Title'].map(title_dictionary)
    data.drop(['Name'], axis=1, inplace=True)
    #one hot encode titles
    titles_dummies = pd.get_dummies(data['Title'], prefix="Title")
    data = pd.concat([data, titles_dummies], axis=1)
    data.drop(['Title'], axis=1, inplace=True)

    return data


# In[ ]:


training_df = process_names(training_df)

test_df = process_names(test_df)
training_df.head()


# In[ ]:


test_df.head(5)


# In[ ]:


##drop ticket
training_df.drop(['Ticket'], axis=1, inplace=True)
test_df.drop(['Ticket'], axis=1, inplace=True)

training_df.head()


# # Scale Numerical variables

# In[ ]:


##Scale numerical variables
scaler = StandardScaler()
training_df[['Age','Fare']] = scaler.fit_transform(training_df[['Age','Fare']])
training_df.head(10)

test_df[['Age','Fare']] = scaler.fit_transform(test_df[['Age','Fare']])


# # Fit Our Machine Learning Model

# We've reached the interesting part of the project, making preditions with our machine learning models

# In[ ]:


X_Train = training_df.drop(['Survived','PassengerId'], axis=1)
Y_Train =training_df.iloc[:,1]
X_Test = test_df
X_Train.head()


# In[ ]:


#Import sklearn models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics

## Fit a Random Forest Classifier
# In[ ]:


#X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.2, random_state=0)

r_forest = RandomForestClassifier(n_estimators=150, min_samples_leaf=3, max_features =0.5)
r_forest.fit(X_Train, Y_Train)
Y_Predicted = r_forest.predict(X_Test)
feature_importances = pd.DataFrame(r_forest.feature_importances_,
                                   index = X_Train.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
feature_importances


# # Drop Features with less importance (x <0.01)

# In[ ]:


#X_Train.drop(['Embarked_C','Title_Royalty','SoloFamily','Embarked_Q',], axis=1, inplace=True)
#X_Test.drop(['Embarked_C','Title_Royalty','SoloFamily','Embarked_Q',], axis=1, inplace=True)
#X.drop('PassengerId', axis=1, inplace=True) ##This is not needed
#X_Train.head()


# ##Fit a Logistics Regression model

# In[ ]:



logistic_model = LogisticRegression(random_state=0)
logistic_model.fit(X_Train, Y_Train)
model_score = logistic_model.score(X_Train, Y_Train)
parameters = logistic_model.coef_
print(parameters)
print(model_score)


# In[ ]:


#Our Logistic Regression model scored 82.8%


# ##Let's fit a Naive Bayes Gaussian model

# In[ ]:


naive_model = GaussianNB()
naive_model.fit(X_Train, Y_Train)
Y_Predict = naive_model.predict(X_Test)
naive_model.score(X_Train, Y_Train)


# In[ ]:


##Fit a Support Vector Classifier


# In[ ]:


support_vector = SVC()
support_vector.fit(X_Train, Y_Train)
support_vector.score(X_Train, Y_Train)


# In[ ]:


##Gradient Boosting Descent Model


# In[ ]:


graident_boosting = GradientBoostingClassifier()
graident_boosting.fit(X_Train, Y_Train)
graident_boosting.score(X_Train, Y_Train)


# # Submit Code

# In[ ]:


#Of all the models tested, it appears the gradient boosting classifier outperformed the others with a score of 89.7%. 
#This explains why we're using it here
passengerId = X_Test["PassengerId"]
New_Test = X_Test.drop('PassengerId', axis=1)
Y_Predicted = graident_boosting.predict(X_Test)


# In[ ]:


#Create a pandas dataframe and convert it to a csv format (see instructions on how to upload your solution)
submission_dataFrame = pd.DataFrame({'PassengerId': passengerId, 'Survived': Y_Predicted})
submission_dataFrame.to_csv("submission.csv",index=False)


# That's it! We're done.
# Don't forget to leave a comment or suggestion at the comment section. 
# And if I missed reference, please DM and I will be willing to cite where necessary.
# 
# Thanks for reading

# In[ ]:





# In[ ]:





# In[ ]:




