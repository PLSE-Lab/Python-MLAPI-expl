#!/usr/bin/env python
# coding: utf-8

# **Machine Learning for Beginners**
# 
# This is an introduction for Machine Learning. There are many different Machine Learning models but they almost all have the same form. This tutorial will teach you how to get started with writing Machine Learning models.
# 
# This Titanic database is not so complicated so it is suitable for beginners.
# 
# This Notebook was of great help to get me started so please check it out and support it. My version includes more explanation and I wish to improve the way of filling missing values in the 'Age' column.
# 
# https://www.kaggle.com/nadintamer/titanic-survival-predictions-beginner

# **Imports**
# 
# The sections below will be some standard imports for data analysis (not model building). Just remember that numpy, pandas and matplot are the most important and common imports and you will likely use them for all your Machine Learning model analysis moving forward.

# In[ ]:


# the standard imports that are used for data analysis
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # for plotting our data
# for plotting
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# **Read the data**
# 
# Now we will read the data and do some data processing on it.

# In[ ]:


# Read the files
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')


# **Inspect the data**
# 
# Now we should inspect our data and see what it looks like. The following code will print out the shape of our data. Then we use the code .head to check the first few rows of data and give us a general idea of what the data looks like.

# In[ ]:


# This shows us the data structure
print(train.shape)

# This will give us an idea of what the data looks like
train.head()


# In[ ]:


# This will give us a summary of our data
train.describe(include = 'all')


# From the data above we can tell what type of data each column is.
# 
# * 1. Numerical Features: Age (Continuous), Fare (Continuous), SibSp (Discrete), Parch (Discrete)
# * 1. Categorical Features: Survived, Sex, Embarked, Pclass
# * 1. Alphanumeric Features: Ticket, Cabin
# 
# Numerical Features are data that contains numeric values. There are Continuous and Discrete. Continuous is where there are many different types of data and Discrete is when the amount of numbers is limited or variation is very few.
# 
# Categorical Features is exactly as the name implies the column is categorized to differnt types.
# 
# Alphanumeric values are strings that have combined letters and numbers. They will be less likely featured in our prediction model but sometime you can gain some information out of these variables.

# **Missing Values**
# 
# Now that we have a better understanding of our data it is time we check how many missing values our data has. How we deal with missing values is important and will affect the accuracu of the model.

# In[ ]:


# This code will give us a summary of the missing data
print(pd.isnull(train).sum())


# The data shows that we are missing values in 'Age', 'Cabin' and 'Embarked'.

# **Going through the Columns**
# 
# We now want to inspect the data and remove certain columns that does not contribute to our model. If we can narrow the most useful data then our model will perform better.
# 
# We also want to alternate the way of how certain columns are presented. You will learn how shortly.

# **PassengerId**
# 
# This Column is a unique value and every passenger has a different passengerId. Thus this column does not contribute to our final prediction model and we should remove it. However we should keep it until we start to make predictions.

# **Survived**
# 
# Obviously we need to keep this column because we need to predict on it but do we need to improve it? This column has numeric values to represent the two different results that we need to predict so it does not need any improvment. We leave this column as is.

# **Pclass**
# 
# This variable indicates which class each passenger was in. This is obviously useful but lets chekc how so.

# In[ ]:


#draw a bar plot of survival by Pclass
sns.barplot(x="Pclass", y="Survived", data=train)

#print percentage of people by Pclass that survived
print("Percentage of Pclass = 1 who survived:", train["Survived"][train["Pclass"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of Pclass = 2 who survived:", train["Survived"][train["Pclass"] == 2].value_counts(normalize = True)[1]*100)

print("Percentage of Pclass = 3 who survived:", train["Survived"][train["Pclass"] == 3].value_counts(normalize = True)[1]*100)


# This indicates that being in a higher class you would have a higher chance of survival so this Column is very important to us for our final predictions.

# **Name**
# 
# We are not interested in the Name themselves but rather we are interested in the titles. A title in a Name can indicate your age and social class so what we truly want from this column is the different types of social classes. We will improve this column later.

# **Sex**
# 
# This is yet another obvious Column that we should include. Many may argue that no matter the sex Man and Woman have an equal chance of surviving. Let's see if that is true.

# In[ ]:


# draw a bar plot of survival by sex
sns.barplot(x="Sex", y="Survived", data=train)

# print percentages of females vs. males that survive
print("Percentage of females who survived:", train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True)[1]*100)

print("Percentage of males who survived:", train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True)[1]*100)


# The output shows that the female have a much higher chance of surviving. This output indicates that we need to keep this column but this column is not exactly the way we want it.
# 
# Machine Learning works best with numeric data so when we have a column that we want to use but only has string values then it would be best if we turn the column into numeric data.
# 
# In this case let's create a new column that make the Males 0 and the Females 1.

# In[ ]:


#map each Sex value to a numerical value
sex_mapping = {"male": 0, "female": 1}
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)


# In[ ]:


# Let's check our data again
train.head()


# We can now see that after our tranformation the 'Sex' column now only contains 0 and 1 and that is what we want.

# **Age**
# 
# This column obviously will prove to be useful but there are many problems with this column. This column is actually the main challenge for us to get an accurate prediction. From the information in the missing value section the 'Age' column is actually missing 177 values which is 20% of the entire column. Whenever we deal with missing values it is sugested that if possible we should never delete any data because they are simply missing values the hard part of machine learning is that we actually need to use the best way possible to try to accurately recreate these missing values. The biggest challenge with this data is how we fill out the missing data in the 'Age' column.

# **sibsp**
# 
# This Column shows us the number of siblings or spouse that a passenger has. We are not so sure if this column is useful or not so we need to visualize it first.

# In[ ]:


#draw a bar plot for SibSp vs. survival
sns.barplot(x="SibSp", y="Survived", data=train)

#I won't be printing individual percent values for all of these.
print("Percentage of SibSp = 0 who survived:", train["Survived"][train["SibSp"] == 0].value_counts(normalize = True)[1]*100)

print("Percentage of SibSp = 1 who survived:", train["Survived"][train["SibSp"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of SibSp = 2 who survived:", train["Survived"][train["SibSp"] == 2].value_counts(normalize = True)[1]*100)


# This is a very interesting development. In general people with more siblings are less likely to survive because they need to find and carry all the family members which will slow them down. The interesting part is people with 1 or 2 siblings or spouse actually have a higher chance of surviving compared to those with none. This column should be kept.

# **Parch**
# 
# This column shows the number of children or parents that passengers have. Let's see if this column is useful or not.

# In[ ]:


#draw a bar plot for Parch vs. survival
sns.barplot(x="Parch", y="Survived", data=train)
plt.show()


# This indicates that people with 1,2 and 3 children or parents have a higher chance of survival but those with none and with too much have a less chance of survival. We need to keep this column because it proves to be useful for our final predictions.

# **Ticket**
# 
# This column is alphanumerical data and this one is hard to actuallty gain any benefit from so we will simply remove this column.

# In[ ]:


# Remove the 'Ticket' column
train = train.drop(['Ticket'],axis = 1)
test = test.drop(['Ticket'],axis = 1)


# **Fare**
# 
# This is an interesting column. I believe that this column can be seen as the same as the 'Pclass' column and I think it should be removed. There are too many things that could affect the 'Fare' value but the 'Pclass' value is solid.

# In[ ]:


# Remove the 'Fare' column
train = train.drop(['Fare'],axis = 1)
test = test.drop(['Fare'],axis = 1)


# **Embarked**
# 
# We need to keep this column but from the missing values above we can see that it is missing 2 values. When we encounter situations like these where the column is missing values but not a lot. It is best to fill in the missing values with the most occured value.

# In[ ]:


# Check the occurance of each 'Embarked' value
print("S:")
s = train[train["Embarked"] == "S"].shape[0]
print(s)

print("C:")
c = train[train["Embarked"] == "C"].shape[0]
print(c)

print("Q:")
q = train[train["Embarked"] == "Q"].shape[0]
print(q)


# S has 644 values which is 72% of the total. 
# 
# Now that S is confirmed to be the most common value we can just fill the missing value in 'Embarked' with S.

# In[ ]:


#replacing the missing values in the Embarked feature with S
train = train.fillna({"Embarked": "S"})


# Now that we have filled in the missing values we need to alter the 'Embarked' column so it is easier for our model to use.

# In[ ]:


# map each Embarked value to a numerical value
embarked_mapping = {"S": 1, "C": 2, "Q": 3}
train['Embarked'] = train['Embarked'].map(embarked_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)

train.head()


# **Cabin**
# 
# Now let's look at the 'Cabin' column. From the missing data compared to the total we can see that there are 687 values missing from 891 total values in the 'Cabin' column. That means that there is 77% of values missing from this column. My initial thought was that this column is missing so much value that thi column isn't very useful for the model. After doing some research on some other Notebooks on this data others have argued that the people who have revorded values for their cabin have a higher chance of surviving. Lets see if this is true.

# We will now create a new column for the 'Cabin' column that will mark all the cabins that are recorded with 1 and all those that are not recorded with 0. This is important because it is much easier for us to operate on integers rather than strings and Machine Learning models work best with integers as well.

# In[ ]:


# To create a new column to identify Cabin types
train["CabinType"] = (train["Cabin"].notnull().astype('int'))
test["CabinType"] = (test["Cabin"].notnull().astype('int'))


# Now lets calculate the percentage of those with a recorded cabin vs those without a recorded cabin

# In[ ]:


# Calculate percentage of survival
print("Percentage of CabinType = 1 who survived:", train["Survived"][train["CabinType"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of CabinType = 0 who survived:", train["Survived"][train["CabinType"] == 0].value_counts(normalize = True)[1]*100)


# Those with a recorded Cabin have a 67% chance of surviving and those without only had a 30% chance of surviving. This can indicate that those with a recorded Cabin could have a higher social status or they have a better location for escape when accidents happen. So now the 'CabinType' is relavant to our final model but we can drop the 'Cabin' Column because it has served its purpose.

# In[ ]:


# Drop the 'Cabin' Column
train = train.drop(['Cabin'],axis = 1)
test = test.drop(['Cabin'],axis = 1)


# **Extract Information from Name**
# 
# Now let's see if we can extract some interesting information from the 'Name' column. What we really want is the titles of the people because they can someimtes indicate their age and social status.
# 
# We will now create a new column to do that.

# In[ ]:


#create a combined group of both datasets so it is easier to manage later
combine = [train, test]

#extract a title for each Name in the train and test datasets
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

# summary
pd.crosstab(train['Title'], train['Sex'])


# In[ ]:


#replace various titles with more common names
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',
    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    
    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# check survival rate for people with different titles
train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# Now we know that people with different titles have various rates of survival. This information could also be useful when we are trying to determine what the missing ages are.

# In[ ]:


#map each of the title groups to a numerical value
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train.head()


# Now we can drop the 'Name' column because we have all the information we need.

# In[ ]:


train = train.drop(['Name'], axis = 1)
test = test.drop(['Name'],axis = 1)


# **Fixing the Age**
# 
# Now we finally get to the most difficult part of this dataset: the missing age values. Before we head on and start trying to fill those out let's first give the 'Age' column less variaton.
# 
# Using the age value we will now alter the 'Age' column and create titles for people in different age groups such as YoungAdult, Adult etc.

# In[ ]:


#sort the ages into logical categories
train["Age"] = train["Age"].fillna(-0.5)
test["Age"] = test["Age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)
test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)

#draw a bar plot of Age vs. survival
sns.barplot(x="AgeGroup", y="Survived", data=train)
plt.show()


# Now using the 'Title' column that we have created we will determine what the missing ages are. We will fill the missing ages by using the mode of the age groups that have the same title as the missing value.

# In[ ]:


# fill missing age with mode age group for each title
mr_age = train[train["Title"] == 1]["AgeGroup"].mode() #Young Adult
miss_age = train[train["Title"] == 2]["AgeGroup"].mode() #Student
mrs_age = train[train["Title"] == 3]["AgeGroup"].mode() #Adult
master_age = train[train["Title"] == 4]["AgeGroup"].mode() #Baby
royal_age = train[train["Title"] == 5]["AgeGroup"].mode() #Adult
rare_age = train[train["Title"] == 6]["AgeGroup"].mode() #Adult

age_title_mapping = {1: "Young Adult", 2: "Student", 3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"}

for x in range(len(train["AgeGroup"])):
    if train["AgeGroup"][x] == "Unknown":
        train["AgeGroup"][x] = age_title_mapping[train["Title"][x]]
        
for x in range(len(test["AgeGroup"])):
    if test["AgeGroup"][x] == "Unknown":
        test["AgeGroup"][x] = age_title_mapping[test["Title"][x]]

train.head()


# I don't believe this is the best way to solve the missing values for 'Age' but that is the job for us. To keep on thinking and innovating and eventually come up with better ways to make the data better.

# In[ ]:


#map each Age value to a numerical value
age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}
train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
test['AgeGroup'] = test['AgeGroup'].map(age_mapping)

#drop the Age feature
train = train.drop(['Age'], axis = 1)
test = test.drop(['Age'], axis = 1)

train.head()


# **Choosing the best model**
# 
# Now is the most exciting part that all of you have been waiting for. The part where you finally get to build your own Machine Learning model!

# **Splitting the Training Data**
#     We will now split our training data to prevent overfitting and to test the accuracy of our different models. In this case we take 22% as out prediction data.

# In[ ]:


from sklearn.model_selection import train_test_split

predictors = train.drop(['Survived', 'PassengerId'], axis=1)
target = train["Survived"]
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.22, random_state = 0)


# **Testing Different Models**
# 
# There are many algorithmns that are very powerful and are already available for us to use.
# 
# Gaussian Naive Bayes
# Logistic Regression
# Support Vector Machines
# Perceptron
# Decision Tree Classifier
# Random Forest Classifier
# KNN or k-Nearest Neighbors
# Stochastic Gradient Descent
# Gradient Boosting Classifier
# 
# Now let's test them out.

# In[ ]:


# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gaussian)


# In[ ]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_logreg)


# In[ ]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_logreg)


# In[ ]:


# Support Vector Machines
from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_svc)


# In[ ]:


# Linear SVC
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_val)
acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_linear_svc)


# In[ ]:


# Perceptron
from sklearn.linear_model import Perceptron

perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_val)
acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_perceptron)


# In[ ]:


#Decision Tree
from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_decisiontree)


# In[ ]:


# Random Forest
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_randomforest)


# In[ ]:


# KNN or k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_knn)


# In[ ]:


# Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_val)
acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_sgd)


# In[ ]:


# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gbk)


# Now we create a list to give us a better visualization of the accuracy

# In[ ]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 'Linear SVC', 
              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],
    'Score': [acc_svc, acc_knn, acc_logreg, 
              acc_randomforest, acc_gaussian, acc_perceptron,acc_linear_svc, acc_decisiontree,
              acc_sgd, acc_gbk]})
models.sort_values(by='Score', ascending=False)


# A bit of an anti-climax wasn't it. You were all hyped up and ready to make your own algorithmns create your own mathematical functions. The truth is Machine Learning has been around longer than you think and there are already lots of great algorithmns ready for us to use and they are hard to beat.

# Now we use the #1 model to submit our results.

# In[ ]:


#set ids as PassengerId and predict survival 
ids = test['PassengerId']
predictions = svc.predict(test.drop('PassengerId', axis=1))

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('submission.csv', index=False)

