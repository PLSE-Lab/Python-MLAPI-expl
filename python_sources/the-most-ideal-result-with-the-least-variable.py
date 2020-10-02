#!/usr/bin/env python
# coding: utf-8

# ### 1) Import Necessary Libraries

# In[ ]:


# data analysis libraries:
import numpy as np
import pandas as pd

# data visualization libraries:
import matplotlib.pyplot as plt
import seaborn as sns

# to ignore warnings:
import warnings
warnings.filterwarnings('ignore')

# # to display all columns:
# pd.set_option('display.max_columns', None)

from sklearn.model_selection import train_test_split, GridSearchCV


# ### 2) Read in and Explore the Data

# In[ ]:


#import train and test CSV files
train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")
#create a copy train
df= train.copy()

#create a combined group of both datasets
combine = [df, test]


# ### 3) Data Analysis

# In[ ]:


#take a look at the training data
df.describe().T


# In[ ]:


(df.columns)


# In[ ]:


test.info()


# Some Observations:
# 1. There are a total of 891 passengers in our training set.
# 2. The Age feature is missing approximately 19.8% of its values. I'm guessing that the Age feature is pretty important to survival, so we should probably attempt to fill these gaps.
# 3. The Cabin feature is missing approximately 77.1% of its values. Since so much of the feature is missing, it would be hard to fill in the missing values. We'll probably drop these values from our dataset.
# 4. The Embarked feature is missing 0.22% of its values, which should be relatively harmless.

# In[ ]:


df.isnull().sum()


# In[ ]:


test.isnull().sum()


# Some Predictions:
# 1. Sex: Females are more likely to survive.
# 2. SibSp/Parch: People traveling alone are more likely to survive.
# 3. Age: Maybe young children are more likely to survive. Maybe it is not important.
# 4. Pclass: People of higher socioeconomic class are more likely to survive.
# 5. Embarked: It is observed that most of the survivors boarded from the S port.
# 6. Fare: I mean its very important.

# In[ ]:


print(df['Pclass'].value_counts())
print(df['SibSp'].value_counts())
print(df['Parch'].value_counts())
print(df['Embarked'].value_counts())
print(df['Sex'].value_counts())
print(df['Ticket'].value_counts())
print(df['Cabin'].value_counts())


# ## 4)Data Visualization

# #### Sex Feature

# In[ ]:


#draw a bar plot of survival by sex
sns.barplot(x="Sex", y="Survived", data=df);


# As predicted, females have a much higher chance of survival than males. The Sex feature is essential in our predictions.

# #### Pclass Feature

# In[ ]:


#draw a bar plot of survival by Pclass
sns.barplot(x="Pclass", y="Survived", data=df);


# As predicted, people with higher socioeconomic class had a higher rate of survival. (62.9% vs. 47.3% vs. 24.2%)

# #### SibSp Feature

# In[ ]:


#draw a bar plot for SibSp vs. survival
sns.barplot(x="SibSp", y="Survived", data=df);


# In[ ]:


df.groupby('SibSp')['PassengerId'].count()


# In general, it's clear that people with more siblings or spouses aboard were less likely to survive. However, contrary to expectations, people with no siblings or spouses were less to likely to survive than those with one or two. (34.5% vs 53.4% vs. 46.4%)

# #### Parch Feature

# In[ ]:


sns.barplot(x="Parch", y="Survived", data=df)
plt.show()


# In[ ]:


df.groupby('Parch')['PassengerId'].count()


# #### Fare Feature

# In[ ]:


sns.boxplot(x = df['Fare']);


# ### 5) Cleaning Data & Outlier Treatment & Feature Engeneering & Predict Missing Values

# Time to clean our data to account for missing values and unnecessary information!

# let's see how our test data looks!

# In[ ]:


test.describe(include="all")


# - We have a total of 418 passengers.
# - 1 value from the Fare feature is missing.
# - Around 20.5% of the Age feature is missing, we will need to fill that in.

# #### Parch Feature

# The number of people with two or more children or parents is very low in the data. Adding them separately to the algorithm can lead to misleading results. So I decided to examine two or more people as a single class.
# 
# Besides I accept the Pclass variable as a categorical ordinal variable.

# In[ ]:


df['Parch'] =  df['Parch'].replace([df.loc[df.Parch>1,'Parch'].values], 2)
test['Parch'] =  test['Parch'].replace([df.loc[df.Parch>1,'Parch'].values], 2)


# In[ ]:


from pandas.api.types import CategoricalDtype 
df['Parch'] = df['Parch'].astype(CategoricalDtype(ordered = True))
test['Parch'] = test['Parch'].astype(CategoricalDtype(ordered = True))


# In[ ]:


df.groupby(['Parch'])['PassengerId'].count()


# #### SibSp Feature

# The number of people with two or more siblings or spouses in the data is very low. Adding them separately to the algorithm can lead to misleading results. So I decided to examine two or more people as a single class.
# 
# I accept the SibSp variable as a categorical ordinal variable.

# In[ ]:


test.groupby(['SibSp'])['PassengerId'].count()


# In[ ]:


df['SibSp'] =  df['SibSp'].replace([df.loc[df.SibSp>1,'SibSp'].values], 2)
test['SibSp'] =  test['SibSp'].replace([df.loc[df.SibSp>1,'SibSp'].values], 2)


# In[ ]:


df['SibSp'] = df['SibSp'].astype(CategoricalDtype(ordered = True))
test['SibSp'] = test['SibSp'].astype(CategoricalDtype(ordered = True))


# In[ ]:


sns.barplot(x="SibSp", y="Survived", data=df);


# #### Cabin Feature

# I think the idea here is that people with recorded cabin numbers are of higher socioeconomic class, and thus more likely to survive. Therefore, I think that the cabin and Pclass columns contain the same information.And the cabin column does not express meaningful information by itself. So I drop it.

# In[ ]:


df[df.Cabin.notnull()].groupby('Pclass')['PassengerId'].count()


# In[ ]:


df = df.drop(['Cabin'], axis = 1)
test = test.drop(['Cabin'], axis = 1)


# #### Ticket Feature

# In[ ]:


#we can also drop the Ticket feature since it's unlikely to yield any useful information
df = df.drop(['Ticket'], axis = 1)
test = test.drop(['Ticket'], axis = 1)


# #### Embarked Feature

# It's clear that the majority of people embarked in Southampton (S). Let's go ahead and fill in the missing values with S.

# In[ ]:


#now we need to fill in the missing values in the Embarked feature

df['Embarked'].fillna(df.Embarked.mode()[0], inplace = True)


# #### Fare Feature

# I noticed that there are outliers in the Fare variable. I made a few observations below to be able to optimize them. I tried to organize these outliers in the most reasonable way in line with my own ideas.

# In[ ]:


# It looks like there is a problem in Fare max data. Visualize with boxplot.
sns.boxplot(x = df['Fare']);


# In[ ]:


Q1 = train['Fare'].quantile(0.25)
Q3 = train['Fare'].quantile(0.75)
IQR = Q3 - Q1

lower_limit = Q1- 1.5*IQR
print(lower_limit)

upper_limit = Q3 + 1.5*IQR
print(upper_limit)

print(df[df.Fare<upper_limit]['Fare'].describe())
print(df[df.Fare>upper_limit]['Fare'].describe())
print(df[df.Fare<150]['Fare'].describe())
print(df[df.Fare>150]['Fare'].describe())


# It's time separate the fare values into some logical groups as well as filling in the single missing value in the test dataset.

# In[ ]:


for x in range(len(test["Fare"])):
    if pd.isnull(test["Fare"][x]):
        pclass = test["Pclass"][x] #Pclass = 3
        test["Fare"][x] = df[df.Pclass==pclass]['Fare'].median()

        
# I replace values greater than 150 with the median of values greater than 150.

df['Fare'] = df['Fare'].replace(df[df.Fare>150]['Fare'], df[df.Fare>150]['Fare'].median())
test['Fare'] = test['Fare'].replace(test[test.Fare>150]['Fare'], test[test.Fare>150]['Fare'].median())


# In[ ]:


# df['FareBand'] = pd.qcut(df['Fare'], 4, labels = [1, 2, 3,4])
# test['FareBand'] = pd.qcut(test['Fare'], 4, labels = [1, 2, 3,4])


binss = [-1,8, 31, 100,np.inf]
labelss = [ 1,2,3,4]
df['FareBand'] = pd.cut(df["Fare"], binss, labels = labelss)
test['FareBand'] = pd.cut(test["Fare"], binss, labels = labelss)


df['FareBand'] = df['FareBand'].astype(CategoricalDtype(ordered = True))
test['FareBand'] = test['FareBand'].astype(CategoricalDtype(ordered = True))

sns.barplot(x="FareBand", y="Survived", data=df)
plt.show()


# In[ ]:


df.groupby('FareBand')['Fare'].mean()


# #### Title Feature

# In[ ]:


#extract a title for each Name in the train and test datasets
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
test['Title'] = test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False) 

pd.crosstab(df['Title'], df['Sex'])


# In[ ]:


#replace various titles with more common names

#for df
df['Title'] = df['Title'].replace(['Lady', 'Capt', 'Col', 'Don',
                                                 'Dr', 'Major', 'Rev', 'Jonkheer', 
                                                 'Dona','Countess', 'Lady', 'Master',
                                                 'Sir'], 'Rare')
df['Title'] = df['Title'].replace(['Mlle', 'Ms'],'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
#for test    
test['Title'] = test['Title'].replace(['Lady', 'Capt', 'Col', 'Don',
                                                 'Dr', 'Major', 'Rev', 'Jonkheer', 
                                                 'Dona','Countess', 'Lady', 'Master',
                                                 'Sir'], 'Rare')
test['Title'] =test['Title'].replace(['Mlle', 'Ms'],'Miss')
test['Title'] = test['Title'].replace('Mme', 'Mrs')


df.Title = pd.Categorical(df.Title)
test.Title = pd.Categorical(test.Title)    

df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# #### Age Feature

# Next we'll fill in the missing values in the Age feature. Since a higher percentage of values are missing, it would be illogical to fill all of them with the same value (as we did with Embarked). Instead, let's try to find a way to predict the missing ages.
# 
# I obtained a median age value by grouping some information of people with age information. Then, I tried to make a prediction by matching the information I grouped with the information of the people whose age is empty. I assigned a median value to several units that I couldn't guess.

# In[ ]:


#I tried to choose the ones that gave the most optimum result.

age_guess = pd.DataFrame(df.groupby(['Title','SibSp','Pclass','Parch','FareBand','Embarked'], as_index=False)['Age'].median())
df_age_new = df.merge(age_guess, on =['Title','SibSp','Pclass','Parch' ,'FareBand','Embarked'], how='inner')
df_age_new['AgeErrors'] = (df_age_new['Age_x']-df_age_new['Age_y']).abs()
df_age_new['AgeErrors'].describe()


# In[ ]:





# In[ ]:


null_Age = df.drop(df[df.Age.notnull()].index, axis=0)
notnull_Age = df.drop(df[df.Age.isnull()].index, axis=0)

age_guess = pd.DataFrame(df.groupby(['Title','SibSp','Pclass','Parch' ,'FareBand','Embarked'], as_index=False)['Age'].median())
null_Age = null_Age.merge(age_guess, on =['Title','SibSp','Pclass', 'Parch','FareBand','Embarked'], how='inner')

null_Age['Age_y']= null_Age['Age_y'].fillna(null_Age['Age_y'].median())

null_Age= null_Age.drop('Age_x', axis=1 ).rename(columns= {"Age_y": "Age"})
df = pd.concat([null_Age,notnull_Age], axis=0, ignore_index = True)


# In[ ]:


null_Age = test.drop(test[test.Age.notnull()].index, axis=0)
notnull_Age = test.drop(test[test.Age.isnull()].index, axis=0)

age_guess = pd.DataFrame(test.groupby(['Title','SibSp','Pclass','Parch' ,'FareBand','Embarked'], as_index=False)['Age'].median())
null_Age = null_Age.merge(age_guess, on =['Title','SibSp','Pclass', 'Parch','FareBand','Embarked'], how='inner')

null_Age['Age_y']= null_Age['Age_y'].fillna(null_Age['Age_y'].median())

null_Age= null_Age.drop('Age_x', axis=1 ).rename(columns= {"Age_y": "Age"})
test = pd.concat([null_Age,notnull_Age], axis=0, ignore_index = True)


# In[ ]:


#sort the ages into logical categories

bins = [0, 5,  64,  np.inf]
labels = [3,4,5]
df['AgeGroup'] = pd.cut(df["Age"], bins, labels = labels)
test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)


df['AgeGroup'] = df['AgeGroup'].astype(CategoricalDtype(ordered = True))
test['AgeGroup'] = test['AgeGroup'].astype(CategoricalDtype(ordered = True))

#draw a bar plot of Age vs. survival
sns.barplot(x="AgeGroup", y="Survived", data=df)
plt.show()


# Babies are more likely to survive than any other age group. However, seniors is unlucky than other groups to survive.

# #### Name Feature
# We can drop the name feature now that we've extracted the titles.

# In[ ]:


#drop the name feature since it contains no more useful information.
df = df.drop(['Name'], axis = 1)
test = test.drop(['Name'], axis = 1)


# #### Fare Features

# I dropped it because the Fare column reduces the success values of the algorithms.

# In[ ]:


df = df.drop(['Fare'], axis = 1)
test = test.drop(['Fare'], axis = 1)


# #### One-Hot Encoding

# I converted non ordinal categorical variables to dummy variables. In doing so, I dropped the variables containing the same information to avoid falling into the dummy variable trap.

# In[ ]:


df = pd.get_dummies(df, columns = ["Pclass"], prefix = ["Pclass"], drop_first= True)
test = pd.get_dummies(test, columns = ["Pclass"], prefix = ["Pclass"], drop_first= True)


# In[ ]:


df = pd.get_dummies(df, columns = ["Embarked"], prefix = ["Embarked"], drop_first= True)
test = pd.get_dummies(test, columns = ["Embarked"], prefix = ["Embarked"], drop_first= True)


# In[ ]:


df = pd.get_dummies(df, columns = ["Title"], prefix = ["Title"], drop_first= True)
test = pd.get_dummies(test, columns = ["Title"], prefix = ["Title"], drop_first= True)


# In[ ]:


df = pd.get_dummies(df, columns = ["Sex"], prefix = ["Sex"], drop_first= True)
test = pd.get_dummies(test, columns = ["Sex"], prefix = ["Sex"], drop_first= True)


# ### 6) Modeling, Evaluation and Model Tuning

# #### Splitting the Training Data
# We will use part of our training data (22% in this case) to test the accuracy of our different models.

# In[ ]:


from sklearn.model_selection import train_test_split

predictors = df.drop(['Survived', 'PassengerId'], axis=1)
target = df["Survived"]
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.22, random_state = 0)


# Testing Different Models
# I will be testing the following models with my training data (got the list from here):
# 
# - Logistic Regression
# - Support Vector Machines
# - Linear SVC
# - Stochastic Gradient Descent
# 
# For each model, we set the model, fit it with 80% of our training data, predict for 20% of the training data and check the accuracy.

# In[ ]:


# Logistic Regression82.74
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_logreg)


# In[ ]:


# Support Vector Machines82.23,
from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_svc)


# In[ ]:


# Linear SVC 82.23, 62.44
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_val)
acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_linear_svc)


# In[ ]:


# Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_val)
acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_sgd)


# In[ ]:


#MLPClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()

scaler.fit(x_train)
X_train_scaled = scaler.transform(x_train)
X_test_scaled = scaler.transform(x_val)

mlpc = MLPClassifier().fit(X_train_scaled, y_train)
y_pred = mlpc.predict(X_test_scaled)
acc_mlpc = round(accuracy_score(y_val, y_pred)*100, 2)
print(acc_mlpc)


# ### Let's compare the accuracies of each model!

# In[ ]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines',  'Logistic Regression', 
               'Linear SVC',  'MLPClassifier' ,'Stochastic Gradient Descent'],
    'Score': [acc_svc, acc_logreg, 
              acc_linear_svc,acc_mlpc,
              acc_sgd]})
models.sort_values(by='Score', ascending=False)


# I decided to use the Multiple Layer Perceptron model for the testing data.

# ### 7) Creating Submission File
# 
# It's time to create a submission.csv file to upload to the Kaggle competition!

# In[ ]:


#set ids as PassengerId and predict survival 
ids = test['PassengerId']
predictions = mlpc.predict(test.drop('PassengerId', axis=1))

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('submission33.csv', index=False)


# In[ ]:




