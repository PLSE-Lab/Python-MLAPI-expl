#!/usr/bin/env python
# coding: utf-8

# # Titanic is sinking. Lifeboats are limited and a decision need to be made. 
# 
# ![](https://www.google.com/url?sa=i&source=images&cd=&cad=rja&uact=8&ved=2ahUKEwip_fLdzarhAhVXcCsKHTwED3sQjRx6BAgBEAU&url=https%3A%2F%2Fwww.goodhousekeeping.com%2Flife%2Fg19809308%2Ftitanic-facts%2F&psig=AOvVaw2W19MLBaByqF5BcG5l0xLA&ust=1554060770519155)
# # What are those decisions? Let's see if we can get some insights from the data and replicate the thought process of 10th April, 1912.

# In[2]:


# Let's import libraries

# for data analysis 
import numpy as np
import pandas as pd

# for visualization
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# to ignore warnings
import warnings
warnings.filterwarnings('ignore')


# In[4]:


#import train and test CSV files
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[5]:


# Let's have a look at the data
train.head()


# In[6]:


train.info()


# * # Initial Observations:
# 
# 
# Passenger ID is an integer (Nominal data) with no random text to filter out
# 
# Survived is an integer (Nominal data) with only two values - 0 and 1
# 
# Pclass is an integer (Nominal data) with only three values - 1, 2 and 3
# 
# SibSp is an integer (Ordinal data) 
# 
# Parch is an integer (Ordinal data)
# 
# Age has some missing values 
# 
# Cabin has many missing values 
# 
# Embarked has a few missing values

# In[8]:


sex_vs_survived = pd.crosstab(train.Sex, train.Survived)
print(sex_vs_survived)
plt.figure(figsize=(25,10))
sex_vs_survived.plot.bar(stacked=True)


# # Insight 1: Female gender has better survival than male gender

# In[9]:


survived_vs_pclass = pd.crosstab([train.Pclass, train.Sex], train.Survived)
print(survived_vs_pclass)
plt.figure(figsize=(25,15))
survived_vs_pclass.plot.bar(stacked = True)


# # Insight 2: Females of first class and second class survived more than females of third class
# # Insight 3: Males of first class has better survival compared to males of second and third class

# In[10]:


sib = pd.crosstab([train.SibSp, train.Sex], train.Survived)
print(sib)
plt.figure(figsize=(25,10))
sib.plot.bar(stacked = True)


# # We can't find any pattern here, atleast visually

# In[11]:


emb = pd.crosstab(train.Embarked, train.Survived)
print(emb)
emb.plot.bar(stacked = True)


# # Can't say much but it looks like C has better chance of survival. Let's explore further

# In[12]:


emb1 = pd.crosstab([train.Embarked, train.Survived], train.Pclass)
print(emb1)
emb1.plot.bar(stacked = True)


# # Probably more first class tickets were sold in C as a percentage compared to Q and S stations which justifies higher survival in C

# In[13]:


train.head()


# # Let us now fill the missing age values in train and test data
# 

# We can fill age in two ways. 
# 
# 1. Take the mean age across the age column and fill it in missing values
# 
# 2. impute mean age with respect to title of the person (Mr. , Master, Miss, etc ) and fill them respectively. 
# 
# 
# Option two would give better age approximation, so let's go with that method 

# In[14]:


name_train = train['Name']
name_train['Title'] = 0
for i in train['Name']:
    name_train['Title']=train['Name'].str.extract('([A-Za-z]+)\.', expand=False)
    
name_test = test['Name']
name_test['Title'] = 0
for i in test['Name']:
    name_test['Title']=test['Name'].str.extract('([A-Za-z]+)\.', expand=False)


# In[15]:


print(name_train.Title.unique())
print(name_test.Title.unique())


# We can see same observations with different names such as Miss, Ms, Mlle, Lady. We may group them into familier titiles

# In[16]:


name_train['Title'] = name_train['Title'].replace(['Miss', 'Ms', 'Mlle', 'Lady'], 'Miss')
name_test['Title'] = name_test['Title'].replace(['Miss', 'Ms', 'Mlle', 'Lady'], 'Miss')
name_test['Title'] = name_test['Title'].replace('Dona', 'Don')


# In[17]:


print(name_train.Title.unique())
print(name_test.Title.unique())


# In[18]:


train['Title'] = name_train['Title']
test['Title'] = name_test['Title']

title_mean = train.groupby('Title')['Age'].mean()


# In[19]:


title_mean


# Let's map the above age values in the missing NAN columns in age of train and test data

# In[20]:


map_title_mean = title_mean.to_dict()
map_title_mean


# In[21]:


# fill missing values in the Age column according to title
train.Age = train.Age.fillna(train.Title.map(map_title_mean))
test.Age = test.Age.fillna(train.Title.map(map_title_mean))


# In[22]:


print(train.head(15))
print(test.head(15))


# In[23]:


train.info()


# In[24]:


test.info()


# ### Age column missing values are now filled with relevant data in train and test datasets

# # Now, Let's remove columns which are not useful in analysis.

# In[25]:


train.drop('Cabin', axis = 1, inplace = True)
train.drop('Name', axis = 1, inplace = True)
train.drop('Ticket', axis = 1, inplace = True)
train.drop('Fare', axis=1, inplace = True)

test.drop('Cabin', axis = 1, inplace = True)
test.drop('Name', axis = 1, inplace = True)
test.drop('Ticket', axis = 1, inplace = True)
test.drop('Fare', axis=1, inplace = True)


# In[26]:


print(train.head(10))
print(test.head(10))


# ### Unnecessary Columns are removed from the train and test dataframe

# In[28]:


title_survival = pd.crosstab(train.Title, train.Survived)
print(title_survival)

plt.figure(figsize=(25,10))
sns.barplot(x='Title', y='Survived', data = train)
plt.xticks(rotation=90);


# In[29]:


# peaks for survived/not survived passengers by their age
facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age', shade = True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()


# # First visualization say that women and children survive better than men and 
# 
# # second visualization probably speaks of people of age group 30 to 40 took the responsibility of making sure children and women are safe and children survive more and elderly don't have much of a chance

# # Let's mark some nominal values to Sex, Embarked and Title

# In[30]:


sex_mapping = {"male": 0, "female": 1}


# In[31]:


embarked_mapping = {'S':0, 'C':1, 'Q':2}


# In[32]:


title_mapping = {'Capt': 1,
 'Col': 2,
 'Countess': 3,
 'Don': 4,
 'Dr': 5,
 'Jonkheer': 6,
 'Major': 7,
 'Master': 8,
 'Miss': 9,
 'Mme': 10,
 'Mr': 11,
 'Mrs': 12,
 'Rev': 13,
 'Sir': 14}


# In[33]:


train['Sex'] = train['Sex'].map(sex_mapping)
train['Embarked'] = train['Embarked'].map(embarked_mapping)
train['Title'] = train['Title'].map(title_mapping)

test['Sex'] = test['Sex'].map(sex_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)
test['Title'] = test['Title'].map(title_mapping)


# In[34]:


print(train.head(10))
print(test.head(10))


# In[35]:


train.Embarked = train.Embarked.fillna(0)
test.Embarked = test.Embarked.fillna(0)


# In[36]:


test.head()


# ## converting float point variables to integer variables

# In[37]:


train.Age = pd.Series(train.Age).astype(int)
train.Embarked = pd.Series(train.Embarked).astype(int)

test.Age = pd.Series(test.Age).astype(int)
test.Embarked = pd.Series(test.Embarked).astype(int)


# In[38]:


train.info()
print(train.head(10))
print(test)


# # Let's see which ML model could replicate year 1912 decision better
# 

# In[44]:


from sklearn.model_selection import train_test_split


# In[45]:


predictors = train.drop(['Survived', 'PassengerId'], axis=1)
target = train["Survived"]
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.2, random_state = 0)


# In[47]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_logreg)


# In[48]:


# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gaussian)


# In[49]:


# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier(learning_rate = 0.05, n_estimators = 3000)
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gbk)


# In[50]:


# Support Vector Machines
from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_svc)


# In[51]:


# Linear SVC
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_val)
acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_linear_svc)


# In[52]:


# Perceptron
from sklearn.linear_model import Perceptron

perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_val)
acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_perceptron)


# In[53]:


#Decision Tree
from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_decisiontree)


# In[54]:


# Random Forest
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier(n_estimators = 1000)
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_randomforest)


# In[55]:


# KNN or k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_knn)


# In[56]:


# Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_val)
acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_sgd)


# In[57]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_logreg)


# In[58]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 'Linear SVC', 
              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],
    'Score': [acc_svc, acc_knn, acc_logreg, 
              acc_randomforest, acc_gaussian, acc_perceptron,acc_linear_svc, acc_decisiontree,
              acc_sgd, acc_gbk]})
models.sort_values(by='Score', ascending=False)


# As Logistic Regression gave better score, we shall extract the predictions from that data for competition

# In[59]:


#set ids as PassengerId and predict survival 
ids = test['PassengerId']
predictions = gbk.predict(test.drop('PassengerId', axis=1))

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('submission2.csv', index=False)


# ## It is done!

# Thank you very much for viewing my work. I am thankful to Kaggle community for teaching me many concepts. 
# 
# Credits: I am thankful to https://www.kaggle.com/rochellesilva/simple-tutorial-for-beginners , https://www.kaggle.com/nadintamer/titanic-survival-predictions-beginner and many others as I have understood their code and added my own insights to improve the model.

# In[ ]:




