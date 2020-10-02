#!/usr/bin/env python
# coding: utf-8

# I rewrite the kernel of [EDA To Prediction (DieTanic)](https://www.kaggle.com/ash316/eda-to-prediction-dietanic) made by [Ashwini Swain](https://www.kaggle.com/ash316), which is awesome as belows:

# # EDA To Prediction (DieTanic)
# 
# *Sometimes life has a cruel sense of humor, giving you the the thing you always wanted at the worst time possible.*
# 
# The sinking of the Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. That's why the name DieTanic. This is a very unforgetable disaster that no one in the world can forget. 
# 
# It took about $7.5 million to build the Titanic and it sunk under the ocean due to collision. The Titanic Dataset is a very good dataset for beginners to start a journey in data science and participate in competition in Kaggle. 
# 
# The objective of this notebook is to give an idea how is the workflow in any predictive modeling problem. How do we check features, how do we add new features and some Machine Learning Concepts. I have tried to keep the notebook as basic as possible so that even newbies can understand every phase of it. 
# 
# If you lie the notebook and think that it helped you, Please UPVOTE. It will keep me motivated. 
# 
# **Contents of the Notebook:**
# 
# **Part1: Exploratory Data Analysis (EDA)**
# 
# 1) Analysis of the features
# 
# 2) Finding any relations or trends considering multiple features. 
# 
# **Part2: Feature Engineering and Data Cleaning**
# 
# 1) Adding any few features.
# 
# 2) Removing redundant features. 
# 
# 3) Converting features into suitable form for modeling
# 
# **Part3: Predictive Modeling**
# 
# 1) Runnding Basick Algorithm
# 
# 2) Cross Validation
# 
# 3) Ensembling
# 
# 4) Important Features Extraction

# ## Part1: Exploratory Data Analysis (EDA)

# In[ ]:


import numpy as np
import pandas as pd
from pandas import Series
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')
sns.set(font_scale=2.5)
import warnings
warnings.filterwarnings('ignore')

import os
print(os.listdir("../input"))

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv('../input/train.csv')


# In[ ]:


data.head()


# **Check for total null values**

# In[ ]:


data.isnull().sum()


# The Age, Cabin and Embarked have null values. I will try to fix them. 

# **How many Survived?**

# In[ ]:


f, ax = plt.subplots(1,2,figsize=(18,8))
data['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot('Survived', data=data, ax=ax[1])
ax[1].set_title('Survived')
plt.show()


# It is evident that not many passengers survived the accident. 
# 
# Out of 891 passengers in training set, only around 350 survived i. only 38.4% of the total training set survived the crash. We need to dig down more to get better insights from the data and see which categories of the passengers did survive and who didn't.
# 
# We will try to check the survival rate by using the different features of the dataset. Some of the features being Sex, Port of Embarcation, Age, etc. 
# 
# First, let us understand the differnt types of features. 
# 
# **Types of Features**
# 
# **Categorical Features:**
# 
# A categorical variable is one that has two or more categories and each value in that feature can be categorized by them. For example, gender is a categorical variable having two categories (male and female). Now we cannot sort of give any ordering to such variables. They are also known as Nominal Variables. 
# 
# Categorical Features in the dataset: Sex, Embarked.
# 
# **Ordinal Features:**
# 
# An ordinal variable is similar to categorical values, but the difference between them is that we can have relative ordering or sorting between the values. For ex: If we have a feature like Height with values Tall, Medium, Short, then Height is an ordinal variable. Here we can have a relative sort in the variable. 
# 
# Oridinal Features in the dataset: Pclass
# 
# **Continuous Features:**
# 
# A feature is said to be continuous if it can take values between any two points or between the minimum or maximum values in the feature column. 
# 
# Continuous Featrues in the dataset: Age
# 

# **Analysing the Features**
# 
# **Sex --> Categorical Feature**

# In[ ]:


data.groupby(['Sex', 'Survived'])['Survived'].count()


# In[ ]:


f, ax = plt.subplots(1,2,figsize=(18,8))
data[['Sex', 'Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Sex')
sns.countplot('Sex', hue='Survived', data=data, ax=ax[1])
ax[1].set_title('Sex: Survived vs Dead')
plt.show()


# This looks interesting. The number of men on the ship is lot more than the number of women. Still the number of women saved is almost twice the number of males saved. The survival rates for women on the ship is around 75% while that for men is around 18-19%.
# 
# This lokks to be a very important feature for modeling. But is it the best? Let's check other features. 

# **Pclass --> Ordinal Feature**

# In[ ]:


pd.crosstab(data['Pclass'], data['Survived'], margins=True).style.background_gradient(cmap='summer_r')


# In[ ]:


f, ax = plt.subplots(1,2,figsize=(18,8))
data['Pclass'].value_counts().plot.bar(color=['#CD7F32', '#FFDF00', '#D3D3D3'], ax=ax[0])
ax[0].set_title('Number of Passengers by Pclass')
ax[0].set_ylabel('Count')
sns.countplot('Pclass', hue='Survived', data=data, ax=ax[1])
ax[1].set_title('Pclass: Survived vs Dead')
plt.show()


# People say Money Can't Buy Everything. But we can clearly see that Passengers of Pclass 1 were given a very high priority while rescue. Even though the number of Passengers in Pclass 3 were a lot higher, still the number of survival from them is very low, somewhere around 25%.
# 
# The survival rate of Pclass1 is around 63% while that of Pclass 2 is around 48%. So money and status matters. Such a materialistic world. 
# 
# Let's dive in a little bit more and check for other interesting observations. Let's check survival rate with Sex and Pclass Together. 

# In[ ]:


pd.crosstab([data['Sex'], data['Survived']], data['Pclass'], margins=True).style.background_gradient(cmap='summer_r')


# In[ ]:


sns.factorplot('Pclass', 'Survived', hue='Sex', data=data)
plt.show()


# We use FactorPlot in this case, because they make the seperation of categorical values easy. 
# 
# Looking at the CrossTab and the FactorPlot, we can easily infer that survival for Women from Pclass 1 is about 95-96%, as only 3 out of 94 Women from Pclass 1 died. 
# 
# It is evident that regardless of Pclass, Women were given first priority while rescue. Let's analyze other features. 

# **Age --> Continuous Feature**

# In[ ]:


print('Oldest Passenger was : {:.2f} Years'.format(data['Age'].max()))
print('Youngest Passenger was : {:.2f} Years'.format(data['Age'].min()))
print('Average Age on the ship was : {:.2f} Years'.format(data['Age'].mean()))


# In[ ]:


f, ax = plt.subplots(1,2,figsize=(18,8))
sns.violinplot('Pclass', 'Age', hue='Survived', data=data, split=True, scale='count', ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0,110,10))
sns.violinplot('Sex', 'Age', hue='Survived', data=data, split=True, scale='count', ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0,110,10))


# Observations:
# 
# 1) The number of children increases with Pclass and the survival rate for passengers below Age 10(i.e children) looks to be good irrespective of the Pclass.
# 
# 2) Survival chances for Passengers aged 20-50 from Pclass 1 is high and is even better for Women. 
# 
# 3) For males, the survival chances decreases with an increase in age. 
# 
# As we had seen earlier, the Age feature has 177 null values. To replace these NaN values, we can assign them the mean age of the dataset. 
# 
# But the problem is, there were many people with many different ages. We just can't assign a 4 year kid with the mean age that is 29 years. Is there any way to find out what age-band does the passenger lie?
# 
# Bingo!! We can check the Name feature. Looking upon the feature, we can see that the names have a salutation like Mr or Mrs. Thus we can assign the man values of Mr and Mrs to the respective groups. 
# 
# "What is in a Name??"

# In[ ]:


data['Initial'] = 0
data['Initial'] = data['Name'].str.extract('([A-Za-z]+)\.')


# In[ ]:


data.head()


# In[ ]:


pd.crosstab(data['Initial'], data['Sex']).T.style.background_gradient(cmap='summer_r')


# There are some misspelled initials like Mlle or Mme that stand for Miss. I will replace them with Miss and same thing for other values. 

# In[ ]:


data['Initial'].replace(['Mlle', 'Mme', 'Ms', 'Dr', 'Major', 'Lady', 'Countess',                         'Jonkheer', 'Col', 'Rev', 'Capt', 'Sir','Don'],                       ['Miss', 'Miss', 'Miss', 'Mr', 'Mr', 'Mrs', 'Mrs', 'Other', 'Other',                       'Other', 'Mr', 'Mr', 'Mr'], inplace=True)


# In[ ]:


data.groupby('Initial')['Age'].mean()


# Filling NaN Ages

# In[ ]:


data.loc[(data['Age'].isnull()) & (data['Initial']=='Mr'), 'Age'] = 33
data.loc[(data['Age'].isnull()) & (data['Initial']=='Mrs'), 'Age'] = 36
data.loc[(data['Age'].isnull()) & (data['Initial']=='Master'), 'Age'] = 5
data.loc[(data['Age'].isnull()) & (data['Initial']=='Miss'), 'Age'] = 22
data.loc[(data['Age'].isnull()) & (data['Initial']=='Other'), 'Age'] = 46


# In[ ]:


data['Age'].isnull().any()


# In[ ]:


f, ax = plt.subplots(1,2,figsize=(20,10))
data[data['Survived']==0]['Age'].plot.hist(ax=ax[0], bins=20, edgecolor='yellow', color='red')
ax[0].set_title('Survived=0')
x1=list(range(0,85,10))
ax[0].set_xticks(x1)
data[data['Survived']==1]['Age'].plot.hist(ax=ax[1], bins=20, edgecolor='black', color='green')
ax[1].set_title('Survived=1')
ax[1].set_xticks(x1)
plt.show()


# **Observations:**
# 
# 1) The toddlers(age<5) were saved in large numbers(the Women and Child First Policy)
# 
# 2) The oldest Passenger was saved(80years).
# 
# 3) Maximum number of deaths were in the age group of 30-40.

# In[ ]:


sns.factorplot('Pclass', 'Survived', col='Initial', data=data)
plt.show()


# The Women and Child First Policy thus holds true irrespective of class.

# **Embarked --> Categorical Value**

# In[ ]:


pd.crosstab([data['Embarked'], data['Pclass']], [data['Sex'], data['Survived']], margins=True).style.background_gradient(cmap='summer_r')


# Chances for Survival by Embarked

# In[ ]:


sns.factorplot('Embarked', 'Survived', data=data)
fig=plt.gcf()
fig.set_size_inches(5,3)
plt.show()


# The chances for survival for Port C is highest around 0.55 while it is lowest for S.

# In[ ]:


f, ax = plt.subplots(2,2,figsize=(20,15))
sns.countplot('Embarked', data=data, ax=ax[0,0])
ax[0,0].set_title('No. of Passengers Boarded')
sns.countplot('Embarked', hue='Sex', data=data, ax=ax[0,1])
ax[0,1].set_title('Male-Female Split for Embarked')
sns.countplot('Embarked', hue='Survived', data=data, ax=ax[1,0])
ax[1,0].set_title('Embarged vs Survived')
sns.countplot('Embarked', hue='Pclass', data=data, ax=ax[1,1])
ax[1,1].set_title('Embarked vs Pclass')
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()


# **Observations:**
# 
# 1) Maximum passengers boarded from S. Majority of them being from Pclass 3. 
# 
# 2) The Passengers from C look to be lucky as a good proportion of them survived. The reason for this maybe the rescue of all the Pclass 1 and Pclass 2 Passengers. 
# 
# 3) The Embark S looks to the port from where majority of the rich people boarded. Still the chances for survival is low here, that is because many passengers from Pclass 3 around 81% didn't survive. 
# 
# 4) Port Q had almost 95% of the passengers were from Pclass 3. 

# In[ ]:


sns.factorplot('Pclass', 'Survived', hue='Sex', col='Embarked', data=data)
plt.show()


# **Observations:**
# 
# 1) The survival chances are almost 1 for women for Pclass 1 and Pclass 2 irrespective of the Pclass. 
# 
# 2) Port S looks to be very unlucky for Pclass 3 Passengers as the survival rate for both men and women is very low. (Money matters)
# 
# 3) Port Q looks to be unluckies for Men, as almost all were from Pclass 3. 

# **Filling Embarked NaN**
# 
# As we saw that maximum passengers boarded from Port S, we replace NaN with S. 

# In[ ]:


data['Embarked'].fillna('S', inplace=True)


# In[ ]:


data['Embarked'].isnull().any()


# **SibSp --> Discriete Feature**

# In[ ]:


pd.crosstab(data['SibSp'], data['Survived'], margins=True).style.background_gradient(cmap='summer_r')


# In[ ]:


f, ax = plt.subplots(1,2,figsize=(20,8))
sns.barplot('SibSp', 'Survived', data=data, ax=ax[0])
ax[0].set_title('SibSp vs Survived')
sns.factorplot('SibSp', 'Survived', data=data, ax=ax[1])
ax[1].set_title('SibSp vs Survived')
plt.close(2)
plt.show()


# In[ ]:


pd.crosstab(data['SibSp'], data['Pclass']).style.background_gradient(cmap='summer_r')


# **Observations:**
# 
# The barplot and factorplot shows that if a passenger is alone onboard with no siblings, he have 34.5% survival rate. The graph roughly decreases if the number of siblings increase. This makes sense. That is, if I have a family on board, I will try to save them instead of saving myself first. Surprisingly the survival for families with 5-8 members is 0%. The reason may be Pclass??
# 
# The reason is Pclass. The crosstab shows that Person with SibSp > 3 were all in Pclass 3. It is imminent that all the large families in Pclass(>) died. 

# **Parch**

# In[ ]:


pd.crosstab(data['Parch'], data['Pclass']).style.background_gradient(cmap='summer_r')


# The crosstab again shows that larger families were in Pclass 3. 

# In[ ]:


f, ax = plt.subplots(1,2,figsize=(20,8))
sns.barplot('Parch', 'Survived', data=data, ax=ax[0])
ax[0].set_title('Parch vs Survived')
sns.factorplot('Parch', 'Survived', data=data,ax=ax[1])
ax[1].set_title('Parch vs Survived')
plt.close(2)
plt.show()


# **Observations:**
# 
# Here too the results are quite similar. Passengers with their parents onboard have greater chance of survival. It however reduces as the number goes up. 
# 
# The chances of survival is good for somebody who has 1-3 parents on the ship. Being alone also proves to be fatal and the chances for survival decreases when somebody has >4 parents on the ship.

# **Fare --> Continous Feature**

# In[ ]:


print('Highest Fare was {:.2f}'.format(data['Fare'].max()))
print('Lowest Fare was {:.2f}'.format(data['Fare'].min()))
print('Average Fare was {:.2f}'.format(data['Fare'].mean()))


# The lowest fare is 0.0. Wow! a free luxorious ride!

# In[ ]:


f, ax = plt.subplots(1,3,figsize=(20,8))
sns.distplot(data[data['Pclass']==1]['Fare'], ax=ax[0])
ax[0].set_title('Fares Pclass 1')
sns.distplot(data[data['Pclass']==2]['Fare'], ax=ax[1])
ax[1].set_title('Fares Pclass 2')
sns.distplot(data[data['Pclass']==3]['Fare'], ax=ax[2])
ax[2].set_title('Fares Pclass 3')
plt.show()


# There looks to be a large distribution in the fares of Passengers in Pclass 1 and this distribution goes on decreasing as the standards reduces. As this is also continous, we can convert into discrete values by using binning. 

# **Observations in a Nutshell for all features:**
# 
# Sex: The chance of survival for women is high as compared to men. 
# 
# Pclass: There is visible trend that being a 1st class passenger gives you better chances of survival. The survival rate for Pclass3 is very low. For women, the chance of survival from Pclass1 is almost 1 and is high too for those from Pclass2. Money Wins!!!
# 
# Age: Children less than 5-10 years do have a high chance of survival. Passengers between age group 15 to 35 died a lot. 
# 
# Embarked: This is a very interesting feature. The chances of survival at C looks to be better than even though the majority of Pclass1 passengers got up at S. Passengers at Q were all from Pclass3. 
# 
# Parch+SibSp: Having 1-2 siblings, spouse on board or 1-3 parents shows a greater chance of probability rather than being alone or having a large family travelling with you. 

# **Correlation Between the Features**

# In[ ]:


sns.heatmap(data.corr(), annot=True, cmap='RdYlGn', linewidth=0.2)
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()


# **Interpreting the Heatmap**
# 
# From the above heatmap, we can see that the features are not much correlated. The highest correlation is between SibSp and Parch i.e 0.41. So we can carry on with all features. 

# ## Part2: Feature Engineering and Data Clearning

# **Age_band**
# 
# As I have mentioned earlier that Age is a continuous feature, there is a problem with Continuous Variables in Machine Learning Models. 
# 
# We need to convert these continuous values into categorical values by either binning or normalization. I will using binning i.e group a range of ages into a single bin or assign them a single value. 
# 
# The maximum age of a passenger was 80. So let's divide the range from 0-80 into 5 bins. So 80/5-16. So bins of size 16.

# In[ ]:


data['Age_band']=0
data.loc[data['Age']<=16, 'Age_band']=0
data.loc[(16<data['Age']) & (data['Age']<=32), 'Age_band']=1
data.loc[(32<data['Age']) & (data['Age']<=48), 'Age_band']=2
data.loc[(48<data['Age']) & (data['Age']<=64), 'Age_band']=3
data.loc[64<data['Age'], 'Age_band']=4
data.head()


# In[ ]:


data['Age_band'].value_counts()


# In[ ]:


sns.factorplot('Age_band', 'Survived', col='Pclass', data=data)
plt.show()


# Ture that... the survival rate decreases as the age increases irrespective of the Pclass.

# **Family_size and Alone**
# 
# At this point, we can create a new feature called 'Family_size' and 'Alone' and analyze it. This feature is the summation of Parch and SibSp. It gives us a combined data so that we can check if survival rate have anything to do with family size of the passengers. Alone will denote whether a passenger is alone or not. 

# In[ ]:


data['Family_size']=0
data['Family_size']=data['SibSp']+data['Parch']
data['Alone']=0
data.loc[data['Family_size']==0, 'Alone']=1


# In[ ]:


f, ax = plt.subplots(1,2, figsize=(18,6))
sns.factorplot('Family_size', 'Survived', data=data, ax=ax[0])
ax[0].set_title('Family_size vs Survived')
sns.factorplot('Alone', 'Survived', data=data, ax=ax[1])
ax[1].set_title('Alone vs Survived')
plt.close(2)
plt.close(3)
plt.show()


# Family_Size=0 means that the passenger is alone. Clearly, if you are alone or family_size=0, then chances for survival is very low. For family_size>4, the changes decreases, too. This also looks to be an important feature for the model. Let's examine this further. 

# In[ ]:


sns.factorplot('Alone', 'Survived', hue='Sex', col='Pclass', data=data)
plt.show()


# It is visible that being alone is harmful irrespective of Sex or Pclass except for Pclass 3 where the chances of females who are alone is high than those with family. 

# **Fare_Range**
# 
# Since fare is also a continous feature, we need to convert it into ordinal value. For this, we will use pandas.qcut. 
# 
# So what qcut does is it splits or arranges the values according the number of bins we have passed. So if we pass for 5 bins, it will arrange the values equally spaced into 5 seperate bins or value ranges. 

# In[ ]:


data['Fare_Range'] = pd.qcut(data['Fare'], 4)
data.groupby(['Fare_Range'])['Survived'].mean().to_frame().style.background_gradient(cmap='summer_r')


# As discussed above, we can clearly see that as the fare_range increases, the chances of survival increases. 
# 
# Now, we cannot pass the Fare_Range values as it is. We should convert it into singleton values same as we did in Age_band.

# In[ ]:


data['Fare_cat']=0
data.loc[data['Fare']<=7.91, 'Fare_cat']=0
data.loc[(7.91<data['Fare']) & (data['Fare']<=14.454), 'Fare_cat']=1
data.loc[(14.454<data['Fare']) & (data['Fare']<=31.0), 'Fare_cat']=2
data.loc[(31.0<data['Fare']) & (data['Fare']<=513), 'Fare_cat']=3


# In[ ]:


type(data)


# In[ ]:


sns.factorplot('Fare_cat', 'Survived', hue='Sex', data=data)
plt.show()


# Clearly, as the Fare_cat increases, the survival chances increases. This feature may become an important feature during modeling along with the Sex. 

# **Converting String Values into Numeric**
# 
# Since we cannot pass strings to a machine learning model, we need to convert features of Sex, Embarked, etc. into numeric values. 

# In[ ]:


data.head()


# In[ ]:


data['Sex'].replace(['female', 'male'], [0,1], inplace=True)
data['Embarked'].replace(['S', 'C', 'Q'], [0,1,2], inplace=True)
data['Initial'].replace(['Mr', 'Mrs', 'Miss', 'Master', 'Other'], [0,1,2,3,4], inplace=True)


# **Dropping UnNeeded Features**
# 
# Name: We don't need name feature as it cannot be converted into any categorical value.
# 
# Age: We have the Age_band feature, so no need to this. 
# 
# Ticket: It is any random string that cannot be categorised. 
# 
# Fare: We have the Fare_cat feature, so unneeded. 
# 
# Cabin: A lot of NaN values and also many passengers have multiple cabins. So, this is a useless feature. 
# 
# Fare_Range: We have the fare_cat feature. 
# 
# PassengerId: Cannot be categorized. 

# In[ ]:


data.drop(['Name', 'Age', 'Ticket', 'Fare', 'Cabin', 'Fare_Range','PassengerId'], axis=1, inplace=True)


# In[ ]:


data.head()


# In[ ]:


sns.heatmap(data.corr(), annot=True, cmap='RdYlGn', linewidth=0.2, annot_kws={'size':20})
fig=plt.gcf()
fig.set_size_inches(18,15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


# Now the above correlation plot, we can see some positively related features. Some of them being SibSp and Family_size and Parch and Family_size and some negative ones like Alone and Family_size. 

# ## Part3: Predictive Modeling
# 
# We have gained some insights from the EDA part. But with that, we cannot accurately predict or tell whether a passenger will survive or die. So, we will predict the whether the Passenger will survive or not using some great Classification Algorithms. Following are the algorithms I will use to make the model:
# 
# 1) Logistic Regression
# 
# 2) Support Vector Machines
# 
# 3) Random Forest
# 
# 4) K-Nearest Neighbours
# 
# 5) Naive Bayes
# 
# 6) Decision Tree

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix


# In[ ]:


data.head()


# In[ ]:


train, test = train_test_split(data, test_size=0.3, random_state=2019, stratify=data['Survived'])


# In[ ]:


train_X = train[train.columns[1:]]
train_Y = train[train.columns[:1]]
test_X = test[test.columns[1:]]
test_Y = test[test.columns[:1]]
X=data[data.columns[1:]]
Y=data['Survived']


# **Radial Support Vector Machines (rbf-SVM)**

# In[ ]:


model = svm.SVC(kernel='rbf', C=1, gamma=0.1)
model.fit(train_X, train_Y)
prediction=model.predict(test_X)
print('Accuracy for rbf SVM is {:.6f}'.format(metrics.accuracy_score(prediction, test_Y)))


# **Linear Support Vector Machine (linear-SVM)**

# In[ ]:


model=svm.SVC(kernel='linear', C=0.1, gamma=0.1)
model.fit(train_X, train_Y)
prediction2=model.predict(test_X)
print('Accuracy for linear SVM is {:.6f}'.format(metrics.accuracy_score(prediction2, test_Y)))


# **Logistic Regression**

# In[ ]:


model=LogisticRegression()
model.fit(train_X, train_Y)
prediction3=model.predict(test_X)
print('Accuracy for Logistic Regression is {:.6f}'.format(metrics.accuracy_score(prediction3, test_Y)))


# **Decision Tree**

# In[ ]:


model=DecisionTreeClassifier()
model.fit(train_X, train_Y)
prediction4=model.predict(test_X)
print('Accuracy for Decision Tree is {:.6f}'.format(metrics.accuracy_score(prediction4, test_Y)))


# **K-Nearest Neighbors (KNN)**

# In[ ]:


model=KNeighborsClassifier()
model.fit(train_X, train_Y)
prediction5=model.predict(test_X)
print('Accuracy for KNN is {:.6f}'.format(metrics.accuracy_score(prediction5, test_Y)))


# Now the accuracy for the KNN model changes as we change the values for n_neighbours attribute. The default value is 5. Let's check the accuracies over vairous values of n_neighbours. 

# In[ ]:


a_index = list(range(1,11))
a=pd.Series()
x=[0,1,2,3,4,5,6,7,8,9,10]
for i in list(range(1,11)):
    model=KNeighborsClassifier(n_neighbors=i)
    model.fit(train_X, train_Y)
    prediction=model.predict(test_X)
    a=a.append(pd.Series(metrics.accuracy_score(prediction, test_Y)))
plt.plot(a_index, a)
plt.xticks(x)
fig=plt.gcf()
fig.set_size_inches(12,6)
plt.show()
print('Accuracies for different values of n are: ', a.values, 'with the max value as ', a.values.max())


# In[ ]:


model=KNeighborsClassifier(n_neighbors=8)
model.fit(train_X, train_Y)
prediction5=model.predict(test_X)
print('Accuracy for KNN is {:.6f}'.format(metrics.accuracy_score(prediction5, test_Y)))


# **Gaussian Naive Bayes**

# In[ ]:


model=GaussianNB()
model.fit(train_X, train_Y)
prediction6=model.predict(test_X)
print('Accuracy for NB is {:.6f}'.format(metrics.accuracy_score(prediction6, test_Y)))


# **Random Forests**

# In[ ]:


model=RandomForestClassifier(n_estimators=100)
model.fit(train_X, train_Y)
prediction7=model.predict(test_X)
print('Accuracy for RF is {:.6f}'.format(metrics.accuracy_score(prediction7, test_Y)))


# The accuracy of a model is not the only factor that determines the robustness of the classifier. Let's say that a classifier is trained over a training ata and tested over the test data and it scores an accuracy of 90%. 
# 
# Now this seems to be very good accuracy for a classifier, but can we confirm that it will be 90% for all the new test sets that come over? The answer is No, because we can't determine which all instances will the classifier will use to train itself. As the training and testing data changes, the accuracy will also change. It may increase or decrease. This is known as model variance. 
# 
# To overcome this and get a generalized model, we use Cross Validation. 

# ### Cross Validation
# 
# Many a times, the data is imbalanced, i.e there may be a high number of class 1 instances but less number of other class instances. Thus we should train and test our algorithm on each and every instance of the dataset. Then we can take an average of all the noted accuracies over the dataset. 
# 
# 1) The K-Fold Cross Validation works by first dividing the dataset into k-subsets. 
# 
# 2) Let's say we divide the dataset into (k=5) parts. We reserve 1 part for testing and train the algorithm over the 4 parts. 
# 
# 3) We continue the process by changing the testing part in each iteration and training the algorithm over the other parts. The accuracies and errors are then averaged to get a average accuracy of the algorithm. 
# 
# This is called K-Fold Cross Validation. 
# 
# 4) An algorithm may underfit over a dataset for some data and sometimes also overfit the data for other training set. Thus with cross-validation, we can achieve a generalized model. 

# In[ ]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict


# In[ ]:


kfold=KFold(n_splits=10, random_state=2019)
xyz=[]
accuracy=[]
std=[]
classifiers = ['Linear Svm', 'Radial Svm', 'Logistic Regression', 'KNN',               'Decision Tree', 'Naive Bayes', 'Random Forest']
models=[svm.SVC(kernel='linear'), svm.SVC(kernel='rbf'), LogisticRegression(),        KNeighborsClassifier(n_neighbors=9), DecisionTreeClassifier(),        GaussianNB(), RandomForestClassifier(n_estimators=100)]
for i in models:
    model=i
    cv_result = cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')
    cf_result=cv_result
    xyz.append(cv_result.mean())
    std.append(cv_result.std())
    accuracy.append(cv_result)
new_models_dataframe2 = pd.DataFrame({'CV Mean':xyz, 'Std':std}, index=classifiers)
new_models_dataframe2


# In[ ]:


plt.subplots(figsize=(12,6))
box=pd.DataFrame(accuracy, index=[classifiers])
box.T.boxplot()


# In[ ]:


new_models_dataframe2['CV Mean'].plot.barh(width=0.8)
plt.title('Average CV Mean Accuracy')
fig=plt.gcf()
fig.set_size_inches(8,5)
plt.show()


# The classification accuracy can be sometimes misleading due to imbalance. We can get a summarized result with the help of confusion matrix, which shows where did the model go wrong, or which class did the model predict wrong. 

# **Confusion Matrix**
# 
# It gives the number of correct and incorrect classifications made by the classifer. 

# In[ ]:


f, ax=plt.subplots(3,3, figsize=(12,10))

y_pred=cross_val_predict(svm.SVC(kernel='rbf'), X, Y, cv=10)
sns.heatmap(confusion_matrix(Y, y_pred), ax=ax[0, 0], annot=True, fmt='2.0f')
ax[0,0].set_title('Matrix for rbf-SVM')

y_pred=cross_val_predict(svm.SVC(kernel='linear'), X, Y, cv=10)
sns.heatmap(confusion_matrix(Y, y_pred), ax=ax[0, 1], annot=True, fmt='2.0f')
ax[0,1].set_title('Matrix for Linear-SVM')

y_pred=cross_val_predict(KNeighborsClassifier(n_neighbors=9), X, Y, cv=10)
sns.heatmap(confusion_matrix(Y, y_pred), ax=ax[0, 2], annot=True, fmt='2.0f')
ax[0,2].set_title('Matrix for KNN')

y_pred=cross_val_predict(RandomForestClassifier(n_estimators=100), X, Y, cv=10)
sns.heatmap(confusion_matrix(Y, y_pred), ax=ax[1, 0], annot=True, fmt='2.0f')
ax[1,0].set_title('Matrix for RF')

y_pred=cross_val_predict(LogisticRegression(), X, Y, cv=10)
sns.heatmap(confusion_matrix(Y, y_pred), ax=ax[1, 1], annot=True, fmt='2.0f')
ax[1,1].set_title('Matrix for Logistic Regression')

y_pred=cross_val_predict(DecisionTreeClassifier(), X, Y, cv=10)
sns.heatmap(confusion_matrix(Y, y_pred), ax=ax[1, 2], annot=True, fmt='2.0f')
ax[1,2].set_title('Matrix for Decision Tree')

y_pred=cross_val_predict(GaussianNB(), X, Y, cv=10)
sns.heatmap(confusion_matrix(Y, y_pred), ax=ax[2, 0], annot=True, fmt='2.0f')
ax[0,0].set_title('Matrix for Naive Bayes')

plt.subplots_adjust(hspace=0.2, wspace=0.2)
plt.show()


# **Interpreting Confusion Matrix**
# 
# The left diagonal shows the number of correct predictions made for each class while the right diagonal shows the number of wrong predictions made. Let's consider the first plot for rbf-SVM:
# 
# 1) The no. of correct predictions are 491(for dead) + 247(for survived) with the mean CV accuracy being (491+247)/891=82.8% which we did get ealier. 
# 
# 2) Errors -> Wrongly Classified 58 dead people as survived and 95 survived as dead. Thus it has made more mistakes by predicting dead as survived. 
# 
# By looking at all the matrices, we can say that rbf-SVM has a higher chance in correctly predicting dead passengers but NaiveBayes has a higher chance in correctly predicting passengers who survived. 

# In[ ]:




