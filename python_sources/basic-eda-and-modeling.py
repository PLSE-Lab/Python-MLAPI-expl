#!/usr/bin/env python
# coding: utf-8

# # Import libraries and check directory

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os


# In[ ]:


print(os.listdir("../input"))


# # Importing data 

# In[ ]:


# Loading data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# # Check first informations about the data 

# In[ ]:


display(train.head(5))

print("Train shape: ", train.shape)


# In[ ]:


display(test.head(5))

print("Test shape: ", test.shape)


# In[ ]:


# Checking null values
train.isnull().sum()


# # Explorating our main feature
# 
# As survive is our "target feature" we will do a further exploration into this variable.

# In[ ]:


plt.pie(train["Survived"].value_counts(),explode=[0, 0.02],autopct='%1.1f%%', labels=train["Survived"].value_counts().index)
plt.title("Survival Rate")
plt.show()


# Looking at our pie chart we can notice that most passengers did not survived. But we want to know more about them, some questions comes up like 'Which type of passenger survived?', 'Which type of class did they buy?',  'Where they embarked?'.

# # Exploring the features
# Now we will understand the relationship of our features against our target feature.

# ### Sex vs Survival Rate
# The first feature we will analyze is the "Sex" feature.
# 
# Which sex had more survivors?

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,5))

sns.countplot('Sex',hue='Survived',data=train, ax=ax[0])
ax[0].set_title("Male/Female survival plot")

survivors = train.query("Survived == 1")
survivors["Sex"].value_counts().plot.pie(explode=[0, 0.02],autopct='%1.1f%%', 
        labels=survivors["Sex"].value_counts().index, ax=ax[1])
ax[1].set_title("Male/Female Survival Rate")
plt.show()


# Looking at the graphics we can realize that most of the survivors were female. Perhaps the story that children and women first land is true, now let's see if this applies for age too.

# ## Concat data frames
# Before go ahead we will concat the train and test data frames, this way both of them will have the same features changes

# In[ ]:


# Removing the target feature from the remaining dataset
new_train = train.drop("Survived", axis=1)
join_df = pd.concat([new_train, test])


# ### Age vs Survival Rate

# In[ ]:


# Let's have a general view of our feature Age
np.unique(join_df["Age"])


# We have a lot of NULL values (if we look at the beginning of the kernel, there are 177 null values to Age) and some number with .5.
# Let's treat our float numbers turning them into integer and fill our nulls.

# In[ ]:


# Replacing the nulls
male_mean = train.query("Sex == 'male' and Survived==1")["Age"].mean()
female_mean = train.query("Sex == 'female' and Survived==1")["Age"].mean()

join_df.loc[(join_df.Age.isnull())&(join_df.Sex=='female'),'Age']=female_mean
join_df.loc[(join_df.Age.isnull())&(join_df.Sex=='male'),'Age']=male_mean

print("Total null: {}".format(join_df.Age.isnull().sum()))

# Rounding the ages
join_df["Age"] = join_df["Age"].map(lambda age: int(age))


# In[ ]:


survivor_list = train.query("Survived == 1")

print("Survivors mean age is {:.0f}".format(survivor_list["Age"].mean()))
print("Survivors mean age for males is {:.0f}".format(survivor_list.query("Sex == 'male'")["Age"].mean()))
print("Survivors mean age for females is {:.0f}".format(survivor_list.query("Sex == 'female'")["Age"].mean()))
print("Minimal Survivor Age is {:.0f}".format(min(survivor_list["Age"])))
print("Maximum Survivor Age is {:.0f}".format(max(survivor_list["Age"])))

plt.figure(figsize=(25,6))
sns.barplot(train['Age'],train['Survived'], ci=None)
plt.xticks(rotation=90);
plt.show()


# With the barplot we can have a clear vision of the ages that had more chances of survival:
# 
# - Children up to 15 years old
# - Some adults from 28 years old to 35 years
# - Some older ages like 48 to 53 years old and 63 years old
# 
# 

# ### Class vs Survival Rate
# 
# Nice! Until now we now that:
#  - More females survived.
#  - We have some ranges of ages that had more chances of survival.
#  
# Now we want to know the behavior of those people. With the feature "class" we can know if the ones in the passengers in the first class had more chances of survival (maybe because of the money and status) than the ones in the second and third classes. Or maybe we find a surprise and people from the lower classes survived most.

# In[ ]:


# Let's see our classes
np.unique(join_df["Pclass"])


# In[ ]:


train["Pclass"].value_counts()


# In[ ]:


class_count_dict = dict(train["Pclass"].value_counts().sort_index())

for k,v in class_count_dict.items():
    print("People from the {} class: {}".format(k, v))


# In[ ]:


f,ax=plt.subplots(3,2,figsize=(15,15))

train["Pclass"].value_counts().plot.pie(explode=[0, 0.02, 0.02],autopct='%1.1f%%', 
        labels=survivors["Pclass"].value_counts().index, ax=ax[0][0])
ax[0][0].set_title("Class Survival Proportion")


sns.countplot(train["Pclass"], ax=ax[0][1])
ax[0][1].set_title("Count passengers count")

sns.countplot('Pclass',hue='Survived',data=train, ax=ax[1][0])
ax[1][0].set_title("General Survivors per Class")

sns.countplot('Pclass',hue='Sex',data=train, ax=ax[1][1])
ax[1][1].set_title("General Class per Sex")

sns.countplot('Pclass',hue='Sex',data=train.query("Survived == 1"), ax=ax[2][0])
ax[2][0].set_title("Survivors Class per Sex")

sns.barplot(x='Pclass',y='Survived',data=train, ax=ax[2][1])
ax[2][1].set_title("Survivors Rate per Class")


# ### Important notes:
# - We can notice that we have more survivors from 1st class, which turns class an important feature.
# - The pattern that females survived more maintain with classes.
# -  Surprisingly (or not!) the rule of the class survival only exist with the 1st class. Maybe it's because the difference in the quantity of people in second class and third class, or only the class really had preference.

# ### Embarked vs Survival Rate
# What if the place they embarked also influences the survival rate? Can we find something useful?
# 
# Data summary:
# - C = Cherbourg
# - Q = Queenstown
# - S = Southampton

# In[ ]:


# Check unique embarked places
print("Unique places: ", train.Embarked.unique())

# First of all, as we have only a few null values, lets fill with the place that had more embarks
f,ax=plt.subplots(1,1,figsize=(6,5))

train["Embarked"].value_counts().plot.pie(explode=[0, 0.02, 0.02],autopct='%1.1f%%', 
                                              labels=train["Embarked"].value_counts().index, ax=ax)


# As 72.4% of the passengers embarked in Southampton, we will fill the nulls with 'S'

# In[ ]:


print("Null values: ", train.Embarked.isnull().sum())
# Treating missing
train['Embarked'].fillna('S',inplace=True)
join_df['Embarked'].fillna('S',inplace=True)
print("Null values after cleaning: ", train.Embarked.isnull().sum())


# In[ ]:


f,ax=plt.subplots(3,2,figsize=(15,15))

sns.countplot(train["Embarked"], ax=ax[0][0])
ax[0][0].set_title("Quantity of people that embarked in place")

sns.countplot('Embarked',hue='Survived',data=train, ax=ax[0][1])
ax[0][1].set_title("Survived quantity by place of embark")

sns.countplot('Embarked',hue='Pclass',data=train, ax=ax[1][0])
ax[1][0].set_title("Quantity of class embarked by place")

sns.countplot('Embarked',hue='Sex',data=train, ax=ax[1][1])
ax[1][1].set_title("Sex by place")

sns.countplot('Embarked',hue='Sex',data=train, ax=ax[1][1])
ax[1][1].set_title("Sex by place")

sns.barplot(x='Embarked',y='Survived',data=train, ax=ax[2][0])
ax[2][0].set_title("Embarked vs Survived rate")

train["Embarked"].value_counts().plot.pie(explode=[0, 0.02, 0.02],autopct='%1.1f%%', 
                                              labels=train["Embarked"].value_counts().index, ax=ax[2][1])
ax[2][1].set_title("Embarked place proportion")


# ### Importants notes:
# - If we join the information that we gathered in the analysis of the classes with the information of the graphs above, we can infere that people from Queenstown had less chance of survival.
# - The relation of the class with the place turn the place of embark important

# ### Family vs Survived
# To finish our features understandment, let's check if the family's size mattered to the survival rate.

# **Data summary**:
# - sibsp: siblings / spouses aboard the Titanic	
# - parch: parents / children aboard the Titanic
# - sibsp + parch = family size

# In[ ]:


# Creating the feature family_size
train["family_size"] = train["SibSp"] + train["Parch"]
train.head(5)

# Replicating to joined data frame
join_df["family_size"] = join_df["SibSp"] + join_df["Parch"]


# In[ ]:


f,ax=plt.subplots(3,2,figsize=(15,15))

sns.countplot('SibSp',hue='Survived',data=train, ax=ax[0][0])
ax[0][0].set_title("Survived by siblings/spouses quantity")
sns.barplot(x='SibSp',y='Survived',data=train, ax=ax[0][1])
ax[0][1].set_title("Survived by siblings/spouses rate")

sns.countplot('Parch',hue='Survived',data=train, ax=ax[1][0])
ax[1][0].set_title("Survived by parents/children quantity")
sns.barplot(x='Parch',y='Survived',data=train, ax=ax[1][1])
ax[1][1].set_title("Survived by parents/children rate")

sns.countplot('family_size',hue='Survived',data=train, ax=ax[2][0])
ax[2][0].set_title("Survived by family size")
sns.barplot(x='family_size',y='Survived',data=train, ax=ax[2][1])
ax[2][1].set_title("Survived by family size rate")

plt.subplots_adjust(wspace=0.2,hspace=0.5)


# ### Importants notes:
# - One interesting point in these graphics is that families with 1-3 people had more chance of survive.
# - So we will consider family size an important feature for our algorithm and ignore the SibSp and Parch features

# ## Main observations
# - Females had more chances of survival than males.
# - People in the range of 5-14 years (kids) had more chance of survival.
# - Even that Southampton has more people embarked, more people from Cherbourg survived.
# - People from 1st class probably had priority and consequently more chances of survival.
# - Family of size 1-3 had more chances of survival.

# # Feature engineering
# Now that we done our EDA let's do some feature engineering into our features.

# In[ ]:


# Getting the correlation between features
sns.heatmap(train.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) 
# Get current figure
fig=plt.gcf()
fig.set_size_inches(12,8)
plt.show()


# In[ ]:


# Converting Sex into numerical values
train["Sex"].replace(["male", "female"], [0, 1], inplace=True)
# Converting the embark place into numerics labels
train["Embarked"].replace(["S", "C", "Q"], [0, 1, 2], inplace=True)


# In[ ]:


# Replicating to joined data frame
join_df["Sex"].replace(["male", "female"], [0, 1], inplace=True)

# Replicating to joined data frame
join_df["Embarked"].replace(["S", "C", "Q"], [0, 1, 2], inplace=True)


# In[ ]:



# Transforming age by range
# Using as base to our range of ages
plt.figure(figsize=(25,6))
sns.barplot(train['Age'],train['Survived'], ci=None)
plt.xticks(rotation=90);
plt.show()

# Creating new field
train["Age_Range"] = 0
train.loc[train["Age"]<=15, "Age_Range"] = 0
train.loc[(train["Age"]>15)&(train["Age"]<=35), "Age_Range"] = 1
train.loc[(train["Age"]>35)&(train["Age"]<=55), "Age_Range"] = 2
train.loc[train["Age"]>55, "Age_Range"] = 3


# In[ ]:


# Creating new field
join_df["Age_Range"] = 0
join_df.loc[join_df["Age"]<=15, "Age_Range"] = 0
join_df.loc[(join_df["Age"]>15)&(join_df["Age"]<=35), "Age_Range"] = 1
join_df.loc[(join_df["Age"]>35)&(join_df["Age"]<=55), "Age_Range"] = 2
join_df.loc[join_df["Age"]>55, "Age_Range"] = 3


# In[ ]:


# Dropping features
train.drop(columns=["PassengerId", "Name", "Ticket", "Cabin", "Fare", "Parch", "SibSp", "Age"], inplace=True)
join_df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin", "Fare", "Parch", "SibSp", "Age"], inplace=True)


# In[ ]:


# Changing the columns names
train.columns = ["survived", "p_class", "sex", "embarked", "family_size", "age_range"]
join_df.columns = ["p_class", "sex", "embarked", "family_size", "age_range"]


# In[ ]:


# Getting the correlation between features
sns.heatmap(train.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) 

# Get current figure
fig=plt.gcf()
fig.set_size_inches(12,8)
plt.show()


# ## Models
# Now that we have our features the way we want, let's create a few models to see which one will perform better

# In[ ]:


from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.naive_bayes import GaussianNB 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn import metrics
import xgboost as xgb


# In[ ]:


# Splitting the data frames again
print("Test Shape: {}".format(test.shape))
print("Train Shape: {}".format(train.shape))
print("Merged Shape: {}".format(join_df.shape))

test_shape = test.shape[0]
train_shape = train.shape[0]


# In[ ]:


# Target feature
y = train["survived"]

# Removing the target feature from the remaining dataset
X = join_df[:train_shape]
test = join_df[train_shape:]

# Splitting in test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print("X_train shape: ", X_train.shape)
print("X_test shape: ", X_test.shape)
print("y_train shape: ", y_train.shape)
print("y_test shape: ", y_test.shape)


# ### Logistic Regression

# In[ ]:


# liblinear because its a small dataset
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)

linear_regression_prediction = model.predict(X_test)

print('Logistic regression accuracy: ',metrics.accuracy_score(linear_regression_prediction, y_test))


# ### Random Forest

# In[ ]:


model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

random_forest_prediction = model.predict(X_test)

print('Random forest accuracy: ', metrics.accuracy_score(random_forest_prediction, y_test))


# ### Decision Tree

# In[ ]:


model = DecisionTreeClassifier()
model.fit(X_train, y_train)

decision_tree_prediction = model.predict(X_test)

print('Decision tree accuracy: ', metrics.accuracy_score(decision_tree_prediction, y_test))


# ### Naive Bayes

# In[ ]:


model = GaussianNB()
model.fit(X_train, y_train)

gaussian_prediction = model.predict(X_test)

print('Naive Bayes accuracy: ', metrics.accuracy_score(gaussian_prediction, y_test))


# ### XGBoost

# In[ ]:


model = xgb.XGBClassifier(n_estimators=100,
                          n_jobs=4,
                          learning_rate=0.03,
                          subsample=0.8,
                          colsample_bytree=0.8)

model.fit(X_train, y_train)
xgb_prediction = model.predict(X_test)

print('XGB prediction: ', metrics.accuracy_score(xgb_prediction, y_test))


# # Submission

# In[ ]:


test_passenger_id = pd.read_csv('../input/gender_submission.csv')['PassengerId']
# We will use the XGB prediction for the submition
xgb_prediction = model.predict(test)
submission = pd.concat([pd.DataFrame(test_passenger_id), pd.DataFrame({'Survived':xgb_prediction})], axis=1)
submission.to_csv('predictions.csv', index=False)

