#!/usr/bin/env python
# coding: utf-8

# Hello all,
# I am trying to use Feynman Technique to get deeper understanding the concept that I thought I know. That is why I am doing this with my broken language. Any question that will make me suffer is highly appreciated. 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn


get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier


# In above cell, we imported nesessary packages. We will understand, why they are needed.

# In[ ]:


train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")
train_test = pd.concat([train, test], ignore_index=True, sort  = False)


# We read the data set using pandas. After that, we've created data frame that include both data set. This will help us while we are filling blank cell in our data set.

# # Understanding Data

# Now it is time to understand the data we are using. It is really important to analyse because causality is a concept that only people can do understand(for now). We will use that understanding while we are filling the blank datas. Lets write some code for understanding our data.

# In[ ]:


train_test.head()


# Now we know what features we have. In next steps, we may add or delete some of features. Who knows?

# In[ ]:


print('Train data shape is: ', train.shape)
print('Test data shape is: ', test.shape)
print('Mixed data shape is: ', train_test.shape)


# In[ ]:


survivedd = train[train["Survived"] == 1]
not_survived = train[train["Survived"] == 0]

print("Survived count: {x} {y:1.3f} %".format(x=len(survivedd), y=float(len(survivedd) / len(train)) * 100))
print("Not Survived count: {z} {n:1.3f} %".format(z=len(not_survived), n=float(len(not_survived) / len(train) * 100)))


# Visualization can be used to show output of above cell.

# In[ ]:


ax=sns.countplot(train["Survived"])

#Just fancy code to write percentages on bar.
totals = []
for i in ax.patches:
    totals.append(i.get_height())

total = sum(totals)
for i in ax.patches:
    # get_x pulls left or right; get_height pushes up or down
    ax.text(i.get_x()+.25, i.get_height()-250.95,             str(round((i.get_height()/total)*100, 2))+'%', fontsize=15,
                color='white')

plt.title('Survival Status')

#Only %38.38 of passengers survived.


# Adding some features may be a good idea. We will come back to this part later.

# In[ ]:


train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1
train_test['FamilySize'] = train_test['SibSp'] + train_test['Parch'] + 1

train['IsAlone'] = 0
train.loc[train['FamilySize'] == 1, 'IsAlone'] = 1

train_test['IsAlone'] = 0
train_test.loc[train_test['FamilySize'] == 1, 'IsAlone'] = 1

test['IsAlone'] = 0
test.loc[test['FamilySize'] == 1, 'IsAlone'] = 1

# We have two new feature.


#  # Correlation Matrix
# 

# Before we dive in to graphs, it is a good idea to understand Correlation Matrix and Covariance Matrix. Covariance indicates the direction of the linear relation between variables.
# Correlation measures strength and direction of linear relationship between columns we have.
# 
# Correlation is always between +1 and -1.
# If the number is close to 1, it may have direct proportion between variables. If the number is close to -1, it may have inverse proportion between variables.
# 
# Correlation does not indicate causality, but not always. For example, lets assume we have a dataset about 
# crime rate and ice cream consumption. They are both increased in selected month so their correlation is close to 1. That doesnt mean there is a direct relation between two of them. 
# 
# It is good for getting an idea. 
# The correlation matrix contains only columns with integer value. 
# If we want to contain other columns we need to convert into integers.

# In[ ]:


train_test.corr()


# Presentation is important so lets use seaborn library for Correlation Matrix.

# In[ ]:


plt.figure(figsize=(16,5))
corr_map = sns.heatmap(train_test.corr(),annot=True)


# It look way better. Isn't it?

# # Added features explained

# Sometimes, even if everything seems okay, we may not get correct results. Lets see with an example.

# In[ ]:


fig, ax = plt.subplots(figsize=(12,6),ncols=2,nrows=2)
ax1 = sns.barplot(x="SibSp",y="Survived", data=train_test, ax = ax[0][0]);
ax2 = sns.barplot(x="Parch",y="Survived", data=train_test, ax = ax[0][1]);
ax3 = sns.countplot(x="SibSp", data=train_test, ax = ax[1][0]);
ax4 = sns.countplot(x="Parch", data=train_test, ax = ax[1][1]);
#ax3.set_yscale('log')
#ax4.set_yscale('log')
ncount = len(train["SibSp"])
for p in ax3.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax3.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
            ha='center', va='bottom')
for p in ax4.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax4.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
            ha='center', va='bottom')


# Barplots in first row, may make us think that there is relationship between Sibs, Parch and Survival Rate. BUT we neeed to look for sample size of higher parch and sibsp values. This is why we created second row.
# 
# As it can be seen, sample size is really small in higher values of parch and sibsp, so it may mislead us. 
# Thats why we created FamilySize feature and isAlone feature.
# 
# I will drop theese columns after we fill the missing age columns.

# **More visualization**

# In[ ]:


fig, ax = plt.subplots(figsize=(16,12),ncols=2,nrows=2)
ax1 = sns.barplot(x="Sex",y="Survived", data=train, ax = ax[0][0]);
ax2 = sns.barplot(x="Pclass",y="Survived", data=train, ax = ax[0][1]);
ax3 = sns.barplot(x="FamilySize",y="Survived", data=train, ax = ax[1][0]);
ax4 = sns.barplot(x="Embarked",y="Survived", data=train, ax = ax[1][1]);


# Observations
# * Pclass of passengers has significicant effect on survival rate.
# * Sex is another factor that affects survival rate. 
# * Embarked may be mislead us, need deeper understanding.(Rate of people who embarked from C(Cherbourg) is using Pclass...etc)

# 

# # Missing Data

# In[ ]:


print("Missing Data in Train Set"+'\n')
total = train.isnull().sum().sort_values(ascending = False)
percentage = total/len(train)*100
missing_data = pd.concat([total, percentage], axis=1, keys=['Total', '%'])
print(missing_data.head(3))

print('\n')
print("Missing Data in Test Set"+'\n')
total1 = test.isnull().sum().sort_values(ascending = False)
percentage1 = total1/len(test)*100
missing_data1 = pd.concat([total1, percentage1], axis=1, keys=['Total', '%'])
print(missing_data1.head(3))


# Cell below is showing missing values on train set. It is not necessary to visualize but it is nice to have.

# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap="magma")
#yticklabes=False removes left side labels.
#cbar=False removes the colorbar.


# # Filling Values

# In our train set, we have 2 missing embarked value. We can fill them with most common one since it is not going to affect remarkably. 
# 
# It is same with our test set for missing a fare value. 
# 
# But I am going to fill these value with different mindset. It is not that common and it is almost impossible to use this in larger data set. 

# **Embarked**

# In[ ]:


display(train_test[train_test.Embarked.isnull()])


# As we can see, passengers who dont have embarked data, have similarities. We can look for passengers who have similar feature values and fill embarked value according to that. 

# In[ ]:


embarked_missing = train_test[(train_test["Sex"] == "female") & (train_test["Pclass"] ==1) & (train_test["Cabin"].str.startswith('B')) 
                        & (train_test["Fare"]>70) & (train_test["Fare"]<100)]
# Embarked_missing shows us passengers who have similar values.
print(embarked_missing["Embarked"].value_counts())


# We have checked passengers who have similar values, and missing values are likely to be from Southampton because 7 out of 9 is embarked from Southampton. Lets assign the values.

# In[ ]:


train["Embarked"] = train["Embarked"].fillna("S")
train_test["Embarked"] = train_test["Embarked"].fillna("S")


# **Fare**

# In[ ]:


fare_missing = train_test[(train_test.Pclass == 3)]
test["Fare"] = test["Fare"].fillna(fare_missing.Fare.mean())
train_test["Fare"] = train_test["Fare"].fillna(fare_missing.Fare.mean())


# Since fare is directly affected by pclass we filled with mean value of same pclass.

# **Age**

# When we look back to correlation matrix, it seems, there are inverse proportion. So basically, people who are older tend to have better ticket class. When we think about what correlation matrix says, it is logical. Older people are richer so they buy higher class. Missing age values are worth more understanding, lets draw some boxplot.

# In[ ]:


fig, ax = plt.subplots(figsize=(16,12),ncols=2,nrows=1)
ax1 = sns.boxplot(x="Survived", y="Age", hue="Pclass", data=train_test, ax = ax[0]);
ax2 = sns.boxplot(x="Pclass", y="Age", hue="Sex", data=train_test, ax = ax[1]);


# As we can see, survival status is good indicator for filling ages. Since only train data have survival data. We are going to write two different filling condition. One of them is for train set, the other one is for test set.

# In[ ]:


test_na_age_index = list(test["Age"][test["Age"].isnull()].index)

for i in test_na_age_index:
    i_Pclass = test.iloc[i]["Pclass"]
    mean_pclass = test[test.Pclass==i_Pclass]['Age'].mean()
    age_pred = test[((test['Sex'] == test.iloc[i]["Sex"]) & (test['Pclass'] == test.iloc[i]["Pclass"]))]["Age"].mean()
    if not np.isnan(age_pred) :
        test['Age'].iloc[i] = age_pred
    else :
        test['Age'].iloc[i] = mean_pclass


# In[ ]:


train_na_age_index = list(train["Age"][train["Age"].isnull()].index)

for i in train_na_age_index:
    i_Pclass = train.iloc[i]["Pclass"]
    mean_pclass = train[train.Pclass==i_Pclass]['Age'].mean()
    age_pred = train[((train['Survived'] == train.iloc[i]["Survived"])&(train['Sex'] == train.iloc[i]["Sex"]) & (train['Pclass'] == train.iloc[i]["Pclass"]))]["Age"].mean()
    if not np.isnan(age_pred) :
        train['Age'].iloc[i] = age_pred
    else :
        train['Age'].iloc[i] = mean_pclass


# In above cells, we find index of values which have missing age cell. Then, we filled the value based on our criteria. 
# For train set:
# We filled missing values with looking Survival Status, Pclass and Sex
# 
# For train set:
# We filled missing values with looking Pclass and Sex.
# 

# **Cabin**

# Since approximately 80% of cabin values are missing, it is logical to drop the feature. It will be not worth to fill and it may cause inaccuracy. 

# In[ ]:


train.drop(["Cabin"], axis = 1, inplace=True)
test.drop(["Cabin"], axis = 1, inplace=True)
train_test.drop(["Cabin"], axis = 1, inplace=True)


# # Converting data that machine learning can understand

# As all we know, libraries, languages are just tools. But deeper understanding can only be achieved by understanding the math behind. This notebook will not include this but I highly recommend the math behind. Basically, we need to convert our features to numbers. There are two type of features. Categorical and continuous. Since most of our features are categorical, we will convert continues one into categorical features. 
# 
# Also we will create new features based by title and get rid of some of features.

# In[ ]:


train['Sex'] = train['Sex'].map({"male": 0, "female": 1})
test['Sex'] = test['Sex'].map({"male": 0, "female": 1})


# We convert Sex features into 1,0.

# In[ ]:


one_hot_train = pd.get_dummies(train["Embarked"], drop_first=True)
one_hot_test = pd.get_dummies(test["Embarked"], drop_first=True)
#This is called one hot encoding. It is same think with above cell. The reason behind dropping first column is effiency.
#If both value is 0, it means it is dropped column. 
one_hot_train


# Now we will drop column embarked and add the one hot encoded data frame.

# In[ ]:


train.drop(["Embarked"], axis = 1, inplace=True)
train = train.join(one_hot_train)

test.drop(["Embarked"], axis = 1, inplace=True)
test = test.join(one_hot_test)


# It is really logical to create features from name title. I am inspired by Kaggle notebook called 'Titanic Top 4% with ensemble modeling'. Thank you for great work.

# In[ ]:


dataset_title = [i.split(",")[1].split(".")[0].strip() for i in train["Name"]]
train["Title"] = pd.Series(dataset_title)

dataset_title2 = [i.split(",")[1].split(".")[0].strip() for i in test["Name"]]
test["Title"] = pd.Series(dataset_title2)


# In[ ]:


train["Title"] = train["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
train["Title"] = train["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
train["Title"] = train["Title"].astype(int)

test["Title"] = test["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
test["Title"] = test["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
test["Title"] = test["Title"].astype(int)


# We convert Name feature into a categorical data which may be understanded by machine learning algorithm. We will use get dummies to get better optimizing at the end.
# 
# It is time to convert continues data into categorical data.

# In[ ]:


pd.cut(train['Age'], 6)


# This command help us to understand how to determine age categories.

# In[ ]:


train.loc[ train['Age'] <= 13, 'Age'] = 0,
train.loc[(train['Age'] > 13) & (train['Age'] <= 27), 'Age'] = 1,
train.loc[(train['Age'] > 27) & (train['Age'] <= 40), 'Age'] = 2,
train.loc[(train['Age'] > 40) & (train['Age'] <= 53), 'Age'] = 3,
train.loc[(train['Age'] > 53) & (train['Age'] <= 66), 'Age'] = 4,
train.loc[ train['Age'] > 66, 'Age'] = 5


# In[ ]:


test.loc[ train['Age'] <= 13, 'Age'] = 0,
test.loc[(train['Age'] > 13) & (train['Age'] <= 27), 'Age'] = 1,
test.loc[(train['Age'] > 27) & (train['Age'] <= 40), 'Age'] = 2,
test.loc[(train['Age'] > 40) & (train['Age'] <= 53), 'Age'] = 3,
test.loc[(train['Age'] > 53) & (train['Age'] <= 66), 'Age'] = 4,
test.loc[ train['Age'] > 66, 'Age'] = 5


# We convert our age feature into categorical so it may fit machine learning algorithm better.

# In[ ]:



e = pd.get_dummies(train["FamilySize"], drop_first=True)
train.drop(["FamilySize"], axis = 1, inplace=True)
train = train.join(e)

f = pd.get_dummies(test["FamilySize"], drop_first=True)
test.drop(["FamilySize"], axis = 1, inplace=True)
test = test.join(f)


# We convert family size into one hot encoding. It is not necessary but nice to have.

# In[ ]:


test.drop(["Name"], axis = 1, inplace=True)
train.drop(["Name"], axis = 1, inplace=True)
test.drop(["SibSp"], axis = 1, inplace=True)
train.drop(["SibSp"], axis = 1, inplace=True)
test.drop(["Parch"], axis = 1, inplace=True)
train.drop(["Parch"], axis = 1, inplace=True)
test.drop(["Ticket"], axis = 1, inplace=True)
train.drop(["Ticket"], axis = 1, inplace=True)
train.drop(["PassengerId"], axis = 1, inplace=True)
test.drop(["Fare"], axis = 1, inplace=True)
train.drop(["Fare"], axis = 1, inplace=True)


# We get rid of the features that is not needed. 

# # Model, prediction and results

# It is time to see our how well it work. 
# 
# Since all of the features are categorical I believe Tree Algorithms for better than the other. Lets find out which one is best.

# In[ ]:


X_train = train.drop("Survived", axis=1)
y_train = train["Survived"]
X_test  = test.drop("PassengerId", axis=1).copy()


# In[ ]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print(logreg.score(X_train, y_train))


# In[ ]:


r_forest = RandomForestClassifier(n_estimators=200)
r_forest.fit(X_train, y_train)
y_pred = r_forest.predict(X_test)
r_forest.score(X_train, y_train)


# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
knn.score(X_train, y_train)


# In[ ]:


svc = SVC()
svc.fit(X_train, y_train)
Y_pred = svc.predict(X_test)
svc.score(X_train, y_train)


# In[ ]:


d_tree = DecisionTreeClassifier()
d_tree.fit(X_train, y_train)
y_pred = d_tree.predict(X_test)
d_tree.score(X_train, y_train)


# In[ ]:


gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
y_pred = gaussian.predict(X_test)
gaussian.score(X_train, y_train)


# In[ ]:


ada=AdaBoostClassifier(n_estimators=200,learning_rate=0.1)
ada.fit(X_train, y_train)
y_pred = ada.predict(X_test)
ada.score(X_train, y_train)


# As it can be seen, tree algorithms works better. Personally I am fan of Random Forest. All algorithms must be learned indivially. I am open for any constructive criticism. Have a wonderful journey!

# In[ ]:


Submission = pd.DataFrame({ 'PassengerId': test["PassengerId"],
                            'Survived': y_pred })
Submission.to_csv("Submission1.csv", index=False)


# **Reference**
# * https://www.kaggle.com/startupsci/titanic-data-science-solutions
# * https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling
# 
