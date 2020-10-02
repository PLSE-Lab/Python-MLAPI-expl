#!/usr/bin/env python
# coding: utf-8

# **How to approach a supervised learning problem:**
# 
# 1. Do some EDA.
# 2. Build a baseline model.
# 3. Do more EDA.
# 4. Engineer features.
# 5. Build a better model.

# # 1

# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.metrics import accuracy_score

get_ipython().run_line_magic('matplotlib', 'inline')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# import datasets
df_train = pd.read_csv("../input/train.csv") 
df_test = pd.read_csv("../input/test.csv") 

# view first five lines of training data
df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


df_train.info()


# In[ ]:


df_train.describe()


# In[ ]:


# plot of count(Survived)
sns.countplot(x="Survived", data=df_train)
plt.show()


# More than 500 people didn't survive.
# 
# A few over 300 people survived.
# 
# So, we will predict that nobody survived as base model.

# In[ ]:


no_survived = pd.Series([0] * df_test.shape[0])


# In[ ]:


out = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': no_survived})


# In[ ]:


out.to_csv('no_survival.csv', index=False)


# Accuracy: 62.7

# # 2

# In[ ]:


# plot count of male and female on titanic
sns.countplot(x="Sex", data=df_train);


# There are more than 575 male and a little over 300 females, so let's check survival according to gender.

# In[ ]:


sns.countplot(x="Survived", hue='Sex', data=df_train);


# In[ ]:


sns.catplot(x="Survived", col="Sex", kind="count", data=df_train);


# We can see out of 300 female passengers, more than 200 survived whereas out of 600 male passengers about 100 survived.
# 
# **Take Away:** Women were more likely to survive than men.

# In[ ]:


df_train.groupby(['Sex']).Survived.sum()


# In[ ]:


df_train.groupby(["Sex"]).Survived.value_counts()


# In[ ]:


print(df_train[df_train["Sex"]=="female"].Survived.sum() / df_train[df_train["Sex"]=="female"].shape[0]) 
# print(df_train[df_train["Sex"]=="female"].Survived.sum() / df_train[df_train["Sex"]=="female"].count()) 
print(df_train[df_train["Sex"]=="male"].Survived.sum() / df_train[df_train["Sex"]=="male"].shape[0]) 


# About 74% of women survived and only 19% men survived.
# 
# Let's build a model that predicts all womens survived and no male survived.

# In[ ]:


women_survived_series = pd.Series(list(map(int, df_test["Sex"]=="female")))


# In[ ]:


out = pd.DataFrame({"PassengerId": df_test.PassengerId, "Survived": women_survived_series})
out.to_csv('all_women_survived.csv', index=False)


# Accuracy: 76.5%

# # 3

# In[ ]:


sns.catplot(x="Survived", col="Pclass", kind="count", data=df_train);


# In[ ]:


sns.catplot(x="Survived", col="Pclass", kind="count", hue="Sex", data=df_train);


# - People with `Pclass=1` are more likely to survive i.e. rich people
# - Very very few females did not survive in `Pclass=1` and `Pclass=2` whereas about 50% female did not survive in `Pclass=3`
# 

# In[ ]:


print(df_train.groupby("Pclass").Survived.sum() / df_train.groupby("Pclass").Survived.count())


# - About 63% people survived in `Pclass=1`
# - About 47% people survived in `Pclass=2`
# - About 24% people survived in `Pclass=2`

# Port of Embarkation
# - C = Cherbourg
# - Q = Queenstown
# - S = Southampton

# In[ ]:


sns.catplot(x="Survived", col="Embarked", kind="count", data=df_train);


# In[ ]:


#sns.catplot(x="Embarked", col="Survived", kind="count", data=df_train);


# - Those who embarked from `C` had greater chances of survival (55%). 
# - Q (39%)
# - S (33%)

# In[ ]:


print(df_train.groupby("Embarked").Survived.sum() / df_train.groupby("Embarked").Survived.count())


# In[ ]:


sns.catplot(x="Survived", col="Embarked", hue="Pclass", kind="count", data=df_train);


# In[ ]:


# df_train.groupby("Embarked").Survived.sum()  # shows number of people survived from each embarked point
# df_train.groupby("Embarked").Survived.count() # shows number of poeple embarked form each port


# **EDA with numerical variables**

# In[ ]:


plt.figure(figsize=(18, 8))
sns.distplot(a=df_train.Fare, kde=False);


# - Three passengers with fare greater than 300 (`Fare=512.3292`), and all of them survived.
# - About 340 passengers paid less than 10\$
# - Very few passengers with fare more than 50\$

# In[ ]:


# Use a pandas plotting method to plot the column 'Fare' for each value of 'Survived' on the same plot.
df_train.groupby('Survived').Fare.hist(alpha=0.5);


# - Those who paid more had more chances of survival.

# In[ ]:


df_train_drop = df_train.Age.dropna()
plt.figure(figsize=(18, 8))
sns.distplot(a=df_train_drop, kde=False);


# - Most of the passengers are young

# In[ ]:


sns.stripplot(x="Survived", y="Fare", data=df_train);


# In[ ]:


sns.swarmplot(x="Survived", y="Fare", data=df_train);


# In[ ]:


df_train.Fare.describe()


# In[ ]:


# Use the DataFrame method .describe() to check out summary statistics of 'Fare' as a function of survival.
df_train.groupby('Survived').Fare.describe()


# In[ ]:


sns.scatterplot(x="Age", y="Fare", hue="Survived", data=df_train, alpha=0.5);


# - Those who survived and paid low fare were more likely to be children.

# In[ ]:


sns.pairplot(data=df_train, hue="Survived");


# # 3

# In[ ]:


# Import modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Figures inline and set visualization style
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()

# Import data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


# In[ ]:


# target variable
survived_train = df_train.Survived
# concatenate train and test set (to perform same data manipulation on both datasets)
data = pd.concat([df_train.drop(['Survived'], axis=1), df_test])


# In[ ]:


data.info()


# We have 2 numerical columns with missing values, so perform imputation

# In[ ]:


data['Age'] = data.Age.fillna(data.Age.median())
data['Fare'] = data.Fare.fillna(data.Fare.median())
data.info()


# In[ ]:


data = pd.get_dummies(data, columns=["Sex"], drop_first=True)
data.head()


# In[ ]:


data = data[["Pclass", "Age", "SibSp", "Fare", "Sex_male"]]
data.head()


# In[ ]:


data_train = data.iloc[:891]
data_test = data.iloc[891:]


# In[ ]:


X = data_train.values
test = data_test.values
y = survived_train.values


# In[ ]:


clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)


# In[ ]:


Y_pred = clf.predict(test)
out = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': Y_pred})


# In[ ]:


out.to_csv('DecisionTree3.csv', index=False)


# Accuracy: 76.5%
# 
# Accuracy: 78% (2nd time due to random initialization)

# In[ ]:


# plt.figure(figsize=(10, 10))
# tree.plot_tree(clf.fit(X, y));


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42, stratify=y)


# In[ ]:


# Setup arrays to store train and test accuracies
dep = np.arange(1, 9)
train_accuracy = np.empty(len(dep))
test_accuracy = np.empty(len(dep))

# Loop over different values of k
for i, k in enumerate(dep):
    # Setup a k-NN Classifier with k neighbors: knn
    clf = tree.DecisionTreeClassifier(max_depth=k)

    # Fit the classifier to the training data
    clf.fit(X_train, y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = clf.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = clf.score(X_test, y_test)

# Generate plot
plt.title('clf: Varying depth of tree')
plt.plot(dep, test_accuracy, label = 'Testing Accuracy')
plt.plot(dep, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Depth of tree')
plt.ylabel('Accuracy')
plt.show()


# # 4

# In[ ]:


# Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from sklearn import tree
from sklearn.model_selection import GridSearchCV

# Figures inline and set visualization style
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()

# Import data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

# Store target variable of training data in a safe place
survived_train = df_train.Survived

# Concatenate training and test sets
data = pd.concat([df_train.drop(['Survived'], axis=1), df_test])

# View head
data.head()


# In[ ]:


data.Name.head()


# In[ ]:


data.Name.tail()


# we can extract titles from Name column to create a new feature

# In[ ]:


data['Title'] = data.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))
sns.countplot(data.Title)
plt.xticks(rotation=90);


# In[ ]:


data['Title'] = data['Title'].replace({'Mlle':'Miss', 'Mme':'Mrs', 'Ms':'Miss'})
data['Title'] = data['Title'].replace(['Don', 'Dona', 'Rev', 'Dr',
                                            'Major', 'Lady', 'Sir', 'Col', 'Capt', 'Countess', 'Jonkheer'],'Special')
sns.countplot(x='Title', data=data);
plt.xticks(rotation=90);


# In[ ]:


data[data.Cabin.isnull()].Fare.hist()


# Several NaN values in `Cabin`. This may suggest those people didn't have a Cabin because it is NaN for those who paid low fare as shown in plot above. 
# 
# So we can create a new feature `hasCabin` showing whether they had cabin or not.

# In[ ]:


data['hasCabin'] = ~data.Cabin.isnull()
data.head()


# In[ ]:


# drop columns ['PassengerId', 'Name', 'Ticket', 'Cabin']
data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
data.head()


# In[ ]:


data.info()


# We have missing values in columns `['Age', 'Fare', 'Embarked']`. Now we need to impute these missing values before we can proceed further

# In[ ]:


data['Age'] = data.Age.fillna(data.Age.median())
data['Fare'] = data.Fare.fillna(data.Fare.median())
data['Embarked'] = data.Embarked.fillna('S')    # as most of passsengers embarked from Southampton
data.info()


# Binning

# In[ ]:


data['CatAge'] = pd.qcut(data.Age, q=4, labels=False)
data['CatFare'] = pd.qcut(data.Fare, q=4, labels=False)
data.info()


# In[ ]:


# Now we can drop 'Age' and 'Fare' column
data.drop(['Age', 'Fare'], axis=1, inplace=True)
data.head()


# - SibSp: Number of siblings or spouse onboard
# - Parch: Number of parents or children onboard
# 
# So we can create a new feature 'Fam_Size' and drop these two colunns

# In[ ]:


data['FamSize'] = data.SibSp + data.Parch
data.head()


# In[ ]:


# drop 'SibSp' and 'Parch'
data.drop(['SibSp', 'Parch'], axis=1, inplace=True)
data.head()


# Now we need to convert non numerical columns to numerical columns.

# In[ ]:


data_dum = pd.get_dummies(data, drop_first=True)
data_dum.head()


# In[ ]:


data_train = data_dum[:891]
data_test = data_dum[891:]

X = data_train.values
y = survived_train.values
test = data_test.values


# Build model

# In[ ]:


# setup the hyperparameter grid
dep = np.arange(1, 9)
param_grid = {'max_depth': dep}

clf = tree.DecisionTreeClassifier()
clf_cv = GridSearchCV(clf, param_grid=param_grid, cv=5)
clf_cv.fit(X, y)

print("Tuned Decision Tree Parameters: {}".format(clf_cv.best_params_))
print("Best score is {}".format(clf_cv.best_score_))


# In[ ]:


y_pred = clf_cv.predict(test)


# In[ ]:


out = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': y_pred})
out.to_csv('feature_engg4.csv', index=False)


# Accuracy: 

# In[ ]:




