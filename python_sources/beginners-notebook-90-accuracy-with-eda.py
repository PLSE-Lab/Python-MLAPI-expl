#!/usr/bin/env python
# coding: utf-8

# <center><h1 style="color:green">Don't forget to upvote if you like it! It's free! :)</h1></center>

# # Contents:
# 1. Include Libraries
# 2. Import DataSet
# 3. EDA(Exploratory Data Analysis)
# 4. Handle Missing Value
# 5. Feature Engineering by OneHotEncoding
# 6. Logistic Regression
# 7. Hyperparameter Tunning
# 8. Train Random Forest Classifier
# 9. Final Submittion

# # Importing Libraries

# In[ ]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


from statistics import mode


# In[ ]:


import re


# # Load DataSet

# In[ ]:


train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')


# # Examine Dataset
# ## Look In to every column one by one 

# In[ ]:


train.head()


# In[ ]:


train.isnull().sum()


# In[ ]:


sns.countplot(train['Survived']);


# In[ ]:


sns.countplot(x='Survived', hue='Sex', data=train);


# ### Wow! 'Sex' looks like a very strong explanatory variable, and it can be our choice for our single feature Logistic Regression model!

# ### You can also find null values by plotting it on graph

# In[ ]:


sns.heatmap(train.isnull(), yticklabels = False, cmap='plasma');


# ### You can skip arguments other than x, cmap is styling the heatmap
# 

# In[ ]:


train.describe()


# In[ ]:


sns.countplot(train['Pclass']);


# In[ ]:


train.Name.value_counts().head()


# In[ ]:


train['Age'].hist(bins=40);


# ### You can always use value_counts to check on data, visualization is just another option 

# In[ ]:


train['SibSp'].value_counts()


# In[ ]:


sns.countplot(train['SibSp'])
plt.title('Count plot for SibSp');


# In[ ]:


sns.countplot(train['Parch'])
plt.title('Count plot for Parch');


# In[ ]:


train.Ticket.value_counts(dropna=False, sort=True).head()


# In[ ]:


train['Fare'].hist(bins=50)
plt.ylabel('Price')
plt.xlabel('Index')
plt.title('Fare Price distribution');


# In[ ]:


train.Cabin.value_counts(0)


# In[ ]:


sns.countplot(train['Embarked'])
plt.title('Count plot for Embarked');


# ### Look in to relationships among dataset

# In[ ]:


sns.heatmap(train.corr(), annot=True);


# ### annot argument is mandatory as you also need data value in each cell
# ### As you can see that Survived as max relation with Pclass, lets vizualize it in chart

# In[ ]:


sns.countplot(x='Survived', hue='Pclass', data=train)
plt.title('Count plot for Pclass categorized by Survived');


# ### Pclass and age, as they had max relation in the entire set we are going to replace missing age values with median age calculated per class

# # Let's Fix data

# In[ ]:


age_group = train.groupby('Pclass')['Age']


# In[ ]:


age_group.median()


# In[ ]:


age_group.mean()


# In[ ]:


train.loc[train.Age.isnull(), 'Age'] = train.groupby("Pclass").Age.transform('median')

train["Age"].isnull().sum()


# In[ ]:


sns.heatmap(train.isnull(), yticklabels = False, cmap='plasma');


# In[ ]:


train['Sex'][train['Sex'] == 'male'] = 0
train['Sex'][train['Sex'] == 'female'] = 1


# In[ ]:


train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2


# In[ ]:


train.head()


# ### You need to do the same changes in test dataset aslo...So lest merge test and train

# In[ ]:


df = pd.read_csv('../input/titanic/train.csv')


# In[ ]:


test['Survived'] = np.nan
full = pd.concat([df, test])


# In[ ]:


full.isnull().sum()


# In[ ]:


full.head()


# In[ ]:


full['Embarked'] = full['Embarked'].fillna(mode(full['Embarked']))


# In[ ]:


# Convert 'Sex' variable to integer form!
full["Sex"][full["Sex"] == "male"] = 0
full["Sex"][full["Sex"] == "female"] = 1

# Convert 'Embarked' variable to integer form!
full["Embarked"][full["Embarked"] == "S"] = 0
full["Embarked"][full["Embarked"] == "C"] = 1
full["Embarked"][full["Embarked"] == "Q"] = 2


# In[ ]:


sns.heatmap(full.corr(), annot=True);


# ### OK, if we look closely, corr(Age, Pclass) is the highest correlation in absolute numbers for 'Age', so we'll use Pclass to impute the missing values:

# In[ ]:


full['Age'] = full.groupby("Pclass")['Age'].transform(lambda x: x.fillna(x.median()))


# In[ ]:


full.isnull().sum()


# ### Also, corr(Fare, Pclass) is the highest correlation in absolute numbers for 'Fare', so we'll use Pclass again to impute the missing values!

# In[ ]:


full['Fare']  = full.groupby("Pclass")['Fare'].transform(lambda x: x.fillna(x.median()))


# In[ ]:


full['Cabin'] = full['Cabin'].fillna('U')


# In[ ]:


full['Cabin'].unique().tolist()[:20]


# ### Did you recognize something? yes, We can get the alphabets(first letter) by running regular expression 

# In[ ]:


full['Cabin'] = full['Cabin'].map(lambda x:re.compile("([a-zA-Z])").search(x).group())


# In[ ]:


full['Cabin'].unique().tolist()


# In[ ]:


cabin_category = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'T':8, 'U':9}
full['Cabin'] = full['Cabin'].map(cabin_category)


# In[ ]:


full['Cabin'].unique().tolist()


# ### Good practice to check the results

# In[ ]:


full['Name'].head()


# In[ ]:


full['Name'] = full.Name.str.extract(' ([A-Za-z]+)\.', expand = False)


# In[ ]:


full['Name'].unique().tolist()


# ### Wohh that's lot's of title

# In[ ]:


full['Name'].value_counts(normalize = True) * 100


# ### Whoops! Apart from Mr, Miss, Mrs, and Master, the rest have percentages close to zero...
# 
# ### So, let's bundle them!

# In[ ]:


full.rename(columns={'Name' : 'Title'}, inplace=True)


# In[ ]:


full['Title'] = full['Title'].replace(['Rev', 'Dr', 'Col', 'Ms', 'Mlle', 'Major', 'Countess', 
                                       'Capt', 'Dona', 'Jonkheer', 'Lady', 'Sir', 'Mme', 'Don'], 'Other')


# In[ ]:


full['Title'].value_counts(normalize = True) * 100


# ### Better! let's convert to numeric

# In[ ]:


title_category = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Other':5}
full['Title'] = full['Title'].map(title_category)
full['Title'].unique().tolist()


# ### Hmmm... but we know from part 2 that Sibsp is the number of siblings / spouses aboard the Titanic, and Parch is the number of parents / children aboard the Titanic... So, what is another straightforward feature to engineer?
# 
# ### Yes, it is the size of each family aboard!

# In[ ]:


full['familySize'] = full['SibSp'] + full['Parch'] + 1


# In[ ]:


# Drop redundant features
full = full.drop(['SibSp', 'Parch', 'Ticket'], axis = 1)


# In[ ]:


full.head()


# In[ ]:


# Recover test dataset
test = full[full['Survived'].isna()].drop(['Survived'], axis = 1)


# In[ ]:


test.head()


# In[ ]:


# Recover train dataset
train = full[full['Survived'].notna()]


# In[ ]:


train['Survived'] = train['Survived'].astype(np.int8)


# # Dateset is completely ready now!

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train.drop(['Survived', 'PassengerId'], axis=1), train['Survived'], test_size = 0.2, random_state=2)


# In[ ]:


from sklearn.linear_model import LogisticRegression
LogisticRegression = LogisticRegression(max_iter=10000)
LogisticRegression.fit(X_train, y_train)


# In[ ]:


predictions = LogisticRegression.predict(X_test)
predictions


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions)


# In[ ]:


acc = (87+54) / (87+54+13+25) * 100
acc


# # Magic Weapon #2: Cross-Validation

#  One of the most popular and efficient CV variants is **k-Fold Cross-Validation**, which we will choose to set our strong local validation scheme below. In a nutshell, k is the number of folds, mentioned above!
# 
#  Nice, now let's apply this key technique ourselves! We will use the basic version of k-Fold with **5 folds** from our friend, Scikit-learn!

# In[ ]:


from sklearn.model_selection import KFold
kf = KFold(n_splits = 5, random_state=2)


# In[ ]:


from sklearn.model_selection import cross_val_score

cross_val_score(LogisticRegression, X_test, y_test, cv = kf).mean() * 100


# ### Didn't Work

# # Magic Weapon #3: Hyperparameter Tuning

#  Secondly, I would like to introduce one of the most popular algorithms for classification (but also regression, etc), **Random Forest!** In a nutshell, Random Forest is an ensembling learning algorithm which combines **decision trees** in order to increase performance and avoid overfitting.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
RandomForest = RandomForestClassifier(random_state=2)


# Below we set the hyperparameter grid of values with 4 lists of values:
# 
# - **'criterion'** : A function which measures the quality of a split.
# - **'n_estimators'** : The number of trees of our random forest.
# - **'max_features'** : The number of features to choose when looking for the best way of splitting.
# - **'max_depth'** : the maximum depth of a decision tree.

# In[ ]:


# Set our parameter grid
param_grid = { 
    'criterion' : ['gini', 'entropy'],
    'n_estimators': [100, 300, 500],
    'max_features': ['auto', 'log2'],
    'max_depth' : [3, 5, 7]    
}


# In[ ]:


from sklearn.model_selection import GridSearchCV

randomForest_CV = GridSearchCV(estimator = RandomForest, param_grid = param_grid, cv = 5)
randomForest_CV.fit(X_train, y_train)


# #### Let's print our optimal hyperparameters set!

# In[ ]:


randomForest_CV.best_params_


# In[ ]:


randomForestFinalModel = RandomForestClassifier(random_state = 2, criterion = 'gini', max_depth = 7, max_features = 'auto', n_estimators = 300)

randomForestFinalModel.fit(X_train, y_train)


# In[ ]:


predictions = randomForestFinalModel.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score

accuracy_score(y_test, predictions) * 100


# ### Let's submit our solutions

# In[ ]:


test['Survived'] = randomForestFinalModel.predict(test.drop(['PassengerId'], axis = 1))


# In[ ]:


test[['PassengerId', 'Survived']].to_csv('MySubmission.csv', index = False)


# In[ ]:


test.info()


# # Plz Upvote!
