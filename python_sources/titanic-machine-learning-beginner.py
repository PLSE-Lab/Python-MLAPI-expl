#!/usr/bin/env python
# coding: utf-8

# ###Loading the data 

# In[ ]:


import pandas as pd
from pandas import Series,DataFrame


# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# ###Preview of the train data

# In[ ]:


train_df.head()


# In[ ]:


train_df.describe(include='all')


# ###EDA

# Explore the distribution to identify whether there is a skewed class distribution problem before analyzing the data in a machine learning model. 
# 

# In[ ]:


import seaborn as sns


# In[ ]:


sns.countplot('Survived',data=train_df)


# Sex and Pclass are very importance features for survival.

# In[ ]:


train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train_df[["Pclass", "Survived"]].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# This is a cruel fact. Male in class1 and class2 are still have more survival rate than female in class1 and class2. 

# In[ ]:


grid = sns.FacetGrid(train_df, row='Pclass', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex','Survived', alpha=.5, ci=None)
grid.add_legend()


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
grid = sns.FacetGrid(train_df, col='Survived', row='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
grid = sns.FacetGrid(train_df, row='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();


# ### Data Cleansing 1: Drop Variables

# drop the PassengerID and Cabin column

# In[ ]:


df = pd.DataFrame(train_df,columns=['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'])




# In[ ]:


df = df.drop(['Cabin','PassengerId','Ticket'], axis = 1)


# In[ ]:


df.head()


# In[ ]:


test_df.head()


# In[ ]:


test = pd.DataFrame(test_df,columns=['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'])



# In[ ]:


test = test.drop(['Cabin','Ticket'], axis = 1)


# In[ ]:


test.head()


# ### Data Cleansing 2 : Imputation
# impute the age with the median 
# impute the Embarked with the most frequent value

# In[ ]:


median = df['Age'].mean()
df['Age']= df['Age'].fillna(median)


# In[ ]:


sns.countplot('Embarked',data=train_df)


# In[ ]:


import matplotlib.pyplot as plt
plt.subplot(121)
sns.boxplot('Pclass', 'Fare', 'Survived', df, orient='v')


# In[ ]:


df.groupby(['Embarked']).mean()


# In[ ]:


df['Embarked'] = df['Embarked'].fillna('S')


# Check whether the imputation is successful. 
# 

# In[ ]:


df.Age.isnull().any()


# In[ ]:


df.Embarked.isnull().any()


# In[ ]:


df.info()


# In[ ]:


median_test = test['Age'].median()
test['Age']= test['Age'].fillna(median_test)
fare = test['Fare'].median()
test['Fare'] = test['Fare'].fillna(fare)
test.info()


# ###Data Cleansing 3: Variable Modification

# collect all the title such as Jr, Mirs, or Miss and make it as a new variable
# 1- honor, 0 - common

# In[ ]:


total_data = [df,test]


# In[ ]:


for dataset in total_data:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)


# In[ ]:


for dataset in total_data:
    dataset['Title'] = dataset['Title'].replace(['Capt','Col','Countess','Don','Dona','Dr','Jonkheer','Lady','Major','Master','Rev','Sir'], 'Honor')

    dataset['Title'] = dataset['Title'].replace(['Mlle', 'Miss','Ms','Mme','Mrs','Mr'],'Common')
    
df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[ ]:


grid = sns.FacetGrid(df, row='Title', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex','Survived', alpha=.5, ci=None)
grid.add_legend()


# In[ ]:


df.groupby(['Title']).mean()


# In[ ]:


test.head()


# Honor - 1 , common - 0

# In[ ]:


for dataset in total_data:
    dataset['Title'] = dataset['Title'].map( {'Honor': 1, 'Common': 0} ).astype(int)


# In[ ]:


df = df.drop('Name', axis = 1)
test = test.drop('Name',axis = 1)


# changing string variable to numeric variable
# Male = 1, female = 0

# In[ ]:


from sklearn import preprocessing

le = preprocessing.LabelEncoder()

df['Gender'] = le.fit_transform(df.Sex)

test['Gender'] = le.fit_transform(test.Sex)


# In[ ]:


df = df.drop('Sex', axis = 1)
test = test.drop('Sex',axis = 1)


# Embarked: "C": 0, "S": 2,"Q": 1

# In[ ]:


from sklearn import preprocessing

le = preprocessing.LabelEncoder()

df['Embarked'] = le.fit_transform(df.Embarked)

test['Embarked'] = le.fit_transform(test.Embarked)


# In[ ]:


df.head()


# In[ ]:


test.head()


# In[ ]:


total_data = [df,test]


# ## Machine Learning: Logistic Regression and Tree Models

# ###Logistic Regression

# In[ ]:


X_train = df.drop("Survived", axis=1)
Y_train = df["Survived"]
X_test  = test.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# In[ ]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# In[ ]:


# get Correlation Coefficient for each feature using Logistic Regression
coeff_df = DataFrame(df.columns.delete(0))
coeff_df.columns = ['Features']
coeff_df["Coefficient Estimate"] = pd.Series(logreg.coef_[0])

# preview
coeff_df


# ### Decision Tree and Random Forest
# Use Decision Tree and Random Forest to predict the result. 

# In[ ]:


from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# Measure the feature importance 

# In[ ]:


clf = DecisionTreeClassifier(random_state=0)
clf = clf.fit(X_train, Y_train)
print(dict(zip(X_train.columns, clf.feature_importances_)))


# In[ ]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred2 = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# #### Submission

# In[ ]:


feature = ['Pclass', 'Age', 'SibSp', 'Parch','Fare','Embarked','Title','Gender']


# In[ ]:


prediction = clf.predict(test[feature])
prediction


# In[ ]:



submission_randomforest = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':prediction})


# In[ ]:


filename = 'Titanic Prediction Beginner.csv'

submission_randomforest.to_csv(filename, index = False)

print('Saved file:' + filename)

