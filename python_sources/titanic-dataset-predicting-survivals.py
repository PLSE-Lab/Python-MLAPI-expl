#!/usr/bin/env python
# coding: utf-8

# **Titanic Dataset. Predicting Survivals.**

# In[ ]:


# Load the neccessary Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# Load the Training Dataset into pandas dataframe. We have never seen the Test dataset before, So do not bother for the moment.

# In[ ]:


df = pd.read_csv('../input/train.csv')


# **Get to know your data. Before any analysis, start to get a feeling on your data.**

# In[ ]:


# Explore the dataset to see values, datatypes, nan values
df.info()


# Previous output indicates that data is made of numbers (integer, floats) and some other types, which might represent categorical variables.
# Moreover some of the columns are incomplete. Each of them must have 891 entries but columns like: "Cabin", "Embarked", "Age" has a different number of non-null entries.
# Computation is done with arrays containing number so this tells us that there is some work to do upfront to pre-process the data. A particular observation here is that Cabin information is available only for 204/891 = 22.9% of the samples, so the question is: do we need this? Would simplify the analysis if we remove this but for the moment is unclear and is better to keep the data.

# In[ ]:


# data dimensions
df.shape


# In[ ]:


# show some initial rows 
df.head(10)


# It is a good idea, after looking to the columns, understand what kind of data do they have? Usually description is in some code books. Here some of the column names are explicit enough but others?
# Parch = Number of parents/children aboard (vertical relationship)
# SibSp = Number of siblings/spouses aboard (horizontal relationship)
# 
# This dataset is made of a) binary variables (Survived, Sex) and other Categorical variables (Embarked, Cabin, Ticket, Name, Pclass). The next step is to explore the categorical variables and their dimension.

# In[ ]:


# For the Cathegorical variables we should see the set of their values over all samples.
print('Pclass:   ', df['Pclass'].unique())
print('Parch:    ', df['Parch'].unique())
print('SibSp:    ', df['SibSp'].unique())
print('Embarked: ', df['Embarked'].unique())

# Let see if the 209 cabines in the dataset host each one person or more
print('Cabin:    ', df['Cabin'].unique().size)

# Are the Ticket unique
print('Ticket:   ', df['Ticket'].unique().size)


# There are 3 classes in the ship. There are 148 different Cabines containing 209 people. There are only 681 ticket for 891 people. Maybe children were registered with parents in the same ticket?

# In[ ]:


# It makes sense to extract those columns which contain booleant values like 0-1 or Trus - False. 
colist = df.columns.tolist()
for col in colist:
    if df[col].unique().size == 2:
        print(col) 


# Survived and Sex are binary values which was expected. If the features are many than this code is useful. The idea can further be extended with ternary values or more. One observation at this point is that most of the data set have categorical values with limited number of values and binary values. 

# **Next step is about manipulating the data set, by Filling missing Values and assigning a number to the categorical variable for the next part of the analysis.**

# In[ ]:


# Embarked have 2 missing values. With low error we can fill them based on the most likely value.
print(df[df['Embarked'] == 'S']['Embarked'].size)
print(df[df['Embarked'] == 'Q']['Embarked'].size)
print(df[df['Embarked'] == 'C']['Embarked'].size)


# In[ ]:


# The most recurrent value is "S" therefore we can fill the missing values in Embarked with S and replace labels with 
# numbers
df["Embarked"] = df["Embarked"].fillna('S')
df['Embarked'] = df['Embarked'].map({'S': 2, 'C': 1, 'Q':0})


# In[ ]:


# Could replace SibSp and Parch with one column, called Family, indicating the overall family number. 
# I do not see much difference if one is a brother or a sister from beeing a mother or a father.
df['Family'] = df['SibSp'] + df['Parch'] + 1


# In[ ]:


# Consider now replacing the Sex type from a label to a number.
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})


# In[ ]:


# Consider the distribution of the age
age_mean = df['Age'].mean()
age_std  = df['Age'].std()

# generate random numbers age_mean
rand_age = np.random.randint(age_mean - age_std, age_mean + age_std, size = df["Age"].isnull().sum())

# fill in the randome values to the missing age values
df["Age"][np.isnan(df["Age"])] = rand_age

# convert from float to int
df['Age'] = df['Age'].astype(int)

# There are some columns which can be dropped already
df.drop(['Parch', 'SibSp', 'PassengerId'], axis=1 , inplace=True)

# finally look at the dataset
df.info()
df.head(6)


# **Ticket, Cabin, Fare, Class, Family have some relation together. Higher Fare realate to tickets acquired for first class cabines. Surely there is some redundant information in here. We could drop Cabin and Ticket in favor of Fare and Pclass. We could also drop 'Name' as we got Sex, Family, Class**

# In[ ]:


df.drop(['Cabin', 'Ticket', 'Name'], axis=1 , inplace=True)
df.info()
df.head(6)


# At this moment we have a dataset with all Numbers and want to start seeing some initials tatistic and trends.

# **Some Initial Statistics**

# Pandas dataframe is a good tool to get a quick statistical indicators over all the data. Examples are toals, mean, standard deviations, min and max. Cathegorical features will be automatically removed. Although for some features the values are no meaningful the analysis still is good value for time, as it allows to get a quick overview and potentially highlite interesting cases/values.

# In[ ]:


df.describe()


# - Average age of the passengers is around 30. 
# - Age might be an important parameter related to the chances to survive.
# - Only 38% of the passengers did survived. Mean is the number of 1 (survive) divided the total number.
# - 64% of the passengers are men.
# For the moment other indicators seems not that meaningful. Need other ways to see trends

# In[ ]:


# To better consider correlation between the various fields, we might use a correlation matrix.
correlation_matrix = df.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation_matrix, vmax=1, square=True, annot=True, cmap='Reds')
plt.title('Overall Correlation Matrix')
plt.show()


# There is some relation with "Fare". Maybe people who paid higher Fair had some more chanches to survive. This is confirmed also with the "Pclass" the lower the class the higher the chances to survive. Maybe Reach people could get better located places in the ship so that they could leave the ship faster then others and/or get to the rescue boats. We could drop Fair in favor of the Pclass as the last seems to be a stronger indicator. Sex and Age seems to be indicator related to survival. Other indicators are weak and we need other investigation.

# In[ ]:


df.drop(['Fare'], axis=1, inplace=True)
df.head(4)


# In[ ]:


# collect all the survival
Survive = df[df['Survived'] == 1]

# create groups based on them
by_families = Survive.groupby(df['Family'])
by_age = Survive.groupby(df['Age'])
by_embark = Survive.groupby('Embarked')
by_sex = Survive.groupby('Sex')
by_class = Survive.groupby('Pclass')

## See how the groups relates with Survived
#print(df.groupby(df['Family'])['Survived'].count())
print(by_families['Survived'].count() / df.groupby(df['Family'])['Survived'].count() * 100 * (df.groupby(df['Family'])['Survived'].count()/891))
print(by_embark['Survived'].count() / df.groupby(df['Embarked'])['Survived'].count() * 100 * (df.groupby(df['Embarked'])['Survived'].count()/891))
print(by_sex['Survived'].count() / df.groupby(df['Sex'])['Survived'].count() * 100 * (df.groupby(df['Sex'])['Survived'].count()/891))
print(by_class['Survived'].count() / df.groupby(df['Pclass'])['Survived'].count() * 100 * (df.groupby(df['Pclass'])['Survived'].count()/891))
#print(by_age['Survived'].count() / df.groupby(df['Age'])['Survived'].count() * 100 *(df.groupby(df['Age'])['Survived'].count()/891))
age_distribution = by_age['Survived'].count() / df.groupby(df['Age'])['Survived'].count() * 100 * (df.groupby(df['Age'])['Survived'].count()/891)
age_distribution.hist()


# There is some learning to be done here. Sometimes initial trends are not as they seems. we should looka at their percentage parts and their weigth among all the population. As an example, it seems that the families of 4 people have a survivale rate of 72%, but thy represent only 3.2% of the entire population, so how strong is this value in order to be considered as a generic trend indicator? One way is to weight this according to their rapresentation of the population. If we do this we can extrapolate a strong tren on smaller familie having more chances to survive. Ideally if one was alone had to take care only on him/herself increasing the chances. Wemen seems to have double the chances to survive with respect to men. Maybe first women and kids to the saving boats :). Who embarked from "S" have higher chances than others to survive. I do not know why is this. Maybe this was the last place where people could take the ship. Class seems to be a weak indicator though, differently from what was seen before. There is a clear trend as well on younger people to have more chances to survive. 

# **Preparing Numpy Arrays for Estimation**

# In[ ]:


# Prepare the data. One of the first things to do is to split between the target to be predicted Y, and the variable 
Y = df['Survived'].values
np.shape(Y)


# In[ ]:


# before creating the feature matriy we need to drop the target "Survived"
df.drop(['Survived'], axis=1 , inplace=True)
df.info()
df.head(4)


# In[ ]:


# create features matrix
X = df.as_matrix()
print(np.shape(X))

# show the second row
X[1]


# **Prepare Test Data**

# In[ ]:


# load and prepare the test data.
df_test = pd.read_csv('../input/test.csv')
df_test.info()


# In[ ]:


# We already know the data to be dropped
ID_test = df_test['PassengerId'].values
df_test.drop(['Cabin', 'Fare', 'Ticket', 'PassengerId', 'Name'], axis=1, inplace=True)
df_test.head(3)


# In[ ]:


# Filling missing data from the Age
age_mean = df_test['Age'].mean()
age_std  = df_test['Age'].std()
rand_age = np.random.randint(age_mean - age_std, age_mean + age_std, size = df_test["Age"].isnull().sum())
df_test["Age"][np.isnan(df_test["Age"])] = rand_age
df_test['Age'] = df_test['Age'].astype(int)

# Consider now replacing the Sex type from a label to a number.
df_test['Sex'] = df_test['Sex'].map({'male': 1, 'female': 0})

# Could replace SibSp and Parch with one column, called Family
df_test['Family'] = df_test['SibSp'] + df_test['Parch']

df_test['Embarked'] = df_test['Embarked'].map({'S': 2, 'C': 1, 'Q':0})

# We can drop now SibSp and Parch
df_test.drop(['Parch', 'SibSp'], axis=1 , inplace=True)

df_test.info()
df_test.head(4)


# In[ ]:


# create features matrix
X_test = df_test.as_matrix()
print(np.shape(X_test))


# **Classifiers**

# In[ ]:


# Linear Regression
from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression()
logistic.fit(X, Y)
Y_pred_lg = logistic.predict(X_test)
logistic.score(X, Y)


# In[ ]:


# Random Forests
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X, Y)
Y_pred_rf = random_forest.predict(X_test)
random_forest.score(X, Y)


# In[ ]:


# Support Vector Machine
from sklearn.svm import SVC
svc = SVC(C=1, kernel='rbf').fit(X, Y)
Y_pred_svc = svc.predict(X_test)
svc.score(X, Y)


# In[ ]:


from sklearn import tree
clf = tree.DecisionTreeClassifier().fit(X, Y)
Y_pred_clf = clf.predict(X_test)
clf.score(X, Y)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knb = KNeighborsClassifier(n_neighbors = 3)
knb.fit(X, Y)
Y_pred_knb = knb.predict(X_test)
knb.score(X, Y)


# In[ ]:


from sklearn.naive_bayes import GaussianNB
gaus = GaussianNB()
gaus.fit(X, Y)
Y_pred_gs = gaus.predict(X_test)
gaus.score(X, Y)


# **There are several classifiers observed in here. What we could do is consider a voting system between them. We remove the one with lowest score so to have odd number of voters.**

# In[ ]:


Y_pred = Y_pred_gs + Y_pred_knb + Y_pred_clf + Y_pred_svc + Y_pred_rf
type(Y_pred)


# In[ ]:


# If 3 or more out of 5 predictors vote for 1 than we have Survived = 1 otherwise we have Survived = 0
Y_pred[Y_pred < 2.5] = 0
Y_pred[Y_pred > 2.5] = 1
Y_pred


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": ID_test,
        "Survived": Y_pred
    })
    
submission.to_csv("titanic_solution.csv", index=False)


# In[ ]:




