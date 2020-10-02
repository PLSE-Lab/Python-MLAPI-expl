#!/usr/bin/env python
# coding: utf-8

# # A Canonical Data Science Procedure
# 
# ---
# 
# ## Contents
# 1. **Data preprocessing**
#     1. Read in data
#     2. See the data
#     3. Create & initialize
#     4. Cleaning
#     5. Normalize
# 
# 2. **Feature Engineering**
#     1. Prepare features to be analyzed
#     2. Analyze features
#     3. Construct features
#     
# 3. **Models**
#     1. Decision tree (0.76555)
#     2. Random forest (0.78469)
# 
# 4. **Save Data**
# 
# 5. **References**
# 
# To be continue ...
# 
# ---
# 

# # 1.  Data preprocessing
# 
# ## 1.1  Read in data

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
combine = [train, test]


# ## 1.2  See the data
# 
# **Use `.head()` or `.tail()`**
# 
# We can see the type and the name of different columns, so that distinguish the numerical and categorial, or alphanumerical features.
# 
# The meaning of each column is explained in the [data description](https://www.kaggle.com/c/titanic/data).

# In[ ]:


train.head()


# **Use `.info()`**
# 
# Very clear. You can see the type of data and spot features with nulls. When the count of one column is less than the whole, it may have missing values ( See column "Age" ).
# 
# **Use `.unique()`**
# 
# See how many different values are in a column.
# 
# **Use `.value_counts()`**
# 
# See how many different values are in the column, and their counts.

# In[ ]:


train.info()
print('_'*80)
print("Unique value in 'Sex': ", train["Sex"].unique())
print("Value counts: ", train["Sex"].value_counts())


# **Use `.describe()` for numerical data**
# 
# In this way we can get some statistical characters of data.
# We can get some observation[1] :
# 
# 1. Data type
#     - Survived is a categorical feature with 0 or 1 values.
# 
# 2. Quantity
#     - Total samples are 891 or 40% of the actual number of passengers on board the Titanic (2,224).
#     - Around 38% samples survived representative of the actual survival rate at 32%.
# 
# 3. Distribution
#     - Most passengers (> 75%) did not travel with parents or children.
#     - Nearly 30% of the passengers had siblings and/or spouse aboard.
#     - Fares varied significantly with few passengers (<1%) paying as high as $512.
#     - Few elderly passengers (<1%) within age range 65-80.

# In[ ]:


train.describe()


# **Use `.describe( include=[ 'O' ] )` for categorical data**
# 
# Observation: 
# 
# 1. Uniqueness
#     - Names are unique across the dataset (count=unique=891)
#     - Cabin values have several dupicates across samples. Alternatively several passengers shared a cabin.
#     - Ticket feature has high ratio (22%) of duplicate values (unique=681).
# 
# 2. Proportion
#     - Sex variable as two possible values with 65% male (top=male, freq=577/count=891).
#     - Embarked takes three possible values. S port used by most passengers (top=S)

# In[ ]:


train.describe(include=['O'])


# ## 1.3 Initialize a new column
# 
# * Use 0
# * Use "NaN"

# In[ ]:


train["family_size"] = float("NaN")
test["family_size"] = float("NaN")


# ## 1.4 Cleaning
# **Null** 
# 
# * Use pandas.DataFrame method: .fillna()
# 
# 

# In[ ]:


for dataframe in combine:
    dataframe["Embarked"] = dataframe["Embarked"].fillna("S")
    dataframe["Age"] = dataframe["Age"].fillna(dataframe["Age"].median())
test.Fare = test.Fare.fillna(test["Fare"].median())


# ## 1.5 Convert data form
# 
# * Use a second `[]`-operator to serve a boolean test

# In[ ]:


for dataframe in combine:
    # Convert the male and female groups to integer form
    dataframe['Sex'] = dataframe['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    # Convert the Embarked classes to integer form
    dataframe['Embarked'] = dataframe['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
# Or use commands like this
# train.loc[train["Sex"] == "male", "Sex"] = 0


# # 2. Feature Engineering[1]
# 
# ## 2.1 Prepare features to be analyzed
# 
# ### Drop some features
# 
# **1. With no correlation of the target**
# 
# Such as "Ticket". It contains high ratio of duplicates (22%) and there may not be a correlation between Ticket and survival. And "PassengerId".
# 
# **2. Incomplete of contains too many nulls both in training and test dataset.**
# 
# Like "Cabin" in this problem.
# 
# **3. Hard to normalize**
# 
# Name feature is relatively non-standard, may not contribute directly to survival, so maybe dropped.
# 
# ### Create features
# 
# Maybe we can:
# 
# - Create "family_size", based on Parch and SibSp to get total count of family members on board.
# - Engineer the Name feature to extract Title as a new feature.
# - Create Age bands. This turns a continous numerical feature into an ordinal categorical feature.
# - Create a Fare range feature.
# 
# ### Select with experiential information
# 
# - Women (Sex=female) were more likely to have survived.
# - Children were more likely to have survived.
# - The upper-class passengers (Pclass=1) were more likely to have survived.
# 
# So we would want to analyze these features to see if they really have effects on the survival results.

# In[ ]:


# Create features
train["family_size"] = train["SibSp"] + train["Parch"] + 1
test["family_size"] = test["SibSp"] + test["Parch"] + 1


# ## 2.2 Analyze features

# 
# ## Pivot features
# To confirm some of our observations and assumptions, we can quickly analyze our feature correlations by pivoting features against each other. We can only do so at this stage for features which **do not have any empty values**. It also makes sense doing so only for features which are **categorical (Sex), ordinal (Pclass) or discrete (SibSp, Parch)** type.
# 
# **Use `.groupby()`**
# 
# Pclass: We can see this ordinal feature have a high correlation with Survived.

# In[ ]:


train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# Similarly, Sex also significant.

# In[ ]:


train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# We can test on the family_size, SibSp, and Parch:

# In[ ]:


train[["family_size", "Survived"]].groupby(['family_size'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# ## Visualize Data
# Use seaborn and matplotlib

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# **1. Correlating numerical features**
# 
# **Histograms:** Helpful for analyzing continuous numberical data. It can indicate distribution of samples using automatically defined bins or equally ranged bands.

# In[ ]:


g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=20)


# ## 2.3 Construct features

# In[ ]:


# Construct features and the target
features = train[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "family_size", "Embarked"]].values
features_test = test[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "family_size", "Embarked"]].values
target = train["Survived"]


# # Models
# 
# ## Decision Tree
# 
# Result: 0.76555

# In[ ]:


from sklearn import tree

# Train on a tree
decision_tree = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)
decision_tree = decision_tree.fit(features, target)
print(decision_tree.score(features, target))
print(decision_tree.feature_importances_)

# Make prediciton
prediction_dt = decision_tree.predict(features_test)
PassengerId =np.array(test["PassengerId"]).astype(int)
solution_dt = pd.DataFrame(prediction_dt, PassengerId, columns = ["Survived"])
solution_dt.to_csv("solution_dt.csv", index_label = ["PassengerId"])


# ## Random Forest
# 
# Result: 0.78469

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

# Train on a tree
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)
random_forest = forest.fit(features, target)
print(random_forest.score(features, target))
print(random_forest.feature_importances_)

# Make prediciton
prediction_rf = random_forest.predict(features_test)
solution_rf = pd.DataFrame(prediction_rf, PassengerId, columns = ["Survived"])
solution_rf.to_csv("solution_rf.csv", index_label = ["PassengerId"])


# # Save data
# 
# * Use [pandas.DataFrame](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html)
# 
#     pd.DataFrame( data, index, columns, dtype, copy )

# In[ ]:


ID = np.arange(0,10)
age = np.arange(10,20)
people = pd.DataFrame(age, ID, columns = ["Age"])
people.to_csv("people.csv", index_label = ["ID"])


# Or in the following form. But note that, in this way the order of the column is determined by the initials of their names.

# In[ ]:


people = pd.DataFrame({"ID":ID, "Age":age})
people.to_csv("People.csv", index=False)


# # References
# 
# [1]  [Titanic Data Science Solutions, Manav Sehgal](https://www.kaggle.com/startupsci/titanic/titanic-data-science-solutions/run/1145136)
# 
# [2] [An Interactive Data Science Tutorial](https://www.kaggle.com/helgejo/an-interactive-data-science-tutorial/notebook)
# 
# [3] [Kaggle Python Tutorial on Machine Learning](https://www.datacamp.com/community/open-courses/kaggle-python-tutorial-on-machine-learning#gs.vvoPxL4)
# 
# [4] [Getting Started with Kaggle](https://www.dataquest.io/m/32/getting-started-with-kaggle)
