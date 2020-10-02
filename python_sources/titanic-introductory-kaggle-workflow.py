#!/usr/bin/env python
# coding: utf-8

# # TITANIC - INTRODUCTORY KAGGLE WORKFLOW
# 
# # Intro
# 
# This notebook serves as both my first Kaggle submission(s) as well as an attempt to establish a personal baseline Kaggle workflow. It incorporates the techniques from some of the popular Kaggle kernels available for the Titanic competition, as well as influences from outside of Kaggle. I want this to serve as a decent reference, both for myself and others, so I have explained things to a fair level of detail. I don't explain every method or function call, or go into detail about how things like the `pandas` library works, but even a beginner should be able to follow along.
# 
# ## _References/Inspiration_
# 
# [Introduction to Ensembling/Stacking in Python](https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python)
# 
# [Titanic Data Science Solutions](https://www.kaggle.com/startupsci/titanic-data-science-solutions)
# 
# [Titanic Top 4% with ensemble modeling](https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling)
# 
# # Dataset
# 
# ## _Source_
# 
# https://www.kaggle.com/c/titanic/data
# 
# ## _Description_
# 
# Dataset containing 891 samples of passenger data labeled for survival, split into `train` and `test` sets. The `train` set contains 12 features while the `test` set contains 11 features.
# 
# ## _Goal_
# 
# Predict the survival of a passenger based on the available features.
# 
# ## _Data Dictionary_
# 
# |Variable|Definition|Key|
# |:--|:--|:--|
# |**survival**|Survival|No = 0, Yes = 1|
# |**pclass**|Ticket class|First Class = 1, Second Class = 2, Third Class = 3|
# |**name**|Passenger name||
# |**sex**|Gender||
# |**age**|Age in years||
# |**sibsp**|# of siblings / spouses aboard the ship||
# |**parch**|# of parents / children aboard the ship||
# |**ticket**|Ticket number||
# |**fare**|Passenger fare||
# |**cabin**|Cabin number||
# |**embarked**|Port of Embarkation|Cherbourg = C, Queenstown = Q, Southampton = S|
# 
# ## _Data Notes_
# 
# **pclass**: Approximate representation of socio-economic status
# - 1st = Upper Class
# - 2nd = Middle Class
# - 3rd = Lower Class
# 
# **age**: Is fractional if less than 1. If age is estimated, will take the form x.5
# 
# **sibsp**:
# - _Sibling_ = brother, sister, stepbrother, stepsister
# - _Spouse_ = husband, wife (mistresses and fiances are ignored)
# 
# **parch**:
# - _Parent_ = mother, father
# - _Child_ = daughter, son, stepdaughter, stepson
# - Some children travelled only with a nanny, and thus parch = 0 in those instances
# 
# # Setup
# 
# ## _Import Libraries_

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import warnings
import xgboost as xgb

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# ## _Configuration_

# In[2]:


# This magic method displays matplotlib plots inside of Jupyter notebooks instead of in a separate window
get_ipython().run_line_magic('matplotlib', 'inline')

# Set plot style using the Seaborn library
sns.set_style("whitegrid")

# Supress the oh so common pandas warnings; use at own risk
warnings.filterwarnings("ignore")


# # Load Data
# 
# We're going to load both train.csv and test.csv into pandas DataFrames. A DataFrame is sort of like an in-memory spreadsheet and an essential tool when working with datasets in Python.
# 
# Why two datasets? The `train` set is used to train the machine learning model. The `test` set is used to test how well the machine learning model works.

# In[3]:


train = pd.read_csv("../input/train.csv")


# In[4]:


# The head() method shows the first rows of a DataFrame; tail() shows the last rows
train.head()


# In[5]:


test = pd.read_csv("../input/test.csv")


# In[6]:


test.head()


# Save the 'PassengerId' from the `test` data because we'll need it when we submit our predictions:

# In[7]:


passenger_id = test["PassengerId"]


# ## _Join Train and Test into One DataFrame_
# 
# A big part of the machine learning workflow is cleaning and transforming data. An important point is that we want to _evaluate_ the `train` dataset but _transform_ data in both datasets. The `test` dataset represents new data that we don't know about. In the 'real world', we would train a model and then use that model to make predictions on new data. The training data _influences_ the model while new data (`test`) does not. Thus we make decisions based on the `train` dataset but not `test`.
# 
# Now before making predictions, even new data needs to be changed. Machine learning algorithms almost universally will not work with text data, so we either need to remove that text data or transform it into numerical data. _This has to happen before making predictions_. So if we drop a column or transform the values in `train`, we should do the same thing in `test`.
# 
# So that data transformations can be done with less code, I am going to create a [hierarchical DataFrame](http://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html) by concatenating the `train` and `test` DataFrames together. The drawback is that there is now an extra level of indexing, which makes selecting data a little more complicated. There are alternative approaches, such as looping through a list of DataFrames, handling each DataFrame individually, or using scikit-learn pipelines.

# In[8]:


all_data = pd.concat([train, test], keys=["Train", "Test"], names=["Dataset", "Id"], sort=False)


# In[9]:


all_data.head()


# In[10]:


all_data.loc["Train"].head()


# In[11]:


all_data.loc["Test"].head()


# # Inspect the Data

# In[12]:


all_data.loc["Train"].info()


# In the training dataset, both 'Age' and 'Cabin' contain a substantial number of null values. 'Embarked' also contains 2 null values.
# 
# The 5 features with a data type of `object` are strings.

# In[13]:


all_data.loc["Test"].info()


# In the testing dataset, 'Age' and 'Cabin' contain a substantial number of null values. 'Fare' also contains 1 null value. Since this is the `test` set, there are no target values (the 'Survived' column.

# In[14]:


all_data.loc["Train"].describe()


# **Observations:**
# - 38% of the passengers in the training set survived
# - The majority of 'Parch' values are 0, as even the 75 percentile is 0
# - Similarly, the majority of 'SibSp' values are also 0
# 
# ## _Create a Histogram_
# 
# Viewing a histogram of the numerical features gives us an idea of the distribution of values.

# In[15]:


# The 'bins' parameter sets the number of vertical bars
# The 'figsize' parameter sets the size of the chart
all_data.loc["Train"].hist(bins=25, figsize=(20, 15))


# **Observations:**
# - 'Age' is not normally distributed &mdash; may be worth normalizing
# - 'Fare', 'Parch', and 'SibSp' are all very positively skewed (data concentrated on the left indicates a positive skew; data concentrated on the right would indicate a negative skew)

# # Feature Engineering
# 
# Feature engineering is the most time consuming and (often) most influential part of the machine learning workflow. How well feature engineering is executed will have a significant impact on how well the machine learning models perform. Take your time and do this right.
# 
# ## _Handle Missing Values_
# 
# **NOTE:** Missing values can be referred to as null, NA, or NaN (Not a Number). They all mean the same thing &mdash; the absence of a value.
# 
# There are two strategies for altering data in DataFrames: editing `inplace` or reassigning values. Some methods, such as `drop()` have a parameter to edit data `inplace`, which directly alters the underlying DataFrame. Alternatively, you can perform a manipulation that returns a _copy_ of a column and then reassign that column with the new values. This can also be done with an entire DataFrame. I use both inplace editing and value reassignment in this notebook.
# 
# **NOTE:** Sometimes pandas returns a _view_ (which edits the underlying DataFrame) and sometimes it returns a _copy_ (which does not edit the underlying DataFrame). It can be confusing when one is returned instead of the other. If you are trying to edit data in a DataFrame and no changes are being made, it's probably an issue of editing a copy instead of a view. This is why I have validation code after each transformation that validates the changes being made. More information on views versus copies can be found [here](http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#indexing-view-versus-copy). 
# 
# ### Embarked
# 
# 'Embarked' is only missing 2 values, so we'll simply fill in the missing values with the most frequently occurring value (mode) in the training dataset. Remember, _evalute_ the training set and _transform_ both datasets:

# In[16]:


# The dropna() method excludes missing values, so if the most common value in the column is null, we don't return that as the mode
embarked_mode = all_data.loc["Train"]["Embarked"].dropna().mode()
embarked_mode


# In[17]:


# The mode() method returns a pandas Series, but we only want the value, so we look at index 0
embarked_mode = embarked_mode[0]
embarked_mode


# In[18]:


# The fillna() method fills in missing values with the value indicated in the first parameter position
all_data["Embarked"].fillna(embarked_mode, inplace=True)


# In[19]:


# The isnull() method returns a pandas Series with the row index and a Boolean (True or False) indicating if the value is null
# The sum() method sums up the Boolean values, with True = 1 and False = 0
all_data["Embarked"].isnull().sum()


# ### Fare
# 
# 'Fare' is only missing 1 value, so we'll again replace it with the mode:

# In[20]:


# Perform the same process as with 'Embarked' only combined into one line of code
all_data["Fare"].fillna(all_data.loc["Train"]["Fare"].dropna().mode()[0], inplace=True)


# In[21]:


all_data["Fare"].isnull().sum()


# ### Age
# 
# 'Age' is a continuous value, and there are a lot missing. We don't want to just fill it in with the mode. Instead, we'll look at features correlated to 'Age', find the median 'Age' value for each combination, and then use those average values to fill in what is missing.

# In[ ]:


# The corr() method returns a correlation matrix for all numerical features
correlation_matrix = all_data.loc["Train"].corr()
correlation_matrix["Age"].sort_values(ascending=False)


# In[22]:


# We can return multiple columns by passing a list to the second indexer
# The groupby() method groups the results by the values in the 'Sex' column, so male and female
# The mean() method returns the average
# The sort_values() method sorts the resulting DataFrame
all_data.loc["Train"][["Age", "Sex"]].groupby(["Sex"]).mean().sort_values(by="Age", ascending=False)


# Here we see that male passengers are on average almost 3 years older than female passengers.

# In[23]:


all_data.loc["Train"][["Age", "Embarked"]].groupby(["Embarked"]).mean().sort_values(by="Age", ascending=False)


# The features with the strongest correlation to 'Age' are 'SibSp' and 'Pclass'. However, there are some combinations of 'SibSp' and 'Pclass' which do not have any age values, resulting in a mean of `NaN`. The same thing occurs with the combination of 'Pclass' and 'Parch'.  Since there is no point to filling in null values with more nulls, we're going to use the combination of 'Pclass' and 'Sex' instead. 
# 
# 'Sex' values are either male or female. 'Pclass' values range from 1 to 3.  We iterate over every combination of 'Sex' and 'Pclass' and find the average 'Age', then fill in the missing 'Age' values:

# In[24]:


for value in ["male", "female"]:
    for i in range(0, 3):
        median_age = all_data.loc["Train"][(all_data.loc["Train"]["Sex"] == value) & (all_data.loc["Train"]["Pclass"] == i+1)]["Age"].dropna().median()
        all_data.loc[(all_data["Age"].isnull()) & (all_data["Sex"] == value) & (all_data["Pclass"] == i+1), "Age"] = median_age


# In[25]:


all_data["Age"].isnull().sum()


# ## _Alternative method for calculating missing age_
# 
# This method not only shows a different programmatic approach, but derives age from 'SibSp' and 'Parch'. I did not see any improvement in predictions using this approach, so I kept the original method in place. Including the below code for reference.

# In[26]:


#missing_age_index = list(all_data.loc["train"][all_data.loc["train"]["Age"].isnull()].index)
#missing_age_index


# In[27]:


# Get the index of any row missing an Age in the dataset
#missing_age_index = list(all_data.loc["train"][dataset["Age"].isnull()].index)
#for i in missing_age_index:
#    age_average = all_data.loc["train"]["Age"].median()
#    age_predict = all_data.loc["train"][(all_data.loc["train"]["SibSp"] == all_data.iloc[i]["SibSp"]) & (all_data.loc["train"]["Parch"] == all_data.iloc[i]["Parch"]) & (all_data.loc["train"]["Pclass"] == all_data.iloc[i]["Pclass"])]["Age"].median()
#    if np.isnan(age_predict):
#        all_data["Age"].iloc[i] = age_average
#    else:
#        all_data["Age"].iloc[i] = age_predict


# ### Cabin
# 
# 'Cabin' contains a lot of missing values, but in this case a null value does not indicate unknown data. Instead, a null indicates that the passenger did not have a cabin room. To handle this, we'll fill in nulls with "None" and handle the cabin numbers later.

# In[28]:


all_data["Cabin"].fillna("None", inplace=True)


# In[29]:


all_data["Cabin"].isnull().sum()


# ## _Create New Features_
# 
# ### Extract Title from Name

# In[30]:


# The extract() method uses a regular expression to extract the title from the 'Name' column
# The expand=False parameter returns a pandas Series
# Assigning values to an index that does not yet exist ('Title') will create it
all_data["Title"] = all_data["Name"].str.extract("([A-Za-z]+)\.", expand=False)


# In[31]:


# The unique() method returns all the unique values in the Series
all_data["Title"].sort_values().unique()


# Using the `crosstab()` function, we look at how each title lines up with gender:

# In[32]:


pd.crosstab(all_data.loc["Train"]["Title"], all_data.loc["Train"]["Sex"])


# We'll reclassify the uncommon titles to consolidate things:

# In[33]:


all_data["Title"].replace(["Capt", "Col", "Countess", "Don", "Dona", "Dr", "Jonkheer", "Lady", "Major", "Rev", "Sir"], "Rare", inplace=True)
all_data["Title"].replace("Mlle", "Miss", inplace=True)
all_data["Title"].replace("Ms", "Miss", inplace=True)
all_data["Title"].replace("Mme", "Mrs", inplace=True)


# In[34]:


all_data["Title"].sort_values().unique()


# In[35]:


all_data.loc["Train"][["Title", "Survived"]].groupby(["Title"]).mean().sort_values(by="Survived", ascending=False)


# Unsurprisingly, we see that female titles (Mrs & Miss) have a much higher survival rate. The title of Mr significantly reduces the rate of survival.
# 
# ### Create Age Groups
# 
# We'll group ages into several age groups, because that's what all the cool kernels are doing. First determine the groups based on the `train` set, then apply the groups to both `train` and `test` sets. 
# 
# I tested predictions with age groups and without them, and there was no impact on my final score. As an instructive exercise, they are included in the workflow:

# In[36]:


# The pandas cut function splits the data into equal sized value ranges
pd.cut(all_data.loc["Train"]["Age"], bins=5).dtype


# Using the groups returned by the `cut()` function, we replace 'Age' with numerical categories. Since most 'Age' values are rounded, we use rounded values when defining the groups.

# In[37]:


all_data.loc[all_data["Age"] <= 16, "Age"] = 0
all_data.loc[(all_data["Age"] > 16) & (all_data["Age"] <= 32), "Age"] = 1
all_data.loc[(all_data["Age"] > 32) & (all_data["Age"] <= 48), "Age"] = 2
all_data.loc[(all_data["Age"] > 48) & (all_data["Age"] <= 64), "Age"] = 3
all_data.loc[all_data["Age"] > 64, "Age"] = 4
# Since the category values are integers, we set the column type to int
# Notice that reassignment is used here as the astype() method does not have an inplace parameter
all_data["Age"] = all_data["Age"].astype(int)


# In[38]:


all_data["Age"].sort_values().unique()


# In[39]:


all_data.loc["Train"][["Age", "Survived"]].groupby(["Age"]).mean().sort_values(by="Survived", ascending=False)


# We can see a clear impact on survival based on age.
# 
# ### Create Family Size
# 
# Combine 'SibSp' and 'Parch' into one 'FamilySize' feature. Add +1 to include the passenger in the size of the family:

# In[40]:


all_data["FamilySize"] = all_data["SibSp"] + all_data["Parch"] + 1


# In[41]:


all_data["FamilySize"].sort_values().unique()


# In[42]:


all_data.loc["Train"][["FamilySize", "Survived"]].groupby(["FamilySize"]).mean().sort_values(by="Survived", ascending=False)


# There seems to be a trend toward larger families having a lower rate of survival.
# 
# ### Label if Passenger is Alone
# 
# If 'FamilySize' = 1, then the passenger is alone aboard the ship.

# In[43]:


all_data["IsAlone"] = 0
all_data.loc[all_data["FamilySize"] == 1, "IsAlone"] = 1


# In[44]:


all_data["IsAlone"].sort_values().unique()


# In[45]:


all_data.loc["Train"][["IsAlone", "Survived"]].groupby(["IsAlone"]).mean()


# It looks like passengers who were alone had a lower rate of survival.
# 
# ### Create Fare Groups
# 
# Just like with age groups, fare groups did not influence my predictions. But we don't know how a new feature will influence predictions until we test it, so don't be afraid to create new features. Machine learning involves a lot of trial and error.

# In[46]:


# The qcut() method splits the data into equal sized bins where each bin has the same number of records
pd.qcut(train["Fare"], 4).dtype


# In[47]:


all_data.loc[all_data["Fare"] <= 7.91, "Fare"] = 0
all_data.loc[(all_data["Fare"] > 7.91) & (all_data["Fare"] <= 14.45), "Fare"] = 1
all_data.loc[(all_data["Fare"] > 14.45) & (all_data["Fare"] <= 31.00), "Fare"] = 2
all_data.loc[all_data["Fare"] > 31.00, "Fare"] = 3
all_data["Fare"] = all_data["Fare"].astype(int)


# In[48]:


all_data["Fare"].sort_values().unique()


# In[49]:


all_data.loc["Train"][["Fare", "Survived"]].groupby(["Fare"], as_index=False).mean().sort_values(by="Survived", ascending=False)


# There is a pretty clear correlation between survival and the fare amount paid. Higher fare groups, where a higher fare was paid, have increased rates of survival. This could be related to socioeconomic standing or simple logistics. Passengers who paid lower fares were likely housed in the bowels of the ship where it would be much more difficult to escape to the deck and available lifeboats.
# 
# ### Create Cabin Groups
# 
# Playing off of the above point, cabin numbers may indication positions on the boat where escape was easier than others. We'll create it and see.
# 
# First, extract the letter from the cabin number:

# In[50]:


all_data["Cabin"] = all_data["Cabin"].loc[all_data["Cabin"].isnull() == False].str.extract("([A-Za-z]+)", expand=False)


# In[51]:


all_data["Cabin"].sort_values().unique()


# In[52]:


all_data.loc["Train"][["Cabin", "Survived"]].groupby(["Cabin"], as_index=False).mean().sort_values(by="Survived", ascending=False)


# It's not surprising that people who were not in cabins had a lower survival rate. But it would be better than having a cabin number starting with T...
# 
# ## _Drop Irrelevant Features_
# 
# There are some features that we can confidently assume have no bearing on survival ('PassengerId', 'Ticket') and others we no longer need ('Name'). We'll drop these features.

# In[53]:


all_data.drop(["PassengerId", "Name", "Ticket"], axis=1, inplace=True)


# In[54]:


all_data.head()


# # Explore the Data
# 
# ## _Correlation Matrix_
# 
# The correlation matrix only works on numerical features, so we're going to convert the labels of the categorical features using scikit-learn's `LabelEncoder`.
# 
# The `LabelEncoder` must first be _fit_ to the data using the `fit()` method, which determines how many labels are in a category and what numeric values to assign to those categories. The `transform()` method then replaces the categorical values with the newly determined numeric values. The `fit_transform` method performs both operations.
# 
# We are going to store the label encoded data in a new variable, _all_data_encoded_, because we'll only use it to explore correlations and feature importances. We're going to use a different kind of encoding for categorical features that we'll feed into our machine learning models.

# In[55]:


# Instantiate a LabelEncoder object
label_encoder = LabelEncoder()
# The apply() method applies a function across a column or row
# Here we apply the LabelEncoder across the entire DataFrame, affecting only categorical (text) features
all_data_encoded = all_data.apply(label_encoder.fit_transform)
all_data_encoded.head()


# Now our entire DataFrame contains only numerical features.

# In[56]:


correlation_matrix = all_data_encoded.loc["Train"].corr()
correlation_matrix["Survived"].sort_values(ascending=False)


# **Observations**:
# - The positive correlation of 'Fare' and the negative correlation of 'Pclass' with survival may be related &mdash; does higher socioeconomic standing increase your chance for survival or is this related to a person's location on the ship?
# - The most significant correlation with survival is the negative correlation for 'Sex'
# - 'IsAlone' has a negative correlation with survival &mdash; maybe due to mothers with children escaping before single men?
# 
# ## _Pearson Correlation Heatmap_
# 
# We'll use a heatmap to see how features are correlated.

# In[57]:


plt.figure(figsize=(10, 8))
plt.title("Pearson Correlation of Features", y=1.05, size=15)
sns.heatmap(
    correlation_matrix,
    linewidths=0.1,
    vmax=1.0,
    square=True,
    cmap=plt.cm.jet,
    linecolor="white",
    annot=True
)


# **Observations**:
# - There are significant correlations between 'FamilySize', 'Parch', and 'SibSp', which is unsurprising since 'FamilySize' was derived from the other two features. We should consider keeping either the set of 'Parch' & 'SibSp' or 'FamilySize'.
# - 'Pclass' is strongly correlated with 'Cabin'. We should consider dropping one or the other.

# ## _Evaluate Feature Importance_
# 
# Scikit-learn's `RandomForestClassifier()` includes a `feature_importances_` property which can be used to measure how important different features are. We'll train a model on the training dataset and then examine the feature importances. In order to do this, we must first split the data into the `X_train` (training data) and `y_train` (training targets) datasets. We'll do this on the `all_data_encoded` DataFrame:

# In[58]:


y_train = all_data_encoded.loc["Train"]["Survived"].astype(int)
X_train = all_data_encoded.loc["Train"].drop(["Survived"], axis=1)


# In[59]:


# Instantiate a Random Forest Classifier object
random_forest_classifier = RandomForestClassifier()
# Train (fit) the Random Forest Classifier on the training data
random_forest_classifier.fit(X_train, y_train)


# In[60]:


# The zip() function takes two iterables and joins them together into an iterable of tuples, in this case with the column name matched up to its feature importance
feature_importances = zip(list(X_train.columns.values), random_forest_classifier.feature_importances_)
# Sort the list by feature importance using a lambda function
feature_importances = sorted(feature_importances, key=lambda feature: feature[1], reverse=True)
# Iterate over the feature_importances list
for name, score in feature_importances:
    # The format() method replaces any set of curly brackets in a string with the specified arguments
    # The :<12 inside the first set of curly brackets aligns the text to the left and sets the character length to 12 characters, making everything print neatly
    print("{:<12} | {}".format(name, score))


# **Observations**
# - Again we see the importance of 'Sex', 'Pclass', and 'Fare'
# - The importance of 'Title' is not surprising given its relation to 'Sex'
# 
# # Final Data Preparation
# 
# ## _Drop Unimportant and Correlated Features_
# 
# We don't want multiple features that measure the same thing, as this skews the importance of the underlying data. We have two decisions to make:
# - **Drop 'FamilySize'**
#     - While 'FamilySize' showed a slightly higher importance, 'Parch' was noticeably more correlated with survival, so we'll keep the set of 'Parch' & 'SibSp' 
# - **Drop 'Cabin'**
#     - 'Pclass' shows a higher correlation to surival and a slightly higher feature importance, so we'll keep it and drop 'Cabin'
#     
# **Note**: While we're dropping features that we created earlier, it's still worth going through the process of creating them. You won't know if it is useful until you measure its importance later on. As mentioned earlier, machine learning involves a lot of trial and error, so throw everything against the wall and see what sticks.

# In[61]:


all_data.drop(["FamilySize", "Cabin"], axis=1, inplace=True)


# In[62]:


all_data.head()


# ## _Convert Categorical Data_
# 
# Most machine learning algorithms only accept numeric data, so we need to convert text categories to numeric values. We could just use the label encoded dataset we created for looking at feature importance, but the machine learning algorithms may interpret meaning with the distances between numeric values.  
# 
# For example, the distance between 'FamilySize' values of 2 and 7 has meaning. One is a small family while the other is a large family. The same principle applies to 'Age'. Our encoded value of 0 means someone is very young while an encoded value of 4 means someone is much older. With similar numerical values applied to 'Title', an algorithm may interpret some meaning if Master has a value of 1 and Mrs has a value of 4. We know however that these numerical values are arbitrary.
# 
# To get around this issue, we use One-Hot Encoding. With One-Hot Encoding, a column is created for each possible feature value. So for 'Sex', there is a column for female and a column for male. If the passenger is female, the 'Sex_female' column = 1 and the 'Sex_male' column = 0. The reverse would be true if the passenger is male.
# 
# We easily apply One-Hot Encoding using pandas `get_dummies()` function below:

# In[63]:


all_data = pd.get_dummies(all_data)


# In[64]:


all_data.loc["Train"].head()


# ## _Split into X and Y Datasets_
# 
# We split the data again, this time using the One-Hot Encoded data:

# In[65]:


y_train = all_data.loc["Train"]["Survived"].astype(int)
X_train = all_data.loc["Train"].drop(["Survived"], axis=1)
X_test = all_data.loc["Test"].drop(["Survived"], axis=1)


# In[66]:


y_train.head()


# In[67]:


X_train.head()


# In[68]:


X_test.head()


# ## _Feature Scaling_
# 
# Before feeding the data into our models, we're going to scale the data. At first I used scikit-learn's `StandardScaler()`. But after seeing the `RobustScaler()` used in a couple other kernels, I tried it. The `RobustScaler()` provides resistance to outliers and results in better predictions, so we'll we use that one.

# In[69]:


#standard_scaler = StandardScaler()
#X_train = standard_scaler.fit_transform(X_train)
#X_test = standard_scaler.transform(X_test)

robust_scaler = RobustScaler()
X_train = robust_scaler.fit_transform(X_train)
X_test = robust_scaler.fit_transform(X_test)


# The `RobustScaler` returns a numpy array, so at this point our data is no longer in the form of a pandas DataFrame.

# In[70]:


X_train[:3]


# In[71]:


X_test[:3]


# # Evaluate Models
# 
# We use scikit-learn's `cross_val_score()` function to test each model. I do a little bit of brute force[](http://) parameter tuning by setting some basic parameters and testing out different values.
# 
# ## _SGD Classifier_

# In[72]:


sgd_classifier = SGDClassifier()
cross_val_score(sgd_classifier, X_train, y_train, cv=10, scoring="accuracy").mean()


# ## _One vs One Classifier_

# In[73]:


one_vs_one_classifier = OneVsOneClassifier(SGDClassifier())
cross_val_score(one_vs_one_classifier, X_train, y_train, cv=10, scoring="accuracy").mean()


# ## _Random Forest Classifier_

# In[74]:


random_forest_classifier = RandomForestClassifier(min_samples_split=10, min_samples_leaf=2)
cross_val_score(random_forest_classifier, X_train, y_train, cv=10, scoring="accuracy").mean()


# ## _Extra Trees Classifier_

# In[75]:


extra_trees_classifier = ExtraTreesClassifier(min_samples_split=7, min_samples_leaf=5)
cross_val_score(extra_trees_classifier, X_train, y_train, cv=10, scoring="accuracy").mean()


# ## _SVM Classifier_

# In[76]:


svm_classifier = SVC(probability=True, C=4.5)
cross_val_score(svm_classifier, X_train, y_train, cv=10, scoring="accuracy").mean()


# ## _AdaBoost Classifier_

# In[77]:


adaboost_classifier = AdaBoostClassifier(n_estimators=200, learning_rate=0.1)
cross_val_score(adaboost_classifier, X_train, y_train, cv=10, scoring="accuracy").mean()


# ## _Gradient Boosting Classifier_

# In[78]:


gradient_boost_classifier = GradientBoostingClassifier(learning_rate=0.03)
cross_val_score(gradient_boost_classifier, X_train, y_train, cv=10, scoring="accuracy").mean()


# ## _Logistic Regression Classifier_

# In[79]:


logistic_regression_classifier = LogisticRegression()
cross_val_score(logistic_regression_classifier, X_train, y_train, cv=5, scoring="accuracy").mean()


# ## _Linear Discriminant Analysis Classifier_

# In[80]:


lda_classifier = LinearDiscriminantAnalysis()
cross_val_score(lda_classifier, X_train, y_train, cv=5, scoring="accuracy").mean()


# ## _XGBoost Classifier_

# In[81]:


xgboost_classifier = xgb.XGBClassifier(gamma=0.7)
cross_val_score(xgboost_classifier, X_train, y_train, cv=5, scoring="accuracy").mean()


# # Fine-Tune the Best Models
# 
# While it is time consuming and does not always result in drastic improved predictions, parameter tuning is still an important part of the machine learning workflow. While it is possible to guess which parameters to use, this is time consuming and inefficient. A better method is to use a grid or random search. Scikit-learn convienently provides functions to do both. The below code shows how to tune the Support Vector Machine classifier using a grid search.

# ## _Support Vector Machine Tuning_

# In[82]:


parameter_grid = [
    {
        "kernel": ["rbf"],
        "C": [4, 4.5, 5],
        "shrinking": [True, False],
        "tol": [0.00001, 0.00003, 0.00005, 0.00008],
        "class_weight": ["balanced", None],
        "gamma": ["auto_deprecated", "scale"],
        "probability": [True]
    },
    {
        "kernel": ["poly"],
        "degree": [1, 3, 5],
        "gamma": ["auto_deprecated", "scale"]
    }
]


# In[83]:


grid_search = GridSearchCV(
    svm_classifier,
    parameter_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=2,
    verbose=1,
    return_train_score=True
)


# In[84]:


grid_search.fit(X_train, y_train)


# In[85]:


grid_search.best_params_


# In[86]:


grid_search.best_estimator_


# In[87]:


svm_classifier = grid_search.best_estimator_


# In[88]:


cross_val_score(svm_classifier, X_train, y_train, cv=5, scoring="accuracy").mean()


# After all of that, I'm just going to cheat and fit the best classifiers to the data using the parameters use in training :)

# In[89]:


svm_classifier.fit(X_train, y_train)
gradient_boost_classifier.fit(X_train, y_train)
logistic_regression_classifier.fit(X_train, y_train)
lda_classifier.fit(X_train, y_train)
xgboost_classifier.fit(X_train, y_train)


# # Emsembling via a Voting Classifier
# 
# By using a voting classifier, we use the predictions of several different models and hopefully make better predictions as a result.

# In[90]:


voting_classifier = VotingClassifier(
    estimators=[
        ("svc", svm_classifier),
        ("gradient_boost", gradient_boost_classifier),
        ("logistic_regression", logistic_regression_classifier),
        ("lda", lda_classifier),
        ("xgboost", xgboost_classifier)
    ],
    voting="soft"
)


# In[91]:


voting_classifier.fit(X_train, y_train)


# In[92]:


cross_val_score(voting_classifier, X_train, y_train, cv=10, scoring="accuracy").mean()


# In[93]:


predictions = voting_classifier.predict(X_test)


# # Submission

# In[94]:


submission = pd.DataFrame(
    {
        "PassengerId": passenger_id,
        "Survived": predictions
    }
)


# In[95]:


submission.head(10)


# In[96]:


# Write the submission DataFrame to a CSV file using the constructed filename
submission.to_csv("submission.csv", index=False)


# In[ ]:




