#!/usr/bin/env python
# coding: utf-8

# # Titanic 79.43% test accuracy with complete steps 
# 
# This is my attempt at predicting survival on the Titanic by experimenting with various machine learning models and pre-processing techniques. This kernel consists of the following:
# 
# - Exploratory data analysis 
# - Visualizations (histograms, barplots, boxplots)
# - Feature engineering (Family Size, Name-Titles, Cabin)
# - Data pre-processing (train/validation split, missing value imputation, binning)
# - Model building and comparison of different classification algorithms
# - Model selection, tuning and interpretation 
# - Prediction and scoring 
# 
# Many observations about this interesting dataset will be noted along the way. 

# In[ ]:


# Importing libraries  
import pandas as pd
import numpy as np
import matplotlib.pylab as py
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import preprocessing, model_selection, metrics
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


# In[ ]:


# Importing the train and test datasets

train_raw = pd.read_csv("../input/train.csv")
test_raw = pd.read_csv("../input/test.csv")

# Creating a copy of the training data which we will later split into train and validation datasets 
data = train_raw.copy(deep = True)

# Creating a copy of the test data to which we will apply the same preprocessing as train and validation
test_data = test_raw.copy(deep = True)


# ## Exploratory Data Analysis 

# In[ ]:


data.head(10)


# In[ ]:


data.tail(10)


# ### Observations:
# - PassengerId is not a useful column
# - Name contains interesting info like marital status (Miss / Mrs) and titles like Reverend (Rev) which denote social status
# - SibSp and Parch could be combined into one feature called Family Size
# - There may be a relationship between Pclass and Ticket (the first digit is the same in most cases)

# In[ ]:


# Checking that the test data has similar features
test_data.head(10)


# In[ ]:


test_data.tail(10)


# In[ ]:


# Getting a feel for the descriptive statistics of each feature
data.describe(include = "all")


# ### Observations :
# - Cabin, Age and Embarked have missing values
# - Cabin is missing 78% of its values 

# In[ ]:


data.info()


# In[ ]:


# Dropping the PassengerId column - will recover it later from test_raw for the predictions file
# Also dropping Ticket 

data.drop(["PassengerId", "Ticket"], axis = 1, inplace = True)
test_data.drop(["PassengerId", "Ticket"], axis = 1, inplace = True)


# ### Visualizing feature distributions and relationships

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Plotting histograms to show the distribution of numeric features in the dataset
data.hist(figsize = (15, 15))


# ### Observations:
# - Age: frequency of passengers aged 18-30 is highest and aged 70+ is lowest
# - Fare: most passengers paid low fares (<50) and there are some outliers with fares around 500
# - Parch and SibSp: most passengers have fewer than two family members on board
# - Pclass: most passengers are travelling 3rd class
# - Survived: Ratio of passengers who died (0) to those who survived (1) is around 61:39 - will check this with a barplot

# In[ ]:


# Plotting barplots for the target variable (Survived) and categorical variables

data.Survived.value_counts().plot(kind = "bar", rot = 0)

print("Ratio of Died to Survived passengers is", int(round(((data.Survived.value_counts()[0] / data.Survived.value_counts().sum()) * 100))),
     ":", int(round((data.Survived.value_counts()[1] / data.Survived.value_counts().sum()) * 100))) 


# In[ ]:


data.Sex.value_counts().plot(kind = "bar", rot = 0)

print("Ratio of male to female passengers is", int(round(((data.Sex.value_counts()[0] / data.Sex.value_counts().sum()) * 100))),
     ":", int(round((data.Sex.value_counts()[1] / data.Sex.value_counts().sum()) * 100))) 


# In[ ]:


data.Embarked.value_counts().plot(kind = "bar", rot = 0)


# In[ ]:


data.Cabin.value_counts().plot(kind = "bar", rot = 0)
# There are too many levels to create a meaningful barplot


# In[ ]:


# Examining the levels in Cabin
data.Cabin.unique()


# ### Observations:
# The letters at the start of each alphanumeric cabin number may indicate the cabin class and/or location which may affect survival

# ### Feature Engineering - Cabin

# In[ ]:


# Replacing the cabin numbers with letters and assigning NaN values the letter X

data["Cabin"] = pd.Series((i[0] if not pd.isnull(i) else "X" for i in data["Cabin"]), dtype = "category")


# In[ ]:


data["Cabin"].unique()


# In[ ]:


test_data["Cabin"] = pd.Series((i[0] if not pd.isnull(i) else "X" for i in test_data["Cabin"]), dtype = "category")


# In[ ]:


test_data["Cabin"].unique()


# In[ ]:


data.Cabin.value_counts().plot(kind = "bar", rot = 0)


# In[ ]:


# Visualizing relationships between features and the target variable

data.groupby(["Sex", "Survived"]).size()


# In[ ]:


sns.countplot(x = "Sex", hue = "Survived", data = data)

print("The survival rate for females is", int(round(100 * (data.groupby(["Sex", "Survived"]).size()[1] / (data.groupby(["Sex", "Survived"]).size()[0] + data.groupby(["Sex", "Survived"]).size()[1])))), "%")
                                               
print("The survival rate for males is", int(round(100 * (data.groupby(["Sex", "Survived"]).size()[3] / (data.groupby(["Sex", "Survived"]).size()[2] + data.groupby(["Sex", "Survived"]).size()[3])))), "%")
                                               
                                                                                        


# In[ ]:


# Plotting survival by Sex
sns.catplot(x = "Sex", y = "Survived", data = data, kind = "bar")


# In[ ]:


data.groupby(["Cabin", "Survived"]).size()


# In[ ]:


sns.countplot(x = "Cabin", hue = "Survived", data = data)


# ### Observations:
# Cabin X has the lowest survival rate - Cabin T has only one passenger so we won't consider it

# In[ ]:


# Plotting survival per cabin
sns.catplot(x = "Cabin", y = "Survived", data = data, kind = "bar", aspect = 2)


# In[ ]:


sns.countplot(x = "Embarked", hue = "Survived", data = data)


# In[ ]:


# Taking a closer look at Embarked
sns.catplot(x = "Embarked", y = "Survived", data = data, kind = "bar", aspect = 2)


# ### Observations:
# - Passengers who embarked at port C have the highest survival rate
# - Passengers who embarked at port S have the lowest survival rate

# In[ ]:


from matplotlib import pyplot
fig, ax = pyplot.subplots(figsize=(40, 20))
sns.countplot(x = "Age", hue = "Survived", data = data)


# ### Observations:
# - All infants under a year old survived
# - Most passengers over 60 died

# In[ ]:


# Plotting relationships between Age and other variables
sns.catplot(x = "Survived", y = "Age", data = data, kind = "box")


# ### Observations:
# Younger people are more likely to survive 

# In[ ]:


sns.catplot(x = "Sex", y = "Age", data = data, kind = "box")


# ### Observations:
# Males are slightly older than females 

# In[ ]:


sns.catplot(x = "Parch", y = "Age", data = data, kind = "box")


# sns.catplot(x = "SibSp", y = "Age", data = data, kind = "box")
# 

# ### Observations:
# 
# Older passengers have more parents/children and younger passengers have more spouses/siblings

# In[ ]:


sns.catplot(x = "Pclass", y = "Age", data = data, kind = "box")


# ### Observations:
# Older passengers are travelling in higher classes

# In[ ]:


# Plotting the relationship between Survived and Fare
sns.catplot(x = "Survived", y = "Fare", data = data, kind = "box")


# ### Observations
# Higher fare and higher survival are correlated but there are also several outliers

# In[ ]:


# Outlier detection

data.plot(kind = "box", figsize = (20, 10))


# ### Observations:
# There is an extreme outlier in the Fare column (500). Let's see what's happening in the test data.

# In[ ]:


test_data.plot(kind = "box", figsize = (20, 10))


# ### Observations:
# The test data has a similar outlier pattern so it can be ignored.

# ### Feature Engineering - Family Size

# In[ ]:


# Combining Parch and SibSp to create a new feature called Family Size

data["FamilySize"] = data["Parch"] + data["SibSp"]
test_data["FamilySize"] = test_data["Parch"] + test_data["SibSp"]


# ### Feature Engineering - Titles 

# In[ ]:


# Extracting titles from the Name column
# The titles contain useful info like marital status, social status and occupation
titles = set()
for name in data["Name"]:
    titles.add(name.split(",")[1].split(".")[0].strip())
print (titles)


# In[ ]:


titles_test = set()
for name in test_data["Name"]:
    titles_test.add(name.split(",")[1].split(".")[0].strip())
print (titles_test)


# In[ ]:


# Creating a titles dictionary to map to the Name column. Some titles can be grouped into a more generic one

title_dict = {
    "Sir" : "Nobility",
    "the Countess" : "Nobility",
    "Miss" : "Miss",
    "Major" : "Army",
    "Col" : "Army",
    "Lady" : "Nobility",
    "Capt" : "Army",
    "Dr" : "Doctor",
    "Jonkheer" : "Nobility",
    "Mlle" : "Miss",
    "Mrs" : "Mrs",
    "Mr" : "Mr",
    "Don" : "Nobility",
    "Rev" : "Clergy",
    "Ms" : "Mrs",
    "Mme" : "Mrs",
    "Master" : "Master",
    "Dona" : "Nobility"
}


# In[ ]:


data["Title"] = data["Name"].map(lambda name:name.split(",")[1].split(".")[0].strip())
data["Title"] = data.Title.map(title_dict)
test_data["Title"] = test_data["Name"].map(lambda name:name.split(",")[1].split(".")[0].strip())
test_data["Title"] = test_data.Title.map(title_dict)


# In[ ]:


data.drop(["Name"], axis = 1, inplace = True)
test_data.drop(["Name"], axis = 1, inplace = True)


# ### Data pre-processing

# In[ ]:


# Identifying missing values
data.isnull().sum()
# We will perform imputation after the train-validation split 


# In[ ]:


# Separating the target column and splitting the dataset into train and validation sets in a 70:30 ratio

target = data["Survived"]
data.drop("Survived", axis = 1, inplace = True)


# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(data, target, test_size = 0.3, random_state = 0)


# In[ ]:


# Missing value imputation
# We will impute Age based on median values grouped by passenger Sex and Title

grouped = X_train.groupby(["Sex", "Title"])  

grouped.Age.median()


# In[ ]:


pd.options.mode.chained_assignment = None

X_train.Age = grouped.Age.apply(lambda x: x.fillna(x.median()))
X_valid.Age = grouped.Age.apply(lambda x: x.fillna(x.median()))
test_data.Age = grouped.Age.apply(lambda x: x.fillna(x.median()))


# In[ ]:


# The few missing values in Embarked and Fare can be imputed with central values

imputer = SimpleImputer(strategy = "most_frequent")
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns = X_train.columns)
X_valid = pd.DataFrame(imputer.transform(X_valid), columns = X_valid.columns)
test_data = pd.DataFrame(imputer.transform(test_data), columns = test_data.columns)


# In[ ]:


# Discretizing Age into bins of equal width
# Binning continuous variables can lead to better predictions when there are patterns within the variable

pd.cut(X_train["Age"], bins = 4)
# Will use these bin sizes as a guide for creating custom bins


# In[ ]:


X_train["Age"] = pd.cut(X_train["Age"], (0, 20, 40, 60, 80), labels = ("0-20", "20-40", "40-60", "60-80"))
X_valid["Age"] = pd.cut(X_valid["Age"], (0, 20, 40, 60, 80), labels = ("0-20", "20-40", "40-60", "60-80"))
test_data["Age"] = pd.cut(test_data["Age"], (0, 20, 40, 60, 80), labels = ("0-20", "20-40", "40-60", "60-80"))


# In[ ]:


# Discretizing Fare into bins of equal frequency (since majority of passengers have low fares, equal width bins would not be appropriate)

X_train["Fare"] = pd.qcut(X_train["Fare"], 4, labels = ("low", "middle", "upper middle", "high"))
X_valid["Fare"] = pd.qcut(X_valid["Fare"], 4, labels = ("low", "middle", "upper middle", "high"))
test_data["Fare"] = pd.qcut(test_data["Fare"], 4, labels = ("low", "middle", "upper middle", "high"))


# In[ ]:


# Converting numeric columns back to numeric

X_train["FamilySize"] = X_train["FamilySize"].astype("int")
X_valid["FamilySize"] = X_valid["FamilySize"].astype("int")
test_data["FamilySize"] = test_data["FamilySize"].astype("int")

X_train["SibSp"] = X_train["SibSp"].astype("int")
X_valid["SibSp"] = X_valid["SibSp"].astype("int")
test_data["SibSp"] = test_data["SibSp"].astype("int")

X_train["Parch"] = X_train["Parch"].astype("int")
X_valid["Parch"] = X_valid["Parch"].astype("int")
test_data["Parch"] = test_data["Parch"].astype("int")


# In[ ]:


X_train.info()


# In[ ]:


# Converting all categorical variables to astype.categorical with same levels across all three datasets
# This will prevent the missing column problem after dummification

X_train["Pclass"] = X_train["Pclass"].astype("category", categories = X_train["Pclass"].unique())
X_valid["Pclass"] = X_valid["Pclass"].astype("category", categories = X_train["Pclass"].unique())
test_data["Pclass"] = test_data["Pclass"].astype("category", categories = X_train["Pclass"].unique())

X_train["Embarked"] = X_train["Embarked"].astype("category", categories = X_train["Embarked"].unique())
X_valid["Embarked"] = X_valid["Embarked"].astype("category", categories = X_train["Embarked"].unique())
test_data["Embarked"] = test_data["Embarked"].astype("category", categories = X_train["Embarked"].unique())

X_train["Sex"] = X_train["Sex"].astype("category", categories = X_train["Sex"].unique())
X_valid["Sex"] = X_valid["Sex"].astype("category", categories = X_train["Sex"].unique())
test_data["Sex"] = test_data["Sex"].astype("category", categories = X_train["Sex"].unique())

X_train["Cabin"] = X_train["Cabin"].astype("category", categories = X_train["Cabin"].unique())
X_valid["Cabin"] = X_valid["Cabin"].astype("category", categories = X_train["Cabin"].unique())
test_data["Cabin"] = test_data["Cabin"].astype("category", categories = X_train["Cabin"].unique())

X_train["Title"] = X_train["Title"].astype("category", categories = X_train["Title"].unique())
X_valid["Title"] = X_valid["Title"].astype("category", categories = X_train["Title"].unique())
test_data["Title"] = test_data["Title"].astype("category", categories = X_train["Title"].unique())


# In[ ]:


X_train.info()


# In[ ]:


# Dummifying (one hot encoding) the categorical variables
X_train = pd.get_dummies(X_train, prefix = ["Pclass", "Sex", "Age", "Fare", "Cabin", "Embarked", "Title"])


# In[ ]:


X_valid = pd.get_dummies(X_valid, prefix = ["Pclass", "Sex", "Age", "Fare", "Cabin", "Embarked", "Title"])


# In[ ]:


test_data = pd.get_dummies(test_data, prefix = ["Pclass", "Sex", "Age", "Fare", "Cabin", "Embarked", "Title"])


# In[ ]:


# Checking that all columns are present
X_train.shape


# In[ ]:


X_valid.shape


# In[ ]:


test_data.shape


# ## Model building and comparison

# In[ ]:


# Creating basic models with different algorithms 

models = []
models.append(("LR", LogisticRegression()))
models.append(("KNN", KNeighborsClassifier(n_neighbors = 5)))
models.append(("DT", DecisionTreeClassifier(max_depth = 6, min_samples_leaf = 2)))
models.append(("NB", GaussianNB()))
models.append(("SVM", SVC()))
models.append(("RF", RandomForestClassifier(n_estimators = 100, max_depth = 6, min_samples_leaf = 2)))
models.append(("GB", GradientBoostingClassifier(n_estimators = 100, learning_rate = 0.1, max_depth = 6, min_samples_leaf = 2)))
models.append(("XGB", XGBClassifier(n_estimators = 100, learning_rate = 0.1, max_depth = 6, min_samples_leaf = 2)))

# Evaluating model performances with cross validation
results = []
names = []
scoring = "accuracy"
for name, model in models:
    kfold = KFold(n_splits=10, random_state=0)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f" % (name, cv_results.mean())
    print(msg)
    
# Boxplots comparing the algorithms
fig = plt.figure()
fig.suptitle("Algorithm Comparison")
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# ### Observations:
# Random Forest, Logistic Regression and SVM have the best score. Let us see how they perform on the validation data.
# 

# In[ ]:


logreg = LogisticRegression(random_state = 0).fit(X_train, y_train)
pred_train = logreg.predict(X_train)
pred_valid = logreg.predict(X_valid)
pred_test = logreg.predict(test_data)

accuracy_score(pred_train, y_train)
# 84.43


# In[ ]:


accuracy_score(pred_valid, y_valid)
# 81.34


# In[ ]:


svm = SVC(random_state = 0).fit(X_train, y_train)
pred_train = svm.predict(X_train)
pred_valid = svm.predict(X_valid)
pred_test = svm.predict(test_data)

accuracy_score(pred_train, y_train)
# 83.46


# In[ ]:


accuracy_score(pred_valid, y_valid)
# 82.09
# Better than Logistic Regression


# In[ ]:


# Random Forest ensemble with hyperparameter tuning
rf = RandomForestClassifier(n_estimators=150, max_depth=8, max_features = 0.7, min_samples_leaf = 2, random_state = 0).fit(X_train, y_train)
pred_train = rf.predict(X_train)
pred_valid = rf.predict(X_valid)
pred_test = rf.predict(test_data)

accuracy_score(pred_train, y_train)
# 87.8


# In[ ]:


accuracy_score(pred_valid, y_valid)
# 83.96


# ### Observations:
#     
# The Random Forest classifier gave the best accuracy on training and validation data after some hyperparameter tuning. 

# In[ ]:


PassengerId = test_raw["PassengerId"]
pred_test = rf.predict(test_data)
# Now we combine these two into a predictions file and upload 


# In[ ]:


Titanicpreds = pd.DataFrame({"PassengerId": PassengerId, "Survived": pred_test})
Titanicpreds.to_csv('Titanicpreds.csv', index=False)


# ### Model interpretation - Feature Importances plot

# In[ ]:


feat_importances = pd.Series(rf.feature_importances_, index=X_train.columns)
feat_importances.nlargest(10).plot(kind='barh')


# ## Final observations 
# - According to the best performing classifier, Title_Mr is the most important feature in predicting survival on the Titanic. 
# - The engineered features Title, FamilySize and Cabin are among the top 10 most important features.
# - Surprisingly, Age does not seem to be important. 
# - Submission score (test data accuracy) : 79.43%
