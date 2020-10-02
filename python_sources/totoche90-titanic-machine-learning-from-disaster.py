#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[ ]:


# Data analysis
import numpy as np
import pandas as pd

# Data viz
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Outputs
import warnings
warnings.filterwarnings('ignore')

# Machine learning
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier


# # Import data

# In[ ]:


# Raw import
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")


# In[ ]:


# Show head of training set
train_df.head()


# In[ ]:


# Show head of testing set
test_df.head()


# # Analyze and refurbish data

# ## 1. Analyze the training set by description

# In[ ]:


# Temporarily convert training set to strings and use the describe function of pandas
train_df.applymap(lambda x: x if pd.isnull(x) else str(x)).describe()


# ### Representativity of the training set
# [Ref stats here](https://www.historyonthenet.com/the-titanic-passenger-and-crew-statistics/)
# 
# In the training set,
# * The **survival rate** is ~39%, which is pretty close to the ~32% of in the real disaster
# * There is **a majority of male passengers** (65%), which is similar to the reality
# * All classes of passengers are represented, with a 55% **majority of 3rd class passengers**, which is similar to the reality
# * All 3 ports of embarkation are represented (C = Cherbourg; Q = Queenstown; S = Southampton)
# * Age feature is quite varied, with 88 different values
# * Same for Fare feature
# 
# With just these observations, we can say that the training set is a representative sample of our desired output, and we can safely proceed with our study.
# 
# ### Safely removable features
# Just by looking at the available features, we can already drop the following features:
# * **PassengerID**: redundant with Name, and not helpful in survival rate
# * **Ticket**: contains duplicates, redundant with Name, and not helpful in survival rate
# * **Cabin**: incomplete, contains duplicates, redundant with Name, and not helpful in survival rate
# 
# ### Modifiable features
# * **Name**: having 1 different name for each passenger, we cannot simply keep this feature as is. However, we can directly see the presence of the title in the name. We can probably form groups of titles.
# * **Fare**: with 248 different values, we can probably form ranges of fare.
# * **Age**: with 88 different values, we can probably form ranges of age.

# ## 2. Drop removable features

# To drop are the following features: **PassengerID**, **Ticket** and **Cabin**

# In[ ]:


# Drop the features
train_df.drop(['PassengerId', 'Ticket', 'Cabin'], inplace=True, axis=1)
test_df.drop(['PassengerId', 'Ticket', 'Cabin'], inplace=True, axis=1)


# ## 3. Decide on the remaining features
# 
# In this section we analyze more carefully the remaining features, by:
# 1. Pre-analyzing the feature
# 2. Correlating it with the output (survival rate)
# 3. Deciding on the actions to do: keep as-is, transform, combine, convert into categories, etc
# 4. Completing the missing data in both the training & testing sets
# 5. Executing the decisions on both the training and testing sets

# ### **Name** feature
# 
# #### Pre-analysis
# Being a categorical feature with as many possibilities as the volume of our training set, we should find a clever way to assess the correlation of this feature with the survival rate.
# 
# One thing we can notice is the presence of the Title in the Name. And since Title is related to Social Class and Gender, it may be a good lead.

# In[ ]:


# We start by extracting the Title from the Name
train_df["Title"] = train_df["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)
# We then observe the distribution of the different titles
train_df[['Title','Name']].groupby(['Title'], as_index=False).count().sort_values(by='Name', ascending=False)


# Having now as many as 17 categorical values, we can still reduce this number by:
# * merging "Mlle" and "Ms" with "Miss"
# * merging "Mme" with "Mrs"
# * creating a "Royal" category including "Sir", "Countess" and "Lady"
# * keeping only Royal, Mr, Miss, Mrs and Master, and merging all the rest into one "Other" category

# In[ ]:


# Make the transformations
train_df["Title"] = train_df["Title"].replace(['Mlle', 'Ms'], 'Miss')
train_df["Title"] = train_df["Title"].replace('Mme', 'Mrs')
train_df["Title"] = train_df["Title"].replace(['Lady', 'Countess', 'Sir'], 'Royal')
train_df["Title"] = train_df["Title"].replace(['Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Other')


# #### Correlation with output
# Having reduced the number of Title values to just 5, correlation with survival rate can be done simply by pivoting features against each other.

# In[ ]:


# We print the influence of Title using pandas' groupby
train_df[["Title", "Survived"]].groupby(['Title'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


# Or we can visualize the trend with seaborn's barplot
sns.barplot(x="Title", y="Survived", data=train_df)


# A clear trend appears with the Title of the passengers:
# * Royal passengers all survived
# * Female passengers have a higher survival rate than male
# 
# #### Decision
# **Name** can definitely be replaced with **Title**.
# 
# #### Data completion

# In[ ]:


# Check for missing values in the training set
print ("Name feature: There are", len(train_df.index) - train_df.Name.count(), "/", len(train_df.index) ,"null values in the training set.")


# In[ ]:


# Check for missing values in the testing set
print ("Name feature: There are", len(test_df.index) - test_df.Name.count(), "/", len(test_df.index) ,"null values in the testing set.")


# No completion is needed in either of the datasets since there are 0 missing values.
# 
# #### Execution of the decision

# In[ ]:


# Replace Name with Title in the training set
train_df.drop(['Name'], inplace=True, axis=1)
# Apply the same transformations to testing set
test_df["Title"] = test_df["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)
test_df["Title"] = test_df["Title"].replace(['Mlle', 'Ms'], 'Miss')
test_df["Title"] = test_df["Title"].replace('Mme', 'Mrs')
test_df["Title"] = test_df["Title"].replace(['Lady', 'Countess', 'Sir'], 'Royal')
test_df["Title"] = test_df["Title"].replace(['Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Other')
test_df.drop(['Name'], inplace=True, axis=1)


# ### **Pclass** feature
# 
# #### Pre-analysis
# **Pclass** is a categorical feature with only 3 possible values. So we can directly proceed to correlation with survival.
# 
# #### Correlation with output
# Correlation with survival rate can be done simply by pivoting features against each other.

# In[ ]:


# Correlation using pandas' groupby
train_df[["Pclass", "Survived"]].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


# Correlation using seaborn's barplot
sns.barplot(x="Pclass", y="Survived", data=train_df)


# We can observe that passengers with higher ticket class have more chance of survival.
# 
# #### Decision
# **Pclass** to keep as-is for sure.
# 
# #### Data completion

# In[ ]:


# Check for missing values in the training set
print ("Pclass feature: There are", len(train_df.index) - train_df.Pclass.count(), "/", len(train_df.index) ,"null values in the training set.")


# In[ ]:


# Check for missing values in the testing set
print ("Pclass feature: There are", len(test_df.index) - test_df.Pclass.count(), "/", len(test_df.index) ,"null values in the testing set.")


# No completion is needed in either of the datasets since there are 0 missing values.
# 
# #### Execution of the decision
# 
# * For training set: nothing to do
# * For testing set: nothing to do

# ### **Sex** feature
# 
# #### Pre-analysis
# **Sex** is a categorical feature only 2 possible values. So we can directly proceed to correlation with survival.
# 
# #### Correlation with output
# Correlation with survival rate can be done simply by pivoting features against each other.

# In[ ]:


# Correlation using pandas' groupby
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


# Correlation using seaborn's barplot
sns.barplot(x="Sex", y="Survived", data=train_df)


# We can observe that female passengers have more chance of survival.
# 
# #### Decision
# **Sex** to keep as-is for sure.
# 
# #### Data completion

# In[ ]:


# Check for missing values in the training set
print ("Sex feature: There are", len(train_df.index) - train_df.Sex.count(), "/", len(train_df.index) ,"null values in the training set.")


# In[ ]:


# Check for missing values in the testing set
print ("Sex feature: There are", len(test_df.index) - test_df.Sex.count(), "/", len(test_df.index) ,"null values in the testing set.")


# No completion is needed in either of the datasets since there are 0 missing values.
# 
# #### Execution of the decision
# 
# * For training set: nothing to do
# * For testing set: nothing to do

# ### **SibSp** feature
# 
# #### Pre-analysis
# **SibSp** is a numerical feature with only 7 possible values. This number is small, hence acceptable. So we can directly proceed to correlation with survival.
# 
# #### Correlation with output
# Correlation with survival rate can be done simply by pivoting features against each other.

# In[ ]:


# Correlation using pandas' groupby
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


# Correlation using seaborn's barplot
sns.barplot(x="SibSp", y="Survived", data=train_df)


# We can observe that passengers with a small family size (number of siblings or spouses < 3) are more likely to survive. Then come passengers with 0 sibling or spouse. And finally passengers with SibSp >= 3.
# 
# #### Decision
# **SibSp** to keep as-is for sure.
# 
# #### Data completion

# In[ ]:


# Check for missing values in the training set
print ("SibSp feature: There are", len(train_df.index) - train_df.SibSp.count(), "/", len(train_df.index) ,"null values in the training set.")


# In[ ]:


# Check for missing values in the testing set
print ("SibSp feature: There are", len(test_df.index) - test_df.SibSp.count(), "/", len(test_df.index) ,"null values in the testing set.")


# No completion is needed in either of the datasets since there are 0 missing values.
# 
# #### Execution of the decision
# * For training set: nothing to do
# * For testing set: nothing to do

# ### **Parch** feature
# 
# #### Pre-analysis
# **Parch** is a numerical feature with only 7 possible values. This number is small, hence acceptable. So we can directly proceed to correlation with survival.
# 
# #### Correlation with output
# Correlation with survival rate can be done simply by pivoting features against each other.

# In[ ]:


# Correlation using pandas' groupby
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


# Correlation using seaborn's barplot
sns.barplot(x="Parch", y="Survived", data=train_df)


# We have kind of the same trending as SibSp: if you're not travelling alone, the bigger your family is, the less chance of survival.
# 
# #### Decision
# **Parch** to keep as-is for sure.
# 
# #### Data completion

# In[ ]:


# Check for missing values in the training set
print ("Parch feature: There are", len(train_df.index) - train_df.Parch.count(), "/", len(train_df.index) ,"null values in the training set.")


# In[ ]:


# Check for missing values in the testing set
print ("Parch feature: There are", len(test_df.index) - test_df.Parch.count(), "/", len(test_df.index) ,"null values in the testing set.")


# No completion is needed in either of the datasets since there are 0 missing values.
# 
# #### Execution of the decision
# * For training set: nothing to do
# * For testing set: nothing to do

# ### **Embarked** feature
# 
# #### Pre-analysis
# **Embarked** is a categorical feature only 3 possible values. So we can directly proceed to correlation with survival.
# 
# #### Correlation with output
# Correlation with survival rate can be done simply by pivoting features against each other.

# In[ ]:


# We visualize the correlation
sns.barplot(x="Embarked", y="Survived", data=train_df)


# A trend appears.
# 
# #### Decision
# **Embarked** to keep as-is.
# 
# #### Data completion

# In[ ]:


# Check for missing values in the training set
print ("Embarked feature: There are", len(train_df.index) - train_df.Embarked.count(), "/", len(train_df.index) ,"null values in the training set.")


# In[ ]:


# Check for missing values in the testing set
print ("Embarked feature: There are", len(test_df.index) - test_df.Embarked.count(), "/", len(test_df.index) ,"null values in the testing set.")


# * For training set: We need to fill the missing values in the training set.
# 
# The feature misses only 2 values. Being a categorical feature, a quick and efficient way in this case is to assign the most occurring value in the feature. We already know it's "S" because of the dataframe description in the beginning.
# 
# But we can always confirm it.

# In[ ]:


# Most frequent port of embarkation
most_freq_port = train_df.Embarked.dropna().mode()[0]
# Fill the training set
train_df['Embarked'] = train_df['Embarked'].fillna(most_freq_port)
# Print the value
print ("We have filled the missing values with: ",most_freq_port)
print ("Length of 'Embarked' series after modification: ",train_df.Embarked.count())


# * For testing set: no completion is needed
# 
# #### Execution of the decision
# * For training set: nothing to do
# * For testing set: nothing to do

# ### **Fare** feature
# 
# #### Pre-analysis
# Being a continuous numerical feature with lots of possibilities, we can already imagine replacing it with fare ranges. However, we still need to confirm the correlation with the output.
# 
# #### Correlation with output
# Since we have lots of possibilities of **Fare** values, correlation with survival rate cannot be done simply by pivoting features against each other. In this case, visualizing the distribution in a histogram is better.

# In[ ]:


g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Fare', bins=6)


# We can observe that passengers who paid less than 100 dollars have less chance of survival.
# 
# #### Decision
# **Fare** to keep for sure, but to be divided into a limited number of ranges.
# 
# #### Data completion

# In[ ]:


# Check for missing values in the training set
print ("Fare feature: There are", len(train_df.index) - train_df.Fare.count(), "/", len(train_df.index) ,"null values in the training set.")


# In[ ]:


# Check for missing values in the testing set
print ("Fare feature: There are", len(test_df.index) - test_df.Fare.count(), "/", len(test_df.index) ,"null values in the testing set.")


# * For training set: no completion is needed
# * For testing set: we need to fill the missing value
# 
# Since the feature only misses 1 value, we can simply fill it with the median of the existing values.

# In[ ]:


# Median of existing values
missing_value = test_df.Fare.median()
# Fill the missing value
test_df['Fare'] = test_df['Fare'].fillna(missing_value)
# Print the value
print ("We have filled the missing value with: ",missing_value)
print ("Length of 'Fare' series after modification: ",test_df.Fare.count())


# #### Execution of the decision
# 
# We need to divide the Fare feature into ranges. For this part, we can use the built-in method of pandas called "qcut" which is a quantile-based discretization function.

# In[ ]:


train_df.head()


# In[ ]:


# We create the feature in the training set
train_df['FareRangeRaw'] = pd.qcut(train_df['Fare'], 4) #if we try 5 we lose the trend
train_df.head()


# In[ ]:


# We visualize the correlation
sns.barplot(x="FareRangeRaw", y="Survived", data=train_df)


# The trend is conserved. Before approving the cut values, we must ensure that the testing set fits the ranges.

# In[ ]:


# Get max fare in testing set
test_df["Fare"].max()


# We can now proceed with the cut.

# In[ ]:


# We apply the binning and labels to both datasets
bins = [-1, 7.91, 14.454, 31.0, 600]
train_df['FareRange'] = pd.cut(train_df['Fare'], bins, labels=[1,2,3,4])
train_df.head()


# In[ ]:


test_df['FareRange'] = pd.cut(test_df['Fare'], bins, labels=[1,2,3,4])


# In[ ]:


# We replace Fare with FareRange in training set
train_df.drop(["FareRangeRaw","Fare"],inplace=True,axis=1)
# Same for testing set
test_df.drop(["Fare"],inplace=True,axis=1)


# ### **Age** feature
# 
# #### Pre-analysis
# Being a continuous numerical feature with lots of possibilities, we can already imagine replacing it with age ranges. However, we still need to confirm the correlation with the output.
# 
# #### Correlation with output
# Since we have lots of possibilities of **Age** values, correlation with survival rate cannot be done simply by pivoting features against each other. In this case, visualizing the distribution in a histogram is better.

# In[ ]:


g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=30)


# We can observe children under 4, and eldrely above 80 have more chance of survival. Most of 15-25 year olds did not survive.
# 
# #### Decision
# **Age** to keep for sure, but to be divided into a limited number of ranges.
# 
# #### Data completion

# In[ ]:


# Check for missing values in the training set
print ("Age feature: There are", len(train_df.index) - train_df.Age.count(), "/", len(train_df.index) ,"null values in the training set.")


# In[ ]:


# Check for missing values in the testing set
print ("Age feature: There are", len(test_df.index) - test_df.Age.count(), "/", len(test_df.index) ,"null values in the testing set.")


# **Age** is missing lots of values in both datasets. It being an important feature for the model, and also having so many missing values, we cannot simply assign the median or even a random value between mean and std variation. In this case, we prefer finding correlation between Age and other features in order to best choose the value to assign. However, in order to avoid useless complexification, we should first assess which features are most likely to be correlated with Age.
# 
# Our correlation options are [Name/Title, Pclass, Sex, Parch, SibSp, Embarked, Fare].
# * Parch, SibSp, Embarked & Fare can already be discarded as there is no immediate logic of correlation
# * We can consider Title, Pclass and Sex
# 
# So, let's visualize the correlation.

# In[ ]:


grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', hue="Title", size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()


# We can see on the graph that taking "Title" into consideration is not a good idea as it is a bit redundant with "Sex" and will also an important number of combinations. If we only keep "Pclass" and "Sex", we get exactly 6 combinations for "Age" possibilities, which can be accepted. Let's visualize it again.

# In[ ]:


grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()


# Looks good. We can now complete the missing values of Age with the median of each combination.

# In[ ]:


# We define a function that will compute the median of a feature corresponding to any given combination of 2 other features
def fillna_custom(df,ax1,val1,ax2,val2,nanax):
    return df[(df[ax1] == val1) & (df[ax2] == val2)][[nanax]].median()[0]
    
# We fill the missing values in both datasets
list_sex = ["male", "female"]
list_pclass = [1, 2, 3]
for sex in list_sex:
    for pclass in list_pclass:
        train_df.loc[(train_df.Age.isnull()) & (train_df.Sex == sex) & (train_df.Pclass == pclass),'Age'] = fillna_custom(train_df.copy(),"Sex",sex,"Pclass",pclass,"Age")
        test_df.loc[(test_df.Age.isnull()) & (test_df.Sex == sex) & (test_df.Pclass == pclass),'Age'] = fillna_custom(test_df.copy(),"Sex",sex,"Pclass",pclass,"Age")


# We can check that we have no more missing values

# In[ ]:


print ("Length of Age in training set:", train_df.Age.count())
print ("Length of Age in testing set:", test_df.Age.count())


# #### Execution of the decision
# Now that we have found a way to complete the missing **Age** values, we can try the division into ranges. For this part, we can use the built-in method of pandas called "cut" which is a binning function.

# In[ ]:


# We cut the Age feature
bins = [0, 10, 20, 30, 40, 50, 75, 80]
train_df['AgeRangeRaw'] = pd.cut(train_df['Age'], bins)
# We visualize the correlation
sns.barplot(x="AgeRangeRaw", y="Survived", data=train_df)


# The trend is conserved. Before approving the cut values, we must ensure that the testing set fits the ranges.

# In[ ]:


test_df["Age"].max()


# We can now proceed with the cut.

# In[ ]:


# We apply the binning and labels to both datasets
bins = [0, 10, 20, 30, 40, 50, 75, 80]
train_df['AgeRange'] = pd.cut(train_df['Age'], bins, labels=[1,2,3,4,5,6,7])
test_df['AgeRange'] = pd.cut(test_df['Age'], bins, labels=[1,2,3,4,5,6,7])


# In[ ]:


# We replace Age with AgeRange in training set
train_df.drop(["AgeRangeRaw","Age"],inplace=True,axis=1)
# Same for testing set
test_df.drop(["Age"],inplace=True,axis=1)


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# ## 4. Finalize X_train, Y_train and X_test
# In this section we finalize our data vectors for the machine learning model, by following these steps:
# 1. Confirm consistency of training set & test set
# 2. Initialize X_train, Y_train & X_test
# 3. Convert all values of X to numerical labels

# ### Consistency of training set & test set
# Here we just make sure that train_df and test_df have the same construction, minus the output.

# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# OK, we can proceed.

# ### Initialization of X_train, Y_train and X_test

# In[ ]:


# X_train is train_df without the output column "Survived"
X_train = train_df.drop("Survived", axis=1)
X_train.head()


# In[ ]:


# Y_train is only the output "Survived" of train_df
Y_train = train_df["Survived"]
Y_train.head()


# In[ ]:


# X_test is simply test_df
X_test = test_df
X_test.head()


# ### Conversion of all features to numerical labels
# The X features to be transformed are:
# * Title
# * Sex
# * Embarked

# In[ ]:


# Title feature mapping
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Other": 6}
# Application to X_train
X_train['Title'] = X_train['Title'].map(title_mapping)
# Application to X_test
X_test['Title'] = X_test['Title'].map(title_mapping)


# In[ ]:


# Sex feature mapping
sex_mapping = {"male": 1, "female": 2}
# Application to X_train
X_train['Sex'] = X_train['Sex'].map(sex_mapping)
# Application to X_test
X_test['Sex'] = X_test['Sex'].map(sex_mapping)


# In[ ]:


# Embarked feature mapping
embarked_mapping = {"S": 1, "C": 2, "Q": 3}
# Application to X_train
X_train['Embarked'] = X_train['Embarked'].map(embarked_mapping)
# Application to X_test
X_test['Embarked'] = X_test['Embarked'].map(embarked_mapping)


# # Machine Learning
# 
# ## Execution of the different algorithms
# 
# Here we use SKLearn tools to run the different ML algorithms on our finalized dataset.

# In[ ]:


# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
# Support Vector Machines
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
# KNN neighbors
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
# Perceptron
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
# Random forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)


# ## Different scores

# In[ ]:


# Scores
acc_decision_tree = decision_tree.score(X_train, Y_train) * 100
acc_linear_svc = linear_svc.score(X_train, Y_train) * 100
acc_svc = svc.score(X_train, Y_train) * 100
acc_knn = knn.score(X_train, Y_train) * 100
acc_gaussian = gaussian.score(X_train, Y_train) * 100
acc_perceptron = perceptron.score(X_train, Y_train) * 100
acc_sgd = sgd.score(X_train, Y_train) * 100
acc_random_forest = random_forest.score(X_train, Y_train) * 100

scores_df = pd.DataFrame({ 'Algorithm' : ["Decision Tree","Linear SVC","Support Vector Machines","KNN neighbors","Gaussian Naive Bayes","Perceptron","Stochastic Gradient Descent","Random forest"],
                          'Score' : [acc_decision_tree,acc_linear_svc,acc_svc,acc_knn,acc_gaussian,acc_perceptron,acc_sgd,acc_random_forest]})

scores_df.sort_values(by='Score', ascending=False)


# ## Graphical visualization of our Decision Tree

# In[ ]:


import graphviz
from sklearn import tree
dot_data = tree.export_graphviz(decision_tree, out_file=None) 
#graph = graphviz.Source(dot_data) 
#graph.render("iris")
graph = graphviz.Source(dot_data)  
graph 


# In[ ]:




