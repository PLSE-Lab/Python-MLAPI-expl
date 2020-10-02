#!/usr/bin/env python
# coding: utf-8

# # Titanic EDA & Visualization Tutorial for Beginners

# The aim of the Titanic competetion is figure out which of the passengers in the "test.csv" file will survive based on the data analysis of people in the "train.csv"
# 
# It is a classification problem, and can be solved by logistic regression /decision tree/random forest
# 
# This notebook is limited to EDA and visualization and does not get into modelling.
# 
# Hope this helps!

# # Import Libraries

# In[ ]:


# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
import statistics

# visualization
import seaborn as sns
sns.set(style='white', context='notebook', palette='deep')
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve


from collections import Counter
import warnings
warnings.filterwarnings("ignore")



# # Read the data

# In[ ]:


train_df = pd.read_csv('../input/titanic/train.csv')
test_df = pd.read_csv('../input/titanic/test.csv')


# # EDA

# In[ ]:


display("train data", train_df)
display("test data", test_df)


# What do you observe?
# 
# The target varibale "Survived" is not present in the test data
# 
# the test data as 1/2 the data of training data
# 
# some data are numberical and some are categorical

# In[ ]:


print(train_df.columns.values)
print(train_df.describe())


# Note, categorical data columns are not shown in the describe function
# 
# Also note missing values in age!
# 
# some values like Pclass which are shown as numerical are actually ordinal or may be assiged as categorical
# 
# 
# which columns are continuous, discrete, ordinal and categorical? 
# 
# 
# 

# Continous: Age, Fare. 
# 
# Discrete: SibSp, Parch.
# 
# Ordinal: Pclass.
# 
# Categorical: Survived, Sex, Ticket, cabin and Embarked. 
# 
# 

# Lets observe the categorical values as well!

# In[ ]:


print(train_df.describe(include=['O']))


# Which of the fields are relevant?

# Is Pclass (Ticket class) a useful metrics or can it be discarded?

# In[ ]:


train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# The 1st class passengers or the rich ones have a higher probability to survive!!

# What about gender?

# In[ ]:


train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# Any thoughts on number of siblings/ spouses on baord (sibsp)?

# In[ ]:


train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# number of parents / children aboard the Titanic?

# In[ ]:


train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# # Some visualizations?

# In[ ]:


# Correlation matrix between numerical values (SibSp Parch Age and Fare values) and Survived 
g = sns.heatmap(train_df[["Survived","SibSp","Parch","Age","Fare"]].corr(),
                annot=True, fmt = ".2f", cmap = "coolwarm")


# what can you infer?

# In[ ]:


# Explore SibSp feature vs Survived
g = sns.catplot(x="SibSp",y="Survived",data=train_df,kind="bar", height = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# In[ ]:


#Explore Parch feature vs Survived
g  = sns.factorplot(x="Parch",y="Survived",data=train_df,kind="bar", height = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# In[ ]:


# Explore Age vs Survived
g = sns.FacetGrid(train_df, col='Survived')
g = g.map(sns.distplot, "Age")


# In[ ]:


# Explore Age distibution 
g = sns.kdeplot(train_df["Age"][(train_df["Survived"] == 0) & (train_df["Age"].notnull())], color="Red", shade = True)
g = sns.kdeplot(train_df["Age"][(train_df["Survived"] == 1) & (train_df["Age"].notnull())], ax =g, color="Blue", shade= True)
g.set_xlabel("Age")
g.set_ylabel("Frequency")
g = g.legend(["Not Survived","Survived"])


# Notice anything interesting in the charts?
# 
# 
# Make some more... 
# 
# Here is how to make a combo chart with 3 variables.. please add more!!
# 

# In[ ]:


# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')
grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()


# # Data Manipulation

# In[ ]:


#Join train and test datasets in order to obtain the same number of features during categorical conversion

combined =  pd.concat(objs=[train_df, test_df], axis=0).reset_index(drop=True)
combined


# In[ ]:


# Fill empty and NaNs values with NaN
combined = combined.fillna(np.nan)

# Check for Null values
combined.isnull().sum()


# In[ ]:


#Fill Embarked nan values of dataset set with the most frequent value
embarked_mode=statistics.mode(combined["Embarked"])
combined["Embarked"] = combined["Embarked"].fillna(embarked_mode)


# In[ ]:


# Explore Age vs Sex, Parch , Pclass and SibSP
g = sns.catplot(y="Age",x="Sex",data=combined,kind="box")
g = sns.catplot(y="Age",x="Sex",hue="Pclass", data=combined,kind="box")
g = sns.catplot(y="Age",x="Parch", data=combined,kind="box")
g = sns.catplot(y="Age",x="SibSp", data=combined,kind="box")


# In[ ]:


# convert Sex into categorical value 0 for male and 1 for female
combined["Sex"] = combined["Sex"].map({"male": 0, "female":1})

g = sns.heatmap(combined[["Age","SibSp","Parch","Pclass"]].corr(),cmap="BrBG",annot=True)


# The correlation map confirms the factorplots observations except for Parch. Age is not correlated with Sex, but is negatively correlated with Pclass, Parch and SibSp.

# In[ ]:


# Filling missing value of Age 

## Fill Age with the median age of similar rows according to Pclass, Parch and SibSp
# Index of NaN age rows
index_NaN_age = list(combined["Age"][combined["Age"].isnull()].index)

for i in index_NaN_age:
    age_med = combined["Age"].median()
    age_pred = combined["Age"][((combined['SibSp'] == combined.iloc[i]["SibSp"]) 
                                & (combined['Parch'] == combined.iloc[i]["Parch"]) 
                                & (combined['Pclass'] == combined.iloc[i]["Pclass"]))].median()
    if not np.isnan(age_pred) :
        combined['Age'].iloc[i] = age_pred
    else :
        combined['Age'].iloc[i] = age_med


# In[ ]:


# Get Title from Name
combined_title = [i.split(",")[1].split(".")[0].strip() for i in combined["Name"]]
combined["Title"] = pd.Series(combined_title)
combined["Title"].head()


# In[ ]:


g = sns.countplot(x="Title",data=combined)
g = plt.setp(g.get_xticklabels(), rotation=45) 


# In[ ]:


title=combined["Title"].unique()
title


# In[ ]:


# Convert to categorical values Title 
combined["Title"] = combined["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
combined["Title"] = combined["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
combined["Title"] = combined["Title"].astype(int)

g = sns.catplot(x="Title",y="Survived",data=combined,kind="bar")
g = g.set_xticklabels(["Master","Miss-Mrs","Mr","Rare"])
g = g.set_ylabels("survival probability")


# Women & children first! Followed by royalty. 
# Commoners are low priority!

# In[ ]:


# Create a family size descriptor from SibSp and Parch
combined["Fsize"] = combined["SibSp"] + combined["Parch"] + 1


# In[ ]:


g = sns.catplot(x="Fsize",y="Survived",data=combined,kind="bar")
g = g.set_ylabels("survival probability")


# In[ ]:


# Create new feature of family size
combined['Single'] = combined['Fsize'].map(lambda s: 1 if s == 1 else 0)
combined['SmallF'] = combined['Fsize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
combined['LargeF'] = combined['Fsize'].map(lambda s: 1 if s >= 5 else 0)


# In[ ]:


# convert to indicator values Title and Embarked 
combined = pd.get_dummies(combined, columns = ["Title"])
combined = pd.get_dummies(combined, columns = ["Embarked"], prefix="Em")
combined.head()


# In[ ]:


combined.columns.values


# In[ ]:


# Create categorical values for Pclass
combined["Pclass"] = combined["Pclass"].astype("category")
combined = pd.get_dummies(combined, columns = ["Pclass"],prefix="Pc")


# In[ ]:


# Drop variable
combined.drop(labels = ["Name",'SibSp', 'Parch','Fsize','Cabin',"Ticket"], axis = 1, inplace = True)

#you may decide some are useful and retain. Adding a few of the above will improve the final solution. 
# try it out. see if you can figure out what helps and what does not


# In[ ]:


combined.info()


# What other vizualisation do you want to try out?
# 
# What else would you like to do as part of EDA?
# 
# Try this out and let me know in comments
# 
# 
# Thanks!!

# # Source and Acknowledgements:
# 
# 1. https://www.kaggle.com/startupsci/titanic-data-science-solutions
# 
# 2. https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling
# 
# 3. https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy
