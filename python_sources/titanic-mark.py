#!/usr/bin/env python
# coding: utf-8

# # Titanic Analyst Report

# ## Step 1: Question or problem definition
# 
# **Project Summary.**  
# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.
# 
# **Goal.**   
# It is your job to predict if a passenger survived the sinking of the Titanic or not. 
# For each in the test set, you must predict a 0 or 1 value for the variable.

# ## Step 2: Acquire training and testing data
# 
# ### 2.1 Load Data Modelling Libraries

# In[ ]:


# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
import re

import warnings
warnings.filterwarnings('ignore')

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style='white', context='notebook', palette='deep')

from collections import Counter

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# Decision_Tree
from sklearn import tree


# ### 2.2 Acquire data

# In[ ]:


# Acquire filenames
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# Load Train and Test data
# Combine Train and Test data
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]
dataset =  pd.concat(objs=[train_df, test_df], axis=0).reset_index(drop=True)


# In[ ]:


train_df.shape, test_df.shape, dataset.shape


# ## Step3: Wrangle, prepare, cleanse the data
# 
# ### 3.1 Analyze by describing data
# 
# **3.1.1 Which features are available in the dataset?**

# In[ ]:


print(train_df.columns.values)


# **3.1.2 Which features are categorical?**  
# *Within categorical features are the values nominal, ordinal, ratio, or interval based?*
# 
# - Categorical: Survived, Sex and Embarked. 
# - Ordinal: Pclass.
# 
# **3.1.3 Which features are numerical?**  
# *Within numerical features are the values discrete, continuous, or timeseries based? *
# 
# - Continous: Age and Fare. 
# - Discrete: SibSp and Parch.

# In[ ]:


# preview the data
train_df.head()


# **3.1.4 Which features are mixed data types?**  
# 
# - Ticket is a mix of numeric and alphanumeric data types. Cabin is alphanumeric.
# 
# **3.1.5 Which features may contain errors or typos?**
# 
# - Name feature may contain errors or typos as there are several ways used to describe a name including titles, round brackets, and quotes used for alternative or short names.

# In[ ]:


print(train_df.info())


# In[ ]:


# Check for Null values
print(train_df.isnull().sum().sort_values(ascending=False).head())
test_df.isnull().sum().sort_values(ascending=False).head()


# **3.1.6 Which features contain blank, null or empty values?**
# 
# - Cabin > Age > Embarked features contain a number of null values in that order for the training dataset.
# - Cabin > Age are incomplete in case of test dataset.
# 
# **3.1.7 What are the data types for various features?**
# 
# - Seven features are integer or floats.
# - Five features are strings (object).

# In[ ]:


train_df.describe(include=['object'])


# **3.1.8 What is the distribution of categorical features?**
# 
# - Names are unique across the dataset (count=unique=891)
# - Sex variable as two possible values with 65% male (top=male, freq=577/count=891).
# - Ticket feature has high ratio (24%) of duplicate values (unique=681).
# - Cabin values have several dupicates across samples. Alternatively several passengers shared a cabin.
# - Embarked takes three possible values. S port used by most passengers (top=S)

# In[ ]:


# Summarie and statistics
train_df.describe()

# Review survived rate using percentiles
#train_df['Survived'].quantile([.61,.62])
#train_df['Pclass'].quantile([.4, .45])
#train_df['SibSp'].quantile([.65, .7])
#train_df[['Age', 'Fare']].quantile([.05,.1,.2,.4,.6,.8,.9,.99])


# **3.1.9 What is the distribution of numerical feature values across the samples?**
# 
# - Total samples are 891 or 40% of the actual number of passengers on board the Titanic (2,224).
# - Survived is a categorical feature with 0 or 1 values.
# - Around 38% samples survived representative of the actual survival rate at 32%.
# - Nearly 45% of the passengers had in No.3 pclass.
# - Few elderly passengers (<1%) within age range 65-80.
# - Nearly 30% of the passengers had siblings and/or spouse aboard.
# - Most passengers (> 75%) did not travel with parents or children.
# - Fares varied significantly with few passengers (<1%) paying as high as $512.

# ### 3.2 Outlier detection

# In[ ]:


# Outlier detection 

def detect_outliers(df,n,features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers   


# In[ ]:


# detect outliers from Age, SibSp , Parch and Fare
Outliers_to_drop = detect_outliers(train_df,2,["Age","SibSp","Parch","Fare"])


# Since outliers can have a dramatic effect on the prediction (espacially for regression problems), i choosed to manage them.
# 
# I decided to detect outliers from the numerical values features (Age, SibSp, Sarch and Fare). Then, i considered outliers as rows that have at least two outlied numerical values.

# In[ ]:


# Show the outliers rows
train_df.loc[Outliers_to_drop]


# In[ ]:


test_df.describe()


# We detect 10 outliers. The 28, 89 and 342 passenger have an high Ticket Fare
# 
# The 7 others have very high values of SibSP.
# 
# because test data have high ticket Fare and high values of SibSp, so we don't drop the outliers.

# ### 3.3 Assumtions based on data analysis
# 
# **Correlating.**
# 
# 1. We want to know how well does each feature correlate with Survival. We want to do this early in our project and match these quick correlations with modelled correlations later in the project.
# 
# **Completing.**
# 
# 1. We may want to complete Age feature as it is definitely correlated to survival.
# 2. We may want to complete the Embarked feature as it may also correlate with survival or another important feature.
# 
# **Correcting.**
# 
# 1. Ticket feature may be dropped from our analysis as it contains high ratio of duplicates (22%) and there may not be a correlation between Ticket and survival.
# 2. Cabin feature may be dropped as it is highly incomplete or contains many null values both in training and test dataset.
# 3. PassengerId may be dropped from training dataset as it does not contribute to survival.
# 4. Name feature is relatively non-standard, may not contribute directly to survival, so maybe dropped.
# 
# **Creating.**
# 
# 1. We may want to create a new feature called Family based on Parch and SibSp to get total count of family members on board.
# 2. We may want to engineer the Name feature to extract Title as a new feature.
# 3. We may want to create new feature for Age bands. This turns a continous numerical feature into an ordinal categorical feature.
# 4. We may also want to create a Fare range feature if it helps our analysis.
# 
# **Classifying.**
# 
# We may also add to our assumptions based on the problem description noted earlier.
# 
# 1. Women (Sex=female) were more likely to have survived.
# 2. Children (Age<?) were more likely to have survived. 
# 3. The upper-class passengers (Pclass=1) were more likely to have survived.

# ### 3.4. Correlating Features analysis
# 
# #### 3.4.1 Numerical features

# In[ ]:


# Correlation matrix between numerical values (SibSp Parch Age and Fare values) and Survived 
g = sns.heatmap(train_df[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot=True,fmt = ".2f",cmap = "coolwarm",alpha=.8,vmin=-1, vmax=1)


# Only Fare feature seems to have a significative correlation with the survival probability.
# 
# It doesn't mean that the other features are not usefull. Subpopulations in these features can be correlated with the survival. To determine this, we need to explore in detail these features

# #### SibSP

# In[ ]:


# Explore SibSp feature vs Survived
g = sns.factorplot(x="SibSp",y="Survived",data=train_df,kind="bar", size = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# It seems that passengers having a lot of siblings/spouses have less chance to survive
# 
# Single passengers (0 SibSP) or with two other persons (SibSP 1 or 2) have more chance to survive
# 
# This observation is quite interesting, we can consider a new feature describing these categories (See feature engineering)

# #### Parch

# In[ ]:


# Explore Parch feature vs Survived
g  = sns.factorplot(x="Parch",y="Survived",data=train_df,kind="bar", size = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# Small families have more chance to survive, more than single (Parch 0), medium (Parch 3,4) and large families (Parch 5,6 ).
# 
# Be carefull there is an important standard deviation in the survival of passengers with 3 parents/children 

# #### Age

# In[ ]:


# Explore Age vs Survived
g = sns.FacetGrid(train_df, col='Survived')
g = g.map(sns.distplot, "Age")


# Age distribution seems to be a tailed distribution, maybe a gaussian distribution.
# 
# We notice that age distributions are not the same in the survived and not survived subpopulations. Indeed, there is a peak corresponding to young passengers, that have survived. We also see that passengers between 60-80 have less survived.
# 
# So, even if "Age" is not correlated with "Survived", we can see that there is age categories of passengers that of have more or less chance to survive.
# 
# It seems that very young passengers have more chance to survive.

# In[ ]:


# Explore Age distibution 
g = sns.kdeplot(train_df["Age"][(train_df["Survived"] == 0) & (train_df["Age"].notnull())], color="Red", shade = True)
g = sns.kdeplot(train_df["Age"][(train_df["Survived"] == 1) & (train_df["Age"].notnull())], ax =g, color="Blue", shade= True)
g.set_xticks(range(0,100,10))
g.set_xlabel("Age")
g.set_ylabel("Frequency")
g = g.legend(["Not Survived","Survived"])


# When we superimpose the two densities , we cleary see a peak correponsing (between 0 and 5) to babies and very young childrens.

# #### Fare

# In[ ]:


dataset["Fare"].isnull().sum()


# In[ ]:


#Fill Fare missing values with the median value
dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())


# Since we have one missing value , i decided to fill it with the median value which will not have an important effect on the prediction.

# In[ ]:


# Explore Fare distribution 
g = sns.distplot(dataset["Fare"], color="r", label="Skewness : %.2f"%(dataset["Fare"].skew()))
g = g.legend(loc="best")


# As we can see, Fare distribution is very skewed. This can lead to overweigth very high values in the model, even if it is scaled.
# 
# In this case, it is better to transform it with the log function to reduce this skew.

# In[ ]:


# Apply log to Fare to reduce skewness distribution
dataset["Fare"] = dataset["Fare"].map(lambda i: np.log(i+1))


# In[ ]:


g = sns.distplot(dataset["Fare"], color="b", label="Skewness : %.2f"%(dataset["Fare"].skew()))
g = g.legend(loc="best")


# Skewness is clearly reduced after the log transformation

# #### 3.4.2 Categorical features

# #### Sex

# In[ ]:


g = sns.barplot(x="Sex",y="Survived",data=train_df)
g = g.set_ylabel("Survival Probability")


# In[ ]:


train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# Sex We confirm the observation during problem definition that Sex=female had very high survival rate at 74%.

# #### Pclass

# In[ ]:


# Explore Pclass vs Survived
g = sns.factorplot(x="Pclass",y="Survived",data=train_df,kind="bar", size = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# In[ ]:


# Explore Pclass vs Survived by Sex
g = sns.factorplot(x="Pclass", y="Survived", hue="Sex", data=train_df,
                   size=6, kind="bar", palette="muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# The passenger survival is not the same in the 3 classes. First class passengers have more chance to survive than second class and third class passengers.
# 
# This trend is conserved when we look at both male and female passengers.

# #### Embarked

# In[ ]:


dataset["Embarked"].isnull().sum()


# In[ ]:


#Fill Embarked nan values of dataset set with 'S' most frequent value
dataset["Embarked"] = dataset["Embarked"].fillna("S")


# Since we have two missing values , i decided to fill them with the most fequent value of "Embarked" (S).

# In[ ]:


# Explore Embarked vs Survived 
g = sns.factorplot(x="Embarked", y="Survived",  data=train_df,
                   size=6, kind="bar", palette="muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# It seems that passenger coming from Cherbourg (C) have more chance to survive.
# 
# My hypothesis is that the proportion of first class passengers is higher for those who came from Cherbourg than Queenstown (Q), Southampton (S).
# 
# Let's see the Pclass distribution vs Embarked

# In[ ]:


# Explore Pclass vs Embarked 
g = sns.factorplot("Pclass", col="Embarked",  data=train_df,
                   size=6, kind="count", palette="muted")
g.despine(left=True)
g = g.set_ylabels("Count")


# Indeed, the third class is the most frequent for passenger coming from Southampton (S) and Queenstown (Q), whereas Cherbourg passengers are mostly in first class which have the highest survival rate.

# ### 3.5 Completing Features
# 
# #### 3.5.1 Age
# 
# As we see, Age column contains 256 missing values in the whole dataset.
# 
# Since there is subpopulations that have more chance to survive (children for example), it is preferable to keep the age feature and to impute the missing values. 
# 
# To adress this problem, i looked at the most correlated features with Age (Sex, Parch , Pclass and SibSP).

# In[ ]:


# convert Sex into categorical value 0 for male and 1 for female
dataset["Sex"] = dataset["Sex"].map({"male": 0, "female":1})


# In[ ]:


g = sns.heatmap(dataset[["Age","Sex","SibSp","Parch","Pclass"]].corr(),fmt = ".2f",cmap="BrBG",annot=True,alpha=.8,vmin=-1, vmax=1)


# The correlation map confirms the factorplots observations except for Parch. Age is not correlated with Sex, but is negatively correlated with Pclass, Parch and SibSp.
# 
# In the plot of Age in function of Parch, Age is growing with the number of parents / children. But the general correlation is negative.
# 
# So, i decided to use SibSP, Parch and Pclass in order to impute the missing ages.
# 
# The strategy is to fill Age with the median age of similar rows according to Pclass, Parch and SibSp.

# In[ ]:


# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
g = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
g.map(plt.hist, 'Age', bins=20, alpha=.7)
g.add_legend()


# Infant passengers in Pclass=2 and Pclass=3 mostly survived. 
# 
# a few of young passengers in Pclass=3 survived.
# 
# Pclass varies in terms of Age distribution of passengers.  

# In[ ]:


# Filling missing value of Age 

## Fill Age with the median age of similar rows according to Pclass, Parch and SibSp
# Index of NaN age rows
index_NaN_age = list(dataset["Age"][dataset["Age"].isnull()].index)

for i in index_NaN_age :
    age_med = dataset["Age"].median()
    age_pred = dataset["Age"][((dataset['SibSp'] == dataset.iloc[i]["SibSp"]) & (dataset['Parch'] == dataset.iloc[i]["Parch"]) & (dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()
    if not np.isnan(age_pred) :
        dataset['Age'].iloc[i] = age_pred
    else :
        dataset['Age'].iloc[i] = age_med


# In[ ]:


dataset['Age'].isnull().sum()


# In[ ]:


g = sns.factorplot(x="Survived", y = "Age",data = train_df, kind="box")
g = sns.factorplot(x="Survived", y = "Age",data = train_df, kind="violin")


# No difference between median value of age in survived and not survived subpopulation.
# 
# But in the violin plot of survived passengers, we still notice that very young passengers have higher survival rate.

# ### 3.6 Feature engineering

# #### 3.6.1 Correcting by dropping features
# 
# Based on our assumptions and decisions we want to drop the Cabin (correcting #2) and Ticket (correcting #1) features.

# In[ ]:


print("Before", dataset.shape)

dataset.drop(['Cabin', 'Ticket', 'PassengerId'], axis=True, inplace=True)

print('After', dataset.shape)


# #### 3.6.2 Creating new feature extracting from existing
# 
# We want to analyze if Name feature can be engineered to extract titles and test correlation between titles and survival, before dropping Name and PassengerId features.

# In[ ]:


dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.', expand=False)


# In[ ]:


pd.crosstab(dataset['Title'], dataset['Sex'])


# In[ ]:


g = sns.countplot(x="Title",data=dataset)
g = plt.setp(g.get_xticklabels(), rotation=45)


# There is 17 titles in the dataset, most of them are very rare and we can group them in 4 categories.

# In[ ]:


# Convert to categorical values Title 
dataset["Title"] = dataset["Title"].replace(['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona',"Ms" ,"Mme","Mlle"],'Rare')


# In[ ]:


g = sns.countplot(dataset["Title"])


# In[ ]:


g = sns.factorplot(x="Title",y="Survived",data=dataset,kind="bar")
g = g.set_ylabels("survival probability")


# It is interesting to note that passengers with rare title have more chance to survive.

# In[ ]:


# Drop Name variable
dataset.drop(labels = ["Name"], axis = 1, inplace = True)


# #### Family size
# 
# We can create a new feature for FamilySize which combines Parch and SibSp. This will enable us to drop Parch and SibSp from our datasets.

# In[ ]:


# Create a family size descriptor from SibSp and Parch
dataset['Familysize'] = dataset['Parch'] + dataset['SibSp'] + 1


# In[ ]:


g = sns.factorplot(x="Familysize",y="Survived",data = dataset)
g = g.set_ylabels("Survival Probability")


# The family size seems to play an important role, survival probability is worst for large families.
# 
# Additionally, i decided to created 4 categories of family size.

# In[ ]:


# Create new feature of family size
bins = [0,1,2,4,8]
labels = ['Single','SmallF','MedF','LargeF']
dataset["Familysize"] = pd.cut(dataset["Familysize"], bins, labels=labels)


# In[ ]:


g = sns.factorplot(x="Familysize",y="Survived",data=dataset,kind="bar")
g = g.set_ylabels("survival probability")


# In[ ]:


# Drop Fsize/SibSp/Parch variable
dataset.drop(labels = ['SibSp','Parch'], axis = 1, inplace = True)


# In[ ]:


# Create categorical values for Pclass
dataset["Pclass"] = dataset["Pclass"].astype("category")
dataset["Sex"] = dataset["Sex"].astype("category")


# In[ ]:


dataset = pd.get_dummies(dataset, columns = ["Pclass"],prefix="Pc")
dataset = pd.get_dummies(dataset, columns = ["Embarked"],prefix="Em")
dataset = pd.get_dummies(dataset, columns = ["Familysize"],prefix="Fs")


# In[ ]:


dataset = pd.get_dummies(dataset)


# In[ ]:


dataset.head()


# In[ ]:


train_df.shape[0]


# ## Step 4: Model, predict and solve

# In[ ]:


## Separate train dataset and test dataset

train = dataset[:train_df.shape[0]]
X_test = dataset[train_df.shape[0]:]
X_test.drop(labels=["Survived"],axis = 1,inplace=True)


# In[ ]:


## Separate train features and label 
X_train = train.drop(labels = ["Survived"],axis = 1)
Y_train = train["Survived"].astype(int)


# In[ ]:


# Decision Tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
acc_decision_tree = round(clf.score(X_train, Y_train)*100, 2)
acc_decision_tree


# ## Step 5: Model evaluation

# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
#submission.to_csv('./input/submission.csv', index=False)


# In[ ]:




