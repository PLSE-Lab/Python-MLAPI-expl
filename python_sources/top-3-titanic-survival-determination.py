#!/usr/bin/env python
# coding: utf-8

# # Titanic survival determination
# Step by Step Guide:
# 1. Importing the Libraries
# 2. Importing the Dataset 
# 3. Dataset Analysis
#     * 3.1 Observing the data
#     * 3.2 Removing outliers   
#     * 3.3 Determining missing values
#     * 3.4 Joining Train/Test Data
# 4. Visualizing and Comparing Features
#     * 4.1 Correlation heatmap 
#     * 4.2 Pclass
#     * 4.3 Sex
#     * 4.4 SibSp
#     * 4.5 Parch 
#     * 4.6 Embarked
#     * 4.7 Cabin    
# 5. Removing Missing Values
#     * 5.1 Age
# 6. Feature Engineering 
#     * 6.1 Title feature
#     * 6.2 Family total
#     * 6.3 Creating age groups
#     * 6.4 Solving fare skewness 
#     * 6.5 Mapping categorical sex feature
#     * 6.6 Mapping categorical embarked feature
#     * 6.7 Removing non-essential features
#     * 6.8 Get categorical dummies
# 7. Building/Training/Evaluating our models
#     * 7.1 Seperating Train/Test dataset
#     * 7.2 Modelling various classifiers
#     * 7.3 Hyperparameter tuning
#     * 7.4 Submitting

# # 1 - Importing the Libraries

# In[ ]:


#Importing the data analysis libraries
import numpy as np # linear algebra
import pandas as pd # data processing

#Importing the visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
#Ensuring that we don't see any warnings while running the cells
import warnings
warnings.filterwarnings('ignore') 

#Importing the counter
from collections import Counter

#Importing sci-kit learn libraries that we will need for this project
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


# # 2 - Importing the Dataset

# In[ ]:


#Reading the data from the given files and creating a training and test dataset
train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")


# # 3 - Dataset Analysis

# ## 3.1 Observing the data

# In[ ]:


train.sample(10)


# In[ ]:


train.describe(include="all")


# ## 3.2 Removing outliers

# To remove outliers I used the code from [here](https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling) as guidance
# * We will use 1.5 times the Inter Quartile Range to have close 1% of outliers as compared to 1 times IQR which can give 5% outliers and 2 times IQR which would include most extreme outliers

# In[ ]:


def detect_outliers(dataframe, n, features):
    
    outliers_indices = []
    
    for feature in features:
        
        #determining the upper and lower quartiles
        Quart1 = dataframe[feature].quantile(0.25)
        Quart3 = dataframe[feature].quantile(0.75)
        
        #determining the upper and lower outlier thresholds to remove the outliers
        upper_outlier_threshold = Quart3 + (Quart3 - Quart1) * 1.5
        lower_outlier_threshold = Quart1 - (Quart3 - Quart1) * 1.5
        
        #finding the outliers and saving their indices in the form of a list, according to the given threshold
        feature_outliers_list = dataframe[(dataframe[feature] > upper_outlier_threshold) | (dataframe[feature] < lower_outlier_threshold)].index
        
        #appending the outliers for each feature to the main outliers_indices list
        outliers_indices.extend(feature_outliers_list)
        
    #Selecting features that have more than 2 outliers
    return list(a for a, b in Counter(outliers_indices).items() if b > n)


# In[ ]:


outliers = detect_outliers(train, 2, ["Age", "SibSp", "Fare", "Parch"])
train.loc[outliers]


# Observations:
# * 3 outliers are due to a very high fare
# * 7 Outliers are due to a very high number of SibSp

# In[ ]:


train = train.drop(outliers, axis = 0).reset_index(drop = True)


# ## 3.3 Determining the missing values

# In[ ]:


print(pd.isnull(train).sum())


# Observations:
# * Only Age(170), Cabin(680) and Embarked(2) have missing values
# * The remaining columns have 0 missing values

# ## 3.4 - Joining Train/Test

# First we will combine the train and test data to ensure that we implement the feature engineering on all data, and we don't have discrepancies when modeling and evaluating.
# We will split the dataframe again after the feature engineering process.

# In[ ]:


df =  pd.concat(objs=[train, test], axis=0).reset_index(drop=True)
df.describe(include = "all")


# # 4 - Visualizing and Comparing the Features

# ## 4.1 - Correlation heatmap

# In[ ]:


sns.heatmap(df[["Survived","Age","Pclass", "SibSp", "Parch", "Fare"]].corr(), cmap = 'coolwarm', annot = True)


# Observations: 
# * Fare has a significant effect on the Survival rate, Age has a negative impact whereas Parch and Sibsp have a small impact
# 
# Assumptions:
# * High Fare payers may get preference due to t

# ## Comparing the effect of different features on the survival

# In[ ]:


#My function function to visualize and count the values in each category of each feature
def bar_plot(variable):
    
    #This code is used to solve problem when there are no survivors for a category, which causes an error in the display code
    feature_categories = df[variable].sort_values().unique()
    for category in feature_categories:
        temp_series = df["Survived"][df[variable] == category].value_counts(normalize = True)
        if temp_series.shape == (1,):
            temp_series = temp_series.append(pd.Series([0], index=[1]))
        elif temp_series.shape == (0,):
            continue
        print("Fraction of {} = {} who survived:".format(variable, category), temp_series[1])
    #visualize
    sns.barplot(x = df[variable],y = df["Survived"],  data = df).set_title('Fraction Survived With Respect To {}'.format(variable))


# ## 4.2 - PClass

# In[ ]:


bar_plot("Pclass")


# Observations:
# * of all the Pclass 1 passengers, more than 60% survived
# * of all the Pclass 2 passengers, just over 47% survived
# * of all the Pclass 3 passengers, only 24% survived
# * This bar plot shows that higher economic status of the passengers will have a higher the survival rate

# ## 4.3 - Sex

# In[ ]:


bar_plot("Sex")


# Observations:
# * of all the females, almost 75% survived
# * of all the males, only about 19% survived
# * This bar plot shows that females are more likely to survive than males 

# ## 4.4 - SibSp

# In[ ]:


bar_plot("SibSp")


# Observations:
# * This feature had some unexpected results
# * passengers no Siblings or Spouses had a lower survival rate (35%) than those with 1(56%) or 2(47%) Siblings/Spouses
# * passengers with 3 or 4 Siblings/Spouses had a lower survival rate than passengers with 0 Siblings/Spouses
# * There were no survivors of passengers with siblings more than 4
# 
# Assumption:
# These observations can have meaning considering passenfers with too many Siblings/Spouses could have been killed in trying to save their large families, and passengers with zero Siblings/Spouses might not have been given preference.

# ## 4.5 - Parch

# In[ ]:


bar_plot("Parch")


# Observations:
# * People with smaller familes (1, 2, 3) had a better survival rate than people with larger familes
# * Contrary to expectations, people with no family members had a lower survival rate than people with a few family members

# ## 4.6 - Embarked

# In[ ]:


#Filling missing value
#Since Embarked only has 2 missing values, I will use the mode from the Series to determine the missing values
df["Embarked"] = df["Embarked"].fillna(df['Embarked'].mode()[0])
bar_plot("Embarked")


# Observations:
# * Passengers that embarked from Cherbourg(C), have the highest survival rate at almost 55%
# * Passengers that embarked from Queenstown(Q), have a 40% survical rate
# * Passengers that embarked from Southampton(S), have the lowest survival rate at just over 30%
# 
# Assumption: 
# * People from Cherbourg may have gotten seat/cabins that are located on the top or closer to the top and they had easier access for escape or these passengers collectively had a higher economic status than the other places
# 
# Let's verify that assumption by correlating the Pclass with the Embarked

# In[ ]:


sns.factorplot("Pclass", col="Embarked",  data=train, size=6, kind="count")


# This validates the assumption about embarked that we made earlier:
# * Queenstown(Q) has almost exclusively Pclass = 3 passengers
# * Southampton(S) has a majority of Pclass = 3 passengers
# * Cherbourg(C) has a majority of Pclass = 1 passengers
# 
# Thus explaining why Cherbourg(C) has a higher survival rate than the other two embarked locations

# ## 4.7 - Cabin

# It's highly like that recorded cabin values would mean a higher economic status and those who don't have a separate cabin belong to lower economic status.
# Let's check the correlation of recorded cabin values with the Pclass to determine if there is any merit in our claim

# In[ ]:


df["Cabin"] = df["Cabin"].notnull().astype(int)

pclass1 = df["Cabin"][df["Pclass"] == 1].value_counts()[1]
print("recorded Cabins with pclass1 = 1: {}".format(pclass1))

sns.barplot(x="Pclass", y="Cabin", data=df)


# As it was assumed that most of the recorded cabins indeed belong to people with the highest economic status (80%) and thus cabin is directly correlated with the Pclass and having both Pclass and Cabin for out features in modeling will be redundant so we will remove the Cabin column when modeling.

# # 5 - Removing missing values

# ## 5.1 - Age

# Inspecting the correlations of various features on age to determine best technique to impute the missing Age data

# In[ ]:


# Exploring Age vs Sex, Parch , Pclass and SibSP
g = sns.factorplot(y="Age",x="Sex",data=df, kind="box")
g = sns.factorplot(y="Age",x="Pclass", data=df, kind="box")
g = sns.factorplot(y="Age",x="Parch", data=df, kind="box")
g = sns.factorplot(y="Age",x="SibSp", data=df, kind="box")


# Observations:
# * Gender has no effect on the Age feature thus it's safe to assume that to impute the Age column we don't have to consider the age
# * In general, The higher the Pclass, the older the people
# * Parch has a mixed correlation with Age so will need to keep this
# * In general, The lesser the SibSp count, the older the age

# In[ ]:


#Using a heatmap to determine the correlation between the remaining features
sns.heatmap(df[["Age","Pclass","SibSp", "Parch"]].corr(), cmap = 'coolwarm', annot = True)


# In[ ]:


#small function that will remove the missing values in the age column
def fill_age_missing_values(df):
    Age_Nan_Indices = list(df[df["Age"].isnull()].index)

    #for loop that iterates over all the missing age indices
    for index in Age_Nan_Indices:
        #temporary variables to hold SibSp, Parch and Pclass values pertaining to the current index
        temp_Pclass = df.iloc[index]["Pclass"]
        temp_SibSp = df.iloc[index]["SibSp"]
        temp_Parch = df.iloc[index]["Parch"]
        age_median = df["Age"][((df["Pclass"] == temp_Pclass) & (df["SibSp"] == temp_SibSp) & (df["Parch"] == temp_Parch))].median()
        if df.iloc[index]["Age"]:
            df["Age"].iloc[index] = age_median
        if np.isnan(age_median):
            df["Age"].iloc[index] = df["Age"].median()
    return df


# In[ ]:


#Using the function to remove missing values in both train and test set
df = fill_age_missing_values(df)
df.describe(include="all")


# In[ ]:


df["Age"].isnull().sum()


# Comments:
# * I decided to use the median technique rather than the mean technique to fill in the missing values to ensure I don't get values in decimal points
# * I used a conditional statement where we check to determine the median age when all three conditions are satified, the conditions being
#     * Same Pclass number
#     * Same number of Siblings/Spouses (SibSp)   
#     * Same number of Parents/Children (Parch)
#     
# As evident from the description of the dataframe above, we have removed all the NaN values in the Age column

# # 6 - Feature Engineering

# # 6.1 - Title feature

# Getting the Title feature from the name

# In[ ]:


df.head()


# The "Title" can be extracted from the "Name" column by splitting the name and the title is between the comma and the period. ex. Braund, "Mr"(Title). Owen Harris

# In[ ]:


#Creating a new column("Title") using list comprehension
df["Title"] = pd.Series([name.split(",")[1].split(".")[0].strip() for name in df["Name"]])
df.head()


# In[ ]:


pd.crosstab(df['Title'], df['Sex'])

Observations:
* Mr is the first category (Mr)
* Miss/Mme/Mlle/Mrs/Ms can be combined into a single category as they resemble the same Title (Miss)
* Jonkhee/Lady/Countess/Sir/Don/Dona can be categorized into Royals (Royals)
* Dr/Major/Col can be categorized into Professional Category (Professionals)
* Master/Rev can be separate category considered both resemble leaders (Mas/Rev)

We have in total 5 separate categories
# In[ ]:


# Convert to categorical values Title 
df["Title"] = df["Title"].replace(['Lady', 'the Countess', 'Countess', 'Don', 'Jonkheer', 'Dona', 'Sir'], 'Royals')
df["Title"] = df["Title"].replace(['Col', 'Dr', 'Major', 'Capt'], 'Professionals')
df["Title"] = df["Title"].replace(["Ms", "Mme", "Mlle", "Mrs"], 'Miss')
df["Title"] = df["Title"].replace(['Master', 'Rev'], 'Mas/Rev')
df["Title"] = df["Title"].map({"Mas/Rev": 0, "Miss": 1, "Mr": 2, "Royals": 3, "Professionals": 4})


# In[ ]:


pd.crosstab(df['Title'], df['Sex'])


# In[ ]:


sns.factorplot(x="Title",y="Survived",data=df, kind="bar").set_xticklabels(["Master","Miss","Mr","Royals","Professionals"]).set_ylabels("survival probability")


# ## 6.2 - Family Total

# Let's convert the Parch and SibSp into a single feature known as Ftotal

# In[ ]:


df["Ftotal"] = 1 + df["SibSp"] + df["Parch"]


# In[ ]:


sns.factorplot(x="Ftotal",y="Survived",data=df, kind="bar").set_ylabels("survival probability")


# Now we can drop the seperate SibSp and Parch columns

# ## 6.3 - Grouping age groups

# Since we have a very big variation in age, we can engineerinf the age feature into age groups

# In[ ]:


df["Age"] = df["Age"].astype(int)
df.loc[(df['Age'] <= 2), 'Age Group'] = 'Baby' 
df.loc[((df["Age"] > 2) & (df['Age'] <= 10)), 'Age Group'] = 'Child' 
df.loc[((df["Age"] > 10) & (df['Age'] <= 19)), 'Age Group'] = 'Young Adult'
df.loc[((df["Age"] > 19) & (df['Age'] <= 60)), 'Age Group'] = 'Adult'
df.loc[(df["Age"] > 60), 'Age Group'] = 'Senior'
df["Age Group"] = df["Age Group"].map({"Baby": 0, "Child": 1, "Young Adult": 2, "Adult": 3, "Senior": 4})


# In[ ]:


df.sample(5)


# In[ ]:


df.describe(include = "all")


# ## 6.4 - Solving fare skewness

# I used help from [here](https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling/) on how to tackle the fare column as fare values were highly skewed towards the low end.

# In[ ]:


sns.distplot(df["Fare"], color="m", label="Skewness : %.2f"%(df["Fare"].skew())).legend(loc="best")


# In[ ]:


# Apply log to Fare to reduce skewness distribution
df["Fare"] = df["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
sns.distplot(df["Fare"], color="g", label="Skewness : %.2f"%(df["Fare"].skew())).legend(loc="best")


# Now the fare is much less skewed as skewness went from 4.51 to 0.56

# ## 6.5 - Mapping categorical sex feature

# In[ ]:


df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df.sample(5)


# ## 6.6 - Mapping categorical embarked feature

# In[ ]:


df["Embarked"] = df["Embarked"].map({"C": 0, "Q": 1, "S": 2})
df.sample(5)


# ## 6.7 - Removing non-essential features

# Some of these features now are reduntant as we have used feature engineering to extract the important details, such features are:
# * Name, after we extracted the Title feature, it is not useful anymore
# * Parch and SibSp feature as we have combined the 2 into a single feature called Ftotal
# * Ticket  and PassnegerID features as they gives nothing significant for the determination
# * Age as we made age groups instead

# In[ ]:


passenger_ID = pd.Series(df["PassengerId"], name = "PassengerId")
df = df.drop(["Name", "PassengerId", "SibSp", "Parch", "Age", "Ticket"], axis=1)
df.sample(5)


# ## 6.8 - Get categorical dummies

# For some features we have categories and these categories require dummies variables

# In[ ]:


df = pd.get_dummies(df, columns = ["Title"])
df = pd.get_dummies(df, columns = ["Embarked"])
df = pd.get_dummies(df, columns = ["Pclass"])
df = pd.get_dummies(df, columns = ["Age Group"])


# # 7 - Building/Training our model

# ## 7.1 - Seperating Train/Test dataset

# In[ ]:


train = df[:train.shape[0]]
test = df[train.shape[0]:].drop(["Survived"], axis = 1)


# ## 7.2 -  Modelling various classifiers

# In[ ]:


#StratifiedKFold aims to ensure each class is (approximately) equally represented across each test fold
k_fold = StratifiedKFold(n_splits=5)

X_train = train.drop(labels="Survived", axis=1)
y_train = train["Survived"]

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

# Creating objects of each classifier
LG_classifier = LogisticRegression(random_state=0)
SVC_classifier = SVC(kernel="rbf", random_state=0)
KNN_classifier = KNeighborsClassifier()
NB_classifier = GaussianNB()
DT_classifier = DecisionTreeClassifier(criterion="entropy", random_state=0)
RF_classifier = RandomForestClassifier(n_estimators=200, criterion="entropy", random_state=0)

#putting the classifiers in a list so I can iterate over there results easily
titanic_classifiers = [LG_classifier, SVC_classifier, KNN_classifier, NB_classifier, DT_classifier, RF_classifier]

#This dictionary is just to grad the name of each classifier
classifier_dict = {
    0: "Logistic Regression",
    1: "Support Vector Classfication",
    2: "K Nearest Neighbor Classification",
    3: "Naive bayes Classifier",
    4: "Decision Trees Classifier",
    5: "Random Forest Classifier",
}

titanic_results = pd.DataFrame({'Model': [],'Mean Accuracy': [], "Standard Deviation": []})

#Iterating over each classifier and getting the result
for i, classifier in enumerate(titanic_classifiers):
    classifier_scores = cross_val_score(classifier, X_train, y_train, cv=k_fold, n_jobs=2, scoring="accuracy")
    titanic_results = titanic_results.append(pd.DataFrame({"Model":[classifier_dict[i]], 
                                                           "Mean Accuracy": [classifier_scores.mean()],
                                                           "Standard Deviation": [classifier_scores.std()]}))


# In[ ]:


print (titanic_results.to_string(index=False))


# Observations:
# * K nearest neighbors and Naive bayes have accuracies below 80% and thus I will not consider them anymore.
# * Even though Logistic regression does a job aswell as the other classifiers, it's accuracy was the lowest among the remaining 5, so I will not use this classifier further
# * Random Forest classifier gives the best results among all classifiers so I will use this classifier as the final classifier for my submission but not before tuning hyper parameters using GridSearchSV

# ## 7.3 - Hyperparameter Tuning

# In[ ]:


from sklearn.model_selection import GridSearchCV

RF_classifier = RandomForestClassifier()


## Search grid for optimal parameters
RF_paramgrid = {"max_depth": [None],
                  "max_features": [1, 3, 10],
                  "min_samples_split": [2, 3, 10],
                  "min_samples_leaf": [1, 3, 10],
                  "bootstrap": [False],
                  "n_estimators" :[100,200,300],
                  "criterion": ["entropy"]}


RF_classifiergrid = GridSearchCV(RF_classifier, param_grid = RF_paramgrid, cv=k_fold, scoring="accuracy", n_jobs= -1, verbose=1)

RF_classifiergrid.fit(X_train,y_train)

RFC_optimum = RF_classifiergrid.best_estimator_

# Best Accuracy Score
RF_classifiergrid.best_score_


# In[ ]:


IDtest = passenger_ID[train.shape[0]:].reset_index(drop = True)


# ## 7.4 - Submitting

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
X_train = train.drop(labels="Survived", axis=1)
y_train = train["Survived"]

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(test)

RFC_optimum.fit(X_train, y_train)

test_predictions = pd.Series(RFC_optimum.predict(X_test).astype(int), name="Survived")
titanic_results = pd.concat([IDtest, test_predictions], axis = 1)
titanic_results.to_csv('submission.csv', index=False)


# If you find this notebook helpful, please upvote and if you have any questions, feel free to ask in the comment section.
# Have a great day :)
