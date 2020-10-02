#!/usr/bin/env python
# coding: utf-8

# **This is my second Kernel on Titanic Survival Challenge. The purpose of this kernel is to try feature engineering and see if it can improve the score. Also, I have been reading different kernels on this problem so, I though it is good to put some of the best practices people are using in their kernels.** 
# 
# ***In Short, learning by doing!!***

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#import libraries

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

#We will use different classifiers and then try Voting Classifier to see if it helps in increasing score.

#Classfiers used in _v1
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

#Ensemble
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

#Model_selection

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

#set data visualization stying

sns.set(style='white', context = 'notebook', palette='deep')


# In[ ]:


#Load data

train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
IDtest = test_data["PassengerId"]


# While doing an online course, that we should remove outliers, if possible, from the data. The method they used was find 1st and 3rd quartiles and get **interquartile ranges** (IQR), $Q3-Q1$ and see if there are data points which are beyond $1.5*IQR$ on both sides. 
# 
# Then if any row which has more than 2 or more outlier columns, can be removed from training data.

# In[ ]:


#Outlier Detection

from collections import Counter

def get_outliers(df,n,features):
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        #1st Quartile
        Q1 = np.percentile(df[col],25)
        #3rd Quartile
        Q3 = np.percentile(df[col],75)
        #Inter-Quartile Range
        IQR = Q3 - Q1
        
        #Outliers Range        
        outliers_boundary = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        list_outlier_cols = df[(df[col] < Q1 - outliers_boundary) | (df[col] > Q3 + outliers_boundary)].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(list_outlier_cols)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    more_than_two_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return more_than_two_outliers   

#Function takes 3 parameters - DataFrame,number of outliers you want to check in an observations, feature columns 
Outliers_to_drop = get_outliers(train_data, 2, ["Age","SibSp","Parch","Fare"])
        


# In[ ]:


train_data.loc[Outliers_to_drop]


# In[ ]:


#Drop outliers

train_data = train_data.drop(Outliers_to_drop, axis = 0)


# Out of 10 outliers, 3 have very high ticket fares as compared to other, and 7 have high value of SibSp

# In this Titanic_v2, I will combine the test and training data and then we will do all the operations on both the data set. This can be done by creating data_processing function and then call over two data sets insdividually as well, which was the idea in Titanic_v1 ( although we didn't create a function then and ran the steps individaully twice. 

# In[ ]:


#Concate datasets
full_dataset = pd.concat(objs=[train_data,test_data], axis = 0).reset_index(drop=True)


# In[ ]:


full_dataset.info()


# In[ ]:


full_dataset.isnull().sum()


# Survived has 418 NaN because the dataset is a combination of test and train data and test data doesn't contain Survived columns.

# ## Feature Engineering

# 
# We will analyse train data set features but whatever operation we will be doing on data, it will be on combined dataset (final_dataset)

# In[ ]:


#In Version V1, we didn't look into SibSp and Parch features, so let's start with them.

train_data[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# Interesting, passengers with less numbers of siblings tends to survive more. For example 0, 1 and 2 sibling passengers have 34%, 46% and 53% survival rate. 
# 
# This can be included in our new features.

# In[ ]:


#Let's do the similar analysis for Parch

train_data[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# Again, small families have more chnaces to survive 3 being highest.

# In[ ]:


#As we can see large family have less survival rate, I am going to make a new feature which we can call family size.
#Family size = SibSp + Parch + Individual

full_dataset["FamilySize"] = full_dataset["SibSp"] + full_dataset["Parch"] + 1


# In[ ]:


full_dataset.head()


# In[ ]:


#Also I am going to drop Cabin Variable as Cabin has more than 70% null

full_dataset.drop(["Cabin"],axis =1, inplace=True)


# In[ ]:


full_dataset.head()


# Last time we imputed age with median age of the population, but this time we are going to do something extra. I will find the correlation of age with different features and see if age can be imputed on the sub-population level.
# 
# 

# # Fill missing Values

# In[ ]:


#Embarked - Since only two values are missing and that is in training data, we can fill it with highest occuring value as we did in V1.
full_dataset["Embarked"].fillna('S', inplace=True)


# In[ ]:


#Fare can be filled with median value as well.

fare_median = full_dataset["Fare"].median()
fare_median

full_dataset["Fare"].fillna(fare_median, inplace=True)


# In[ ]:


#Let's do one hot encoding like V1.

from sklearn.preprocessing import LabelEncoder

le_pClass = LabelEncoder()
le_sex = LabelEncoder()
le_embarked = LabelEncoder()
full_dataset['PClass_encoded'] = le_pClass.fit_transform(full_dataset.Pclass)
full_dataset['Sex_encoded'] = le_sex.fit_transform(full_dataset.Sex)
full_dataset['Embarked_encoded'] = le_embarked.fit_transform(full_dataset.Embarked)


# In[ ]:


full_dataset.head()


# In[ ]:


#One hot encoding for categorical columns (PClass, Sex, Embarked)

from sklearn.preprocessing import OneHotEncoder

pClass_ohe = OneHotEncoder()
sex_ohe = OneHotEncoder()
embarked_ohe = OneHotEncoder()

Xp =pClass_ohe.fit_transform(full_dataset.PClass_encoded.values.reshape(-1,1)).toarray()
Xs =sex_ohe.fit_transform(full_dataset.Sex_encoded.values.reshape(-1,1)).toarray()
Xe =embarked_ohe.fit_transform(full_dataset.Embarked_encoded.values.reshape(-1,1)).toarray()


# In[ ]:


#Add back to original dataframe

train_dataOneHot = pd.DataFrame(Xp, columns = ["PClass_"+str(int(i)) for i in range(Xp.shape[1])])
full_dataset = pd.concat([full_dataset, train_dataOneHot], axis=1)

train_dataOneHot = pd.DataFrame(Xs, columns = ["Sex_"+str(int(i)) for i in range(Xs.shape[1])])
full_dataset = pd.concat([full_dataset, train_dataOneHot], axis=1)

train_dataOneHot = pd.DataFrame(Xe, columns = ["Embarked_"+str(int(i)) for i in range(Xe.shape[1])])
full_dataset = pd.concat([full_dataset, train_dataOneHot], axis=1)


# In[ ]:


full_dataset.head()


# Last time we didn't do anything with two features, Name and Ticket. Also, we imputed Age with median. let's see if we can do something more this time.

# In[ ]:


#First, let us take age.
#Let us see how other features are correlated with age and if we can impute age as per other features.

g = sns.catplot(y="Age",x="Sex",data=full_dataset,kind="box")
g = sns.catplot(y="Age",x="Sex",hue="Pclass", data=full_dataset,kind="box")
g = sns.catplot(y="Age",x="Parch", data=full_dataset,kind="box")
g = sns.catplot(y="Age",x="SibSp", data=full_dataset,kind="box")


# In[ ]:


#Convert Sex feature into 0 and 1 and then check correlation matrix.

full_dataset["Sex"] = full_dataset["Sex"].map({"male": 0, "female":1})

g = sns.heatmap(full_dataset[["Age","Sex","SibSp","Parch","Pclass"]].corr(),cmap="BrBG",annot=True)


# Factorplot and correlation matrix tells us that while age is not related to Sex of the passenger but it is negatively correlated to SibSp, Parch and PClass, so we can impute the age of the passenger, where it is not present , with the median of age of similar rows of PClass, SibSp, and Parch.

# In[ ]:


#Get indexes of rows with NaN as age.
#We are getting indexes of all the columns and then getting back all the indexes where Age is null

index_NaN_age = list(full_dataset["Age"][full_dataset["Age"].isnull()].index)


for i in index_NaN_age:
    age_med = full_dataset["Age"].median()
    age_pred = full_dataset["Age"][((full_dataset['SibSp'] == full_dataset.iloc[i]["SibSp"]) 
                                    & (full_dataset['Parch'] == full_dataset.iloc[i]["Parch"]) 
                                    & (full_dataset['Pclass'] == full_dataset.iloc[i]["Pclass"]))].median()
    if not np.isnan(age_pred) :
        full_dataset['Age'].iloc[i] = age_pred
    else :
        full_dataset['Age'].iloc[i] = age_med


# We have no Nan in our dataset now, but we have not done feature engineering on Ticket and Name feature yet. Can we do something there?

# In[ ]:


#Name, Get title from the name.

full_dataset_title = [i.split(",")[1].split(".")[0].strip() for i in full_dataset["Name"]]
full_dataset["Title"] = pd.Series(full_dataset_title)
full_dataset["Title"].head()


# In[ ]:


#Histogram for Titles

g = sns.countplot(x="Title", data = full_dataset)
g = plt.setp(g.get_xticklabels(), rotation = 45)


# There are mainly 4 titles and all other titles are very rare. We will make 4 titles, Mr, Mrs/Miss, Master, Others

# In[ ]:



#Replace with Rare
full_dataset["Title"] = full_dataset["Title"].replace(['Lady', 'the Countess','Countess','Capt', 
                                             'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
#Replace feminine titles with Ms.
full_dataset["Title"] = full_dataset["Title"].replace(['Miss', 'Ms','Mme','Mlle', 'Mrs'], 'Ms')

#Map titles
full_dataset["Title"] = full_dataset["Title"].map({"Master":0, "Ms":1 ,"Mr":2, "Rare":3})
full_dataset["Title"] = full_dataset["Title"].astype(int)


# In[ ]:


#Histogram Again
g = sns.countplot(x="Title", data = full_dataset)
g = g.set_xticklabels(["Master","Ms","Mr","Rare"])


# In[ ]:


#Let us see if survival chnaces depends on titles.

g = sns.catplot(x="Title", y= "Survived", data = full_dataset, kind ='bar')
g = g.set_xticklabels(["Master","Ms","Mr","Rare"])


# Clearly women and childrens have higher rate of survivals.

# In[ ]:


#We can now remove Name column

full_dataset.drop(["Name"], axis=1, inplace=True)


# I am going to work on Ticket, Cabin and FamilySize in the next iteration.

# In[ ]:


full_dataset.head()


# # Modeling

# In[ ]:


#Drop extra columns

full_dataset.drop(["PassengerId","Embarked","Pclass","Sex", "Ticket","Parch", "SibSp", 
                   "PClass_encoded","Sex_encoded","Embarked_encoded"]
                , axis =1, inplace=True)

full_dataset.head()


# In[ ]:


train_len = len(train_data)
train_len


# In[ ]:


#Let's divide data into Train and test now.

train_data = full_dataset[:train_len]
test_data = full_dataset[train_len:]

#drop survived column from test_data

test_data.drop(["Survived"], axis =1, inplace=True)

print(train_data.shape)
print(test_data.shape)
      


# In[ ]:


#Separating features and target variable from training data

train_data["Survived"] = train_data["Survived"].astype(int)
train_data["Fare"] = train_data["Fare"].astype(float)


y_train = train_data["Survived"]
X_train = train_data.drop(labels=["Survived"], axis =1)


# In this version 2 I will be using:
# 
# * SVC
# * RandomForest
# * AdaBoost
# * Decision Tree
# * Extra Trees
# * Gradient Boosting
# * KNN
# * Logistic Regression
# 
# 
# 

# In[ ]:


#10 fold cross validation

kfold = StratifiedKFold(n_splits=10)


# In[ ]:


#Modeling Steps.

random_state = 42
classifiers = []
classifiers.append(SVC(random_state = random_state))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),
                                      random_state=random_state,learning_rate =0.1))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state=random_state))



cv_results = []

for classifier in classifiers:
    cv_results.append(cross_val_score(classifier, X_train, y_train, scoring="accuracy", cv = kfold, n_jobs=-1))
    
cv_means = []
cv_std = []

for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())
    
cv_res = pd.DataFrame({"CrossValMeans": cv_means, "CrossValErrors": cv_std, 
                       "Algorithm": ["SVC", "RandomForestClassifier", "AdaBoostClassifier",
                                     "Decision Tree Classifier", "Extra Trees" ,"Gradient Boosting", 
                                     "K Nearest Neighbors", "Logistic regression"]})

cv_res = cv_res.sort_values(by = "CrossValMeans", ascending=False)
cv_res


# In[ ]:


g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")


# In[ ]:


#Predictions on test data
gbc_clf = GradientBoostingClassifier(random_state=random_state)
gbc_clf.fit(X_train, y_train)


# In[ ]:


Y_pred = gbc_clf.predict(test_data)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": IDtest,
        "Survived": Y_pred
    })

submission.to_csv('Titanic_Prediction_v3.csv', index=False)


# # To do in next Version

# *  More feature engineering
# *  Hyper parameter Tuning
# *  Ensemble Modeling
# *  Learning Curve Graphs

# In[ ]:




