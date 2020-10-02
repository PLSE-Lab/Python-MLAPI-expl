#!/usr/bin/env python
# coding: utf-8

# # Titanic Survival Classification - Nazih Kalo

# ### Load Packages

# In[ ]:


#Visualization and display option packages
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt 
from datetime import timedelta, date
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
sns.set(font_scale=2)

#Math
import numpy as np

#Table manipulation
import pandas as pd 

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

#Model Selection and parameter tuning
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import sklearn.model_selection as cv

# import the metrics class
from sklearn import metrics
from sklearn.metrics import classification_report

#Supress deprecated/future warnings 
import warnings
def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Read the data 

# In[ ]:


#Import data
train = pd.read_csv('/kaggle/input/train.csv')
test = pd.read_csv('/kaggle/input/test.csv')


# In[ ]:


train.head()


# ### Variable Definitions:
# ![image.png](attachment:image.png)

# In[ ]:


#Basic info on train set
train.info()


# In[ ]:


#Summary stats for numerical features
train.describe()


# In[ ]:


#Summary stats for categorical features
train.describe(include=['O'])


# ### Exploratory Data Analysis

# #### Overall Survival (train set)

# In[ ]:


g = sns.countplot(x='Survived', data=train)


print('Mean Survival Rate: ', str(round(train.Survived.mean()*100,2))+'%')


# It looks like the survival rate in the train set was quite low, overall 38.38%.

# #### Survival by Sex

# In[ ]:


g = sns.catplot(x = 'Survived', col = 'Sex', kind = 'count', data = train)
g.set_xticklabels(["No", "Yes"])
g.set_titles("{col_name}", fontsize = 50)


#Mean survival rate by sex
print('Mean Survival Rate (%): \n', train.groupby('Sex')['Survived'].mean()*100)


# The survival rate was significantly higher among females with 78% vs. 18% for males. This suggests that sex will be an important feature for our model.

# #### Survival by Pclass

# In[ ]:


sns.catplot(x = 'Survived', col = 'Pclass', kind = 'count', data = train)

g.set_xticklabels(["No", "Yes"])
g.set_titles("{col_name}", fontsize = 50)


#Mean survival rate by sex
print('Mean Survival Rate (%): \n', train.groupby('Pclass')['Survived'].mean()*100)


# Similarly there are significant differences in the survival rate between ticket classes, with higher classes being more likely to survive. 

# ### Survival Age distribution

# In[ ]:


g = sns.FacetGrid(train, col='Survived', height = 10)
g.map(plt.hist, 'Age', bins=20)
g.set_titles("Survived = {col_name}", fontsize = 50)


# Looking at the distribution of ages between the survived/dead group it looks like the survived distribution has a stronger left skew, suggesting that young people represent a large proportion of those who survived than those who didn't.

# # Finding interaction terms

# Looking at survival rates by class and embarked/port it looks like there may some interaction between these two variables. It is clear that Southhampton had a lower survival rate on average than the other locations. 

# In[ ]:


g = sns.catplot(x="Pclass", y ='Survived', hue="Embarked", kind="point", edgecolor=".6",
            data=train.groupby(['Pclass', 'Embarked']).mean().reset_index(), height = 7)

g.set_titles("Survived", fontsice = 50)


# ## Plotting pairplot to look for other interesting relationships in our features.

# In[ ]:


#Create copy to change sex column to a boolean 'female': 1, 'male': 0
train_copy = train.copy()
train_copy['Sex'] = train_copy['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

#Plot pairplot of numerical components 
sns.pairplot(train_copy.iloc[:, [1,2,4,5,6,7,9]], hue = 'Survived', markers=["o", "s"])


# This pairplot contains a lot of information about our features. 
# 
# First it confirms the idea that the age of survivors was lower overall (in the Age dist plot). 
# 
# Second it looks like the fare among survivors has a stronger right skew (in the fare hist), this suggests that those who paid higher fares (and likely were in the higher classes) were more likely to survive. This makes sense given our analysis of pclass.
# 
# Thirdly, it looks like younger passengers tended to have higher more sibling and/or spouses on the ship with them.
# 
# Lastly, there seems to be a small relationship between the SibSp and survival, with larger sibsp numbers resulting in higher survival odds. 

# # Feature Engineering 
# 
# Based on the analysis above we can now move on to feature engineering to keep/build features that seem to be useful.

# In[ ]:


#Combine train and test for feature engineering 
train_test_df = [train, test]


# ### Create title column from Name 

# In[ ]:


for dataset in train_test_df:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

train['Title'].value_counts()


# Wow, there are many titles on this ship. Let us reduce these to five groups (Miss, Mrs, Master, Mr & Rare). 

# In[ ]:


#Reduce the number of groups to 5 groups
for dataset in train_test_df:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean().sort_values(by = 'Survived',ascending = False)


# There seems to be quite significant survival rates just based on the passengers title alone!
# 
# These values seem to confirm some of our previous observations, with women having higher survival than men & wealthier sounding titles also surviving better than regular sounding ones. 

# ### Mapping titles categories to numeric
# #### Title mapping Mr =  0, Miss =  1, Mrs = 2, Master =  3, Rare = 4

# In[ ]:


#Change to ordinal 
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4}
for dataset in train_test_df:
    dataset['Title'] = dataset['Title'].map(title_mapping)

train.head()


# In[ ]:


#Drop the Name column from both datasets
for dataset in train_test_df:
    dataset.drop('Name', axis = 1, inplace = True)

train_test_df = [train, test]
train.head()


# ### Converting Sex column to dummy for female
# #### Sex to boolean Female = 1 and Male = 0

# In[ ]:


for dataset in train_test_df:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_test_df = [train, test]

train.head()


# ### Visualizing our missing data

# In[ ]:


#TRAIN MISSING VALUES
total_num = train.isnull().sum().sort_values(ascending=False)
perc = train.isnull().sum()/train.isnull().count() *100
perc1 = (round(perc,2).sort_values(ascending=False))

# Creating a data frame:
train_null = pd.concat([total_num, perc1], axis =1 , keys =["Total Missing Values", "Percentage %"]).sort_values(by ="Percentage %", ascending = False)

#Top nulls
top_null = train_null[train_null["Percentage %"]>0]
top_null.reset_index(inplace=True)
top_null


# In[ ]:


#Visualising missing data 

sns.set(font_scale = 1)
fig, axes = plt.subplots(2,1, figsize = (15,12))
sns.heatmap(train.isnull(), cbar = False, yticklabels=False, cmap="magma", ax=axes[0])
sns.heatmap(test.isnull(), cbar = False, yticklabels=False, cmap="magma", ax=axes[1])

#Reset to 2
sns.set(font_scale = 2)


# It looks like the main missing values in our datasets are age and cabin. With a couple of values for embarked and fare. 
# 
# Let us take care of those now. 

# ### Fill Age NA using median from Title
# 

# In[ ]:


# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')
g = sns.FacetGrid(train, row='Title', height=4, aspect=1.6)
g.map(plt.hist, 'Age', alpha=.5, bins=20)
g.add_legend()
g.set_ylabels('Count')


# #### Let us start by preparing an empty array to contain guessed Age values based on Pclass x Gender combinations.
# 
# 

# In[ ]:


train.groupby(['Title'])['Age'].median()


# In[ ]:


for dataset in train_test_df:
    dataset['Age'] = dataset.groupby(['Title'])['Age'].transform(lambda x: x.fillna(x.median()))


# In[ ]:


#Let us create Age bands and determine correlations with Survived.
train['AgeBand'] = pd.cut(train['Age'], 5)
train.groupby(['AgeBand'])['Survived'].mean().sort_values(ascending=True).to_frame().style.background_gradient(cmap='summer_r')


# In[ ]:


#Let us replace Age with ordinals based on these bands.
for dataset in train_test_df:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
    dataset['Age'] = dataset['Age'].astype(int)
    
    train.head()


# In[ ]:


#Drop ageband column from train dataset
train = train.drop('AgeBand', axis=1)

train_test_df = [train, test]


# ### Create interaction term between Age and Pclass 
# 
# ##### Rationale for this is that the effect of Age varies with Pclass.  For example, older people in first class are more likely to survive than old people in lower classes.

# In[ ]:


g = sns.catplot(x = 'Pclass',  y ='Survived', hue="Age", kind="point", edgecolor=".6",
            data=train.groupby(['Pclass', 'Age']).mean().reset_index(), height = 7)

g.set_titles("Survived", fontsice = 50)


# In[ ]:


for dataset in train_test_df:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass
    dataset['Age'] = dataset['Age'].astype(int)
    dataset['Age*Class'] = dataset['Age*Class'].astype(int)
    
train.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)


# # Create is_child indicator

# In[ ]:


train_test_df = [train, test]

for dataset in train_test_df:
    dataset['is_child'] = dataset['Age'].apply(lambda x: 1 if x == 0 else 0)

train_test_df = [train, test]

train.head()


# In[ ]:


train.groupby(['is_child'])['Survived'].mean().sort_values(ascending=False).to_frame().style.background_gradient(cmap='summer_r')


# ### Fill NAs of Embarked with mode (most frequent)

# In[ ]:


mode_embarked = train.Embarked.dropna().mode()[0]
mode_embarked 


# In[ ]:


for dataset in train_test_df:
    dataset['Embarked'] = dataset['Embarked'].fillna(mode_embarked)
    
train.groupby(['Embarked'])['Survived'].mean().sort_values(ascending=False).to_frame().style.background_gradient(cmap='summer_r')


# ### Fill NAs of Fare in test with the median
# 
# There is only one value missing, we can resort to just filling this fare value with the median of its Pclass.

# In[ ]:


#The null entry in 
test[test.Fare.isnull()] #Pclass = 3


# In[ ]:


test['Fare'] = test['Fare'].fillna(test.loc[test.Pclass == 3 ,'Fare'].dropna().median())


# ### Create FareBand to segment different fare values and create ordinal categorical variable

# In[ ]:


train['Fare_Band']=pd.qcut(train['Fare'],4)
train.groupby(['Fare_Band'])['Survived'].mean().to_frame().style.background_gradient(cmap='summer_r')


# In[ ]:


#Convert the Fare feature to ordinal values based on the FareBand.

for dataset in train_test_df:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train = train.drop('Fare_Band', axis=1)
train_test_df = [train, test]
    
train.head(10)


# ### Create FamilySize to replace sibsp and parch
# 
# Since there wasnt significant relationship between each of these features with survival, let us combine them to try and get a more meaningful metric.

# In[ ]:


#Create new feature combining existing features

#We can create a new feature for FamilySize which combines Parch and SibSp. 
#This will enable us to drop Parch and SibSp from our datasets.

for dataset in train_test_df:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch']

train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


#We can group FamilySize into four groups 

for dataset in train_test_df:
    dataset.loc[(dataset['FamilySize'] >= 1) & (dataset['FamilySize'] <= 2), 'FamilySize'] = 1
    dataset.loc[dataset['FamilySize'] == 3, 'FamilySize'] = 2
    dataset.loc[dataset['FamilySize'] >= 4, 'FamilySize'] = 3
    
train.groupby(['FamilySize'])['Survived'].mean().sort_values(ascending=False).to_frame().style.background_gradient(cmap='summer_r')


# In[ ]:


g = sns.catplot(x = 'FamilySize', y="Survived", kind="bar", edgecolor=".6",
            data=train.groupby(['FamilySize']).mean().reset_index(), height = 7)


# In[ ]:


#Let us drop Parch, SibSp features in favor of Family Size

for dataset in train_test_df:
    dataset.drop(['Parch', 'SibSp'], axis=1, inplace = True)

train_test_df = [train, test]

train.head()


# # Create Alone dummy variable

# In[ ]:


train_test_df = [train, test]
#We can create another feature called IsAlone.
for dataset in train_test_df:
    dataset['Alone'] = 0
    dataset.loc[dataset['FamilySize'] == 0, 'Alone'] = 1

train[['Alone', 'Survived']].groupby(['Alone'], as_index=False).mean()


# # Create has cabin indicator
# 
# Although we see that 77.10% of cabin rows are NaN in our training set, this could be indicative of the fact that these passengers were not in a superiod class to have their cabin recorded. Let us create a dummy variable to track whether the passenger had a cabin recorded or not. 

# In[ ]:


for dataset in train_test_df:
    dataset['Has_Cabin'] = ~dataset.Cabin.isnull()
    dataset['Has_Cabin'] = dataset['Has_Cabin'].astype(int)


# In[ ]:


#Let us drop Cabin  in favor of Has_Cabin
for dataset in train_test_df:
    dataset.drop('Cabin', axis=1, inplace = True)

train_test_df = [train, test]
train.head()
train.head()


# In[ ]:


print(train[['Has_Cabin', 'Survived']].groupby(['Has_Cabin'], as_index=False).mean().sort_values(by='Survived', ascending=False))

display(g = sns.catplot(x = 'Has_Cabin', y="Survived", kind="bar", edgecolor=".6",
            data=train.groupby(['Has_Cabin']).mean().reset_index(), height = 7))


# It looks like all three classes had some cabins registered but overall those with recorded cabins had higher overall survival rates across all classes.

# In[ ]:


g = sns.catplot(x = 'Has_Cabin', y="Survived",hue = 'Pclass', kind="bar", edgecolor=".6",
            data=train, height = 7)


# ## Convert Embarked feature to numeric

# In[ ]:


for dataset in train_test_df:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train.head()


# ## Add embarked * Pclass interaction term
# 
# As our EDA showed earlier, there seems to be an interaction between the embarked location and pclass. Let us highlight this to our models by creating an interaction term between the two.

# In[ ]:


train_test_df = [train, test]

for dataset in train_test_df:
    dataset['Embarked*Pclass'] = dataset['Embarked']*dataset['Pclass']

train.head()


# ## Dropping Columns

# #### Dropping the ticket column from both datasets

# In[ ]:


#looking at the percentage of unique values for ticket in both datasets

print('Percent of unique ticket values in train: ',round(train.Ticket.nunique()/len(train)*100,2), '%')
print('Percent of unique ticket values in test: ', round(test.Ticket.nunique()/len(test)*100,2),'%')


# In[ ]:


for dataset in train_test_df:
    dataset.drop('Ticket', axis = 1, inplace = True)

train_test_df = [train, test]

train.head()


# #### Dropping the passengerID column from the train data (will keep it for test to show results)

# In[ ]:


train = train.drop('PassengerId', axis = 1)


# # Final Mapping Dictionary
# 
# #### Sex:
# - Female = 1
# - Male = 0
# 
# #### Fare:
# - Fare <=7.91 : 0 
# - 7.91 < Fare <= 14.454 : 1
# - 14.454 < Fare <= 31 : 2
# - 31 < Fare : 3
# 
# #### Age:  
# - Age <= 16 : 0 
# - 16 < Age <= 32 : 1
# - 32 < Age <= 48 : 2
# - 48 < Age <= 64 : 3
# - 64 < Age <= 64 : 4
# 
# #### Embarked: 
# - S =  0
# - C = 1
# - Q = 2
# 
# #### Title:  
# - Mr =  0
# - Miss =  1 
# - Mrs = 2
# - Master =  3
# - Rare = 4
# 
# #### is_child	
# - Age <=16 : 1
# - Age > 16 : 0
# 
# #### FamilySize
# - sum of SibSp + Parch
# 
# #### Alone 
# - = 1 if FamilySize = 0, 0 otherwise
# 
# #### Has_Cabin 
# - 1 for passengers with a cabin value and 0 otherwise

# # Model Building 

# In[ ]:


# extract target variable (interest rate) from training data
target = train["Survived"]

# remove interest rate column from training data
predictors = train.drop("Survived", axis=1)


# In[ ]:


predictors.head()


# ## Train-Test Split of training set to perform CV and measure accuracy.

# In[ ]:


#SET SEED 
SEED = 1 # We will be using this seed for all models that include a random_state parameter.

# Split dataset into 80% train and 20% test
X_train, X_test, Y_train, Y_test = train_test_split(predictors, target,
test_size=0.2,
random_state=SEED)


# # Logistic Regression
# 
# After applying some hyperparameter tuning to the Inverse of regularization strength parameter (C) the best value that balanced Test/train accuracy was 4.

# In[ ]:


#TRAIN Accuracy score
logreg = LogisticRegression(C=4, solver= 'lbfgs', random_state=SEED)
logreg.fit(X_train, Y_train)
Y_pred_train = logreg.predict(X_train)
acc_log_train = round(logreg.score(X_train, Y_train) * 100, 2)


print(metrics.classification_report(Y_train, Y_pred_train))
print("Accuracy:",acc_log_train,'%') #Total accuracy 


# In[ ]:


#TEST Accuracy score
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_test, Y_test) * 100, 2)


print(metrics.classification_report(Y_test, Y_pred))
print("Accuracy:",acc_log,'%') #Total accuracy 


# In[ ]:


cnf_matrix = metrics.confusion_matrix(Y_test, Y_pred)

# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[ ]:


#Logistic coefficients 
coeff_df = pd.DataFrame(train.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)


# ### Interesting insights from the coefficients of the logistic regression
# 
# 1. Sex is highest positivie coefficient, implying as the Sex value increases (male: 0 to female: 1), the probability of Survived=1 increases the most.
# <br>
#      
# 2. Title has a large positive coefficient. Looking at the order of our variables this also makes sense as the larger indicators values for Miss(1)/Mrs(2)/Master(3)/Rare(4) should have a better chance of survivival than Mr(0). Since the first four are either indicative of being women/wealthy and the last one is indicative of being an older man.
# <br>
# 
# 3. As Pclass increases, probability of Survived=1 decreases. Which makes sense since we saw lower class passengers had lower survival rates. 
# <br>
# 
# 4. Alone seems to be a good feature to have included in the model as it has second highest negative correlation with Survived and suggests that people traveling alone were more likely to die. 
# <br>
# 
# 5. is_child is also a good feature to have added as it is very positively correlated with survival as we saw from the EDA
# <br>
# 
# 6. Interestingly, Has_Cabin also provided a good predictor of survival. This agrees with the analysis we did previously that showed that those with non-null cabin had significantly higher survival rates.

# # Support Vector Machines
# 
# Similarly for SVM, after applying some hyperparameter tuning to the optimal gamma and C values were used in our models. The code for parameter optimazation was included below. 

# In[ ]:


#TRAIN Accuracy score
svc = SVC(gamma=0.01,C=100, random_state=SEED)
svc.fit(X_train, Y_train)
Y_pred_train = svc.predict(X_train)
acc_svc_train = round(svc.score(X_train, Y_train) * 100, 2)

print(metrics.classification_report(Y_train, Y_pred_train))
print("Accuracy:",acc_svc_train,'%') #Total accuracy 


# In[ ]:


#TEST Accuracy score
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_test, Y_test) * 100, 2)

print(metrics.classification_report(Y_test, Y_pred))
print("Accuracy:",acc_svc,'%') #Total accuracy 


# In[ ]:


# Set the parameters by cross-validation
#tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
#                     'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
#                    {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
#                     'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
#                    {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}
#                   ]

#scores = ['precision', 'recall']

#for score in scores:
#    print("# Tuning hyper-parameters for %s" % score)
#    print()

#    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
#                       scoring='%s_macro' % score)
#    clf.fit(X_train, Y_train)
#
#    print("Best parameters set found on development set:")
#    print()
#    print(clf.best_params_)
#    print()
#    print("Grid scores on development set:")
#    print()
#    means = clf.cv_results_['mean_test_score']
#    stds = clf.cv_results_['std_test_score']
#    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#        print("%0.3f (+/-%0.03f) for %r"
#              % (mean, std * 2, params))
#    print()


# In[ ]:


cnf_matrix = metrics.confusion_matrix(Y_test, Y_pred)

# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# # K-Nearest Neighbors 

# In[ ]:


#TRAIN Accuracy score
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, Y_train)
Y_pred_train = knn.predict(X_train)
acc_knn_train = round(knn.score(X_train, Y_train) * 100, 2)

print(metrics.classification_report(Y_train, Y_pred_train))
print("Accuracy:",acc_knn_train,'%') #Total accuracy 


# In[ ]:


#TEST Accuracy score
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_test, Y_test) * 100, 2)

print(metrics.classification_report(Y_test, Y_pred))
print("Accuracy:",acc_knn,'%') #Total accuracy 


# In[ ]:


cnf_matrix = metrics.confusion_matrix(Y_test, Y_pred)

# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# # Decision Tree

# In[ ]:


#TRAIN Accuracy score
decision_tree = DecisionTreeClassifier(max_depth=6, ## Model complexity
                           min_samples_leaf=20,
                           random_state=SEED, criterion='entropy')
decision_tree.fit(X_train, Y_train)
Y_pred_train = decision_tree.predict(X_train)
acc_decision_tree_train = round(decision_tree.score(X_train, Y_train) * 100, 2)

print(metrics.classification_report(Y_train, Y_pred_train))
print("Accuracy:",acc_decision_tree_train,'%') #Total accuracy 


# In[ ]:


#TEST Accuracy score
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_test, Y_test) * 100, 2)

print(metrics.classification_report(Y_test, Y_pred))
print("Accuracy:",acc_decision_tree,'%') #Total accuracy 


# In[ ]:


cnf_matrix = metrics.confusion_matrix(Y_test, Y_pred)

# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# ## Let us visualize our decision tree to get an idea of how our classification is being made

# In[ ]:


from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
from sklearn.datasets import load_wine
from IPython.display import SVG
from graphviz import Source
from IPython.display import display

graph = Source(tree.export_graphviz(decision_tree, out_file=None
   , feature_names=X_train.columns, class_names=['0', '1'] 
   , filled = True))
display(SVG(graph.pipe(format='svg')))


# **Based on this decision tree it is clear that title seems to be playing the most important role here, given it is the root node. Other important variables include 'Has_Cabin', 'Pclass' & 'Sex'.**

# # Random Forest

# In[ ]:


#TRAIN Accuracy score
random_forest = RandomForestClassifier(n_estimators=100, 
                                       min_samples_leaf=25,
                                       min_samples_split=0.1,
                                       random_state=SEED, 
                                       criterion='gini')
random_forest.fit(X_train, Y_train)
Y_pred_train = random_forest.predict(X_train)
acc_random_forest_train = round(random_forest.score(X_train, Y_train) * 100, 2)

print(metrics.classification_report(Y_train, Y_pred_train))
print("Accuracy:",acc_random_forest_train,'%') #Total accuracy 


# In[ ]:


#TEST Accuracy score
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
acc_random_forest = round(random_forest.score(X_test, Y_test) * 100, 2)

print(metrics.classification_report(Y_test, Y_pred))
print("Accuracy:",acc_random_forest,'%') #Total accuracy 


# In[ ]:


cnf_matrix = metrics.confusion_matrix(Y_test, Y_pred)

# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
# Create a pd.Series of features importances
importances_rf = pd.Series(random_forest.feature_importances_,
index = X_train.columns)
# Sort importances_rf
sorted_importances_rf = importances_rf.sort_values()
# Make a horizontal bar plot
sorted_importances_rf.plot(kind='barh', color='blue')
plt.show()


# ### Now that we have obtained our own accuracy metrics using train-test split, let us check what scores we get from CV. In theory, these scores should give us a more representative and unbiased measure of our accuracy since we are using K number of train-test splits and averaging their results. 

# # Cross validation Scores
# 
# We will be using the **balanced accuracy** scoring metric for our cross validation. The reasoning for this is that balanced accuracy provides a better measure when **there is a large class imbalance**, which is the case here. There are a lot more people that didn't survive than those who did. A model can predict the value of the majority class for all predictions and achieve a high classification accuracy, the problem is that this model is not useful in the problem domain, so we will use balanced accuracy.

# In[ ]:


from sklearn.model_selection import cross_val_score

logref_score = cross_val_score(logreg, X_train, Y_train, cv = 5, scoring='balanced_accuracy')

logref_score_mean = round(logref_score.mean() * 100, 2)

print('Logreg Accuracy : {:.2f}% (+/- {:.3f}%)'.format(logref_score_mean, logref_score.std()*100*1.96))


# In[ ]:


KNN_score = cross_val_score(knn, X_train, Y_train, cv = 5, scoring='balanced_accuracy')
KNN_score_mean = round(KNN_score.mean()*100, 2)

print('KNN Accuracy : {:.2f}% (+/- {:.3f}%)'.format(KNN_score_mean, KNN_score.std()*100*1.96))


# In[ ]:


RF_score = cross_val_score(random_forest, X_train, Y_train, cv = 5, scoring='balanced_accuracy')
RF_score_mean = round(RF_score.mean()*100, 2)

print('RF Accuracy : {:.2f}% (+/- {:.3f}%)'.format(RF_score_mean, RF_score.std()*100*1.96))


# In[ ]:


svc_score = cross_val_score(svc, X_train, Y_train, cv = 5, scoring='balanced_accuracy')
svc_score_mean = round(svc_score.mean()*100, 2)

print('SVM Accuracy : {:.2f}% (+/- {:.3f}%)'.format(svc_score_mean, svc_score.std()*100*1.96))


# In[ ]:


DT_score = cross_val_score(decision_tree, X_train, Y_train, cv = 5, scoring='balanced_accuracy')
DT_score_mean = round(DT_score.mean()*100, 2)

print('DT Accuracy : {:.2f}% (+/- {:.3f}%)'.format(DT_score_mean, DT_score.std()*100*1.96))


# # Evaluating all models accuracy

# In[ ]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 
              'Decision Tree'],
    'Train Score': [acc_svc_train, acc_knn_train, acc_log_train, 
              acc_random_forest_train, acc_decision_tree_train],
    'Test Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_decision_tree],
    'CV Score': [svc_score_mean, KNN_score_mean, logref_score_mean,
                RF_score_mean, DT_score_mean]})

models.sort_values(by='CV Score', ascending=False).style.background_gradient(cmap='RdYlGn')


# # Step 3: Apply Best Model to Holdout_Test set
# 
# Test your models using the data found within the "Holdout_testing" file. Save the results of the final model (remember you will only predict the Survived column in holdout test set with your best model results) in a single, separate CSV titled "Titanic Results from" *insert your name or UChicago net ID.

# In[ ]:


holdout_test = test.drop(['PassengerId'], axis =1)

holdout_test.head()


# In[ ]:


# Support Vector Machines
svc = SVC(gamma=0.01,C=100, kernel='rbf')
svc.fit(predictors, target)
Y_pred = svc.predict(holdout_test)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": Y_pred
    })


# In[ ]:


#Save result to CSV
#submission.to_csv('submission.csv', index=False)


# # Final Write Up 
# 
# Given this was a classification problem I chose to use the main supervised algorithms I am currently familiar with. The results from the five models used to classify the titanic survivors was quite surprising. Firstly, I expected that the logisitic regression would not perform as well as it did because it assumes a linear relationship between the independent and dependent variables, homoskedastic errors  and no collinearity between independent variables. Many of these assumptions were violated, yet the model still outperformed more complex non-linear models. 
# 
# Given the non-linear aspects of the problem, I expected decision trees/random forest to show superior performance. This is because DT/RF require no significant preprocessing, no assumptions on distribution and automatically take care of colinearity. However they seemed to underperform. This may be due to having overly complex models which caused them to overfit. Nevertheless, I was unable to stablize the train/test/CV accuracy measures any further given my time constraint. 
# 
# SVM performed the best in my case, this was not suprising in retrospect as it is a non-linear model so it can handle many different features/high dimensional data. Furthermore, the kernel trick is real strength of SVM as it is very flexible classifier and does not solve for local optima. Lastly, the risk of over-fitting is relatively small in SVM. However, the disadvantage of this model is that it is difficult to understand and interpret the final model. Unlike logisitic regression or decision trees the variable weights and individual impact are not easy to extract and visualize. 
