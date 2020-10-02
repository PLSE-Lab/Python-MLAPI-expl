#!/usr/bin/env python
# coding: utf-8

# # **Titanic Survival Prediction**

# *The RMS Titanic was a British passenger liner that sank in the North Atlantic Ocean in the early morning hours of 15 April 1912, after it collided with an iceberg during its maiden voyage from Southampton to New York City. There were an estimated 2,224 passengers and crew aboard the ship, and more than 1,500 died, making it one of the deadliest commercial peacetime maritime disasters in modern history. The RMS Titanic was the largest ship afloat at the time it entered service and was the second of three Olympic-class ocean liners operated by the White Star Line.*
# 
# **The goal of this project is to complete the analysis of what sorts of people were likely to survive.**

# #  Table of Contents
# 
#   1. Missing Data Analysis
#   2. Exploratory Data Analysis
#   3. Feature Engineering
#   4. Feature Selection
#   5. Feature Importance
#   6. Classification algorithmsand metrics explained
#   7. Hyperparameter Tuning
#   8. Ensemble Techniques for Prediction
#   9. Model Evaluation
#   10. Submission

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import numpy as np
import  pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


train_df=pd.read_csv('../input/titanic/train.csv')
test_df=pd.read_csv('../input/titanic/test.csv')


# In[ ]:


train_df.columns


# In[ ]:


test_df.columns


# In[ ]:


train_df.info()
# Age, Cabin, Embarked has null values


# In[ ]:


train_df.describe()


# **Missing values Imputation**

# In[ ]:


Gend_male = pd.DataFrame(train_df[train_df["Sex"] == "male"])
mean_age_male = Gend_male['Age'].mean()

Gend_female = pd.DataFrame(train_df[train_df["Sex"] == "female"])
mean_age_female = Gend_female['Age'].mean()
train_df.loc[(train_df['Age'].isna()) & (train_df['Sex']=='male'), 'Age']=mean_age_male
train_df.loc[(train_df['Age'].isna()) & (train_df['Sex']=='female'), 'Age']=mean_age_female

Gend_male_test = pd.DataFrame(test_df[test_df["Sex"] == "male"])
mean_age_male_test = Gend_male_test['Age'].mean()

Gend_female_test = pd.DataFrame(test_df[test_df["Sex"] == "female"])
mean_age_female_test = Gend_female_test['Age'].mean()
test_df.loc[(test_df['Age'].isna()) & (test_df['Sex']=='male'), 'Age']=mean_age_male
test_df.loc[(test_df['Age'].isna()) & (test_df['Sex']=='female'), 'Age']=mean_age_female





# In[ ]:


train_df['Cabin'].fillna('NaN',inplace=True)
test_df['Cabin'].fillna('NaN',inplace=True)


# In[ ]:


train_df['Embarked'].fillna('Unknown',inplace=True)


# # **Total number of survived and not survived**

# In[ ]:



Survival_rate = {'Survived_count': [0],
                'Not_Survived_count' : [0],
                'Total' : [0]}

Survival_rate['Survived_count'] =  train_df.Survived.value_counts()[0]
Survival_rate['Not_Survived_count'] =  train_df.Survived.value_counts()[1]
Survival_rate['Total'] =  Survival_rate['Survived_count'] + Survival_rate['Not_Survived_count']

# Create the index 
index_ = ['Survival_Rate'] 
  
# Set the index 

Survival=pd.DataFrame([Survival_rate])
Survival.index = index_ 
Survival.transpose()


# In[ ]:


Survival.hist(figsize=(10,10),grid=False)
#plt.plot()


# # **Survived and Not Survived by Age and Embarked**

# In[ ]:



s_count = {'Not_Survived':[0],
          'Survived':[0]}
sf_count = {'Not_Survived':[0],
          'Survived':[0]}
c_count = {'Not_Survived':[0],
          'Survived':[0]}
cf_count = {'Not_Survived':[0],
          'Survived':[0]}
q_count = {'Not_Survived':[0],
          'Survived':[0]}
qf_count = {'Not_Survived':[0],
          'Survived':[0]}
s_count['Not_Survived'] = train_df.loc[(train_df['Sex']=='male') & (train_df['Embarked']=='S'),'Survived'].value_counts()[0]
s_count['Survived'] = train_df.loc[(train_df['Sex']=='male') & (train_df['Embarked']=='S'),'Survived'].value_counts()[1]
#Create the index 
index_ = ['Male passengers : Embarked S'] 
# Set the index 
male_S=pd.DataFrame([s_count])
male_S.index = index_ 
#male_S.transpose()
#male_S
sf_count['Not_Survived'] = train_df.loc[(train_df['Sex']=='female') & (train_df['Embarked']=='S'),'Survived'].value_counts()[0]
sf_count['Survived'] = train_df.loc[(train_df['Sex']=='female') & (train_df['Embarked']=='S'),'Survived'].value_counts()[1]
#Create the index 
index_ = ['Female passengers : Embarked S'] 
# Set the index 
female_S=pd.DataFrame([sf_count])
female_S.index = index_
c_count['Not_Survived'] = train_df.loc[(train_df['Sex']=='male') & (train_df['Embarked']=='C'),'Survived'].value_counts()[0]
c_count['Survived'] = train_df.loc[(train_df['Sex']=='male') & (train_df['Embarked']=='C'),'Survived'].value_counts()[1]
#Create the index 
index_ = ['Male passengers : Embarked C'] 
# Set the index 
male_C=pd.DataFrame([c_count])
male_C.index = index_ 
#male_C
cf_count['Not_Survived'] = train_df.loc[(train_df['Sex']=='female') & (train_df['Embarked']=='C'),'Survived'].value_counts()[0]
cf_count['Survived'] = train_df.loc[(train_df['Sex']=='female') & (train_df['Embarked']=='C'),'Survived'].value_counts()[1]
#Create the index 
index_ = ['Female passengers : Embarked C'] 
# Set the index 
female_C=pd.DataFrame([cf_count])
female_C.index = index_ 
q_count['Not_Survived'] = train_df.loc[(train_df['Sex']=='male') & (train_df['Embarked']=='Q'),'Survived'].value_counts()[0]
q_count['Survived'] = train_df.loc[(train_df['Sex']=='male') & (train_df['Embarked']=='Q'),'Survived'].value_counts()[1]
#Create the index 
index_ = ['Male passengers : Embarked Q'] 
# Set the index 
male_Q=pd.DataFrame([q_count])
male_Q.index = index_ 
#male_Q
qf_count['Not_Survived'] = train_df.loc[(train_df['Sex']=='female') & (train_df['Embarked']=='Q'),'Survived'].value_counts()[0]
qf_count['Survived'] = train_df.loc[(train_df['Sex']=='female') & (train_df['Embarked']=='Q'),'Survived'].value_counts()[1]
#Create the index 
index_ = ['female passengers : Embarked Q'] 
# Set the index 
female_Q=pd.DataFrame([qf_count])
female_Q.index = index_ 
#male_Q
frames=[male_S,female_S,male_C,female_C,male_Q,female_Q]
result=pd.concat(frames)
Gender_Embarked = pd.DataFrame(result)
Gender_Embarked


# In[ ]:


Gender_Embarked[["Not_Survived","Survived"]].plot(kind="bar",stacked=True)


# # **Survived and Non Survived male and female by PClass**

# In[ ]:



a = {'Not_Survived':[0],
          'Survived':[0]}
b = {'Not_Survived':[0],
          'Survived':[0]}
c = {'Not_Survived':[0],
          'Survived':[0]}
d = {'Not_Survived':[0],
          'Survived':[0]}
e = {'Not_Survived':[0],
          'Survived':[0]}
f = {'Not_Survived':[0],
          'Survived':[0]}
a=train_df.loc[(train_df['Sex']=='female') & (train_df['Pclass']== 1),'Survived'].value_counts()
#Create the index 
index_ = ['Female passengers : PClass 1'] 
# Set the index 
female_1=pd.DataFrame([a])
female_1.index = index_ 
b=train_df.loc[(train_df['Sex']=='female') & (train_df['Pclass']== 2),'Survived'].value_counts()
#Create the index 
index_ = ['Female passengers : PClass 2'] 
# Set the index 
female_2=pd.DataFrame([b])
female_2.index = index_ 
c=train_df.loc[(train_df['Sex']=='female') & (train_df['Pclass']== 3),'Survived'].value_counts()
#Create the index 
index_ = ['Female passengers : PClass 3'] 
# Set the index 
female_3=pd.DataFrame([c])
female_3.index = index_ 
d=train_df.loc[(train_df['Sex']=='male') & (train_df['Pclass']== 1),'Survived'].value_counts()
#Create the index 
index_ = ['Male passengers : PClass 1'] 
# Set the index 
male_1=pd.DataFrame([d])
male_1.index = index_ 
e=train_df.loc[(train_df['Sex']=='male') & (train_df['Pclass']== 2),'Survived'].value_counts()
#Create the index 
index_ = ['Male passengers : PClass 2'] 
# Set the index 
male_2=pd.DataFrame([e])
male_2.index = index_ 
f=train_df.loc[(train_df['Sex']=='male') & (train_df['Pclass']== 3),'Survived'].value_counts()
#Create the index 
index_ = ['Male passengers : PClass 3'] 
# Set the index 
male_3=pd.DataFrame([f])
male_3.index = index_ 
#male_Q
frames=[male_1,female_1,male_2,female_2,male_1,female_2,male_3,female_3]
result=pd.concat(frames)
Gender_PClass = pd.DataFrame(result)
Gender_PClass


# In[ ]:


ax2 = Gender_PClass.plot.pie(subplots=True,figsize=(20,20), autopct='%1.1f%%',shadow=True)
plt.legend(loc='center left')
plt.show()


# **Survived and not survived by Pclass**

# In[ ]:


ax=sns.countplot(x='Pclass',hue='Survived',data=train_df)


# Survived and Not Survived by Embarked

# In[ ]:


ax=sns.FacetGrid(train_df,col="Survived")
ax=ax.map(plt.hist,'Age',color="g",bins=10)


# **Categorize by Age bins, PClass and Sex**

# In[ ]:


sns.catplot(x="Pclass",y="Age",hue="Sex",data=train_df)


# # Feature Engineering

# The feature engineering process includes :
#     1. Testing features.
#     2. Deciding what features to create.
#     3. Creating features.
#     4. Checking how the features work with your model.
#     5. Improving your features if needed.
#     6. Create more features until the work is done.
# Feature Engineering techniques include imputation, handling outliers, binning, log transform one-hot encoding, grouping operations, feature split, scaling and extracting date.

# ***Computation of Age Bins*******

# In[ ]:


train_df['age_bins'] = pd.cut(x=train_df['Age'], bins=8, labels=False, retbins=False, include_lowest=True)
test_df['age_bins'] = pd.cut(x=test_df['Age'], bins=8, labels=False, retbins=False, include_lowest=True)


# **Computation of Fare Range****

# In[ ]:


train_df['Fare_cat']=0
train_df.loc[train_df['Fare']<=7.91,'Fare_cat']=0
train_df.loc[(train_df['Fare']>7.91)&(train_df['Fare']<=14.454),'Fare_cat']=1
train_df.loc[(train_df['Fare']>14.454)&(train_df['Fare']<=31),'Fare_cat']=2
train_df.loc[(train_df['Fare']>31)&(train_df['Fare']<=93.5),'Fare_cat']=3
train_df.loc[(train_df['Fare']>93.5)&(train_df['Fare']<=164.8667),'Fare_cat']=4
train_df.loc[(train_df['Fare']>164.8667)&(train_df['Fare']<=512.3292),'Fare_cat']=5

test_df['Fare_cat']=0
test_df.loc[test_df['Fare']<=7.91,'Fare_cat']=0
test_df.loc[(test_df['Fare']>7.91)&(test_df['Fare']<=14.454),'Fare_cat']=1
test_df.loc[(test_df['Fare']>14.454)&(test_df['Fare']<=31),'Fare_cat']=2
test_df.loc[(test_df['Fare']>31)&(test_df['Fare']<=93.5),'Fare_cat']=3
test_df.loc[(test_df['Fare']>93.5)&(test_df['Fare']<=164.8667),'Fare_cat']=4
test_df.loc[(test_df['Fare']>164.8667)&(test_df['Fare']<=512.3292),'Fare_cat']=5


# **Extract Initials from the Name feature. Categorize the Initials by different values**

# In[ ]:


name = train_df['Name']
#Extract the initials
train_df['Title'] = name.str.extract(pat = "(Mr|Master|Mrs|Miss|Major|Rev|Lady|Dr|Mme|Mlle|Col|Capt)\\.")
test_df['Title'] = name.str.extract(pat = "(Mr|Master|Mrs|Miss|Major|Rev|Lady|Dr|Mme|Mlle|Col|Capt)\\.")
train_df['Title'].astype(str)
test_df['Title'].astype(str)
#Assign Rare for the rare initials
train_df.Title[train_df.Title == 'Rev'] = 'Rare'
train_df.Title[train_df.Title == 'Major'] = 'Rare'
train_df.Title[train_df.Title == 'Lady'] = 'Rare'
train_df.Title[train_df.Title == 'Dr'] = 'Rare'
train_df.Title[train_df.Title == 'Mme'] = 'Rare'
train_df.Title[train_df.Title == 'Mlle'] = 'Rare'
train_df.Title[train_df.Title == 'Col'] = 'Rare'
train_df.Title[train_df.Title == 'Capt'] = 'Rare'

test_df.Title[test_df.Title == 'Rev'] = 'Rare'
test_df.Title[test_df.Title == 'Major'] = 'Rare'
test_df.Title[test_df.Title == 'Lady'] = 'Rare'
test_df.Title[test_df.Title == 'Dr'] = 'Rare'
test_df.Title[test_df.Title == 'Mme'] = 'Rare'
test_df.Title[test_df.Title == 'Mlle'] = 'Rare'
test_df.Title[test_df.Title == 'Col'] = 'Rare'
test_df.Title[test_df.Title == 'Capt'] = 'Rare'
# Categorize the Initial
train_df['Title'].replace(['Mr','Mrs','Miss','Master','Rare'],[1,2,3,4,5],inplace=True)
test_df['Title'].replace(['Mr','Mrs','Miss','Master','Rare'],[1,2,3,4,5],inplace=True)
#train_df

# Missing values Imputation
train_df['Title'].fillna(0,inplace=True)
test_df['Title'].fillna(0,inplace=True)


# **Categorize sex to numeric variable**

# In[ ]:


train_df['Sex'].replace(['male','female'],[0,1],inplace=True)
test_df['Sex'].replace(['male','female'],[0,1],inplace=True)


# **Categorize Embarked to numeric variable**

# In[ ]:


#train_df['Embarked'].unique()
train_df['Embarked'] = train_df['Embarked'].map({'S': 1, 'C': 2, 'Q': 3,'Unknown':0} ).astype(int)
test_df['Embarked'] = test_df['Embarked'].map({'S': 1, 'C': 2, 'Q': 3,'Unknown':0} ).astype(int)
#dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


# **Family Size Computation**

# In[ ]:



train_df['Family_Size']=0
train_df['Family_Size'] = train_df['SibSp'] + train_df['Parch']
train_df['IsAlone']=0
train_df.loc[(train_df['Family_Size']==1),'IsAlone']=1
train_df.loc[(train_df['Family_Size']==0) | (train_df['Family_Size']>1),'IsAlone']=0

test_df['Family_Size']=0
test_df['Family_Size'] = test_df['SibSp'] + test_df['Parch']
test_df['IsAlone']=0
test_df.loc[(test_df['Family_Size']==1),'IsAlone']=1
test_df.loc[(test_df['Family_Size']==0) | (test_df['Family_Size']>1),'IsAlone']=0


# # Feature Selection

# Feature selection is one of the core concepts in machine learning which hugely impacts the performance of the model. Irrelavant or partially relavant features can negatively impact the performance of the model. Feature selection and data cleaning should be the first and most important step of your model designing.
# **Benefits of Feature Selection :**
#     1. Reduces Overfitting
#     2. Improves accuracy
#     3. Reduces training time
# **Feature Selection Methods :**
#     1. Intrinsic
#     2. Wrapper methods
#     3. Filter methods
# The **intrinsic method** uses an algorithm ExtraTrees classifier.
# The **Wrapper method** uses techniques such as Forward feature selection, Backward elimination, Recursive feature elimination.
# The **filter method** is divided into two types. Statistical approach and Feature Importance. The statistical approaches are Pearson Coefficient, Spearman Coefficient, ANOVA, Chisquared Test and mutual information test. Corelation Heatmap is drawn to identify the feature importances.
# The **embedded method** is a combination of wrapper and filter methods. The techniques in embedded methods are Ridge regression and Lasso regression.

# In[ ]:


train_df.info()


# In[ ]:


test_df.info()


# **Corelation of all the attributes by Heatmap**

# In[ ]:


sns.heatmap(train_df.corr(), annot=True).set_title("Corelation of attributes")
fig=plt.gcf()
fig.set_size_inches(15,15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


# The positive corelated attributes are age : age_bins (0.97), SibSp : Family_size (0.89), Parch : Family_size (0.78), Sex:Title (0.58), Survived : Sex(0.54). Negative corelated : Name, Ticket, Cabin, Passenger_Id, Fare, Fare_Cat, Age

# In[ ]:


sns.heatmap(test_df.corr(), annot=True).set_title("Corelation of attributes")
fig=plt.gcf()
fig.set_size_inches(15,15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


# # Feature Importance by ExtraTreeClassifier

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
y = train_df['Survived'] 
#X = pd.DataFrame(train_df)
#df.drop(['A'], axis = 1)
X = train_df.drop(['Survived','Name','Ticket','Cabin'],axis=1) 


# In[ ]:


# Building the model 
extra_tree_forest = ExtraTreesClassifier(n_estimators = 5,criterion ='entropy', max_features = 5) 
  
# Training the model 
extra_tree_forest.fit(X, y) 
  
# Computing the importance of each feature 
feature_importance = extra_tree_forest.feature_importances_ 
  
# Normalizing the individual importances 
feature_importance_normalized = np.std([tree.feature_importances_ for tree in extra_tree_forest.estimators_], axis = 0) 

# Plotting a Bar Graph to compare the models 
plt.figure(figsize=(20,6))
plt.bar(X.columns,feature_importance_normalized,align='edge', width=0.3) 
plt.xlabel('Feature Labels') 
plt.ylabel('Feature Importances') 
plt.title('Comparison of different Feature Importances') 

plt.show() 


# **Chisquare Test for Feature Selection**

# In[ ]:


from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import chi2 
X.shape
# Two features with highest chi-squared statistics are selected 
chi2_features = SelectKBest(chi2, k = 9) 
X_kbest_features = chi2_features.fit_transform(X, y) 
  
# Reduced features 
print('Original feature number:', X.shape[1]) 
print('Reduced feature number:', X_kbest_features.shape[1])
#X
X_kbest_features
#PClass, Age, Sex, Fare, Fare_cat, Title, Family Size - Top 7 features


# In[ ]:


X.head()


# # Prediction - Classification Algorithms

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


# Based on the feature selection techniques, retaining the most important features in the dataset for prediction and dropping the unnecessary features.

# **Drop unnecessary columns in train and test set before predictions**

# In[ ]:


train_df = train_df.drop("Name", axis=1)
train_df = train_df.drop("Ticket", axis=1)
train_df = train_df.drop("Cabin", axis=1)
train_df = train_df.drop("Fare", axis=1)
#train_df = train_df.drop("Embarked", axis=1)
train_df = train_df.drop("IsAlone", axis=1)
#train_df = train_df.drop("SibSp", axis=1)
#train_df = train_df.drop("Parch", axis=1)
#train_df = train_df.drop("Pclass", axis=1)
#train_df = train_df.drop("Age", axis=1)
train_df = train_df.drop("age_bins", axis=1)
train_df = train_df.drop("PassengerId", axis=1)


test_df = test_df.drop("Name", axis=1)
test_df = test_df.drop("Ticket", axis=1)
test_df = test_df.drop("Cabin", axis=1)
test_df = test_df.drop("Fare", axis=1)
#test_df = test_df.drop("Embarked", axis=1)
test_df = test_df.drop("IsAlone", axis=1)
#test_df = test_df.drop("SibSp", axis=1)
#test_df = test_df.drop("Parch", axis=1)
#test_df = test_df.drop("Pclass", axis=1)
#test_df = test_df.drop("Age", axis=1)
test_df = test_df.drop("age_bins", axis=1)


# **Corelation map (Heatmap) after Feature Selection and dropping columns for prediction**

# In[ ]:


sns.heatmap(train_df.corr(), annot=True).set_title("Corelation of attributes")
fig=plt.gcf()
fig.set_size_inches(15,15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


# **Split the data into train and test set for classifcation predictions**

# In[ ]:


X_train = train_df.drop("Survived", axis=1)
y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, y_train.shape, X_test.shape


# In[ ]:


X_test.head()


# **Logistic Regression**

# In[ ]:


model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
#acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
logistic_score = round(model.score(X_train,y_train) * 100,2)
print(logistic_score)


# **SVM Classification**

# RBF SVM

# In[ ]:


from sklearn import svm
model = svm.SVC(kernel='rbf',C=1,gamma=0.1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc_svc = round(model.score(X_train, y_train) * 100, 2)
acc_svc


# **Linear SVC**

# In[ ]:


model = svm.SVC(kernel='linear',C=0.1,gamma=0.1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc_l_svc = round(model.score(X_train, y_train) * 100, 2)
acc_l_svc


# **KNN Classification**

# In[ ]:


#KNN Classification
a_index=list(range(1,11))
a=pd.Series()
x=[0,1,2,3,4,5,6,7,8,9,10]
for i in list(range(1,11)):
    model=KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    a=a.append(pd.Series(model.score(X_train,y_train)))
plt.plot(a_index, a)
plt.xticks(x)
fig=plt.gcf()
fig.set_size_inches(12,6)
plt.show()
print('Accuracies for different values of n are:',a.values,'with the max value as ',a.values.max())
acc_knn = round(model.score(X_train,y_train)*100,2)
acc_knn


# **Gaussian Naive Bayes Classification**

# In[ ]:


model=GaussianNB()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
acc_gaus = round(model.score(X_train,y_train)*100,2)
acc_gaus


# **Decision Tree Classification**

# In[ ]:


model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc_dec = round(model.score(X_train, y_train) * 100, 2)
acc_dec


# **Feature Selection using RFE (Recursive Feature Elimination)**

# In[ ]:


from sklearn.feature_selection import RFE
rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=5)
rfe.fit(X_train, y_train)
y_pred = rfe.predict(X_test)
rfe.score(X_train, y_train)
rfe_scc = round(rfe.score(X_train, y_train) * 100, 2)

for i in range(X_train.shape[1]):
    print('Column: %d, Name: %s, Selected %s, Rank: %.3f' % (i, X_train.columns[i],rfe.support_[i], rfe.ranking_[i]))


# Lists of the 6 input features and whether or not they were selected as well as their relative ranking of importance.

# **Random Forest Classification**

# In[ ]:


model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
y_pred_ran = model.predict(X_test)
model.score(X_train, y_train)
acc_forest = round(model.score(X_train, y_train) * 100, 2)
acc_forest


# # **Cross Validation**

# In[ ]:


from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts
xyz=[]
accuracy=[]
std=[]
classifiers=['Linear Svm','Radial Svm','Logistic Regression','KNN','Decision Tree','Naive Bayes','Random Forest']
models=[svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),KNeighborsClassifier(n_neighbors=9),DecisionTreeClassifier(),GaussianNB(),RandomForestClassifier(n_estimators=100)]
for i in models:
    model = i
    cv_result = cross_val_score(model,X_train,y_train, cv = kfold,scoring = "accuracy")
    cv_result=cv_result
    xyz.append(cv_result.mean())
    std.append(cv_result.std())
    accuracy.append(cv_result)
new_models_dataframe2=pd.DataFrame({'CV Mean':xyz,'Std':std},index=classifiers)       
new_models_dataframe2


# In[ ]:


plt.subplots(figsize=(12,6))
box=pd.DataFrame(accuracy,index=[classifiers])
box.T.boxplot()


# In[ ]:


new_models_dataframe2['CV Mean'].plot.barh(width=0.8)
plt.title('Average CV Mean Accuracy')
fig=plt.gcf()
fig.set_size_inches(12,5)
plt.show()


# **Confusion Matrix**

# In[ ]:


from sklearn.metrics import confusion_matrix
f,ax=plt.subplots(3,3,figsize=(12,10))
y_pred = cross_val_predict(svm.SVC(kernel='rbf'),X_train,y_train,cv=10)
sns.heatmap(confusion_matrix(y_train,y_pred),ax=ax[0,0],annot=True,fmt='2.0f')
ax[0,0].set_title('Matrix for rbf-SVM')

y_pred = cross_val_predict(svm.SVC(kernel='linear'),X_train,y_train,cv=10)
sns.heatmap(confusion_matrix(y_train,y_pred),ax=ax[0,1],annot=True,fmt='2.0f')
ax[0,1].set_title('Matrix for Linear-SVM')

y_pred = cross_val_predict(KNeighborsClassifier(n_neighbors=9),X_train,y_train,cv=10)
sns.heatmap(confusion_matrix(y_train,y_pred),ax=ax[0,2],annot=True,fmt='2.0f')
ax[0,2].set_title('Matrix for KNN')

y_pred = cross_val_predict(RandomForestClassifier(n_estimators=100),X_train,y_train,cv=10)
sns.heatmap(confusion_matrix(y_train,y_pred),ax=ax[1,0],annot=True,fmt='2.0f')
ax[1,0].set_title('Matrix for Random-Forests')

y_pred = cross_val_predict(LogisticRegression(),X_train,y_train,cv=10)
sns.heatmap(confusion_matrix(y_train,y_pred),ax=ax[1,1],annot=True,fmt='2.0f')
ax[1,1].set_title('Matrix for Logistic Regression')

y_pred = cross_val_predict(DecisionTreeClassifier(),X_train,y_train,cv=10)
sns.heatmap(confusion_matrix(y_train,y_pred),ax=ax[1,2],annot=True,fmt='2.0f')
ax[1,2].set_title('Matrix for Decision Tree')

y_pred = cross_val_predict(GaussianNB(),X_train,y_train,cv=10)
sns.heatmap(confusion_matrix(y_train,y_pred),ax=ax[2,0],annot=True,fmt='2.0f')
ax[2,0].set_title('Matrix for Naive Bayes')

plt.subplots_adjust(hspace=0.2,wspace=0.2)
plt.show()


# # **Hyperparameter Tuning**

# A hyperparameter is a parameter whose value is set before the learning process begins. Hyperparameter tuning is choosing a set of optimal parameters for a learning algorithm. Two different methods for optimizing hyperparameters : **GridSearch** and **RandomSearch**

# Hyperparameters tuning for Kernel SVM, Decision Tree and Random Forest.

# Kernel SVM

# In[ ]:


from sklearn.model_selection import GridSearchCV
C=[2,2.1,2.5]
gamma=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
kernel=['rbf','linear']
hyper={'kernel':kernel,'C':C,'gamma':gamma}
gd=GridSearchCV(estimator=svm.SVC(),param_grid=hyper,verbose=True)
gd.fit(X_train,y_train)
print(gd.best_score_)
print(gd.best_estimator_)


# Random Forest Classification

# In[ ]:


n_estimators=range(100,1000,1100)
hyper={'n_estimators':n_estimators}
gd=GridSearchCV(estimator=RandomForestClassifier(random_state=0),param_grid=hyper,verbose=True)
gd.fit(X_train,y_train)
print(gd.best_score_)
print(gd.best_estimator_)


# The best score for RBF-SVM is 82.2697 with C=1 and gamma = 0.5. The best score for Random Forest is 82.0456 with n_estimators=200

# # **Ensemble Algorithms**

# Ensemble methods are tecghniques that create multiple models and then combine them to produce improved results. Ensemble methods usually produce more accurate solutions than a single model would.
# In ensemble algorithms, **bagging methods** form a class of algorithms which build several instances of a black-box estimator on random subsets of the original training set and then aggregate their individual predictions to form a final prediction.
# **Boosting** is an ensemble meta-algorithm for primarily reducing bias, and also variance.

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import xgboost as xgb
import lightgbm as lgb


# **Voting Classifier**

# In[ ]:


from sklearn.ensemble import VotingClassifier
ensemble_lin_rbf=VotingClassifier(estimators=[('KNN',KNeighborsClassifier(n_neighbors=10)),
                                              ('RBF',svm.SVC(probability=True,kernel='rbf',C=0.5,gamma=0.1)),
                                              ('RFor',RandomForestClassifier(n_estimators=500,random_state=0)),
                                              ('LR',LogisticRegression(C=0.05)),
                                              ('DT',DecisionTreeClassifier(random_state=0)),
                                              ('NB',GaussianNB()),
                                              ('svm',svm.SVC(kernel='linear',probability=True))
                                             ], 
                       voting='soft').fit(X_train,y_train)
print('The accuracy for ensembled model is:',round(ensemble_lin_rbf.score(X_train,y_train)*100,2))
vot = round(ensemble_lin_rbf.score(X_train,y_train)*100,2)
cross=cross_val_score(ensemble_lin_rbf,X_train,y_train, cv = 10,scoring = "accuracy")
print('The cross validated score is',cross.mean())




# **Bagging**

# Bagging also known as Bootstrap aggregation is a way to decrease the variance in the prediction and helps to avoid overfitting. Although it is usually applied to decision tree methods, it can be used with any type of method.
# Bagging works best with models with high variance. An example for this can be Decision Tree or Random Forests. We can use KNN with small value of n_neighbours, as small value of n_neighbours.

# Bagged KNN

# In[ ]:


from sklearn.ensemble import BaggingClassifier
model=BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=3),random_state=0,n_estimators=700)
model.fit(X_train,y_train)
prediction=model.predict(X_test)
print('The accuracy for bagged KNN is:',round(model.score(X_train,y_train)*100,2))
bag_knn = round(model.score(X_train,y_train)*100,2)
result=cross_val_score(model,X_train,y_train,cv=10,scoring='accuracy')
print('The cross validated score for bagged KNN is:',round(result.mean()*100,2))



# Bagged DecisionTree

# In[ ]:


model=BaggingClassifier(base_estimator=DecisionTreeClassifier(),random_state=0,n_estimators=100)
model.fit(X_train,y_train)
prediction=model.predict(X_test)
print('The accuracy for bagged Decision Tree is:',round(model.score(X_train,y_train)*100,2))
bag_ran = round(model.score(X_train,y_train)*100,2)
result=cross_val_score(model,X_train,y_train,cv=10,scoring='accuracy')
print('The cross validated score for bagged Decision Tree is:',round(result.mean()*100,2))


# Boosting

# Boosting is an ensemble method for improving the model predictions of any given learning algorithm. The idea of boosting is to train weak learners sequentially, each trying to correct its predecessor.
# Boosting is an iterative technique which adjusts the weight of an observation based on the last classification.

# **Ada Boost Classifier**

# In[ ]:


ada = AdaBoostClassifier(random_state=1,n_estimators=1000)
ada.fit(X_train, y_train)
y_pred = ada.predict(X_test)
ada.score(X_train, y_train)
ada_boost = round(ada.score(X_train, y_train) * 100, 2)
ada_boost


# **Gradient Boosting**

# In[ ]:


grad = GradientBoostingClassifier(n_estimators=1000,learning_rate=0.01,random_state=0)
grad.fit(X_train, y_train)
y_pred = grad.predict(X_test)
grad.score(X_train, y_train)
grad_boost = round(grad.score(X_train, y_train) * 100, 2)
grad_boost


# **Extreme Gradient Boosting**

# In[ ]:


extreme = xgb.XGBClassifier(n_estimators=1000,learning_rate=0.1)
extreme.fit(X_train,y_train)
y_pred = extreme.predict(X_test)
extreme.score(X_train, y_train)
extreme_boost = round(extreme.score(X_train, y_train) * 100, 2)
extreme_boost


# Confusion matrix for the best model

# In[ ]:


xgb=XGBClassifier(n_estimators=1000,random_state=0,learning_rate=0.01)
result=cross_val_predict(xgb,X_train,y_train,cv=10)
sns.heatmap(confusion_matrix(y_train,result),cmap='winter',annot=True,fmt='2.0f')
plt.show()


# Feature Importance

# In[ ]:


import xgboost as xgb
f,ax=plt.subplots(2,2,figsize=(15,12))
model=RandomForestClassifier(n_estimators=1000,random_state=0)
model.fit(X_train,y_train)
pd.Series(model.feature_importances_,X_train.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[0,0])
ax[0,0].set_title('Feature Importance in Random Forests')

model=AdaBoostClassifier(n_estimators=1000,learning_rate=0.01,random_state=0)
model.fit(X_train,y_train)
pd.Series(model.feature_importances_,X_train.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[0,1],color='#ddff11')
ax[0,1].set_title('Feature Importance in AdaBoost')

model=GradientBoostingClassifier(n_estimators=1000,learning_rate=0.1,random_state=0)
model.fit(X_train,y_train)
pd.Series(model.feature_importances_,X_train.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[1,0],cmap='RdYlGn_r')
ax[1,0].set_title('Feature Importance in Gradient Boosting')

model=xgb.XGBClassifier(n_estimators=1000,learning_rate=0.1)
#extreme = xgb.XGBClassifier(n_estimators=1000,learning_rate=0.1)
model.fit(X_train,y_train)
pd.Series(model.feature_importances_,X_train.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[1,1],color='#FD0F00')
ax[1,1].set_title('Feature Importance in XgBoost')
plt.show()


# In[ ]:


submission = pd.DataFrame({"PassengerId": test_df["PassengerId"],"Survived": y_pred_ran})
submission
submission.to_csv('submission.csv',index=False)
#test_df=pd.read_csv('../input/titanic/test.csv')


# In[ ]:


models = pd.DataFrame({'Model': ['Radial SVC', 'KNN', 'Logistic Regression','Random Forest', 'Naive Bayes', 'Linear SVC', 'Decision Tree','VotingClassifier','Bagged KNN','Bagged DecisionTree','AdaBoost','GradientBoost','XGBoost'],
    'Score': [acc_svc, acc_knn, logistic_score,acc_forest, acc_gaus, acc_l_svc, acc_dec,vot,bag_knn,bag_ran,ada_boost,grad_boost,extreme_boost]})
models.sort_values(by='Score', ascending=False)

