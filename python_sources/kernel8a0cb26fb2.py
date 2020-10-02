#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Predicting Survival on the Titanic
# 

# # Predict survival on the Titanic
# 
#     Defining the problem statement
#  
#     Collecting the data
#     
#     Exploratory data analysis
#     
#     Feature engineering
#     
#     Modelling
#     
#     Testing

# # 1. Defining the problem statement

# In[ ]:



I apply the tools of machine learning to predict which passengers survived the Titanic tragedy.


# # Load the libraries

# In[ ]:


# Import Dependencies
get_ipython().run_line_magic('matplotlib', 'inline')

# Start Python Imports
import math, time, random, datetime

# Data Manipulation and mathematics operator
import numpy as np
import pandas as pd

# Visualization 
import matplotlib.pyplot as plt
import missingno
import seaborn as sns
plt.style.use('seaborn-whitegrid')

# Preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize

# Machine learning
import catboost
from sklearn.model_selection import train_test_split
from sklearn import model_selection, tree, preprocessing, metrics, linear_model
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier, Pool, cv
from catboost import CatBoostRegressor, Pool, cv

#Data scaling
from sklearn.preprocessing import StandardScaler

# Let's be rebels and ignore warnings for now
import warnings
warnings.filterwarnings('ignore')


# # Load the test and trian data

# In[ ]:


# Import train & test data 
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
gender_submission = pd.read_csv('data/gender_submission.csv') # example of what a submission should look like


# In[ ]:


# View top head row of all the dataset
display(train.head(), test.head(),gender_submission.head())


# # Data Descriptions

# Survival: 0 = No, 1 = Yes
#     
# pclass (Ticket class): 1 = 1st, 2 = 2nd, 3 = 3rd
#     
# sex: Sex
#     
# Age: Age in years
#     
# sibsp: number of siblings/spouses aboard the Titanic
#     
# parch: number of parents/children aboard the Titanic
#     
# ticket: Ticket number
#     
# fare: Passenger fare
#     
# cabin: Cabin number
#     
# embarked: Port of Embarkation, C = Cherbourg, Q = Queenstown, S = Southampton

# In[ ]:


train.describe()


# Missing values can be checked and visualised

# In[ ]:


train.isnull().head()


# # Check for  missing values 

# In[ ]:


# graphically determine the  missing values by missingno
missingno.matrix(train, figsize = (30,12))


# In[ ]:


# graphically determine the  missing values by seaborn
sns.heatmap(train.isnull())


# In[ ]:


# the correlation matrix about the locations of missing values in columns.
missingno.heatmap(train)


# In[ ]:


# Confirm the number of missing values in each column.
train.info()


# In[ ]:


#Let determine the number of missing scores in each feature

def find_missing_values(df, columns):
    """
    Finds number of rows where certain columns are missing values.
    ::param_df:: = target dataframe
    ::param_columns:: = list of columns
    """
    missing_vals = {}
    print("Number of missing or NaN values for each column:")
    df_length = len(df)
    for column in columns:
        total_column_values = df[column].value_counts().sum()
        missing_vals[column] = df_length-total_column_values
        #missing_vals.append(str(column)+ " column has {} missing or NaN values.".format())
    return missing_vals

missing_values = find_missing_values(train, columns=train.columns)
missing_values


# In[ ]:


# The following features will be droped Name , Age, Ticket and Cabin
train= train.drop(['Name','Age','Cabin','Ticket','PassengerId'], axis=1)
train.head()


# # How many survive?

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
train['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot('Survived',data=train,ax=ax[1])
ax[1].set_title('Survived')
plt.show()


# # Let look into  some of these features 

# Feature: Embarked

# In[ ]:


# How many missing values does Embarked have?
missing_values['Embarked']


# In[ ]:


# What kind of values are in Embarked?
train.Embarked.value_counts()


# In[ ]:


# What do the counts look like?
sns.countplot(y='Embarked', data=train);


# I will delete the  two missing record  from Embarke

# In[ ]:


train= train.dropna(subset=['Embarked'])


# # Feature :Sex

# In[ ]:


train.groupby(['Sex','Survived'])['Survived'].count()


# In[ ]:


# Let's view the distribution of Sex
f,ax=plt.subplots(1,2,figsize=(18,8))

train[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Sex')
sns.countplot('Sex',hue='Survived',data=train,ax=ax[1])
ax[1].set_title('Sex:Survived vs Dead')
plt.show()


# In[ ]:


#Let change 0 for male and 1 for female
train['Sex'] = train['Sex']


# In[ ]:


train.head()


# # Feature: Fare

# In[ ]:


# What kind of variable is Fare?
train.Fare.dtype


# In[ ]:


print('Highest Fare was:',train['Fare'].max())
print('Lowest Fare was:',train['Fare'].min())
print('Average Fare was:',train['Fare'].mean())


# In[ ]:


# How many unique kinds of Fare are there?
print("There are {} unique Fare values.".format(len(train.Fare.unique())))


# In[ ]:



train['Fare'] = train['Fare']


# In[ ]:


# What do our Fare bins look like?
train.Fare.value_counts().head()


# In[ ]:


f,ax=plt.subplots(1,3,figsize=(20,8))
sns.distplot(train[train['Pclass']==1].Fare,ax=ax[0])
ax[0].set_title('Fares in Pclass 1')
sns.distplot(train[train['Pclass']==2].Fare,ax=ax[1])
ax[1].set_title('Fares in Pclass 2')
sns.distplot(train[train['Pclass']==3].Fare,ax=ax[2])
ax[2].set_title('Fares in Pclass 3')
plt.show()


# There looks to be a large distribution in the fares of Passengers in Pclass1 and this distribution goes on decreasing as the standards reduces. As this is also continous, we can convert into discrete values by using binning.

# In[ ]:


display(test.head(), train.head())


# # Feature encoding(Label encoding)

# In[ ]:


# Label Encode all continuous values using LabelEncoder()
train = train.apply(LabelEncoder().fit_transform)

train.head()


# # Correlation Between The Features

# In[ ]:


sns.heatmap(train.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()


# # Predictive Modeling

# In[ ]:


# Split the dataframe into data and labels
x_train = train.drop('Survived', axis=1) # data
y_train = train.Survived # labels


#target = train["Survived"]
#x_train, x_val, y_train, y_val = train_test_split(train, target, test_size = 0.22, random_state = 0)


# # Gaussian Naive Bayes

# In[ ]:


# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.cross_validation  import cross_val_score

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_train)


cr_gaussian = classification_report(y_train,y_pred)
acc_gaussian = round(accuracy_score(y_pred, y_train) * 100, 2)
acc_cv_gaussian =round(cross_val_score(gaussian, x_train,y_train, cv=10, scoring='accuracy').mean() * 100,2)

print("Accuracy: %s" % acc_gaussian)
print("Accuracy CV 10-Fold: %s" % acc_cv_gaussian)
print(cr_gaussian)


# # Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_train)

cr_logreg = classification_report(y_train,y_pred)
acc_logreg = round(accuracy_score(y_pred, y_train) * 100, 2)


acc_cv_logreg =round(cross_val_score(logreg, x_train,y_train, cv=10, scoring='accuracy').mean() * 100,2)

print("Accuracy: %s" % acc_logreg)
print("Accuracy CV 10-Fold: %s" % acc_cv_logreg)
print(cr_logreg)


# In[ ]:


# Support Vector Machines


# In[ ]:


from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_train)



cr_svc = classification_report(y_train,y_pred)
acc_svc = round(accuracy_score(y_pred, y_train) * 100, 2)


acc_cv_svc =round(cross_val_score(svc, x_train,y_train, cv=10, scoring='accuracy').mean() * 100,2)

print("Accuracy: %s" % acc_svc)
print("Accuracy CV 10-Fold: %s" % acc_cv_svc)
print(cr_svc)


# In[ ]:


# Linear SVC


# In[ ]:



from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_train)

cr_linear_svc = classification_report(y_train,y_pred)
acc_linear_svc = round(accuracy_score(y_pred, y_train) * 100, 2)

acc_cv_linear_svc =round(cross_val_score(linear_svc, x_train,y_train, cv=10, scoring='accuracy').mean() * 100,2)

print("Accuracy: %s" % acc_linear_svc)
print("Accuracy CV 10-Fold: %s" % acc_cv_linear_svc)
print(cr_linear_svc)


# In[ ]:


#Decision Tree


# In[ ]:


from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_train)

cr_decisiontree = classification_report(y_train,y_pred)
acc_decisiontree = round(accuracy_score(y_pred, y_train) * 100, 2)


acc_cv_decisiontree =round(cross_val_score(decisiontree, x_train,y_train, cv=10, scoring='accuracy').mean() * 100,2)

print("Accuracy: %s" % acc_decisiontree)
print("Accuracy CV 10-Fold: %s" % acc_cv_decisiontree)
print(cr_decisiontree)


# In[ ]:


# Random Forest


# In[ ]:



from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_train)

cr_randomforest = classification_report(y_train,y_pred)
acc_randomforest = round(accuracy_score(y_pred, y_train) * 100, 2)

acc_cv_randomforest =round(cross_val_score(randomforest, x_train,y_train, cv=10, scoring='accuracy').mean() * 100,2)

print("Accuracy: %s" % acc_randomforest)
print("Accuracy CV 10-Fold: %s" % acc_cv_randomforest)
print(cr_randomforest)


# In[ ]:


# KNN or k-Nearest Neighbors


# In[ ]:



from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_train)

cr_knn = classification_report(y_train,y_pred)
acc_knn = round(accuracy_score(y_pred, y_train) * 100, 2)


acc_cv_knn =round(cross_val_score(knn, x_train,y_train, cv=10, scoring='accuracy').mean() * 100,2)

print("Accuracy: %s" % acc_knn)
print("Accuracy CV 10-Fold: %s" % acc_cv_knn)
print(cr_knn)


# In[ ]:


# Stochastic Gradient Descent


# In[ ]:


from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_train)

cr_sgd = classification_report(y_train,y_pred)
acc_sgd = round(accuracy_score(y_pred, y_train) * 100, 2)


acc_cv_sgd =round(cross_val_score(sgd, x_train,y_train, cv=10, scoring='accuracy').mean() * 100,2)

print("Accuracy: %s" % acc_sgd)
print("Accuracy CV 10-Fold: %s" % acc_cv_sgd)
print(cr_sgd)


# In[ ]:


# Gradient Boosting Classifier


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_train)

cr_gbk = classification_report(y_train,y_pred)
acc_gbk = round(accuracy_score(y_pred, y_train) * 100, 2)


acc_cv_gbk =round(cross_val_score(gbk, x_train,y_train, cv=10, scoring='accuracy').mean() * 100,2)

print("Accuracy: %s" % acc_gbk)
print("Accuracy CV 10-Fold: %s" % acc_cv_gbk)
print(cr_gbk)


# # Let's compare the accuracies of each model!
# 

# In[ ]:




models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Linear SVC', 
              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],
    'Score': [acc_svc, acc_knn, acc_logreg, 
              acc_randomforest, acc_gaussian,acc_linear_svc, acc_decisiontree,
              acc_sgd, acc_gbk]})
models.sort_values(by='Score', ascending=False)


# In[ ]:



cv_models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Linear SVC', 
              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],
    'Score': [acc_cv_svc, acc_cv_knn, acc_cv_logreg, 
              acc_cv_randomforest, acc_cv_gaussian,acc_cv_linear_svc, acc_cv_decisiontree,
              acc_cv_sgd, acc_cv_gbk]})
print('---Cross-validation Accuracy Scores---')
cv_models.sort_values(by='Score', ascending=False)


# We can see from the tables, the Gradient Boosting Classifier had the best results with 81% , It is the best to pay more attention to the cross-validation figure.
# Cross-validation is more robust than just the .fit() models as it does multiple passes over the data instead of one.
# Because the CatBoost model got the best resultsThere fore the best model to be considered is Gradient Boosting Classifier

# In[ ]:


# Feature Importance
def feature_importance(model, data):
    """
    Function to show which features are most important in the model.
    ::param_model:: Which model to use?
    ::param_data:: What data to use?
    """
    fea_imp = pd.DataFrame({'imp': model.feature_importances_, 'col': data.columns})
    fea_imp = fea_imp.sort_values(['imp', 'col'], ascending=[True, False]).iloc[-30:]
    _ = fea_imp.plot(kind='barh', x='col', y='imp', figsize=(20, 10))
    return fea_imp
    #plt.savefig('catboost_feature_importance.png') 


# In[ ]:


# Plot the feature importance scores
feature_importance(gbk, X_train)


# In[ ]:


test1 =test.drop(['PassengerId','Name','Age','Ticket','Cabin'], axis=1)
test1.head()


# In[ ]:


# Make a prediction using the CatBoost model on the wanted columns
predictions = gbk.predict(test1.apply(LabelEncoder().fit_transform))


# In[ ]:


# Our predictions array is comprised of 0's and 1's (Survived or Did Not Survive)
predictions[:20]


# In[ ]:


# Create a submisison dataframe and append the relevant columns
submission = pd.DataFrame()
submission['PassengerId'] = test['PassengerId']
submission['Survived'] = predictions # our model predictions on the test dataset

# Let's convert our submission dataframe 'Survived' column to ints
submission['Survived'] = submission['Survived'].astype(int)
print('Converted Survived column to integers.')
submission.head(20)


# In[ ]:


# Convert submisison dataframe to csv for submission to csv 
# for Kaggle submisison
submission.to_csv('C:/Users/sfagb/Desktop/kaggle/Titanic/submission.csv', index=False)
print('Submission CSV is ready!')


# In[ ]:




