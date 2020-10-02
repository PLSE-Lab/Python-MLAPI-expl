#!/usr/bin/env python
# coding: utf-8

# Hello Friends! I am excited to present my perspective of Titanic problem. 
# 
# Targeted Audience
# 1. Who wants to start with Kaggle and reach Top 5% in the competition
# 2. Implement various classification algorithms inlcuing most trending one i.e XGBoost
# 3. Understand Titanic use case and solve similar classification problem.
# 
# Let me first summarize what I have done in this project.
# 
# 1. Data Preprocessing (Includes Exploratory data Analysis and Feature Engineering )
# 2. Implement various classification algorithms
# 3. Model Evaluation
# 
# lastly Very important one. :P  Download Submission csv
# Kindly UPVOTE if you find it helful. Thank You:)
# 
# Lets START

# In[ ]:


#Below are packages used in project

import numpy as np # used for numpy data structures viz. 1D and 2D arrays and math functions
import pandas as pd # To extract data from cav file and store in table like sttructure ( dataframe)
import datetime 
import matplotlib.pyplot as plt #get 2D plots
import matplotlib #get 2D plots
import seaborn as sns #get 2D plots
from sklearn.metrics import confusion_matrix #get confusion matrrix viz. actual vs predicted output for model evaluation
import math  #math functions
import xgboost as xgb #to implement XGBoost algorithm
np.random.seed(2019) #Seed is set so that results (random numbers) are repeated even after running code several times
from scipy.stats import skew #statistical analysis
from scipy import stats #statistical analysis

import statsmodels #statistical analysis
from sklearn.metrics import accuracy_score #for model Evaluation
from IPython.display import FileLink #For creating download link to submission csv file

#to get plots below the code blocks
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **1. Data Preprocessing (Includes Exploratory data Analysis and Feature Engineering )**

# In[ ]:


#Read input csv files using pandas functions
train=pd.read_csv("/kaggle/input/titanic/train.csv")
test=pd.read_csv("/kaggle/input/titanic/test.csv")
submission=pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

#We will combine test and train for applying data tranformations 
train['train'] = 1 
test['train'] = 0
data = train.append(test,sort=False ,ignore_index=True)


# In[ ]:


#lets view few records in data
data.head()

#In Data Preprocessing, we will consider each column and transform to make them usable for model application.
#1. Age has lot many missing values, so need to impute Age
#2. From name we can get title for each passenger and use it to impute missing values in Age field
#3. Ticket and cabin column will be preprocessed and converted into categorical column
#4. Fare and Embarked column have 2 and 1 missing value respectively, we will impute them using most repetitive values


# In[ ]:


#lets analyse input data
data.describe()

#It can be seen that we have lot of nulls in Age and survived.
#Survived is expected to have so many nulls as we combined test and train.


# In[ ]:


data.Name.head()


# In[ ]:


#Age is very important variable and we need to impute missing values with best estimates.
#Data is Name variable can be used to extract Titles and then we can impute Age missing values with mean of those groups
#lets see how we will do this.
data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
pd.crosstab(data['Title'], data['Sex'])
data = data.drop('Name',axis=1)

#let's replace a few titles -> "other" and fix a few titles
data['Title'] = np.where((data.Title=='Capt') |(data.Title=='Dr') | (data.Title=='Countess') | (data.Title=='Don') | (data.Title=='Dona')
                        | (data.Title=='Jonkheer') | (data.Title=='Lady') | (data.Title=='Sir') | (data.Title=='Major') | (data.Title=='Rev') | (data.Title=='Col'),'Other',data.Title)
data['Title'] = data['Title'].replace('Ms','Miss')
data['Title'] = data['Title'].replace('Mlle','Miss')
data['Title'] = data['Title'].replace('Mme','Mrs')


# In[ ]:


#Lets observe how age is correlated with Title
sns.boxplot(data = data, x = "Title", y = "Age")


# In[ ]:


data.groupby('Title').Age.mean()


# In[ ]:


#We will impute Age missing values as below
data['Age'] = np.where((data.Age.isnull()) & (data.Title=='Master'),5,
                        np.where((data.Age.isnull()) & (data.Title=='Miss'),22,
                                 np.where((data.Age.isnull()) & (data.Title=='Mr'),32,
                                          np.where((data.Age.isnull()) & (data.Title=='Mrs'),37,
                                                  np.where((data.Age.isnull()) & (data.Title=='Other'),45,data.Age))))) 

#Converting Age to 


# In[ ]:


#Ticket column is quite random but we can use information from it and check how it describes our dependent varaible i.e survided
import string
TypeOfTicket = []
for i in range(len(data.Ticket)):
    ticket = data.Ticket.iloc[i]
    for c in string.punctuation:
                ticket = ticket.replace(c,"")
                splited_ticket = ticket.split(" ")   
    if len(splited_ticket) == 1:
                TypeOfTicket.append('NO')
    else: 
                TypeOfTicket.append(splited_ticket[0])
            
data['TypeOfTicket'] = TypeOfTicket

data.TypeOfTicket.value_counts()
data['TypeOfTicket'] = np.where((data.TypeOfTicket!='NO') & (data.TypeOfTicket!='PC') & (data.TypeOfTicket!='CA') & 
                                (data.TypeOfTicket!='A5') & (data.TypeOfTicket!='SOTONOQ'),'other',data.TypeOfTicket)
data = data.drop('Ticket',axis=1)


# In[ ]:


#Lets observe how age is correlated with Title
sns.lineplot(data = data, x = "TypeOfTicket", y = "Survived")

#It can be seen that typeOfTicket is certainly explaining Survived.
#passengers holding certain type of tickets had better chance of survival


# In[ ]:


data.Embarked.value_counts()
# imputing value of Embarked with 'S' which has most number of occurences
data.Embarked=data.Embarked.fillna("S")


# In[ ]:


data["Cabin"].head()


# In[ ]:


# Replace the Cabin number by first letter and with 'M' if Null 
data["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'M' for i in data['Cabin'] ])


# In[ ]:


sns.countplot(data["Cabin"],order=['A','B','C','D','E','F','G','T','M'])
#it cn be seen that most passengers have unkown cabin number which may suggest that 
#these passengers were not alloted any cabins


# In[ ]:


#lets see Survival probablity vs Cabin 
g = sns.factorplot(y="Survived",x="Cabin",data=data,kind="bar",order=['A','B','C','D','E','F','G','T','M'])
g = g.set_ylabels("Survival Probability")


# It can be seen that passengers not having cabins i.e M have low survival probality compared to others

# In[ ]:


#Fill Fare NA with 0 
data.Fare=data.Fare.fillna(0)


# In[ ]:


#check which columns are having nulls
data.isnull().sum()


# In[ ]:


#Convert all categorical variables to their dummies for model application
data = pd.get_dummies(data)


# In[ ]:


data.columns


# In[ ]:


#split data into train and test
from sklearn.model_selection import train_test_split
trainX, testX, trainY, testY = train_test_split(data[data.Survived.isnull()==False].drop('Survived',axis=1),data.Survived[data.Survived.isnull()==False],test_size=0.30, random_state=2019)


# In[ ]:


#Model_eval will store the results of accuracy scroe of each model
model_eval = pd.DataFrame({'Model': [],'Accuracy Score': []})


# **2. Implement various classification algorithms

# We will apply below classification algorithms in the order and compare their accuracy score
# 1. Logistic Regression 
# 2. Decision Tree
# 3. Random Forest
# 4. SVC
# 5. XGBoost

# In[ ]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(trainX, trainY)
y_pred = model.predict(testX)
from sklearn.metrics import accuracy_score
res = pd.DataFrame({"Model":['LogisticRegression'],
                    "Accuracy Score": [accuracy_score(y_pred,testY)]})
model_eval = model_eval.append(res)
pd.crosstab(testY, y_pred, rownames=['Actual'], colnames=['Predicted'])


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=4)
model.fit(trainX, trainY)
y_pred = model.predict(testX)
from sklearn.metrics import accuracy_score
res = pd.DataFrame({"Model":['DecisionTreeClassifier'],
                    "Accuracy Score": [accuracy_score(y_pred,testY)]})
model_eval = model_eval.append(res)
pd.crosstab(testY, y_pred, rownames=['Actual'], colnames=['Predicted'])


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=2500, max_depth=4)
model.fit(trainX, trainY)
y_pred = model.predict(testX)
from sklearn.metrics import accuracy_score
res = pd.DataFrame({"Model":['RandomForestClassifier'],
                    "Accuracy Score": [accuracy_score(y_pred,testY)]})
model_eval = model_eval.append(res)
pd.crosstab(testY, y_pred, rownames=['Actual'], colnames=['Predicted'])


# In[ ]:


from sklearn.svm import SVC
model = SVC()
model.fit(trainX, trainY)
y_pred = model.predict(testX)
from sklearn.metrics import accuracy_score
res = pd.DataFrame({"Model":['SVC'],
                    "Accuracy Score": [accuracy_score(y_pred,testY)]})
model_eval = model_eval.append(res)
pd.crosstab(testY, y_pred, rownames=['Actual'], colnames=['Predicted'])


# In[ ]:


from xgboost.sklearn import XGBClassifier
model = XGBClassifier(learning_rate=0.0001,n_estimators=2500,
                                max_depth=4, min_child_weight=0,
                                gamma=0, subsample=0.7,
                                colsample_bytree=0.7,
                                scale_pos_weight=1, seed=21,
                                reg_alpha=0.00006)
model.fit(trainX, trainY)
y_pred = model.predict(testX)
from sklearn.metrics import accuracy_score
res = pd.DataFrame({"Model":['XGBClassifier'],
                    "Accuracy Score": [accuracy_score(y_pred,testY)]})
model_eval= model_eval.append(res)
pd.crosstab(testY, y_pred, rownames=['Actual'], colnames=['Predicted'])


# **3. Model Evaluation****

# In[ ]:


#lets see how the models performed overall
model_eval


# In[ ]:


#It can be seen that XGboost fared really good and we can use it to submit our results
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
trainX = data[data.Survived.isnull()==False].drop(['Survived','train'],axis=1)
trainY = data.Survived[data.Survived.isnull()==False]
testX = data[data.Survived.isnull()==True].drop(['Survived','train'],axis=1)
model = XGBClassifier(learning_rate=0.0001,n_estimators=2500,
                                max_depth=4, min_child_weight=0,
                                gamma=0, subsample=0.7,
                                colsample_bytree=0.7,
                                scale_pos_weight=1, seed=21,
                                reg_alpha=0.00006)
model.fit(trainX, trainY)
test = data[data.train==0]
test['Survived'] = model.predict(testX).astype(int)
test = test.reset_index()
test[['PassengerId','Survived']].to_csv("submissionXGB.csv",index=False)
FileLink(r'submissionXGB.csv')

