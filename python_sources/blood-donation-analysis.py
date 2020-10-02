#!/usr/bin/env python
# coding: utf-8

# **About the Project:**
# 
# One of the interesting aspects about blood is that it is not a typical commodity. First, there is the perishable nature of blood. Grocery stores face the dilemma of perishable products such as milk, which can be challenging to predict accurately so as to not lose sales due to expiration. Blood has a shelf life of approximately 42 days according to the American Red Cross (Darwiche, Feuilloy et al. 2010). However, what makes this problem more challenging than milk is the stochastic behavior of blood supply to the system as compared to the more deterministic nature of milk supply. Whole blood is often split into platelets, red blood cells, and plasma, each having their own storage requirements and shelf life. For example, platelets must be stored around 22 degrees Celsius, while red blood cells 4 degree Celsius, and plasma at -25 degrees Celsius. Moreover, platelets can often be stored for at most 5 days, red blood cells up to 42 days, and plasma up to a calendar year.
# 
# Amazingly, only around 5% of the eligible donor population actually donate (Linden, Gregorio et al. 1988, Katsaliaki 2008). This low percentage highlights the risk humans are faced with today as blood and blood products are forecasted to increase year-on-year. This is likely why so many researchers continue to try to understand the social and behavioral drivers for why people donate to begin with. The primary way to satisfy demand is to have regularly occurring donations from healthy volunteers.

# **Aim Of Project:**
# 
# To build a model which can identify who is likely to donate blood again.
# 
# Models implemented:
# 
# Logistic Regression
# 
# Suport Vector Machine
# 
# Random Forest
# 
# Decision Tree
# 
# MLP Classifier

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#Importing library for visualization
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

#Importing all the required model for model comparision
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

#Importing library for splitting model into train and test and for data transformation
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


#Filter the unwanted warning
import warnings
warnings.simplefilter("ignore")


# **Reading the Data**

# In[ ]:


path = '/kaggle/input/predicting-blood-analysis/blood-train.csv'
path1 = '/kaggle/input/predicting-blood-analysis/blood-test.csv'
blood = path + 'blood-train.csv'
blood1 = path1 + 'blood-test.csv'


# In[ ]:


train = pd.read_csv(path)
test = pd.read_csv(path1)


# In[ ]:


train.head()


# In[ ]:


#Printing the train and test size
print("Train Shape : ",train.shape)
print("Test Shape : ",test.shape)


# In[ ]:


#Printing first five rows of data
train.head()


# In[ ]:


#Counting the number of people who donated and not donated
train["Made Donation in March 2007"].value_counts()


# In[ ]:


#Storing dependent variable in Y
Y=train.iloc[:,-1]
Y.head()


# In[ ]:


#Printing last 5 rows
train.tail()


# In[ ]:


#Removing Unnamed: 0 columns
old_train=train
train=train.iloc[:,1:5]
test=test.iloc[:,1:5]


# In[ ]:


#Printing firsr  rows
train.head()


# In[ ]:


#Merging both train and test data
df=pd.merge(train,test)


# In[ ]:


df.head()


# In[ ]:


#Setting the independent variable and dependent variable
X=df.iloc[:,:]
X.head()


# **Data Exploration**

# In[ ]:


# Statistics of the data
train.describe()


# In[ ]:


#Boxplot for Months since Last Donation
plt.figure(figsize=(20,10)) 
sns.boxplot(y="Months since Last Donation",data=old_train)


# We see from the above boxplot that the maximum people have donated blood in nearby 10 months.

# In[ ]:


#Correlation between all variables [Checking how different variable are related]
corrmat=X.corr()
f, ax = plt.subplots(figsize =(9, 8)) 
sns.heatmap(corrmat, ax = ax, cmap ="YlGnBu", linewidths = 0.1,fmt = ".2f",annot=True) 


# In[ ]:


#Printing all unique value for Month Since Last donation
train["Months since Last Donation"].unique()


# **Feature Engineering**
# 
# Volume donated is also a good feature to know wether the donor will donate or not.

# In[ ]:


#Creating new variable for calculating how many times a person have donated
X["Donating for"] = (X["Months since First Donation"] - X["Months since Last Donation"])


# In[ ]:


#Seeing first five rows of the DataFrame
X.head()


# In[ ]:


#Correlation between all variables
corrmat=X.corr()
f, ax = plt.subplots(figsize =(9, 8)) 
sns.heatmap(corrmat, ax = ax, cmap ="YlGnBu", linewidths = 0.1,fmt = ".2f",annot=True) 


# Since Total Volume Donated (c.c.) have the very high correlation with other variables so we are dropping the variable.

# In[ ]:


#Dropping the unnecessary column
X.drop([ 'Total Volume Donated (c.c.)'], axis=1, inplace=True)


# In[ ]:


X.head()


# In[ ]:


#Shape of independent variable
X.shape


# **Feature Transformation**

# In[ ]:


#Feature Scaling
from sklearn.preprocessing import StandardScaler
scale=StandardScaler()

#Fitting and transforming data
X=scale.fit_transform(X)


# In[ ]:


train=X[:576]


# In[ ]:


train.shape


# In[ ]:


test=X[576:]


# In[ ]:


Y=Y[:576]


# In[ ]:


Y.shape


# **Model Building**

# In[ ]:


#Splitting into train and test set
xtrain,xtest,ytrain,ytest=train_test_split(train,Y,test_size=0.2,random_state=0)


# **StepsTo Follow**
# 
# 
# - Create the object
# - Do the necessary hyperparameter tuning
# - Fit the model
# - Predict the test set
# - Compute roc_auc_score
# - Repeat above step for all model
# - Compare roc_auc_Score of all model and choose the best model

# **Logistic regression**

# In[ ]:


#Building the model
logreg = LogisticRegression(random_state=7)
#Fitting the model
logreg.fit(xtrain,ytrain)


# In[ ]:


#Predicting on the test data
pred=logreg.predict(xtest)


# In[ ]:


accuracy_score(pred,ytest)


# In[ ]:


#Printing the roc_auc_score
roc_auc_score(pred,ytest)


# **Support Vector Machine**

# In[ ]:


### SVC classifier
SVMC = SVC(probability=True)
#Fitting the model
SVMC.fit(train,Y)


# In[ ]:


#Predicting on the test data
pred=SVMC.predict(xtest)


# In[ ]:


accuracy_score(pred, ytest)


# In[ ]:


#Printing the confusion matrix
confusion_matrix(pred,ytest)


# In[ ]:


#Printing the roc auc score
roc_auc_score(pred,ytest)


# **Random Forest**

# In[ ]:


#Buildin the model
RFC = RandomForestClassifier()
#Fitting the model
RFC.fit(xtrain,ytrain)


# In[ ]:


#Predicting the test data result
pred=RFC.predict(xtest)


# In[ ]:


#Printing the confusion matrix
confusion_matrix(pred,ytest)


# In[ ]:


accuracy_score(pred, ytest)


# In[ ]:


#Printingthe roc auc score
roc_auc_score(pred,ytest)


# **Decision Tree**

# In[ ]:


#Building the model
model=DecisionTreeClassifier(max_leaf_nodes=4,max_features=3,max_depth=15)


# In[ ]:


#Fitting the model
model.fit(xtrain,ytrain)


# In[ ]:


#Predicting the test data
pred=model.predict(xtest)


# In[ ]:


accuracy_score(pred, ytest)


# In[ ]:


#printing the confusion matrix
confusion_matrix(pred,ytest)


# In[ ]:


#Printing accuracy score
accuracy_score(pred,ytest)


# In[ ]:


#Printing roc auc score
roc_auc_score(pred,ytest)


# **MLP Classifier**

# In[ ]:


#Building the Model
clf_neural = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(25,),random_state=1)
clf_neural.fit(train, Y)


# In[ ]:


#Predicting from the fitted model on test data
print('Predicting...\nIn Test Data')
predicted = clf_neural.predict(xtest)


# In[ ]:


#printing confusion matrix
confusion_matrix(predicted,ytest)


# In[ ]:


#Printing roc auc score
roc_auc_score(pred,ytest)


# In[ ]:


accuracy_score(pred, ytest)


# In[ ]:




