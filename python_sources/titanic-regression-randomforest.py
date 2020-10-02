#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import scale
import statsmodels.api as sm


# In[ ]:


# To have proper visibility of all columns and rows setting these options
pd.set_option("display.max_rows", 999)
pd.set_option("display.max_columns", 999)


# In[ ]:


# 01. Import Data
f_train = '../input/titanic/train.csv'
df_train = pd.read_csv(f_train, index_col=0)

f_new = '../input/titanic/test.csv'
df_test = pd.read_csv(f_new, index_col=0)


# In[ ]:


df_train.head()


# In[ ]:


df_train.info()


# Age Cabin and Embarked ha null values. We need to impute them in the test set.
# We also need to check the test set.

# In[ ]:


df_test.info()


# The test dataset has Age, Fare, and Cabin with null values.
# We replace Age with median or mean in both dataset. When Cabin is null, it means No Cabin, so we replace it with None.
# For Embarked, we need to check the data and maybe replace with the mode value.
# 

# In[ ]:


df_train.describe()


# In[ ]:


# Age has some null columns, we replace it with the median value of 28
df_train['Age'] = df_train['Age'].replace(np.nan,28 )


# In[ ]:


df_train.describe()


# In[ ]:


df_test.describe()


# In[ ]:


# Age has some null columns, we replace it with the median value of 27
df_test['Age'] = df_test['Age'].replace(np.nan,27 )


# In[ ]:


# We replace Fare with the mean value
df_test['Fare'] = df_test['Fare'].replace(np.nan,35 )


# In[ ]:


# For cabin we replace the numbers with just the letters. This replcaes null values too with 'n'

df_train['Cabin'] = df_train['Cabin'].astype(str).str[0]
df_test['Cabin'] = df_test['Cabin'].astype(str).str[0]


# In[ ]:


df_train.head()


# In[ ]:


#df_train['Cabin'] = df_train['Cabin'].replace(np.nan,'None' )
#df_test['Cabin'] = df_test['Cabin'].replace(np.nan,'None' )


# In[ ]:


df_train['Embarked'].value_counts()


# In[ ]:


# We replace Embarked in Train set with the mode value
df_train['Embarked'] = df_train['Embarked'].replace(np.nan,'S' )


# In[ ]:


df_train['Cabin'].value_counts()


# In[ ]:


df_test['Cabin'].value_counts()


# In[ ]:


# Since some cabin types are very less, we can club them together to Other or letter 'O'
df_train['Cabin'] = df_train['Cabin'].replace('F', 'O')
df_train['Cabin'] = df_train['Cabin'].replace('G', 'O')
df_train['Cabin'] = df_train['Cabin'].replace('T', 'O')
df_train['Cabin'] = df_train['Cabin'].replace('A', 'O')
df_train['Cabin'].value_counts()


# In[ ]:


# We club the cabin types for test set too
df_test['Cabin'] = df_test['Cabin'].replace('F', 'O')
df_test['Cabin'] = df_test['Cabin'].replace('G', 'O')
df_test['Cabin'] = df_test['Cabin'].replace('T', 'O')
df_test['Cabin'] = df_test['Cabin'].replace('A', 'O')
df_test['Cabin'].value_counts()


# In[ ]:


#Checking the survival rate
df_train['Survived'].value_counts()


# 

# In[ ]:


df_train.info()


# In[ ]:


df_test.info()


# In[ ]:


df_train


# In[ ]:


# I will now drop the Name and Ticket column as it does not have any use.
df_train = df_train.drop(columns=['Name','Ticket'])
df_test = df_test.drop(columns=['Name','Ticket'])


# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# We now split the data to Test and train set.

# In[ ]:


# Putting feature variable to X
X = df_train.drop(['Survived'],axis=1)

# Putting response variable to y
y = df_train['Survived']


# In[ ]:


X.head()


# In[ ]:


X.tail()


# In[ ]:


y.head()


# In[ ]:


print(X.shape)
print(y.shape)


# In[ ]:


# we create dummy values for the categorical columns
cat_var = ['Sex', 'Cabin', 'Embarked']
dummy = pd.get_dummies(X[cat_var], drop_first=True)
#Adding the results to master dataframe
X = pd.concat([X, dummy], axis=1)

# Dropping the categorical variables for which dummy variables are present
X = X.drop(cat_var, axis=1)

X.head()


# In[ ]:


X.info()


# In[ ]:


X['SibSp'].value_counts()


# In[ ]:


X['Parch'].value_counts()


# In[ ]:


X.info()


# In[ ]:


X.tail()


# In[ ]:


# scaling the features

# storing column names in cols
cols = X.columns
num_cols = ['Pclass', 'Age', 'SibSp', 'Fare','Parch']
X[num_cols] = pd.DataFrame(scale(X[num_cols]))
#X.columns = cols
#X.columns


# In[ ]:


X


# In[ ]:


X.info()


# In[ ]:


# somehow the last row has null values, so we drop it
X = X.drop(X.index[890])
X.info()


# In[ ]:


y = y.drop(y.index[890])


# In[ ]:


# We will now do the split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# In[ ]:


X_train.shape


# In[ ]:


X_train.info()


# In[ ]:





# In[ ]:


# Logistic regression model
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()


# In[ ]:


# We use RFE for feature selection
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


# In[ ]:


from sklearn.feature_selection import RFE
rfe = RFE(logreg, 6)             # running RFE with 10 variables as output
rfe = rfe.fit(X_train, y_train)


# In[ ]:


list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[ ]:


col = X_train.columns[rfe.support_]


# In[ ]:


X_train.columns[~rfe.support_]


# In[ ]:


#Assesing model with StatsModel
X_train_sm = sm.add_constant(X_train[col])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[ ]:


# Getting the predicted values on the train set
y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]


# In[ ]:


y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# Creating a dataframe with the actual churn flag and the predicted probabilities

# In[ ]:


y_train_pred_final = pd.DataFrame({'Survived':y_train.values, 'Survived_Prob':y_train_pred})
y_train_pred_final['PassengerId'] = y_train.index
y_train_pred_final.head()


# In[ ]:


y_train_pred_final['predicted'] = y_train_pred_final.Survived_Prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.head()


# In[ ]:


# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Survived_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[ ]:


# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[ ]:


# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# In[ ]:


# From curve we see that the optimum cutoff is 0.35
y_train_pred_final['final_predicted'] = y_train_pred_final.Survived_Prob.map( lambda x: 1 if x > 0.35 else 0)

y_train_pred_final.head()


# In[ ]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_train_pred_final.Survived, y_train_pred_final.final_predicted)


# In[ ]:


confusion2 = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final.final_predicted )
confusion2


# Here accuracy is 78% with the Logistic Regression model.
# 
# We will try nce with Decision Tree

# In[ ]:


df_train.head()


# In[ ]:


# Putting feature variable to X
X = df_train.drop('Survived',axis=1)

# Putting response variable to y
y = df_train['Survived']


# In[ ]:


# we create dummy values for the categorical columns
cat_var = ['Sex', 'Cabin', 'Embarked']
dummy = pd.get_dummies(X[cat_var], drop_first=True)
#Adding the results to master dataframe
X = pd.concat([X, dummy], axis=1)

# Dropping the categorical variables for which dummy variables are present
X = X.drop(cat_var, axis=1)

X.head()


# In[ ]:


# we create dummy values for the categorical columns for the test set too
dummy = pd.get_dummies(df_test[cat_var], drop_first=True)
#Adding the results to master dataframe
df_test = pd.concat([df_test, dummy], axis=1)

# Dropping the categorical variables for which dummy variables are present
df_test = df_test.drop(cat_var, axis=1)

df_test.head()


# In[ ]:


from sklearn.tree import DecisionTreeClassifier

# Fitting the decision tree with default hyperparameters, apart from
# max_depth which is 5 so that we can plot and read the tree.
dt_default = DecisionTreeClassifier(max_depth=3)
dt_default.fit(X, y)


# In[ ]:


pip install pydotplus


# In[ ]:


pip install graphviz


# In[ ]:


# Importing required packages for visualization
from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydotplus, graphviz

# Putting features
features = list(X.columns[0:])
features


# In[ ]:


# Importing random forest classifier from sklearn library
from sklearn.ensemble import RandomForestClassifier

# Running the random forest with default parameters.
rfc = RandomForestClassifier()


# In[ ]:


# fit
rfc.fit(X_train,y_train)


# In[ ]:


# Making predictions
predictions = rfc.predict(X_test)


# In[ ]:


# Importing classification report and confusion matrix from sklearn metrics
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score


# In[ ]:


# Let's check the report of our default model
print(classification_report(y_test,predictions))


# In[ ]:


# Printing confusion matrix
print(confusion_matrix(y_test,predictions))


# In[ ]:


print(accuracy_score(y_test,predictions))


# In[ ]:


# Create the parameter grid based on the results of random search 
param_grid = {
    'max_depth': [3,4,8,10],
    'min_samples_leaf': range(3, 5, 10),
    'min_samples_split': range(3, 5, 10),
    'n_estimators': [5, 10, 20, 50, 100], 
    'max_features': [2, 3, 4, 5, 10]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1,verbose = 1)


# In[ ]:


# Fit the grid search to the data
grid_search.fit(X_train, y_train)


# In[ ]:


# printing the optimal accuracy score and hyperparameters
print('We can get accuracy of',grid_search.best_score_,'using',grid_search.best_params_)


# In[ ]:


# model with the best hyperparameters
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(bootstrap=True,
                             max_depth=3,
                             min_samples_leaf=3, 
                             min_samples_split=3,
                             max_features=3,
                             n_estimators=5)


# In[ ]:


# fit
rfc.fit(X_train,y_train)


# In[ ]:


# predict
predictions = rfc.predict(X_test)


# In[ ]:


# evaluation metrics
from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


print(classification_report(y_test,predictions))


# In[ ]:


print(confusion_matrix(y_test,predictions))


# In[ ]:


Y_pred = rfc.predict(df_test)
rfc.score(X_train, y_train)
acc_random_forest = round(rfc.score(X_train, y_train) * 100, 2)
acc_random_forest


# In[ ]:


df_test.head()


# In[ ]:


Y_pred


# In[ ]:


submission = df_test["Pclass"]


# In[ ]:


submission.head()


# In[ ]:


submission = pd.DataFrame({"Pclass": df_test["Pclass"],"Survived": Y_pred})
#submission.to_csv('../output/submission.csv', index=False)
submission.head()


# In[ ]:


# Need to drop Pclass as it is not needed
subission = submission.drop(columns="Pclass")


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv('../submission.csv', index=False)


# In[ ]:




