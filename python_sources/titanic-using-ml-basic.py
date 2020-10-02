#!/usr/bin/env python
# coding: utf-8

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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# #### Importing training data

# In[ ]:


data_train = pd.read_csv('../input/train.csv')


# In[ ]:


data_train.head()


# ### Seprating continuous and categorical data

# In[ ]:


numeric_var_names=[key for key in dict(data_train.dtypes) if dict(data_train.dtypes)[key] in ['float64', 'int64', 'float32', 'int32']]
cat_var_names=[key for key in dict(data_train.dtypes) if dict(data_train.dtypes)[key] in ['object']]
print(numeric_var_names)
print(cat_var_names)


# ### For Continuous Data

# In[ ]:


data_num = data_train[numeric_var_names]


# In[ ]:


data_num.info()


# #### Missing value treatment

# In[ ]:


print(data_num.Age.mean())
print(data_num.Age.median())


# #### replacing missing data for Age with mean values

# In[ ]:


data_num.Age = data_num.Age.fillna(data_num.Age.mean())


# In[ ]:


data_num.info()


# #### Outliers treatment

# In[ ]:


data_num.quantile(0.99)


# In[ ]:


data_num.max()


# #### capping data to 99 percentile

# In[ ]:


def outlier_capping(x):
    x = x.clip_upper(x.quantile(0.99))
    x = x.clip_lower(x.quantile(0.01))
    return x

data_num[['Age' ,'Fare']]=data_num[['Age' ,'Fare']].apply(lambda x: outlier_capping(x))


# #### Creating Data SUmmary to better understand data

# In[ ]:


# Use a general function that returns multiple values
def var_summary(x):
    return pd.Series([x.count(), x.isnull().sum(), x.sum(), x.mean(), x.median(),  x.std(), x.var(), x.min(), x.dropna().quantile(0.01), x.dropna().quantile(0.05),x.dropna().quantile(0.10),x.dropna().quantile(0.25),x.dropna().quantile(0.50),x.dropna().quantile(0.75), x.dropna().quantile(0.90),x.dropna().quantile(0.95), x.dropna().quantile(0.99),x.max()], 
                  index=['N', 'NMISS', 'SUM', 'MEAN','MEDIAN', 'STD', 'VAR', 'MIN', 'P1' , 'P5' ,'P10' ,'P25' ,'P50' ,'P75' ,'P90' ,'P95' ,'P99' ,'MAX'])

num_summary=data_num.apply(lambda x: var_summary(x)).T
num_summary


# ### Now handling categorical data

# In[ ]:


data_cat = data_train[cat_var_names]


# In[ ]:


data_cat.head()


# #### dropping ir relevant column NAME as passenger id is a more better alternative to it

# In[ ]:


data_cat = data_cat.drop(columns= ['Name'])


# In[ ]:


data_cat.head()


# In[ ]:


data_cat.info()


# #### dropping variable CABIN as it has 75 % missing data

# In[ ]:


data_cat = data_cat.drop(columns= ['Cabin'])


# #### Removing unwanted characters from ticket variable to make it better usage for analysis

# In[ ]:


data_cat.Ticket = data_cat.Ticket.apply(lambda x: x.split(' ')[1] if x.find(' ')!= -1 else x )


# In[ ]:


data_cat.Ticket.head()


# #### removing errornous entries

# In[ ]:


data_cat.Ticket = data_cat.Ticket[data_cat.Ticket != 'LINE']
data_cat.Ticket = data_cat.Ticket[data_cat.Ticket != 'Basle']


# In[ ]:


data_cat.Ticket = data_cat.Ticket[data_cat.Ticket != 'Basle']


# In[ ]:


data_cat.Ticket = pd.to_numeric(data_cat.Ticket)


# In[ ]:


data_cat.head()


# ### Converting categorical variables to Dummy variables

# In[ ]:


def create_dummies( df, colname ):
    col_dummies = pd.get_dummies(df[colname], prefix=colname)
    col_dummies.drop(col_dummies.columns[0], axis=1, inplace=True)
    df = pd.concat([df, col_dummies], axis=1)
    df.drop( colname, axis = 1, inplace = True )
    return df


# In[ ]:


for c_feature in ['Sex', 'Embarked']:
    data_cat[c_feature] = data_cat[c_feature].astype('category')
    data_cat = create_dummies(data_cat , c_feature )


# In[ ]:


data_cat.head()


# #### Now combining numerical and categorical data

# In[ ]:


data_new = pd.concat([data_num, data_cat], axis =1)


# In[ ]:


data_new.head()


# In[ ]:


get_ipython().run_line_magic('pinfo', 'sns.boxplot')


# #### Exploratory Analysis on Data

# ##### Scatter plots to see if numerical variables can differ Survived and not survived people

# In[ ]:


fig,axes = plt.subplots(2,2,figsize=(16,10))
sns.boxplot(x = data_new.Survived, y = data_new.Age,ax=axes[0,0] )
sns.boxplot(x = data_new.Survived, y = data_new.Fare,ax=axes[0,1])
sns.boxplot(x = data_new.Survived, y = data_new.Ticket,ax=axes[1,0])
sns.boxplot(x = data_new.Survived, y = data_new.PassengerId,ax=axes[1,1])


# ##### factor plots  to see if categorical variables can differ Survived and not surviving people

# In[ ]:


cat_var_names = ['Pclass','SibSp','Parch','Sex_male','Embarked_Q','Embarked_S']


# In[ ]:


for cat_variable in cat_var_names:
    t =sns.catplot(data=data_new,kind='count',x='Survived',col=cat_variable)    


# ### Splitting data for train and test

# In[ ]:


X = data_new


# In[ ]:


from sklearn.model_selection import train_test_split
import statsmodels.formula.api as sm
train_features = X.columns.difference(['Survived'])
train_X, test_X = train_test_split(X, test_size=0.3, random_state=42)
train_X.columns


# In[ ]:


train_X.Survived.shape


# In[ ]:


train_features


# ### Logistic regression

# In[ ]:


logreg = sm.logit(formula='Survived ~ ' + "+".join(train_features), data=train_X)
result = logreg.fit()


# In[ ]:


result.summary2()


# #### Removing less important variables

# In[ ]:


my_formula='Survived ~ ' + "+".join(['Age', 'Embarked_S',
       'Pclass', 'Sex_male', 'SibSp'])


# In[ ]:


logreg = sm.logit(formula=my_formula, data=train_X)
result = logreg.fit()
result.summary2()


# #### Checking model accuracy

# In[ ]:


train_X.head(2)


# In[ ]:


from sklearn import metrics
train_gini = 2*metrics.roc_auc_score(train_X['Survived'], result.predict(train_X)) - 1
print("The Gini Index for the model built on the Train Data is : ", train_gini)

test_gini = 2*metrics.roc_auc_score(test_X['Survived'], result.predict(test_X)) - 1
print("The Gini Index for the model built on the Test Data is : ", test_gini)


# In[ ]:


#for train data
## Intuition behind ROC curve - predicted probability as a tool for separating the '1's and '0's
train_predicted_prob = pd.DataFrame(result.predict(train_X))
train_predicted_prob.columns = ['prob']
train_actual = train_X['Survived']
# making a DataFrame with actual and prob columns
train_predict = pd.concat([train_actual, train_predicted_prob], axis=1)
train_predict.columns = ['actual','prob']
train_predict.head()


# In[ ]:


#for test data
## Intuition behind ROC curve - predicted probability as a tool for separating the '1's and '0's
test_predicted_prob = pd.DataFrame(result.predict(test_X))
test_predicted_prob.columns = ['prob']
test_actual = test_X['Survived']
# making a DataFrame with actual and prob columns
test_predict = pd.concat([test_actual, test_predicted_prob], axis=1)
test_predict.columns = ['actual','prob']
test_predict.head()


# ### Calculating Sensitivity and specificity for various Cut offs

# In[ ]:


## Intuition behind ROC curve - confusion matrix for each different cut-off shows trade off in sensitivity and specificity
roc_like_df = pd.DataFrame()
train_temp = train_predict.copy()

for cut_off in np.linspace(0,1,50):
    train_temp['cut_off'] = cut_off
    train_temp['predicted'] = train_temp['prob'].apply(lambda x: 0.0 if x < cut_off else 1.0)
    train_temp['tp'] = train_temp.apply(lambda x: 1.0 if x['actual']==1.0 and x['predicted']==1 else 0.0, axis=1)
    train_temp['fp'] = train_temp.apply(lambda x: 1.0 if x['actual']==0.0 and x['predicted']==1 else 0.0, axis=1)
    train_temp['tn'] = train_temp.apply(lambda x: 1.0 if x['actual']==0.0 and x['predicted']==0 else 0.0, axis=1)
    train_temp['fn'] = train_temp.apply(lambda x: 1.0 if x['actual']==1.0 and x['predicted']==0 else 0.0, axis=1)
    sensitivity = train_temp['tp'].sum() / (train_temp['tp'].sum() + train_temp['fn'].sum())
    specificity = train_temp['tn'].sum() / (train_temp['tn'].sum() + train_temp['fp'].sum())
    roc_like_table = pd.DataFrame([cut_off, sensitivity, specificity]).T
    roc_like_table.columns = ['cutoff', 'sensitivity', 'specificity']
    roc_like_df = pd.concat([roc_like_df, roc_like_table], axis=0)


# In[ ]:


roc_like_df.head()


# In[ ]:


## Finding ideal cut-off for checking if this remains same in OOS validation
roc_like_df['total'] = roc_like_df['sensitivity'] + roc_like_df['specificity']


# #### Now finding the cut off based on ROC curve ( using just Logistic regression takes cut off default by 0.5 so this method is much better)

# In[ ]:


roc_like_df[roc_like_df['total']==roc_like_df['total'].max()]


# ##### cut off = 0.306

# In[ ]:


test_predict['predicted'] = test_predict['prob'].apply(lambda x: 1 if x > 0.306122 else 0)
train_predict['predicted'] = train_predict['prob'].apply(lambda x: 1 if x > 0.306122 else 0)
sns.heatmap(pd.crosstab(train_predict['actual'], train_predict['predicted']), annot=True, fmt='.0f')
plt.title('Train Data Confusion Matrix')
plt.show()
sns.heatmap(pd.crosstab(test_predict['actual'], test_predict['predicted']), annot=True, fmt='.0f')
plt.title('Test Data Confusion Matrix')
plt.show()


# In[ ]:


print("The overall accuracy score for the Train Data is : ", metrics.accuracy_score(train_predict.actual, train_predict.predicted))
print("The overall accuracy score for the Test Data  is : ", metrics.accuracy_score(test_predict.actual, test_predict.predicted))


# #### checking actual vs predicted values

# In[ ]:


train_predict.head()


# ### Getting data from test file for predicting later based on best Model

# #### reading data from file

# In[ ]:


data_test = pd.read_csv('../input/test.csv')


# In[ ]:


data_test.head()


# In[ ]:


data_test.info()


# #### missing value treatment

# In[ ]:


data_test.Age = data_num.Age.fillna(data_num.Age.mean())


# In[ ]:


data_test.Fare = data_num.Fare.fillna(data_num.Age.mean())


# #### outlier data treatment

# In[ ]:


def outlier_capping(x):
    x = x.clip_upper(x.quantile(0.99))
    x = x.clip_lower(x.quantile(0.01))
    return x

data_test[['Age' ,'Fare']]=data_num[['Age' ,'Fare']].apply(lambda x: outlier_capping(x))


# In[ ]:


data_test = data_test.drop(columns= ['Name'])


# In[ ]:


data_test = data_test.drop(columns= ['Cabin'])


# In[ ]:


data_test.Ticket = data_test.Ticket.apply(lambda x: x.split(' ')[1] if x.find(' ')!= -1 else x )


# In[ ]:


data_test.Ticket = data_test.Ticket[data_test.Ticket != 'LINE']
data_test.Ticket = data_test.Ticket[data_test.Ticket != 'Basle']


# #### Converting categorical data to dummy variables

# In[ ]:


def create_dummies( df, colname ):
    col_dummies = pd.get_dummies(df[colname], prefix=colname)
    col_dummies.drop(col_dummies.columns[0], axis=1, inplace=True)
    df = pd.concat([df, col_dummies], axis=1)
    df.drop( colname, axis = 1, inplace = True )
    return df


# In[ ]:


for c_feature in ['Sex', 'Embarked']:
    data_test[c_feature] = data_test[c_feature].astype('category')
    data_test = create_dummies(data_test , c_feature )


# In[ ]:


data_test.head()


# ### Now implementing  with ML algorithms

# #### RANDOM FOREST

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


data_new = data_new.dropna()


# In[ ]:


X = data_new


# #### train test split

# In[ ]:


train_features = X.columns.difference(['Survived'])
train_X, test_X = train_test_split(X, test_size=0.3, random_state=42)
train_X.columns


# In[ ]:


train_y = train_X.Survived
test_y = test_X.Survived


# In[ ]:


train_X = train_X.drop(columns= ['Survived'])
test_X = test_X.drop(columns= ['Survived'])


# In[ ]:


data_new.head()


# In[ ]:


param_grid = {'min_samples_split':[5,6,7],
              'n_estimators': [100,200,300],
              'min_samples_leaf':[2,3,4,5],
              'max_depth':[3,4,5,]
              }


# #### Paramter Tuning

# In[ ]:


tree3 = GridSearchCV(RandomForestClassifier(), param_grid, cv = 5)
tree3.fit( train_X,train_y )
print(tree3.best_score_)
print(tree3.best_params_)


# #### Model fit

# In[ ]:


model_rf = RandomForestClassifier(min_samples_leaf=2, min_samples_split= 6, n_estimators= 300,max_depth = 5,oob_score=True)  #fine tuning the model
model_rf.fit(train_X, train_y)


# ### Testing Model accuracy

# In[ ]:


print('---------Accuracy score')
print ('Train Accuracy score:' , metrics.accuracy_score(model_rf.predict(train_X), train_y))
print ('Test Accuracy score:' , metrics.accuracy_score(model_rf.predict(test_X), test_y))
train_gini = 2*metrics.roc_auc_score(train_y, model_rf.predict(train_X)) - 1
print('---------Gini score')
print("Train Gini score:", train_gini)

test_gini = 2*metrics.roc_auc_score(test_y, model_rf.predict(test_X)) - 1
print("Test Gini score:", test_gini)


# ### GBM Model

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier


# In[ ]:


data_new.head()


# In[ ]:


param_grid = {'learning_rate':[0.3,0.4],
              'n_estimators':[50,60,70],
              'min_samples_split':[4,5],
              'max_depth':[2,3,4,5]
    
                }


# #### Parameters tuning

# In[ ]:


tree2 = GridSearchCV(GradientBoostingClassifier(), param_grid, cv = 5)
tree2.fit( train_X, train_y )
print(tree2.best_score_)
print(tree2.best_params_)


# In[ ]:


model_gbm = GradientBoostingClassifier( learning_rate =0.3,n_estimators= 70,min_samples_split =4,max_depth = 3)  #fine tuning the model

model_gbm.fit(train_X, train_y)


# ### Testing Model Accuracy

# In[ ]:


print('---------Accuracy score')
print ('Train Accuracy score:' , metrics.accuracy_score(model_gbm.predict(train_X), train_y))
print ('Test Accuracy score:' , metrics.accuracy_score(model_gbm.predict(test_X), test_y))
train_gini = 2*metrics.roc_auc_score(train_y, model_gbm.predict(train_X)) - 1
print('---------Gini score')
print("Train Gini score:", train_gini)

test_gini = 2*metrics.roc_auc_score(test_y, model_gbm.predict(test_X)) - 1
print("Test Gini score:", test_gini)


# ### Using KNN

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


param_grid = {'n_neighbors': np.arange(1,99,2),
              'p':[1,2]
            }


# In[ ]:


tree2 = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
tree2.fit( train_X, train_y )


# In[ ]:


print(tree2.best_score_)
print(tree2.best_params_)


# In[ ]:


model_knn = KNeighborsClassifier(n_neighbors=81,p=1)  #fine tuning the model

model_knn.fit(train_X, train_y)


# ### Testing Model Accuracy

# In[ ]:


print('---------Accuracy score')
print ('Train Accuracy score:' , metrics.accuracy_score(model_knn.predict(train_X), train_y))
print ('Test Accuracy score:' , metrics.accuracy_score(model_knn.predict(test_X), test_y))
train_gini = 2*metrics.roc_auc_score(train_y, model_knn.predict(train_X)) - 1
print('---------Gini score')
print("Train Gini score:", train_gini)

test_gini = 2*metrics.roc_auc_score(test_y, model_knn.predict(test_X)) - 1
print("Test Gini score:", test_gini)


# ### Naive bayes

# In[ ]:


import sklearn.naive_bayes as nb
from sklearn.naive_bayes import MultinomialNB


# In[ ]:


model_nb = MultinomialNB(alpha= 0)
model_nb.fit( train_X, train_y )


# ##### Testing Model Performance

# In[ ]:


print('---------Accuracy score')
print ('Train Accuracy score:' , metrics.accuracy_score(model_nb.predict(train_X), train_y))
print ('Test Accuracy score:' , metrics.accuracy_score(model_nb.predict(test_X), test_y))
train_gini = 2*metrics.roc_auc_score(train_y, model_nb.predict(train_X)) - 1
print('---------Gini score')
print("Train Gini score:", train_gini)

test_gini = 2*metrics.roc_auc_score(test_y, model_nb.predict(test_X)) - 1
print("Test Gini score:", test_gini)


# ### SVM

# In[ ]:


from sklearn.svm import SVC


# In[ ]:


svc = SVC()
param_grid = {
    'kernel': ['rbf'],
    'C': [0.01,0.1, 10, 100, 1000,10000]}


# #### Fine Tuning

# In[ ]:


svc= GridSearchCV(svc, param_grid, cv=5)
svc.fit(train_X, train_y)


# In[ ]:


print( svc.best_params_)
print(svc.best_score_)


# In[ ]:


model = SVC(C = 0.01, kernel= 'rbf')
model.fit(train_X,train_y)


# #### Testing Model Performance

# In[ ]:


print('---------Accuracy score')
print ('Train Accuracy score:' , metrics.accuracy_score(model.predict(train_X), train_y))
print ('Test Accuracy score:' , metrics.accuracy_score(model.predict(test_X), test_y))
train_gini = 2*metrics.roc_auc_score(train_y, model.predict(train_X)) - 1
print('---------Gini score')
print("Train Gini score:", train_gini)

test_gini = 2*metrics.roc_auc_score(test_y, model.predict(test_X)) - 1
print("Test Gini score:", test_gini)


# In[ ]:





# 

# In[ ]:


models = pd.DataFrame({'Model' :['LOGISTIC REGRESSION','RANDOM FOREST','GBM','KNN','Naive Bayes','SVM'],
                       'Score':[ metrics.accuracy_score(test_predict.actual, test_predict.predicted),
                                 metrics.accuracy_score(model_rf.predict(test_X), test_y),
                                 metrics.accuracy_score(model_gbm.predict(test_X), test_y),
                                 metrics.accuracy_score(model_knn.predict(test_X), test_y),
                                 metrics.accuracy_score(model_nb.predict(test_X), test_y),
                                 metrics.accuracy_score(model.predict(test_X), test_y),
                                 ]})
models.sort_values(by = 'Score',ascending=True)


# #### Based on the Testing Accuracy we can select Random Forest ( 81.2 % test accuracy) model as the best model for predicting Whether the passengers survived or not

# ### Predicting for test data now using RF model

# #### reading data from file

# In[ ]:


data_test.head()


# #### prediction using random forest model

# In[ ]:


data_test['Survived'] = model_rf.predict(data_test)


# #### final output

# In[ ]:


data_test.head(10)


# Submission File

# In[ ]:


ids = data_test['PassengerId']
predictions = model_rf.predict(data_test)

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('submission.csv', index=False)


# In[ ]:


output


# In[ ]:




