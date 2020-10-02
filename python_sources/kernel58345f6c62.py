#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


get_ipython().run_line_magic('matplotlib', 'inline')
import os
import warnings
warnings.filterwarnings('ignore')


# ## Load and check data

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.describe()


# In[ ]:


train.info()


# Combine train and test datasets for further work with features

# In[ ]:


dataset = pd.concat((train, test))


# Check for null and missing values

# In[ ]:


dataset = dataset.fillna(np.nan)
dataset.isnull().sum()


# Age and Cabin features have an important part of missing values. Missing values in Survived correspond to the join testing dataset(Survived column doesn't exist in test set ). 

# ## Numerical values

# In[ ]:


g = sns.heatmap(train[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")


# Only Fare feature seems to have a significative correlation with the survival probability.

# In[ ]:


g = sns.factorplot(x="SibSp",y='Survived',data=train,kind="bar", size = 6, 
)
g.despine(left=True)
g = g.set_ylabels("survival probability")


# It seems that passengers having a lot of siblings/spouses have less chance to survive

# In[ ]:


g  = sns.factorplot(x="Parch",y="Survived",data=train,kind="bar", size = 6,)
g.despine(left=True)
g = g.set_ylabels("survival probability")


# Small families have more chance to survive, more than single (Parch 0), medium (Parch 3,4) and large families (Parch 5,6 ).

# In[ ]:


dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean())


# In[ ]:


g = sns.FacetGrid(train, col='Survived')
g = g.map(sns.distplot, "Age")


# Distributions are not the same in the survived and not survived subpopulations. There is a peak corresponding to young passengers, that have survived.

# We have only one missing value. I decided to fill it with the mean value

# In[ ]:


dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].mean())


# In[ ]:


sns.distplot(dataset['Fare'])
print(dataset["Fare"].skew())


# Fare distribution is very skewed. This can lead to overweigth very high values in the model. We need to use log transformation to Fare

# In[ ]:


dataset["Fare"] = dataset["Fare"].map(lambda x: np.log(x) if x > 0 else 0)


# In[ ]:


sns.distplot(dataset['Fare'])
print(dataset["Fare"].skew())


# Skewness is reduced after the log transformation

# ## Categorical values

# In[ ]:


g = sns.barplot(x="Sex",y="Survived",data=train)
g = g.set_ylabel("Survival Probability")


# In[ ]:


train[["Sex","Survived"]].groupby('Sex').mean()


# Male have less chance to survive than Femal

# In[ ]:


g = sns.factorplot(x="Pclass",y="Survived",data=train,kind="bar", size = 6)
g.despine(left=True)
g = g.set_ylabels("survival probability")


# First class passengers have more chance to survive than second class and third class passengers.

# We have two missing values , i decided to fill them with the most fequent value of "Embarked" (S)

# In[ ]:


dataset["Embarked"] = dataset["Embarked"].fillna("S")


# In[ ]:


g = sns.factorplot(x="Embarked", y="Survived",  data=train,
                   size=6, kind="bar")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# It seems that passenger coming from Cherbourg (C) have more chance to survive.

# All of these categorical variables are important for prediction

# ## Feature engineering
# 

# ### Name

# The Name feature contains information on passenger's title.

# In[ ]:


dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]
dataset["Title"] = pd.Series(dataset_title)
dataset["Title"].value_counts()


# There is 17 titles in the dataset, most of them are very rare and we can group them in 4 categories

# In[ ]:


dataset["Title"] = dataset["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
dataset["Title"] = dataset["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
dataset["Title"] = dataset["Title"].astype(int)


# In[ ]:


g = sns.countplot(dataset["Title"])
g = g.set_xticklabels(["Master","Miss/Ms/Mme/Mlle/Mrs","Mr","Rare"])


# In[ ]:


g = sns.factorplot(x="Title",y="Survived",data=dataset,kind="bar")
g = g.set_xticklabels(["Master","Miss-Mrs","Mr","Rare"])
g = g.set_ylabels("survival probability")


# Women are more likely to survive than men

# ### Family size

#  Family size feature which is the sum of SibSp , Parch and 1 (including the passenger). I decided to created 4 categories of family size.

# In[ ]:


dataset["FamilySize"] = dataset["SibSp"] + dataset["Parch"] + 1
dataset['Single'] = dataset['FamilySize'].map(lambda s: 1 if s == 1 else 0)
dataset['SmallF'] = dataset['FamilySize'].map(lambda s: 1 if  s == 2  else 0)
dataset['MedF'] = dataset['FamilySize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
dataset['LargeF'] = dataset['FamilySize'].map(lambda s: 1 if s >= 5 else 0)


# ### Cabin

# In[ ]:


dataset['Cabin'].describe()


# In[ ]:


dataset['Cabin'].isnull().sum()


# The Cabin feature column contains 295 values and 10014 missing values. Replace the Cabin number by the type of cabin 'X' if not

# In[ ]:


dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin'] ])


# ### Ticket

# I decided to replace the Ticket feature column by the ticket prefixe

# In[ ]:


dataset["Ticket"].head()


# In[ ]:


Ticket = []
for i in list(dataset.Ticket):
    if not i.isdigit() :
        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0])
    else:
        Ticket.append("X")
        
dataset["Ticket"] = Ticket
dataset["Ticket"].head()


# ### Encoding categorical variables

# In[ ]:


dataset = pd.get_dummies(dataset, columns = ["Sex", "Title"])
dataset = pd.get_dummies(dataset, columns = ["Embarked"], prefix="Em")
dataset = pd.get_dummies(dataset, columns = ["Ticket"], prefix="T")
dataset["Pclass"] = dataset["Pclass"].astype("category")
dataset = pd.get_dummies(dataset, columns = ["Pclass"],prefix="Pc")
dataset = pd.get_dummies(dataset, columns = ["Cabin"],prefix="Cabin")


# In[ ]:


dataset.drop(labels = ["Name"], axis = 1, inplace = True)
dataset.drop(labels = ["PassengerId"], axis = 1, inplace = True)


# In[ ]:


dataset.head(10)


# ## Splitting the training data

# In[ ]:


X_train = dataset[:train.shape[0]]
X_test = dataset[train.shape[0]:]
y = train['Survived']


# In[ ]:


X_train = X_train.drop(labels='Survived', axis=1)
X_test = X_test.drop(labels='Survived', axis=1)


# ## Feature Scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


headers_train = X_train.columns
headers_test = X_test.columns


# In[ ]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Datasets after scaling

# In[ ]:


pd.DataFrame(X_train, columns=headers_train).head()


# In[ ]:


pd.DataFrame(X_test, columns=headers_test).head()


# ## Modeling

# As models, I used the most popular methods such as Logistic Regression, KNN, SVM, Random Forest and etc. In addition, a grid approach was applied to each model to optimize parameters and cross validation to check the model prediction on test data.

# ### Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score


# In[ ]:


cv = StratifiedShuffleSplit(n_splits = 10, test_size = .25, random_state = 0 )
accuracies = cross_val_score(LogisticRegression(solver='liblinear'), X_train, y, cv  = cv)
print ("Cross-Validation accuracy scores:{}".format(accuracies))
print ("Mean Cross-Validation accuracy score: {}".format(round(accuracies.mean(),5)))


# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


C_vals = [0.2,0.3,0.4,0.5,1,5,10]

penalties = ['l1','l2']

cv = StratifiedShuffleSplit(n_splits = 10, test_size = .25)


param = {'penalty': penalties, 'C': C_vals}

logreg = LogisticRegression(solver='liblinear')
 
grid = GridSearchCV(estimator=LogisticRegression(), 
                           param_grid = param,
                           scoring = 'accuracy',
                            n_jobs =-1,
                           cv = cv
                          )

grid.fit(X_train, y)


# In[ ]:


print (grid.best_score_)
print (grid.best_params_)
print(grid.best_estimator_)


# In[ ]:


logreg_grid = grid.best_estimator_
logreg_score = round(logreg_grid.score(X_train,y), 4)
logreg_score


# ### KNN

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


k_range = range(1,31)
weights_options=['uniform','distance']
param = {'n_neighbors':k_range, 'weights':weights_options}
cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)
grid = GridSearchCV(KNeighborsClassifier(), param,cv=cv,verbose = False, n_jobs=-1)

grid.fit(X_train,y)


# In[ ]:


print (grid.best_score_)
print (grid.best_params_)
print(grid.best_estimator_)


# In[ ]:


knn_grid= grid.best_estimator_
knn_score = round(knn_grid.score(X_train,y), 4)
knn_score


# ### Naive Bayes

# In[ ]:


from sklearn.naive_bayes import GaussianNB

cv = StratifiedShuffleSplit(n_splits = 10, test_size = .25, random_state = 0 )
accuracies = cross_val_score(GaussianNB(), X_train, y, cv  = cv)
print ("Cross-Validation accuracy scores:{}".format(accuracies))
print ("Mean Cross-Validation accuracy score: {}".format(round(accuracies.mean(),5)))

bayes_score = round(accuracies.mean(), 4)
bayes_score


# ### SVM

# In[ ]:


from sklearn.svm import SVC

C = [0.1, 1,1.5]
gammas = [0.001, 0.01, 0.1]
kernels = ['rbf', 'poly', 'sigmoid']
param_grid = {'C': C, 'gamma' : gammas, 'kernel' : kernels}

cv = StratifiedShuffleSplit(n_splits=10, test_size=.25, random_state=8)

grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=cv)
grid_search.fit(X_train,y)


# In[ ]:


print(grid_search.best_score_)
print(grid_search.best_params_)
print(grid_search.best_estimator_)


# In[ ]:


svm_grid = grid_search.best_estimator_
svm_score = round(svm_grid.score(X_train,y), 4)
svm_score


# ### Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
max_depth = range(1,31)
max_feature = [21,22,23,24,25,26,28,29,30,'auto']
criterion=["entropy", "gini"]

param = {'max_depth':max_depth, 
         'max_features':max_feature, 
         'criterion': criterion}

cv=StratifiedShuffleSplit(n_splits=10, test_size =.25, random_state=9)

grid = GridSearchCV(DecisionTreeClassifier(), 
                                param_grid = param, 
                                 verbose=False, 
                                 cv=cv,
                                n_jobs = -1)
grid.fit(X_train, y) 


# In[ ]:


print( grid.best_params_)
print (grid.best_score_)
print (grid.best_estimator_)


# In[ ]:


dectree_grid = grid.best_estimator_
dectree_score = round(dectree_grid.score(X_train,y), 4)
dectree_score


# In[ ]:


import os     
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydot
from IPython.display import Image
dot_data = StringIO()  
export_graphviz(dectree_grid, out_file=dot_data,  
                feature_names=headers_train,  class_names = (["Survived" if int(i) is 1 else "Not_survived" for i in y.unique()]),
                filled=True, rounded=True,
                proportion=True,
                special_characters=True)  
(graph,) = pydot.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())


# ### Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
n_estimators = [140,145,150];
max_depth = range(1,10);
criterions = ['gini', 'entropy'];
cv = StratifiedShuffleSplit(n_splits=10, test_size=.25, random_state=10)


parameters = {'n_estimators':n_estimators,
              'max_depth':max_depth,
              'criterion': criterions
              
        }
grid = GridSearchCV(estimator=RandomForestClassifier(max_features='auto'),
                                 param_grid=parameters,
                                 cv=cv,
                                 n_jobs = -1)
grid.fit(X_train,y)


# In[ ]:


print (grid.best_score_)
print (grid.best_params_)
print (grid.best_estimator_)


# In[ ]:


rf_grid = grid.best_estimator_
rf_score = round(rf_grid.score(X_train,y), 4)
rf_score


# ### XGBoost

# In[ ]:


from xgboost import XGBClassifier


# In[ ]:


XGBClassifier = XGBClassifier(colsample_bytree = 0.3, subsample = 0.7, reg_lambda = 1)

#colsample_bytree = [0.3, 0.5]
#subsample = [0.7, 1]
n_estimators = [400, 450]
max_depth = [2,3,4]
learning_rate = [0.01, 0.1]
reg_alpha = [0, 0.0001, 0.0005]
#reg_lambda = [0.3, 1, 5]
cv = StratifiedShuffleSplit(n_splits=10, test_size=.25, random_state=13)


parameters = {#'colsample_bytree':colsample_bytree,
              #'subsample': subsample,
              'n_estimators':n_estimators,
              'max_depth':max_depth,
              'learning_rate':learning_rate,
              'reg_alpha':reg_alpha,
              #'reg_lambda':reg_lambda
        }
grid = GridSearchCV(estimator=XGBClassifier,
                                 param_grid=parameters,
                                 cv=cv,
                                 n_jobs = -1)
grid.fit(X_train,y)


# In[ ]:


print (grid.best_score_)
print (grid.best_params_)
print (grid.best_estimator_)


# In[ ]:


xgb_grid = grid.best_estimator_
xgb_score = round(xgb_grid.score(X_train,y), 4)
xgb_score


# I also implement classifications neural network using the keras library

# ### Keras

# In[ ]:


from keras.models import Sequential
from keras.layers.core import Dense
import keras
from keras.optimizers import *
from keras.initializers import *


# In[ ]:


NN_train = dataset[:train.shape[0]]
NN_test = dataset[train.shape[0]:]
NN_y = train['Survived'].values
#NN_y = NN_y.reshape(-1,1)

NN_train = NN_train.drop(labels='Survived', axis=1)
NN_test = NN_test.drop(labels='Survived', axis=1)


# In[ ]:


sc = StandardScaler()
NN_train = sc.fit_transform(NN_train.values)
NN_test = sc.transform(NN_test.values)


# In[ ]:


n_cols = NN_train.shape[1]


# In[ ]:


model = Sequential()


# Input shape was added according to the number of features of train dataset after preprocessing. Adam was used as the optimaseira, binary_accuracy was used as the metrics

# In[ ]:


model.add(Dense(128, activation='relu', input_shape=(n_cols,)))

model.add(Dense(128, activation="elu"))
model.add(Dense(256, activation="elu"))
model.add(Dense(128, activation="elu"))
model.add(keras.layers.Dropout(0.3))

model.add(Dense(512, activation="elu"))
model.add(Dense(1024, activation="elu"))
model.add(Dense(512, activation="elu"))
model.add(keras.layers.Dropout(0.3))

model.add(Dense(1024, activation="elu"))
model.add(Dense(2048, activation="elu"))
model.add(Dense(1024, activation="elu"))
model.add(keras.layers.Dropout(0.3))

model.add(Dense(512, activation="elu"))
model.add(Dense(1024, activation="elu"))
model.add(Dense(512, activation="elu"))
model.add(keras.layers.Dropout(0.3))

model.add(Dense(256, activation="elu"))
model.add(Dense(128, activation="elu"))
model.add(Dense(64, activation="elu"))
model.add(Dense(32, activation="elu"))
model.add(keras.layers.Dropout(0.3))

model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer="Adam", loss='binary_crossentropy', metrics=["binary_accuracy"])


# In[ ]:


model.summary()


# In[ ]:


#from keras.callbacks import EarlyStopping
#early_stopping_monitor = EarlyStopping(patience = 10)


# Cross validation has also been applied.

# In[ ]:


model_result = model.fit(NN_train, NN_y, batch_size=100, epochs=200, validation_split = 0.25, shuffle = True)


# In[ ]:


keras_score = round(max(model_result.history["val_binary_accuracy"]), 4)
keras_score


# ## Results

# In[ ]:


results = pd.DataFrame({
    'Model': ['Logistic Regression', 'KNN', 'Naive Bayes','Support Vector Machines',   
               'Decision Tree', 'Random Forest', 'XGBoost', 'Keras'],
    'Score': [logreg_score, knn_score, bayes_score, 
              svm_score, dectree_score, rf_score, xgb_score, keras_score]})
results.sort_values(by='Score', ascending=False)
#print df.to_string(index=False)


# You can see that the highest accuracy scores are XGBoost, Keras. Other models have similar accuracy of the estimate. While Naive Bayes predicts the worst of all other models. 

# **Although the logistic regression did not show the best result on the training data, it is the one that has the highest submission accuracy. Therefore, I decided to use it as the main model**

# ## Prediction

# In[ ]:


predict = logreg_grid.predict(X_test)


# In[ ]:


my_submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predict})
my_submission.to_csv('submission__logreg.csv', index=False)

