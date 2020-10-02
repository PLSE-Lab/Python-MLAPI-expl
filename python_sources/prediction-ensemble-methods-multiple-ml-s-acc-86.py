#!/usr/bin/env python
# coding: utf-8

# # Adult Census Income Analysis - Decision TREE, Random Forest, CV, Tuning the model with Ensemble Techniques(Baaging , ADAboost)
# 

# ### A stable and optimized model to predict the income of a given population, which is labelled as <= 50K and >50K. The attributes (predictors) are age, working class type, marital status, gender, race etc.
# #### Following are the steps, 
# #### 1.clean and prepare the data,
# #### 2.Analyze Data,
# #### 3.Label Encoding,
# #### 4.Build a decision tree and Random forest with default hyperparameters,
# #### 5.Build several classifier models to compare, cross validate and for voting classifier model
# #### 6.choose the optimal hyperparameters using grid search cross-validation.
# #### 7.Build optimized Random forest model with tuned hyperparameters from grid search model
# #### 8.Increase Accuracy by Applying Ensemble technique BAGGING to our tuned random forest model
# #### 9.Increase Accuracy by Applying Ensemble technique ADABOOST to our tuned random forest model
# ####  I hope you enjoy this notebook and find it useful!

# ## Clean & Analyze Data,

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
# from sklearn.cross_validation import train_test_split
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


from sklearn.model_selection import cross_val_score


# In[ ]:


from sklearn.model_selection import train_test_split


# In[2]:


data =  pd.read_csv("../input/adult.csv")


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.isnull().sum()


# In[ ]:


# select all categorical variables
df_categorical = data.select_dtypes(include=['object'])

# checking whether any other columns contain a "?"
df_categorical.apply(lambda x: x=="?", axis=0).sum()


# In[ ]:


data[data['workclass'] == '?' ].count()


# In[ ]:


data[data['occupation'] == '?' ].count()


# In[ ]:


data[data['native.country'] == '?' ].count()


# In[ ]:


(1836/32561)/100


#  ### Missing Value % is very insignificant  so we will drop those values

# In[ ]:


data.count()


# In[ ]:


data = data[data["workclass"] != "?" ]


# In[ ]:


data = data[data["occupation"] != "?" ]


# In[ ]:


data = data[data["native.country"] != "?" ]


# In[ ]:


data.count()


# In[ ]:


data.head()


# In[ ]:


data["income"].unique()


# In[ ]:


data["income"] = data["income"].map({'<=50K' : 0, '>50K': 1})
data.head()


# In[ ]:


data["income"].unique()


# ## Label Encoding

# In[ ]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()

catogorical_data = data.select_dtypes(include =['object'])


# In[ ]:


catogorical_data.head()


# In[ ]:


catogorical_data = catogorical_data.apply(le.fit_transform)


# In[ ]:


catogorical_data.head()


# In[ ]:


data = data.drop(catogorical_data.columns, axis=1)
data = pd.concat([data, catogorical_data], axis=1)
data.head()


# In[ ]:


data.info()


# In[ ]:


data['income'] = data['income'].astype('category')


# ## Decision Tree Model with Default parameters

# In[ ]:


x=data.drop('income',axis=1)
y=data['income']
#Train & Test split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state= 476)


# In[ ]:


tree = DecisionTreeClassifier()
model_tree = tree.fit(x_train,y_train)
model_tree


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[ ]:


model_tree = tree.fit(x_train,y_train)
pred_tree = tree.predict(x_test)
a1 = accuracy_score(y_test,pred_tree)
print("The Accuracy of Desicion Tree is ", a1)


# In[ ]:


confusion_matrix(y_test,pred_tree)


# In[ ]:


print(classification_report(y_test, pred_tree))


# ## Random Forest Model with Default parameters

# In[ ]:


rf = RandomForestClassifier()
model_rf = rf.fit(x_train,y_train)
pred_rf = rf.predict(x_test)
a2 = accuracy_score(y_test, pred_rf)
print("The Accuracy of Random Forest is ", a2)


# ## Logistic Regression & KNN model

# In[ ]:


from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()

model_lg = lg.fit(x_train,y_train)
pred_lg = lg.predict(x_test)
a3 = accuracy_score(y_test, pred_lg)
print("The Accuracy of logistic regression is ", a3)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier()


# In[ ]:


model_knn =knn.fit(x_train,y_train) 
pred_knn = knn.predict(x_test)
a4 = accuracy_score(y_test, pred_knn)
print("The Accuracy of KNN is ", a4)


# # Build optimized Random forest model with tuned hyperparameters from grid search model  

# In[ ]:


rf_param = {
    "n_estimators": [25,50,100],
    "criterion" : ["gini"],
    "max_depth" : [3,4,5,6],
    "max_features" : ["auto","sqrt","log2"],
    "random_state" : [123]
}


# In[ ]:


GridSearchCV(rf, rf_param, cv = 5)


# In[ ]:


grid =GridSearchCV(rf, rf_param, cv = 5)


# In[ ]:


grid.fit(x_train,y_train).best_params_


# In[ ]:


rf1 = RandomForestClassifier(criterion = 'gini',
    max_depth = 6,
    max_features = 'auto',
    n_estimators = 100,
    random_state = 123)
model_rf1 = rf1.fit(x_train,y_train)
pred_rf1 = rf1.predict(x_test)
accuracy_score(y_test, pred_rf1)


# # K FOLD Cross Validation

# In[ ]:


cross_val_score(tree,x_train,y_train,scoring= "accuracy", cv=10)


# In[ ]:


cross_val_score(tree,x,y,scoring= "accuracy", cv=5).mean()


# In[ ]:


cross_val_score(rf,x_train,y_train,scoring= "accuracy", cv=5).mean()


# In[ ]:


cross_val_score(lg,x_train,y_train,scoring= "accuracy", cv=5).mean()


# In[ ]:


cross_val_score(knn,x_train,y_train,scoring= "accuracy", cv=5).mean()


# # Voting Classifier model

# In[ ]:


from sklearn.ensemble import VotingClassifier


# In[ ]:


model_vote = VotingClassifier(estimators=[('logistic Regression', lg), ('random forrest', rf), ('knn neighbors', knn),(' decision tree', tree)], voting='soft')
model_vote = model_vote.fit(x_train, y_train)


# In[ ]:


vote_pred = model_vote.predict(x_test)


# In[ ]:


a5 =  accuracy_score(y_test, vote_pred)
print("The Accuracy of voting classifier is ", a5)


# In[ ]:


print(classification_report(y_test, vote_pred))


# # Ensemble Technique Bagging 
# 
# ## Increase Accuracy by Applying Ensemble technique BAGGING to our tuned random forest model

# In[ ]:


from sklearn.ensemble import BaggingClassifier


# In[ ]:


bagg = BaggingClassifier(base_estimator=rf1,n_estimators=15)


# In[ ]:


model_bagg =bagg.fit(x_train,y_train) 
pred_bagg = bagg.predict(x_test)


# In[ ]:


a6 = accuracy_score(y_test, pred_bagg)
print("The Accuracy of BAAGING is ", a6)


# In[ ]:


confusion_matrix(y_test,pred_bagg)


# In[ ]:


print(classification_report(y_test, pred_bagg))


# #  Ensemble Technique  ADA Boost 
# 
# ## Increase Accuracy by Applying Ensemble technique ADABOOST to our tuned random forest model

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier


# In[ ]:


Adaboost = AdaBoostClassifier(base_estimator=rf1, n_estimators=15)


# In[ ]:


model_boost =Adaboost.fit(x_train,y_train) 
pred_boost = Adaboost.predict(x_test)


# In[ ]:


a7 = accuracy_score(y_test, pred_boost)
print("The Accuracy of BOOSTING is ", a7)


# In[ ]:


confusion_matrix(y_test,pred_boost)


# In[ ]:


print(classification_report(y_test, pred_boost))


# In[ ]:




