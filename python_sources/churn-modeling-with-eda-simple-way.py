#!/usr/bin/env python
# coding: utf-8

# # Churn Modeling
# 
# This data set contains details of a bank's customers and the target variable is a binary variable reflecting the fact whether the customer left the bank (closed his account) or he continues to be a customer.
# 
# Here we have 13 feature columns and **Exited** is a target column.
# 
# **Row Numbers:-**
# Row Numbers from 1 to 10000.
# 
# 
# **CustomerId:-**
# Unique Ids for bank customer identification.
# 
# **Surname:-**
# Customer's last name.
# 
# **CreditScore:-**
# Credit score of the customer.
# 
# **Geography:-**
# The country from which the customer belongs(Germany/France/Spain).
# 
# **Gender:-**
# Male or Female(Female/Male).
# 
# **Age:-**
# Age of the customer.
# 
# **Tenure:-**
# Number of years for which the customer has been with the bank.
# 
# **Balance:-**
# Bank balance of the customer.
# 
# **NumOfProducts:-**
# Number of bank products the customer is utilising.
# 
# **HasCrCard:-**
# Binary Flag for whether the customer holds a credit card with the bank or not(0=No,1=Yes).
# 
# **IsActiveMember:-**
# Binary Flag for whether the customer is an active member with the bank or not(0=No,1=Yes).
# 
# **EstimatedSalary:-**
# Estimated salary of the customer in Dollars.
# 
# **Exited:-**
# Binary flag 1 if the customer closed account with bank and 0 if the customer is retained(0=No,1=Yes).

# # 1. Import Liberary

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # 2. Exploratory Data Analysis

# In[ ]:


df = pd.read_csv("../input/churn-modelling/Churn_Modelling.csv")


# In[ ]:


df.shape


# In[ ]:


df.columns


# In[ ]:


df.head()


# In[ ]:


df.drop_duplicates(inplace=True)


# In[ ]:


df.shape


# We don't have duplicate values.

# In[ ]:


Catagorical_Features = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']


# **1. CreditScore**

# In[ ]:


sns.violinplot(x=df['Exited'], y=df['CreditScore'])


# In[ ]:


sns.boxplot(x=df['Exited'], y=df['CreditScore'])


# CreditScore is lessthen 400 then high chance that account will closed.

# In[ ]:





# **2. Geography**

# In[ ]:


sns.barplot(x=df['Geography'], y=df['Exited'])


# German person have almost double probablity of close account compare to other.

# In[ ]:





# **3. Gender**

# In[ ]:


sns.barplot(x=df['Gender'], y=df['Exited'])


# female close more account compare to male.

# In[ ]:





# **4. Age**

# In[ ]:


sns.violinplot(x=df['Exited'], y=df['Age'])


# In[ ]:


sns.boxplot(x=df['Exited'], y=df['Age'])


# age is between 30 to 40 then less chance of close account but age between 41 to 50 then more chance of close account.

# In[ ]:





# **5. Tenure**

# In[ ]:


sns.barplot(x=df['Tenure'], y=df['Exited'])


# In[ ]:





# **6. Balance**

# In[ ]:


sns.kdeplot(data=df['Balance'],shade=True)


# In[ ]:


sns.violinplot(x='Exited', y='Balance', data=df)


# if balance is between 90000 to 150000 then it is more chance the account is close but balance is 0 then less chance for close account.

# In[ ]:





# **7. NumOfProducts**

# In[ ]:


sns.barplot(x=df['NumOfProducts'], y=df['Exited'])


# customer utilize more then 2 bank products then there are higher chance that customer close account.

# In[ ]:





# **8. HasCrCard**

# In[ ]:


sns.barplot(x='HasCrCard', y='Exited', data=df)


# we don't show any mejor diffrence who have cradit card or not.

# In[ ]:





# **9. IsActiveMember**

# In[ ]:


sns.barplot(x = df['IsActiveMember'], y= df['Exited'])


# here higher chance of close account who is not active member.

# In[ ]:





# **10. EstimatedSalary**

# In[ ]:


sns.boxplot(x=df['Exited'], y=df['EstimatedSalary'])


# In[ ]:


sns.violinplot(x=df['Exited'], y=df['EstimatedSalary'])


# we don't get any usefull information from EstimatedSalary column.

# In[ ]:





# # 3. Data Preprocesing

# In[ ]:


df.head(5)


# **convert catagorical value in numeric value**

# In[ ]:


df = pd.get_dummies(df,columns=['Geography','Gender'],drop_first=True)


# In[ ]:


df.head(5)


# Here RowNumber, CustomerId and Surname is not use in churn modeling

# In[ ]:


df.drop(columns=['RowNumber', 'CustomerId', 'Surname'],inplace=True)


# In[ ]:


df.head()


# In[ ]:


df.shape


# here we have 12 columns and we find correlation between them. Threshold = 0.85 

# In[ ]:


def person_corr(df):
    df_dup = df.copy()
    df_corr = df.corr() # Find Correlation of dataframe
    col_name = df_corr.columns
    col = list()
    for i in df_corr:
        for j in col_name:
            if (df_corr[i][j]>0.0) & (i!=j) & (i not in col): # set threshold 0.85
                col.append(j)
    df_dup.drop(columns=col,inplace=True)
    return df_dup


# In[ ]:


df_diff_col = person_corr(df)


# here we don't have correlated columns.

# In[ ]:


df.corr()


# In[ ]:





# # 4. Train Model

# In[ ]:


X_train = df.drop(columns=['Exited'])
y_train = df['Exited']


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=42)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold


# In[ ]:


# Cross validate model with Kfold stratified cross val
kfold = StratifiedKFold(n_splits=10)


# In[ ]:


# Modeling step Test differents algorithms 
random_state = 2
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(LinearDiscriminantAnalysis())

cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, X_train, y = y_train, scoring = "accuracy", cv = kfold, n_jobs=4))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",
"RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis"]})

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res)
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")


# Here RandomForest, ExtraTrees, GradientBoosting have high score so we use that classifier

# In[ ]:


#ExtraTrees 
ExtC = ExtraTreesClassifier()


## Search grid for optimal parameters
ex_param_grid = {
#               "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[50,100,200,300],
              "criterion": ["gini"]}
etc_folds = []
etcc = []
for i in range(5,18,2):

    kfold =i
    gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)

    gsExtC.fit(X_train,y_train)

    ExtC_best = gsExtC.best_estimator_
    etc_folds.append(gsExtC.best_score_)
    etcc.append(ExtC_best)
# Best score
gsExtC.best_score_


# In[ ]:





# In[ ]:


# RFC Parameters tunning 
RFC = RandomForestClassifier()


## Search grid for optimal parameters
rf_param_grid = {
#               "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[50,100,200,300],
              "criterion": ["gini"]}

rfc_folds =[]
rfcc = []
for i in range(5,18,2):
    kfold = i

    gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)

    gsRFC.fit(X_train,y_train)

    RFC_best = gsRFC.best_estimator_
    
    rfc_folds.append(gsRFC.best_score_)
    rfcc.append(RFC_best)
# Best score
gsRFC.best_score_


# In[ ]:





# In[ ]:


# Gradient boosting tunning

GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [50,100,200,300,400],
              'learning_rate': [0.1, 0.05, 0.01,10],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
              }
gbdt_folds = []
gbdtt = []
for i in range(3,10,2):
    kfold = i
    gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

    gsGBC.fit(X_train,y_train)

    GBC_best = gsGBC.best_estimator_
    
    gbdt_folds.append(gsGBC.best_score_)
    gbdtt.append(GBC_best)
# Best score
gsGBC.best_score_


# In[ ]:





# In[ ]:


votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),('gbc',GBC_best)], voting='soft', n_jobs=4)


# In[ ]:


votingC = VotingClassifier(estimators=[('etc', etcc[etc_folds.index(max(etc_folds))]),('rfc', rfcc[rfc_folds.index(max(rfc_folds))]), ('gbdt',gbdtt[gbdt_folds.index(max(gbdt_folds))])], voting='soft', n_jobs=4)

votingC = votingC.fit(X_train, y_train)

exited_pred = pd.Series(votingC.predict(X_test), name="Exited_pred")

y_test.reset_index(drop=True, inplace=True)

results = pd.concat([y_test, exited_pred],axis=1)

results.to_csv("churn_modling.csv",index=False)


# In[ ]:




