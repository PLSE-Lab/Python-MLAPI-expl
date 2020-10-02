#!/usr/bin/env python
# coding: utf-8

# # BUSINESS PROBLEM STATEMENT
# - Assume you are working with MyBank. The Bank executed a campaign to cross-sell Personal Loans. As part of their Pilot Campaign, 20000 customers were sent campaigns through email, sms, and direct mail.
# - They were given an offer of Personal Loan at an attractive interest rate of 12% and processing fee waived off if they respond within 1 Month. 
# - 2512 customer expressed their interest and are marked as Target = 1
# - Many Demographics and Behavioural variables provided. 
# - You have to build a Model using Supervised Learning Technique to finds profitable segments to target for cross-selling personal loans. Make necessary assumptions where required.
# 

# # EDA

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[ ]:


data_train=pd.read_csv('../input/PL_XSELL.csv')


# In[ ]:


data_train.shape


# In[ ]:


data_train.head().T


# - since customer id and random number has no meaning we can drop those column

# In[ ]:


data_train=data_train.drop(['random','CUST_ID'],axis=1)


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


plt.pie(data_train['GENDER'].value_counts(),labels=data_train['GENDER'].value_counts().index,autopct='%1.1f%%')


# - Above chart shows male account holders are high when compared to others

# In[ ]:


plt.subplots(figsize=(16,8))
sns.heatmap(data_train.corr()[data_train.corr().abs()>0.1],annot=True)


# - From the above you can understand there is very small amount of correlation for target coumn with independent variable.
# - Multi-collinearity exists 

# In[ ]:


# convert factors to labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data_train['GENDER'] = le.fit_transform(data_train['GENDER'])
data_train['AGE_BKT'] = le.fit_transform(data_train['AGE_BKT'])
data_train['OCCUPATION'] = le.fit_transform(data_train['OCCUPATION'])
data_train['ACC_TYPE'] = le.fit_transform(data_train['ACC_TYPE'])
data_train['OCCUPATION'] = le.fit_transform(data_train['OCCUPATION'])


# In[ ]:


data_train.info()


# In[ ]:


data_train['ACC_OP_DATE']=pd.to_datetime(data_train['ACC_OP_DATE'])


# In[ ]:


data_train['ACC_OP_YR']=data_train['ACC_OP_DATE'].dt.year


# - age and age_bucket express same data

# In[ ]:


from sklearn.preprocessing import StandardScaler
x=data_train.drop(['ACC_OP_DATE'],axis=1)
scaler=StandardScaler().fit(x)
y=pd.DataFrame(scaler.transform(x),columns=x.columns)
y.boxplot(vert=False,figsize=(15,10))


# - Many outliers are there except some column

# In[ ]:


data_train.describe().T


# In[ ]:


# Load libraries
from matplotlib import pyplot
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# - Feature Selection based on VIF

# In[ ]:


X=data_train.drop(['AGE','TARGET','ACC_OP_DATE'],axis=1)
Y=data_train[['TARGET']]


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
x_features = list(X)

x_features.remove('TOT_NO_OF_L_TXNS')
x_features.remove('ACC_OP_YR')


data_mat = X[x_features].as_matrix()
data_mat.shape
vif = [ variance_inflation_factor( data_mat,i) for i in range(data_mat.shape[1]) ]
vif_factors = pd.DataFrame()
vif_factors['column'] = list(x_features)
vif_factors['vif'] = vif
print(vif_factors)


# In[ ]:


X=data_train[x_features]
validation_size = 0.30
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)


# In[ ]:


# Test options and evaluation metric
num_folds = 10
seed = 7
scoring = 'accuracy'
# Spot-Check Algorithms
models = []
models.append(('Logistic', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('GB',GradientBoostingClassifier()))


# In[ ]:


# evaluate each model in turn
results = []
names = []
model_comp=pd.DataFrame(columns=['Model','Test Accuracy','Std.Dev'])
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    model_comp=model_comp.append([{'Model':name, 'Test Accuracy':cv_results.mean(), 'Std.Dev':cv_results.std()}],ignore_index=True)
    
model_comp


# In[ ]:


model=DecisionTreeClassifier(max_depth=15)
model=model.fit(X_train,Y_train)
model.score(X_train,Y_train)


# In[ ]:


model.score(X_validation,Y_validation)


# - the model is underfit,so regularization is done to become best model

# In[ ]:


model=DecisionTreeClassifier(max_depth=5)
model=model.fit(X_train,Y_train)
model.score(X_train,Y_train)


# In[ ]:


model.score(X_validation,Y_validation)


# In[ ]:


from IPython.display import Image  
from sklearn import tree
from os import system

train_char_label = ['yes', 'no']
Loan_campaign_File = open('Loan_campaign_tree.dot','w')
dot_data = tree.export_graphviz(model, out_file=Loan_campaign_File, feature_names = list(X), class_names = list(train_char_label))

Loan_campaign_File.close()


# importance of features in the tree building ( The importance of a feature is computed as the 
#(normalized) total reduction of the criterion brought by that feature. It is also known as the Gini importance )

print (pd.DataFrame(model.feature_importances_, columns = ["Imp"], index = X.columns).sort_values(by='Imp',ascending=False))


# - To Create tree diagram,use this link https://dreampuf.github.io/GraphvizOnline/
# - Open .dot file generated and copy it in this url, you will get the tree diagram

# - To view the tree diagram use the below link
# - https://www.kaggle.com/dineshmk594/loan-campaign#download.png

# - Because I have given max_depth=5 only five levels are there(exclude the root)
# - Feature selection used for Decision Tree is based on the model importance variables

# In[ ]:




