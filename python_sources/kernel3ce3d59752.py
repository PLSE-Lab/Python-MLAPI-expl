#!/usr/bin/env python
# coding: utf-8

# **Employee Attrition Submission**
# 
# Firstly thanking Consulting and Analytics Club IITG for organising the Summer Analytics Course and this Kaggle competition. This being my first completely attempted Kaggle competition was really insightful and thought me a lot. I have tried to explain my notebook at my best and tried to keep the code simple and clear. I am open to any suggestions you have and would love to learn from you.

# **Importing files**

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


# **Loading Data**

# In[ ]:


train_data = pd.read_csv("../input/summeranalytics2020/train.csv",index_col= "Id")
test_data = pd.read_csv("../input/summeranalytics2020/test.csv",index_col= "Id")

X = train_data.drop(['Attrition'],axis='columns').copy()
y = train_data["Attrition"].copy()
X_test = test_data.copy()


# **Spliting the data to create validation set**
# 
# Test size is suggested to be between 0.2-0.3 .

# In[ ]:


from sklearn.model_selection import train_test_split

X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size=0.3,random_state=0)


# In[ ]:


# Get list of categorical variables
s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables:")
print(object_cols)

# Get list of numerical variables
numerical_cols = [cname for cname in X_train.columns if 
                X_train[cname].dtype in ['int64', 'float64']]

print((numerical_cols))


# In[ ]:


X_train.nunique()


# In[ ]:


#"Behaviour" is a redundant column hence droping it.

X_train.drop(['Behaviour'],inplace=True,axis='columns')
X_valid.drop(['Behaviour'],inplace=True,axis='columns')
X_test.drop(['Behaviour'],inplace=True,axis='columns')

numerical_cols.remove('Behaviour')
print(numerical_cols)


# **Visualization of Data**
# 
# Since there were no missing values in the data now checking for the skewness of the data to apply appropriate transformations.

# In[ ]:


import seaborn as sns
from matplotlib import pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')

#all numerical columns except performance rating because there was some error in producing it graph

cols = ['Age', 'DistanceFromHome', 'Education', 'EmployeeNumber', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobSatisfaction', 'MonthlyIncome', 'NumCompaniesWorked', 'PercentSalaryHike', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager', 'CommunicationSkill']

for col in cols:
    fig,ax = plt.subplots(1,3)
    sns.distplot(X_train[col],ax=ax[0])
    sns.distplot(np.sqrt(X_train[col]),ax=ax[1])
    sns.distplot(np.log(X_train[col]+1),ax=ax[2])


# In[ ]:


#depending upon the skewness of data appling appropriate transformations 

sqrt_col = ['DistanceFromHome','TotalWorkingYears','NumCompaniesWorked','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']
log_col = ['MonthlyIncome','PercentSalaryHike']

X_train[sqrt_col] = np.sqrt(X_train[sqrt_col])
X_train[log_col] = np.log(np.log(X_train[log_col]))

X_valid[sqrt_col] = np.sqrt(X_valid[sqrt_col])
X_valid[log_col] = np.log(np.log(X_valid[log_col]))

X_test[sqrt_col] = np.sqrt(X_test[sqrt_col])
X_test[log_col] = np.log(np.log(X_test[log_col]))


# In[ ]:


#checking distribution of both the train and test data ( and found them almost same)
cols = ['Age', 'DistanceFromHome', 'Education', 'EmployeeNumber', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobSatisfaction', 'MonthlyIncome', 'NumCompaniesWorked', 'PercentSalaryHike', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager', 'CommunicationSkill']

for col in cols:
    fig,ax = plt.subplots(1,2)
    sns.distplot(X_train[col],ax=ax[0])
    sns.distplot(X_test[col],ax=ax[1])


# **Defining models and Data preprocessing**
# 
# To make the data clean and clear Pipeline is used.
# Multiple models are defined with multiple grids appropriately and GridSearchCV is used for hyperparameter tuning.

# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.naive_bayes import BernoulliNB,GaussianNB
from sklearn.calibration import CalibratedClassifierCV

# Preprocessing for numerical data
numerical_transformer = StandardScaler()

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore',sparse=False)),
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, object_cols),
    ])

model1 = LogisticRegression(random_state=0)
model2 = DecisionTreeClassifier(random_state=0)
model3 = RandomForestClassifier(random_state=0)
model4 = SVC(random_state=0,probability=True)
model5 = XGBClassifier(random_state=0)
model6 = GaussianNB()
model7 = CalibratedClassifierCV(LinearSVC(max_iter=1000000,random_state=0,loss='hinge'))

lcm = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('model',model1)
])

dtm = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('model',model2)
])

rfcm = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('model',model3)
])

svcm = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('model',model4)
])

xgbcm = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('model',model5)
])

nbcm = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('model',model6)
])

lsvcm = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('model',model7)
])

grid_val1 = [
    {'model__penalty':['l2'],'model__tol':[0.001],'model__C':[1],'model__solver':['liblinear','saga','newton-cg','lbfgs'],'model__max_iter':[100]},
    {'model__penalty':['l1'],'model__tol':[0.001],'model__C':[1],'model__solver':['liblinear','saga'],'model__max_iter':[100]},
    {'model__penalty':['elasticnet'],'model__tol':[0.001],'model__C':[0.1],'model__solver':['saga'],'model__max_iter':[100],'model__l1_ratio':[0.25,0.5,0.75]},
]

grid_val2 = [
    {'model__criterion':['gini','entropy'],'model__max_depth':[10],'model__max_features':['sqrt','log2','none']}
]

grid_val3 = [
    {'model__criterion':['gini'],'model__max_depth':[6],'model__n_estimators':[20]}
]

grid_val4 = [
    {'model__C':[0.1],'model__kernel':['poly'],'model__degree':[2],'model__gamma':[0.1],'model__tol':[0.01]}
]

grid_val5 = [
    {'model__max_depth':[5],'model__objective':['binary:logistic'],'model__eval_metric':['auc'],'model__booster':['gblinear'],'model__learning_rate':[0.2]}
]

grid_val6 = [
    {'model__var_smoothing':[1e-09,1e-10,1e-08]}
]

grid_val7 =[
    {'model__method':['sigmoid']}
]

fm = GridSearchCV(lcm,param_grid = grid_val1,scoring = 'roc_auc')

# sfm = SelectFromModel(fm,threshold=0.001)

# sfm.fit(X_train,y_train)

# X_train = sfm.transform(X_train)
# X_valid = sfm.transform(X_valid)
# X_test = sfm.transform(X_test)

fm.fit(X_train,y_train)

print(fm.best_params_)

y_valid_pred = fm.predict_proba(X_valid)[:,1]
y_train_pred = fm.predict_proba(X_train)[:,1]

print(roc_auc_score(y_train,y_train_pred))
print(roc_auc_score(y_valid,y_valid_pred))

y_test_pred = fm.predict_proba(X_test)[:,1]


# In[ ]:


output = pd.DataFrame({'Id': X_test.index,
                       'Attrition': y_test_pred})
output.to_csv('submission.csv',index=False)


# In[ ]:


#Checking if the output is in the desired format

output.head()


# In[ ]:





# In[ ]:




