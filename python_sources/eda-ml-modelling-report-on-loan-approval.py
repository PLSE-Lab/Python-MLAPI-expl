#!/usr/bin/env python
# coding: utf-8

# # EDA & ML Modelling Report on Loan Approval
# 
# Analysis By: NEELESH DUGAR
# 
# Email: dugar.nilesh23@gmail.com
# 
# Mob: +91-7838823636

# ##### 1. We are importing WARNINGS class to suppress any warning

# In[ ]:


import warnings 
warnings.filterwarnings('ignore')


# ##### 2. We are now importing all necessary packages for our report

# In[ ]:


import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import gc
from datetime import datetime 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegressionCV, SGDClassifier, LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier
import lightgbm as lgbm

pd.set_option('display.max_columns', 500)


# ##### 3. We will now load the dataset

# In[ ]:


data = pd.read_csv("../input/LoanApproval.csv")


# ##### 4. Let us have a preview of the data

# In[ ]:


data.drop('Loan_ID', axis=1, inplace= True)
data.head()


# ##### 5. Let us see the no. of rows and columns in this dataset

# In[ ]:


print("Dataset contains -",data.shape[0],"rows and",data.shape[1],"columns")


# ##### 6. Let us see a basic Descriptive Stats of this dataset (Before Cleaning)

# In[ ]:


data.describe(include="all")


# ##### 7. How many null values are there column-wise?

# In[ ]:


data.isnull().sum()


# ##### 8. We will now handle Missing Values and Categorical Variables

# (a) Converting Categorical Variable 'Gender' to Numerical

# In[ ]:


data.Gender[data.Gender == 'Male'] = 1
data.Gender[data.Gender == 'Female'] = 2


# (b) Filling missing values in 'Gender' by random function

# In[ ]:


dict_gender = [1,2]
data.Gender.fillna(np.random.choice(dict_gender), inplace=True)


# (c) Converting Categorical Variable 'Married' to Numerical

# In[ ]:


data.Married[data.Married == 'Yes'] = 1
data.Married[data.Married == 'No'] = 0


# (d) Filling missing values in 'Married' by random function

# In[ ]:


dict_married = [0,1]
data.Married.fillna(np.random.choice(dict_married), inplace=True)


# (e) Filling missing values in 'Dependents' by random function

# In[ ]:


dict_dependents = [0,1,2,3]
data.Dependents.fillna(np.random.choice(dict_dependents), inplace=True)


# (f) Converting Categorical Variable 'Self_Employed' to Numerical

# In[ ]:


data.Self_Employed[data.Self_Employed == 'Yes'] = 1
data.Self_Employed[data.Self_Employed == 'No'] = 0


# (g) Filling missing values in 'Self_Employed' by random function

# In[ ]:


dict_self_employed = [0,1]
data.Self_Employed.fillna(np.random.choice(dict_self_employed), inplace=True)


# (h) Filling missing values in 'LoanAmount' by mean function

# In[ ]:


data.LoanAmount.fillna(data.LoanAmount.mean(), inplace=True)


# (i) Filling missing values in 'Loan_Amount_Term' by random function

# In[ ]:


dict_loan_amount_term = [120,240,360,480]
data.Loan_Amount_Term.fillna(np.random.choice(dict_loan_amount_term), inplace=True)


# (j) Filling missing values in 'Credit_History' by random function

# In[ ]:


dict_credit_history = [0,1]
data.Credit_History.fillna(np.random.choice(dict_credit_history), inplace=True)


# In[ ]:


#Now that we have treated all Missing values, there should be no NULL/NaN values
data.isnull().sum()


# ##### 9. Let us now again see a basic Descriptive Stats of this dataset (After Cleaning)

# In[ ]:


data.describe(include="all")


# ##### 10. Now lets see some visualizations for possible combinations

# In[ ]:


# We have converted datatype of 'Gender' from 'int64'->'object', for Visualization purpose
data.Gender=data.Gender.astype(object)

# We have converted datatype of 'Married' from 'int64'->'object', for Visualization purpose
data.Married=data.Married.astype(object)

# We have converted datatype of 'Self_Employed' from 'int64'->'object', for Visualization purpose
data.Self_Employed=data.Self_Employed.astype(object)

# This command below will show the datatypes of all the columns
data.info()


# In[ ]:


# Now we are creating a new DataFrame named "obj_cols" from the old DataFrame 'data' which will have columns with only OBJECT as its datatype
obj_cols = [*data.select_dtypes('object').columns]
obj_cols1 = obj_cols
obj_cols.remove('Loan_Status')

# Setting up the height & width of the plot we will make below
plt.figure(figsize=(24, 18))

# We are using a For-loop to plot 6 graphs in one plot using "obj_cols" as our refernce dataset
for idx, cols in enumerate(obj_cols):
    
    plt.subplot(3, 3, idx+1)
    
    sns.countplot(cols, data= data, hue='Loan_Status')


# ### Now we will start preparing ML models and compare their scores to select the best accurate model

# (a) Converting Categorical Variable 'Loan_Status' to Numerical

# In[ ]:


data.Loan_Status.replace({'Y': 0, 'N': 1}, inplace= True)
data['Loan_Status']= data.Loan_Status.astype(int)
data.info()


# In[ ]:


# Here we are creating a DataFrame named 'dummies' by using '.get_dummies' function of pandas which will convert all categorical variables to dummy variables
dummies = pd.get_dummies(data, drop_first=True)
dummies.info()

# 'SimpleImputer()' function is used to fill missing values in a DataFrame
SimImp = SimpleImputer()

# We are now creating a new DataFrame named 'train' which will be like our original dataset named 'data' but with no missing values
train= pd.DataFrame(SimImp.fit_transform(dummies), columns=dummies.columns)
train.info()

# We are selecting all the numerical columns and making a new DataFrame with name 'num_cols'
num_cols = [*data.select_dtypes(['Int64', 'Float64']).columns]
num_cols.remove('Loan_Amount_Term')
num_cols.remove('Credit_History')


# In[ ]:


# We are creating a new DataFrame named 'obj_train' from the old Train DataFrame by removing all the numerical columns from it
# obj_train = train.drop(num_cols, axis=1)
# obj_train.info()

# For ML modelling, we'll only use the categorical features for training 
X, y = train.drop('Loan_Status', axis=1), train.Loan_Status

# We will split the data to train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify= y)


# 1. Logistic Regression

# In[ ]:


log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_pred = log_reg.predict(X_test)
log_score = round(log_reg.score(X_train, y_train) * 100, 2)


# 2. Ada Boost Classifier 

# In[ ]:


abc = AdaBoostClassifier()
abc.fit(X_train,y_train)
abc_pred = abc.predict(X_test)
abc_score = round(abc.score(X_train, y_train) * 100, 2)


# 3. LogisticRegressionCV

# In[ ]:


lrcv = LogisticRegressionCV()
lrcv.fit(X_train,y_train)
lrcv_pred = lrcv.predict(X_test)
lrcv_score = round(lrcv.score(X_train, y_train) * 100, 2)


# 4. Stochastic Gradient Descent (SGD) Classifier

# In[ ]:


sgd = SGDClassifier()
sgd.fit(X_train,y_train)
sgd_pred = sgd.predict(X_test)
sgd_score = round(sgd.score(X_train, y_train) * 100, 2)


# 5. XG Boost

# In[ ]:


xgb = XGBClassifier()
xgb.fit(X_train,y_train)
xgb_pred = xgb.predict(X_test)
xgb_score = round(xgb.score(X_train, y_train) * 100, 2)


# ##### We will now check scores of all the models by comparing against each other

# In[ ]:


models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Ada Boost Classifier', 
              'LogisticRegressionCV', 'SGD Classifier', 
              'XG Boost'],
    'Score': [log_score, abc_score, 
              lrcv_score, sgd_score, xgb_score]})

models.sort_values(by='Score', ascending=False)


# ### We can see that the XG Boost Classifier is giving the "Best Accuracy" with a score of 84.93.

# #### I hope this report helps you in understanding a few more concepts of Data Science & Analytics. This is my second kernel posted and a lot more will be coming soon. Stay Tuned!!
# And you can contact me for any queries/collaboration/discussion. My contact details are available at the Top.
# 
# Welcome to Data Science & Machine Learning Club! All the best for future endeavours! :)
