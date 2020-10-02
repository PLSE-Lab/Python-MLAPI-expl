#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import accuracy_score
import statsmodels.api as sm 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.svm import SVC
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


dataset = pd.read_excel(io='/kaggle/input/bank-loan-modelling/Bank_Personal_Loan_Modelling.xlsx' ,sheet_name='Data')


# In[ ]:


dataset.head()


# * **ID** : Customer ID
# * **Age** : Customer's age in completed years
# * **Experience** : #years of professional experience
# * **Income** : Annual income of the customer (000)
# * **ZIP Code** : Home Address ZIP code.
# * **Family** : Family size of the customer
# * **CCAvg** : Avg. spending on credit cards per month (000)
# * **Education** : Education Level. 1: Undergrad; 2: Graduate; 3:Advanced/Professional
# * **Mortgage** : Value of house mortgage if any. (000)
# * **Personal Loan** : Did this customer accept the personal loan offered in the last campaign?
# * **Securities Account** : Does the customer have a securities account with the bank?
# * **CD Account** : Does the customer have a certificate of deposit (CD) account with the bank?
# * **Online** : Does the customer use internet banking facilities?
# * **Credit card** : Does the customer use a credit card issued byUniversalBank?
# 

# In[ ]:


dataset.info()


# # Correlation

# In[ ]:


colormap = plt.cm.viridis # Color range to be used in heatmap
plt.figure(figsize=(15,15))
plt.title('Dataset Correlation of attributes', y=1.05, size=19)
sns.heatmap(dataset.corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)


# #### There is no strong correlation between any two variables.
# #### There is no strong correlation between any independent variable and class variable.

#  

# # Detection of null data.

# In[ ]:


dataset.isnull().sum()


# In[ ]:


dataset.isnull().values.any()


# **There is no empty value in the data set.**

# ## Data distribution in each feature and target variable.

# In[ ]:


dataset.describe().transpose()


# ## The amount of unique values in each column.

# In[ ]:


dataset.apply(lambda x: len(x.unique()))


# ## Unique values in each column.

# In[ ]:


for col in dataset.columns:
    print(col + ' Col Unique values: ', dataset[col].unique(), '\n\n')


# # Number of people with zero mortgage

# In[ ]:


zero_mortgage = 0
for zero in dataset['Mortgage']:
    if zero == 0:
        zero_mortgage += 1
print('Number of people with zero mortgage ', zero_mortgage)


# ## Number of people with zero credit card spending per month

# In[ ]:


cc_avg = 0
for avg in dataset['CCAvg']:
    if avg == 0:
        cc_avg += 1
print('Number of people with zero credit card spending per month: ', cc_avg)


# ### **Categorical columns in the dataset:**
# 
# * **Personal Loan**
# 
# 
# * **Securities Account**
# 
# 
# * **CD Account**
# 
# 
# * **Online Col**
# 
# 
# * **CreditCard**

# ### **Value counts of all categorical columns:**

# In[ ]:


categorical_col = ['Personal Loan', 'Securities Account', 'CD Account', 'Online Col', 'CreditCard']

for col in categorical_col:
    val = 0
    for value in col:
        val += 1
    print('Value count of ' + col + ':', val)


# ## Univariate
# * **Age Column**

# In[ ]:


plt.figure(figsize=(16,4))
sns.set_color_codes()
sns.countplot(dataset["Age"])


# In[ ]:


plt.figure(figsize=(18,5))
sns.set_color_codes()
sns.distplot(dataset["Age"])


# ## Bivariate
# * **Age and Personal Loan**

# In[ ]:


plt.figure(figsize=(12,4))
sns.set_color_codes()
sns.barplot(dataset["Age"],dataset["Personal Loan"])


# In[ ]:


plt.figure(figsize=(12,4))
sns.set_color_codes()
sns.boxplot(y=dataset["Age"],x=dataset["Personal Loan"])


#  

# In[ ]:


X = dataset.drop(columns = ['ID', 'Personal Loan'])
y = dataset['Personal Loan']


#  

# ## Best features

# In[ ]:


kbest = SelectKBest(k=5)
k_best_features = kbest.fit_transform(X, y)
list(dataset.columns[kbest.get_support (indices=True)])


# In[ ]:


dataset.corrwith(dataset["Personal Loan"]).abs().nlargest(5)


# In[ ]:


X = dataset.drop(columns = ['ID','Age','Experience','ZIP Code', 'Family', 'Education','Personal Loan','Securities Account', 'Online','CreditCard']).values
y = dataset['Personal Loan'].values


# ## Dividing the dataset

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# ## Scaling

# In[ ]:


sc = MinMaxScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)


# ## Modelling
# * **A small function for easier and easier use of the model. It can make your work quite easy when using more than one model.**

# In[ ]:


def model_evaluate(model, test):
    y_pred = model.predict(test)
    print('Metrics: \n', classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm, cmap = 'Blues', fmt = '', annot = True)

    plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual values"   , fontdict = {'size':14}, labelpad = 10)
    plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)


#  

# * **Logistic Regression**

# In[ ]:


model = LogisticRegression(random_state = 0)
model.fit(x_train, y_train)

model_evaluate(model, x_test)


#  

# * **Support Vector Classifier**

# In[ ]:


model = SVC(kernel = 'rbf')
model.fit(X_train, y_train)

model_evaluate(model, X_test)


#  

# * **K Neighbors Classifier**

# In[ ]:


model = KNeighborsClassifier(n_neighbors = 7, metric = 'euclidean')
model.fit(X_train, y_train)

model_evaluate(model, X_test)


#  

# * **Gaussian Naive Bayes**

# In[ ]:


model = GaussianNB()
model.fit(X_train, y_train)

model_evaluate(model, X_test)


#  

# * **XGBoost Classifier**

# In[ ]:


xgb = XGBClassifier()
xgb.fit(X_train, y_train)

model_evaluate(xgb, X_test)


# In[ ]:


crossVal= cross_val_score(estimator = xgb, X = X_train, y = y_train, cv = 10)
print('XGBoost Accuracy: ', crossVal.mean())
print('XGBoost Std: ', crossVal.std())


# In[ ]:


skf = StratifiedKFold(n_splits=10)
scores = cross_val_score(xgb, X_train, y_train, cv=skf)
print("scores:\n{}".format(scores))
print("average score:\n{}".format(scores.mean()))


# In[ ]:


params = [{'learning_rate':[0.1,0.01],
           'colsample_bytree':[1,3],
           'gamma':[0,1],
           'reg_alpha':[2,3],
           'reg_lambda':[1,2,4,16],
           'n_estimators':[50,100,150],
           'colsample_bylevel':[1,2],
           'missing':[False, True],
           'subsample':[1,2],
           'base_score':[0.2,0.5]
           }
    ]
from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(estimator = xgb,
                  param_grid = params,
                  scoring = 'accuracy',
                  cv = 10,
                  n_jobs = -1)
grid_search = gs.fit(x_train, y_train)
best_result = grid_search.best_score_
best_params = grid_search.best_params_
print('Best_Result', best_result)
print('Best_Params', best_params)


# In[ ]:


xgb = XGBClassifier(base_score = 0.2, colsample_bylevel = 1, colsample_bytree = 1, gamma = 0, learning_rate = 0.1, missing = True, n_estimators = 150, reg_alpha = 3, reg_lambda = 1, subsample = 1)
xgb.fit(X_train, y_train)

model_evaluate(xgb, X_test)


#  

# ### **The bank's expectation from the dataset is to take out loans to its debtor customers and turn them into loan customers. Therefore, it is desired to create a new marketing campaign by making inferences about the connection between the variables.**
# 
# ### **It seems that the XGBoost algorithm successfully accomplishes the necessary correlation.**
