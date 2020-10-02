#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


import pandas as pd
import numpy as np
import datetime as dt

#data viz
#for better viz
import plotly.express as px
import plotly.graph_objects as go

#for quick viz
import seaborn as sns

#ml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report
from imblearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


# In[ ]:


df = pd.read_csv('/kaggle/input/auto-insurance-claims-data/insurance_claims.csv')
df.head()


# # EDA and cleaning

# In[ ]:


df.isnull().sum()


# There are no null values in the dataset.

# In[ ]:


# removing column named _c39 as it contains only null values

df = df.drop(['_c39'], axis = 1)


# In[ ]:


df.info()


# There doesn't seem to be need to change column data types except for `policy_bind_date` which we will convert to `datetime` type

# In[ ]:


df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'])


# In[ ]:


df.describe().T


# In[ ]:


for i in df.columns:
    if df[i].dtype == 'object':
        print(i, ":", df[i].nunique())


# We will drop some of the columns: `policy_state', 'policy_csl', 'incident_date', 'incident_state', 'incident_city' and 'incident_location`

# In[ ]:


drop_columns = ['policy_state', 'policy_csl', 'incident_date', 'incident_state', 'incident_city', 'incident_location']
df = df.drop(drop_columns, axis = 1)
df.head()


# In[ ]:


for i in df.columns:
    if df[i].dtype == 'object':
        print(i, ":", df[i].nunique())


# `fraud_reported` is going to be our target column. We will convert it to 1 and 0.

# In[ ]:


df['fraud_reported'] = df['fraud_reported'].str.replace('Y', '1')
df['fraud_reported'] = df['fraud_reported'].str.replace('N', '0')
df['fraud_reported'] = df['fraud_reported'].astype(int)


# In[ ]:


df['fraud_reported'].unique()


# `policy_deductable`: In an insurance policy, the deductible is the amount paid out of pocket by the policy holder before an insurance provider will pay any expenses.

# In[ ]:


sns.countplot(df['fraud_reported'])


# Our data is very imbalanced.

# We will visualize the data and see if there is any feature which might influence the claims

# In[ ]:


def vis_data(df, x, y = 'fraud_reported', graph = 'countplot'):
    if graph == 'hist':
        fig = px.histogram(df, x = x)
        fig.update_layout(title = 'Distribution of {x}'.format(x = x))
        fig.show()
    elif graph == 'bar':
      fig = px.bar(df, x = x, y = y)
      fig.update_layout(title = '{x} vs. {y}'.format(x = x, y = y))
      fig.show()
    elif graph == 'countplot':
      a = df.groupby([x,y]).count()
      a.reset_index(inplace = True)
      no_fraud = a[a['fraud_reported'] == 0]
      yes_fraud = a[a['fraud_reported'] == 1]
      trace1 = go.Bar(x = no_fraud[x], y = no_fraud['policy_number'], name = 'No Fraud')
      trace2 = go.Bar(x = yes_fraud[x], y = yes_fraud['policy_number'], name = 'Fraud')
      fig = go.Figure(data = [trace1, trace2])
      fig.update_layout(title = '{x} vs. {y}'.format(x=x, y = y))
      fig.update_layout(barmode = 'group')
      fig.show()


# In[ ]:


vis_data(df, 'insured_sex')


# In[ ]:


vis_data(df, 'insured_education_level')


# In[ ]:


vis_data(df, 'insured_occupation')


# From the data, it looks like people in exec-managerial positions have more number of frauds compared to other occupations.
# 
# Sales, tech-support and transport moving also have relatively high cases of fraud.

# In[ ]:


vis_data(df, 'insured_relationship')


# In[ ]:


vis_data(df, 'incident_type')


# Multi-vehicle and single vehicle collisions have more number of frauds compared to parked and vehicle theft. One of the reasons could be that in a collision, there is high possibility of more damage to car, as well as the passengers and hence the need to file false insurance claims.

# In[ ]:


vis_data(df, 'collision_type')


# While there are significant numbers of false claims in front and side collisions, rear collisions are the highest.
# 
# This data is for the US and there, many people use dash cams while driving to record whatever is happening while they drive. In rear collisions, the footage from dash cams is not very helpful to onclusively prove whose mistake it was (insurance owner or other car owner). Maybe that is the reason for more fradulent claims in rear collisions.

# In[ ]:


vis_data(df, 'incident_severity')


# Here, compared to minor damage, total loss and trivial damage, fraudulent claims are highest in major damage.
# 
# One reason could be that the high amount of repair cost which will be incurred by the insurer due to major damage. 

# In[ ]:


vis_data(df, 'authorities_contacted')


# In[ ]:


vis_data(df, 'insured_hobbies')


# One thing which is striking in this graph is that people with chess and cross-fit as hobby have extremely high number of fraudulent claims.
# 
# We will keep them and rename other values as 'other'

# In[ ]:


hobbies = df['insured_hobbies'].unique()
for hobby in hobbies:
  if (hobby != 'chess') & (hobby != 'cross-fit'):
    df['insured_hobbies'] = df['insured_hobbies'].str.replace(hobby, 'other')

df['insured_hobbies'].unique()


# In[ ]:


df.head()


# In[ ]:


vis_data(df, 'age', 'anything', 'hist')


# We will bin the ages and then check the trend for fraud vs. no fraud according to age.

# In[ ]:


df['age'].describe()


# In[ ]:


bin_labels = ['15-20', '21-25', '26-30', '31-35', '36-40', '41-45', '46-50', '51-55', '56-60', '61-65']
bins = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]

df['age_group'] = pd.cut(df['age'], bins = bins, labels = bin_labels, include_lowest = True)


# In[ ]:


vis_data(df, 'age_group')


# People in the age group of 31-35 and 41-45 have more number of frauds

# In[ ]:


vis_data(df, 'months_as_customer', 'not', 'hist')


# Like we did for the age column, we will create a new column grouping the months_as_customer column data.

# In[ ]:


df['months_as_customer'].describe()


# In[ ]:


bins = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
bin_labels = ['0-50','51-100','101-150','151-200','201-250','251-300','301-350','351-400','401-450','451-500']

df['months_as_customer_groups'] = pd.cut(df['months_as_customer'], bins = 10, labels = bin_labels, include_lowest= True)


# In[ ]:


vis_data(df, 'months_as_customer_groups')


# In[ ]:


vis_data(df, 'auto_make')


# In[ ]:


vis_data(df, 'number_of_vehicles_involved')


# In[ ]:


vis_data(df, 'witnesses', 'fraud_reported')


# In[ ]:


vis_data(df, 'bodily_injuries')


# In[ ]:


vis_data(df, 'total_claim_amount', 'y', 'hist')


# In[ ]:


vis_data(df, 'incident_hour_of_the_day')


# In[ ]:


vis_data(df, 'number_of_vehicles_involved')


# In[ ]:


vis_data(df, 'witnesses')


# In[ ]:


vis_data(df, 'auto_year')


# In[ ]:


df['policy_annual_premium'].describe()


# In[ ]:


bins = list(np.linspace(0,2500, 6, dtype = int))
bin_labels = ['very low', 'low', 'medium', 'high', 'very high']

df['policy_annual_premium_groups'] = pd.cut(df['policy_annual_premium'], bins = bins, labels=bin_labels)


# In[ ]:


vis_data(df, 'policy_annual_premium_groups')


# In[ ]:


df['policy_deductable'].describe()


# In[ ]:


bins = list(np.linspace(0,2000, 5, dtype = int))
bin_labels = ['0-500', '501-1000', '1001-1500', '1501-2000']

df['policy_deductable_group'] = pd.cut(df['policy_deductable'], bins = bins, labels = bin_labels)

vis_data(df, 'policy_deductable_group')


# In[ ]:


vis_data(df, 'property_damage')


# In[ ]:


vis_data(df, 'police_report_available')


# In[ ]:


#removing columns for which we created groups
df = df.drop(['age', 'months_as_customer', 'policy_deductable', 'policy_annual_premium'], axis = 1)
df.columns


# Based on the EDA, we remove some of the columns

# In[ ]:


required_columns = ['policy_number', 'insured_sex', 'insured_education_level', 'insured_occupation',
       'insured_hobbies', 'capital-gains', 'capital-loss', 'incident_type', 'collision_type', 'incident_severity',
       'authorities_contacted', 'incident_hour_of_the_day', 'number_of_vehicles_involved',
       'witnesses', 'total_claim_amount',
       'injury_claim', 'property_claim', 'vehicle_claim',
       'fraud_reported', 'age_group',
       'months_as_customer_groups', 'policy_annual_premium_groups']

print(len(required_columns))


# In[ ]:


df1 = df[required_columns]

corr_matrix = df1.corr()

fig = go.Figure(data = go.Heatmap(
                                z = corr_matrix.values,
                                x = list(corr_matrix.columns),
                                y = list(corr_matrix.index)))

fig.update_layout(title = 'Correlation')

fig.show()


# From the correlation matrix, we see there is high correlation between `vehicle claim`, `total_claim_amount`, `property_claim` and `injury_claim`
# 
# The reason for it is that `total_claim_amount` is the sum of columns `vehicle claim`,`property_claim` and `injury_claim`.
# 
# We will remove the other 3 columns and only keep `total_claim_amount` as it captures the information and removes collinearity.

# In[ ]:


t = df['total_claim_amount'].iloc[1]
a = df['vehicle_claim'].iloc[1]
b = df['property_claim'].iloc[1]
c = df['injury_claim'].iloc[1]

print(t)
a+b+c


# Keeping only the `total_claim_amount` column from these.

# In[ ]:


required_columns = ['insured_sex', 'insured_occupation',
       'insured_hobbies', 'capital-gains', 'capital-loss', 'incident_type', 'collision_type', 'incident_severity',
       'authorities_contacted', 'incident_hour_of_the_day', 'number_of_vehicles_involved',
       'witnesses', 'total_claim_amount', 'fraud_reported', 'age_group',
       'months_as_customer_groups', 'policy_annual_premium_groups']

print(len(required_columns))


# In[ ]:


df1 = df1[required_columns]
df1.head()


# ## Encoding data for modelling

# In[ ]:


cat_cols = ['age_group', 'months_as_customer_groups', 'policy_annual_premium_groups']
for col in cat_cols:
  df1[col] = df1[col].astype('object')

columns_to_encode = []
for col in df1.columns:
  if df1[col].dtype == 'object':
    columns_to_encode.append(col)

columns_to_encode


# In[ ]:


df1.info()


# In[ ]:


df1.head()


# In[ ]:


df2 = pd.get_dummies(df1, columns = columns_to_encode)

df2.head()


# ## Features and Target

# In[ ]:


features = []
for col in df2.columns:
  if col != 'fraud_reported':
    features.append(col)

target = 'fraud_reported'

X = df2[features]
y = df2[target]


# ## Scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X = sc.fit_transform(X)


# # Modelling

# ## Splitting in train and test data

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)


# ## Logistic Regression

# In[ ]:


lr = LogisticRegression()

lr.fit(X_train, y_train)
preds = lr.predict(X_test)

score = lr.score(X_test, y_test)
print(score)


# In[ ]:


print(classification_report(y_test, preds))


# ## Synthetic Minority Over-sampling Technique (SMOTE)
# 
# We saw that our data is not balanced. Therefore, we will apply SMOTE and then predict.

# In[ ]:


oversample = SMOTE(random_state=9)


# In[ ]:


X_over, y_over = oversample.fit_resample(X, y)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, random_state = 1)


# We will use the LogisticRegression we defined earlier

# In[ ]:


lr.fit(X_train, y_train)
preds = lr.predict(X_test)
score = lr.score(X_test, y_test)
print(score)
print()
print(classification_report(y_test, preds))


# ### The accuracy increased from 84.8% to **85.1%**

# ## Decision Tree

# In[ ]:


dtc = DecisionTreeClassifier()

dtc.fit(X_train, y_train)
preds = dtc.predict(X_test)

score = dtc.score(X_test, y_test)
print(score)
print()
print(classification_report(y_test, preds))


# ## There is some more increase in accuracy

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# ## Random Forest

# In[ ]:


rfc = RandomForestClassifier(random_state = 1)
rfc.fit(X_train, y_train)


# In[ ]:


preds = rfc.predict(X_test)

score = rfc.score(X_test, y_test)
print(score*100)
print()
print(classification_report(y_test, preds))


# ## Using Random Forest, we got an accuracy of 90.18%.
# We were able to increase our model accuracy from 84.8% to over 90%.
# 
# We will see if we can increase it further using hyperparameter tuning.

# In[ ]:


#implementing 

svc = SVC(kernel='linear')
svc.fit(X_train, y_train)

preds = svc.predict(X_test)

print('Score:' , svc.score(X_test, y_test))
print('Classification report:', classification_report(y_test, preds))


# # Hyperparameter Tuning

# In SVC, we can change the kernel and degree inorder to improve our model. We will do that see if accuracy improves.

# In[ ]:


degrees = [2,3,4,5,6,7,8]
kernels = ['poly', 'rbf', 'sigmoid']
c_value = [1,2,3]


# In[ ]:


scores = {}
for degree in degrees:
    for kernel in kernels:
        for c in c_value:
            svc_t = SVC(kernel = kernel, degree = degree, C = c)
            svc_t.fit(X_train, y_train)
            
            preds = svc_t.predict(X_test)
            score = svc_t.score(X_test,y_test)
#             print('Score with degree as {d}, kernel as {k}, C as {c} is:'.format(d = degree, k = kernel, c = c), score)
            scores['Score with degree as {d}, kernel as {k}, C as {c} is best'.format(d = degree, k = kernel, c = c)] = score

print(max(scores, key=scores.get))


# In[ ]:


svc_tuned = SVC(kernel='poly', degree = 4, C = 2)
svc_tuned.fit(X_train, y_train)

preds = svc_tuned.predict(X_test)

print('Score:' , svc_tuned.score(X_test, y_test))
print('Classification report:', classification_report(y_test, preds))


# ## Accuracy increased to 91.77% from 90.18% using SVM classifier.
# This is the best score we have got till now various ML algorithms we tried.

# In[ ]:


rfc_tuned = RandomForestClassifier(n_estimators = 1000, random_state = 1, min_samples_split = 2)
rfc_tuned.fit(X_train, y_train)
preds_tuned = rfc_tuned.predict(X_test)
score = rfc_tuned.score(X_test, y_test)
print(score)


# There was no improvement. We will use GridSearch to check for the best parameters ad use them for tuning.

# ## GridSearch

# In[ ]:


n_estimators = [100, 300, 500, 800, 1200]
max_depth = [5, 8, 15, 25, 30]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10] 

hyper = dict(n_estimators = n_estimators, max_depth = max_depth,  
              min_samples_split = min_samples_split, 
             min_samples_leaf = min_samples_leaf)

grid = GridSearchCV(rfc, hyper, cv = 3, verbose = 1, 
                      n_jobs = -1)
best = grid.fit(X_train, y_train)


# In[ ]:


print(best)


# Using the best parameters from GridSearch

# In[ ]:


rfc_tuned = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0,
                                              class_weight=None,
                                              criterion='gini', max_depth=None,
                                              max_features='auto',
                                              max_leaf_nodes=None,
                                              max_samples=None,
                                              min_impurity_decrease=0.0,
                                              min_impurity_split=None,
                                              min_samples_leaf=1,
                                              min_samples_split=2,
                                              min_weight_fraction_leaf=0.0,
                                              n_estimators=100, n_jobs=None,
                                              oob_score=False, random_state=1,
                                              verbose=0, warm_start=False)

rfc_tuned.fit(X_train, y_train)
preds_tuned = rfc_tuned.predict(X_test)

score = rfc_tuned.score(X_test, y_test)

print(score)


# # Conclusion
# 
# ## Out of all the algorithms, we got best accuracy (91.77%) with SVM classifier. 
# 
# We were able to increase our accuracy from 84% to 91.7% using data cleaning, feature engineering, feature selection and hyperparameter tuning.
# 
# Note: Using GridSearch method for random forest we got accuracy of over 90% but it took a lot of time.
# 
# ### Upvote and like if you find it useful :)
