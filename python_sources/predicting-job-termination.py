#!/usr/bin/env python
# coding: utf-8

# # Prediction Job Termination

# ## Importing Necesary Code

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

import os
print(os.listdir("../input"))


# ## Reading in the Dataset

# In[ ]:


df = pd.read_csv('../input/MFG10YearTerminationData.csv')


# ## Preliminary Cleaning of the Dataset

# In[ ]:


df.head(20)


# In[ ]:


df.tail(20)


# In[ ]:


df.info()


# ## Observations
# 
# - No null values
# - Dataset is a mix of integers and strings
# - Some columns are repetitive, and can be removed without further analysis
#     - age can be determined from birthdate_key and recorddate_key, one set can be discarded
#     - length_of_service can be determined from orighiredate_key and recorddate_key, one set can be discarded
#     - depatment_name and job_title are also pretty similar, so one column can be discarded
#     - city_name also has little relevance
# - Some columns give away information that or machine learning model is trying to predoct, so these colums must be removed
#     - termreason_desc gives away the fact that those employees were terminated
#     - termtype_desc also gives away the fact that those employees were terminated
# - Various entries in the columns are in the form of strings, we must change them to integral values
# 
# Based on these observations, we can start removing and reformatting data

# ### Dropping Columns

# In[ ]:


df = df.drop(columns = ['birthdate_key', 'recorddate_key', 'orighiredate_key', 'terminationdate_key', 'termreason_desc', 'termtype_desc', 'department_name', 'gender_full'])


# ### Reformating strings into integral data

# In[ ]:


df['job_title'].value_counts()


# In[ ]:


'''
Job_title is the most tedious column, as it has many distinct entries, we will generalize like jobs into categories, and
then turn them into numerical values
'''

board = ['VP Stores', 'Director, Recruitment', 'VP Human Resources', 'VP Finance',
         'Director, Accounts Receivable', 'Director, Accounting',
         'Director, Employee Records', 'Director, Accounts Payable',
         'Director, HR Technology', 'Director, Investments',
         'Director, Labor Relations', 'Director, Audit', 'Director, Training',
         'Director, Compensation']

executive = ['Exec Assistant, Finance', 'Exec Assistant, Legal Counsel',
             'CHief Information Officer', 'CEO', 'Exec Assistant, Human Resources',
             'Exec Assistant, VP Stores']

manager = ['Customer Service Manager', 'Processed Foods Manager', 'Meats Manager',
           'Bakery Manager', 'Produce Manager', 'Store Manager', 'Trainer', 'Dairy Manager']


employee = ['Meat Cutter', 'Dairy Person', 'Produce Clerk', 'Baker', 'Cashier',
            'Shelf Stocker', 'Recruiter', 'HRIS Analyst', 'Accounting Clerk',
            'Benefits Admin', 'Labor Relations Analyst', 'Accounts Receiveable Clerk',
            'Accounts Payable Clerk', 'Auditor', 'Compensation Analyst',
            'Investment Analyst', 'Systems Analyst', 'Corporate Lawyer', 'Legal Counsel']

def changeTitle(row):
    if row in board:
        return 'board'
    elif row in executive:
        return 'executive'
    elif row in manager:
        return 'manager'
    else:
        return 'employee'
    
df['job_title'] = df['job_title'].apply(changeTitle)

df.head()


# In[ ]:


df['job_title'] = df['job_title'].map({'board': 3, 'executive': 2, 'manager': 1, 'employee': 0})
df.head()


# In[ ]:


df['city_name'].value_counts()

# We will sort these cities by population, and then turn them into integral values 


# In[ ]:


city_pop_2011 = {'Vancouver':2313328,
                 'Victoria':344615,
                 'Nanaimo':146574,
                 'New Westminster':65976,
                 'Kelowna':179839,
                 'Burnaby':223218,
                 'Kamloops':85678,
                 'Prince George':71974,
                 'Cranbrook':19319,
                 'Surrey':468251,
                 'Richmond':190473,
                 'Terrace':11486,
                 'Chilliwack':77936,
                 'Trail':7681,
                 'Langley':25081,
                 'Vernon':38180,
                 'Squamish':17479,
                 'Quesnel':10007,
                 'Abbotsford':133497,
                 'North Vancouver':48196,
                 'Fort St John':18609,
                 'Williams Lake':10832,
                 'West Vancouver':42694,
                 'Port Coquitlam':55985,
                 'Aldergrove':12083,
                 'Fort Nelson':3561,
                 'Nelson':10230,
                 'New Westminister':65976,
                 'Grand Forks':3985,
                 'White Rock':19339,
                 'Haney':76052,
                 'Princeton':2724,
                 'Dawson Creek':11583,
                 'Bella Bella':1095,
                 'Ocean Falls':129,
                 'Pitt Meadows':17736,
                 'Cortes Island':1007,
                 'Valemount':1020,
                 'Dease Lake':58,
                 'Blue River':215}


# In[ ]:


#Make a copy of city names
df['Pop'] = df['city_name']

# Map from city name to population
df['Pop'] = df.Pop.map(city_pop_2011)

# Make a new column for population category
df['Pop_category'] = df.Pop

# Categorise according to population size
# >= 100,000 is City
# 10,000 to 99,999 is Rural
# < 10,000 is Remote
# Guidance from Australian Institute of Health and Welfare
# http://www.aihw.gov.au/rural-health-rrma-classification/
city_ix = (df['Pop'] >= 100000)
rural_ix = ((df['Pop'] < 100000) & (df['Pop'] >= 10000))
remote_ix = (df['Pop'] < 10000)
df.loc[city_ix, 'Pop_category'] = 'City'
df.loc[rural_ix, 'Pop_category'] = 'Rural'
df.loc[remote_ix, 'Pop_category'] = 'Remote'

df['Pop_category'] = df['Pop_category'].map({'City' : 0, 'Rural' : 1, 'Remote' : 2})

# Check the replacement went to plan
df.Pop_category.value_counts()


# In[ ]:


df['gender_short'] = df['gender_short'].map({'M': 1, 'F': 0})


# In[ ]:


df['STATUS'] = df['STATUS'].map({'ACTIVE': 1, 'TERMINATED': 0})
df.head()


# In[ ]:


df['BUSINESS_UNIT'].value_counts()


# In[ ]:


df['BUSINESS_UNIT'] = df['BUSINESS_UNIT'].map({'STORES': 0, 'HEADOFFICE' :1})


# ## Interpreting the Data

# In[ ]:


out_of_co = df[df.STATUS == 0]
in_co = df[df.STATUS == 1]


# In[ ]:


df.head()


# In[ ]:


a = sns.jointplot(out_of_co.age, out_of_co.length_of_service, color='r')


# In[ ]:


a = sns.FacetGrid(out_of_co, col='Pop_category', row='job_title', palette='Set1_r', 
                  hue='gender_short', margin_titles=True)
b = (a.map(plt.scatter, 'age', 'length_of_service').add_legend())


# In[ ]:


c = sns.FacetGrid(in_co, col='Pop_category', row='job_title', palette='Set1_r', 
                  hue='gender_short', margin_titles=True)
d = (c.map(plt.scatter, 'age', 'length_of_service').add_legend())


# ## Observations
# 
# - Nobody was terminated if they were at the executive level, or higher.
# - No major discrepancy between male and female termination
# - Executives and board members live in the cities

# ## Preparing The Model

# In[ ]:


# Deleting columns that are not relevant to predictions
short_df = df.drop(columns = ['EmployeeID', 'store_name','job_title','BUSINESS_UNIT', 'city_name'])


# In[ ]:


y = short_df['STATUS']
X = short_df.drop('STATUS', axis=1)


# In[ ]:


y.head()


# In[ ]:





# In[ ]:


X.head()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=10)


# # Machine Learning Models 

# ### K Nearest Neighbor

# In[ ]:


model = KNeighborsClassifier(n_neighbors=5, weights='uniform')
model.fit(X_train, y_train)
score = model.score(X_test, y_test)


# In[ ]:


print(score)


# ### Random Forest Regressor

# In[ ]:


model = RandomForestClassifier(n_estimators = 100)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)


# In[ ]:


print(score)


# #### Finding the Feature Importance in a Random Forest

# In[ ]:


# Finds how important each feature is for the model to make a prediction
feature_importances = pd.DataFrame(model.feature_importances_,
                                   index = X_train.columns,
                                   columns=['importance']).sort_values('importance',ascending=False)


# In[ ]:


feature_importances


# In[ ]:


list(feature_importances.index)


# In[ ]:


list(feature_importances.values)


# In[ ]:


list(feature_importances.index)


# In[ ]:


# fig, ax = plt.subplots()

importances = model.feature_importances_
std = np.std([model.feature_importances_ for tree in model.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
names = list(feature_importances.index)
# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")

plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), names, rotation=90)
plt.xlim([-1, X.shape[1]])

plt.show()


# ## Next Steps
# 
# I would use a nueral network, as they might be more effective in predicting based on differnt sized datasets
