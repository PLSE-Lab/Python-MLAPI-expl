#!/usr/bin/env python
# coding: utf-8

# # ANALYSIS & PREDICTION OF JOB TERMINATION

# ## ***Importing necesssary packages for the analysis***

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


## loading and reading data from the file
df=pd.read_csv('../input/empatt/emp.csv')
df


# In[ ]:


df.STATUS.value_counts()


# In[ ]:


df.info()


# In[ ]:


df.STATUS.unique()


# In[ ]:


df.isnull().sum()


# ### ***Dropping features that does not account for status of the employee like EMPLOYEE ID,BIRTH_DATE KEY,ORIGHIRE DATE,TERMINATION DATE KEY,GENDER FULL,the mentioned featured does not have any impact on termination od the employee***

# In[ ]:


df.drop(['EmployeeID','recorddate_key','birthdate_key','orighiredate_key','gender_full','terminationdate_key'],axis=1,inplace=True)
df


# In[ ]:


df.describe()


# ## ***Basic Exploratory data analysis***

# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x = 'STATUS', data = df)


# In[ ]:


plt.figure(figsize=(10,6))
sns.boxplot(x = 'STATUS', y='age', data = df)


# ### ***As we can observe above,2014 has seen many terminations compared to other periods***

# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(hue= 'STATUS', data = df, x = 'BUSINESS_UNIT')


# We can observe,almost all the termination are from STORES business unit

# In[ ]:


plt.subplots(figsize=(20,10))
sns.countplot(x= 'age',hue = 'STATUS',data = df,palette='colorblind')


# ## Termination are more after the 60 years of age

# In[ ]:


df.corr()


# In[ ]:


plt.figure(figsize=(20,10))
sns.heatmap(df.corr(),annot=True,fmt='.0%')


# ## Length of Service and Age are highly correlated

# In[ ]:


df.columns


# In[ ]:


plt.subplots(figsize=(14,5))
sns.countplot(x= 'gender_short',hue = 'STATUS',data = df,palette='colorblind')


# ## Female gender has more active and terminated status

# In[ ]:


df.job_title.value_counts()


# ##
# 
# 1. Meat Cutter 
# 2. Dairy Person                      
# 3. Produce Clerk  
# 4. Baker
# 5. Cashier   
# 
# Above are the top 5 job titles ##

# In[ ]:


plt.subplots(figsize=(14,5))
sns.countplot(x= 'termreason_desc',hue = 'STATUS',data = df,palette='colorblind')


# In[ ]:


plt.subplots(figsize=(14,5))
sns.countplot(x= 'termtype_desc',hue = 'STATUS',data = df,palette='colorblind')


# In[ ]:


df=pd.get_dummies(columns=['STATUS'],data=df,drop_first=True)
df


# In[ ]:


from collections import Counter
out_of_co = df[df.STATUS_TERMINATED == 0]
term_per_year = Counter(out_of_co.STATUS_YEAR)
term_per_year_df = pd.DataFrame.from_dict(term_per_year, orient='index')
term_per_year_df = term_per_year_df.sort_index()
term_per_year_df.plot(kind='bar')


# In[ ]:


df.job_title.value_counts()


# ### As we can see there are many sub categories in job_title lets bin them under somee category and use accordingly

# In[ ]:


##Job_title is the most tedious column, as it has many distinct entries, we will generalize like jobs into categories, 
## and then turn them into numerical values

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


df=pd.get_dummies(columns=['job_title'],data=df,drop_first=True)
df.head()


# In[ ]:


df.STATUS_YEAR.unique()


# In[ ]:


df.city_name.unique()


# ### As we can see the list of city we will segrregate thme as RURAL,URBAN AND OTHER

# In[ ]:


city_population = {'Vancouver':2313328,
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


##Make a copy of city names
df['Population'] = df['city_name']

# Map from city name to population
df['Population'] = df.Population.map(city_population)

# Make a new column for population category
df['Population_category'] = df.Population

# Categorise according to population size
# >= 100,000 is Urban
# 10,000 to 99,999 is Rural
# < 10,000 is Other
# Guidance from Australian Institute of Health and Welfare

urban_ix = (df['Population'] >= 100000)
rural_ix = ((df['Population'] < 100000) & (df['Population'] >= 10000))
other_ix = (df['Population'] < 10000)
df.loc[urban_ix, 'Population_category'] = 'Urban'
df.loc[rural_ix, 'Population_category'] = 'Rural'
df.loc[other_ix, 'Population_category'] = 'Other'

df['Population_category'] = df['Population_category'].map({'Urban' : 0, 'Rural' : 1, 'Other' : 2})

df.Population_category.value_counts()


# In[ ]:


df.head(2)


# In[ ]:


df['gender_short'] = df['gender_short'].map({'M': 1, 'F': 0})
df['BUSINESS_UNIT'] = df['BUSINESS_UNIT'].map({'STORES': 0, 'HEADOFFICE' :1})


# In[ ]:


df.head()


# ### Observations
# 1. Nobody was terminated if they were at the executive level, or higher.
# 2. Usually Terminations where done above the age of 60
# 3. No major difference between male and female termination,however female are more in both
# 4. Executives and board members live in the Urban areas

# In[ ]:


df.termreason_desc.unique()


# In[ ]:


df.termtype_desc.unique()


# In[ ]:


df.department_name.unique()


# In[ ]:


df = df.drop(columns = [ 'store_name','BUSINESS_UNIT', 'city_name','department_name'])


# In[ ]:


df['termtype_desc'] = df['termtype_desc'].map({'Not Applicable': 0, 'Voluntary' :1,'Involuntary':2})
df['termreason_desc'] = df['termreason_desc'].map({'Not Applicable': 0, 'Retirement' :1,'Resignaton':2,'Layoff':3})


# In[ ]:


df.head(2)


# In[ ]:


df.info()


# In[ ]:


df.termreason_desc.isna().sum()


# # Building Models 
# 
# ## ***1. LOGISTIC REGRESSION MODEL***

# In[ ]:


x = df.drop('STATUS_TERMINATED', axis=1)
y = df['STATUS_TERMINATED']


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


# In[ ]:


# 30% of the data will be used for testing
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.3, random_state=0)


# In[ ]:


logm=LogisticRegression()


# In[ ]:


logm.fit(x_train,y_train)


# ## ***Predicting the Results***

# In[ ]:


y_pred=logm.predict(x_test)
y_pred


# ## ***Evaluation of model performance***

# In[ ]:


# Classification Report
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[ ]:


print(classification_report(y_test,y_pred))


# In[ ]:


print(confusion_matrix(y_test,y_pred))


# ## ***K-Fold Cross Validation***

# In[ ]:


from sklearn.model_selection import cross_val_score
lg_Kfold_accu = cross_val_score(estimator = logm,X = x_train, y = y_train, cv = 10)
lg_Kfold_accu=lg_Kfold_accu.mean()
print("Accuracy: {:.2f} %".format(lg_Kfold_accu*100))


# ## ***Executing Hyper Paramter Tuning***

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


parameters = [{'penalty': [11,12,'elasticnet'], 'C': [1,10,50,100,200]},
              {'tol': [0.001,0.0001,0.00001]}]
lg_grid_search = GridSearchCV(estimator = logm,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
lg_grid_search = lg_grid_search.fit(x_train, y_train)
lg_best_accuracy = lg_grid_search.best_score_
lg_best_parameters = lg_grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(lg_best_accuracy*100))
print("Best Parameters:", lg_best_parameters)


# In[ ]:


parameters = [{'tol': [0.01,0.001,0.002,0.003]}]
lg_grid_search = GridSearchCV(estimator = logm,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
lg_grid_search = lg_grid_search.fit(x_train, y_train)
lg_best_accuracy = lg_grid_search.best_score_
lg_best_parameters = lg_grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(lg_best_accuracy*100))
print("Best Parameters:", lg_best_parameters)


# ### ***We have achieved a accuracy score of 97.10 % under logistic regression ***

# In[ ]:


df.to_csv('empatt.csv', index = False)


# In[ ]:




