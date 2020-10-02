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
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


# loading the data
data = pd.read_csv('/kaggle/input/glassdoor_jobs.csv')
data.head(3)


# 
# ### Data Cleaning
# - inorder to perform any kind of operations first we need to clean the data otherwise model says **garbage in = garbage out**

# In[ ]:


data.info()


# In[ ]:


data.columns


# In[ ]:


data.isna().sum()


# In[ ]:


# take a copy of data and remove unnecessary attributes
emp_data = data.copy(deep= True)
emp_data.drop(columns= ['Unnamed: 0'], inplace = True)
emp_data.head()


# In[ ]:


emp_data.columns


# ### Job Title Handling

# In[ ]:


emp_data['Job Title'].unique()


# In[ ]:



# job title cleaning

def jobtitle_cleaner(title):
    if 'data scientist' in title.lower():
        return 'D-sci'
    elif 'data engineer' in title.lower():
        return 'D-eng'
    elif 'analyst' in title.lower():
        return 'analyst'
    elif 'machine learning' in title.lower():
        return 'ML'
    elif 'manager' in title.lower():
        return 'manager'
    elif 'director' in title.lower():
        return 'director'
    elif 'research' in title.lower():
        return 'R&D'
    else:
        return 'na'


# In[ ]:


emp_data['JobTitles'] = emp_data['Job Title'].apply(jobtitle_cleaner)


# In[ ]:


emp_data['Job Title'].unique()


# In[ ]:


emp_data['JobTitles'].unique()


# In[ ]:



emp_data['JobTitles'].value_counts()


# In[ ]:


senior_list = ['sr','sr.','senior','principal', 'research', 'lead', 'R&D','II', 'III']
junior_list = ['jr','jr.','junior']


def jobseniority(title):
    for i in senior_list:
        if i in title.lower():
            return 'Senior Prof'
            
    for j in junior_list:
        if j in title.lower():
            return 'Junior Prof'
        else:
            return 'No Desc'


# In[ ]:



emp_data['Job Seniority'] = emp_data['Job Title'].apply(jobseniority)


# In[ ]:



emp_data['Job Seniority'].unique()


# In[ ]:


emp_data['Job Seniority'].value_counts()


# ### Job Description Handling

# In[ ]:


# job descriptions
jobs_list = ['python', 'excel','r studio', 'spark','aws']

for i in jobs_list:
    emp_data[i+'_'+'job'] = emp_data['Job Description'].apply(lambda x : 1 if i in x.lower() else 0)


# In[ ]:



for i in jobs_list:
    print(emp_data[i+'_'+'job'].value_counts())


# ### Company Name Handling

# In[ ]:


emp_data['Company Name'].unique()


# In[ ]:


emp_data['Company Name'][0].split('\n')[0]


# In[ ]:


# remove numbers from company name
emp_data['Company Name'] = emp_data['Company Name'].apply(lambda x : x.split("\n")[0])
emp_data['Company Name'].value_counts()


# ### Head quarters Handling

# In[ ]:


emp_data['Headquarters'].unique()


# In[ ]:



emp_data['Hquarters'] = emp_data['Headquarters'].str.split(',').str[1]
emp_data['Hquarters'].value_counts().head()


# ### Location Handling

# In[ ]:


emp_data['Location'].unique()


# In[ ]:



emp_data['loaction spots'] = emp_data['Location'].str.split(',').str[1]
emp_data['loaction spots'].value_counts().head()


# ### Compitators Handling

# In[ ]:


emp_data['Competitors'].unique()


# In[ ]:


emp_data['compitator company'] = emp_data['Competitors'].str.split(',').str[0].replace('-1', 'no compitator')


# In[ ]:



emp_data['compitator company'].value_counts()


# ### Type of ownership Handling

# In[ ]:


emp_data['Type of ownership'].unique()


# In[ ]:



emp_data['Ownership'] = emp_data['Type of ownership'].str.split('-').str[1].replace(np.NaN, 'others')
emp_data['Ownership'].value_counts()


# ### Revenue Handling

# In[ ]:


emp_data['Revenue'].unique()


# In[ ]:


emp_data['Revenue'] = emp_data['Revenue'].str.replace('-1','others')


# In[ ]:


emp_data['Revenue'].value_counts()


# ### Size Handling

# In[ ]:


emp_data['Size'].unique()


# In[ ]:


emp_data['Size'] = emp_data['Size'].str.replace('-1','others')
emp_data['Size'].value_counts()


# ### Salary estimate Handling

# In[ ]:


emp_data["Salary Estimate"].unique()


# In[ ]:



emp_data['min_sal'] = emp_data['Salary Estimate'].str.split(",").str[0].str.replace('(Glassdoor est.)','')


# In[ ]:


emp_data['min_sal'] = emp_data['min_sal'].str.replace('(Glassdoor est.)','').str.split('-').str[0].str.replace('$','').str.replace('K','')


# In[ ]:


emp_data['min_sal'].unique()


# In[ ]:


emp_data['min_sal'] = emp_data['min_sal'].str.replace('Employer Provided Salary:','')
emp_data['min_sal'].unique()


# In[ ]:


emp_data['max_sal'] = emp_data['Salary Estimate'].str.split(",").str[0].str.replace('(Glassdoor est.)','')
emp_data['max_sal']


# In[ ]:


emp_data['max_sal'] = emp_data['max_sal'].str.replace('(Glassdoor est.)','').str.split('-').str[1].str.replace('$','').str.replace('K','')


# In[ ]:



emp_data['max_sal'] = emp_data['max_sal'].str.replace('(Employer est.)','')


# In[ ]:


emp_data['max_sal'] = emp_data['max_sal'].str.split().str[0].str.replace('(','').str.replace(')','')


# In[ ]:


emp_data['max_sal'].unique()


# In[ ]:


emp_data['min_sal'] = pd.to_numeric(emp_data['min_sal'], errors='coerce')
type(emp_data['min_sal'])


# In[ ]:


emp_data['min_sal'].isna().sum()


# In[ ]:


emp_data['min_sal'].hist()
plt.show()


# In[ ]:


emp_data['max_sal'].isna().sum()


# In[ ]:



emp_data['min_sal'] = emp_data['min_sal'].replace(np.nan, emp_data['min_sal'].mean())


# In[ ]:


emp_data['min_sal'].isna().sum()


# In[ ]:



emp_data['max_sal'] = pd.to_numeric(emp_data['max_sal'], errors='coerce')
type(emp_data['max_sal'])


# In[ ]:


emp_data['max_sal'].isnull().sum()


# In[ ]:



emp_data['max_sal'].hist()
plt.show()


# In[ ]:



emp_data['avg.salary'] = (emp_data['min_sal'] + emp_data['max_sal'])/ 2


# In[ ]:


emp_data['avg.salary'].hist()
plt.show()


# ### Data gathering

# In[ ]:



emp_data.head()


# In[ ]:


final_data = emp_data[['Rating',
       'Company Name', 'Size',
       'Type of ownership','Sector', 'Revenue',
       'JobTitles', 'Job Seniority', 'python_job', 'excel_job', 'r studio_job',
       'spark_job', 'aws_job', 'Hquarters', 'loaction spots',
       'compitator company', 'Ownership','avg.salary']]
final_data.head()


# ### Getting dummies

# In[ ]:


final_data = pd.get_dummies(data = final_data, columns = ['Company Name', 'Size', 'Type of ownership', 'Sector',
       'Revenue', 'JobTitles', 'Job Seniority','Hquarters', 'loaction spots',
       'compitator company', 'Ownership'])


# In[ ]:



final_data.head()


# ### Scaling

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
ms = MinMaxScaler()
final_data[['Rating', 'avg.salary']] = ms.fit_transform(final_data[['Rating', 'avg.salary']])


# In[ ]:


final_data.head()


# In[ ]:


# split the data into attributes and lable
X = final_data.drop(columns= 'avg.salary').values
y = final_data.iloc[:, 6].values


# ### train and test

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# ### Model selection

# In[ ]:


# Using GridSearchCV to find the best algorithm for this problem
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

def find_best_model(X, y):
    models = {
        'linear_regression': {
            'model': LinearRegression(),
            'parameters': {
                'n_jobs': [-1]
            }
            
        },
        
        'decision_tree': {
            'model': DecisionTreeRegressor(criterion='mse', random_state= 0),
            'parameters': {
                'max_depth': [5,10]
            }
        },
        
        'random_forest': {
            'model': RandomForestRegressor(criterion='mse', random_state= 0),
            'parameters': {
                'n_estimators': [10,15,20,50,100,200]
            }
        },
        
        'svm': {
            'model': SVR(gamma='auto'),
            'parameters': {
                'C': [1,10,20],
                'kernel': ['rbf','linear']
            }
        }

    }
    
    scores = [] 
    cv_shuffle = ShuffleSplit(n_splits=5, test_size=0.20, random_state=0)
        
    for model_name, model_params in models.items():
        gs = GridSearchCV(model_params['model'], model_params['parameters'], cv = cv_shuffle, return_train_score=False)
        gs.fit(X, y)
        scores.append({
            'model': model_name,
            'best_parameters': gs.best_params_,
            'Test score': gs.best_score_
        })
        
    return pd.DataFrame(scores, columns=['model','best_parameters','Test score'])

find_best_model(X_train, y_train)


# In[ ]:


# Creating linear regression model
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
# Fitting the dataset to the model
lr_model.fit(X_train, y_train)
print("Accuracy of the Linear Regression Model on Training set is : {}% and on Test set is {}%".format(round(lr_model.score(X_train, y_train),4)*100, round(lr_model.score(X_test, y_test),4)*100))


# In[ ]:


# Creating decision tree regression model
from sklearn.tree import DecisionTreeRegressor
decision_model = DecisionTreeRegressor(criterion='mse', max_depth=10, random_state=0)
# Fitting the dataset to the model
decision_model.fit(X_train, y_train)
print("Accuracy of the Decision Tree Regression Model on Training set is : {}% and on Test set is {}%".format(round(decision_model.score(X_train, y_train),4)*100, round(decision_model.score(X_test, y_test),4)*100))


# In[ ]:


# Creating random forest regression model
from sklearn.ensemble import RandomForestRegressor
forest_model = RandomForestRegressor(n_estimators=100, criterion='mse', random_state=0)
# Fitting the dataset to the model
forest_model.fit(X_train, y_train)
print("Accuracy of the Random Forest Regression Model on Training set is : {}% and on Test set is {}%".format(round(forest_model.score(X_train, y_train),4)*100, round(forest_model.score(X_test, y_test),4)*100))


# In[ ]:


# Creating AdaBoost regression model
from sklearn.ensemble import AdaBoostRegressor
adb_model = AdaBoostRegressor(base_estimator=decision_model, n_estimators=250, learning_rate=1, random_state=0)
# Fitting the dataset to the model
adb_model.fit(X_train, y_train)
print("Accuracy of the AdaBoost Regression Model on Training set is : {}% and on Test set is {}%".format(round(adb_model.score(X_train, y_train),4)*100, round(adb_model.score(X_test, y_test),4)*100))


# In[ ]:




