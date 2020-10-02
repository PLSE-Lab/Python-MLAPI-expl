#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pyforest')


# In[ ]:


from pyforest import *  # By importing pyforest, we don't need to include the package of pandas and numpy.
import os
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
from sklearn import metrics
from imblearn.over_sampling import SMOTE


# In[ ]:


churn = pd.read_csv('../input/employee-attrition/Employee_Churn.csv')
churn.head()


# In[ ]:


churn.describe()    # describe() gives us the information of the numerical features. 


# In[ ]:


churn.info()   # info() gives us the information of all the attributes' datatypes.


# In[ ]:


churn.isnull().sum()       # isnull() let's us know if there is any null values in any particular feature.


# In[ ]:


churn.memory_usage(deep=True)    # memory_usage gives us the clear idea of which attribute consumes more memory in bytes.


# In[ ]:


churn.BusinessTravel.value_counts()   # value_counts() let's us know the particular data appears how many times in a particular column.


# In[ ]:


churn.Department.value_counts()


# In[ ]:


churn.EducationField.value_counts()


# In[ ]:


churn.JobRole.value_counts()


# In[ ]:


from sklearn.preprocessing import LabelEncoder       # helps us to convert categorical into numerical.
le=LabelEncoder()
churn['Attrition'] = le.fit_transform(churn['Attrition'])
churn['BusinessTravel'] = le.fit_transform(churn['BusinessTravel'])
churn['Department'] = le.fit_transform(churn['Department'])
churn['Gender'] = le.fit_transform(churn['Gender'])
churn['MaritalStatus'] = le.fit_transform(churn['MaritalStatus'])
churn['OverTime'] = le.fit_transform(churn['OverTime'])
churn['EducationField'] = le.fit_transform(churn['EducationField'])
churn['JobRole'] = le.fit_transform(churn['JobRole'])


# In[ ]:


churn.head()


# In[ ]:


churn.memory_usage(deep=True)   # by converting all the categorical columns into numerical, the memory usage decreases.


# In[ ]:


# let's drop the attributes that are not required for implementation.

churn.drop(columns=['YearsWithCurrManager', 'WorkLifeBalance', 'TrainingTimesLastYear', 'StockOptionLevel', 'NumCompaniesWorked', 'MaritalStatus', 'EnvironmentSatisfaction', 'EmployeeNumber','StandardHours', 'Over18'], inplace=True, axis=1)
churn.info()


# In[ ]:


churn.shape  #shape let's us know the number of rows and columns in dataset.


# In[ ]:


print(churn.Age.max())  # we can get the data of a person who is of max age
print(churn.Age.min())  # we can get the data of a person who is of min age


# In[ ]:


print(churn.loc[churn.Age == 60])
print('-------------------------------------------------------------------------------------------------------')
print(churn.loc[churn.Age == 18])


# In[ ]:


churn.loc[(churn.Age == 18) & (churn.Attrition == 1)]   # Gives us the data of a person who is 18 years and who left the company.


# In[ ]:


churn.loc[(churn.Attrition == 1) & (churn.Gender == 1)]  #Male


# In[ ]:


print(churn.DailyRate.max())
print(churn.DailyRate.min())


# In[ ]:


print(churn.MonthlyIncome.max())   #employee who has max monthly income
print(churn.MonthlyIncome.min())   #employee who has min monthly income


# In[ ]:


churn.loc[(churn.MonthlyIncome == 19999) & (churn.Attrition == 0)]  


# In[ ]:


churn.loc[churn.MonthlyIncome == 1009]  # We can get an idea behind this employee's attrition that the monthly income is low, distance from home is also more, he is also doing overtime so it is obvious that the employee will definitely churn (leave the company)


# In[ ]:


print(churn.DistanceFromHome.max())
print(churn.DistanceFromHome.min())


# In[ ]:


churn.loc[(churn.DistanceFromHome >= 10) & (churn.Attrition == 1)]


# In[ ]:


churn.loc[(churn.Department == 0) & (churn.Attrition == 1)] #Human Resources


# In[ ]:


churn.loc[(churn.Department == 1) & (churn.Attrition == 1)] #Research & Development


# In[ ]:


churn.loc[(churn.Department == 2) & (churn.Attrition == 1)] #Sales


# From the above analysis we can come to know that, most of the employees who churned are of Research and Development department

# In[ ]:


print(churn.PerformanceRating.max())
print(churn.PerformanceRating.min())


# In[ ]:


churn[(churn.PerformanceRating == 4) & (churn.Attrition == 1)]


# In[ ]:


churn.loc[churn.YearsSinceLastPromotion == 15]


# Now, we will generate the Pearson matrix to know the correlation between each and every attributes.

# In[ ]:


numerical = [u'Age', u'DailyRate', u'DistanceFromHome', 
             u'Education',
             u'HourlyRate',
             u'MonthlyIncome', u'MonthlyRate',
             u'PercentSalaryHike', u'PerformanceRating',
             'TotalWorkingYears',
             'YearsAtCompany',
             'YearsSinceLastPromotion']
data = [
    go.Heatmap(
        z= churn[numerical].astype(float).corr().values, # Generating the Pearson correlation
        x=churn[numerical].columns.values,
        y=churn[numerical].columns.values,
        colorscale='Viridis',
        reversescale = False,
#         text = True ,
        opacity = 1.0
        
    )
]


layout = go.Layout(
    title='Pearson Correlation of numerical features',
    xaxis = dict(ticks='', nticks=36),
    yaxis = dict(ticks='' ),
    width = 900, height = 700,
    
)


fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='labelled-heatmap')


# In[ ]:


churn.Attrition.sum() # Total 237 employees have churned (left the company).


# In[ ]:


churn.MonthlyIncome.plot(kind='hist')


# In[ ]:


churn.DailyRate.plot(kind='hist')


# In[ ]:


churn.TotalWorkingYears.plot(kind='hist')


# In[ ]:


churn.Attrition.plot(kind='hist')


# In[ ]:


ax = sns.boxplot(x='Attrition', y='Department', data=churn)
ax = sns.stripplot(x='Attrition', y='Department', data=churn, jitter=True, edgecolor='gray')


# In[ ]:


sns.violinplot(x='Attrition',y='Department',data=churn,size=6)


# In[ ]:


# Now, let's split our data into training and testing.

train, test = train_test_split(churn, test_size=0.2)
print(train.shape)
print(test.shape)


# In[ ]:


#Pie chart will give the percentage of attrition and non-attrition

import plotly.offline as ply
values = pd.Series(churn["Attrition"]).value_counts()
trace = go.Pie(values=values)
ply.iplot([trace])


# In[ ]:


X=churn.loc[:, churn.columns != 'Attrition']
y=churn.loc[:, churn.columns == 'Attrition']


# In[ ]:


print("Shape of X is: {}".format(X.shape))
print("Shape of y is: {}".format(y.shape))


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Now we will implement some of machine learning algorithms to know the accuracy, precision and recall.

# In[ ]:


from sklearn.metrics import (accuracy_score, log_loss, classification_report)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
prediction = model.predict(X_test)
print('Accuracy',metrics.accuracy_score(prediction, y_test))
print(classification_report(prediction, y_test))


# In[ ]:


import xgboost as xgb
model = xgb.XGBClassifier()
model.fit(X_train, y_train)
prediction = model.predict(X_test)
print('Accuracy',metrics.accuracy_score(prediction, y_test))
print(classification_report(prediction, y_test))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
prediction = model.predict(X_test)
print(classification_report(prediction, y_test))


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
prediction = model.predict(X_test)
print(classification_report(prediction, y_test))


# In[ ]:




