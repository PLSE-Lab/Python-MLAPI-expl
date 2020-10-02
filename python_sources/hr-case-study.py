#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


#Importing the datasets
employee_df = pd.read_csv('/kaggle/input/hr-analytics-case-study/employee_survey_data.csv')
manager_df = pd.read_csv('/kaggle/input/hr-analytics-case-study/manager_survey_data.csv')
general_df = pd.read_csv('/kaggle/input/hr-analytics-case-study/general_data.csv')
In_df = pd.read_csv('/kaggle/input/hr-analytics-case-study/in_time.csv')
Out_df = pd.read_csv('/kaggle/input/hr-analytics-case-study/out_time.csv')


# **We have 5 CSV's given, Their details are as below:**
# * **employee_survey_data.csv** - having detail of survey taken by employess
# * **manager_survey_data.csv** - having details of ratings given by manager to employess
# * **general_data.csv** - having general details regarding employess
# * **in_time.csv** - having details of in time of employess over a year
# * **out_time.csv** - having details of out time of employess over a year

# # Exploratory Data Analysis

# **Checking general_df**

# In[ ]:


general_df.info()


# In[ ]:


general_df.describe()


# In[ ]:


general_df.isnull().sum()


# In[ ]:


sns.pairplot(general_df, hue='Attrition',  diag_kind='hist')


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(general_df.corr(),cmap ="YlGnBu", linewidths = 0.1, ax = ax)


# From the above tho plots **pairplot and heatmap**, it can be inferred that there are some variables which are co related to each other, however most of the variables are independent and not corellated.
# 
# Now further we will be analyzing the categorical variable through count plot, For this we have created a list **feature_cat** having the required categorical variable, Lets see how the plot comes out and what can be inferred from this.

# In[ ]:


feature_cat = ['BusinessTravel', 'Department', 'Education', 'EducationField', 'Gender', 'JobLevel',  'MaritalStatus', 'NumCompaniesWorked', 'StockOptionLevel'
           ,  'TrainingTimesLastYear', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
i = 0
j = 0
fig, ax = plt.subplots(4, 3, figsize=(24, 20))
for feature in feature_cat:
    sns.countplot(x=feature, hue='Attrition', data=general_df, ax = ax[i,j])
    j = j + 1
    if j > 2:
        j = 0
        i = i + 1 


# From this plot we can see that for different categories what are the number of people who stayed and who left.
# Observetions are as follows:
# * Employess those travel very frequently are most likely to leave.
# * A lot of people from R&D department has left there can be some issue in this department, which need to be analysed.
# * People from HR department also leaves the company more frequently.
# * Single Employess has a high tendency to leave.
# * Employess for whom this is second company are most likely to leave.
# * If an employee promotion is pending from almost 6-7 years, he is likely to leave.

# Now to futher we will pe plotting pie chart to deeply analyse above points.

# In[ ]:


def pie(category):
    label = []
    label_percent = []
    for cat in general_df[category].unique():
        label.append(cat)
        t1 = general_df[(general_df[category] == cat) & (general_df['Attrition'] == 'Yes')].shape[0]
        t2 = general_df[general_df[category] == cat].shape[0]
        label_percent.append(t1/t2 * 100)
    fig1, ax1 = plt.subplots()
    ax1.pie(label_percent, labels=label, autopct='%1.1f%%', shadow=True, startangle=180)
    centre_circle = plt.Circle((0,0),0.75,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    ax1.axis('equal')
    plt.title(category)
    plt.show()
        
pie('EducationField')


# In[ ]:


pie('Department')


# As obeserved earlier people from HR background has high tendency to leave.

# In[ ]:


pie('BusinessTravel')


# From the above graph it seems like company need to check its policy on travelling as almost 83% people who travels has a tendency to leave
# Among them frequent travellers has a higher possibility of leaving the company.

# In[ ]:


pie('MaritalStatus')


# Married people havbe 50% chances of leaving the company.

# Now we will be analysing 4 parameters **(Age, YearsAtCompany, MonthlyIncome, NumCompaniesWorked)**
# wrt **BusinessTravel, EducationField and Department**

# In[ ]:


def violin_plot(main_cat):
    fig, ax = plt.subplots(figsize=(24, 0.1))
    plt.title(main_cat)
    plt.axis('off')
    plt.show
    sns.catplot(x=main_cat, y='Age', data=general_df, hue='Attrition', kind='violin', bw=0.2, col='Gender', inner='quartile', height=6, aspect=1.5)
    sns.catplot(x=main_cat, y='YearsAtCompany', data=general_df, hue='Attrition',kind='violin', bw=0.2, col='Gender', inner='quartile', height=6, aspect=1.5)
    sns.catplot(x=main_cat, y='MonthlyIncome', data=general_df, hue='Attrition',kind='violin', bw=0.2, col='Gender', inner='quartile', height=6, aspect=1.5)
    sns.catplot(x=main_cat, y='NumCompaniesWorked', data=general_df, hue='Attrition',kind='violin', bw=0.2, col='Gender', inner='quartile', height=6, aspect=1.5)
    


# In[ ]:


violin_plot('BusinessTravel')


# In[ ]:


violin_plot('Department')


# In[ ]:


violin_plot('EducationField')


# From the above violin plots it can be seen the frquency distribution for attrition among male and female for various parameters.

# **Checking employee_df**

# In[ ]:


employee_df.info()


# In[ ]:


employee_df.describe()


# In[ ]:


employee_df.isnull().sum()


# **Checking manager_df**

# In[ ]:


manager_df.info()


# In[ ]:


manager_df.describe()


# In[ ]:


manager_df.isnull().sum()


# **Final Analysis**
# * Company should check its policy wrt HR Department as attrition rate is almost 50% there.
# * Frequent travelers are most likely to leave hence travelling compensation should be revised to stop the attrition.
# * While recruting people it should be kept in mind that the people for whom it will be the second company are most likely to leave.
# * Policy regarding promotion should be reviewed and promotion should not be kept pending for 4+ years
# 

# # Data Cleaning

# We are having null values in employee_df and general_df, lets see how to handel them.

# **cleaning employee_df**

# In[ ]:


print(employee_df['EnvironmentSatisfaction'].median())
print(employee_df['JobSatisfaction'].median())
print(employee_df['WorkLifeBalance'].median())


# In[ ]:


#since all the 3 columns haas 3 as median value so we fill null value with 3 rating
employee_df.fillna(3.0, inplace=True)
employee_df.isnull().sum()


# **cleaning general_df**

# Since Only two colums NumCompaniesWorked and TotalWorkingYears have null values, lets analyse these one at a time

# In[ ]:


general_df[general_df['NumCompaniesWorked'].isnull()]


# In[ ]:


general_df[general_df['TotalWorkingYears'].isnull()]


# * Since both the parameter are very important in respect of attrition and we have no way to guess these values so its better to drop the data corresponding to null values as there are a toatl of 27 null values in 4410 employee.
# * We will drop thes values after merging all the dataframe
# 
# * Since we have to predict the probability of an employee leaving or staying thus we need to consolidate all data in a single dataframe
# 
# * Lets start with in_time and out_time
# 
# * We will find the diff of time between in and out time of the day and then find the average over time and rate the employee on basis of the average time

# In[ ]:


#Correcting the column name
In_df.rename(columns={'Unnamed: 0' : 'EmployeeID'}, inplace=True)
Out_df.rename(columns={'Unnamed: 0' : 'EmployeeID'}, inplace=True)


# In[ ]:


#Code for generating average in out time diff
time_diff = []
for i in range(4410):
    time_diff.append([])
    for j in range(262):
        time_diff[i].append(j)
time_diff_sec = []
for i in range(4410):
    time_diff_sec.append([])
    for j in range(262):
        time_diff_sec[i].append(j)

from datetime import datetime
In_df.fillna(In_df.iloc[2,2], inplace=True)
Out_df.fillna(In_df.iloc[2,2], inplace=True)
for i in range(0,4410):
    for j in range(1, 262):
        In_Time = In_df.iloc[i,j]
        out_Time = Out_df.iloc[i,j]
        d1 = datetime.strptime(In_Time, "%Y-%m-%d %H:%M:%S")
        d2 = datetime.strptime(out_Time, "%Y-%m-%d %H:%M:%S")
        time_diff[i][j] = d2 - d1
        a = d2.hour*3600 + d2.minute*60 + d2.second
        b = d1.hour*3600 + d1.minute*60 + d1.second
        time_diff_sec[i][j] = a - b
        
In_Out_Diff = pd.DataFrame(time_diff, columns=In_df.columns)
In_Out_Diff_sec = pd.DataFrame(time_diff_sec, columns=In_df.columns)
In_Out_Diff['EmployeeID'] = In_df['EmployeeID']
In_Out_Diff_sec['EmployeeID'] = In_df['EmployeeID']        


# In[ ]:


#Time format in seconds
In_Out_Diff_sec.head(5)


# In[ ]:


#Time format in HH:MM:SS
In_Out_Diff.head(5)


# In[ ]:


#Code for generating mean time for each employee
import datetime
mean_time = []
mean_sec = []
for i in range(4410):
    mean = In_Out_Diff_sec.iloc[i : i+1, 1: ].values.mean()
    mean_format = str(datetime.timedelta(seconds = mean))
    mean_sec.append(mean)
    mean_time.append(mean_format)

In_Out_Diff['Mean_Sec'] = mean_sec
In_Out_Diff['Mean'] = mean_time
In_Out_Diff.head()


# In[ ]:


Emp_in_out = In_Out_Diff[['EmployeeID', 'Mean', 'Mean_Sec']]
#Code for giving rating basis on meant time
#25200 - less than this - 1
#between two - 2
#30600 - moe than - 3
rating = []
for i in range(Emp_in_out.shape[0]):
    mean_time = Emp_in_out['Mean_Sec'].iloc[i]
    if mean_time < 25200:
        rating.append(1) 
    elif (mean_time > 25200) & (mean_time < 30600):
        rating.append(2) 
    else:
        rating.append(3)
rating
Emp_in_out['Work_Time_Rating'] = rating

Emp_in_out.head()


# In[ ]:


Employee = general_df.merge(Emp_in_out[['EmployeeID','Work_Time_Rating']], how='outer', on='EmployeeID')
Employee = Employee.merge(employee_df, how='outer', on='EmployeeID')
Employee = Employee.merge(manager_df, how='outer', on='EmployeeID')
df = Employee[['EmployeeID']]
Employee = Employee.drop('EmployeeID', axis=1)
Employee = pd.concat([df, Employee], axis=1)
df = Employee[['Attrition']]
Employee = Employee.drop('Attrition', axis=1)
Employee = pd.concat([Employee, df], axis=1)


# In[ ]:


Employee.isnull().sum()


# In[ ]:


#Droping Null values
Employee.dropna(inplace=True)
Employee.isnull().sum()


# In[ ]:


a = ['Attrition', 'BusinessTravel', 'Department', 'Education', 'EducationField', 'EmployeeCount', 'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'StandardHours', 'StockOptionLevel', 'TrainingTimesLastYear', 'Work_Time_Rating', 'EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance', 'JobInvolvement', 'PerformanceRating']
for col in a:
    print( '_______________________________________  ' + col + '  _______________________________________' )
    print(str(Employee[col].unique()))
    print( '_________________________________________________________________________________' )


# * **To be dropped** - EmployeeCount , Over18 , StandardHours
# * **To be Encoded **- Attrition , BusinessTravel , Department , EducationField , Gender , JobRole , MaritalStatus, StockOptionLevel

# In[ ]:


Employee.drop(['EmployeeCount', 'Over18', 'StandardHours'], axis=1, inplace=True)


# In[ ]:


Employee


# # Machine Learning Model

# Now based on the data provided by the company lets develope a Machine Learning model to predict the attrition.

# In[ ]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# In[ ]:


def encoder(X_i):
    label_encoder = LabelEncoder()
    X_i =  label_encoder.fit_transform(X_i)
    return X_i


# In[ ]:


temp = Employee.drop('Attrition', axis=1)


# In[ ]:


#Transforming BussineesTravel
temp['BusinessTravel'] = encoder(temp['BusinessTravel'])

#Transforming Department
temp['Department'] = encoder(temp['Department'])

#Transforming EducationField
temp['EducationField'] = encoder(temp['EducationField'])

#Transforming Gender
temp['Gender'] = encoder(temp['Gender'])

#Transforming JobRole
temp['JobRole'] = encoder(temp['JobRole'])

#Transforming MaritialStatus
temp['MaritalStatus'] = encoder(temp['MaritalStatus'])

#Transforming StockOptionLevel
temp['StockOptionLevel'] = encoder(temp['StockOptionLevel'])


# In[ ]:


temp


# In[ ]:


i = temp[['BusinessTravel', 'Department', 'EducationField', 'Gender' , 'JobRole', 'MaritalStatus']]
j = temp.drop(['BusinessTravel', 'Department', 'EducationField', 'Gender' , 'JobRole', 'MaritalStatus'], axis=1)


# In[ ]:


onehotencoder = OneHotEncoder(categorical_features=None, categories ='auto' , drop = 'first'  )
i = onehotencoder.fit_transform(i).toarray()


# In[ ]:


X = np.concatenate((j, i),axis=1)
pd.DataFrame(X)


# In[ ]:


X = X[:, 1:]
pd.DataFrame(X)


# In[ ]:


Y = Employee.iloc[:, 26].values
label_encoder1 = LabelEncoder()
Y =  label_encoder1.fit_transform(Y)
Y


# In[ ]:


#Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X  = sc_x.fit_transform(X)
pd.DataFrame(X)


# In[ ]:


#Train Test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state =0 )


# In[ ]:


#Logistic Regression

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


# In[ ]:


#SVC
from sklearn.svm import SVC
classifier_svc = SVC(kernel='rbf', random_state = 0)
classifier_svc.fit(X_train, y_train)
y_pred = classifier_svc.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm


# In[ ]:


#Decision tree
from sklearn.tree import DecisionTreeClassifier
classifier_dt = DecisionTreeClassifier()
classifier_dt.fit(X_train, y_train)
y_pred = classifier_dt.predict(X_test)
cm_dt = confusion_matrix(y_test, y_pred)
cm_dt


# In[ ]:


#Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators=50, criterion='gini')
classifier_rf.fit(X_train, y_train)
y_pred = classifier_rf.predict(X_test)
cm_rf = confusion_matrix(y_test, y_pred)
cm_rf


# **Evaluating the Model Performance**

# In[ ]:


#K Fold Cross Validation for logistic Regression
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y = y_train, cv = 10)
accuracies.mean()


# In[ ]:


#K Fold Cross Validation for svc
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier_svc, X=X_train, y = y_train, cv = 10)
accuracies.mean()


# In[ ]:


#K Fold Cross Validation for descision tree
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier_dt, X=X_train, y = y_train, cv = 10)
accuracies.mean()


# In[ ]:


#K Fold Cross Validation for random forest
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier_rf, X=X_train, y = y_train, cv = 10)
accuracies.mean()


# **parameter tuning using Grid Search**

# In[ ]:


#Model tuning for Logistic Regression
from sklearn.model_selection import GridSearchCV
parameters = [{ 'C' : [ 0.5, 1, 5 , 10], 'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag' ],'penalty' : ['l2']},
               {'C' : [ 0.5, 1, 5 , 10], 'solver' : [ 'saga'],'penalty' : ['elasticnet'], 'l1_ratio' : [0.1, 0.2, 0.3]}
              ]
grid_search = GridSearchCV(estimator=classifier, param_grid = parameters, scoring = 'accuracy', cv=10)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameter = grid_search.best_params_
print(best_accuracy)
print(best_parameter)


# In[ ]:


#Model Tuning for SVC
from sklearn.model_selection import GridSearchCV
parameters = [
               {'C' : [ 0.5, 1, 5 ], 'gamma' : [0.1, 0.2],'kernel' : ['poly',  'sigmoid'], 'coef0' : [0.05, 0.1]},
                 {'C' : [ 0.5, 1, 5 ], 'gamma' : [0.1, 0.2],'kernel' : [ 'rbf']}
              ]
grid_search = GridSearchCV(estimator=classifier_svc, param_grid = parameters, scoring = 'accuracy', cv=10)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameter = grid_search.best_params_
print(best_accuracy)
print(best_parameter)


# In[ ]:


#Model Tuning for Decision Tree
from sklearn.model_selection import GridSearchCV
parameters = [
               { 'criterion' : ['gini', 'entropy'], }
              ]
grid_search = GridSearchCV(estimator=classifier_dt, param_grid = parameters, scoring = 'accuracy', cv=10)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameter = grid_search.best_params_
print(best_accuracy)
print(best_parameter)


# In[ ]:


#Model Tuning for Random Forest
from sklearn.model_selection import GridSearchCV
parameters = [
               {'n_estimators' : [ 10, 100, 200 ], 'criterion' : ['gini', 'entropy']}
              ]
grid_search = GridSearchCV(estimator=classifier_rf, param_grid = parameters, scoring = 'accuracy', cv=10)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameter = grid_search.best_params_
print(best_accuracy)
print(best_parameter)


# **Running the model with best parameters**

# In[ ]:


classifier_rf = RandomForestClassifier(n_estimators=200, criterion='gini')
classifier_rf.fit(X_train, y_train)
y_pred = classifier_rf.predict(X_test)
cm_rf = confusion_matrix(y_test, y_pred)
cm_rf


# In[ ]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

