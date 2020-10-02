#!/usr/bin/env python
# coding: utf-8

# # SoftServe DS Hackathon 2020
# 
# ![img](https://s.dou.ua/CACHE/images/img/announces/Hackathon_2-01_1_HRdziq2/3af11affb7e010c0d2440a7a4ed16181.png)

# ### Task
# 
# Solution should be made with respect of the following business goals:
# 
# 1. Managers should have a notification about Employees even in case of insignificant risk of Dismissal
# 2. Managers should get an information regarding Dismissal drivers - the interpretability of the prediction result is desirable
# 3. Managers should get the recommendations regarding potential actions for Employees retention.

# ### Content
# 
# 1. [Import](#Import)
# 2. [Constants](#Constants)
# 3. [Read data](#Read-data)
# 4. [Data research](#Data-research)
# 5. [Visualization](#Visualization)
# 6. [Data preparation](#Data-preparation)
# 7. [Train preparation](#Train-preparation)
# 8. [Split data](#Split-data)
# 9. [Logistic Regression coef research](#Logistic-Regression-coef-research)
# 10. [Training](#Training)
# 11. [Evaluation](#Evaluation)
# 12. [Submission](#Submission)
# 13. [Analysis of the historical data and modelling results](#Analysis-of-the-historical-data-and-modelling-results)

# # Tableau Visualization

# In[ ]:


get_ipython().run_cell_magic('html', '', "\n<div class='tableauPlaceholder' id='viz1589830533527' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Dy&#47;DynamicHRstatistics&#47;DynamicHRstatistics&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='DynamicHRstatistics&#47;DynamicHRstatistics' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Dy&#47;DynamicHRstatistics&#47;DynamicHRstatistics&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1589830533527');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.minWidth='420px';vizElement.style.maxWidth='650px';vizElement.style.width='100%';vizElement.style.minHeight='587px';vizElement.style.maxHeight='887px';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.minWidth='420px';vizElement.style.maxWidth='650px';vizElement.style.width='100%';vizElement.style.minHeight='587px';vizElement.style.maxHeight='887px';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else { vizElement.style.width='100%';vizElement.style.height='977px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# In[ ]:





# In[ ]:





# # Import

# In[ ]:


import os


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from tqdm import tqdm


# # Constants

# In[ ]:


DATA_PATH = "/kaggle/input/softserve-ds-hackathon-2020/"


# # Read data

# In[ ]:


employees_df = pd.read_csv(os.path.join(DATA_PATH, "employees.csv"))
submission_df = pd.read_csv(os.path.join(DATA_PATH, "submission.csv"))
history_df = pd.read_csv(os.path.join(DATA_PATH, "history.csv"))


# In[ ]:





# # Join tables
# 

# In[ ]:


# join history to employees by EmployeeID

history_df = history_df.join(employees_df.set_index('EmployeeID'), on='EmployeeID')


# In[ ]:


# create target
history_df['target'] = 0
history_df.target.loc[pd.notnull(history_df.DismissalDate)] = 1


# In[ ]:


# date columns to date time
history_df.Date = pd.to_datetime(history_df.Date)
employees_df.HiringDate = pd.to_datetime(employees_df.HiringDate)
employees_df.DismissalDate = pd.to_datetime(employees_df.DismissalDate)


# # Data research

# ### Employees dataframe 

# In[ ]:


print('Shape: ', employees_df.shape)
print('Columns: ', employees_df.columns)
employees_df.head()


# In[ ]:


print('Number of unique employees: ', len(employees_df.EmployeeID.unique()))
employees_df.isna().sum()


# ### history dataframe

# In[ ]:


print('Shape: ', history_df.shape)
print('Number of unique employees: ', len(history_df.EmployeeID.unique()))
print('Columns list: ', history_df.columns)
history_df.head()


# ### submission dataframe

# In[ ]:


print('Shape: ', submission_df.shape)
print('Number of unique employees: ', len(submission_df.EmployeeID.unique()))
submission_df.head()


# # Visualization

# In[ ]:


def value_counts_bar_chart(column_name):
    sns.set(font_scale=1.4)
    history_df[column_name].loc[history_df.target == 1].value_counts()[:20].plot(kind='bar', figsize=(15, 6), rot=0)
    plt.xlabel(column_name, labelpad=14)
    plt.ylabel("Count of Dismissal", labelpad=14)
    plt.title("Count of Dismissal by "+column_name, y=1.02)


# In[ ]:


def value_counts_bar_chart_last_year(column_name):
    last_year = history_df.Date.max() - pd.DateOffset(years=1)

    sns.set(font_scale=1.4)
    history_df[column_name].loc[(history_df.target == 1)&(history_df.Date>=last_year)].value_counts()[:20].plot(kind='bar', figsize=(15, 6), rot=0)
    plt.xlabel(column_name, labelpad=14)
    plt.ylabel("Count of Dismissal", labelpad=14)
    plt.title("Count of Dismissal by "+column_name, y=1.02);


# ### DevCenterID
# - DevCenterID - Employee Location in terms of Company Geography

# In[ ]:


value_counts_bar_chart('DevCenterID')


# In[ ]:


value_counts_bar_chart_last_year('DevCenterID')


# ### SBUID 
# 
# - Employee Location in terms of Company Structure

# In[ ]:


value_counts_bar_chart('SBUID')


# In[ ]:


value_counts_bar_chart_last_year('SBUID')


# ### PositionID 
# 
# - Identifier of Employee Position (like QC Engineer, Development Consultant, etc)

# In[ ]:


value_counts_bar_chart('PositionID')


# In[ ]:


value_counts_bar_chart_last_year('PositionID')


# ### IsTrainee 
# 
# - Trainee flag of Employee

# In[ ]:


value_counts_bar_chart('IsTrainee')


# In[ ]:


value_counts_bar_chart_last_year('IsTrainee')


# ### LanguageLevelID
# 
# - English Level Identifier (like Intermediate low, Upper-intermediate, etc)

# In[ ]:


value_counts_bar_chart('LanguageLevelID')


# In[ ]:


value_counts_bar_chart_last_year('LanguageLevelID')


# ### CustomerID 
# 
# - Client Identifier (one client may be related to the several projects)

# In[ ]:


value_counts_bar_chart('CustomerID')


# In[ ]:


value_counts_bar_chart_last_year('CustomerID')


# ### ProjectID 
# 
# - Employee Main Project Identifier

# In[ ]:


value_counts_bar_chart('ProjectID')


# In[ ]:


value_counts_bar_chart_last_year('ProjectID')


# ### IsInternalProject 
# 
# - Internal / External project flag

# In[ ]:


value_counts_bar_chart('IsInternalProject')


# In[ ]:


value_counts_bar_chart_last_year('IsInternalProject')


# ### Utilization 
# 
# - percent of Employee load on Non-Internal Projects during last month

# In[ ]:


# value_counts_bar_chart('Utilization')


# In[ ]:


# value_counts_bar_chart_last_year('Utilization')


# ### HourVacation 
# 
# - vacation hours are spent as on the last month

# In[ ]:


value_counts_bar_chart('HourVacation')


# In[ ]:


value_counts_bar_chart_last_year('HourVacation')


# ### HourMobileReserve 
# 
# - total hours in Mobile reserve as on the last month

# In[ ]:


value_counts_bar_chart('HourMobileReserve')


# In[ ]:


value_counts_bar_chart_last_year('HourMobileReserve')


# ### HourLockedReserve 
# 
# - total hours in Locked reserve as on the last month

# In[ ]:


value_counts_bar_chart('HourLockedReserve')


# In[ ]:


value_counts_bar_chart_last_year('HourLockedReserve')


# ### OnSide 
# 
# - was Employee involved to OnSite visit last month

# In[ ]:


# value_counts_bar_chart('OnSide')


# In[ ]:


# value_counts_bar_chart_last_year('OnSide')


# ### MonthOnPosition 
# 
# - month without position changing as on the last month

# In[ ]:


value_counts_bar_chart('MonthOnPosition')


# In[ ]:


value_counts_bar_chart_last_year('MonthOnPosition')


# ### MonthOnSalary 
# 
# - month without salary increasing as on the last month

# In[ ]:


value_counts_bar_chart('MonthOnSalary')


# In[ ]:


value_counts_bar_chart_last_year('MonthOnSalary')


# ### CompetenceGroupID 
# 
# - Employee Competency Group (like QC, Big Data, Data Science, etc)

# In[ ]:


value_counts_bar_chart('CompetenceGroupID')


# In[ ]:


value_counts_bar_chart_last_year('CompetenceGroupID')


# ### FunctionalOfficeID 
# 
# - Functional Office Identifier (like SDO, QMO, etc)

# In[ ]:


value_counts_bar_chart('FunctionalOfficeID')


# In[ ]:


value_counts_bar_chart_last_year('FunctionalOfficeID')


# ### PaymentTypeId 
# 
# - Payment with respect to the country-specifics employment

# In[ ]:


value_counts_bar_chart('PaymentTypeId')


# In[ ]:


value_counts_bar_chart_last_year('PaymentTypeId')


# ### WageGross 
# 
# - Compensation GROSS

# In[ ]:


# value_counts_bar_chart('WageGross')


# In[ ]:


# value_counts_bar_chart_last_year('WageGross')


# ### BonusOneTime 
# 
# - One Time Bonus

# In[ ]:


value_counts_bar_chart('BonusOneTime')


# In[ ]:


value_counts_bar_chart_last_year('BonusOneTime')


# ### APM 
# 
# - Employee APM

# In[ ]:


# value_counts_bar_chart('APM')


# In[ ]:


# value_counts_bar_chart_last_year('APM')


# ### PositionLevel 
# 
# - Employee Seniority Level (Junior, Middle, Senior, etc)

# In[ ]:


value_counts_bar_chart('PositionLevel')


# In[ ]:


value_counts_bar_chart_last_year('PositionLevel')


# In[ ]:


history_df.columns


# # Data preparation

# In[ ]:


history_df.target.unique()


# In[ ]:


# encode CustomerID
labelencoder= LabelEncoder()
history_df['CustomerID'] = labelencoder.fit_transform(history_df['CustomerID'])


# In[ ]:


# encode ProjectID
history_df['ProjectID'] = history_df['ProjectID'].astype(str)
labelencoder= LabelEncoder()
history_df['ProjectID'] = labelencoder.fit_transform(history_df['ProjectID'])


# In[ ]:


# create column with month number
history_df['month'] = pd.DatetimeIndex(history_df['Date']).month


# In[ ]:


# add work duration

last_date = history_df.Date.max()
history_df.DismissalDate = history_df.DismissalDate.fillna(last_date)
history_df.DismissalDate.isna().sum()
history_df.DismissalDate = pd.to_datetime(history_df.DismissalDate)
history_df.HiringDate = pd.to_datetime(history_df.HiringDate)


history_df['workDuration'] = (history_df.DismissalDate - history_df.HiringDate) / np.timedelta64(1, 'M')


# In[ ]:


# Add mean and sum of bonuses

column_name = 'BonusOneTime'
history_df['BonusMean'] = 0
history_df['BonusSum'] = 0
    
for index, row in tqdm(history_df.iterrows()):  
    row_employee = row['EmployeeID']
    row_date = row['Date']
    
    row_bonuses = history_df[['Date', column_name]].loc[(history_df.EmployeeID==row_employee)&(history_df.Date<=row_date)]
    history_df.loc[index, 'BonusMean'] = row_bonuses[column_name].mean()
    history_df.loc[index, 'BonusSum'] = row_bonuses[column_name].sum()


# In[ ]:


history_df[['BonusMean', 'BonusSum']].describe()


# In[ ]:


# add mean and max-min MonthOnSalary

column_name = 'MonthOnSalary'
history_df['MonthOnSalaryMean'] = 0
history_df['MonthOnSalaryDifference'] = 0
    
for index, row in tqdm(history_df.iterrows()): 
    row_employee = row['EmployeeID']
    row_date = row['Date']
    
    row_value = history_df[['Date', column_name]].loc[(history_df.EmployeeID==row_employee)&(history_df.Date<=row_date)]
    history_df.loc[index, 'MonthOnSalaryMean'] = row_value[column_name].mean()
    history_df.loc[index, 'MonthOnSalaryDifference'] = row_value[column_name].sum()


# In[ ]:


history_df[['MonthOnSalaryMean', 'MonthOnSalaryDifference']].describe()


# # Train preparation

# In[ ]:


history_df.to_csv('prepareted_history.csv', index=False)


# In[ ]:


# history without last mount


train_history_df = history_df.loc[history_df.Date != history_df.Date.max()]
dev_history_df = history_df.loc[history_df.Date == history_df.Date.max()]


# In[ ]:


train_history_df.columns


# In[ ]:


drop_columns = [
    'EmployeeID',
    'Date',
    'HiringDate',
    'DismissalDate'
]


# In[ ]:


train = train_history_df.drop(drop_columns, axis=1)


# # Split data

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
    train.drop(['target'], axis=1), 
    train['target'], 
    test_size=0.3, 
    random_state=1
)


# # Logistic Regression coef research

# In[ ]:


def model_scoring(clf, X_test, y_test):
    predicted= clf.predict(X_test)
    print(classification_report(y_test, predicted))


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)
model_scoring(logreg, X_test, y_test)


# In[ ]:


coeff_df = pd.DataFrame(train.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False).round(5)


# # Random forest coef research

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


model = RandomForestClassifier().fit(X_train, y_train)


# In[ ]:


# Actual class predictions
rf_predictions = model.predict(X_test)
# Probabilities for each class
rf_probs = model.predict_proba(X_test)[:, 1]


# In[ ]:


# Extract feature importances
fi = pd.DataFrame(
    {'feature': list(X_train.columns),
     'importance': model.feature_importances_}
).sort_values('importance', ascending = False)

# Display
fi


# # Training

# In[ ]:


# Note: we can try one class classification: https://machinelearningmastery.com/one-class-classification-algorithms/


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from lightgbm import LGBMClassifier


# In[ ]:


# clf = RandomForestClassifier().fit(X_train, y_train) # give 0.20435
clf = GradientBoostingClassifier().fit(X_train, y_train) # give 0.21048


# In[ ]:


# clf = LGBMClassifier()
# clf.fit(X_train, y_train)


# # Evaluation

# In[ ]:


model_scoring(clf, X_test, y_test)


# # Submission

# In[ ]:


for employee in submission_df.EmployeeID.unique():
    input_data = dev_history_df.loc[dev_history_df.EmployeeID == employee]
    input_data = input_data.drop(drop_columns, axis=1)
    input_data = input_data.drop(['target'], axis=1)
    # predict = clf.predict(input_data)[0]
    predict = clf.predict_proba(input_data)[0]
    
    if predict[1] >= 0.1 or predict[0] < 0.9:
        predict = 1
    else: 
        predict = 0
    
    submission_df.target.loc[submission_df.EmployeeID==employee] = predict


# In[ ]:


submission_df.target.value_counts()


# In[ ]:


submission_df.to_csv('submission.csv', index=False)


# # Analysis of the historical data and modelling results

# Our database has a lot of columns, and I don't have much time for in-depth analysis. Therefore, I need to prioritize features to research. I used logistic regression for this. (*See [Logistic Regression coef research](#Logistic-Regression-coef-research)*). I chose the largest and smallest coefficients of the trained model. There are:
# 
# - **HourLockedReserve**	= 0.01237
# - **SBUID**	= 0.00768
# - **HourMobileReserve**	= 0.00578
# 
# - **MonthOnSalary**	= -0.01054
# - **workDuration**	= -0.02227
# - **BonusOneTime**	= -0.03363 
# 
# It looks like, that first three item can affect negatively, and last three item have a positive effect. But Logistic Regression can not be source for conclusions, because it did not give high test results. Therefore, we need to visualize these indicators under different conditions.

# ### BonusOneTime

# In[ ]:


history_df_0 = history_df.loc[history_df.target == 0] # those who resigned
history_df_1 = history_df.loc[history_df.target == 1] # those who not resigned


# In[ ]:


print('Max bonus: ', history_df.BonusOneTime.unique().max())
print('Min bonus: ', history_df.BonusOneTime.unique().min())
print('Mean bonus: ', history_df.BonusOneTime.unique().mean())


# In[ ]:


print('Max bonus: ', history_df_0.BonusOneTime.unique().max())
print('Min bonus: ', history_df_0.BonusOneTime.unique().min())
print('Mean bonus: ', history_df_0.BonusOneTime.unique().mean())


# In[ ]:


print('Max bonus: ', history_df_1.BonusOneTime.unique().max())
print('Min bonus: ', history_df_1.BonusOneTime.unique().min())
print('Mean bonus: ', history_df_1.BonusOneTime.unique().mean())


# In[ ]:


history_df.BonusOneTime.value_counts()[:10]/history_df.shape[0]*100


# In[ ]:


history_df_0.BonusOneTime.value_counts()[:10]/history_df_0.shape[0]*100


# In[ ]:


history_df_1.BonusOneTime.value_counts()[:10]/history_df_1.shape[0]*100


# Bonuses seem to reduce the risk of dismissal. See comparing in table:
# 
# | fired employee|     |not fired employee  ||
# |----------|----------|----------|----------|
# | Bonus |% in category|Bonus  |% in category|
# | 0        | 87.86    | 0       | 82.83    |
# | 100      | 1.43     | 100     | 1.03     |
# | 200      | 1.07     | 200     | 0.71     |
#  
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




