#!/usr/bin/env python
# coding: utf-8

# <h1>HR Analytics</h1>
# 
# Per the requirement of the competition I will be predicting which employees will be next to leave the company based on the data provided by Kaggle.

# In[ ]:


#import all nessary packages 
import pandas as pd
from pandas import DataFrame
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from IPython.display import HTML, display
get_ipython().run_line_magic('matplotlib', 'inline')


# Using the pandas package importing the data into a dataframe print out the info to see the data types of the imported data.  Check to see if there are any null values. 

# In[ ]:


#import the csv file into a panda data frame.

hr_df = pd.read_csv('../input/HR_comma_sep.csv')

hr_df.info()


# Convert the Sales, and Salary Feature to numerical values. 
# sales = 0,
# accounting = 1, 
# hr = 2, 
# technical = 3, 
# support = 4, 
# management = 5, 
# IT = 6,
# product_mng = 7, 
# marketing = 8, 
# RandD = 9
# 

# In[ ]:


hr_df['sales'].replace(['sales', 'accounting', 'hr', 'technical', 'support', 'management','IT','product_mng','marketing','RandD'],[0,1,2,3,4,5,6,7,8,9],
                      inplace=True)

hr_df['salary'].replace(['low','medium','high'],[0,1,2], inplace=True)


# In[ ]:


hr_df.head()


# <h1>Visualize the Data</h1>

# In[ ]:



correlation = hr_df.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')

plt.title('Correlation between different fearures')


# In[ ]:


plt.figure(figsize=(9,6))
sns.boxplot('number_project', 'last_evaluation', data=hr_df)


# In[ ]:


plt.figure(figsize=(9,5))
sns.boxplot('salary', 'satisfaction_level', data=hr_df)
plt.title('Average Satisfied Employee By Salary')


# In[ ]:


plt.figure(figsize=(9,5))
sns.distplot(hr_df['satisfaction_level'], kde=False)
plt.title('Total Employee by Satisfaction Level')
plt.ylabel('Employee Count')
plt.xlabel('Satisfaction Level')


# The Above Histogram shows the distribution of employee by satisfaction level.  We can tell by looking at the graph there is a majority of employee who are not satisfied.  I will explore in detail as to why employee are not satisfied.  

# Plotting Left column by department to determined how many employee have left and how many stayed.  1=Left and 0=Stay. Based on the plot below we can see that the sales department has the majority of the employees therefore they also have the majority of the employee who have left.   

# In[ ]:


sns.factorplot(x='left', col='sales', data=hr_df, kind='count', col_wrap=4,              size=2.5)
plt.title('Total Employee count by Left')


# In[ ]:


plt.figure(figsize=(18,10))
sns.pointplot('time_spend_company', 'satisfaction_level', hue='left', data=hr_df, kde=False)
plt.title('Employee Time Spend')
plt.ylabel('Avg Satisfaction')
plt.xlabel('Time Spend in Company')


# In[ ]:


plt.figure(figsize=(18,10))
x = np.arange(0, len(hr_df))
y = hr_df['satisfaction_level']

sns.regplot(x,y, 'o', scatter_kws={'s':15})


# <h1>Separete Left from Stay</h1>
# 
# Using the left column I am seperating both 0 Stays and 1 Left into two different data frames this will help us do our predictions.

# In[ ]:


hr_drop = hr_df.drop(['salary', 'sales'], axis=1)

hr_drop.head()


# In[ ]:


stay_df = hr_df[hr_drop['left']==0]

stay_fi = stay_df.drop(['left'], axis=1)


# In[ ]:


stay_df.shape


# In[ ]:


stay_fi.head()


# In[ ]:


from sklearn.model_selection import train_test_split
label = hr_df.pop('left')
data_train, data_tests, label_train, label_test = train_test_split(hr_df, label, test_size=0.2, random_state=42)


# In[ ]:


data_train.head()


# In[ ]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression
logis = LogisticRegression()
a = logis.fit(data_train, label_train)
y_pred = logis.predict(stay_fi)
logis_score_train = logis.score(data_train, label_train)
print("Training Score: ", logis_score_train)
logis_score_test = logis.score(data_tests, label_test)
print("Test Score: ", logis_score_test)


# In[ ]:


coeff_df = pd.DataFrame(hr_drop.columns.delete(0))
coeff_df.columns = ['Features']
coeff_df['Correlation'] = pd.Series(logis.coef_[0])
coeff_df.sort_values(by='Correlation', ascending=False) 


# In[ ]:


#SVN
from sklearn.svm import SVC 
svm = SVC() 
svm.fit(data_train, label_train)
y_pred = svm.predict(stay_fi)
svm_score_train = svm.score(data_train, label_train)
print('Training Score: ', svm_score_train)
svm_score_test = svm.score(data_tests, label_test)
print('Test Score: ', svm_score_test)


# In[ ]:


#KNN
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier() 
knn.fit(data_train, label_train)
y_pred = knn.predict(stay_fi)
knn_score_train = knn.score(data_train, label_train)
print('Training score: ', knn_score_train)
knn_score_test = knn.score(data_tests, label_test)
print('Test score: ', knn_score_test)


# In[ ]:


#Decision Tree
from sklearn import tree 
dt = tree.DecisionTreeClassifier() 
dt.fit(data_train, label_train)
y_pred = dt.predict(stay_fi)
dt_score_train = dt.score(data_train, label_train)
print('Training Score: ', dt_score_train)
dt_score_test = dt.score(data_tests, label_test)
print('Test Score: ', dt_score_test)


# In[ ]:


#Random Forrest 
from sklearn.ensemble import RandomForestClassifier 
rfc = RandomForestClassifier() 
rfc.fit(data_train, label_train)
y_pred = rfc.predict(stay_fi)
rfc_score_train = rfc.score(data_train, label_train)
print('Training Score: ', rfc_score_train)
rfc_score_test = rfc.score(data_tests, label_test)
print('Test Score: ', rfc_score_test)


# In[ ]:


models = pd.DataFrame({
    'Model' : ['Logistic Regression', 'SVM', 'kNN', 'Decision Tree', 'Random Forest'],
    'Training_Score' : [logis_score_train, svm_score_train, knn_score_train, dt_score_train, rfc_score_train],
    'Testing_Score' : [logis_score_test, svm_score_test, knn_score_test, dt_score_test, rfc_score_train]
})
models.sort_values(by='Testing_Score', ascending=False)


# In[ ]:




