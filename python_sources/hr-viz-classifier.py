#!/usr/bin/env python
# coding: utf-8

# # Import Libraries 

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import statsmodels.formula.api as smf
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.metrics import precision_recall_fscore_support as score

from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from time import time

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.linear_model import Lasso, Ridge


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sb


# # Import Files

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df1 = pd.read_csv("../input/human-resources-data-set/core_dataset.csv")


# In[ ]:


df1.head()


# In[ ]:


df1.info()


# In[ ]:


df1.columns


# In[ ]:


df1.rename(columns={'Pay Rate': 'Pay'}, inplace=True)


# In[ ]:


df1['State'].value_counts()


# In[ ]:


df1.groupby(['Sex']).mean().Age.plot(kind='bar')


# In[ ]:


df1['Sex'].replace({'male': 'Male'}, inplace=True)


# In[ ]:


df1.groupby(['Sex']).mean().Age.plot(kind='bar')


# In[ ]:


df1.groupby(['Sex']).mean().Pay.plot(kind='bar')


# In[ ]:


df1.groupby(['Sex', 'Performance Score']).size().unstack().plot(kind='bar', stacked=True, figsize=(10,10))


# In[ ]:


df1['Paymean'] = df1['Pay'].mean()


plt.figure(figsize=(15,15))
sns.countplot(x='Sex', data=df1, hue = 'Performance Score', palette="Set1")
plt.title('Age vs. Sex')


# In[ ]:


plt.figure(figsize=(12,4))
sns.heatmap(df1.isnull(),cbar=False,cmap='viridis',yticklabels=False)
plt.title('Missing value in the dataset')


# In[ ]:


df1['Performance Score'].value_counts()


# In[ ]:


sns.pairplot(df1)


# In[ ]:


df1.corr()


# In[ ]:


df1.drop(columns=['Date of Termination'], inplace=True)


# In[ ]:


df1.columns


# In[ ]:


df1.drop(columns=['Employee Name', 'Employee Number', 'Reason For Term' , 'Hispanic/Latino', 'Zip', 'DOB', 'Date of Hire', 'Manager Name', 'Position', 'State', 'Paymean'], inplace=True)


# In[ ]:


df1.sample(10)


# In[ ]:


df1.columns


# In[ ]:


df1.dropna(inplace=True)


# In[ ]:


dfcorr = pd.get_dummies(df1)


# In[ ]:


dfcorr.info()


# In[ ]:



pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# In[ ]:


dfcorr.corr()


# In[ ]:


from sklearn import tree
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


dfcorr.columns


# In[ ]:


y = df1['Pay']
X = dfcorr.drop(['Pay'], axis=1)


# In[ ]:


dfcorr


# In[ ]:





# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


# In[ ]:


X, y = make_classification(n_samples=1000, n_features=4,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)


# In[ ]:


dfcorr.columns


# In[ ]:


clf = RandomForestClassifier(n_estimators=100, max_depth=2,
                             random_state=0)


# In[ ]:


df1['Pay'].unique()


# In[ ]:


feature_cols = ['Age', 'Pay', 'Sex_Female', 'Sex_Male', 'MaritalDesc_Divorced',
       'MaritalDesc_Married', 'MaritalDesc_Separated', 'MaritalDesc_Single',
       'MaritalDesc_widowed', 
       'Employment Status_Active', 'Employment Status_Future Start',
       'Employment Status_Leave of Absence',
       'Employment Status_Terminated for Cause',
       'Employment Status_Voluntarily Terminated', 'Department_Admin Offices',
       'Department_Executive Office', 'Department_IT/IS',
       'Department_Production       ', 'Department_Sales',
       'Department_Software Engineering',
       'Department_Software Engineering     ', 'Employee Source_Billboard',
       'Employee Source_Careerbuilder',
       'Employee Source_Company Intranet - Partner',
       'Employee Source_Diversity Job Fair',
       'Employee Source_Employee Referral', 'Employee Source_Glassdoor',
       'Employee Source_Information Session',
       'Employee Source_Internet Search', 'Employee Source_MBTA ads',
       'Employee Source_Monster.com', 'Employee Source_Newspager/Magazine',
       'Employee Source_On-campus Recruiting',
       'Employee Source_On-line Web application', 'Employee Source_Other',
       'Employee Source_Pay Per Click',
       'Employee Source_Pay Per Click - Google',
       'Employee Source_Professional Society',
       'Employee Source_Search Engine - Google Bing Yahoo',
       'Employee Source_Social Networks - Facebook Twitter etc',
       'Employee Source_Vendor Referral', 'Employee Source_Website Banner Ads',
       'Employee Source_Word of Mouth', 'Performance Score_90-day meets',
       'Performance Score_Exceptional',
       'Performance Score_N/A- too early to review',
       'Performance Score_Needs Improvement', 'Performance Score_PIP']
X = dfcorr[feature_cols]
y = dfcorr['Performance Score_Exceeds']
X_train, X_test, y_train, y_test=train_test_split(X, y, 
                                                  test_size = 0.25,
                                                  random_state = 123)


# In[ ]:


clf = RandomForestClassifier(n_estimators=500,criterion = 'entropy', max_depth=5,random_state=123)
clf.fit(X_train,y_train)


# In[ ]:


#Classify the test subset using .predict()
y_pred = clf.predict(X_test)


# In[ ]:



#Calculate the accuracy using metrics.accuracy_score(y_test,y_pred)
print("Accuracy:", metrics.accuracy_score(y_test,y_pred))


# In[ ]:


# y_test is a dataframe and to use the function confusion_matrix it is necessary to convert y_test to a list
from sklearn.metrics import confusion_matrix
y_true = y_test.tolist()
mat = confusion_matrix(y_true,y_pred)
sns.heatmap(mat.T, square =True, annot = True, fmt = 'd', cbar= False)
plt.xlabel('True data')
plt.ylabel('predicted values')


# In[ ]:


feature_importances = pd.Series(clf.feature_importances_, index = X.columns)
feature_importances = feature_importances.sort_values()
feature_importances.plot(kind='barh', figsize = (12,12))


# In[ ]:


from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
estimator = clf.estimators_[90]
export_graphviz(estimator, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names=X_train.columns,class_names = ['no', 'yes'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




