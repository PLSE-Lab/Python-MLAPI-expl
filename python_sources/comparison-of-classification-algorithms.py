#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import sklearn.metrics
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, classification_report

#import plotly.offline as py
#py.init_notebook_mode(connected=True)
#import plotly.graph_objs as go
#import plotly.tools as tls

# Import and suppress warnings
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


data = pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')
data.head()


# In[ ]:


data.isnull().any()


# In[ ]:


numerical = [u'Age', u'DailyRate', u'DistanceFromHome', u'Education', u'EmployeeNumber', u'EnvironmentSatisfaction',
       u'HourlyRate', u'JobInvolvement', u'JobLevel', u'JobSatisfaction',
       u'MonthlyIncome', u'MonthlyRate', u'NumCompaniesWorked',
       u'PercentSalaryHike', u'PerformanceRating', u'RelationshipSatisfaction',
       u'StockOptionLevel', u'TotalWorkingYears',
       u'TrainingTimesLastYear', u'WorkLifeBalance', u'YearsAtCompany',
       u'YearsInCurrentRole', u'YearsSinceLastPromotion',
       u'YearsWithCurrManager']
data1=data[numerical]
a4_dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.heatmap(data1.corr())


# In[ ]:


data.dtypes


# In[ ]:


obj_data = data.select_dtypes(include=['object']).copy()
obj_data.head()


# In[ ]:



categorical = []
for col, value in data.iteritems():
    if value.dtype == 'object':
        categorical.append(col)

# Store the numerical columns in a list numerical
numerical = data.columns.difference(categorical)


# In[ ]:


attrition_cat = data[categorical]
attrition_cat = attrition_cat.drop(['Attrition'], axis=1) # Dropping the target column


# In[ ]:


attrition_cat = pd.get_dummies(attrition_cat)
attrition_cat.head(3)


# In[ ]:


# Store the numerical features to a dataframe attrition_num
attrition_num = data[numerical]


# In[ ]:


data_final = pd.concat([attrition_num, attrition_cat], axis=1)


# In[ ]:


target_map = {'Yes':1, 'No':0}
# Use the pandas apply method to numerically encode our attrition target variable
target = data["Attrition"].apply(lambda x: target_map[x])
target.head(3)


# In[ ]:


train, test, target_train, target_val = train_test_split(data_final, target, train_size= 0.75,random_state=0);


# # Decision tree

# In[ ]:


from sklearn import tree


decision_tree = tree.DecisionTreeClassifier(max_depth = 5)
decision_tree.fit(train, target_train)

# Predicting results for test dataset
y_pred1 = decision_tree.predict(test)


from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(decision_tree, out_file=dot_data,feature_names=list(train),class_names=['0','1'],
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# In[ ]:


accuracy_tree = sklearn.metrics.accuracy_score(target_val, y_pred1)
print(accuracy_tree)


# In[ ]:


sklearn.metrics.confusion_matrix(target_val, y_pred1,sample_weight=None)
report = classification_report(target_val, y_pred1)
print(report)


# In[ ]:


kfold = model_selection.KFold(n_splits=10, random_state=None)
cv_results = model_selection.cross_val_score(decision_tree, train, target_train, cv=kfold, scoring='accuracy')
cv_results.mean()


# # Logistic Regression

# In[ ]:


from sklearn import model_selection
from sklearn import metrics
from sklearn import feature_selection
from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression


# In[ ]:


model = LogisticRegression()

# Fit Recursive Feature Extraction with our model to the data
# This intuitively tries combinations of 26 features that maximizes our accuracy
selector = feature_selection.RFE(model, 26, step=1) # I did an exhaustive search of all the number of features to try. 26 features is just the best
selector = selector.fit(train, target_train)



y_pred2 = selector.predict(test)

# Get Accuracy
accuracy_logistic = metrics.accuracy_score(target_val, y_pred2)
print(accuracy_logistic)


# In[ ]:


sklearn.metrics.confusion_matrix(target_val, y_pred2,sample_weight=None)
report = classification_report(target_val, y_pred2)
print(report)


# In[ ]:


kfold = model_selection.KFold(n_splits=10, random_state=None)
cv_results = model_selection.cross_val_score(selector, train, target_train, cv=kfold, scoring='accuracy')
cv_results.mean()


# ## Linear Regression

# In[ ]:


from sklearn import linear_model

# Create linear regression object
linear = linear_model.LinearRegression()
# Train the model using the training sets and check score
linear.fit(train, target_train)
linear.score(train, target_train)
#Equation coefficient and Intercept
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)
#Predict Output
y_pred3= linear.predict(test)


# In[ ]:


y_pred3_new=[]
for i in y_pred3:
    if i<0.7:
        i=0
        y_pred3_new.append(i)
        
    else:
        i=1
        y_pred3_new.append(i)


# In[ ]:



accuracy_linear = metrics.accuracy_score(target_val, y_pred3_new)
print(accuracy_linear)


# In[ ]:


sklearn.metrics.confusion_matrix(target_val, y_pred3_new,sample_weight=None)
report = classification_report(target_val, y_pred2)
print(report)


# **As we can see Logistic Regression performs the best in the above case, with the highest accuracy. Let me know if yu have any suggestions or ideas. This is my first submission on Kaggle.**

# In[ ]:




