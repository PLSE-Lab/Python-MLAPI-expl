#!/usr/bin/env python
# coding: utf-8

# In[61]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve
from sklearn.metrics import  recall_score, classification_report
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

# Plotting the graphs
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import plotly.graph_objs as go
import plotly.offline as py
py.init_notebook_mode(connected=True)

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# > **Data**

# In[62]:


#Importing data
lowafilepath = '../input/WA_Fn-UseC_-Telco-Customer-Churn.csv'
data = pd.read_csv(lowafilepath)


# In[63]:


data.head()


# In[64]:


print ("Rows     : " ,data.shape[0])
print ("Columns  : " ,data.shape[1])
print ("\nFeatures : \n" ,data.columns.tolist())
print("\nData Information : \n",data.info())


# Column TotalCharges is having some missing values

# > **Missing Value Removal**

# In[65]:


# Removing the missing values
data = data[pd.notnull(data['TotalCharges'])]
print("Number of null values in total charges:",data['TotalCharges'].isna().sum())
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'],errors='coerce')


# > **EDA**

# In[66]:


sns.pairplot(data,vars = ['tenure','MonthlyCharges','TotalCharges'], hue="Churn")


# People having lower tenure and higher monthly charges are tend to churn more.

# In[67]:


# Getting the categorical variables 
all_cat_var = data.nunique()[data.nunique()<5].keys().tolist()

# Getting the categorical variables without churn
cat_var = all_cat_var[:-1]


# In[68]:


def pie_plot(Column):    
    ct1 = pd.crosstab(data[Column],data['Churn'])
    trace1 = go.Pie(labels = ct1.index,
                    values = ct1.iloc[:,0],
                    hole=0.3,
                    domain=dict(x=[0,.45]))
    trace2 = go.Pie(labels = ct1.index,
                    values = ct1.iloc[:,1],
                    domain=dict(x=[.55,1]),
                    hole=0.3)

    layout = go.Layout(dict(title = Column + " distribution in customer attrition ",
                                plot_bgcolor  = "rgb(243,243,243)",
                                paper_bgcolor = "rgb(243,243,243)",
                                annotations = [dict(text = "churn customers",
                                                    font = dict(size = 13),
                                                    showarrow = False,
                                                    x = .15, y = 1),
                                               dict(text = "Non churn customers",
                                                    font = dict(size = 13),
                                                    showarrow = False,
                                                    x = .88,y = 1)

                                              ]
                               )
                          )

    fig = go.Figure(data=[trace1,trace2],layout=layout)
    py.iplot(fig)


# In[69]:


for i in cat_var:
    pie_plot(i)


# In[70]:


# Removing the customer id
del data['customerID'] #customerID is a uninque id so it dosn't give any information


# > **Statistical Test**

# **Chi-square Test for Feature Extraction:**
# Chi-square test is used for categorical features in a dataset. We calculate Chi-square between each feature and the target and select the desired number of features with best Chi-square scores.It determines if the association between two categorical variables of the sample would reflect their real association in the population.

# In[71]:


# chi square test 

i = 0
for nam in cat_var:
    crosstab = pd.crosstab(data[nam], data['Churn'])
#     crosstab
    chi = stats.chi2_contingency(crosstab)
    if chi[1]<0.05:
        i=i+1
        print(i,nam, " is important for predicting churn with p value: ",chi[1] )


# Columns such as Gender and PhoneService are not important in predicting the churn

# > **Label Encoding**

# Label encodingrefers to converting the labels into numeric form so as to convert it into the machine-readable form. Machine learning algorithms can then decide in a better way on how those labels must be operated. It is an important pre-processing step for the structured dataset in supervised learning.

# In[72]:


le = LabelEncoder()

# apply "le.fit_transform"
data_new = data.apply(le.fit_transform)


# Finding out the Multicollinearity using VIF

# In[73]:


X = add_constant(data_new)
pd.Series([variance_inflation_factor(X.values, i) 
           for i in range(X.shape[1])], index=X.columns)


# Total charges and tenure is having high VIF score which means multicoliniarity

# In[74]:


# Splitiing to x and y

X = (data_new.loc[:, data_new.columns != 'Churn'])
y = (data_new.loc[:, data_new.columns == 'Churn'])
print('Shape of X: {}'.format(X.shape))
print('Shape of y: {}'.format(y.shape))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
columns = X_train.columns


# > ** Base Model-Logistic Regression**

# In[75]:


# Logistic regression Model

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

#  Model metrics
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# > **Feature selection**

# In[76]:


# Removing unimportant features ['gender','PhoneService','TotalCharges','tenure']
X_train.drop(['gender','PhoneService','TotalCharges','tenure'], axis=1, inplace=True)
X_test.drop(['gender','PhoneService','TotalCharges','tenure'], axis=1, inplace=True)


# In[77]:


# New model
logreg_new = LogisticRegression()
logreg_new.fit(X_train, y_train)

#  Model metrics
y_pred = logreg_new.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg_new.score(X_test, y_test)))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[78]:


# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth = 3)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Report : ",  classification_report(y_test, y_pred)) 


# In[ ]:




