#!/usr/bin/env python
# coding: utf-8

# I have test 3 basic algorithms. The maximum accuracy received is 79% using SVM using linear kernel.

# In[1]:


#importing modules
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/diabetes.csv')
data.head()


# Additional details about the attributes
# 
#  - Pregnancies: Number of times pregnant
# 
#  - Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
# 
#  - BloodPressure: Diastolic blood pressure (mm Hg)
# 
#  - SkinThickness: Triceps skin fold thickness (mm)
# 
#  - Insulin: 2-Hour serum insulin (mu U/ml)
# 
#  - BMI: Body mass index (weight in kg/(height in m)^2)
# 
#  - DiabetesPedigreeFunction: Diabetes pedigree function
# 
#  - Age: Age (years)
# 
#  - Outcome: Class variable (0 or 1)

# In[2]:


data.tail()


# In[3]:


data.shape


# In[4]:


data.describe()


# In[5]:


data.groupby('Outcome').size()


# In[6]:


# Visualizing datas
data.hist(figsize=(16,14))


# In[8]:


data.plot(kind= 'box' , subplots=True, layout=(3,3), sharex=False, sharey=False, figsize=(16,14))


# In[9]:


column_x = data.columns[0:len(data.columns) - 1]
column_x


# In[10]:


corr = data[data.columns].corr()
corr


# In[11]:


#Extracting the important features for better accuracy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
X = data.iloc[:,0:8]
Y = data.iloc[:,8]
select_top_4 = SelectKBest(score_func=chi2, k = 5)


# In[12]:


fit = select_top_4.fit(X,Y)
features = fit.transform(X)


# In[13]:


features[0:5]


# In[14]:


data.head()


# In[15]:


features = ['Pregnancies', 'Glucose', 'Insulin', 'BMI', 'Age']
req_features = data[features]
req_features.head()


# In[16]:


req_outcome = data['Outcome']
req_outcome.head()


# In[17]:


#Splitting data for train and test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(req_features, req_outcome, test_size=.25, random_state = 12)


# In[19]:


#Decision Tree
from sklearn import tree
t_clf = tree.DecisionTreeClassifier()
t_clf = t_clf.fit(X_train, Y_train)
t_acc = t_clf.score(X_test, Y_test)
print (t_acc)


# In[20]:


# SVM with linear kernel
from sklearn import svm
s_clf = svm.SVC(kernel='linear')
s_clf = s_clf.fit(X_train, Y_train)
s_acc = s_clf.score(X_test, Y_test)
print (s_acc)


# In[22]:


#SVM with rbf kernel
from sklearn import svm
s_clf = svm.SVC(kernel='rbf')
s_clf = s_clf.fit(X_train, Y_train)
s_acc = s_clf.score(X_test, Y_test)
print (s_acc)


# In[21]:


#Naive Bayes
from sklearn.naive_bayes import GaussianNB
n_clf = GaussianNB()
n_clf = n_clf.fit(X_train, Y_train)
n_acc = n_clf.score(X_test, Y_test)
print (n_acc)


# In[ ]:




