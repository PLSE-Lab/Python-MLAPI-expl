#!/usr/bin/env python
# coding: utf-8

# ## Binary Classification using Graduate Admission Dataset
# 
# This notebook compares performance of various Machine Learning classifiers on the "Graduate Admission" data. I'm still just a naive student implementing Machine Learning techniques. You're most welcome to suggest me edits on this kernel, I am happy to learn.

# ## Setting up Google Colab
# 
# You can skip next 2 sections if you're not using Google Colab.

# In[ ]:


## Uploading my kaggle.json (required for accessing Kaggle APIs)
from google.colab import files
files.upload()


# In[ ]:


## Install Kaggle API
get_ipython().system('pip install -q kaggle')
## Moving the json to appropriate place
get_ipython().system('mkdir -p ~/.kaggle')
get_ipython().system('cp kaggle.json ~/.kaggle/')
get_ipython().system('chmod 600 /root/.kaggle/kaggle.json')


# ### Getting the data
# I'll use Kaggle API for getting the data directly into this instead of pulling the data from Google Drive.

# In[ ]:


get_ipython().system('kaggle datasets download mohansacharya/graduate-admissions')
get_ipython().system('echo "========================================================="')
get_ipython().system('ls')
get_ipython().system('unzip graduate-admissions.zip')
get_ipython().system('echo "========================================================="')
get_ipython().system('ls')


# ### Imports

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
pd.set_option('display.max_columns', 60)
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Data exploration

# In[ ]:


FILE_NAME = "../input/Admission_Predict_Ver1.1.csv"

raw_data = pd.read_csv(FILE_NAME)
raw_data.head()


# In[ ]:


## Are any null values persent ?
raw_data.isnull().values.any()


# In[ ]:


## So no NaNs apparently
## Let's just quickly rename the dataframe columns to make easy references
## Notice the blankspace after the end of 'Chance of Admit' column name
raw_data.rename(columns = {
    'Serial No.' : 'srn',
    'GRE Score'  : 'gre',
    'TOEFL Score': 'toefl',
    'University Rating' : 'unirating',
    'SOP'        : 'sop',
    'LOR '        : 'lor',
    'CGPA'       : 'cgpa',
    'Research'   : 'research',
    'Chance of Admit ': 'chance'
}, inplace=True)
raw_data.describe()


# ### Analyzing the factors influencing the admission :
# From what I've heard from my relatives, seniors, friends is that you need an excellent CGPA from a good university , let's verify that first.

# In[ ]:


fig, ax = plt.subplots(ncols = 2)
sns.regplot(x='chance', y='cgpa', data=raw_data, ax=ax[0])
sns.regplot(x='chance', y='unirating', data=raw_data, ax=ax[1])


# ### Effect of GRE/TOEFL :
# Let's see if GRE/TOEFL score matters at all. From what I've heard from my seniors, relatives, friends; these exams don't matter if your score is above some threshold.

# In[ ]:


fig, ax = plt.subplots(ncols = 2)
sns.regplot(x='chance', y='gre', data=raw_data, ax=ax[0])
sns.regplot(x='chance', y='toefl', data=raw_data, ax=ax[1])


# ### Effect of SOP / LOR / Research :
# I decided to analyze these separately since these are not academic related.and count mostly as an extra curricular skill / writing.

# In[ ]:


fig, ax = plt.subplots(ncols = 3)
sns.regplot(x='chance', y='sop', data=raw_data, ax=ax[0])
sns.regplot(x='chance', y='lor', data=raw_data, ax=ax[1])
sns.regplot(x='chance', y='research', data=raw_data, ax=ax[2])


# ### Conclusions :
# CGPA, GRE and TOEFL are extremely important and they vary almost linearly. (TOEFL varies almost scaringly linearly to chance of admit). On other factors, you need to have _just enough_ score to get admission.
# 
# We will convert the 'Chance of Admission' column into 0 or 1 and then use binary classification algorithms to see if you can get an admission or not.

# In[ ]:


THRESH = 0.6
# I think we can also drop srn as it is not doing absolutely anything
raw_data.drop('srn', axis=1, inplace=True)
raw_data['chance'] = np.where(raw_data['chance'] > THRESH, 1, 0)
raw_data.head()


# In[ ]:


raw_data.describe()


# ### Train-Test split :
# Since we have less data, I am goign to use traditional 70-30 train test split. May update this in future if author adds more data.

# In[ ]:


X = raw_data.drop(columns='chance')
Y = raw_data['chance'].values.reshape(raw_data.shape[0], 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)


print("Training set ...")
print("X_train.shape = {}, Y_train.shape =  is {}".format(X_train.shape, 
                                                          Y_train.shape))

print("Test set ...")
print("X_test.shape = {}, Y_test.shape =  is {}".format(X_test.shape, 
                                                          Y_test.shape))


# In[ ]:


from sklearn.metrics import mean_absolute_error as mae


# ### Trying out Logistic Regression :
# Let's apply good old logistic regression to see if it acn classify the dataset properly

# In[ ]:


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(penalty='l1', solver='liblinear', max_iter=3000000, 
                         tol=1e-8)
clf.fit(X_train, Y_train)


# In[ ]:


Y_pred = clf.predict(X_test)
print("Mean Absolute Error = ", mae(Y_test, Y_pred))


# ### Trying out LinearSVC :
# Linearity of data with respect to 3 features suggest us to use Linear SVC.

# In[ ]:


from sklearn.svm import LinearSVC
clf = LinearSVC(verbose=1, max_iter=3000000, tol=1e-8, C=1.25)
clf.fit(X_train, Y_train)


# In[ ]:


Y_pred = clf.predict(X_test)
print("Mean Absolute Error = ", mae(Y_test, Y_pred))


# ### Trying out Bernoulli Naive Bayes
# As described [here](https://towardsdatascience.com/naive-bayes-classifier-81d512f50a7c), Bernoulli Naive Bayes can be used for binary classification assuming all factors equally influence the output.

# In[ ]:


from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB()
clf.fit(X_train, Y_train)


# In[ ]:


Y_pred = clf.predict(X_test)
print("Mean Absolute Error = ", mae(Y_test, Y_pred))


# ### Trying out Decision Tree Classifier

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=10)
clf.fit(X_train, Y_train)


# In[ ]:


Y_pred = clf.predict(X_test)
print("Mean Absolute Error = ", mae(Y_test, Y_pred))


# ### Conclusion :
# 
# Most of the classifiers don't work very well on the data with the currently chosen hyper-parameters. This maybe due to smaller size of dataset. 
# 
# I'm still new to machine learning, if you think I've done something wrong and you want to correct me, you're most welcome.
# 
# 

# In[ ]:




