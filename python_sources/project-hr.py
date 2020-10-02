# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# coding: utf-8

# # Project HR
# 
# Predict attrition of your valuable employees
# 
# [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset)

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')


# In[2]:


df = pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv")
df.head()


# In[3]:


df.shape


# In[4]:


df.pop('EmployeeNumber')
df.pop('Over18')
df.pop('StandardHours')
df.pop('EmployeeCount')


# In[5]:


df.columns


# In[6]:


df.shape


# In[7]:


df.head()


# In[8]:


y = df['Attrition']
tmp = df['Attrition']
X = df
X.pop('Attrition')


# In[9]:


y.unique()


# In[10]:


df.head()


# In[11]:


from sklearn import preprocessing
le = preprocessing.LabelBinarizer()


# In[12]:


y = le.fit_transform(y)


# In[13]:


y.shape


# In[14]:


tmp = le.fit_transform(tmp)


# In[15]:


type(tmp)


# In[16]:


tmp = pd.Series(list(tmp))


# In[17]:


tmp.value_counts()


# In[18]:


tmp.value_counts() / tmp.count()


# In[19]:


df.info()


# In[20]:


df.select_dtypes(['object'])


# In[21]:


ind_BusinessTravel = pd.get_dummies(df['BusinessTravel'], prefix='BusinessTravel')
ind_Department = pd.get_dummies(df['Department'], prefix='Department')
ind_EducationField = pd.get_dummies(df['EducationField'], prefix='EducationField')
ind_Gender = pd.get_dummies(df['Gender'], prefix='Gender')
ind_JobRole = pd.get_dummies(df['JobRole'], prefix='JobRole')
ind_MaritalStatus = pd.get_dummies(df['MaritalStatus'], prefix='MaritalStatus')
ind_OverTime = pd.get_dummies(df['OverTime'], prefix='OverTime')


# In[22]:


ind_BusinessTravel.head()


# In[23]:


df['BusinessTravel'].unique()


# In[24]:


df1 = pd.concat([ind_BusinessTravel, ind_Department, ind_EducationField, ind_Gender, 
                 ind_JobRole, ind_MaritalStatus, ind_OverTime])


# In[25]:


df.select_dtypes(['int64'])


# In[26]:


df1 = pd.concat([ind_BusinessTravel, ind_Department, ind_EducationField, ind_Gender, 
                 ind_JobRole, ind_MaritalStatus, ind_OverTime, df.select_dtypes(['int64'])], axis=1)


# In[27]:


df1.shape


# # Decision Tree

# In[28]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df1, y)


# In[29]:


from sklearn.tree import DecisionTreeClassifier


# In[30]:


clf = DecisionTreeClassifier(random_state=42)


# In[31]:


clf.fit(X_train, y_train)


# In[32]:


from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[33]:


def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    '''
    print the accuracy score, classification report and confusion matrix of classifier
    '''
    if train:
        '''
        training performance
        '''
        print("Train Result:\n")
        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_train, clf.predict(X_train))))
        print("Classification Report: \n {}\n".format(classification_report(y_train, clf.predict(X_train))))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_train, clf.predict(X_train))))

        res = cross_val_score(clf, X_train, y_train.ravel(), cv=10, scoring='accuracy')
        print("Average Accuracy: \t {0:.4f}".format(np.mean(res)))
        print("Accuracy SD: \t\t {0:.4f}".format(np.std(res)))
        
    elif train==False:
        '''
        test performance
        '''
        print("Test Result:\n")        
        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_test, clf.predict(X_test))))
        print("Classification Report: \n {}\n".format(classification_report(y_test, clf.predict(X_test))))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_test, clf.predict(X_test))))    
        


# In[34]:


print_score(clf, X_train, y_train, X_test, y_test, train=True)


# In[35]:


print_score(clf, X_train, y_train, X_test, y_test, train=False)


# The result is clearly not satisfactory. We will revisit this project after we covered ensemble model.

# ****

# # Bagging

# In[36]:


from sklearn.ensemble import BaggingClassifier


# In[37]:


bag_clf = BaggingClassifier(base_estimator=clf, n_estimators=5000,
                            bootstrap=True, n_jobs=-1, random_state=42)


# In[38]:


bag_clf.fit(X_train, y_train.ravel())


# In[39]:


print_score(bag_clf, X_train, y_train, X_test, y_test, train=True)
print_score(bag_clf, X_train, y_train, X_test, y_test, train=False)


# ***

# # Random Forest

# In[40]:


from sklearn.ensemble import RandomForestClassifier


# In[41]:


rf_clf = RandomForestClassifier()


# In[42]:


rf_clf.fit(X_train, y_train.ravel())


# In[43]:


print_score(rf_clf, X_train, y_train, X_test, y_test, train=True)
print_score(rf_clf, X_train, y_train, X_test, y_test, train=False)


# In[68]:


plt.figure(figsize=(12,6))
pd.Series(rf_clf.feature_importances_,
          index=X_test.columns).sort_values(ascending=False).plot(kind='bar')


# In[51]:


import seaborn as sns


# In[45]:


pd.Series(rf_clf.feature_importances_, 
         index=X_train.columns).sort_values(ascending=False).plot(kind='bar', figsize=(12,6));


# # AdaBoost

# In[69]:


from sklearn.ensemble import AdaBoostClassifier


# In[70]:


ada_clf = AdaBoostClassifier()


# In[71]:


ada_clf.fit(X_train, y_train.ravel())


# In[72]:


print_score(ada_clf, X_train, y_train, X_test, y_test, train=True)
print_score(ada_clf, X_train, y_train, X_test, y_test, train=False)


# ***

# # AdaBoost + RandomForest

# In[73]:


ada_clf = AdaBoostClassifier(RandomForestClassifier())
ada_clf.fit(X_train, y_train.ravel())


# In[74]:


print_score(ada_clf, X_train, y_train, X_test, y_test, train=True)
print_score(ada_clf, X_train, y_train, X_test, y_test, train=False)


# ***

# # Gradient Boosting Classifier

# In[75]:


from sklearn.ensemble import GradientBoostingClassifier


# In[76]:


gbc_clf = GradientBoostingClassifier()
gbc_clf.fit(X_train, y_train.ravel())


# In[77]:


print_score(gbc_clf, X_train, y_train, X_test, y_test, train=True)
print_score(gbc_clf, X_train, y_train, X_test, y_test, train=False)


# ***

# # XGBoost

# In[78]:


import xgboost as xgb


# In[79]:


xgb_clf = xgb.XGBClassifier()
xgb_clf.fit(X_train, y_train.ravel())


# In[80]:


print_score(xgb_clf, X_train, y_train, X_test, y_test, train=True)
print_score(xgb_clf, X_train, y_train, X_test, y_test, train=False)

