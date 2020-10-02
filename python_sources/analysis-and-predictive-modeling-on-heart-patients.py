#!/usr/bin/env python
# coding: utf-8

# ## ****Objective****
# * EDA on dataset
# * Building a predictive model
# ![heart](https://2f4izj3opteu3l5obc1sh0bb-wpengine.netdna-ssl.com/wp-content/uploads/sites/14/2016/09/heartbeat-heart-attack.jpg)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


hp=pd.read_csv('/kaggle/input/heart-patients/US_Heart_Patients.csv')


# In[ ]:


hp.info()


# # MISSING VALUE TREATMENT

# In[ ]:


hp.isnull().sum()*100/len(hp)


# The above code shows how much value percentage of value missing in the dataset from following attributes.

# In[ ]:


hp.shape


# In[ ]:


hp.drop('education',axis=1,inplace=True)


# Education has no significance to our analysis so dropped it.

# In[ ]:


hp.dropna(subset=['cigsPerDay'],inplace=True)


# Cigsperday has only 0.62% of missing value so we can drop nan from it.

# In[ ]:


hp.dropna(subset=['BPMeds'],inplace=True)


# BP Meds is an categorical value attribute so nan value cannot be filled.

# In[ ]:


hp.totChol


# total cholesterol attribute can be filled with mean/average.

# In[ ]:


hp.totChol.fillna(hp.totChol.mean(),inplace=True)


# In[ ]:


hp.glucose


# Glucose can be filled with mean/average.

# In[ ]:


hp.glucose.fillna(hp.glucose.mean(),inplace=True)


# In[ ]:


hp.dropna(subset=['heartRate'],inplace=True)


# In[ ]:


hp.BMI.describe()


# In[ ]:


hp.BMI.fillna(hp.BMI.mean(),inplace=True)


# In[ ]:


hp.isnull().sum()*100/len(hp)


# # EDA

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as  plt


# In[ ]:


hp.TenYearCHD.value_counts().plot.barh(grid=True)
plt.show()


#  ***15% of total patients(4157) will suffer with heart disease within 10 year of span. ***

# In[ ]:


plt.figure(figsize=(20,8))
sns.heatmap(hp.corr(),annot=True)
plt.show()


# There is high correlation between diaBP and sysBP

# In[ ]:


plt.figure(figsize=(20,8))
sns.stripplot(data=hp,y='age',x='cigsPerDay',hue='TenYearCHD')
plt.show()


# Age and CigPerday is not much affecting whether you have an heart attack or not.

# In[ ]:


sns.scatterplot(data=hp,y='diaBP',x='sysBP',hue='TenYearCHD')
plt.show()


# DiaBP and sysBP are not much impacting on whether the patient will have heart attack within 10yrs or not.
# > As patients with low diaBP and sysBP are also having a high chance of having heart attack and also patients with high diaBP and sysBP are also having high chance of heart attack.

# In[ ]:


sns.scatterplot(data=hp,y='diaBP',x='sysBP',hue='prevalentHyp')
plt.show()


# Patients with high Blood Pressure and are having diabetes have prevalent Hypertension.
# Similarly patients with high blood pressure and donot have diabetes also have prevelant Hypertension.
# >>*So high blood pressure can lead to hypertension or vice versa.*

# In[ ]:


sns.barplot(data=hp,x='diabetes',y='glucose',hue='TenYearCHD')
plt.show()


# Diabetic patients have high glucose level as compared to  non diabetic patients.
# >>But glucose level above 150 will lead to an heart attack.

# In[ ]:


sns.barplot(data=hp,x='currentSmoker',y='cigsPerDay',hue='TenYearCHD')
plt.show()


# ***Current smoker and smoking 17 or more than 17 cigarettes per day than you would have an heart attack within 10 years.***

# In[ ]:


hp.shape


# In[ ]:


bmi=hp['BMI']
def cat_BMI(bmi):
    if float(bmi)<=18.5 :
        return 'Underweight'
    elif 18.5<float(bmi)<=25:
        return 'Normal'
    elif 25<float(bmi)<=30:
        return 'Overweight'
    else:
        return 'Obese'


# In[ ]:


hp['cat_BMI']=hp['BMI'].apply(cat_BMI)


# In[ ]:


hp['cat_BMI'].value_counts()*100/len(hp['cat_BMI'])


# * ***Approximately 44% of patients are having normal weight***
# * ***Approximately 42% of patients are overweight ***
# * ***Approximately 13% of patients are Obese***
# * ***Only 1% is underweight***

# In[ ]:


ct_bmi_tychd=pd.crosstab(hp['cat_BMI'],hp['TenYearCHD'])


# In[ ]:


ct_bmi_tychd.plot.bar()
plt.show()


# ****Overweight patients are having high chances of getting an heart diease****
# > Overweight falls in between 25 to 30 BMI(Body Mass Index)
# ![bmi formula](https://bmicalculatorusa.com/wp-content/uploads/2018/10/body-mass-index-formulas.jpg)

# # APPLYING ML MODELS
# >> As the target variable is categorical,so applying SLC

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.feature_selection import RFE


# In[ ]:


X=hp.drop(['TenYearCHD','cat_BMI'],axis=1)
y=hp.TenYearCHD


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# splitting into 70:30 train test 

# In[ ]:


dtr=DecisionTreeClassifier()
np.random.seed(42)

param_dist = {'criterion':['gini','entropy'],'min_samples_split':[2,3,4],'max_depth':[4,5,6],'min_samples_leaf':[1,2,3]}

cv_dtr = GridSearchCV(dtr, cv = 5,param_grid=param_dist, n_jobs = 3)

cv_dtr.fit(X_train, y_train)
print('Best Parameters using grid search: \n', cv_dtr.best_params_)


# In[ ]:


dtr=dtr.set_params(criterion= 'gini', max_depth= 4, min_samples_leaf= 1, min_samples_split= 3)


# In[ ]:


dtr_fit=dtr.fit(X_train,y_train)
dtr_predict=dtr.predict(X_test)
dtr_train_predict=dtr.predict(X_train)
print('accuracy score of train:',accuracy_score(y_train,dtr_train_predict))
print('accuracy score of test:',accuracy_score(y_test,dtr_predict))
print('classification report:\n',classification_report(y_test,dtr_predict))

