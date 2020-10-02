#!/usr/bin/env python
# coding: utf-8

# The classification goal is to predict whether the patient has 10-year risk of future coronary heart disease (CHD).

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import statsmodels.api as sm
from statsmodels.tools import add_constant as add_constant
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("/kaggle/input/framingham-heart-study-dataset/framingham.csv")
df.info()


# #### Columns/Variables Explained:
# * **sex**: male or female (male=1)
# * **age**: age of the patient;(Continuous - Although the recorded ages have been truncated to whole numbers, the concept of age is continuous)
# * **education**: 1 = Some High School; 2 = High School or GED; 3 = Some College or Vocational School; 4 = college
# * **currentSmoker**: whether or not the patient is a current smoker (current smoker=1)
# * **cigsPerDay**: the number of cigarettes that the person smoked on average in one day.
# * **BPMeds**: whether or not the patient was on blood pressure medication (on BP medication = 1)
# * **prevalentStroke**: whether or not the patient had previously had a stroke (previous stroke = 1)
# * **prevalentHyp**: whether or not the patient was hypertensive (hypertensive = 1)
# * **diabetes**: whether or not the patient had diabetes (diabetes = 1)
# * **totChol**: total cholesterol level.
# * **sysBP**: systolic blood pressure.
# * **diaBP**: diastolic blood pressure.
# * **BMI**: Body Mass Index.
# * **heartRate**: heart rate.
# * **glucose**: glucose level
# * **TenYearCHD**: 10 year risk of coronary heart disease CHD (yes=1)

# Find out percentage of nulls

# In[ ]:


100* df.isnull().sum()/df.count()


# Since the percentage of nulls is very less, let's drop those rows.

# In[ ]:


df.dropna(inplace=True)


# In[ ]:


df.head()


# In[ ]:


print("Percentage of People with heart disease: {0:.2f} %".format(100*df.TenYearCHD.value_counts()[1]/df.TenYearCHD.count()))


# There seems to be an Imbalance in data with respect to TenYearCHD. Only 15.23% rows have Positive value. We will first go ahead with the imbalance and see if it affects our analysis.

# Let's create dummy variables for education.

# In[ ]:


df = pd.concat([df, pd.get_dummies(df.education, prefix="ed_",drop_first=True)],axis=1)
df.drop(['education'], axis=1, inplace=True)
df.columns


# We will use statsmodels' Logit funtion to perform logistic regression so that we get a clear picture of p-values , so that we find out statistically significant variables.

# In[ ]:


X = df.drop(['TenYearCHD'], axis=1)
Y = df.TenYearCHD

X_const=sm.add_constant(X)
model=sm.Logit(Y, X_const)
result=model.fit()
result.summary()


# There are lot of columns which are statistically insignificant. (p-value>0.05). We remove these columns in the next step

# In[ ]:


def back_feature_elim(data_frame,dep_var,col_list):
    """ Takes in the dataframe, the dependent variable and a list of column names, runs the regression repeatedly eliminating
    feature with the highest p-value above alpha(0.05) one at a time and returns the regression summary with all 
    p-values below alpha"""

    while len(col_list)>0 :
        model=sm.Logit(dep_var,data_frame[col_list])
        result=model.fit(disp=0)
        largest_pvalue=round(result.pvalues,3).nlargest(1)
        if largest_pvalue[0]<(0.05):
            return result
            break
        else:
            col_list=col_list.drop(largest_pvalue.index)


# Let's remove statistically insignificant columns and run logistic regression again

# In[ ]:


result=back_feature_elim(df,df.TenYearCHD, X.columns)
result.summary()


# In[ ]:


params = np.exp(result.params)
conf = np.exp(result.conf_int())
conf['OR'] = params
pvalue=round(result.pvalues,3)
conf['pvalue']=pvalue
conf.columns = ['CI 95%(2.5%)', 'CI 95%(97.5%)', 'Odds Ratio','pvalue']
print ((conf))


# Inferences from the above table<br>
# * Odds of getting diagnosed with heart disease for males is higher by 49.69% than females
# * Odds of getting diagnosed with heart disease increases by 2.83% every year.
# * Odds of getting diagnosed with heart disease increases by 1.42% with every additional cigarette
# and so on...

# Now that we have the dataset all cleaned up, Let's split the dataset with statistically significant columns in the ratio train:test=70:30

# In[ ]:


new_df=df[['male','age','cigsPerDay','prevalentHyp','diabetes','sysBP','diaBP','BMI','heartRate','ed__2.0','ed__3.0','ed__4.0',
           'TenYearCHD']]
X = new_df.drop(['TenYearCHD'], axis=1)
Y = new_df.TenYearCHD
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=5)


# In[ ]:


logreg=LogisticRegression()
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)


# In[ ]:


sklearn.metrics.accuracy_score(y_test,y_pred)


# In[ ]:


cm = confusion_matrix(y_test,y_pred)
print(cm)


# 
# 
# The confusion matrix shows 937+8 = 945 correct predictions and 147+6= 153 incorrect ones.
# 
# True Positives: 8
# 
# True Negatives: 937
# 
# False Positives: 6 (Type I error)
# 
# False Negatives: 147 ( Type II error)

# **Since we are dealing with Predicting heart disease, Higher value of false negatives is dangerous. <br>
# False Negative = There is Heart Disease, but we predict it wrongly as No Heart Disease. **

# In[ ]:


TN=cm[0,0]
TP=cm[1,1]
FN=cm[1,0]
FP=cm[0,1]
sensitivity=TP/float(TP+FN)
specificity=TN/float(TN+FP)


# In[ ]:


print('The accuracy of the model [TP+TN/(TP+TN+FP+FN)] = ',(TP+TN)/float(TP+TN+FP+FN),'\n',

'Missclassification [1-Accuracy] = ',1-((TP+TN)/float(TP+TN+FP+FN)),'\n',

'Sensitivity or True Positive Rate [TP/(TP+FN)] = ',TP/float(TP+FN),'\n',

'Specificity or True Negative Rate [TN/(TN+FP)] = ',TN/float(TN+FP),'\n')


# As we can see, True Negatives rate is very high (since, the dataset had around 85% No Heart Disease). Because of this, our model was not able to learn much of Positives

# In[ ]:


smt = SMOTE()
x_train, y_train = smt.fit_sample(x_train, y_train)


# In[ ]:


np.bincount(y_train)


# In[ ]:


logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)


# In[ ]:


print(sklearn.metrics.accuracy_score(y_test,y_pred))
cm = confusion_matrix(y_test,y_pred)
print(cm)


# 
# The confusion matrix shows 621+104 = 805 correct predictions and 322+51= 373 incorrect ones.
# 
# True Positives: 104
# 
# True Negatives: 621
# 
# False Positives: 322 (Type I error)
# 
# False Negatives: 51 ( Type II error)

# In[ ]:


TN=cm[0,0]
TP=cm[1,1]
FN=cm[1,0]
FP=cm[0,1]
sensitivity=TP/float(TP+FN)
specificity=TN/float(TN+FP)


# In[ ]:


print('The accuracy of the model [TP+TN/(TP+TN+FP+FN)] = ',(TP+TN)/float(TP+TN+FP+FN),'\n',

'Missclassification [1-Accuracy] = ',1-((TP+TN)/float(TP+TN+FP+FN)),'\n',

'Sensitivity or True Positive Rate [TP/(TP+FN)] = ',TP/float(TP+FN),'\n',

'Specificity or True Negative Rate [TN/(TN+FP)] = ',TN/float(TN+FP),'\n')


# As we can see, True positive rate has increased since we have balanced the data now(increasing Positive samples). Also, we were worried about False Negatives. This has decreased from 147 to 51 which is a good sign. Now our model will have lesser Type II errors.
