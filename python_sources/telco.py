#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from pylab import rcParams


# **1. Introduction**
# This Data sets has information about TELCO clients, if they lefts company in last months (CHURN). And I do a model who can to guess which clients are thinking about new TELCO company and they are planing get out from this Company. 
# 
# **Basic Information**

# In[ ]:


df = pd.read_csv('../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')


# In[ ]:


df.head()


# In[ ]:


uniq = pd.DataFrame({'Diferent_value_count': df.nunique(), 'DTypes': df.dtypes})
print ("Rows     : " ,df.shape[0])
print ("Columns  : " ,df.shape[1])
print ("\nFeatures : \n" ,df.columns.tolist())
print ("\nMissing values :  ", df.isnull().sum().values.sum())
print ("\nUnique values :  \n", uniq)


# **1.2 Missing values**
# 
# There are 11 missing values in TotalCharges column. So I decided in those missing values inpute from MonthlyCharges	column. It is possible that they are missing because it is the first months for this client in this company.

# In[ ]:


df[df['TotalCharges'] ==' ']


# In[ ]:


df.loc[df['TotalCharges'] ==' ','TotalCharges'] = df['MonthlyCharges'] 


# In[ ]:


df[df['TotalCharges'] ==' ']


# In[ ]:


df["TotalCharges"] = df["TotalCharges"].astype(float)


# **2. Data correction**
# 
# I created a new column. In this column I input information from Contract term. I splited the client into two groups:
# first group - 1, is loyal clients (who is company clients for more than 1 year)
# second group - new clients (who is company clients less than 1 year)
# I changed columns where is information about "yes" or "no" to "1" and "0". Columns su information about - "No service", I changed to "0".

# In[ ]:


df['Contract_term'] = df["Contract"].replace({'Month-to-month':0, 'One year':1, 'Two year':1})
df.drop(['customerID', 'Contract'], axis = 1, inplace = True)
df['MultipleLines'] = df['MultipleLines'].replace({'No phone service':0, 'No' : 0, 'Yes' : 1})


# In[ ]:


replace_col_val = [ 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
for i in replace_col_val : 
    df[i]  = df[i].replace({'No internet service' : 'No'})


# In[ ]:


replace_col_val2 = ['Partner', 'Dependents', 'PhoneService', 'OnlineSecurity',
                   'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                   'StreamingMovies', 'PaperlessBilling', 'Churn']
for i in replace_col_val2 : 
    df[i]  = df[i].replace({'Yes' : 1, 'No': 0})


# In[ ]:


df.head()


# **3. Visual data analysis**
# 
# Firstly I want to see how much clients is "CHURN"

# In[ ]:


colors = ['gold', 'yellowgreen']
explode = (0.1, 0)
labels = 'Yes', 'No'
No, Yes= df['Churn'].value_counts(sort = True)
sizes = [Yes, No]
rcParams['figure.figsize'] = 5,5
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140,)
plt.title('Percent of churn in customer')
plt.show()
print('Churn Yes -', Yes)
print('Churn No -', No)


# Then I see how clients distributed by gender

# In[ ]:


fig, axis = plt.subplots(1,2, figsize = (10,10))
colors = ['gold', 'yellowgreen']
explode = (0.1, 0)
labels = 'Yes', 'No'
axis[0].set_title('Percent of churn in Male')
axis[1].set_title('Percent of churn in Female')

No, Yes= df[df['gender']== 'Male'].Churn.value_counts()
sizes = [Yes, No]
axis[0].pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140,)

No, Yes= df[df['gender']!= 'Male'].Churn.value_counts()
sizes = [Yes, No]
axis[1].pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140,)

plt.show()


# And then how clients distributed by another parameters.
# And here I see, That in unloyal clients group is more clients who wants to go out.

# In[ ]:


colum = ['SeniorCitizen', 'Partner', 'Dependents', 'MultipleLines',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
       'Contract_term']


# In[ ]:


i = 0
j= 0
fig, axis = plt.subplots(6, 4,figsize = (20,30))
colors = ['gold', 'yellowgreen']
explode = (0.1, 0)
labels = 'Yes', 'No'
for r in colum:
    if i == 6:
        pass
    else:
        No, Yes= df[df[r]== 1].Churn.value_counts()
        sizes = [Yes, No]
        axis[i,j].pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
        axis[i,j].set_title(r + ' is Yes')
        No, Yes= df[df[r]== 0].Churn.value_counts()
        sizes = [Yes, No]
        j+=1
        axis[i,j].pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
        axis[i,j].set_title(r + ' is No')
        if j < 3:
            j+=1
        else:
            i += 1
            j = 0


# Loyal clients decided to get out to another company just 214 of 2954. But unloyal clients proportion is bigger. So I will do analysis just in unloyal clients group.

# In[ ]:


df.groupby('Churn').Contract_term.value_counts()


# In[ ]:


df_Contract_term = df[df['Contract_term'] == 0]


# In[ ]:


df_Contract_term.drop(['Contract_term'], axis = 1, inplace = True)


# In[ ]:


df_Contract_term.head()


# Let's see how clients are distributed by all parameters at now. 

# In[ ]:


colum2 = ['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'PaperlessBilling']


# In[ ]:


i = 0
j= 0
fig, axis = plt.subplots(6, 4,figsize = (20,30))
colors = ['gold', 'yellowgreen']
explode = (0.1, 0)
labels = 'Yes', 'No'
for r in colum2:
    if i == 6:
        pass
    else:
        No, Yes= df_Contract_term[df_Contract_term[r]== 1].Churn.value_counts()
        sizes = [Yes, No]
        axis[i,j].pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
        axis[i,j].set_title(r + ' is Yes')
        No, Yes= df_Contract_term[df_Contract_term[r]== 0].Churn.value_counts()
        sizes = [Yes, No]
        j+=1
        axis[i,j].pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
        axis[i,j].set_title(r + ' is No')
        if j < 3:
            j+=1
        else:
            i += 1
            j = 0


# I dropped "gender" column.

# In[ ]:


df_Contract_term.drop(['gender'], axis = 1, inplace = True)


# In[ ]:


df3 = df_Contract_term


# In[ ]:


df3.describe()


# I decided to see how related those parametre by "CHURN":
# 'tenure','MonthlyCharges','TotalCharges'

# In[ ]:


df3[['tenure','MonthlyCharges','TotalCharges','Churn']].corr()


# In[ ]:


sns.pairplot(df_Contract_term[['tenure','Churn']], hue='Churn', height=5)


# In[ ]:


sns.pairplot(df_Contract_term[['MonthlyCharges','Churn']], hue='Churn', height=5)


# In[ ]:


sns.pairplot(df_Contract_term[['TotalCharges','Churn']], hue='Churn', height=5)


# In[ ]:


plt.figure(figsize = (15,7))
sns.countplot(df_Contract_term['tenure'])


# In[ ]:


sns.pairplot(df_Contract_term[['MonthlyCharges','Churn']], hue='Churn', height=5)


# In[ ]:


sns.pairplot(df_Contract_term[['tenure','MonthlyCharges','TotalCharges','Churn']], hue='Churn')


# I decided to creat a new column "Tenure_group". In this column I splited terms by this groups: less 4 month, 5-36 months and more 36 months.

# In[ ]:


bins = [0, 4, 36, 500]
labels = ['<=4','5 - 36','>36']
df3['tenure_group'] = pd.cut(df3['tenure'], bins=bins, labels=labels)


# In[ ]:


matrix = np.triu(df3.corr())
plt.figure(figsize=(30,10))
cmap = sns.diverging_palette(500, 1)
sns.heatmap(df3.corr().round(2), mask=matrix, vmin=-0.7, vmax=0.7, annot=True, 
            cmap=cmap,
            square=True)


# I changed all parametres to 0 and 1 values.

# In[ ]:


df3_dummies = pd.get_dummies(df3, drop_first=True)
df3 = df3_dummies


# In[ ]:


matrix = np.triu(df3.corr())
plt.figure(figsize=(30,10))
cmap = sns.diverging_palette(500, 1)
sns.heatmap(df3.corr().round(2), mask=matrix, vmin=-0.7, vmax=0.7, annot=True, 
            cmap=cmap,
            square=True)


# I dropped TotalCharges

# In[ ]:


df3.drop(['TotalCharges'], axis = 1, inplace = True)


# **4. Logistic Regression model******

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import statsmodels.formula.api as smf


# In[ ]:


X = df3.drop('Churn', axis = 1)
y = df3['Churn']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
logistic = LogisticRegression()
logistic.fit(X_train, y_train)


# In[ ]:


predictions = logistic.predict(X_test)
predictions


# In[ ]:


conf_matrix = metrics.confusion_matrix(y_test, predictions)
conf_matrix


# In[ ]:


fig, ax = plt.subplots()
ax= plt.subplot()
sns.heatmap(conf_matrix,annot=True, ax = ax, cmap='YlGnBu', fmt='d')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual') 
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['0 - Not churn', '1 - Churn'])
ax.yaxis.set_ticklabels(['0 - Not churn', '1 - Churn'])


# In[ ]:


Accuracy = metrics.accuracy_score(y_test, predictions)
Precision = metrics.precision_score(y_test, predictions)
recall = metrics.recall_score(y_test, predictions)
F1 = 2 * (Precision * recall) / (Precision + recall)


# In[ ]:


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


# In[ ]:


probs = logistic.predict_proba(X_test)
probs = probs[:, 1]
auc = roc_auc_score(y_test, probs)
print('AUC: %.2f' % auc)


# In[ ]:


fpr, tpr, thresholds = roc_curve(y_test, probs)
plot_roc_curve(fpr, tpr)


# In[ ]:


print('Accuracy :', Accuracy.round(3))
print('Precision :', Precision.round(3))
print('recall :', recall.round(3))
print('F-1 :', F1.round(3))
print('Area under the curve :', auc.round(3))


# In[ ]:


df3.columns = ['SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'PaperlessBilling', 'MonthlyCharges',
       'Churn', 'InternetService_Fiber_optic', 'InternetService_No',
       'PaymentMethod_Credit_card_automatic',
       'PaymentMethod_Electronic_check', 'PaymentMethod_Mailed_check','tenure_group_5_36', 'tenure_group_36']


# In[ ]:


est2 = smf.logit('Churn ~ SeniorCitizen + Partner+Dependents +tenure+PhoneService+MultipleLines + OnlineSecurity + OnlineBackup +DeviceProtection+ TechSupport + StreamingTV + StreamingMovies + PaperlessBilling + MonthlyCharges+InternetService_Fiber_optic + InternetService_No + PaymentMethod_Credit_card_automatic+PaymentMethod_Electronic_check+PaymentMethod_Mailed_check+tenure_group_5_36+tenure_group_36',df3).fit()


# In this model is too much interrelated meaning. So I dropped it.

# In[ ]:


est2.summary()


# In[ ]:


df3.drop(['Partner', 'Dependents', 'PhoneService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
          'TechSupport', 'StreamingTV',
          'StreamingMovies', 'MonthlyCharges',
          'InternetService_No', 'PaymentMethod_Credit_card_automatic','PaymentMethod_Mailed_check', 'tenure_group_36'], axis = 1, inplace = True)


# In[ ]:


X = df3.drop('Churn', axis = 1)
y = df3['Churn']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
logistic = LogisticRegression()
logistic.fit(X_train, y_train)


# Let see model now. And now model is a litle bit better.

# In[ ]:


predictions = logistic.predict(X_test)
predictions


# In[ ]:


conf_matrix = metrics.confusion_matrix(y_test, predictions)
conf_matrix


# In[ ]:


fig, ax = plt.subplots()
ax= plt.subplot()
sns.heatmap(conf_matrix,annot=True, ax = ax, cmap='YlGnBu', fmt='d')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual') 
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['0 - Not churn', '1 - Churn'])
ax.yaxis.set_ticklabels(['0 - Not churn', '1 - Churn'])


# In[ ]:


Accuracy = metrics.accuracy_score(y_test, predictions)
Precision = metrics.precision_score(y_test, predictions)
recall = metrics.recall_score(y_test, predictions)
F1 = 2 * (Precision * recall) / (Precision + recall)


# In[ ]:


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


# In[ ]:


probs = logistic.predict_proba(X_test)
probs = probs[:, 1]
auc = roc_auc_score(y_test, probs)
print('AUC: %.2f' % auc)


# In[ ]:


fpr, tpr, thresholds = roc_curve(y_test, probs)
plot_roc_curve(fpr, tpr)


# In[ ]:


print('Accuracy :', Accuracy.round(3))
print('Precision :', Precision.round(3))
print('recall :', recall.round(3))
print('F-1 :', F1.round(3))
print('Area under the curve :', auc.round(3))


# In[ ]:


df3.columns


# In[ ]:


est2 = smf.logit('Churn ~ SeniorCitizen+tenure+MultipleLines+PaperlessBilling+InternetService_Fiber_optic+PaymentMethod_Electronic_check+tenure_group_5_36',df3).fit()


# In[ ]:


est2.summary()


# * **5.Conclusions**
# 
# After discarding some of the interrelated parameters, the model improved. 
# In order to get an even better model, more methods should be applied that would reduce the customer ratio of CHURN to 50 percent and then look this segment in detail. 
# I was able to reduce this ratio by visual analysis alone. Other regression models should be tried.

# In[ ]:




