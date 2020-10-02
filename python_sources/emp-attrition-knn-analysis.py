#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn import metrics

import matplotlib.pyplot as plt
from scipy.stats import norm, skew
plt.rc("font", size=14)

import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


warnings.filterwarnings("ignore")
empData=pd.read_csv(r"../input/HR-Employee-Attrition.csv",header=0)
#empData.head()
empData.shape


# In[ ]:


print(list(empData.columns))


# In[ ]:


empData.head()


# In[ ]:


#empData.drop(['EmployeeNumber'],axis=1,inplace=True)
#Dropped Emp Number, which is not required for analysis


# In[ ]:


empData.describe().T


# 
# 

# 

# In[ ]:


empData.columns.groupby(empData.dtypes)


# In[ ]:


empData.info()


# In[ ]:



isnaa=empData.isna().sum()
isnaa



# In[ ]:


empData.duplicated().sum()


# In[ ]:


"""
(mu, sigma) = norm.fit(empData.loc[empData['Attrition'] == 'Yes', 'Age'])
print("Ave age of emp left the organization : ",mu)
(mu, sigma) = norm.fit(empData.loc[empData['Attrition'] == 'No', 'Age'])
print("Ave age of emp in the organization : ",mu)
"""


# In[ ]:


empData['EducationField'].unique()


# In[ ]:


ed_field=empData['EducationField'].unique()
ed_field
for i in ed_field:
    ratio = empData.loc[empData['EducationField'] == i,'Attrition'].shape[0]/empData.loc[empData['Attrition'] == 'Yes'].shape[0]
    print("Attrition Rate for EduField {0}:\t {1}".format(i,ratio))


# In[ ]:


#empData.Gender.value_counts()
ed_field=empData['Gender'].unique()
ed_field
for i in ed_field:
    ratio = (empData.loc[empData['Gender'] == i,'Attrition'].shape[0])/(empData.loc[empData['Attrition'] == 'Yes'].shape[0])
    print("Attrition Rate for {0}: {1}".format(i,ratio))
    
# Changing the Attrition Rate 
empDataAna = empData.copy()
empDataAna['Target'] = empDataAna['Attrition'].apply(
    lambda x: 0 if x == 'No' else 1)    




# In[ ]:


empData.head()


# In[ ]:


empData['JobSatisfaction'].value_counts()




#for i in empData['EnvironmentSatisfaction'].unique():
    #ratio = df_HR[(df_HR['EnvironmentSatisfaction']==field)&(df_HR['Attrition']=="Yes")].shape[0] / df_HR[df_HR['EnvironmentSatisfaction']==field].shape[0]
    #ratio = empData.loc[empData['EnvironmentSatisfaction'] == i ,empData['Attrition'] == 'Yes'].shape(0) / empData.loc[empData['EnvironmentSatisfaction'] == i].shape(0)
    
 #   empData[empData['EnvironmentSatisfaction'] == i ,'Attrition'].shape(0)
    
    
    
    


# In[ ]:


empDataAna.columns


# In[ ]:





# In[ ]:


# Dropping columns which are not significant
empDataAna = empDataAna.drop(['Attrition', 'EmployeeCount', 'EmployeeNumber', 'StandardHours','Over18'], axis=1)
empDataAna.head()



# In[ ]:


empDataAna.corr()['Target'].sort_values()


# 

# **Basic Infering with plots**

# In[ ]:


num_cols = empDataAna.select_dtypes(include = np.number)
a = num_cols[num_cols.columns].hist(bins=15, figsize=(15,35), layout=(9,3),color = 'blue',alpha=0.6)


# In[ ]:


cat_cols = empDataAna.select_dtypes(exclude=np.number)


# In[ ]:


"""
fig, ax = plt.subplots(4, 2, figsize=(20, 20))
for variable, subplot in zip(cat_col, ax.flatten()):
    sns.countplot(empDataAna[variable], ax=subplot,palette = 'Set3')
    for label in subplot.get_xticklabels():
        label.set_rotation(360)
plt.tight_layout()
"""


# Correlation Between Target and other variables

# In[ ]:


"""
corr = data.drop(columns=['StandardHours','EmployeeCount']).corr()
corr.style.background_gradient(cmap='YlGnBu')
"""


# **ENCODING**

# In[ ]:


print("Cat Columns --- {0} and Count ---- {1} ".format(cat_cols.columns,cat_cols.columns.shape[0]))


# In[ ]:


num_cols = empDataAna.select_dtypes(include = np.number)
print("Num Columns --- {0} and Count ---- {1} ".format(num_cols.columns,num_cols.columns.shape[0]))


# In[ ]:


# ENCODING CAT COLUMNS...
cat_col_encoded = pd.get_dummies(cat_cols)
cat_col_encoded.head()


# In[ ]:


empDatafin = pd.concat([num_cols,cat_col_encoded,],axis=1)
empDatafin.head()


# In[ ]:


x = empDatafin.drop(columns='Target')
y = empDatafin['Target']


# **ANALYSIS**

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

mLogReg = LogisticRegression()
mLogReg.fit(x_train, y_train)


# 

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

x_train, x_test, y_train, y_test = train_test_split(
x, y, test_size = 0.3, random_state = 100)
y_train=np.ravel(y_train)
y_test=np.ravel(y_test)


# In[ ]:


for i in range(25):
    K = i+1
    neigh = KNeighborsClassifier(n_neighbors = K, weights='uniform', algorithm='auto')
    neigh.fit(x_train, y_train) 
    y_pred = neigh.predict(x_test)
    print ("Accuracy : {0}% for K-Value {1}".format(accuracy_score(y_test,y_pred)*100,K))


# In[ ]:


accuracy_dict = {}
accuracy_list = []
for k in range(1,39):
    model = KNeighborsClassifier(n_neighbors = k,weights='uniform', algorithm='auto').fit(x_train,y_train)
    Y_predict = model.predict(x_test)
    accuracy = metrics.accuracy_score(y_test,Y_predict)
    accuracy_dict.update({k:accuracy})
    accuracy_list.append(accuracy)
    print("Accuracy ---> k = {} is {}" .format(k,accuracy))


# In[ ]:



key_max = max(accuracy_dict.keys(), key=(lambda k: accuracy_dict[k]))

print( "The Accuracy value is ",accuracy_dict[key_max], "with k= ", key_max)


# 

# In[ ]:





# In[ ]:


elbow_curve = pd.DataFrame(accuracy_list,columns = ['accuracy'])


# In[ ]:


elbow_curve.plot()


# In[ ]:





# **ROC CURVE**

# In[ ]:


"""
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, mLogReg.predict(x_test))
fpr, tpr, thresholds = roc_curve(y_test, mLogReg.predict_proba(x_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
"""

