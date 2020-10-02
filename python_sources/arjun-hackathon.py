#!/usr/bin/env python
# coding: utf-8

# ## Predict the churn for customer data

# In[ ]:


# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import LabelEncoder 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# Importing the dataset
#../input/predict-the-churn-for-customer-dataset/
data_train = pd.read_csv('../input/predict-the-churn-for-customer-dataset/Train File.csv')
data_test = pd.read_csv('../input/predict-the-churn-for-customer-dataset/Test File.csv')


# ## Removing Irrelevant data

# In[ ]:


# Removing irrelevant data
test_customerId = data_test['customerID']
data_train.drop(['customerID'], axis=1, inplace=True)
data_test.drop(['customerID'], axis=1, inplace=True)


# # Visualization of Customer data with features
# ### Customer Gender Distribution 

# In[ ]:


a4_dims = (11.7, 8.27)
ax = plt.subplots(figsize=a4_dims)
colors = ['#94aa2a','#F7DC6F']
ax = (data_train['gender'].value_counts()*100.0 /len(data_train)).plot.pie(autopct='%.1f%%', labels = ['Male', 'Female'],figsize =(5,5), fontsize = 12,colors = colors )


# ## Distribution of services and customers are availing

# In[ ]:


features = ['PhoneService','MultipleLines','InternetService','OnlineSecurity',
           'OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
colors1 =['#f45905','#c70d3a','#512c62']
colors2 =['#94aa2a','#e47312','#d55252']
colors3= ['#A04000','#D68910','#F7DC6F']
fig, axes = plt.subplots(nrows = 3,ncols = 3,figsize = (15,12))
for i, item in enumerate(features):
    if i < 3:
        ax = data_train[item].value_counts().plot(kind = 'bar',ax=axes[i,0],rot = 0, color=colors1)
        
    elif i >=3 and i < 6:
        ax = data_train[item].value_counts().plot(kind = 'bar',ax=axes[i-3,1],rot = 0,color = colors2)
        
    elif i < 9:
        ax = data_train[item].value_counts().plot(kind = 'bar',ax=axes[i-6,2],rot = 0,color=colors3)
    ax.set_title(item)


# ## Distrimution of churn rate

# In[ ]:


a4_dims = (11.7, 8.27)
ax = plt.subplots(figsize=a4_dims)
colors = ['#512c62','#f45905']
ax = (data_train['Churn'].value_counts()*100.0 /len(data_train)).plot.pie(autopct='%.1f%%', labels = ['NO', 'Yes'],figsize =(5,5), fontsize = 12,colors = colors )


# ## churn vs monthly charges distribution

# In[ ]:


dims = (10, 8)
ax = plt.subplots(figsize=dims)

ax = sns.kdeplot(data_train.MonthlyCharges[(data_train["Churn"] == 'No') ],
                color="red", shade = True)
ax = sns.kdeplot(data_train.MonthlyCharges[(data_train["Churn"] == 'Yes') ],
                ax =ax, color="skyblue", shade= True)
ax.legend(["Not Churn","Churn"],loc='upper right')
ax.set_ylabel('Density')
ax.set_xlabel('Monthly Charges')
ax.set_title('churn vs monthly charges distribution')


# # Data Manipulation

# In[ ]:


# Check and null values in the data
#data_train.info()
data_train.fillna(data_train.mean(), inplace=True)
data_test.fillna(data_test.mean(), inplace=True)


# In[ ]:


#data_train.info()


# In[ ]:


# Datatype conversion
data_train['TotalCharges'] = pd.to_numeric(data_train['TotalCharges'])
data_test['TotalCharges'] = pd.to_numeric(data_test['TotalCharges'])


# In[ ]:


# Convert categorical data to numerical values
labelEncoder = LabelEncoder()

data_train['gender'] = labelEncoder.fit_transform(data_train['gender'])
data_train['Partner'] = labelEncoder.fit_transform(data_train['Partner'])
data_train['Dependents'] = labelEncoder.fit_transform(data_train['Dependents'])
data_train['PhoneService'] = labelEncoder.fit_transform(data_train['PhoneService'])
data_train['MultipleLines'] = labelEncoder.fit_transform(data_train['MultipleLines'])
data_train['InternetService'] = labelEncoder.fit_transform(data_train['InternetService'])
data_train['OnlineSecurity'] = labelEncoder.fit_transform(data_train['OnlineSecurity'])
data_train['OnlineBackup'] = labelEncoder.fit_transform(data_train['OnlineBackup'])
data_train['DeviceProtection'] = labelEncoder.fit_transform(data_train['DeviceProtection'])
data_train['TechSupport'] = labelEncoder.fit_transform(data_train['TechSupport'])
data_train['StreamingTV'] = labelEncoder.fit_transform(data_train['StreamingTV'])
data_train['StreamingMovies'] = labelEncoder.fit_transform(data_train['StreamingMovies'])
data_train['Contract'] = labelEncoder.fit_transform(data_train['Contract'])
data_train['PaperlessBilling'] = labelEncoder.fit_transform(data_train['PaperlessBilling'])
data_train['PaymentMethod'] = labelEncoder.fit_transform(data_train['PaymentMethod'])
data_train['Churn'] = labelEncoder.fit_transform(data_train['Churn'])

data_test['gender'] = labelEncoder.fit_transform(data_test['gender'])
data_test['Partner'] = labelEncoder.fit_transform(data_test['Partner'])
data_test['Dependents'] = labelEncoder.fit_transform(data_test['Dependents'])
data_test['PhoneService'] = labelEncoder.fit_transform(data_test['PhoneService'])
data_test['MultipleLines'] = labelEncoder.fit_transform(data_test['MultipleLines'])
data_test['InternetService'] = labelEncoder.fit_transform(data_test['InternetService'])
data_test['OnlineSecurity'] = labelEncoder.fit_transform(data_test['OnlineSecurity'])
data_test['OnlineBackup'] = labelEncoder.fit_transform(data_test['OnlineBackup'])
data_test['DeviceProtection'] = labelEncoder.fit_transform(data_test['DeviceProtection'])
data_test['TechSupport'] = labelEncoder.fit_transform(data_test['TechSupport'])
data_test['StreamingTV'] = labelEncoder.fit_transform(data_test['StreamingTV'])
data_test['StreamingMovies'] = labelEncoder.fit_transform(data_test['StreamingMovies'])
data_test['Contract'] = labelEncoder.fit_transform(data_test['Contract'])
data_test['PaperlessBilling'] = labelEncoder.fit_transform(data_test['PaperlessBilling'])
data_test['PaymentMethod'] = labelEncoder.fit_transform(data_test['PaymentMethod'])


# In[ ]:


#data_train.info()


# # Modeling using Logistic Regression

# In[ ]:


# Seperating the data
X_train = data_train.drop(['Churn'], axis=1)
y_train = data_train['Churn']

X_test = data_test


# In[ ]:


# Building the Logistic Regression model
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
result = lr_model.fit(X_train, y_train)


# In[ ]:


from sklearn import metrics
prediction = lr_model.predict(X_test)


# In[ ]:


prediction


# In[ ]:


df_predicted = pd.DataFrame(columns=['customerID','Churn'])


# In[ ]:


df_predicted['customerID'] = test_customerId
df_predicted['Churn'] = prediction

df_predicted['Churn'] = df_predicted['Churn'].replace(dict({1:'Yes',0:'No'}))

df_predicted.head()


# In[ ]:


df_predicted.count()


# In[ ]:


df_predicted.to_csv('submission.csv', index=False)


# In[ ]:


from IPython.display import FileLink
FileLink(r'submission.csv')

