#!/usr/bin/env python
# coding: utf-8

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


import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score, confusion_matrix, precision_score, f1_score, accuracy_score, classification_report


# In[ ]:


df = pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')


# ### Overview

# In[ ]:


df.head()


# In[ ]:


print('Rows: {}, Columns: {}'.format(df.shape[0], df.shape[1]))
features = df.columns.to_list()
features.remove('Churn')
print('Features:\n', features, sep='')


# Let's drop the customer ids as they do not have anything to do with churning.

# In[ ]:


df.drop(["customerID"], axis = 1,inplace = True)


# Checking if there are any null values.

# In[ ]:


df.isnull().sum()


# Seems like there are no NaNs.
# 
# But to be 100% sure, let's see if there is something else in place of a value.

# In[ ]:


def check_values():
    for i in range(df.columns.size):
        print(df.columns[i] + ':')
        for j in range(df[df.columns[i]].size):
            if df[df.columns[i]][j] == ' ' :
                print('Found space')
            elif df[df.columns[i]][j] == '-' :
                print('Found hyphen')
            elif df[df.columns[i]][j] == 'NA' :
                print('Found NA')
        print('Done!')


# In[ ]:


check_values()


# We have found 'Spaces' in TotalCharges column which can't be used for training.
# 
# Let's fix this.

# In[ ]:


# replacing spaces with null values
df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)


# In[ ]:


check_values()


# In[ ]:


df.isnull().sum()


# In[ ]:


df = df[df["TotalCharges"].notnull()]


# In[ ]:


df.isnull().sum()


# Fixing the index.

# In[ ]:


df.reset_index(drop = True, inplace = True)


# Let's check if the data types are defined properly.

# In[ ]:


df.dtypes


# TotalCharges is of object data type, we need to convert it into float.

# In[ ]:


df.TotalCharges = df.TotalCharges.astype(float)


# In[ ]:


df.dtypes


# Let's see the number of unique values in our data.

# In[ ]:


df.nunique()


# In[ ]:


for i in range(df.columns.size) :
    if df[df.columns[i]].nunique() <= 4:
        print(df[df.columns[i]].unique())


# Mapping yes to 1, no to 0 and no internet service to -1.

# In[ ]:


col_map = ['Partner', 
          'Dependents', 
          'PhoneService', 
          'MultipleLines',
          'OnlineSecurity',
          'OnlineBackup',
          'DeviceProtection',
          'TechSupport',
          'StreamingTV',
          'StreamingMovies',
          'PaperlessBilling', 
          'Churn']
for col in col_map:
    df[col] = [1 if val == "Yes" else 0 if val == "No" else -1 for val in df[col]]


# In[ ]:


for i in range(df.columns.size) :
    if df[df.columns[i]].nunique() <= 4:
        print(df[df.columns[i]].unique())


# Mapping male to 1 and female to 0 in Gender column.

# In[ ]:


df['gender'] = [1 if gen == 'Male' else 0 for gen in df['gender']]


# In[ ]:


df.head()


# ### Exploratory Data Analysis and Visualization

# Let's explore the dataset.

# In[ ]:


plt.figure(figsize = [15, 6])
plt.pie(df['Churn'].value_counts(), 
        labels = ['No', 'Yes'], 
        startangle = 90, 
        autopct='%1.1f%%', 
        wedgeprops = {'width' : 0.2},
        counterclock = True);
plt.title('Customer churn')
plt.legend()
plt.axis('equal');


# We observe that **26.6%** of the total customers have churned out.

# In[ ]:


plt.figure(figsize = [15, 6])
plt.suptitle('Gender distribution')

plt.subplot(1, 2, 1)
plt.pie(df[df['Churn'] == 1]['gender'].value_counts(), 
        labels = ['Female', 'Male'], 
        startangle = 90, 
        autopct='%1.1f%%', 
        wedgeprops = {'width' : 0.2},
        counterclock = True);
plt.legend()
plt.text(-0.13,-0.03, 'Churn',fontsize = 14)
plt.axis('equal')

plt.subplot(1, 2, 2)
plt.pie(df[df['Churn'] == 0]['gender'].value_counts(), 
        labels = ['Male', 'Female'], 
        startangle = 90, 
        autopct='%1.1f%%', 
        wedgeprops = {'width' : 0.2},
        counterclock = True);
plt.legend()
plt.text(-0.22,-0.03, 'Not Churn',fontsize = 14)
plt.axis('equal');


# We do not observe any drastic difference between churn and not churn customers based on their _gender_.

# Analysizing the probability distributions of churn and not churn customers against various features using Kernel Density Estimate(KDE) plot.

# In[ ]:


bluish = sns.color_palette()[0]
orangish = sns.color_palette()[1]


# In[ ]:


plt.figure(figsize = [15, 8])
ten_dist = sns.kdeplot(df['tenure'][df["Churn"] == 0], color = bluish, shade = True)
ten_dist = sns.kdeplot(df['tenure'][df["Churn"] == 1], color = orangish, shade= True)
ten_dist.legend(['Not Churn', 'Churn'])
ten_dist.set_xlabel('Tenure')
ten_dist.set_ylabel('Frequency')
plt.xticks(np.arange(0, 80, 5))
plt.title('Distribution of tenure for churn and not churn customers');


# We observe that probability of churning out is maximum for customers who have a **short tenure** (between 0 to approx. 15).

# In[ ]:


plt.figure(figsize = [15, 8])
ten_dist = sns.kdeplot(df['MonthlyCharges'][df["Churn"] == 0], color = bluish, shade = True)
ten_dist = sns.kdeplot(df['MonthlyCharges'][df["Churn"] == 1], color = orangish, shade= True)
ten_dist.legend(['Not Churn', 'Churn'])
ten_dist.set_xlabel('Monthly charges')
ten_dist.set_ylabel('Frequency')
plt.title('Distribution of monthly charges for churn and not churn customers');


# We observe that most of the customers who have churned out have **high monthly charages** when compared to the ones who have not churned out.

# In[ ]:


plt.figure(figsize = [15, 8])
ten_dist = sns.kdeplot(df['TotalCharges'][df["Churn"] == 0], color = bluish, shade = True)
ten_dist = sns.kdeplot(df['TotalCharges'][df["Churn"] == 1], color = orangish, shade= True)
ten_dist.legend(['Not Churn', 'Churn'])
ten_dist.set_xlabel('Total charges')
ten_dist.set_ylabel('Frequency')
plt.title('Distribution of total charges for churn and not churn customers');


# We observe that distribution of the customers who have churned out is high between 0 to approx. 1000.

# In[ ]:


plt.figure(figsize = [10,6])
sns.countplot(data = df, x = 'Contract', hue = 'Churn')
plt.legend(['Not Churn', 'Churn'])
plt.title('Contracts against churn and not churn customers', fontsize = 14);


# 
# We learn that most of the customers who have churned out were part of the month-to-month contract.
# 
# We also observe that the participation in one and two year contract is very less for the customers who have churned out whereas the participation is significant for those who have not.

# In[ ]:


plt.figure(figsize = [10,6])
sns.countplot(data = df, x = 'InternetService', hue = 'Churn')
plt.legend(['Not Churn', 'Churn'])
plt.title('Internet service against churn and not churn customers', fontsize = 14);


# We observe that the customers who opted for Fiber optic were likely to churn out.
# 
# We also see that the customers who opted for DSL were less likely to churn out.

# ### Data Pre-processing

# Converting categorical variables to indicator variables.

# In[ ]:


df = pd.get_dummies(data = df)
df.head()


# Let's build the correlation matrix to analyse the relation.

# In[ ]:


corr = df.corr()
fig = plt.figure(figsize = (8, 8))
ax = fig.add_subplot(111)
p = ax.matshow(corr, vmin = -1, vmax = 1)
fig.colorbar(p)
ticks = np.arange(0, 27, 1) 
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(df.columns.to_list(), rotation = 90)
ax.set_yticklabels(df.columns.to_list());


# In[ ]:


df.corr()['Churn'].sort_values()


# We observe that Contract_Month-to-month, InternetService_Fiber optic, PaymentMethod_Electronic check are highly positively correlated and Tenure, Contract_Two year are negatively highly correlated with Churn.

# In[ ]:


df.describe()


# We observe that our feautres are on different scales and this can slow the process of convergence of our learning algorithm. 
# 
# To fix this let's normalize our dataset.

# #### Mean normalization

# In[ ]:


X = df.drop(["Churn"], axis = 1)
X = (X - np.mean(X, axis = 0)) / np.std(X, axis = 0)
X.describe()


# In[ ]:


y = df['Churn'].values


# #### Splitting the dataset into train and test set

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


# In[ ]:


print('Test: ', X_train.shape[0], ', ', y_train.shape[0], sep = '')
print('Train: ', X_test.shape[0],',', y_test.shape[0], sep = '')


# ### Modelling

# #### Logistic Regression

# In[ ]:


lr_model = LogisticRegression()
lr_model.fit(X_train,y_train)
lr_train_acc = lr_model.score(X_train, y_train)
lr_test_acc = lr_model.score(X_test, y_test)
print('Logistic Regression')
print('Training accuracy:', lr_train_acc)
print('Testing accuracy:', lr_test_acc) 


# #### Support Vector Machine

# In[ ]:


svc_model = SVC(random_state = 1)
svc_model.fit(X_train, y_train)
svm_train_acc = svc_model.score(X_train,y_train)
svm_test_acc = svc_model.score(X_test,y_test)
print('SVM')
print('Training accuracy:', svm_train_acc)
print('Testing accuracy:', svm_test_acc)


# #### K-Nearest Neighbours

# Let's find out the optimal value for k.

# In[ ]:


plt.figure(figsize = (15, 6))
acc = []
acc_k = []
for k in range(1, 25):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    acc.append(knn.score(X_test,y_test))
    acc_k.append([knn.score(X_test,y_test), k])
    
plt.plot(range(1, 25), acc)
plt.xticks(np.arange(1, 26, 1))
plt.xlabel("Range")
plt.ylabel("Score")
plt.title('Finding k for KNN');


# In[ ]:


max(acc_ind)


# We find that k = 9 is the one that we should use for maximum accuracy.

# In[ ]:


knn_model = KNeighborsClassifier(n_neighbors = 9)
knn_model.fit(X_train, y_train)
knn_train_acc = knn_model.score(X_train, y_train)
knn_test_acc = knn_model.score(X_test, y_test)
print('KNN for k = 15')
print('Training accuracy:', knn_train_acc)
print('Testing accuracy:', knn_test_acc)


# ### Model Selection

# Since the data is imbalanced we can't just rely upon training and test accuracy.
# 
# We'll build a confusion matrix and use f1 score to select our model.

# In[ ]:


def scores(name, y, y_hat):
    acc = accuracy_score(y, y_hat)
    precision = precision_score(y, y_hat)
    recall = recall_score(y, y_hat)    
    f1 = f1_score(y, y_hat, average='weighted')
    
    print(name)    
    print('Accuracy:', acc)
    print('Precision: ', precision)                   
    print('Recall:', recall)
    print('F1_score:', f1)
    print()


# In[ ]:


scores("Logistic Regression",y_test, lr_model.predict(X_test))
scores("Support Vector Machine", y_test, svc_model.predict(X_test))
scores("K-Nearest Neighbors", y_test, knn_model.predict(X_test))


# We observe that Logistic Regression is the one with highest accuracy and F1 score thus we select it as our model.
# 
# Let's now visualize the confusion matrix for our model.

# In[ ]:


lr_matrix = confusion_matrix(y_test, lr_model.predict(X_test))
f, ax = plt.subplots(figsize = (8, 8))
sns.heatmap(lr_matrix, annot = True, color = "red", fmt = ".0f")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix");


# ### Conclusion

# - 26.6% of the total customers have churned out.
# - There isn't any drastic difference between churn and not churn customers based on their gender.
# - The probability of churning out is maximum for customers who have a short tenure (between 0 to approx. 15).
# - Most of the customers who have churned out have high monthly charges when compared to the ones who have not churned out.
# - Most of the customers who have churned out were part of the month-to-month contract.
# - The participation in one and two year contract is very less for the customers who have churned out whereas the participation is significant for those who have not
# - The customers who opted for Fiber optic were likely to churn out.
# - The customers who opted for DSL were less likely to churn out.
# - Features like Contract_Month-to-month, InternetService_Fiber optic, PaymentMethod_Electronic check are highly positively correlated and Tenure, Contract_Two year are negatively highly correlated with Churn.
# - We used F1 score to choose our model as the data was imbalanced.
# - Logistic Regression gave the best results in terms of accuracy and F1 score thus it is the best model.
