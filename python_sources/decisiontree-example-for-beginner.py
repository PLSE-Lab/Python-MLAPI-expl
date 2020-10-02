# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier # for decision tree
from sklearn.model_selection import train_test_split # to split the data
%matplotlib inline


import os
#print(os.listdir("../input/loan_data_demo_hitanshu1212/loan_data_demo_hitanshu1212.csv"))

# Any results you write to the current directory are saved as output.

# Load the dataset
dt_data = pd.read_csv('../input/loan_data_demo_h1221221.csv')
dt_data.head()

# To get the information about the dataset
dt_data.info()

dt_data.describe()

df_credit = dt_data['credit.policy'].value_counts()
df_credit

# plot hist graph to get the deep information and relation of columns
x = dt_data[dt_data['credit.policy'] == 1]['fico']
y = dt_data[dt_data['credit.policy'] == 0]['fico']
plt.subplots(figsize = (12,6))
plt.plot()
plt.hist(x,bins=30,alpha=0.5,color='blue', label='Credit.Policy=1')
plt.hist(y,bins=30,alpha=0.5,color='red', label='Credit.Policy=0')
plt.title ("Histogram of FICO score by approved or disapproved credit policies", fontsize=16)
plt.xlabel("FICO score", fontsize=14)
plt.show()

# To explore the data more deply
sns.boxplot(x = dt_data['credit.policy'], y = dt_data['int.rate'])
plt.title("Interest rate varies between risky and non-risky borrowers", fontsize=15)
plt.xlabel("Credit policy",fontsize=15)
plt.ylabel("Interest rate",fontsize=15)

# To explore the data more deply
sns.boxplot(x = dt_data['credit.policy'], y = dt_data['log.annual.inc'])
plt.title("Interest rate varies between risky and non-risky borrowers", fontsize=15)
plt.xlabel("Credit policy",fontsize=15)
plt.ylabel("log.annual.inc",fontsize=15)

# To explore the data more deply
sns.boxplot(x = dt_data['credit.policy'], y = dt_data['installment'])
plt.title("Interest rate varies between risky and non-risky borrowers", fontsize=15)
plt.xlabel("Credit policy",fontsize=15)
plt.ylabel("installment",fontsize=15)

plt.figure(figsize=(10,6))
sns.countplot(x='purpose',hue='not.fully.paid',data=dt_data, palette='Set1')
plt.title("Bar chart of loan purpose colored by not fully paid status", fontsize=17)
plt.xlabel("Purpose", fontsize=15)

# Setting up the data
df_final = pd.get_dummies(dt_data,['purpose'],drop_first=True)

# Split data in traning and testing
X = df_final.drop('not.fully.paid',axis=1)
Y = df_final['not.fully.paid']
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=0)

# Creating a dicision tree model
dtree = DecisionTreeClassifier(criterion='gini',max_depth=None)

dtree.fit(x_train,y_train)

# Predictions and Evaluation of Decision Tree
y_pred = dtree.predict(x_test)

from sklearn.metrics import classification_report,confusion_matrix

cm = confusion_matrix(y_test,y_pred)
cm

cr = classification_report(y_test,y_pred)
print(cr)