#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Few necessary imports
import os
import subprocess
import matplotlib.pyplot as plt
import graphviz
import seaborn as sns


# In[ ]:


# Load data
data = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Don't need customerID to find out trends in the data
data = data.drop(columns=['customerID'])
# Convert TotalCharges from string to numeric datatype and fill NaN values witht the median
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median())
# Convert SeniorCitizen from integer to string
data['SeniorCitizen'] = data['SeniorCitizen'].apply(lambda x: 'Yes' if x==1 else 'No')

columns = list(data.columns)
non_numeric_cols = list(set(columns) - set(['tenure', 'TotalCharges', 'MonthlyCharges']))
numeric_cols = ['tenure', 'TotalCharges', 'MonthlyCharges']


# ## Desriptive Analysis

# In[ ]:


for column in non_numeric_cols:
    print('------------------------')
    print(data[column].value_counts())  


# In[ ]:


data[['tenure', 'TotalCharges', 'MonthlyCharges']].describe()


# ## Exploratory Analysis

# In[ ]:


# Remove Churn from feature columns since it is the target value
non_numeric_cols.remove('Churn')


# In[ ]:


# Exploratory analysis on non-continuous features
for idx in range(0, len(non_numeric_cols), 2):
    plt.subplots(figsize=(12,4))
    plt.subplot(1, 2, 1)
    ax = sns.countplot(x=non_numeric_cols[idx], data=data, alpha=0.6)
    ax = sns.countplot(x=non_numeric_cols[idx], data=data[data['Churn'] == 'Yes'])
    ax.set(ylabel='churned / total')
    plt.subplot(1, 2, 2)
    ax = sns.countplot(x=non_numeric_cols[idx+1], data=data, alpha=0.6)
    ax = sns.countplot(x=non_numeric_cols[idx+1], data=data[data['Churn'] == 'Yes'])
    ax.set(ylabel='churned / total')
    plt.tight_layout()
    plt.show()


# #### Conclusions Drawn:
# - Customers **without online security** have a significantly high churn rate
# - Customers who pay using **electronic check** have a significantly high churn rate
# - Customers having a **month-to-month contract** have a significantly higher rate of leaving
# - Customers who **don't have multiple lines** almost certainly have left in the last month
# - Customers having **fiber optic internet service** have higher churn rates
# - Customers without services like tech support, online backup, device protection have a high churn rate
# - Customers who have phone services tend to have higher churn rates
# - Number of customers who left in the last month is almost equal for males and females
# - Senior citizens tend to have higher churn rates
# - Customers who don't have dependents tend to leave more often tha the ones that do
# - Customers who don't have partners have a higher churn rate
# - Customers with and without StreamingTV facility have similar churn rates
# - Customers who have paperless billing have a higher rate of leaving

# In[ ]:


# Exploratory analysis on continuous features
for column in numeric_cols:
    plt.subplots(figsize=(12,4))
    plt.subplot(1, 2, 1)
    sns.distplot(data[data['Churn'] == 'Yes'][column], label='Yes')
    sns.distplot(data[data['Churn'] == 'No'][column], label='No')
    plt.legend()
    plt.subplot(1, 2, 2)
    sns.boxplot(x=column, y='Churn', hue='Churn', data=data)
    plt.tight_layout()
    plt.show()


# #### Conclusions drawn:
# - Customers who have not been with the company for long (0-20 months) tend to have a higher rate of leaving than customers who have been with the company for a long time
# - The median tenure for customers who have left in the last month is close to 10 while the median tenure for customers who have stayed is close to 40
# - Customers having higher monthly charges (>65) have a higher tendency to leave than customers having lower monthly charges
# - The median monthly charge of the customers who left in the last month is close to 80
# - Total Charges has a lot of outliers and is not very intutive, dropping this feature is probably the best option

# In[ ]:


# Correlation matrix to find the dependencies amongst the continuous features
correlation_matrix = data[numeric_cols].corr()
sns.heatmap(correlation_matrix,
            xticklabels=correlation_matrix.columns.values,
            yticklabels=correlation_matrix.columns.values)


# #### Conclusions drawn:
# - Tenure and Monthly Charges have a weak correlation
# - Total Charges is significantly dependent on both Tenure and Monthly Charges, which reinforces our decision to drop the TotalCharges as a feature

# In[ ]:


# Dropping TotalCharges column from dataframe
data = data.drop(columns=['TotalCharges'])
numeric_cols.remove('TotalCharges')
columns = list(data.columns)


# ## Predictive Analysis

# In[ ]:


from sklearn import tree, svm
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel


# In[ ]:


# List of columns containg boolean data
bool_cols = ['Partner', 'Dependents', 'SeniorCitizen', 'PaperlessBilling', 'PhoneService']

# Convert string data with two classes to boolean values
for column in bool_cols:
    data[column] = data[column].apply(lambda x: 1 if x in 'Yes' else 0)
data['gender'] = data['gender'].apply(lambda x: 1 if x in 'Female' else 0)

# Create dummy variables for features with more than two classes
non_numeric_data = pd.get_dummies(data[non_numeric_cols])
non_numeric_data.head()


# In[ ]:


# Standardization of continuous features
numeric_data = pd.DataFrame(scale(data[numeric_cols]), index=data.index, columns=numeric_cols)
numeric_data.head()


# In[ ]:


# Create the final feature dataframe to perform predictive analysis
features = pd.concat([numeric_data, non_numeric_data], axis=1)
features.head()


# In[ ]:


# Create target dataframe for predictive analysis
labels = data['Churn'].apply(lambda x: 1 if x in 'Yes' else 0)
labels.head()


# In[ ]:


# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2)
print('X_train: ', X_train.shape)
print('y_train: ', y_train.shape)
print('X_val: ', X_val.shape)
print('y_val: ', y_val.shape)

kfold = model_selection.KFold(n_splits=10, random_state=101)
scoring = 'accuracy'


# ### 10-fold Cross Validation

# In[ ]:


models = {'decision_tree': tree.DecisionTreeClassifier(min_samples_split=20, max_depth=11),
         'random_forest': RandomForestClassifier(criterion='entropy', random_state=101, n_estimators=200, max_depth=11),
         'logistic_regression': LogisticRegressionCV(),
         'svm_model': svm.SVC(kernel='linear', C=1),
         'mlp': MLPClassifier(hidden_layer_sizes=(16, 16), max_iter=1000)
        }

for key, model in models.items():
    results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    print("10-fold cross-validation average accuracy for {}: %.3f".format(key) % (results.mean()))


# #### Conclusions drawn:
# - The Logistic Regression model outperforms the other models in the 10-fold cross validation tests
# - The SVM with linear SVC kernel comes close, but is heavier compared to the logistic regression model which makes it the worse option out of the two

# ### Decision Tree
# Even though the decision tree model performed the worst in our cross-validation test, we could still draw several useful insights after visualizing the trained decision tree.

# In[ ]:


decision_tree = models['decision_tree'].fit(X_train, y_train)


# In[ ]:


y_pred = decision_tree.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print("Accuracy: {}".format(acc))
print()
print(classification_report(y_val,y_pred))


# In[ ]:


# Visualize the decision tree and save it as a png file
dot_data = tree.export_graphviz(decision_tree, out_file=None, feature_names=features.columns,
                         filled=True, rounded=True, special_characters=True)

graph = graphviz.Source(dot_data)
graph


# In[ ]:


with open("decision_tree.dot", 'w') as f:
    dot_data = tree.export_graphviz(decision_tree, out_file=f, feature_names=features.columns,
                         filled=True, rounded=True, special_characters=True)
try:
    subprocess.check_call(["dot", "-Tpng", "decision_tree.dot", "-o", "decision_tree.png"])
    subprocess.check_call(["rm", "decision_tree.dot"])
    print("Graph successfully stored as decision_tree.png")
except:
    exit("Could not run dot, ie graphviz, to produce visualization")


# ### Logistic Regression

# In[ ]:


log_reg = models['logistic_regression'].fit(X_train, y_train)


# In[ ]:


y_pred = log_reg.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print("Accuracy: {}".format(acc))
print()
print(classification_report(y_val,y_pred))


# ### Multi-Layered Perceptron

# In[ ]:


mlp = models['mlp'].fit(X_train, y_train)


# In[ ]:


y_pred = mlp.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print("Accuracy: {}".format(acc))
print()
print(classification_report(y_val,y_pred))


# ## Solutions
# We will now try to find out how changing certain important features affects the churn propensity

# In our exploratory analysis, we made the following discoveries:
# - Customers **without online security** have a significantly high churn rate
# - Customers who **don't have multiple lines** almost certainly have left in the last month
# - Customers without services like tech support, online backup, device protection have a high churn rate
# 
# What happens if we assume that these services are offered to future customers?

# In[ ]:


churn = log_reg.predict(X_val)
churn_rate = np.sum(churn)/len(churn)

print("Current churn rate: {}%".format(np.round(churn_rate*100,2)))


# In[ ]:


# Returns the churn rate after removing specific features
# The coefficient of that particular feature is made 0 so that it doesn't contribute to the predictions
def remove_features(feat_idx):
    lr = LogisticRegressionCV()
    lr.coef_ = np.copy(log_reg.coef_)
    for idx in feat_idx:
        lr.coef_[0, idx] = 0
    lr.intercept_ = log_reg.intercept_
    lr.classes_ = log_reg.classes_
    churn = lr.predict(X_val)
    churn_rate = np.sum(churn)/len(churn)
    return np.round(churn_rate*100,2)


# In[ ]:


feature_list = ['OnlineSecurity', 'MultipleLines', 'TechSupport', 'OnlineBackup', 'DeviceProtection']
feat_idx = []

for feat in feature_list:
    idx = list(features.columns).index(feat+'_No')
    feat_idx.append(idx)
    print("Churn rate if all customers had {}: {}%".format(feat, remove_features([idx])))

print("\nChurn rate if all customers had all of the above facilities: {}%".format(remove_features(feat_idx)))


# Though, this might not be the best way to make such inferences, but this method offers an approximate way to see how each feature affects the churn propensity. The churn rate went down from **21%** to **13%** if all of the above facilities were being taken up by all of the customers. This inference intutively seems to be correct since offering more and better services to customers should make them want to stay with the company for longer.

# In[ ]:


idx = list(features.columns).index('PaymentMethod_Electronic check')
print("Churn rate if the company discontinues the electronic check payment method: {}%".format(remove_features([idx])))


# In[ ]:


idx = list(features.columns).index('InternetService_Fiber optic')
print("Churn rate if the company starts offering better fiber optic internet service: {}%".format(remove_features([idx])))


# In[ ]:


idx = list(features.columns).index('Contract_Month-to-month')
print("Churn rate if the company discontinues the month-to-month contract: {}%".format(remove_features([idx])))


# #### Solutions:
# - There seems to be a problem with the tech services that the company offers to its customers. A huge proportion of the customers who don't have these services tend to leave the company. Services such as online security, tech support, online backup, and device protection, if made better, could lead to a higher customer retention rate.
# - Customers seem uncomfortable with the lectronic check payment method. The payment method should be looked into for problems or discontinued all together.
# - The fiber optic internet services being offered by the company seem to be having some problems. Customers who have opted for this service in the past have had higher rates of leaving the company. Improving the quality of the fibe optic service could definitely reduce the number of customers leaving the company.
# - The month-to-month contract seems to be the biggest reason why a lot of the customers leave the company. A reason for this could be that the month-to-month contract has an extremely high monthly charge when compared to the other contract offers. Reducing the monthly charge on the month-to-month contract is a sure shot way of increasing customer retention.
