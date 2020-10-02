#!/usr/bin/env python
# coding: utf-8

# # Predicting churn for bank

# **Importing all the necessary libraries**

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel


# In[ ]:


data = pd.read_csv("../input/predicting-churn-for-bank-customers/Churn_Modelling.csv")


# **For checking correlation between features**

# In[ ]:


data.corr()


# In[ ]:


data.shape


# **Let's observer data**

# In[ ]:


data.head()


# In[ ]:


data["Geography"].unique() #checking for unique values in Geography


# In[ ]:


data.describe()


# In[ ]:


data.dtypes


# # Data Visualization

# In[ ]:


plt.figure(figsize = (15,15))
sns.catplot(x = 'Geography', kind = 'count', data = data, palette = 'pink')
plt.title('Customers distribution across Countries')
plt.show()


# **Maximum customers from France**

# In[ ]:


plt.figure(figsize = (15,15))
sns.catplot(x = 'Gender', kind = 'count', data = data, palette = 'pastel')
plt.title("Males vs Females")
plt.show()


# **We have more male customers**

# In[ ]:


plt.figure(figsize = (15,15))
sns.catplot(x = 'IsActiveMember', kind = 'count', data = data, palette = 'pink')
plt.title("Active VS Non-Active Members")
plt.show()


# **We have more active members**

# In[ ]:


plt.figure(figsize = (15,15))
sns.catplot(x = 'HasCrCard', kind = 'count', palette = 'pastel', data = data)
plt.title("Credit Card VS No Credit Card")
plt.show()


# **Most of the customers have credit card**

# In[ ]:


plt.figure(figsize = (15,15))
sns.catplot(x = 'Exited', kind = 'count', hue = 'Gender', palette = 'pink', data = data)
plt.title("Gender and Exited")
plt.show()


# **Females are more likely to exit**

# In[ ]:


plt.figure(figsize = (15,15))
sns.catplot(x = 'HasCrCard', kind = 'count', hue = 'Gender', palette = 'pastel', data = data)
plt.title("Gender and Credit Card")
plt.show()


# **Males generally have credit card**

# But on other hand they are more likely not to have credit cards too

# In[ ]:


plt.figure(figsize = (15,15))
sns.catplot(x = 'IsActiveMember', kind = 'count', hue = 'Gender', palette = 'pink', data = data)
plt.title("Gender and Active Members")
plt.show()


# **Males are more likely to be active members**

# **But on other hand males are also likely to be non active members**

# In[ ]:


plt.figure(figsize = (15,15))
sns.catplot(x = "NumOfProducts", kind = 'count', palette = 'pastel', data = data )
plt.title('Number of Products')
plt.show()


# **Most of the customers have 1 or 2 products from bank**

# In[ ]:


plt.figure(figsize = (15,15))
sns.catplot(x = 'Tenure', kind = 'count', palette = 'pastel', data = data)
plt.title("Tenure of Customer")
plt.show()


# **Most customers have tenure of in between 1-8 years in bank **

# In[ ]:


plt.figure(figsize = (15,15))
sns.catplot(x = 'Exited', kind = 'count', hue = 'IsActiveMember', palette = 'pink', data = data)
plt.title("Exited and Active Members")
plt.show()


# **Non active members are likely to exit more, quite understandable**

# In[ ]:


plt.figure(figsize = (15,15))
sns.catplot(x = 'Exited', kind = 'count', hue = 'HasCrCard', palette = 'pastel', data = data)
plt.title("Exited and Card")
plt.show()


# **Customers with credit card are likely to exit more**

# In[ ]:


plt.figure(figsize = (15,15))
sns.catplot(x = 'IsActiveMember', kind = 'count', hue = 'HasCrCard', palette = 'pink', data = data)
plt.title('Active Member and Card')
plt.show()


# In[ ]:


plt.figure(figsize = (15,15))
sns.scatterplot(x = 'Balance', y = 'EstimatedSalary', hue = 'Exited',palette = 'pastel', data = data)
plt.title("Balance vs Estimated Salary")
plt.show()


# In[ ]:


plt.figure(figsize = (15,15))
sns.scatterplot(x = 'Balance', y = 'CreditScore', hue = 'Exited',palette = 'pink', data = data)
plt.title("Balance vs Credit Score")
plt.show()


# In[ ]:


plt.figure(figsize = (15,15))
sns.scatterplot(x = 'Balance', y = 'EstimatedSalary', hue = 'Gender',palette = 'pastel', data = data)
plt.title("Estimated Salary vs Credit Score")
plt.show()


# In[ ]:


plt.figure(figsize = (15,15))
sns.scatterplot(x = 'Balance', y = 'EstimatedSalary', hue = 'IsActiveMember',palette = 'pastel', data = data)
plt.title("Estimated Salary vs Credit Score")
plt.show()


# # Data Preprocessing

# In[ ]:


data.drop(['RowNumber', 'CustomerId', 'Surname'], axis = 1, inplace = True)


# In[ ]:


data.isnull().sum() #checking for null values


# For checking skwness in the data

# In[ ]:


plt.figure(figsize = (15,15))
sns.distplot(data['Age'])
plt.title("Age")
plt.show()


# In[ ]:


plt.figure(figsize = (15,15))
sns.distplot(data["CreditScore"])
plt.title("Credit Score")
plt.show()


# In[ ]:


plt.figure(figsize = (15,15))
sns.distplot(data["EstimatedSalary"])
plt.title("Estimated Salary")
plt.show()


# In[ ]:


plt.figure(figsize = (15,15))
sns.distplot(data["Balance"])
plt.title("Balance")
plt.show()


# **For detecting outliers in data**

# In[ ]:


column = ["Age", "Balance", "EstimatedSalary", "CreditScore"]
for i in column:
    plt.figure(figsize = (15,15))
    sns.boxplot(data[i])
    plt.title('Box Plot')
    plt.show()


# In[ ]:


data = data[(data["Age"] <60)]
data = data[(data["CreditScore"] >400)]


# In[ ]:


data.describe()


# **Normalizing the data**

# In[ ]:


data["Balance"] = QuantileTransformer().fit_transform(data["Balance"].values.reshape(-1,1))
data["CreditScore"] = QuantileTransformer().fit_transform(data["CreditScore"].values.reshape(-1,1))
data["EstimatedSalary"] = QuantileTransformer().fit_transform(data["EstimatedSalary"].values.reshape(-1,1))
data["Age"] = QuantileTransformer().fit_transform(data["Age"].values.reshape(-1,1))


# In[ ]:


data["Balance"] = StandardScaler().fit_transform(data["Balance"].values.reshape(-1,1))
data["CreditScore"] = StandardScaler().fit_transform(data["CreditScore"].values.reshape(-1,1))
data["EstimatedSalary"] = StandardScaler().fit_transform(data["CreditScore"].values.reshape(-1,1))


# In[ ]:


data.describe()


# **Label Encoding for categorical columns**

# In[ ]:


data["Geography"] = LabelEncoder().fit_transform(data["Geography"])
data["Gender"] = LabelEncoder().fit_transform(data["Gender"])


# In[ ]:


data.head()


# In[ ]:


data.corr()


# # Splitting Train and Test Data

# In[ ]:


y = data["Exited"]


# In[ ]:


y.head()


# In[ ]:


data.drop(["Exited"], axis = 1, inplace = True)


# In[ ]:


data.head()


# In[ ]:


train_x,test_x,train_y,test_y = train_test_split(data,y, test_size = 0.3, random_state = 50)


# # Model Fitting

# # Logistic Regression

# In[ ]:


logistic = LogisticRegression()
logistic.fit(train_x,train_y)
log_y = logistic.predict(test_x)
print(accuracy_score(log_y,test_y))


# **Tuning the model**

# In[ ]:


random_parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] ,'penalty':['l1','l2']}
print(random_parameters)


# In[ ]:


random_para = RandomizedSearchCV(estimator = logistic, param_distributions = random_parameters, n_iter = 50, cv = 10, verbose=2, random_state= 50, n_jobs = -1)
random_para.fit(train_x,train_y)


# In[ ]:


random_para.best_params_


# In[ ]:


logistic2 = LogisticRegression(penalty ='l2', C =1)
logistic2.fit(train_x,train_y)
log_y = logistic2.predict(test_x)
print(accuracy_score(log_y,test_y))


# **Feature Selection**

# In[ ]:


feature = SelectFromModel(LogisticRegression())
feature.fit(train_x,train_y)
feature_support = feature.get_support()
feature_selected = train_x.loc[:,feature_support].columns.tolist()
print(str(len(feature_selected)), 'selected features')


# In[ ]:


print(feature_selected)


# In[ ]:


train_x_feature = train_x[["Age", "IsActiveMember"]]
train_x_feature.head()


# In[ ]:


test_x_feature = test_x[["Age", "IsActiveMember"]]
test_x_feature.head()


# In[ ]:


logistic.fit(train_x_feature, train_y)
log_y_feature = logistic.predict(test_x_feature)
print(accuracy_score(log_y_feature, test_y))


# # Random Forest Classifier

# In[ ]:


random = RandomForestClassifier()
random.fit(train_x,train_y)
random_y = random.predict(test_x)
print(accuracy_score(random_y,test_y))


# **Tuning Parameters**

# In[ ]:


n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto','sqrt']
max_depth = [int(x) for x in np.linspace(10,110,num=11)]
max_depth.append(None)
min_samples_split = [2,5,10]
min_samples_leaf = [1,2,4]
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,
'max_features': max_features,
'max_depth': max_depth,
'min_samples_split': min_samples_split,
'min_samples_leaf': min_samples_leaf,
'bootstrap': bootstrap
}
print(random_grid)


# In[ ]:


random_para = RandomizedSearchCV(estimator = random, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
random_para.fit(train_x,train_y)


# In[ ]:


random_para.best_params_


# In[ ]:


random_2 = RandomForestClassifier(n_estimators=1400,min_samples_split =10,min_samples_leaf= 2,max_features = 'sqrt',max_depth=80,bootstrap= True)
random_2.fit(train_x,train_y)
random_2_y = random_2.predict(test_x)
print(accuracy_score(random_2_y,test_y)) 


# **Feature Selection**

# In[ ]:


feature = SelectFromModel(RandomForestClassifier(n_estimators=1400,min_samples_split =10,min_samples_leaf= 2,max_features = 'sqrt',max_depth=80,bootstrap= True))
feature.fit(train_x,train_y)
feature_support = feature.get_support()
feature_selected = train_x.loc[:,feature_support].columns.tolist()
print(str(len(feature_selected)), 'selected features')


# In[ ]:


feature_selected


# In[ ]:


train_x_feature = train_x[['Age', 'Balance', 'NumOfProducts']]
train_x_feature.head()


# In[ ]:


test_x_feature = test_x[['Age', 'Balance', 'NumOfProducts']]
test_x_feature.head()


# In[ ]:


random_2.fit(train_x_feature,train_y)
random_2_feature_y = random_2.predict(test_x_feature)
print(accuracy_score(random_2_feature_y,test_y))


# # Naive Bayes

# In[ ]:


bayes = GaussianNB()
bayes.fit(train_x,train_y)
bayes_y = bayes.predict(test_x)
print(accuracy_score(bayes_y,test_y))


# **Feature Selection**

# In[ ]:


train_x_feature = train_x[["Age", "Balance"]] #based on correlation values
train_x_feature.head()


# In[ ]:


test_x_feature = test_x[["Age", "Balance"]] #based on correlation values
test_x_feature.head()


# In[ ]:


bayes.fit(train_x_feature,train_y)
bayes_feature_y =bayes.predict(test_x_feature)
print(accuracy_score(bayes_feature_y, test_y))


# # Conclusion

# The highest accuracy we achieved is by hypertuning RandomForestClassifier and using all the features.

# In[ ]:


print(str((accuracy_score(random_2_y,test_y)) * 100) + "%")


# **When we get time this notebook will be updated**

# If you like my work, please upvote.
