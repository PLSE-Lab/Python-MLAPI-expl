#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# importing Libraries
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from pylab import plot, show, subplot, specgram, imshow, savefig
from sklearn import preprocessing
from sklearn import cross_validation, metrics
from sklearn.preprocessing import Normalizer
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import Imputer

import matplotlib.pyplot as plote

get_ipython().run_line_magic('matplotlib', 'inline')

plt.style.use('ggplot')


# In[ ]:


train = pd.read_csv("../input/bank.csv")


# In[ ]:


train.head()


# In[ ]:


train.info()


# In[ ]:


train.describe()


# In[ ]:


train.columns


# In[ ]:


list(set(train.dtypes.tolist()))


# ## function for inding missing values in dataset

# In[ ]:


# To check how many columns have missing values - this can be repeated to see the progress made
def show_missing():
    missing = train.columns[train.isnull().any()].tolist()
    return missing


# In[ ]:


train[show_missing()].isnull().sum()


# ### this shows that there is no any missing data

# In[ ]:





# # Correlation in Data

# In[ ]:



corr=train.corr()["y"]
corr[np.argsort(corr, axis=0)[::-1]]


# # VISUALIZATION

# In[ ]:


#plotting correlations
num_feat=train.columns[train.dtypes!=object]
num_feat=num_feat[1:-1] 
labels = []
values = []
for col in num_feat:
    labels.append(col)
    values.append(np.corrcoef(train[col].values, train.y.values)[0,1])

ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(12,8))
rects = ax.barh(ind, np.array(values), color='red')
ax.set_yticks(ind+((width)/2.))
ax.set_yticklabels(labels, rotation='horizontal')
ax.set_xlabel("Correlation coefficient")
ax.set_title("Correlation Coefficients w.r.t y");


# ### Heat map

# In[ ]:


corrMatrix=train[num_feat].corr()
sns.set(font_scale=1.10)
plt.figure(figsize=(10, 10))
sns.heatmap(corrMatrix, vmax=.8, linewidths=0.01,
            square=True,annot=True,cmap='viridis',linecolor="white")
plt.title('Correlation between features');


# the above correlation plot shows that cons_conf_idx,previous,duration causing y=1 and the others causing y =0

# ## above shows the correlation of data wrt y ie how much these columns affecting the y

# In[ ]:


#sns.pairplot(train)


# finging age distribution

# In[ ]:


plt.hist(train.age,bins=10)
plt.show()


# this shows that the coustomer between 25 to 40 is more likely to subscribe the bank

# In[ ]:


plt.hist((train.duration),bins=152)
plt.show()


# In[ ]:


# Let's see how the numeric data is distributed.

train.hist(bins=10, figsize=(20,15), color='#E14906')
plt.show()


# ### this plot shows relation between y and durations

# In[ ]:


ax = train.groupby('y').duration.mean().plot(kind='bar')
ax.set_xlabel("y(outcome)")
ax.set_ylabel("mean durations")


#                                                                            this shows that average duration for not subscribing is less than subscribing

# ### this plot shows relation between y and age

# In[ ]:


ax = train.groupby('y').age.mean().plot(kind='bar')
ax.set_xlabel("y(outcome)")
ax.set_ylabel("mean ages")


# In[ ]:


ax = train.groupby('y').previous.mean().plot(kind='bar')
ax.set_xlabel("y(outcome)")
ax.set_ylabel("mean previous")


# In[ ]:


# This is to create each of the categories.
lst = [train]
for column in lst:
    column.loc[column["age"] < 30,  "age_category"] = 20
    column.loc[(column["age"] >= 30) & (column["age"] <= 39), "age_category"] = 30
    column.loc[(column["age"] >= 40) & (column["age"] <= 49), "age_category"] = 40
    column.loc[(column["age"] >= 50) & (column["age"] <= 59), "age_category"] = 50
    column.loc[column["age"] >= 60, "age_category"] = 60
 
train['age_category'] = train['age_category'].astype(np.int64)
train.dtypes


# In[ ]:


import seaborn as sns
sns.set(style="white")
fig, ax = plt.subplots(figsize=(12,8))
sns.countplot(x="age_category", data=train, palette="Set2")
ax.set_title("Different Age Categories", fontsize=20)
ax.set_xlabel("Age Categories")
plt.show()


# In[ ]:


# There was a positive ratio of Suscribing Term Deposits  of people in their 20s (or younger) and 60s (or older)
sns.set(style="white")
fig, ax = plt.subplots(figsize=(15, 4))
colors = ["#F08080", "#00FA9A"]
labels = ['No Deposit', 'Deposit']
sns.countplot(y="age_category", hue='y', data=train, palette=colors).set_title('Employee Salary Turnover Distribution')
ax.set_ylabel("Age Category")
legend_name = plt.legend()
legend_name.get_texts()[0].set_text('Refused a T.D Suscription')
legend_name.get_texts()[1].set_text('Accepted a T.D Suscription')


# In[ ]:


sns.set(style="white")
fig, ax = plt.subplots(figsize=(14,8))
sns.countplot(x="job", data=train, palette="Set1")
ax.set_title("Occupations of Potential Clients", fontsize=20)
ax.set_xlabel("Types of Jobs")
plt.show()


# In[ ]:


sns.factorplot('marital','age',hue='y',data=train )


# In[ ]:


sns.factorplot('marital','duration',hue='y',data=train )


# In[ ]:


sns.factorplot('marital','previous',hue='y',data=train )


# In[ ]:


sns.factorplot('housing','age',hue='y',data=train )


# In[ ]:


sns.factorplot('housing','cons_conf_idx',hue='y',data=train )


# In[ ]:


sns.factorplot('housing','emp_var_rate',hue='y',data=train )


# In[ ]:


train.groupby('job').y.mean().plot(kind='bar')


# ### Above plot shows that the students and retired man are most likely to subscribe

# In[ ]:


train.groupby('day_of_week').y.mean().plot(kind='bar')


# In[ ]:


train.groupby('month').y.mean().plot(kind='bar')


# In[ ]:


train.groupby('education').y.mean().plot(kind='bar')


# ### this shows that the illiterate person is more likely to subscribe

# #  prediction through different algorithms

# In[ ]:


encoding_list = ['job', 'marital', 'education', 'default', 'housing', 'loan',
       'contact', 'month', 'day_of_week','poutcome']
train[encoding_list] = train[encoding_list].apply(LabelEncoder().fit_transform)


# ### Splitting data set to train and test in 7:3 ratio 

# In[ ]:


Y = train['y']
X = train.drop('y', axis=1)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,Y ,test_size=0.3, random_state=2)
print (X_train.shape)
print (X_test.shape)
print (y_train.shape)
print (y_test.shape)


# # Different Classification Models:

# In[ ]:



from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


# In[ ]:


# Logistic Regression
log_reg = LogisticRegression()
log_scores = cross_val_score(log_reg, X_train, y_train, cv=3)
log_reg_mean = log_scores.mean()
print(log_reg_mean)


# In[ ]:


# KNearestNeighbors
knn_clf = KNeighborsClassifier()
knn_scores = cross_val_score(knn_clf, X_train, y_train, cv=3)
knn_mean = knn_scores.mean()
print(knn_mean)


# In[ ]:


# Decision Tree
tree_clf = tree.DecisionTreeClassifier()
tree_scores = cross_val_score(tree_clf, X_train, y_train, cv=3)
tree_mean = tree_scores.mean()
print(tree_mean)


# In[ ]:


# Gradient Boosting Classifier
grad_clf = GradientBoostingClassifier()
grad_scores = cross_val_score(grad_clf, X_train, y_train, cv=3)
grad_mean = grad_scores.mean()
print(grad_mean)


# In[ ]:


# Random Forest Classifier
rand_clf = RandomForestClassifier(n_estimators=18)
rand_scores = cross_val_score(rand_clf, X_train, y_train, cv=3)
rand_mean = rand_scores.mean()
print(rand_mean)


# In[ ]:


# NeuralNet Classifier
neural_clf = MLPClassifier(alpha=1)
neural_scores = cross_val_score(neural_clf, X_train, y_train, cv=3)
neural_mean = neural_scores.mean()
print(neural_mean)


# In[ ]:


# Naives Bayes
nav_clf = GaussianNB()
nav_scores = cross_val_score(nav_clf, X_train, y_train, cv=3)
nav_mean = neural_scores.mean()
print(nav_mean)


# ## Clearly We can see that Gradient Boosting gives the best score
