#!/usr/bin/env python
# coding: utf-8

# # **Introduction**

# ![](https://miro.medium.com/max/600/1*IfOF_QNqAwFJY0tbU7UlZQ.jpeg)
# <br>**Hello, everyone! That's my EDA and prediction of a pulsar star in the universe.** Here you can find some exploration using Seaborn and applying our data on several algorithms. If you can help me with my code and say what I'm doing wrong feel free to do it. Enjoy the notebook :)

# # **Preparing the environment and uploading data**

# **Import packages**

# In[ ]:


# data analysis and wrangling
import pandas as pd
import numpy as np

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style='ticks', rc={'figure.figsize':(15, 10)})

# machine learning
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# **Acquire data**

# In[ ]:


data = pd.read_csv("../input/predicting-a-pulsar-star/pulsar_stars.csv")


# # **Exploratory Data Analysis (EDA)**

# **Take a first look**

# In[ ]:


data.head()


# **Get detailed statistics about the data.** I created a function that shows us various stats about our data

# In[ ]:


def detailed_analysis(df):
  obs = df.shape[0]
  types = df.dtypes
  counts = df.apply(lambda x: x.count())
  nulls = df.apply(lambda x: x.isnull().sum())
  distincts = df.apply(lambda x: x.unique().shape[0])
  missing_ratio = (df.isnull().sum() / obs) * 100
  uniques = df.apply(lambda x: [x.unique()])
  skewness = df.skew()
  kurtosis = df.kurt()
  print('Data shape:', df.shape)

  cols = ['types', 'counts', 'nulls', 'distincts', 'missing ratio', 'uniques', 'skewness', 'kurtosis']
  details = pd.concat([types, counts, nulls, distincts, missing_ratio, uniques, skewness, kurtosis], axis=1)

  details.columns = cols 
  dtypes = details.types.value_counts()
  print('________________________\nData types:\n', dtypes)
  print('________________________')

  return details


# In[ ]:


details = detailed_analysis(data)
details


# In[ ]:


data.describe()


# **It seems we haven't got null values, so we can start to work with the data without any troubles. However we have to rename our columns**

# In[ ]:


data.columns.values


# In[ ]:


data.rename(columns={ ' Mean of the integrated profile':'mean_profile', ' Standard deviation of the integrated profile':'std_profile', 
                     ' Excess kurtosis of the integrated profile':'kurt_profile', ' Skewness of the integrated profile':'skew_profile',
                     ' Mean of the DM-SNR curve':'mean_dmsnr_curve', ' Standard deviation of the DM-SNR curve':'std_dmsnr_curve',
                     ' Excess kurtosis of the DM-SNR curve':'kurt_dmsnr_curve', ' Skewness of the DM-SNR curve': 'skew_dmsnr_curve'}, inplace=True)

data.head()


# **Target class distribution**

# In[ ]:


values = data.target_class.value_counts()
indexes = values.index

sns.barplot(indexes, values)

plt.xlabel('target_class')
plt.ylabel('Number of values')


# **Comparison of all atributes for target classes**

# In[ ]:


sns.pairplot(data=data, vars=data.columns.values[:-1], hue='target_class')


# **Relationship between mean_profile and std_profile by target class**

# In[ ]:


sns.jointplot(x='mean_profile', y='std_profile', data=data, kind='kde', height=12)


# **Distplots of numeric values**

# In[ ]:


fig = plt.figure(figsize=(25, 25))

fig1 = fig.add_subplot(421)
sns.distplot(data.mean_profile, color='r')

fig2 = fig.add_subplot(422)
sns.distplot(data.std_profile, color='g')

fig3 = fig.add_subplot(423)
sns.distplot(data.kurt_profile, color='b')

fig4 = fig.add_subplot(424)
sns.distplot(data.skew_profile, color='y')

fig5 = fig.add_subplot(425)
sns.distplot(data.mean_dmsnr_curve, color='purple')

fig6 = fig.add_subplot(426)
sns.distplot(data.std_dmsnr_curve, color='grey')

fig6 = fig.add_subplot(427)
sns.distplot(data.kurt_dmsnr_curve, color='black')

fig6 = fig.add_subplot(428)
sns.distplot(data.skew_dmsnr_curve, color='orange')


# **Correlation heatmap**

# In[ ]:


fig = plt.figure(figsize=(12, 10))

correlation = data.corr()
sns.heatmap(correlation, annot=True, cmap='GnBu_r', center=1)


# **Comparison of mean values between stars and not stars**

# In[ ]:


fig = plt.figure(figsize=(20, 20))

values = data.groupby('target_class')[data.columns.values[:-1]].mean()
for item in range(len(values.iloc[0])):
  plot = fig.add_subplot(4, 2, item + 1)
  sns.barplot(x=data.target_class.value_counts().index, y=values[values.columns.values[item]], palette='Greens')


# # **Model**

# **Split our data**

# In[ ]:


X = data.drop('target_class', axis=1)
y = data.target_class

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print('Input train shape', X_train.shape)
print('Output train shape', y_train.shape)
print('Input test shape', X_test.shape)
print('Output test shape', y_test.shape)


# **Decision Tree Classifier**

# In[ ]:


model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print('Accuracy:', accuracy * 100)


# In[ ]:


sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)


# **Gradient Boosting Classifier**

# In[ ]:


gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)

y_pred = gbc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print('Accuracy:', accuracy * 100)


# In[ ]:


sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)


# **Random Forest Classifier**

# In[ ]:


model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print('Accuracy:', accuracy * 100)


# In[ ]:


sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)


# **Gaussian Naive Bayes**

# In[ ]:


model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print('Accuracy:', accuracy * 100)


# In[ ]:


sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)


# **XGBoost Classifier**

# In[ ]:


xgb = XGBClassifier()
xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print('Accuracy:', accuracy * 100)


# In[ ]:


sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)


# **We got nearly perfect accuracy. Thanks for reading my notebook! If you like it, please upvote :)**
