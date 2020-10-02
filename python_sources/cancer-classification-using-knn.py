#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[ ]:


# numerical analysis
import numpy as np
# storing and processing in dataframes
import pandas as pd
# simple plotting
import matplotlib.pyplot as plt
# advanced plotting
import seaborn as sns

# splitting dataset into train and test
from sklearn.model_selection import train_test_split
# scaling features
from sklearn.preprocessing import StandardScaler
# selecting important features
from sklearn.feature_selection import RFECV
# k nearest neighbors model
from sklearn.neighbors import KNeighborsClassifier
# accuracy
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc


# ### Theme

# In[ ]:


# plot style
sns.set_style('whitegrid')
# color palettes
pal = ['#0e2433', '#ff007f']


# # Data

# In[ ]:


# read data
df = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')

# first few rows
df.head()


# ### Data properties

# In[ ]:


# no. of rows and columns
df.shape


# In[ ]:


# columns names
df.columns


# In[ ]:


# random rows
# df.sample(5)


# In[ ]:


# descriptive statistics
# df.describe(include='all')


# In[ ]:


# consise summary of dataframe
# df.info()


# In[ ]:


# no. of na values in each columns
# df.isna().sum()


# # Exploring the data

# ### Class distribution

# In[ ]:


# no of values in each class
print(df['diagnosis'].value_counts())

# plot class distribution
sns.countplot(df['diagnosis'], palette=pal, alpha=0.8)
plt.show()


# * There is an inbalance in no. of observations in each class.
# * This will possibley lead to a biased model.
# * Idealy we want to have approximatly equal no. of observation all the classes.
#   
# * In this case we can bring down the no. of observation in the 'B' class to no. of obeservation in the 'M'

# ### Is the mean radius of malignant and benign cancers the same?

# In[ ]:


fig, ax = plt.subplots()
m = ax.hist(df[df["diagnosis"] == "M"]['radius_mean'], bins=20, range=(0, 30), 
            label = "Malignant", alpha=0.7, color='#232121')
b = ax.hist(df[df["diagnosis"] == "B"]['radius_mean'], bins=20, range=(0, 30), 
            label = "Benign", alpha=0.7, color='#df2378')
plt.xlabel("Radius")
plt.ylabel("Count")
plt.title("Mean Radius")
plt.legend()
plt.show()


# In[ ]:


print('Min radius of benign cancer :', df[df['diagnosis']=='B']['radius_mean'].min())
print('Max radius of benign cancer :', df[df['diagnosis']=='B']['radius_mean'].max())
print('Min radius of malignant cancer :', df[df['diagnosis']=='M']['radius_mean'].min())
print('Min radius of malignant cancer :', df[df['diagnosis']=='M']['radius_mean'].max())


# * It looks like mean radius of benign is on the smaller than malignant
# * we can see benign is of the range 5-17 and malignant is of the range 10-28
# * mean_radius looks like good feature that can help us to distiguish between benign and malignant cancers

# ### Correlation heatmap

# In[ ]:


# figure size
plt.figure(figsize=(25, 12))
# plot heatmap
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='RdGy')
# show figure
plt.show()


# ### Pairplot

# In[ ]:


# # figure size
# plt.figure(figsize=(6, 6))
# # plot pairplot
# sns.pairplot(df, hue="diagnosis", palette=pal)
# # show figure
# plt.plot()


# In[ ]:


sns.boxplot(data=df, x)


# In[ ]:


get_ipython().run_line_magic('pinfo', 'sns.boxplot')


# # Preprocessing

# In[ ]:


# Equalize class distribution


# In[ ]:


# Drop unwanted columns

df.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)
print(df.shape)


# In[ ]:


# encoding diagnosis data

df['diagnosis'] = df['diagnosis'].apply(lambda x: 1 if x=='M' else 0)


# In[ ]:


# scale features
# scaler = StandardScaler()
# df_trans = pd.DataFrame(scaler.fit_transform(X))


# In[ ]:


# pca


# # Visual Exploration

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# plt.figure(figsize=(18, 6))
# sns.swarmplot(x="features", y="value", hue="diagnosis", data=df_melt)
# plt.show()


# In[ ]:


# plt.scatter_matrix(df)


# # ML Model

# In[ ]:


# features and labels
X = df.drop(['diagnosis'], axis=1)
y = df['diagnosis']


# In[ ]:


# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


# model initialization
model = KNeighborsClassifier()

# model fitting
model.fit(X_train, y_train)

# predict using the model
pred = model.predict(X_test)

# model validation
print(accuracy_score(pred, y_test))
print(confusion_matrix(pred, y_test))
print(classification_report(pred, y_teset))


# In[ ]:


sns.heatmap(confusion_matrix(pred, y_test), annot=True, fmt="d")


# In[ ]:


# # random forest classifier
# model = RandomForestClassifier() 

# # recursive feature elimination with cross validation
# rfecv = RFECV(estimator=model, step=1, cv=5, scoring='accuracy')
# rfecv = rfecv.fit(X_train, y_train)

# # predict using the model
# pred = model.predict(X_test)

# print(rfecv.n_features_)
# print(X_train.columns[rfecv.support_])

# plt.figure()
# plt.xlabel("Number of features selected")
# plt.ylabel("Cross validation score of number of selected features")
# plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
# plt.show()

# # model validation
# print(accuracy_score(pred, y_test))
# print(confusion_matrix(pred, y_test))
# sns.heatmap(confusion_matrix(pred, y_test), annot=True, fmt="d")

# # roc-auc
# probs = model.predict_proba(X_test)
# preds = probs[:,1]
# fpr, tpr, threshold = roc_curve(y_test, preds)
# roc_auc = auc(fpr, tpr)

# # roc-auc plot
# plt.title('ROC')
# plt.plot(fpr, tpr, 'b--', label='AUC = %0.2f' % roc_auc)
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.legend(loc='lower right')
# plt.show()

