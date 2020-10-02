#!/usr/bin/env python
# coding: utf-8

# # Dealing with Imbalanced Dataset
# **Our goal**
# 1. To create undersample and oversampled data
# 2. To remove detect and remove the outliers
# 3. Train on multiple algorithms for the classification 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# Reading the csv file through pandas and view the dataset.
# 
# **Analysis**
# 1. To protect the identity of the user and to make the anonymoused data, the team has already performed the PCA and given the component variable which need to be used for the further analysis
# 2. After looking at the data, column "Time" and "Amount" has not been scaled. Hence we will go through certain scaling algorithm which can be used to scale the data as similar as other variable.
# 3. Class has two values as "0" --> Non Fraud Transactions and "1" --> Fraud Transactions.
# 4. There are no null values present in the dataset

# In[ ]:


original_df = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
original_df.head()


# Now we will count the total values present in each class and plotted a count plot.
# 
# **Analysis**
# 
# 1. It can be observed that, Non Fraud transactions in too much quantity compared to fraud transaction.
# 2. If we try to apply machine learning to train a classification algorithm, then it will be biased towards non fraud transactions.
# 3. Hence we will explore different oversampling and undersampling techniques.

# In[ ]:


print("Count of classes available")
print(pd.Index(original_df['Class']).value_counts())
sns.set(style="darkgrid")
ax = sns.countplot(x="Class", data=original_df)


# We will have the distribution plot for the "Time" and "Amount" to see, how distributed are they in the gaussian curve. We can see that, the graphs are highly skewed. Hence it becomes very important to scale the data.

# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(18,4))

amount_val = original_df['Amount'].values
time_val = original_df['Time'].values

sns.distplot(amount_val, ax=ax[0], color='b')
ax[0].set_title('Distribution of Transaction Amount', fontsize=14)
ax[0].set_xlim([min(amount_val), max(amount_val)])

sns.distplot(time_val, ax=ax[1], color='g')
ax[1].set_title('Distribution of Transaction Time', fontsize=14)
ax[1].set_xlim([min(time_val), max(time_val)])


plt.show()


# **Scaling**
# 
# There are various types of scaling techniques available under sklearn package. It will be interesting to check which scaling method could scale our variable best, which will help to center our data for our gaussian curve. We tested with various scaling techniques like StandardScaler, MinMaxScaler, and RobustScaler and have decided to go with RobustScaler. Robust Scaler is less prone to outlier. Visit below page for more details
# 
# [Comparision between other scaling method.](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html)

# In[ ]:


from sklearn.preprocessing import RobustScaler
rob_scaler = RobustScaler()
original_df['scaled_amount'] = rob_scaler.fit_transform(original_df['Amount'].values.reshape(-1,1))
original_df['scaled_time'] = rob_scaler.fit_transform(original_df['Time'].values.reshape(-1,1))
original_df.drop(['Time','Amount'], axis=1, inplace=True)


# In[ ]:


# Shuffling our data
original_df.sample(frac=1).head()


# In order to view our dataset we need to reduce the dimensionality to 2. We have 3 options.
# 
# 1. Princple Component Analysis (PCA)
# 2. Singular Value Decomposition (SVD)
# 3. t-Distributed Stochastic Neighbor Embedding (t-SNE)
# 
# We observed that, tSNE shows accurate which can depicts clearly separated Fraud and Non Fraud cases.

# In[ ]:


from sklearn.manifold import TSNE
def plot_graph_tsne(X,y):
    pca_2d = TSNE(n_components=2, random_state=42).fit_transform(X.values)
    colors = ['#ef8a62' if v == 0 else '#f7f7f7' if v == 1 else '#67a9cf' for v in y]
    kwarg_params = {'linewidth': 1, 'edgecolor': 'black'}
    fig = plt.Figure(figsize=(12,6))
    plt.scatter(pca_2d[:, 0],pca_2d[:, 1], c=colors, **kwarg_params, label="Fraud")
    plt.legend()
    sns.despine()


# First we will perform the undersampling and further check, how our model preforms on undersampled data. There is great library which can be explored further for different [undersampling](https://imbalanced-learn.readthedocs.io/en/stable/api.html#module-imblearn.under_sampling) techniques.
# 
# Our aim to create equally sampled data from each class, in order to remove bias from the dataset. As we understood before, we have only 492 classes for non fraud samples, then we must take only 492 fraud samples. Hence we will go with RandomUnderSampler for imblearn library.

# In[ ]:


from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
X = original_df.drop('Class', axis=1)
y = original_df['Class']
sampler = RandomUnderSampler(sampling_strategy='auto')
X_rs, y_rs = sampler.fit_sample(X, y)
print('Resampled dataset shape %s' % Counter(y_rs))
plot_graph_tsne(X_rs, y_rs)


# In[ ]:


df_under = pd.concat([X_rs, y_rs], axis=1)
df_under.head()


# Lets plot the correlation matrix for different features. We plotted correlation matrix for undersampled data and for original data. We had following observation:
# 
# 1.  Negative Correlation for anomaly check - v3, v9, v10, v12, v14, v16, v17 are highly negatively correlated. 
# 2.  Positive Correlation for anomaly chcek - v2, v4, v11, v19 are highly positively correlated. 
# 
# We will go ahead and create box for above feature to look for available outliers.

# In[ ]:


f, (ax1, ax2) = plt.subplots(2, 1, figsize=(50,30))      
sns.heatmap(df_under.corr(), annot=True, cmap = 'coolwarm', ax=ax1)
ax1.set_title("Imbalanced Correlation Matrix of UnderSampled Data", fontsize=14)

       
sns.heatmap(original_df.corr(), cmap = 'coolwarm', ax=ax2)
ax2.set_title("Imbalanced Correlation Matrix of Original Data", fontsize=14)


# Once plotted the boxplot for the above negatively and positively feature, we observed great deal of available outliers (it is also called as one of the anomaly detected techniques). 

# In[ ]:


f, axes = plt.subplots(ncols=7, figsize=(60,8))
sns.boxplot(x="Class", y="V3", data=df_under,  ax=axes[0])
axes[0].set_title('V3 vs Class Negative Correlation')

sns.boxplot(x="Class", y="V9", data=df_under,  ax=axes[1])
axes[1].set_title('V9 vs Class Negative Correlation')

sns.boxplot(x="Class", y="V10", data=df_under,  ax=axes[2])
axes[2].set_title('V10 vs Class Negative Correlation')

sns.boxplot(x="Class", y="V12", data=df_under,  ax=axes[3])
axes[3].set_title('V12 vs Class Negative Correlation')

sns.boxplot(x="Class", y="V14", data=df_under,  ax=axes[4])
axes[4].set_title('V14 vs Class Negative Correlation')

sns.boxplot(x="Class", y="V16", data=df_under,  ax=axes[5])
axes[5].set_title('V16 vs Class Negative Correlation')

sns.boxplot(x="Class", y="V17", data=df_under,  ax=axes[6])
axes[6].set_title('V17 vs Class Negative Correlation')

plt.show()


# In[ ]:


f, axes = plt.subplots(ncols=4, figsize=(20,4))
sns.boxplot(x="Class", y="V2", data=df_under,  ax=axes[0])
axes[0].set_title('V2 vs Class Positive Correlation')

sns.boxplot(x="Class", y="V4", data=df_under,  ax=axes[1])
axes[1].set_title('V4 vs Class Positive Correlation')

sns.boxplot(x="Class", y="V11", data=df_under,  ax=axes[2])
axes[2].set_title('V11 vs Class Positive Correlation')

sns.boxplot(x="Class", y="V19", data=df_under,  ax=axes[3])
axes[3].set_title('V19 vs Class Positive Correlation')

plt.show()


# We also also plotted the distribution graph for the features to look, how deviated are they from the normal distribution

# In[ ]:


from scipy.stats import norm
fig, ax = plt.subplots(1, 4, figsize=(50,8))

V2 = df_under['V2'].values
V4 = df_under['V4'].values
V11 = df_under['V11'].values
V19 = df_under['V19'].values

sns.distplot(V2, ax=ax[0], fit=norm, color='b')
ax[0].set_title('Distribution of V2', fontsize=8)
ax[0].set_xlim([min(V2), max(V2)])

sns.distplot(V4, ax=ax[1], fit=norm, color='g')
ax[1].set_title('Distribution of V4', fontsize=8)
ax[1].set_xlim([min(V4), max(V4)])

sns.distplot(V11, ax=ax[2], fit=norm, color='g')
ax[2].set_title('Distribution of V11', fontsize=8)
ax[2].set_xlim([min(V11), max(V11)])

sns.distplot(V19, ax=ax[3], fit=norm, color='g')
ax[3].set_title('Distribution of V19', fontsize=8)
ax[3].set_xlim([min(V19), max(V19)])

plt.show()


# **Outlier removal**
# 
# To remove sufficient amount outlier we have tried to method Z-score method and IQR method.
# 
# 1. Z-Score method : The intuition behind Z-score is to describe any data point by finding their relationship with the Standard Deviation and Mean of the group of data points. Z-score is finding the distribution of data where mean is 0 and standard deviation is 1 i.e. normal distribution. In most of the cases a threshold of 3 or -3 is used i.e if the Z-score value is greater than or less than 3 or -3 respectively, 
# 
# 2. IQR Method : IQR is somewhat similar to Z-score in terms of finding the distribution of data and then keeping some threshold to identify the outlier. Then we multiply with the threshold (1.5) and try to include only those values.
# 
# We observed that, IQR method tends to remove too many outlier and hence we will proceed with Z-score method.

# In[ ]:


from scipy import stats
import numpy as np
z = np.abs(stats.zscore(df_under[["V3","V9","V10","V12","V14","V2","V4","V11","V19"]]))
df_under_filtered = df_under[(z < 3).all(axis=1)]
print("Before outlier removal",df_under.shape)
print("Remaining after outlier removal",df_under_filtered.shape)
sns.set(style="darkgrid")
ax = sns.countplot(x="Class", data=df_under_filtered)


# In[ ]:


Q1 = df_under[["V3","V9","V10","V12","V14","V2","V4","V11","V19"]].quantile(0.25)
Q3 = df_under[["V3","V9","V10","V12","V14","V2","V4","V11","V19"]].quantile(0.75)
IQR = Q3 - Q1
df_under_out = df_under[~((df_under < (Q1 - 1.5 * IQR)) |(df_under > (Q3 + 1.5 * IQR))).any(axis=1)]
print("Remaining after outlier removal",df_under_out.shape)
sns.set(style="darkgrid")
ax = sns.countplot(x="Class", data=df_under_out)


# After removing those outlier, we created boxplot for those feature. We observed that, outliers has been reduced upto great extent.

# In[ ]:


f, axes = plt.subplots(ncols=9, figsize=(60,8))
sns.boxplot(x="Class", y="V3", data=df_under_filtered,  ax=axes[0])
axes[0].set_title('V3 Reduced Outlier')
axes[0].annotate('Fewer extreme \n outliers', xy=(0.98, -17.5), xytext=(0, -12),
            arrowprops=dict(facecolor='black'),
            fontsize=14)

sns.boxplot(x="Class", y="V9", data=df_under_filtered,  ax=axes[1])
axes[1].set_title('V9 Reduced Outlier')

sns.boxplot(x="Class", y="V10", data=df_under_filtered,  ax=axes[2])
axes[2].set_title('V10 Reduced Outlier')

sns.boxplot(x="Class", y="V12", data=df_under_filtered,  ax=axes[3])
axes[3].set_title('V12 Reduced Outlier')

sns.boxplot(x="Class", y="V14", data=df_under_filtered,  ax=axes[4])
axes[4].set_title('V14 Reduced Outlier')

sns.boxplot(x="Class", y="V2", data=df_under_filtered,  ax=axes[5])
axes[5].set_title('V2 Reduced Outlier')

sns.boxplot(x="Class", y="V4", data=df_under_filtered,  ax=axes[6])
axes[6].set_title('V4 Reduced Outlier')

sns.boxplot(x="Class", y="V11", data=df_under_filtered,  ax=axes[7])
axes[7].set_title('V11 Reduced Outlier')

sns.boxplot(x="Class", y="V19", data=df_under_filtered,  ax=axes[8])
axes[8].set_title('V19 Reduced Outlier')


plt.show()


# In[ ]:


X = df_under_filtered.drop('Class', axis=1)
y = df_under_filtered['Class']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=4)


# In[ ]:


X_test.shape


# We train different model with the above data and cross validated it. We observed XGBoost Classifier did a great job. It could be because of the of random forest based boosting technique which it follow within.

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
import xgboost
from sklearn import svm, tree
from sklearn import metrics

classifiers = []
nb_model = GaussianNB()
classifiers.append(("Gausian Naive Bayes Classifier",nb_model))
lr_model= LogisticRegression()
classifiers.append(("Logistic Regression Classifier",lr_model))
# sv_model = svm.SVC()
# classifiers.append(sv_model)
dt_model = tree.DecisionTreeClassifier()
classifiers.append(("Decision Tree Classifier",dt_model))
rf_model = RandomForestClassifier()
classifiers.append(("Random Forest Classifier",rf_model))
xgb_model = xgboost.XGBClassifier()
classifiers.append(("XG Boost Classifier",xgb_model))
lda_model = LinearDiscriminantAnalysis()
classifiers.append(("Linear Discriminant Analysis", lda_model))
gp_model =  GaussianProcessClassifier()
classifiers.append(("Gaussian Process Classifier", gp_model))
ab_model =  AdaBoostClassifier()
classifiers.append(("AdaBoost Classifier", ab_model))

cv_scores = []
names = []
for name, clf in classifiers:
    print(name)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities  
    y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.
    print("Model Score : ",clf.score(X_test, y_pred))
    print("Number of mislabeled points from %d points : %d"% (X_test.shape[0],(y_test!= y_pred).sum()))
    scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
    cv_scores.append(scores)
    names.append(name)
    print("Cross validation scores : ",scores.mean())
    confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
    print("Confusion Matrix \n",confusion_matrix)
    classification_report = metrics.classification_report(y_test,y_pred)
    print("Classification Report \n",classification_report)


# In[ ]:




