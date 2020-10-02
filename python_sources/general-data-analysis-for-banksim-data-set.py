#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
from matplotlib import pyplot
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/bs140513_032310.csv")


# In[ ]:


df.head(5)


# In[ ]:


df.isnull().values.any()


# In[ ]:


df.describe(include='all')


# In[ ]:


trans_by_cust = df[df['customer'] == "'C1978250683'"]
fraud_by_cust = df[(df['customer'] == "'C1978250683'") & (df['fraud'] == 1)]
no_fraud_by_cust = df[(df['customer'] == "'C1978250683'") & (df['fraud'] == 0)]
trans_by_cust.head()


# In[ ]:


num_fraud_trans, num_safe_trans, total_trans = len(fraud_by_cust), len(no_fraud_by_cust), len(trans_by_cust)
percent_frauds = (num_fraud_trans/total_trans * 100)
percent_safe = (100 - percent_frauds)
print("Percentage of frauds by customer C1978250683: ", round(percent_frauds, 2))
print("Percentage of  no frauds by customer C1978250683: ", round(percent_safe, 2))


# 41.51 % of the total transactions made by this customer are fraudulent. This is something to look out for. Lets see the corresponding amounts.

# In[ ]:


trans_by_cust['amount'].describe()


# In[ ]:


fraud_by_cust['amount'].hist(edgecolor='black', color='r', linewidth=1.2)


# In[ ]:


no_fraud_by_cust['amount'].hist(edgecolor='black', color='g', linewidth=1.2)


# In[ ]:


ax = sns.boxplot(x="fraud", y="amount", data=trans_by_cust)


# In[ ]:


df['gender'].describe()


# In[ ]:


ax = sns.boxplot(x="gender", y="amount", hue="fraud", data=df)


# In[ ]:


sns.catplot(x="gender", y="amount", hue="fraud", data=df);


# In[ ]:


df.dtypes


# In[ ]:


for col_name in df.columns:
    if(df[col_name].dtype == 'object'):
        df[col_name]= df[col_name].astype('category')
        df[col_name] = df[col_name].cat.codes


# In[ ]:


len(df.columns)


# In[ ]:


df.dtypes


# In[ ]:


num_obs = len(df)


# In[ ]:


df.describe(include='all')


# In[ ]:


class_counts = df.groupby('fraud').size()
print("Total Number of observations: ", num_obs)
print(class_counts)


# In[ ]:


df.hist(edgecolor='black')


# In[ ]:


df['age'].hist(bins=8, edgecolor='black', linewidth=1.2)


# In[ ]:


df['gender'].hist(edgecolor='black', linewidth=1.2)


# In[ ]:


sns.pairplot(df, hue = 'fraud', diag_kind = 'kde',
             plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'},
             size = 4)


# **Feature Selection**

# **Pearson Correlation** 

# Pearson Correlation is given by: $\rho = \frac{\text{cov}(X,Y)}{\sigma_x \sigma_y}$. The values are between -1 and +1. A value of -1 means negative correlation and +1 means positive correlation. 0 means no correlation. Dividing covariance by product of standard deviations.

# In[ ]:


corr = df.corr(method='pearson')
print(corr)


# In[ ]:


corr = df.corr()
corr.style.background_gradient().set_precision(2)


# In[ ]:


skew = df.skew()
print(skew)


# #  $\chi^2$ Chi Squared Test
# This test is generally used for categorical values. Not on values like means, percentages etc. 

# In[ ]:


# Chi squared test
# Feature selection
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
array = df.values
X = array[:,0:9]
Y = array[:,9]
# Extracting Features
test = SelectKBest(score_func=chi2, k=5)
fit = test.fit(X, Y)
# Summarize the scores
set_printoptions(precision=2)
print(fit.scores_)
features = fit.transform(X)
# Summarize the features selected
print(features[0:5,:])


# Just attach the corresponding attributes with the values. Higher the value, better is the variable to be accomodated in the model.
# 
# 

# # Recursive Feature Elimination

# In[ ]:


# Recursive Feature Selection (RFE)
# feature extraction
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
rfc = RandomForestClassifier()
# create the RFE model and select 4 attributes
rfe = RFE(rfc, 4, step=1)
rfe_fit = rfe.fit(X, Y)
 
# summarize the selection of the attributes
print(rfe_fit.n_features_)
print(rfe_fit.support_)
print(rfe_fit.ranking_)


# #  Decision tree for feature selection
# Our model is a mix of categorical and numeric attributes. Its a good idea, to use decision tree to get importance of the features. Just like Recursive Feature Elimination. Another way to select important features

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
mdl = DecisionTreeClassifier()
mdl.fit(X, Y)


# In[ ]:


cols = df.columns.tolist()
df_fimp = pd.Series(mdl.feature_importances_ , index = cols[0:len(cols)-1]).sort_values(ascending=False)
df_fimp


# High score corresponds to a better feature. It has been sorted in descending order. This means amount is the best feature, then second best is merchant and so on. But our intuition, how in line it is with our business use case and insights that we consolidated through data exploration should be used to select the final features and not entirely rely on feature selection techniques.

# # General

# In[ ]:


is_fraud, not_fraud = df[df['fraud'] == 1], df[df['fraud'] == 0]
frauds = len(is_fraud)
no_frauds = len(not_fraud)
total_obs = len(df)


# In[ ]:


# For reference: https://www.kaggle.com/funkyong13/fraud-detect-visualization-to-classification
# Why re-invent the wheel?
num_bins = 10
tran_amount = df['amount']
n, bins, patches = pyplot.hist(tran_amount, num_bins, normed = False, stacked = True, facecolor= '#f26a6a', alpha=0.5)
pyplot.close()
n_fraud = np.zeros(num_bins)
for i in range(num_bins):
    for j in range(frauds):
        if bins[i] < is_fraud['amount'].iloc[j] <= bins[i+1]:
            n_fraud[i] += 1
range_amount = []
for i in range(num_bins):
    lower_lim, higher_lim = str(int(bins[i])), str(int(bins[i+1]))
    range_amount.append("$ " + lower_lim + " ~ " + higher_lim )


# In[ ]:


df_hist = pd.DataFrame(index = range_amount)
df_hist.index.name = 'Transaction Amount[$]'
df_hist['# Total'] = n
df_hist['# Frauds'] = n_fraud
df_hist['# Safe'] = df_hist['# Total'] - df_hist['# Frauds']
df_hist['% Frauds'] = (df_hist['# Frauds'] / df_hist['# Total'] * 100).round(2)
df_hist['% Safe'] = (df_hist['# Safe'] / df_hist['# Total'] * 100).round(2)
df_hist


# In[ ]:


print("Percentage of frauds: ", frauds/total_obs*100)
print("Percentage of no frauds: ", no_frauds/total_obs*100)


# The results above show us that the dataset is highly unbalanced.
