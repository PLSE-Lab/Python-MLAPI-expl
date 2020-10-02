#!/usr/bin/env python
# coding: utf-8

# My references:
# - https://www.kaggle.com/tboyle10/methods-for-dealing-with-imbalanced-data
# - https://www.kaggle.com/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets/data

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score, precision_score, roc_curve


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# # Data Exploration

# There are 30 features and 1 target variable in our dataset. Out of the 30 features, there are time, amount and 28 principal components.

# In[ ]:


# Class 1 for fraudulent transactions
df = pd.read_csv('../input/creditcardfraud/creditcard.csv')
df.describe()


# ## Percentage of fraud transactions
# The percentage of fraud transactions is 0.17% which makes our dataset hugely imbalanced.

# In[ ]:


nonfraud, fraud = df.Class.value_counts()
print('Percentage of fraud transactions', fraud / (nonfraud + fraud))


# There are more two order of magnitude  of nonfraud transactions compared with fraud ones. Notice the y scale is log scale and it is better than the linear scale in this case since you get a order of magnitude comparison.

# In[ ]:



sns.countplot('Class', data = df)
plt.gca().set_yscale('log')
df['Class'].value_counts()


# ## Distribution of transaction amount
# 
# Most fraud transactions are less than 1000 dollars. It seems that the identity thieves know large amount of transaction will likely to be declined by the banks.

# In[ ]:


fig, ax = plt.subplots(1, 2, figsize = (18, 4))

ax[0].hist(df[df.Class == 0]['Amount'], bins = 100, density = True, log = True, label = 'nonfraud')
ax[0].set_title('nonfraud')
ax[1].hist(df[df.Class == 1]['Amount'], bins = 100, density = True, log = True, label = 'fraud')
ax[1].set_title('fraud')
plt.show()


# ## Distribution of transaction times
# 
# The dsitribution of nonfraud transaction time agrees with our intuition: people generally do not make a lot of transactions at midnight. Though we do not know what time 0 corresponds to, it is reasonable to guess the two minima are midnights.\
# 
# As for fraud transactions, not much pattern.

# In[ ]:


fig, ax = plt.subplots(1, 2, figsize = (18, 4))

ax[0].hist(df[df.Class == 0]['Time'], bins = 100, density = True, label = 'nonfraud')
ax[0].set_title('nonfraud')
ax[1].hist(df[df.Class == 1]['Time'], bins = 100, density = True, label = 'fraud')
ax[1].set_title('fraud')
plt.show()


# ## Correlation analysis
# 
# The principal components are barely correlated as is expected.

# In[ ]:


plt.figure(figsize = (12, 8))
df_corr = df.corr()
sns.heatmap(df_corr, cmap = 'coolwarm_r')
plt.show()


# One point I do not understand in the boxplots below is why there seems to be so many outliers? One possible answer is that the features are not normally distributed. Another question is should you remove the outliers? I do not have any good reason to remove the outliers. The outliers are the actual data points and are not mistakes by any means. 

# In[ ]:


# Correlation with class label sorted from largest to smallest
index_sorted = df_corr.sort_values('Class', ascending = False).index

fig, ax = plt.subplots(2, 5, figsize = (20, 8))
plt.subplots_adjust(hspace = 0.3, wspace = 0.3)

for i in range(0, 5):
    sns.boxplot(x = 'Class', y = index_sorted[i + 1], data = df, ax = ax[0][i])
    ax[0][i].set_title('%s vs %s (corr = %.3f)' % (index_sorted[i + 1], 'Class', df_corr['Class'][index_sorted[i + 1]]))

for i in range(0, 5):
    var = index_sorted[- (i + 1)]
    sns.boxplot(x = 'Class', y = var, data = df, ax = ax[1][i])
    ax[1][i].set_title('%s vs %s (corr = %.3f)' % (var, 'Class', df_corr['Class'][var]))


# If we look at the distribution of V17, which is the most negatively correlated feature with class label, the tails of the nofraud data is indeed large. And the distribution is slightly assymetric. 

# In[ ]:


sns.distplot(df[df.Class == 1]['V17'], bins = 100, norm_hist = True, label = 'fraud', color = 'red')
sns.distplot(df[df.Class == 0]['V17'], bins = 100, norm_hist = True, label = 'nonfraud', color = 'blue')
plt.legend()
plt.show()


# In[ ]:


# 


# In[ ]:


y = df.Class
X = df.drop('Class', axis = 1) # axis = 1 means to drop a column
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 27) # random_state is the seed for rand generator


# ## Training using dummy classifier
# The dummy classifier will look at what the majority class is in the training data and classify all data as that class. Though dumb, it gives pretty good accuracy on a highly imbalanced data set like this where data of one class outnumbers the other by orders of magnitude. A lesson to learn is accuracy is not always the best measure of model performance.

# In[ ]:


# Try dummy classification for this highly imbalanced dataset
dummy = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)


# In[ ]:


# The prediction of dummy should be all the same
np.unique(dummy_pred)


# In[ ]:


# accuracy = (true positive + true negative) / (total predictions made)
accuracy_score(y_test, dummy_pred)
# Seems good but it is due to the highly skewed the dataset


# # Training using logistic regression
# Do I expect logistic regression to work? Not sure.
# 

# In[ ]:


# Try logistic regression
lr = LogisticRegression(solver = 'liblinear').fit(X_train, y_train) # What is liblinear solver?
lr_pred = lr.predict(X_test)
lr_test_score = lr.predict_proba(X_test)[:, 1]
lr_train_score = lr.predict_proba(X_train)[:, 1]
accuracy_score(y_test, lr_pred)


# In[ ]:


# Check logistic regression does not give the same answer as dummy classifier
np.unique(lr_pred, return_counts = True)


# In[ ]:


# sklearn gives error since there is no positive prediction in the dummy classifier
precision_score(y_test, lr_pred), precision_score(y_test, dummy_pred)


# In[ ]:


recall_score(y_test, lr_pred), recall_score(y_test, dummy_pred)


# In[ ]:


# Compute precision and recall directly from confusion matrix
conf_mat = confusion_matrix(y_test, lr_pred)
true_neg, false_pos, false_neg, true_pos = conf_mat.ravel()
# precision, recall
true_pos / (true_pos + false_pos), true_pos / (true_pos + false_neg)


# In[ ]:


# Take a look at ROC curve
def plot_roc(y_true, y_score, label = 'none'):
    fp, tp, _ = roc_curve(y_true, y_score)
    plt.plot(fp, tp, label = label + ' (%.4f)' % np.trapz(tp, fp))

#fpr, tpr, thresholds = roc_curve(y_test, lr_scores)
#plt.plot(fpr, tpr, label = 'logistic regression (%.4f)' % np.trapz(tpr, fpr))
plot_roc(y_test, lr_test_score, 'lr test')
plot_roc(y_train, lr_train_score, 'lr train')
plt.xlabel('false positive')
plt.ylabel('true postive')
plt.legend(loc = 'lower right')
plt.title('ROC')
#plt.xlim(0, 0.1)


# 

# In[ ]:




