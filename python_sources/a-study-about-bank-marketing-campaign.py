#!/usr/bin/env python
# coding: utf-8

# # Bank Marketing Dataset
# 
# This dataset contains information about a Portuguese bank marketing campaign. It's goal is to predict if the client will subscribe to a term deposit, indicated by the variable 'y'.
# 
# More information about this database can be found at: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

# ## Imports

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from scipy import stats
from scipy.stats import norm
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
import os
print(os.listdir("../input/bank-marketing-dataset"))


# ## Opening database

# In[ ]:


df = pd.read_csv('../input/bank-marketing-dataset/bank.csv')
df = df.rename(columns={'deposit': 'y'})
df.head(10)


# As we can see, this dataset have a lot of categorical features.
# 
# To begin our study, we should have a look at the 'y' distribution.

# In[ ]:


df['y'].value_counts().plot.bar()
plt.ylabel('Count')
plt.show()


# The plot shows that there are a considerable unbalance between the classes, our model should consider that.
# 
# Now, let's check the categorical columns

# In[ ]:


categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month','poutcome']
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20,20))
for i, col in enumerate(categorical_cols):
    row_index = (i % 3)
    col_index = int(i / 3)
    fig.add_subplot(df[col].value_counts().plot.bar(ax=axes[row_index, col_index], title=col))


# In[ ]:


numerical_cols = ['balance', 'day','duration', 'campaign', 'pdays', 'previous']
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(20,20))
for i, col in enumerate(numerical_cols):
    row_index = (i % 3)
    col_index = i // 3
    fig.add_subplot(df[col].value_counts().plot.hist(ax=axes[row_index, col_index], title=col))


# ## Analyzing the influence of the age
# 
# To check how the age influence the 'y', we can calculate the success rate (number of 'yes' divided by number of 'no') for each age and plot it

# In[ ]:


data = df[['age', 'y']]
data['y'] = data['y'].apply(lambda y: 1 if y == 'yes' else 0)
data = data.groupby('age')['y'].mean() * 100
data.plot.bar(figsize=(15, 5), title='Success Rate x Age')
plt.ylabel('Success Rate (%)')
plt.show()


# With this plot, we can conclude that older people tend to have higher chance tp subscribe to a term deposit. Also, young people can also have a high chance to subscribe, which reduces as the person gets older and get stabilize at 30 years old, growing again only after the 60 years.

# ## Analyzing the influence of the job
# 
# With a similar analysis, we can plot the number of 'yes' per job

# In[ ]:


data = df[df['y'] == 'yes'].groupby(df['job'])
N = 12
data['job'].count().nlargest(N).plot.bar(figsize=(15, 5),
                                           title='Number of subscriptions per Job')
plt.grid(True)
plt.ylabel("Number of Subscriptions")
plt.xlabel("Job")
plt.show()


# As we can see, this plot shows that 'jobs' with a possible economical stability have a higher number of subscriptions, as 'management', 'technician', 'blue-collar' and 'admin'. 

# ## Analyzing the influence of the marital
# 
# We will apply the same logic as before

# In[ ]:


data = df[df['y'] == 'yes'].groupby(df['marital'])
N = 3
data['marital'].count().nlargest(N).plot.bar(figsize=(15, 5),
                                           title='Number of subscriptions per Marital')
plt.ylabel("Number of Subscriptions")
plt.show()
print('Proportion from the biggest number to the second one: {}'.format(
      data['marital'].count().nlargest(N)[0] / data['marital'].count().nlargest(N)[1]))


# Married people have a higher number of subscriptions (almost 70% bigger than single), followed by single and divorced (in last).

# ## Preprocessing the data
# 
# Now, we could analyze the correlation of the variables to have a better knowledge of how much each one is important to predict 'y'.
# 
# First, we need to transform the categorical column into numerical:

# In[ ]:


data = df.copy()
data['loan'] = data['loan'].apply(lambda x: 1 if x == 'yes' else 0)
data['housing'] = data['housing'].apply(lambda x: 1 if x == 'yes' else 0)
data['default'] = data['default'].apply(lambda x: 1 if x == 'yes' else 0)
data['y'] = data['y'].apply(lambda x: 1 if x == 'yes' else 0)
data = pd.get_dummies(data)


# Now, let's check the skewness of the numerical values

# In[ ]:


sns.distplot(df['duration'], fit=norm);
fig = plt.figure()
res = stats.probplot(df['duration'], plot=plt)


# In[ ]:


# The data is not in a normal distribution, so let's transform it
data['duration'] = np.log(data['duration'] + 1)
sns.distplot(data['duration'], fit=norm);
fig = plt.figure()
res = stats.probplot(data['duration'], plot=plt)


# In[ ]:


sns.distplot(data['age'], fit=norm)
fig = plt.figure()
res = stats.probplot(data['age'], plot=plt)


# In[ ]:


# The data is not in a normal distribution, so let's transform it
data['age'] = np.log(data['age'])
sns.distplot(data['age'], fit=norm)
fig = plt.figure()
res = stats.probplot(data['age'], plot=plt)


# In[ ]:


sns.distplot(data['balance'], fit=norm)
fig = plt.figure()
res = stats.probplot(data['balance'], plot=plt)


# Balance has a lot of zero values, so let's keep it for this study.
# 
# Finally, separating the 'y' column:

# In[ ]:


y_train = data['y']
data = data.drop(columns=['y'])


# ## Defining the model
# 
# For this study, we will test the acuraccy of some classifiers, like RandomForest, kNN, SVC and LogistcRegression.
# 
# Our final model will be the best between those

# In[ ]:


def model_scores(model):
    kf = KFold(n_folds, shuffle=True).get_n_splits(data.values)
    score = cross_val_score(model, data.values, y_train, scoring='accuracy', cv = kf, n_jobs=-1)
    return(score)


# In[ ]:


params = {
    'n_estimators':[50, 100, 150, 200, 250],
    'max_depth': [None, 2, 3, 4, 5]
}
forest = Pipeline([('scaler', RobustScaler()),
                   ('model', GridSearchCV(RandomForestClassifier(n_jobs=-1), params))])
params = {'kernel':('linear', 'rbf'), 'C':range(1, 10)}
svc = Pipeline([('scaler', RobustScaler()),
                ('model', GridSearchCV(SVC(class_weight='balanced', max_iter=1000, gamma='scale'), params))])
params = {'n_neighbors' : range(3, 10)}
knn = Pipeline([('scaler', RobustScaler()),
                ('model', GridSearchCV(KNeighborsClassifier(n_jobs=-1), params))])
logistic = Pipeline([('scaler', RobustScaler()), ('model', LogisticRegression(class_weight='balanced', n_jobs=-1))])

models = [forest, svc, knn, logistic]
n_folds = 5


# For each model, we will perform a grid search, for the best parameters, and then perform a 5-Fold Cross Validation, shuffling the data for each fold

# In[ ]:


scores = [model_scores(model) for model in tqdm(models)]


# In[ ]:


precision = [np.mean(score) for score in scores]
final_precision = max(precision)
print('Final model accuracy-score: {:.2f}%'.format(final_precision))

