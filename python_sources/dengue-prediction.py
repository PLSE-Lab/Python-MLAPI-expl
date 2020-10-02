#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[ ]:


train=pd.read_csv('../input/DengAI_Predicting_Disease_Spread_-_Training_Data_Features.csv')


# In[ ]:


train.sample(5)


# In[ ]:


test=pd.read_csv('../input/DengAI_Predicting_Disease_Spread_-_Test_Data_Features.csv')


# In[ ]:


test.shape


# In[ ]:


feat_train=pd.read_csv('../input/DengAI_Predicting_Disease_Spread_-_Training_Data_Labels.csv')


# In[ ]:


feat_train.shape


# In[ ]:


train.fillna(train.mean(), inplace=True)


# In[ ]:


test.fillna(train.mean(), inplace=True)


# In[ ]:


test.isnull().sum()


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


plt.figure(figsize=(10,10))
train['ndvi_ne'].plot()
train['ndvi_nw'].plot()
plt.legend()


# In[ ]:


df=pd.merge(train, feat_train)


# In[ ]:


df.head()


# In[ ]:


c1=df[df['city']=='sj']


# In[ ]:


c1.shape


# In[ ]:


c2=df[df['city']=='iq']


# In[ ]:


c2.shape


# In[ ]:





# In[ ]:


c1['week_start_date']=pd.to_datetime(c1['week_start_date'])


# In[ ]:


c1.dtypes


# In[ ]:


get_ipython().run_line_magic('timeit', "pd.to_datetime(c1['week_start_date'], infer_datetime_format=True)")


# In[ ]:


c1.dtypes


# In[ ]:


c1.set_index(c1['week_start_date'], inplace=True)


# In[ ]:


c1.shape


# In[ ]:


plt.figure(figsize=(10,10))
c1['ndvi_ne'].plot()
c1['ndvi_nw'].plot()
c1['ndvi_se'].plot()
c1['ndvi_sw'].plot()
plt.legend()
plt.show()


# In[ ]:


get_ipython().run_line_magic('timeit', "pd.to_datetime(c2['week_start_date'], infer_datetime_format=True)")


# In[ ]:


c2.set_index(c2['week_start_date'], inplace=True)


# In[ ]:


plt.figure(figsize=(10,10))
c2['ndvi_ne'].plot()
c2['ndvi_nw'].plot()
c2['ndvi_se'].plot()
c2['ndvi_sw'].plot()
plt.legend()
plt.show()


# In[ ]:


plt.subplot(1,2,1)
c1['ndvi_ne'].plot()
c1['ndvi_nw'].plot()
c1['ndvi_se'].plot()
c1['ndvi_sw'].plot()
plt.legend()
plt.subplot(1,2,2)
c2['ndvi_ne'].plot()
c2['ndvi_nw'].plot()
c2['ndvi_se'].plot()
c2['ndvi_sw'].plot()
plt.legend()


# In[ ]:





# In[ ]:





# In[ ]:


get_ipython().run_line_magic('timeit', "pd.to_datetime(df['week_start_date'], infer_datetime_format=True)")


# In[ ]:





# In[ ]:


c2['week_start_date']=pd.to_datetime(c2['week_start_date'])


# In[ ]:





# In[ ]:


c2.set_index(c2['week_start_date'], inplace=True)


# In[ ]:


plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
c1['ndvi_ne'].plot()
c1['ndvi_nw'].plot()
c1['ndvi_se'].plot()
c1['ndvi_sw'].plot()
plt.legend()
plt.subplot(1,2,2)
c2['ndvi_ne'].plot()
c2['ndvi_nw'].plot()
c2['ndvi_se'].plot()
c2['ndvi_sw'].plot()
plt.legend()


# In[ ]:





# In[ ]:


plt.figure(figsize=(15,15))
c1['reanalysis_air_temp_k'].plot()
c1['reanalysis_avg_temp_k'].plot()
c1['reanalysis_dew_point_temp_k'].plot()
c1['reanalysis_max_air_temp_k'].plot()
c1['reanalysis_min_air_temp_k'].plot()
plt.legend()


# In[ ]:


plt.figure(figsize=(15,15))
c2['reanalysis_air_temp_k'].plot()
c2['reanalysis_avg_temp_k'].plot()
c2['reanalysis_dew_point_temp_k'].plot()
c2['reanalysis_max_air_temp_k'].plot()
c2['reanalysis_min_air_temp_k'].plot()
plt.legend()


# In[ ]:





# In[ ]:


plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
c1['reanalysis_air_temp_k'].plot()
c1['reanalysis_avg_temp_k'].plot()
c1['reanalysis_dew_point_temp_k'].plot()
c1['reanalysis_max_air_temp_k'].plot()
c1['reanalysis_min_air_temp_k'].plot()
plt.legend()
plt.subplot(1,2,2)
c2['reanalysis_air_temp_k'].plot()
c2['reanalysis_avg_temp_k'].plot()
c2['reanalysis_dew_point_temp_k'].plot()
c2['reanalysis_max_air_temp_k'].plot()
c2['reanalysis_min_air_temp_k'].plot()
plt.legend()


# In[ ]:


df.dtypes


# In[ ]:


g=sns.lmplot(x='total_cases', y='ndvi_ne', data=c1, markers='o')
g=sns.lmplot(x='total_cases', y='ndvi_nw', data=c1, markers='v')
g=sns.lmplot(x='total_cases', y='ndvi_se', data=c1, markers='^')
g=sns.lmplot(x='total_cases', y='ndvi_sw', data=c1, markers='s')


# In[ ]:





# In[ ]:


g=sns.lmplot(x='total_cases', y='ndvi_ne', data=df, markers='o', col='city', hue='city')
g=sns.lmplot(x='total_cases', y='ndvi_nw', data=df, markers='o', col='city', hue='city')
g=sns.lmplot(x='total_cases', y='ndvi_se', data=df, markers='o', col='city', hue='city')
g=sns.lmplot(x='total_cases', y='ndvi_sw', data=df, markers='o', col='city', hue='city')


# In[ ]:





# In[ ]:


g=sns.lmplot(x='year', y='ndvi_ne', data=df, markers='o', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='year', y='ndvi_nw', data=df, markers='v', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='year', y='ndvi_se', data=df, markers='s', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='year', y='ndvi_sw', data=df, markers='^', col='city', hue='city',aspect=1.5, x_jitter=.1)


# In[ ]:





# In[ ]:





# In[ ]:


g=sns.lmplot(x='year', y='reanalysis_air_temp_k', data=df, markers='o', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='year', y='reanalysis_avg_temp_k', data=df, markers='v', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='year', y='reanalysis_dew_point_temp_k', data=df, markers='s', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='year', y='reanalysis_max_air_temp_k', data=df, markers='^', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='year', y='reanalysis_min_air_temp_k', data=df, markers='^', col='city', hue='city',aspect=1.5, x_jitter=.1)


# In[ ]:


g=sns.lmplot(x='total_cases', y='ndvi_ne', data=df, markers='o', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='total_cases', y='ndvi_nw', data=df, markers='v', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='total_cases', y='ndvi_se', data=df, markers='s', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='total_cases', y='ndvi_sw', data=df, markers='^', col='city', hue='city',aspect=1.5, x_jitter=.1)


# In[ ]:


g=sns.lmplot(x='total_cases', y='reanalysis_air_temp_k', data=df, markers='o', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='total_cases', y='reanalysis_avg_temp_k', data=df, markers='v', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='total_cases', y='reanalysis_dew_point_temp_k', data=df, markers='s', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='total_cases', y='reanalysis_max_air_temp_k', data=df, markers='^', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='total_cases', y='reanalysis_min_air_temp_k', data=df, markers='^', col='city', hue='city',aspect=1.5, x_jitter=.1)


# In[ ]:





# In[ ]:


g=sns.lmplot(x='year', y='reanalysis_precip_amt_kg_per_m2', data=df, markers='o', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='year', y='reanalysis_relative_humidity_percent', data=df, markers='v', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='year', y='reanalysis_sat_precip_amt_mm', data=df, markers='s', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='year', y='reanalysis_specific_humidity_g_per_kg', data=df, markers='^', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='year', y='reanalysis_tdtr_k', data=df, markers='^', col='city', hue='city',aspect=1.5, x_jitter=.1)


# In[ ]:


g=sns.lmplot(x='total_cases', y='reanalysis_precip_amt_kg_per_m2', data=df, markers='o', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='total_cases', y='reanalysis_relative_humidity_percent', data=df, markers='v', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='total_cases', y='reanalysis_sat_precip_amt_mm', data=df, markers='s', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='total_cases', y='reanalysis_specific_humidity_g_per_kg', data=df, markers='^', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='total_cases', y='reanalysis_tdtr_k', data=df, markers='^', col='city', hue='city',aspect=1.5, x_jitter=.1)


# In[ ]:


g=sns.lmplot(x='year', y='station_avg_temp_c', data=df, markers='o', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='year', y='station_diur_temp_rng_c', data=df, markers='v', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='year', y='station_max_temp_c', data=df, markers='s', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='year', y='station_min_temp_c', data=df, markers='^', col='city', hue='city',aspect=1.5, x_jitter=.1)


# In[ ]:


g=sns.lmplot(x='total_cases', y='station_avg_temp_c', data=df, markers='o', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='total_cases', y='station_diur_temp_rng_c', data=df, markers='v', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='total_cases', y='station_max_temp_c', data=df, markers='s', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='total_cases', y='station_min_temp_c', data=df, markers='^', col='city', hue='city',aspect=1.5, x_jitter=.1)


# In[ ]:


feature_col=['city','week_start_date', 'total_cases','weekofyear']
X=df.drop(feature_col, axis=1)


# In[ ]:


X.shape


# In[ ]:





# In[ ]:


y=df.iloc[:,24]


# In[ ]:


y.shape


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[ ]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)


# In[ ]:


clf = KNeighborsClassifier(n_neighbors = 13)
scoring = 'accuracy'
score = cross_val_score(clf, X, y, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[ ]:


np.mean(score)*100


# In[ ]:


clf = DecisionTreeClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, X, y, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[ ]:


np.mean(score)*100


# In[ ]:


clf = RandomForestClassifier(n_estimators=13)
scoring = 'accuracy'
score = cross_val_score(clf, X, y, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[ ]:


np.mean(score)*100


# In[ ]:


clf = GaussianNB()
scoring = 'accuracy'
score = cross_val_score(clf, X, y, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[ ]:


np.mean(score)*100


# In[ ]:


clf = SVC()
scoring = 'accuracy'
score = cross_val_score(clf, X, y, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[ ]:


np.mean(score)*100


# In[ ]:


test.head()


# In[ ]:


feature_col=['city','week_start_date', 'weekofyear']
test=test.drop(feature_col, axis=1)


# In[ ]:


clf=SVC()
clf.fit(X, y)

pred=clf.predict(test)


# In[ ]:


submission = pd.DataFrame({
        "total": pred
})

submission.to_csv('submission_format.csv', index=False)


# In[ ]:


submission = pd.read_csv('submission_format.csv')
submission.head()

