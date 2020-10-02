#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


# In this kernel, the followng question is trying to be answered: **What are the most important measurements (regarding mobility) that must be taken in order to reduce the rate of cases/day in Latin America vs. The rest of the world?**
# 
# To given an answer to this question, the following databases were used:
# * current-data-on-the-geographic-distribution-of-covid-19-cases-worldwide.csv - to obtain the rate of cases/day
# * apple_mobility_trends/mobility-trends.csv - to have information about the different methods of transportation used in the last months
# * google_mobility/regional-mobility - to have information about the distribution of places regarding mobility
# 
# Therefore, the first thing to do is to organize and clean up each database. This can be seen in the following cells.

# In[ ]:


sa_countries = ['Belize','Costa_Rica','El_Salvador','Guatemala','Honduras','Mexico','Nicaragua','Panama','Argentina',
                'Bolivia','Brazil','Chile','Colombia','Ecuador','French_Guiana','Guyana','Paraguay','Peru','Suriname','Uruguay',
                'Venezuela','Cuba','Dominican_Republic','Haiti','Guadeloupe','Martinique','Puerto_Rico']

sta_1 = pd.read_csv('/kaggle/input/uncover/UNCOVER/ECDC/current-data-on-the-geographic-distribution-of-covid-19-cases-worldwide.csv') #Statistics
mob_1 = pd.read_csv('/kaggle/input/uncover/UNCOVER/apple_mobility_trends/mobility-trends.csv') #Mobility
mob_2 = pd.read_csv('/kaggle/input/uncover/UNCOVER/google_mobility/regional-mobility.csv') #Mobility

sta_1 = sta_1.drop(sta_1.columns[[0,7,8,10]], axis=1)
sta_1 = sta_1.assign(date = pd.to_datetime(sta_1[['year', 'month', 'day']]))
sta_1 = sta_1.drop(sta_1.columns[[0,1,2]], axis=1)
sta_1 = sta_1.dropna()
sta_1['countriesandterritories'].replace({'El Salvador':'El_Salvador', 'French Guiana':'French_Guiana',
                                          'Dominican Republic':'Dominican_Republic','Puerto Rico':'Puerto_Rico'},inplace=True)
mob_1 = mob_1.drop(mob_1[mob_1.geo_type == 'city'].index)
mob_1 = mob_1.drop(mob_1.columns[0], axis=1)
mob_1 = mob_1.dropna()
mob_1['region'].replace({'El Salvador':'El_Salvador', 'French Guiana':'French_Guiana',
                         'Dominican Republic':'Dominican_Republic','Puerto Rico':'Puerto_Rico'},inplace=True)

mob_2 = mob_2.drop(mob_2.columns[1], axis=1)
mob_2 = mob_2.dropna()
mob_2['country'].replace({'El Salvador':'El_Salvador', 'French Guiana':'French_Guiana',
                         'Dominican Republic':'Dominican_Republic','Puerto Rico':'Puerto_Rico'},inplace=True)


# In[ ]:


sta1_la = sta_1.loc[sta_1['countriesandterritories'].isin(sa_countries)] #Latin America
sta1_rw = sta_1.drop(sta1_la['countriesandterritories'].index) #Rest of the world
sta1_la = sta1_la.drop(sta1_la.columns[[2,3]], axis=1)
sta1_rw = sta1_rw.drop(sta1_rw.columns[[2,3]], axis=1)
sta1_la = sta1_la.groupby(['date']).sum().reset_index()
sta1_rw = sta1_rw.groupby(['date']).sum().reset_index()

mob1_la = mob_1.loc[mob_1['region'].isin(sa_countries)]
mob1_rw = mob_1.drop(mob1_la['region'].index)
mob1_la = mob1_la.drop(mob1_la.columns[0], axis=1)
mob1_rw = mob1_rw.drop(mob1_rw.columns[0], axis=1)
mob1_la = mob1_la.groupby(['date','transportation_type']).sum().reset_index()
mob1_rw = mob1_rw.groupby(['date','transportation_type']).sum().reset_index()

mob2_la = mob_2.loc[mob_2['country'].isin(sa_countries)]
mob2_rw = mob_2.drop(mob2_la['country'].index)
mob2_la = mob2_la.drop(mob2_la.columns[0], axis=1)
mob2_rw = mob2_rw.drop(mob2_rw.columns[0], axis=1)
mob2_la = mob2_la.groupby(['date']).sum().reset_index()
mob2_rw = mob2_rw.groupby(['date']).sum().reset_index()


# In[ ]:


df1 = mob1_la[mob1_la.transportation_type == 'driving']
df2 = mob1_la[mob1_la.transportation_type == 'walking']
df3 = mob1_la[mob1_la.transportation_type == 'transit']
df1 = df1.rename(columns={'value':'driving'})
df1['walking'] = df2.value.values
df1['transit'] = df3.value.values

mob1_la = df1.drop(df1.columns[[1]], axis=1)

df1 = mob1_rw[mob1_rw.transportation_type == 'driving']
df2 = mob1_rw[mob1_rw.transportation_type == 'walking']
df3 = mob1_rw[mob1_rw.transportation_type == 'transit']
df1 = df1.rename(columns={'value':'driving'})
df1['walking'] = df2.value.values
df1['transit'] = df3.value.values

mob1_rw = df1.drop(df1.columns[[1]], axis=1)


# So as it can be seen in the next print, the three data bases were divided for Latin America and the rest of the world. And, the format of each is consistent, were the columns are: 'date', 'feature1', 'feature2', ...
# 
# It is important to mention that the intial date and the final date in each database is the same. This, to have a consistent time-window for the data analysis.

# In[ ]:


sta1_la = sta1_la[sta1_la.date >= mob2_la.date[0]]
sta1_la = sta1_la[sta1_la.date <= mob2_la.date[62]]
sta1_rw = sta1_rw[sta1_rw.date >= mob2_la.date[0]]
sta1_rw = sta1_rw[sta1_rw.date <= mob2_la.date[62]]

mob1_la = mob1_la[mob1_la.date >= mob2_la.date[0]]
mob1_la = mob1_la[mob1_la.date <= mob2_la.date[62]]
mob1_rw = mob1_rw[mob1_rw.date >= mob2_la.date[0]]
mob1_rw = mob1_rw[mob1_rw.date <= mob2_la.date[62]]

print(sta1_la.head(),'\n')
print(mob1_la.head(),'\n')
print(mob2_la.head())


# Having cleaned up the databases, the following graphs show every index that will be studied. This, considering the two different groups mentioned before, Latin America and the rest of the world. The first graph shows the rate of cases/day and deaths/day.

# In[ ]:


fig = plt.figure(figsize = (19,3))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
sta1_la.set_index('date').plot(ax=ax1, title='Latin America')
sta1_rw.set_index('date').plot(ax=ax2, title='Rest of the world')


# The second graph shows the index for different methods of transportation/day used.

# In[ ]:


fig = plt.figure(figsize = (19,3))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
mob1_la.set_index('date').plot(ax=ax1, title='Latin America')
mob1_rw.set_index('date').plot(ax=ax2, title='Rest of the world')


# And the third one shows the index of locations/day.

# In[ ]:


fig = plt.figure(figsize = (19,3))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
mob2_la.set_index('date').plot(ax=ax1, title='Latin America')
mob2_rw.set_index('date').plot(ax=ax2, title='Rest of the world')


# Now, with the information obtained from cleaning up the databases, a full study can be done. But first, it is important to mention that the amount of data is very limited (only 63 instances - refering to almost 2 full months of data) which is a clear constraint and restriction for the models that will be deployed.

# In[ ]:


import sklearn.ensemble
import sklearn.model_selection
import sklearn.metrics
import sklearn.neural_network
from sklearn.preprocessing import StandardScaler


# First, the importance and effect of the methods of transportation will be studied. To have consistent results, the data needs to be scaled and normalize as it can be seen.

# In[ ]:


total_la = mob1_la
total_la['cases'] = sta1_la.cases.values

total_rw = mob1_rw
total_rw['cases'] = sta1_rw.cases.values

atributes_name = total_rw.columns
total_la = total_la.to_numpy()
total_rw = total_rw.to_numpy()

atributes_la = total_la[:,1:-1].astype(float)
target_la = total_la[:,-1].astype(float)

atributes_rw = total_rw[:,1:-1].astype(float)
target_rw = total_rw[:,-1].astype(float)


# In[ ]:


scalerla = StandardScaler()
xla_train, xla_test, yla_train, yla_test = sklearn.model_selection.train_test_split(atributes_la, target_la, train_size=0.7)
xla_train = scalerla.fit_transform(xla_train)
xla_test = scalerla.transform(xla_test)

scalerrw = StandardScaler()
xrw_train, xrw_test, yrw_train, yrw_test = sklearn.model_selection.train_test_split(atributes_rw, target_rw, train_size=0.7)
xrw_train = scalerrw.fit_transform(xrw_train)
xrw_test = scalerrw.transform(xrw_test)


# The following cell implements a method to find the best number of trees in the Random Forest Regressor for each group (Latin America vs. Rest of the world). Here, the variable that is being predicted is the amount of cases/day using the features from the second data base.

# In[ ]:


n_trees = np.arange(1,100,1)
score_la = []
score_rw = []

for n_tree in n_trees:
    clf_la = sklearn.ensemble.RandomForestRegressor(n_estimators=n_tree, max_features='sqrt')
    clf_la.fit(xla_train, yla_train)
    score_la.append(clf_la.score(xla_test,yla_test))
    
    clf_rw = sklearn.ensemble.RandomForestRegressor(n_estimators=n_tree, max_features='sqrt')
    clf_rw.fit(xrw_train, yrw_train)
    score_rw.append(clf_rw.score(xrw_test,yrw_test))

best_Mla = n_trees[np.argmax(score_la)]
clfla_best = sklearn.ensemble.RandomForestRegressor(n_estimators=best_Mla, max_features='sqrt')
clfla_best.fit(xla_train,yla_train)
importancesla = clfla_best.feature_importances_

best_Mrw = n_trees[np.argmax(score_rw)]
clfrw_best = sklearn.ensemble.RandomForestRegressor(n_estimators=best_Mrw, max_features='sqrt')
clfrw_best.fit(xrw_train,yrw_train)
importancesrw = clfrw_best.feature_importances_

fig = plt.figure(figsize = (15,5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ala = pd.Series(importancesla, index=atributes_name[1:-1])
ala.nlargest(9).plot(kind='barh', ax=ax1, title='Latin America: $R^2$ = {:.3f} , {:.0f} Trees'.format(clfla_best.score(xla_test,yla_test),best_Mla))
ax1.set_xlabel('Feature Importance')
arw = pd.Series(importancesrw, index=atributes_name[1:-1])
arw.nlargest(9).plot(kind='barh', ax=ax2, title='Rest of the world: $R^2$ = {:.3f} , {:.0f} Trees'.format(clfrw_best.score(xrw_test,yrw_test),best_Mrw))
ax2.set_xlabel('Feature Importance')
plt.tight_layout()


# As it can be seen from the graphs shown before, it seems as transit, walking and driving are all significant when determining the amount of cases/day for the rest of the world. But, for Latin America, it seems as if the only feature that has a high importance is transit. Considering the score for both models, the results of the rest of the world seem to have a better fit. But, in order to actually confirm the importance of this features LASSO was used. **It is important to mention that LASSO implies a linear regression, meaning that this is a very rough approximation to the data seen when cleaning up the databases.**

# In[ ]:


n_alpha = 100
alpha = np.logspace(-5, 5, n_alpha)

scores_la = []
betas_la = []

scores_rw = []
betas_rw = []

for a in alpha:
    lasso_la = sklearn.linear_model.Lasso(alpha=a, max_iter=10000)
    lasso_la.fit(xla_train, yla_train)
    scores_la.append(lasso_la.score(xla_test, yla_test))
    betas_la.append(lasso_la.coef_)
    
    lasso_rw = sklearn.linear_model.Lasso(alpha=a, max_iter=10000)
    lasso_rw.fit(xrw_train, yrw_train)
    scores_rw.append(lasso_rw.score(xrw_test, yrw_test))
    betas_rw.append(lasso_rw.coef_)

print("")
print("Best model (LASSO): Latin America R^2 = {}".format(max(scores_la)))
print("")
best_la = np.argmax(scores_la) 
beta_la = betas_la[best_la]
ii = np.argsort(-beta_la)
for i in ii:
    if(abs(beta_la[i])>0):
        print(atributes_name[1:-1][i], beta_la[i])
        
print("")
print("Best model (LASSO): Rest of the world R^2 = {}".format(max(scores_rw)))
print("")
best_rw = np.argmax(scores_rw) 
beta_rw = betas_rw[best_rw]
ii = np.argsort(-beta_rw)
for i in ii:
    if(abs(beta_rw[i])>0):
        print(atributes_name[1:-1][i], beta_rw[i])


# The results obtained are consistent with those obtained with the Random Forest Regressor. This, since the LASSO linear regression suggests that, for Latin America, t. And, regarding the rest of the world, the significant features in this case are walking and driving. 
# 
# **So, studying the intersection between the Random Forest Regressor and the LASSO linear regression, it seems as if the most significant feature for Latin America, regarding methods of transportation, when trying to determine the rate of cases/day is transit. While for the rest of the world, are both walking and driving. But, clearly, the lack of data and the rough approximation of using LASSO as a linear regression is a constraint and limits the accuracy of the results.**
# 
# Now, to study the effect of the locations, the same process was done, but in this case, considering the third database. Once again the variable to predict is simply the amount of cases/day.

# In[ ]:


total_la = mob2_la
total_la['cases'] = sta1_la.cases.values

total_rw = mob2_rw
total_rw['cases'] = sta1_rw.cases.values

atributes_name = total_rw.columns
total_la = total_la.to_numpy()
total_rw = total_rw.to_numpy()

atributes_la = total_la[:,1:-1].astype(float)
target_la = total_la[:,-1].astype(float)

atributes_rw = total_rw[:,1:-1].astype(float)
target_rw = total_rw[:,-1].astype(float)


# In[ ]:


scalerla = StandardScaler()
xla_train, xla_test, yla_train, yla_test = sklearn.model_selection.train_test_split(atributes_la, target_la, train_size=0.7)
xla_train = scalerla.fit_transform(xla_train)
xla_test = scalerla.transform(xla_test)

scalerrw = StandardScaler()
xrw_train, xrw_test, yrw_train, yrw_test = sklearn.model_selection.train_test_split(atributes_rw, target_rw, train_size=0.7)
xrw_train = scalerrw.fit_transform(xrw_train)
xrw_test = scalerrw.transform(xrw_test)


# In[ ]:


n_trees = np.arange(1,100,1)
score_la = []
score_rw = []

for n_tree in n_trees:
    clf_la = sklearn.ensemble.RandomForestRegressor(n_estimators=n_tree, max_features='sqrt')
    clf_la.fit(xla_train, yla_train)
    score_la.append(clf_la.score(xla_test,yla_test))
    
    clf_rw = sklearn.ensemble.RandomForestRegressor(n_estimators=n_tree, max_features='sqrt')
    clf_rw.fit(xrw_train, yrw_train)
    score_rw.append(clf_rw.score(xrw_test,yrw_test))

best_Mla = n_trees[np.argmax(score_la)]
clfla_best = sklearn.ensemble.RandomForestRegressor(n_estimators=best_Mla, max_features='sqrt')
clfla_best.fit(xla_train,yla_train)
importancesla = clfla_best.feature_importances_

best_Mrw = n_trees[np.argmax(score_rw)]
clfrw_best = sklearn.ensemble.RandomForestRegressor(n_estimators=best_Mrw, max_features='sqrt')
clfrw_best.fit(xrw_train,yrw_train)
importancesrw = clfrw_best.feature_importances_

fig = plt.figure(figsize = (15,5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ala = pd.Series(importancesla, index=atributes_name[1:-1])
ala.nlargest(9).plot(kind='barh', ax=ax1, title='Latin America: $R^2$ = {:.3f} , {:.0f} Trees'.format(clfla_best.score(xla_test,yla_test),best_Mla))
ax1.set_xlabel('Feature Importance')
arw = pd.Series(importancesrw, index=atributes_name[1:-1])
arw.nlargest(9).plot(kind='barh', ax=ax2, title='Rest of the world: $R^2$ = {:.3f} , {:.0f} Trees'.format(clfrw_best.score(xrw_test,yrw_test),best_Mrw))
ax2.set_xlabel('Feature Importance')
plt.tight_layout()


# Looking at both of these graphs, there are a lot of differences between Latin America and the rest of the world. It seems as if the most important feature for Latin America is the workplaces, while for the rest of the world are the parks, transit stations, groceries and pharmacies. All of these features seem to have a big importance when trying to determine the amount of cases/day. As well as this, the score for both models is not so high, meaning that there might be a bias and the results should not be taken as a final statement. Once again, to confirm the importance of these features LASSO linear regression was used.

# In[ ]:


n_alpha = 100
alpha = np.logspace(-5, 5, n_alpha)

scores_la = []
betas_la = []

scores_rw = []
betas_rw = []

for a in alpha:
    lasso_la = sklearn.linear_model.Lasso(alpha=a, max_iter=10000)
    lasso_la.fit(xla_train, yla_train)
    scores_la.append(lasso_la.score(xla_test, yla_test))
    betas_la.append(lasso_la.coef_)
    
    lasso_rw = sklearn.linear_model.Lasso(alpha=a, max_iter=10000)
    lasso_rw.fit(xrw_train, yrw_train)
    scores_rw.append(lasso_rw.score(xrw_test, yrw_test))
    betas_rw.append(lasso_rw.coef_)

print("")
print("Best model (LASSO): Latin America R^2 = {}".format(max(scores_la)))
print("")
best_la = np.argmax(scores_la) 
beta_la = betas_la[best_la]
ii = np.argsort(-beta_la)
for i in ii:
    if(abs(beta_la[i])>0):
        print(atributes_name[1:-1][i], beta_la[i])
        
print("")
print("Best model (LASSO): Rest of the world R^2 = {}".format(max(scores_rw)))
print("")
best_rw = np.argmax(scores_rw) 
beta_rw = betas_rw[best_rw]
ii = np.argsort(-beta_rw)
for i in ii:
    if(abs(beta_rw[i])>0):
        print(atributes_name[1:-1][i], beta_rw[i])


# In this case, not every feature has a coefficient different from zero, which means that some of the destinations studied do not play an important when trying to predict the number of cases per day in both Latin America and the rest of the world. Considering the results obtained for Latin America, these are consistent with those obtained with the Random Tree Regressor. This, since the features that need to be taken into account when determining the amount of cases/day are transit stations and workplaces. Also, this is consistent with the results from the first data base, since transit was the most significant feature when looking at the methods of transportation. Now, taking into account the results obtained for the rest of the world, they are clearly consistent with those obtained with the Random Forest Regressor. But, it is important to mention that the parks where knocked off when using LASSO. 
# 
# **So, studying the intersection between the Random Forest Regressor and the LASSO linear regression, it seems as if the most significant feature for Latin America, regarding destinations, when trying to determine the rate of cases/day are transit stations and workplaces. While for the rest of the world, are groceries, pharmacies and transit station. Lastly I cannot stress enough that the lack of data and the rough approximation of using LASSO as a linear regression is a constraint and limits the accuracy of the results.**
# 
