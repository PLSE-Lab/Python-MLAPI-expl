#!/usr/bin/env python
# coding: utf-8

# ## Abstract
# 
# In this work I'm concetrated on math exam scores both 4th and 8th grade. I found the main influence on score have YEAR of exam, everage money spend on one student (in my dataset asigned as AVERAGE) and the STATE where students are examed. Depanding on model (linear or random forest) AVERAGE cash play high role or STATE.

# ## Loading libraries and data file

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df=pd.read_csv('../input/states_all.csv')


# In[ ]:


df.isna().sum()*100/df.shape[0]


# Quick look what is going on in the data. On ecan see two main groups of data
# 1.  revenue and expenditure 
# 1.  exams
# 1. Correlation between revenue/expenditure and grades suggest if there are more students than there is more maney for the state for them.

# In[ ]:



corr = df.corr()
fig, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(corr, annot=True,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# ## Let's see what is the influence of state name on the revenue

# In[ ]:


fig, ax = plt.subplots(figsize=(15, 15))
sns.violinplot(x="TOTAL_REVENUE", y="STATE", data=df)


# In[ ]:


df['average_reveneue']=df['TOTAL_REVENUE']/df['GRADES_ALL_G']
df['average_expenditure']=df['TOTAL_EXPENDITURE']/df['GRADES_ALL_G']


# ## Now let's take TOTAL_REVENUE/GRADES_ALL_G what means we have more equality, but not perfect

# In[ ]:


fig, ax = plt.subplots(figsize=(15, 15))
sns.violinplot(x="average_reveneue", y="STATE", data=df[df.STATE!='VIRGINIA'])


# In[ ]:


fig, ax = plt.subplots(figsize=(15, 15))
sns.violinplot(x="average_expenditure", y="STATE", data=df[df.STATE!='VIRGINIA'])


# ## Lets look on the influence of average cash on the exams

# In[ ]:


sns.jointplot("average_expenditure", "AVG_MATH_4_SCORE", data=df, kind="reg")


# In[ ]:


sns.jointplot("TOTAL_REVENUE", "AVG_MATH_4_SCORE", data=df, kind="reg")


# ## It looks like it makes difference if calculating money for all students and while making average for each student

# ## Creating model  for AVG_MATH_4_SCORE

# In[ ]:


df1=df.drop(['PRIMARY_KEY','AVG_MATH_8_SCORE','AVG_READING_4_SCORE','AVG_READING_8_SCORE','ENROLL'],axis=1)
df2=df1.dropna()
df3 = pd.get_dummies(df2, columns=['STATE'])
df4=(df3-df3.mean())/df3.std()


# In[ ]:


y=df4.loc[:,'AVG_MATH_4_SCORE'].values
X=df4.drop(['AVG_MATH_4_SCORE'],axis=1).loc[:,:].values


# In[ ]:


print(X.shape,' ',y.shape)


# In[ ]:


rf=RandomForestRegressor()

parameters = {'n_estimators': [4, 6, 9], 
              'max_features': ['log2', 'sqrt','auto'], 
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }

# Run the grid search
grid_obj = GridSearchCV(rf, parameters, cv=5)
grid_obj = grid_obj.fit(X, y)

# Set the clf to the best combination of parameters
rf = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
print('Params ',rf)
print('Score ',rf.score(X, y))


# ## It is hard to fit X and y because X is multicolumn so i fit Y redict and y data- this shoud be stright line.

# In[ ]:


Y_rf=rf.predict(X)
plt.plot(Y_rf, y, 'ro')
plt.show()


# In[ ]:


feature_importances_rf = pd.DataFrame(rf.feature_importances_,
                                   index = df4.drop(['AVG_MATH_4_SCORE'],axis=1).columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
feature_importances_rf.head()


# In[ ]:


from sklearn.linear_model import Lasso, ElasticNet
clf = Lasso()

parameters = {'alpha': [0.00001,0.0001,0.001, 0.01],
              'tol': [0.00001,0.0001,0.001, 0.01]
             }

# Run the grid search
grid_obj = GridSearchCV(clf, parameters, cv=5)
grid_obj = grid_obj.fit(X, y)

# Set the clf to the best combination of parameters
clf = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
print('Params ',clf)
print('Score ',clf.score(X, y))


# In[ ]:


Y_clf=clf.predict(X)
plt.plot(Y_clf, y, 'ro')
plt.show()


# In[ ]:


feature_importances_clf = pd.DataFrame(clf.coef_,
                                   index = df4.drop(['AVG_MATH_4_SCORE'],axis=1).columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
feature_importances_clf.head(10)


# In[ ]:


eln = ElasticNet()

parameters = {'alpha': [0.00001,0.0001,0.001, 0.01],
              'l1_ratio': [0.0001,0.001,0.01, 0.1],
              'tol': [0.00001,0.0001,0.001, 0.01],
              'max_iter': [1000,2000,5000, 10000],
             }

# Run the grid search
grid_obj = GridSearchCV(eln, parameters, cv=5)
grid_obj = grid_obj.fit(X, y)

# Set the clf to the best combination of parameters
eln = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
print('Params ',eln)
print('Score ',eln.score(X, y))


# In[ ]:


Y_eln=eln.predict(X)
plt.plot(Y_eln, y, 'ro')
plt.show()


# ## Lowest and highest value are best predicted by Random Forest, middle are simmilar for all models

# In[ ]:


fig, ax = plt.subplots(figsize=(15, 15))
plt.plot(Y_eln, y, 'ro',label='Elastic Net')#red Elastic Net
plt.plot(Y_clf, y, 'bs',label='Lasso')#blue Lasso
plt.plot(Y_rf, y, 'g^',label='Random Forest')#green Random Forest
plt.plot([-5,2],[-5,2])#PREDICTION LIKE IT SHOUD BE
plt.legend()
plt.show()


# In[ ]:


feature_importances_eln = pd.DataFrame(eln.coef_,
                                   index = df4.drop(['AVG_MATH_4_SCORE'],axis=1).columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
feature_importances_eln.head(10)


# * Non-linear Random Forest makes slightly better score than linear Lasso and Elastic Net. 
# * From linear calculations highest influence on the math 4 exam have GRADES, year, expenditure and on 8th place average expenditure on student. This result is an exchange reason with effect. Analyze suggests the more trudents are at grades 9, 12 and 8 the better is 4th grade result. Correlation is the same but the REASON why there are many students in grade higher than 4th is that more of them passed exam. 
# * Expenditure is more importand than revenue. In this cale RF is better for best and worst results so it may mean average revenue is more important for best and worst while expenditure for the rest.

# ## Influence of time:from correlation matrix we could see that there is an influence time on expenditure what makes more money for student in time

# In[ ]:


fig, ax = plt.subplots(figsize=(15, 15))
sns.lineplot(x="YEAR", y="AVG_MATH_4_SCORE", data=df)


# ## Building model for AVG_MATH_8_SCORE

# In[ ]:


df1=df.drop(['PRIMARY_KEY','AVG_MATH_4_SCORE','AVG_READING_4_SCORE','AVG_READING_8_SCORE','ENROLL'],axis=1)
df2=df1.dropna()
df3 = pd.get_dummies(df2, columns=['STATE'])
df4=(df3-df3.mean())/df3.std()


# In[ ]:


y=df4.loc[:,'AVG_MATH_8_SCORE'].values
X=df4.drop(['AVG_MATH_8_SCORE'],axis=1).loc[:,:].values


# In[ ]:


rf=RandomForestRegressor()

parameters = {'n_estimators': [4, 6, 9], 
              'max_features': ['log2', 'sqrt','auto'], 
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }

# Run the grid search
grid_obj = GridSearchCV(rf, parameters, cv=5)
grid_obj = grid_obj.fit(X, y)

# Set the clf to the best combination of parameters
rf = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
print('Params ',rf)
print('Score ',rf.score(X, y))


# In[ ]:


Y_rf=rf.predict(X)
plt.plot(Y_rf, y, 'ro')
plt.show()


# In[ ]:


feature_importances_rf = pd.DataFrame(rf.feature_importances_,
                                   index = df4.drop(['AVG_MATH_8_SCORE'],axis=1).columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
feature_importances_rf.head()


# In[ ]:


clf = Lasso()

parameters = {'alpha': [0.00001,0.0001,0.001, 0.01],
              'tol': [0.00001,0.0001,0.001, 0.01]
             }

# Run the grid search
grid_obj = GridSearchCV(clf, parameters, cv=5)
grid_obj = grid_obj.fit(X, y)

# Set the clf to the best combination of parameters
clf = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
print('Params ',clf)
print('Score ',clf.score(X, y))


# In[ ]:


Y_clf=clf.predict(X)
plt.plot(Y_clf, y, 'ro')
plt.show()


# In[ ]:


feature_importances_clf = pd.DataFrame(clf.coef_,
                                   index = df4.drop(['AVG_MATH_8_SCORE'],axis=1).columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
feature_importances_clf.head(10)


# In[ ]:


eln = ElasticNet()

parameters = {'alpha': [0.00001,0.0001,0.001, 0.01],
              'l1_ratio': [0.0001,0.001,0.01, 0.1],
              'tol': [0.00001,0.0001,0.001, 0.01],
              'max_iter': [1000,2000,5000, 10000],
             }

# Run the grid search
grid_obj = GridSearchCV(eln, parameters, cv=5)
grid_obj = grid_obj.fit(X, y)

# Set the clf to the best combination of parameters
eln = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
print('Params ',eln)
print('Score ',eln.score(X, y))


# In[ ]:


Y_eln=clf.predict(X)
plt.plot(Y_eln, y, 'ro')
plt.show()


# In[ ]:


feature_importances_eln = pd.DataFrame(eln.coef_,
                                   index = df4.drop(['AVG_MATH_8_SCORE'],axis=1).columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
feature_importances_eln.head(10)


# Result is simmilar. RF is best but not significantly for lowest and highest results.

# In[ ]:


fig, ax = plt.subplots(figsize=(15, 15))
plt.plot(Y_eln, y, 'ro',label='Elastic Net')#red Elastic Net
plt.plot(Y_clf, y, 'bs',label='Lasso')#blue Lasso
plt.plot(Y_rf, y, 'g^',label='Random Forest')#green Random Forest
plt.plot([-5,2],[-5,2])#PREDICTION LIKE IT SHOUD BE
plt.legend()
plt.show()


# * If RF is best for worst and best students in this area most important are money-calculated averages
# * for other students most important is where are they studying- the STATE
# * it means (what was already shown) each STATE gives slightly different sum of money for each student
# * . MASSACHUSETT gives highest part, then MINESOTA, NORTH DACOTA
# * good correlation are for states given highest part, so RF works better for wors students that best students. IN this case average money plays most important role

# In[ ]:


fig, ax = plt.subplots(figsize=(15, 15))
sns.boxplot(x="AVG_MATH_8_SCORE", y="STATE", data=df)


# In[ ]:


states=df.groupby(['STATE']).mean()
normalized_df=(states-states.mean())/states.std()
fig, ax = plt.subplots(figsize=(15, 15))
normalized_df['AVG_MATH_8_SCORE'].plot(kind='bar');

