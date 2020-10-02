#!/usr/bin/env python
# coding: utf-8

# # Predicting chance of admission in a Masters Program.
# in this notebook, we will be looking at different parameters which influence selection for a Masters Program. we will analyse  [Graduate Admissions](https://www.kaggle.com/mohansacharya/graduate-admissions) data set, build multiple probabilistic models and evaluate them. <br/>
# **Objective :**
#     *  Data exploration and Visualization     
#     *  feature importance/selection.
#     *  Build Model     

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import warnings
warnings.simplefilter('ignore')
import missingno as msn


# ## 1. Loading Data

# In[ ]:


os.listdir('../input/')


# In[ ]:


data_1 = pd.read_csv('../input/Admission_Predict.csv')
data_2 = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')


# In[ ]:


print("Data Columns: ",np.array(data_1.columns.tolist()))


#  [GRE](https://en.wikipedia.org/wiki/Graduate_Record_Examinations): Graduate Record Examinations 
# [TOEFL](https://en.wikipedia.org/wiki/Test_of_English_as_a_Foreign_Language/): Test of English as a Foreign Language
# [SOP](https://en.wikipedia.org/wiki/Standard_operating_procedure): Statement of Purpose
# [LOR](https://en.wikipedia.org/wiki/Letter_of_recommendation): Letter of Recommendation
# [CGPA](https://en.wikipedia.org/wiki/Grading_in_education): Grades

# In[ ]:


#original columns has trailing sapaces in LOR and Chance of Admit. changed Serial no. with Id
cols = ['Id', 'GRE Score', 'TOEFL Score', 'University Rating', 'SOP',
       'LOR', 'CGPA', 'Research', 'Chance of Admit']
data_1.columns = cols
data_2.columns = cols


# In[ ]:


data_1.head()


# In[ ]:


data_2.head()


# In[ ]:


print("data_1 length: ",len(data_1))
print("data_2 length: ",len(data_2))


# In[ ]:


(data_1 != data_2[:len(data_1)]).sum()


# as data_1 is a subset of data_2, we will continue with data_2 only.

# In[ ]:


data = data_2.copy()


# ### Missing Value Visualization

# In[ ]:


msn.matrix(data,figsize=(10,5))


# in this data we don't have any missing values. when we have lot of features  it's a good idea to visualize missing values. this give us an idea about inter correlation between features and will help in feature selection.

# ## 2. Data Exploration and Visualization

# In[ ]:


data.describe()


# In[ ]:


data.info()


# **Take away:** 
# * from info() we can see there is no null values in data and also give info about the data types.
# * describe() tells that some students got full marks in GRE and TOEFL and there is no student with 10 CGPA. Chance of selection variable has maximum value 0.97 which also makes sense because max CGPA is 9.92. 

# In[ ]:


## numerical and categorical columns
num_cols = ['GRE Score','TOEFL Score', 'CGPA']
cat_cols = ['University Rating', 'SOP','LOR']
bin_cols = ['Research']


# In[ ]:


g = sns.pairplot(data,hue='Research',vars=num_cols,diag_kind='kde')


#  let's explore **CGPA** and **Research Experience**!
#  we will be looking for two things:
# * Does students with **research
# expereince** **research
# expereince** have better **CGPA** ?
# * Does students with better **CGPA** go for **research**?

# In[ ]:


g = sns.catplot(x='Research',y='CGPA',data=data,kind='box')
g.fig.set_size_inches([10,5])


# In[ ]:


plt.figure(figsize=(10,5))
sns.distplot(data[data['Research'] == 0]['CGPA'],label='0',bins=20)
sns.distplot(data[data['Research'] == 1]['CGPA'],label='1',bins=20)
plt.legend()
plt.show()


# In[ ]:


print("Num of students with research experience: ",len(data[data.Research == 1]))
print("Num of students with NO research experience: ",len(data[data.Research == 0]))
print("Avg CGPA of students with research experience: ", data[data.Research == 1]['CGPA'].mean())
print("Avg CGPA of students with NO research experience: ", data[data.Research == 0]['CGPA'].mean())


# now let's define 7.5 as better CGPA criterion. and see percentage of students with research experience with CGPA >= 7.5 and < 7.5 

# In[ ]:


CGPA_thre = 7.5
print("Num of students with CGPA >= {}: {}".format(CGPA_thre,len(data[data.CGPA >= CGPA_thre])))
print("Num of students with CGPA < {}: {}".format(CGPA_thre,len(data[data.CGPA < CGPA_thre])))
print("percentage of students with CGPA >= {} and go for research: {}".format(CGPA_thre,
                                                                                   len(data[(data.CGPA>=CGPA_thre)&(data.Research == 1)])/len(data[data.CGPA >= CGPA_thre])))
print("percentage of students with CGPA < {} and go for research: {}".format(CGPA_thre,
                                                                                   len(data[(data.CGPA<CGPA_thre)&(data.Research == 1)])/len(data[data.CGPA < CGPA_thre])))


# **Take Away**:
# We know that both of these statment are very obvious. research guys usually have better CGPA and students with better CGPA (better academics) go for research<br/>
# But from this data we can't make a strong claim on the second statement because we have very small sample set with CGPA <7.5 (only 3.6% students). <br/>
# Asking questions from the data and answering them by giving some stats from the data is really a great way for data analysis. it gives better understanding of data and helps in feature engineering. 

# let's look at some plots!

# In[ ]:


fig,[axs1,axs2] = plt.subplots(ncols=2,figsize=(20,5))
sns.scatterplot(x='GRE Score',y='CGPA',data=data,hue='SOP',ax=axs1,size='SOP')
sns.scatterplot(x='GRE Score',y='CGPA',data=data,hue= 'LOR',ax=axs2,size='LOR')
fig.show()


# In[ ]:


fig,axs = plt.subplots(nrows=2,ncols=2,figsize=(20,10))
sns.countplot(x='LOR',data=data,ax=axs[0,0])
sns.countplot(x='SOP',data=data,ax=axs[0,1])
sns.countplot(x='University Rating',data=data,ax=axs[1,0])
sns.countplot(x='Research',data=data,ax=axs[1,1])
fig.show()


# ### Relation with target (chance of Admit)

# In[ ]:


_,axs = plt.subplots(ncols=3,figsize=(30,5))
sns.scatterplot(x='CGPA',y='Chance of Admit',data=data,hue='Research',ax=axs[0])
sns.scatterplot(x='GRE Score',y='Chance of Admit',data=data,hue='Research',ax=axs[1])
sns.scatterplot(x='TOEFL Score',y='Chance of Admit',data=data,hue='Research',ax=axs[2])


# ### Categorical Features

# In[ ]:


sns.set_style('whitegrid')
g = sns.catplot(x='SOP',y='Chance of Admit',data=data,kind='swarm',col='Research')
g.fig.set_size_inches([20,5])
g.set_xticklabels("")
sns.set_style('whitegrid')
g = sns.catplot(x='LOR',y='Chance of Admit',data=data,kind='swarm',col='Research')
g.fig.set_size_inches([20,5])
g.set_titles("")


# In[ ]:


g = sns.relplot(x='CGPA',y='Chance of Admit',data=data,col='Research',hue='University Rating',size='University Rating')
g.fig.set_size_inches([20,5])
g = sns.catplot(x='Research',y='Chance of Admit',data=data,kind='box')
g.fig.set_size_inches([10,5])


# ### Correlation Matrix

# In[ ]:


cols = ['GRE Score','TOEFL Score','SOP','LOR','CGPA','Chance of Admit']
g = sns.heatmap(data[cols].corr(),annot=True,xticklabels=cols,yticklabels=cols,fmt=".2f")
g.figure.set_size_inches([10,10])


# ## 3. Feature Importance Visualization
# 

# Main things I consider for feature selection
# * feature importance provided by tree based models usually random forest / xgboost.
# * Percentage of missing values.
# * correlation with target value
# * insights from data visualization
# * also try dimensionality reduction methods. like PCA
# * Recursive feature elimination<br/>
# we will take the full data set and see feature importance. we have very less number of feature so feature selection is not that important. but still we ll have look!

# In[ ]:


X_cols = ['GRE Score','TOEFL Score','University Rating','SOP','LOR','CGPA','Research']
target = 'Chance of Admit'
X = data[X_cols]
y = data['Chance of Admit']


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from xgboost import plot_importance
rfc = RandomForestRegressor(max_depth=3)
xgb = XGBRegressor()


# In[ ]:


rfc.fit(X,y)
xgb.fit(X,y)


# In[ ]:


df_feature_imp = pd.DataFrame()
df_feature_imp['feature_name'] = X.columns
df_feature_imp['xgb'] = xgb.feature_importances_
df_feature_imp['rfc'] = rfc.feature_importances_
df_feature_imp


# In[ ]:


_,axs = plt.subplots(ncols=2,sharey=True,figsize=(10,5))
sns.barplot(x='xgb',y='feature_name',data=df_feature_imp,ax=axs[0])
sns.barplot(x='rfc',y='feature_name',data=df_feature_imp,ax=axs[1])


# **Take Away:**
# we can see CGPA, GRE Score and TOEFL Score are the most important features from both the models. Thses three also have highest correlation values with target value 0.88 (CGPA), 0.81 (GRE Score), 0.79 (TOEFL Score)

# ## Model

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


# In[ ]:


xgb = XGBRegressor(max_depth=3,learning_rate=0.1)
xgb.fit(X_train,y_train)


# In[ ]:


from sklearn.metrics import mean_squared_error
print("Using XGB")
print("RMSE score on training data: ",np.sqrt(mean_squared_error(y_true=y_train,y_pred=xgb.predict(X_train))))
print("RMSE score on testing data: ",np.sqrt(mean_squared_error(y_true=y_test,y_pred=xgb.predict(X_test))))


# In[ ]:


rfc = RandomForestRegressor(max_depth=3)
rfc.fit(X_train,y_train)


# In[ ]:


print("Using rfc")
print("RMSE score on training data: ",np.sqrt(mean_squared_error(y_true=y_train,y_pred=rfc.predict(X_train))))
print("RMSE score on testing data: ",np.sqrt(mean_squared_error(y_true=y_test,y_pred=rfc.predict(X_test))))


# **The most important thing usually people miss is looking at the data on which model is not performing well. doing visualization after building model can give an idea about features because of which model is confusing or performing poorly and help improving model performance.** <br/>
# things we should look:
# * seperate out data on which model is performing good and poor.
# * Analyse both of these data set together and look for features which are:
#     1. different while looking at true positive / false negative data<br\
#     2. similar in case of true positive / false positive data. 

# In[ ]:


df_eval_train = pd.DataFrame()
df_eval_test = pd.DataFrame()

df_eval_train['rfc'] = rfc.predict(X_train)
df_eval_train['xgb'] = xgb.predict(X_train)
df_eval_train['target'] = y_train.values

df_eval_test['rfc'] = rfc.predict(X_test)
df_eval_test['xgb'] = xgb.predict(X_test)
df_eval_test['target'] = y_test.values


# ### **let's visualize how these models are doing on train and test**

# In[ ]:


df_eval_train.sort_values('target',inplace=True)
df_eval_train.reset_index(drop=True,inplace=True)
_,axs = plt.subplots(2,2,figsize=(20,10))
axs[0,0].plot(df_eval_train['xgb'])
axs[0,0].plot(df_eval_train['target'])
axs[0,1].plot(df_eval_train['rfc'])
axs[0,1].plot(df_eval_train['target'])

axs[1,0].plot(df_eval_test['target'])
axs[1,0].plot(df_eval_test['xgb'],'o')
axs[1,1].plot(df_eval_test['target'])
axs[1,1].plot(df_eval_test['rfc'],'o')
axs[0,0].legend(['target','XGB'])
axs[0,1].legend(['target','RFC'])

axs[0,0].set_ylabel('Train')
axs[1,0].set_ylabel('Test')

axs[1,0].set_xlabel('XGB')
axs[1,1].set_xlabel('RFC')


# ### It's difficult to interpret plots on testing data. we'll try something else.<br/>
# let's plot squared error for individual sample 

# In[ ]:


df_eval_test['rfc_diff'] = np.square(df_eval_test['target']-df_eval_test['rfc'])
df_eval_test['xgb_diff'] = np.square(df_eval_test['target']-df_eval_test['xgb'])
plt.figure(figsize=(10,5))
plt.plot(df_eval_test['rfc_diff'],'o')
plt.plot(df_eval_test['xgb_diff'],'o')
plt.legend(['RFC','XGB'])


# **Take Away:**
# * both models are having problem while making prediction on samples with low target values (<0.6)
# * XGB is performing better than RFC on testing data <br/>
# **Sorry for naming RFC(Random forest classifier) it should be RFR :P**

# **Success is no accident. It is hard work, perseverance, learning, studying, sacrifice and most of all, love of what you are doing or learning to do - [pele](https://en.wikipedia.org/wiki/Pel%C3%A9)**

# Please let me know if you see any thing wrong.
# Thanks!
