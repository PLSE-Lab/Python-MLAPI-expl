#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


data = pd.read_csv("../input/df1_backup.csv")
len(data)


# In[ ]:


tt = data[data["Role"] == 4]
tt = tt[["Number of Individual Goals", "Individual Goals Achieved", "% goal"]]
tt
data["Number of Individual Goals"].unique()


# In[ ]:


data.head()


# In[ ]:


data.columns


# * Features such as 'Employee ID',  'Number of Individual Goals', 'Individual Goals Achieved' can lead to data leakage hence not considering them into training features.
# * Features such as 'day' is not of any use.

# In[ ]:


train_features = ['Gender', 'Nationality', 'Birth Year', 'Prior Industry',
       'Tech Aptitude', 'Linguistic Aptitude', 'Finance Aptitude',
       'People Skills', 'Decision Skills', 'Year', 'Intake', 'Division',
       'Start-End', 'Group Size', 'Role', 'Neuroticism',
       'Extraversion', 'Openness to Experience', 'Agreeableness',
       'Conscientiousness', 'year', 'Month', 'Age_test', 'Age_joining']
target_feature = ["% goal"]


# In[ ]:


# for i,col in enumerate(X.columns):
#     print(str(i+1) + ". " + col + ":\n" +  str(data[col].unique()) + "\n\n")


# In[ ]:


X, Y =  data[train_features], data[target_feature]
X.head()


# In[ ]:


data["year"].unique()


# #Just testing K best for features importance

# In[ ]:


from sklearn.feature_selection import SelectKBest, chi2, f_regression


# In[ ]:


X = X.astype('category')
X = X.apply(lambda x: x.cat.codes)


# In[ ]:


X.head()


# In[ ]:


Y.head()


# In[ ]:


Y = np.ravel(Y)


# In[ ]:


X_kbest = SelectKBest(score_func=f_regression, k=20)
X_kbest.fit_transform(X, Y)


# In[ ]:


dfscores = pd.DataFrame(X_kbest.pvalues_)


# In[ ]:


dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Features','Score']  #naming the dataframe columns
featureScores = featureScores.sort_values(by = "Score", ascending=False)


# In[ ]:


#featureScores.index = featureScores.Features
#featureScores = featureScores.drop("Features", axis =1)


# ## Feature importance of different features related to target variable.

# In[ ]:


plt.figure(figsize=(15,7))
sns.barplot(y=featureScores.Features,x=featureScores.Score, orient="h", palette="magma")
plt.title('Most Important Features',color = 'c',fontsize=15)


# In[ ]:



   


# From Above the we know the categorical and Numerical features as.

# Categorical Features List:
# 

# In[ ]:


cat_featrs = ['Gender', 'Nationality', 'Birth Year', 'Prior Industry','Year','Intake', 'Division','Start-End', 
              'Group Size', 'Role', 'Age_test', 'Age_joining', "year"]
num_featrs = ['Tech Aptitude', 'Linguistic Aptitude', 'Finance Aptitude', 'People Skills', 'Decision Skills', 'Neuroticism',
       'Extraversion', 'Openness to Experience', 'Agreeableness','Conscientiousness']


# In[ ]:


data.info()


# In[ ]:





# ## Analysing Target Feature "% goal"

# In[ ]:


#histogram

sns.distplot(data['% goal'], bins = 15, color = 'r');


# In[ ]:


#skewness and kurtosis
print("Skewness: %f" % data['% goal'].skew())
print("Kurtosis: %f" % data['% goal'].kurt())


# > We can see target feature is skewed between 60 - 90 percent target completeion.

# ## Checking dependency of target feature towards gender of individuls.

# In[ ]:


data.describe()
data['Gender'].unique()


# In[ ]:





# In[ ]:


#help(pd.Series.unique)
g_count = data['Gender'].value_counts().rename({"M": "Male", "F": "Female"})
g_count


# In[ ]:


bar = g_count.plot(kind = 'barh', color=["b","r"], fontsize=15)


# > Examining population distribution

# In[ ]:


f, ax = plt.subplots(figsize=(6, 6))
pie = g_count.plot(kind='pie', colormap="RdBu", fontsize=15, legend = True, label="Gender Proportion" )

# fig, ax = plt.subplots(figsize=(6,6))
# myaximage = ax.imshow(aspect='auto',
#                       extent=(20, 80, 20, 80),
#                       alpha=0.5)
# g_count.plot(kind='pie', colormap="RdBu")
# plt.show()


# In[ ]:


plt.figure()
sns.set_style("whitegrid")
ax = sns.boxplot(x="Gender", y='% goal', hue="Gender", data=data[["Gender", '% goal']], palette="brg")


# > We can see that goals achieved are not having any good dependency over gender. As both gender are having similar avg. performances.

# In[ ]:


#correlation matrix for categorical features.
data1 = pd.concat([X, data["% goal"]], axis=1)
corrmat = data1.corr()
f, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(corrmat, annot=True, fmt=".2f", annot_kws={'size': 8}, square=True);


# In[ ]:


#correlation matrix for categorical features.
data1 = pd.concat([X[cat_featrs], data["% goal"]], axis=1)
corrmat = data1.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, annot=True, fmt=".2f", annot_kws={'size': 8}, square=True);


# > We can see 

# In[ ]:


#correlation matrix for categorical features.
data1 = pd.concat([X[num_featrs], data["% goal"]], axis=1)
corrmat = data1.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, annot=True, fmt=".2f", cmap = "magma", annot_kws={'size': 10}, square=True);


# In[ ]:


#Plotting features having maximum correlations with target feature.
data3 = pd.concat([X, data["% goal"]], axis=1)
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, '% goal')['% goal'].index
cm = np.corrcoef(data3[cols].values.T)
sns.set(font_scale=1.25)
f, ax = plt.subplots(figsize=(12, 9))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',cmap = "Greys_r", annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sn = sns.set()
fig = plt.figure()
pair = sns.pairplot(data = data[['Tech Aptitude', 'Linguistic Aptitude', 'Finance Aptitude',
       'People Skills', 'Decision Skills',"Prior Industry",
       '% goal']], hue ="Prior Industry", diag_kind = 'kde', palette = "plasma")


# In[ ]:


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, cross_val_score
X_R = X.copy()
#X_R = X_R.drop(['Role', 'Number of Individual Goals', 'Individual Goals Achieved'], axis=1)
#X_train,X_test,y_train,y_test = train_test_split(X, Y,test_size=0.3)
X_train,X_test,y_train,y_test = train_test_split(X_R, Y,test_size=0.3)
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)


# In[ ]:


sel = (RandomForestRegressor(n_estimators = 100))
sel.fit(X_train, y_train)
#sel.score(X_test, y_test)


# In[ ]:


feature_import = pd.Series(sorted(zip(X_R.columns,map(lambda x: round(x, 4), sel.feature_importances_)), reverse=True, key = lambda x:x[1]))             
feature_import =  pd.DataFrame.from_records(list(feature_import), columns = ["Features", "Score"])
feature_import.index = feature_import.Features
plt.figure(figsize=(15,7))
sns.barplot(y=feature_import.Features,x=feature_import.Score, orient="h")
#plt.xticks(rotation=45)
plt.title('Most Important Features',color = 'm',fontsize=15)


# ## From above analysis we have found the list of important features those are having good correlation with our target feature.

# 

# In[ ]:


imp_feature_list = ['Tech Aptitude', 'Linguistic Aptitude', 'Finance Aptitude', 'People Skills', 'Decision Skills', 'Neuroticism',
       'Extraversion', 'Openness to Experience', 'Agreeableness','Conscientiousness', "Nationality", 'Age_joining',"Prior Industry"]


# * > Let's see the distribution of % goals achieved by different group of roles.

# In[ ]:


plt.figure(figsize=(15,7))
sns.violinplot(x="Role", y="% goal", hue="Gender", data=data, palette="viridis")
plt.show()


# > From above Violin plot we can see goals achieved by different departments is different, people of department 0,1,2 have achieved goals between 50 - 90% where as there are people in 3rd and 4th department those were not able to achieve goals even 20 percent. 

# Let's see scatter plot of % goals and role

# Coverting % Goal into categorical values and drawing it's stacked bar plot.

# In[ ]:



#############Stack plot later.
# labels = ["<=10", "<=20", "<=30", "<=40", "<=50", '<=60', "<=70", "<=80", "<=90"]
# bins = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# category = pd.cut(data["% goal"],bins = bins, labels=labels)
# data['Goal range'] = category

# stacke_plt = data[["Goal range", "% goal", "Role"]].copy()
# stacke_plt.index = stacke_plt.


# In[ ]:





# In[ ]:


plt.figure(figsize=(15,7))
sns.stripplot(x="Role", y="% goal", hue="Gender", data=data, palette="RdPu")
plt.show()


# From above it's clear that features such as below are having good corre

# 

# In[ ]:


pd.series(sel.estimator_,feature_importances_,.ravel()).hist()


# In[ ]:


pd.series(sel.estimator_,feature_importances_,.ravel()).hist()


# In[ ]:





# In[ ]:




