#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_data = pd.read_csv("/kaggle/input/fmcgfast-moving-consumer-goods/train1.csv")


# In[ ]:


test_data = pd.read_csv("/kaggle/input/fmcgfast-moving-consumer-goods/test1.csv")


# importing required libraries

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, norm
from scipy import stats
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# getting the discriptive information about the train data

# In[ ]:



train_data.head(10) 


# droping ['id'] column from train data

# In[ ]:


train_data = train_data.loc[:, ~train_data.columns.str.contains('^Unnamed')]


# In[ ]:


print(train_data.keys())


# ['PROD_CD', 'SLSMAN_CD', 'PLAN_MONTH', 'PLAN_YEAR', 'TARGET_IN_EA','ACH_IN_EA']
# are the columns name in train data

# In[ ]:



train_data.dtypes #datatype of colunms in train data


# #ENCODING  (using this we have remove string part and kept integer in following colunms for EDA and better insights )

# In[ ]:


train_data['PROD_CD'] = train_data['PROD_CD'].str.replace(r'\D', '').astype(int)


# In[ ]:


train_data['SLSMAN_CD'] = train_data['SLSMAN_CD'].str.replace(r'\D', '').astype(int)


# In[ ]:


train_data['TARGET_IN_EA'] = train_data['TARGET_IN_EA'].str.replace(r'\D', '').astype(int)


# In[ ]:


train_data['ACH_IN_EA'] = train_data['ACH_IN_EA'].str.replace(r'\D', '').astype(int)


# In[ ]:


train_data.dtypes


# In[ ]:


test_data.dtypes


# In[ ]:


test_data = test_data.loc[:, ~test_data.columns.str.contains('^Unnamed')]


# CHECK FOR CORRELATION

# In[ ]:


corr=train_data.corr()
sns.heatmap(corr,annot=True)


# In[ ]:


corr


# by heatmap we can see that target and achivement have good correlation that is 0.719321
# and month and year do not have good correlation that is -0.98

# GROUPBY IS USED FOR GRUOPING OF PARTICULAR PARAMETERS TO GET INSIGHTS OF PRODUCT AND SALESMAN DATA ACCORDING TO THEIR FREQUENCY IN A PERTICULAR MONTH AND YEAR 
# 

# In[ ]:


train_data.groupby(['SLSMAN_CD','PLAN_MONTH'])['SLSMAN_CD'].count()


# In[ ]:


train_data.groupby(['PROD_CD','PLAN_MONTH'])['PROD_CD'].count()


# In[ ]:


train_data.groupby(['SLSMAN_CD','PLAN_MONTH','PLAN_YEAR'])['SLSMAN_CD'].count()


# STANDARD DEVIATION
# 

# In[ ]:


np.std(train_data) 


# #VARIANCE
# 

# In[ ]:


np.var(train_data)


# In[ ]:


skew(train_data)


# # KURTOSIS
#  

# In[ ]:


kurtosis(train_data)


# HISTOGRAM
# 

# In[ ]:


plt.hist(train_data['PROD_CD']);plt.title('Histogram of PROD_CD'); plt.xlabel('PROD_CD'); plt.ylabel('Frequency')


# In[ ]:


plt.hist(train_data['SLSMAN_CD'], color = 'coral');plt.title('Histogram of SLSMAN_CD'); plt.xlabel('SLSMAN_CD'); plt.ylabel('Frequency')


# In[ ]:


plt.hist(train_data['TARGET_IN_EA'], color= 'brown');plt.title('Histogram of TARGET_IN_EA'); plt.xlabel('TARGET_IN_EA'); plt.ylabel('Frequency')


# In[ ]:


plt.hist(train_data['ACH_IN_EA'], color = 'violet');plt.title('Histogram of ACH_IN_EA'); plt.xlabel('ACH_IN_EA'); plt.ylabel('Frequency')


# BOXPLOT
# 
# 
# 
# 
# 
# 

# In[ ]:


sns.boxplot(train_data["PROD_CD"])


# In[ ]:


sns.boxplot(train_data["SLSMAN_CD"])


# In[ ]:


sns.boxplot(train_data["PLAN_MONTH"])


# In[ ]:


sns.boxplot(train_data["TARGET_IN_EA"])


# SCATTERPLOT

# In[ ]:


sns.scatterplot(x='PROD_CD', y='ACH_IN_EA', data=train_data).set_title('Scatterplot of ACH_IN_EA & PROD_CD')


# In[ ]:


sns.scatterplot(x='SLSMAN_CD', y='ACH_IN_EA', data=train_data).set_title('Scatterplot of ACH_IN_EA & SLSMAN_CD')


# In[ ]:


sns.scatterplot(x='PLAN_MONTH', y='ACH_IN_EA', data=train_data).set_title('Scatterplot of ACH_IN_EA & PLAN_MONTH')


# In[ ]:


sns.scatterplot(x='PLAN_YEAR', y='ACH_IN_EA', data=train_data).set_title('Scatterplot of ACH_IN_EA & PLAN_YEAR')


# In[ ]:


sns.scatterplot(x='TARGET_IN_EA', y='ACH_IN_EA', data=train_data).set_title('Scatterplot of ACH_IN_EA & TARGET_IN_EA')


# COUNTPLOT

# In[ ]:


sns.countplot(train_data["PROD_CD"])


# In[ ]:


sns.countplot(train_data["SLSMAN_CD"])


# In[ ]:


sns.countplot(train_data["PLAN_MONTH"])


# In[ ]:


sns.countplot(train_data["PLAN_YEAR"])


# UNIQUE VALUES and COUNTS
# 

# In[ ]:


train_data.PROD_CD.unique()               


# In[ ]:


train_data.PROD_CD.value_counts()                    


# In[ ]:


train_data.SLSMAN_CD.unique()


# In[ ]:


train_data.SLSMAN_CD.value_counts()


# In[ ]:


train_data.PLAN_YEAR.unique()


# In[ ]:


train_data.PLAN_YEAR.value_counts()


# In[ ]:


train_data.PLAN_MONTH.unique()


# In[ ]:


train_data.PLAN_MONTH.value_counts()


# In[ ]:


train_data.TARGET_IN_EA.unique()


# In[ ]:


train_data.TARGET_IN_EA.value_counts()


# In[ ]:


train_data.ACH_IN_EA.unique()


# In[ ]:


train_data.ACH_IN_EA.value_counts()


# HERE IS SOME INSIGHTS ABOUT HOW MUCH SALESMAN'S TARGETS AND ACHIEVEMENTS

# In[ ]:


train_data.plot(x="ACH_IN_EA",y="SLSMAN_CD")


# In[ ]:


train_data.plot(x="TARGET_IN_EA",y="SLSMAN_CD")


# we can finally plot the targets and acheivements  made for each salesman and products for each month:

# In[ ]:


fig,ax= plt.subplots(figsize =(15,7))
fig= train_data.groupby(['PROD_CD','PLAN_MONTH']).count()['ACH_IN_EA'].unstack().plot(ax=ax)


# In[ ]:


fig,ax= plt.subplots(figsize =(15,7))
fig= train_data.groupby(['SLSMAN_CD','PLAN_MONTH'])['ACH_IN_EA'].count().unstack().plot(ax=ax)


# In[ ]:


fig,ax= plt.subplots(figsize =(15,7))
fig= train_data.groupby(['PROD_CD','PLAN_MONTH']).count()['TARGET_IN_EA'].unstack().plot(ax=ax)


# In[ ]:


fig,ax= plt.subplots(figsize =(15,7))
fig= train_data.groupby(['SLSMAN_CD','PLAN_MONTH'])['TARGET_IN_EA'].count().unstack().plot(ax=ax)


# #ANALYIZING MORE INSGHTS
# 

# #GETTING THE ACCURING FREQUENCY OF EACH PRODUCTS AND SALESMAN IN BAR CHART FOR EACH MONTH AND YEAR 

# In[ ]:


pd.crosstab(train_data.PROD_CD,train_data.PLAN_MONTH).plot(kind="bar")


# In[ ]:


pd.crosstab(train_data.PROD_CD,train_data.PLAN_YEAR).plot(kind="bar")


# In[ ]:


pd.crosstab(train_data.SLSMAN_CD,train_data.PLAN_MONTH).plot(kind="bar")


# In[ ]:


pd.crosstab(train_data.SLSMAN_CD,train_data.PLAN_YEAR).plot(kind="bar")


# #distribution plot
# 

# #By distribution plot we can see how much our data is normally distributed

# In[ ]:


sns.distplot(train_data['PROD_CD'], fit=norm, kde=False)


# In[ ]:


sns.distplot(train_data['SLSMAN_CD'], fit=norm, kde=False, color = 'coral')


# In[ ]:


sns.distplot(train_data['PLAN_MONTH'], fit=norm, kde=False, color = 'skyblue')


# In[ ]:


sns.distplot(train_data['PLAN_YEAR'], fit=norm, kde=False, color = 'orange')


# In[ ]:


sns.distplot(train_data['TARGET_IN_EA'], fit=norm, kde=False, color = 'brown')


# VISUALISATION OF DENSITY DISTRIBUTION OF TARGETS AND ACHEIVEMENT AND COMPARIOSION BETWEEN THEM

# In[ ]:


sns.kdeplot(train_data['TARGET_IN_EA'],shade = True, bw = .5, color = "red")


# In[ ]:


import seaborn as sns


# In[ ]:


train_data["ACH_IN_EA"].describe()


# In[ ]:


sns.kdeplot(train_data['ACH_IN_EA'],shade = True, bw = .5, color = "BLUE")


# VISIUALIZING DISTRIBUTION OF DATA ACCORDING TO THE MONTH

# In[ ]:


sns.violinplot(y=train_data['PROD_CD'],x=train_data['PLAN_MONTH'])


# In[ ]:


sns.violinplot(y=train_data['SLSMAN_CD'],x=train_data['PLAN_MONTH'])


# In[ ]:


sns.violinplot(y=train_data['TARGET_IN_EA'],x=train_data['PLAN_MONTH'])


# In[ ]:


sns.violinplot(y=train_data['ACH_IN_EA'],x=train_data['PLAN_MONTH'])


# GETTING EXACTLY WHERE SALES MA COMPLETED THEIR TARGET OR ACHIEVED THE TARGET OF EVERY PRODUCT IN DATASET 

# In[ ]:


target=list(train_data.TARGET_IN_EA)


# In[ ]:


achiv=list(train_data.ACH_IN_EA)


# In[ ]:


yn=[]     


# In[ ]:


for x in range(22646):
    if(target[x]<=achiv[x]):
        yn.append(1)
    else:
        yn.append(0)


# In[ ]:


train_data['result'] = yn


# HERE IT SHOWING HOW MUCH TARGETS ARE ACHIEVED OR NOT ACHIEVED

# In[ ]:


pd.crosstab(train_data.result,train_data.PLAN_YEAR).plot(kind="bar")


# In[ ]:


pd.crosstab(train_data.result,train_data.PLAN_MONTH).plot(kind="bar")


# In[ ]:


prod = np.array(train_data['PROD_CD'])


# In[ ]:


salesman = np.array(train_data['SLSMAN_CD'])


# In[ ]:


month = np.array(train_data['PLAN_MONTH'])


# In[ ]:


year = np.array(train_data['PLAN_YEAR'])


# In[ ]:


target = np.array(train_data['TARGET_IN_EA'])


# In[ ]:


achieved = np.array(train_data['ACH_IN_EA'])


# # Normal Probability distribution 
# As we know data is not normally Distributed so we can process and form the Normal Probability Distribution of data column wise.
# 
# 
# 

# # ACHIEVED

# In[ ]:


x_ach = np.linspace(np.min(achieved), np.max(achieved))


# In[ ]:


y_ach = stats.norm.pdf(x_ach, np.mean(x_ach), np.std(x_ach))


# In[ ]:


plt.plot(x_ach, y_ach,); plt.xlim(np.min(x_ach), np.max(x_ach));plt.xlabel('achieved');plt.ylabel('Probability');plt.title('Normal Probability Distribution of achieved')


# # Product_code

# In[ ]:


x_prod = np.linspace(np.min(prod), np.max(prod))


# In[ ]:


y_prod = stats.norm.pdf(x_prod, np.mean(x_prod), np.std(x_prod))


# In[ ]:


plt.plot(x_prod, y_prod, color = 'coral'); plt.xlim(np.min(x_prod), np.max(x_prod));plt.xlabel('prod_cd');plt.ylabel('Probability');plt.title('Normal Probability Distribution of prod_cd')


# # salesman_code
# 

# In[ ]:


x_sale = np.linspace(np.min(salesman), np.max(salesman))
y_sale = stats.norm.pdf(x_sale, np.mean(x_sale), np.std(x_sale))
plt.plot(x_sale, y_sale, color = 'coral'); plt.xlim(np.min(x_sale), np.max(x_prod));plt.xlabel('Sale_cd');plt.ylabel('Probability');plt.title('Normal Probability Distribution of sales_cd')


# # target
# 

# In[ ]:


x_target = np.linspace(np.min(target), np.max(target))
y_target = stats.norm.pdf(x_target, np.mean(x_target), np.std(x_target))
plt.plot(x_target, y_target, color = 'coral'); plt.xlim(np.min(x_target), np.max(x_target));plt.xlabel('target');plt.ylabel('Probability');plt.title('Normal Probability Distribution of target')


# # Unsquish the pie.
# 

# In[ ]:


train_data['PLAN_MONTH'].value_counts().head(10).plot.pie()
train_data['PLAN_YEAR'].value_counts().head(10).plot.pie()
plt.gca().set_aspect('equal')


# By all the plots and graphs we have come to conclusion that highest achivement is 232,000 
# by saleman code i.e SLSMAN_CD 94 in month of november 2019 product_CD 31 
# 

# In[ ]:


X = train_data.iloc[:,:6]  #independent columns
y = train_data.iloc[:,-1]    #target column i.e price range


# #apply SelectKBest class to extract top 10 best features
# 

# In[ ]:


bestfeatures = SelectKBest(score_func=chi2, k=5)
fit = bestfeatures.fit(X,y)


# In[ ]:


dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)


# concat two dataframes for better visualization

# In[ ]:



featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['imp','importance']  #naming the dataframe columns


# In[ ]:


featureScores


# By using feature selection we get the importance of particular column in data.
# so by feature selection we get that product_CD and Achivement is having more importance respectively followed by target
# column and column with very less importance can be drop.
# 
# 
