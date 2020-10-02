#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
import seaborn as sns


import plotly.offline as py
import cufflinks as cf
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
get_ipython().run_line_magic('matplotlib', 'inline')


import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
import gc
gc.enable()
PATH = '../input/'


# ### Exploration Road Map:
# * Load Data
# * Take same insight of categorical variable
# * count na value for each column
# * Na value replace with zero
# * categorical lable convert into int
# * feature engineering
#     * ExtraTreesClassifier
#     * PCA
# * Histogram
# 

# In[2]:


## load all data
application_train = pd.read_csv(PATH+"/application_train.csv")
application_test = pd.read_csv(PATH+"/application_test.csv")
bureau = pd.read_csv(PATH+"/bureau.csv")
bureau_balance = pd.read_csv(PATH+"/bureau_balance.csv")
credit_card_balance = pd.read_csv(PATH+"/credit_card_balance.csv")
installments_payments = pd.read_csv(PATH+"/installments_payments.csv")
previous_application = pd.read_csv(PATH+"/previous_application.csv")
POS_CASH_balance = pd.read_csv(PATH+"/POS_CASH_balance.csv")


# In[3]:


print("application_train -  rows:",application_train.shape[0]," columns:", application_train.shape[1])
print("application_test -  rows:",application_test.shape[0]," columns:", application_test.shape[1])
print("bureau -  rows:",bureau.shape[0]," columns:", bureau.shape[1])
print("bureau_balance -  rows:",bureau_balance.shape[0]," columns:", bureau_balance.shape[1])
print("credit_card_balance -  rows:",credit_card_balance.shape[0]," columns:", credit_card_balance.shape[1])
print("installments_payments -  rows:",installments_payments.shape[0]," columns:", installments_payments.shape[1])
print("previous_application -  rows:",previous_application.shape[0]," columns:", previous_application.shape[1])
print("POS_CASH_balance -  rows:",POS_CASH_balance.shape[0]," columns:", POS_CASH_balance.shape[1])


# In[4]:


application_train.head()


# In[5]:


### First Exploration of Train data set
gc.collect()


# In[6]:


NAME_CONTRACT_TYPE = application_train.groupby(['TARGET','NAME_CONTRACT_TYPE']).size().unstack(level=0)
CODE_GENDER = application_train.groupby(['TARGET','CODE_GENDER']).size().unstack(level=0)
FLAG_OWN_CAR = application_train.groupby(['TARGET','FLAG_OWN_CAR']).size().unstack(level=0)
NAME_INCOME_TYPE = application_train.groupby(['TARGET','NAME_INCOME_TYPE']).size().unstack(level=0)
NAME_EDUCATION_TYPE = application_train.groupby(['TARGET','NAME_EDUCATION_TYPE']).size().unstack(level=0)
OCCUPATION_TYPE = application_train.groupby(['TARGET','OCCUPATION_TYPE']).size().unstack(level=0)
ORGANIZATION_TYPE = application_train.groupby(['TARGET','ORGANIZATION_TYPE']).size().unstack(level=0)


# In[7]:



plt.subplot(NAME_CONTRACT_TYPE.plot(kind='bar', stacked=True, title="NAME_CONTRACT_TYPE"))
plt.subplot(CODE_GENDER.plot(kind='bar', stacked=True, title="CODE_GENDER"))
plt.subplot(FLAG_OWN_CAR.plot(kind='bar', stacked=True, title="FLAG_OWN_CAR"))
plt.subplot(NAME_INCOME_TYPE.plot(kind='bar', stacked=True, title="NAME_INCOME_TYPE"))
plt.subplot(NAME_EDUCATION_TYPE.plot(kind='bar', stacked=True, title="NAME_EDUCATION_TYPE"))
plt.subplot(OCCUPATION_TYPE.plot(kind='bar', stacked=True, title="OCCUPATION_TYPE"))
plt.subplot(ORGANIZATION_TYPE.plot(kind='bar', stacked=True, title="ORGANIZATION_TYPE"))


# ####  NAME_CONTRACT_TYPE
# IN NAME_CONTRACT_TYPE cash loans more received compare to revolving loan
# #### CODE_GENDER
# Female get more loan as compare to male
# ####  FLAG_OWN_CAR
# person has not own car getting more loan approve as campare to own car
# #### NAME_INCOME_TYPE
# working employee approval loan ratio is high as compare to others
# #### NAME_EDUCATION_TYPE
# only secondary and higer education class getting loan
# #### OCCUPATION_TYPE
# laborers, sales staff , driver, manager geeting more approval loan as compare to hr staff, it staff and secretaries

# In[8]:


### extract the categorical column name from the applicaiton train dataset
category = application_train.dtypes


# In[9]:


cate = []
for i,value in enumerate(category):
    if value == 'object':
        cate.append(category.index[i])
print ("Categorical Variable", cate)


# ### Count na value for each column

# In[10]:


na_count = application_train.isnull().sum().sort_values(ascending=False)


# In[11]:


na_count.head(5)


# ### Now handling missing value 
# missing value replace with zero and check those column is important for our train data set 
# if the column is not important for train dataset we remove those column where we have getting more null value 

# In[12]:


application_train = application_train.fillna(0)


# In[13]:


application_train.isnull().sum().head()


# ### Feature Selection Method

# In[14]:


#before feature engineering we have convert all category variable into int
from sklearn import preprocessing
le = preprocessing.LabelEncoder()


# In[15]:


for i in application_train.columns:
    if application_train.dtypes[i] == 'object':
        le.fit(list(application_train[i].unique()))
        application_train[i] = le.transform(list(application_train[i].values))
print("Done")


# ### ExtraTreesClassifier
# Bagged decision trees like Random Forest and Extra Trees can be used to estimate the importance of features.
# In the example below we construct a ExtraTreesClassifier classifier for the Pima Indians onset of diabetes dataset. You can learn more about the ExtraTreesClassifier class in the scikit-learn API.

# In[16]:


array = application_train.values


# In[17]:


# load data
X = array[:,2:]
Y = array[:,1]
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
print (X.shape)
clf = ExtraTreesClassifier()
clf = clf.fit(X, Y)


# In[30]:


imp_value = clf.feature_importances_.round(4)
imp_col = application_train.columns[2:]
d = {'fea_name': imp_col, 'value':imp_value}
imp_fea = pd.DataFrame(data = d)
imp_fea = imp_fea.sort_values('value',ascending=False)
imp_fea = imp_fea.reset_index(drop=True)[:50]
imp_fea.fea_name[:5].values


# 
# ### below plot represent only 5 feature are important
# 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'DAYS_ID_PUBLISH',
#        'DAYS_REGISTRATION'

# In[31]:


imp_fea = imp_fea.set_index('fea_name')
imp_fea.plot(kind='barh',figsize=(15,10),title="Feature IMportant ")


# 
# ### Principal Component Analysis
# Principal Component Analysis (or PCA) uses linear algebra to transform the dataset into a compressed form.
# Generally this is called a data reduction technique. A property of PCA is that you can choose the number of dimensions or principal component in the transformed result.
# In the example below, we use PCA and select 3 principal components.
# Learn more about the PCA class in scikit-learn by reviewing the PCA API. Dive deeper into the math behind PCA on the Principal Component Analysis Wikipedia article.

# In[20]:


from sklearn.decomposition import PCA

# feature extraction
pca = PCA(n_components=120,svd_solver='full')
fit = pca.fit(X)
# summarize components
print("Explained Variance: ", fit.explained_variance_ratio_)
print(fit.components_)


# In[ ]:





# In[21]:


var=np.cumsum(np.round(fit.explained_variance_ratio_, decimals=3)*100)
var #cumulative sum of variance explained with [n] features


# ### Based on the plot below it's clear we should pick 3 features.
# 
# i have observed that base on pca plot we have to need require step increase our fearture
# 1. Normalise the data
# 2. Remove na column
# 3. Rest of column na value replace with (min,max,mean or else)
# 4. Remove the outlier
# 
# In pca and ExtraTreesClassifier extract same feature
# * pac = 3 feature
# * ExtraTreesClassifier = 5 feature above (0.025 value)

# In[22]:


plt.ylabel('% Variance Explained')
plt.xlabel('# of Features')
plt.title('PCA Analysis')
plt.ylim(30,100.5)
plt.style.context('seaborn-whitegrid')
plt.plot(var)


# ###  Analysis of all important column and outliers
# 

# In[37]:


fea_df = pd.DataFrame(application_train, columns=imp_fea.index[:20])


# In[53]:


fea_df.hist(figsize=(16,16))


# In[ ]:




