#!/usr/bin/env python
# coding: utf-8

# # Santander Value Prediction Challenge  
# 
# ## _**Objective: To predict the value of transactions for each potential customers**_
# 
# It seems to be a regression problem, where we have to predict a continuos variable.
# 

# ## _Let's import the required modules_

# In[ ]:


#base modules
import numpy as np
import pandas as pd

#visualization
import seaborn as sns
from matplotlib import pyplot as plt

#Scipy
import scipy

#scikit-learn
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

#LightGbm
import lightgbm

#Model validation
from sklearn.model_selection import cross_validate
from sklearn.metrics.scorer import make_scorer


# In[ ]:


from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')


# ## _Let's load the dataset_

# In[ ]:


# Let's import the dataset
train_data  = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")


# ## _Let's get started with understanding the data, it's time for some exploratory data analysis(EDA)_
# _Let's first see how the data looks like_

# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# In[ ]:


train_id = train_data['ID']
train_target = train_data['target']
test_id = test_data['ID']
del train_data['ID']
del train_data['target']
del test_data['ID']


# In[ ]:


print("The number of columns in train dataset are %i" % len(train_data.columns))
print("The number of rows in train dataset are %i" % len(train_data))


# In[ ]:


print("The number of columns in test dataset are %i" % len(test_data.columns))
print("The number of rows in test dataset are %i" % len(test_data))


# > Note: 
# * _The data is masked, we do not know what the columns really mean_
# * _The number of columns in train dataset are greater than the number of rows_
# * _The test data is almost 10 times larger than train dataset_ 
# 

# ## _Let's understand the distribution of the target variable_

# In[ ]:


plt.figure(figsize=(15,8))
sns.distplot(train_target)
plt.xlabel("Target",fontsize=14)
plt.title("Histogram-KDE plot of target variable",fontsize=14)


# _The target variable is highly skewed towards positively. So, let's apply some transformation to make it normally distributed._

# In[ ]:


plt.figure(figsize=(15,10))
sns.distplot(np.log(train_target))
plt.xlabel("Target",fontsize=14)
plt.title("Histogram-KDE of log transformation of the target varible",fontsize=14)


# _This looks much better than the actual target variable, but let's try out box cox transformation also._

# In[ ]:


box_cox_trans = scipy.stats.boxcox(train_target.values,lmbda=0.1)
box_cox_trans


# In[ ]:


plt.figure(figsize=(15,10))
sns.distplot(box_cox_trans)
plt.xlabel("Target",fontsize=14)
plt.title("Histogram-KDE plot of box-cox transformation of the target variable",fontsize=14)


# _This looks pretty much normally distributed and better._

# ## _Let's check if there are any columns with constant values through out_

# In[ ]:


train_data.nunique()[train_data.nunique(axis=0)==1]


# In[ ]:


constant_column_names = train_data.columns[train_data.nunique(axis=0)==1].tolist() #Saving the redundant column names 


# _There are 256 variables that has constant values. Therefore it can be dropped._

# ## _Let's now check for missing values_

# In[ ]:


train_data.isnull().values.any()


#  _There are'nt any missing value in the train data_

#  

# ## _Let's now understand the datatype of the variables_

# In[ ]:


combined_data.dtypes.value_counts()


# ## _Let's drop the columns with constant values_

# In[ ]:


train_data = train_data.drop(columns = constant_column_names,axis=1)
test_data = test_data.drop(columns = constant_column_names,axis=1)


# ## _Let's now use use PCA for dimensionality reduction the dimensions_

# In[ ]:


combined_data = pd.concat([train_data,test_data],axis=0)
print(combined_data.shape)


# In[ ]:


# combined_data = scale(combined_data)


# In[ ]:


pca = PCA(n_components=2000)


# In[ ]:


pca.fit_transform(combined_data)


# In[ ]:


cumsum_variance = np.cumsum(np.round(pca.explained_variance_ratio_,decimals=4)*100)


# In[ ]:


plt.plot(cumsum_variance)
plt.xlabel("Number of components")
plt.ylabel("Cumulative percentage of explained variance")
plt.title("Plot of explained variance in percentage")


#  _Looks like the dimensionality reduction using PCA is not that useful, so going to run my baseline model on the actual data with out dimensionality reduction._

# ## _Let's now try truncated SVD for reducing the dimensions_

# In[ ]:


tsvd = TruncatedSVD(n_components=2000)
tsvd.fit_transform(combined_data)


# In[ ]:


cumsum_variance = np.cumsum(np.round(tsvd.explained_variance_ratio_,decimals=4)*100)


# In[ ]:


cumsum_variance


# In[ ]:


plt.plot(cumsum_variance)
plt.xlabel("Number of components")
plt.ylabel("Cumulative percentage of explained variance")
plt.title("Plot of explained variance in percentage-tsvd")


# The t-svd performs slightly better than PCA in dimensionality reduction.

# ## _Let me now transform some of the variables as categorical based on their frequecies of occurence_

# In[ ]:


column_name = []
distinct_values = []
for col in combined_data.columns:
#     if combined_data[col].dtype == 'int64':
    column_name.append(col)
    distinct_values.append(combined_data[col].nunique())


# In[ ]:


plt.plot(sorted(distinct_values))


# In[ ]:


count=0
for col in combined_data.columns:
    if combined_data[col].nunique()<=500:
        combined_data[col]=combined_data[col].astype('category')
#         count=count+1


# In[ ]:


train_data = combined_data[:len(train_data)]
test_data = combined_data[len(train_data):]


# ## _Baseline model building using Lightgbm_

# In[ ]:


# Let's now define the root mean squared logarthmic error
def rmsle(y_pred,y_act):
    y_pred = scipy.special.inv_boxcox(y_pred,0.1)
    y_act = scipy.special.inv_boxcox(y_act,0.1)
    return np.sqrt(np.mean(np.square(np.log(y_pred+1)-np.log(y_act+1))))

scorer = make_scorer(rmsle)


# In[ ]:


model = lightgbm.LGBMRegressor()


# In[ ]:


rmsle_scores = cross_validate(model,train_data,box_cox_trans,scoring=scorer,cv=5)


# In[ ]:


rmsle_scores


# In[ ]:


model.fit(train_data,box_cox_trans)


# In[ ]:


test_pred = model.predict(test_data)


# In[ ]:


test_pred = scipy.special.inv_boxcox(test_pred,0.1)


# In[ ]:


dat = pd.DataFrame()


# In[ ]:


dat["ID"] = test_id
dat['target'] = test_pred
dat.to_csv("first_sub_2.csv",index=False)


# This kernel is all about understanding the data and building a baseline model, there are many things which needs to be fine tuned and a lot of work to be done in preprocessing. 
