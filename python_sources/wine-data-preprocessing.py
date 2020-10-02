#!/usr/bin/env python
# coding: utf-8

# # Wine Reviews Introduction

# ![](https://253qv1sx4ey389p9wtpp9sj0-wpengine.netdna-ssl.com/wp-content/uploads/2015/01/GettyImages-80992924-700x461.jpg)

# ## Target 
# In this dataset, price estimation based on some features.

# ## Context
#    After watching Somm (a documentary on master sommeliers) I wondered how I could create a predictive model to identify wines through blind tasting like a master sommelier would. The first step in this journey was gathering some data to train a model. I plan to use deep learning to predict the wine variety using words in the description/review. The model still won't be able to taste the wine, but theoretically it could identify the wine based on a description that a sommelier could give. If anyone has any ideas on how to accomplish this, please post them!

# ## Content
# * Load and Check Data
# * Variable Description
# * Data Visualization
# * Outliers Detection
# * Missing Value Imputation
# * Missing Value Visualization
# - Fill Missing Value

# # Load and Check Data

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df=pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv")
df=df.iloc[:,1:]
df.head()


# In[ ]:


df.shape


# # Variable Description

# 1. country : The country that the wine is from.
# 2. description : A few sentences from a sommelier describing the wine's taste, smell, look, feel, etc.
# 3. designation : The vineyard within the winery where the grapes that made the wine are from.
# 4. points : The number of points WineEnthusiast rated the wine on a scale of 1-100 (though they say they only post reviews for wines that score &gt;=80)
# 5. price : The cost for a bottle of the wine.
# 6. province : The province or state that the wine is from
# 7. region_1 : The wine growing area in a province or state (ie Napa)
# 8. region_2 : Sometimes there are more specific regions specified within a wine growing area (ie Rutherford inside the Napa Valley), but this value can sometimes be blank
# 9. variety : The type of grapes used to make the wine (ie Pinot Noir)

# In[ ]:


df.info()


# * float64(1): price.
# * int64(2): Unnamed: 0, points.
# * object(8): country, description, designation, province, region_1, region_2, variety, winery.
# * number of rows: 150.930
# * number of columns: 11

# In[ ]:


df.dtypes


# In[ ]:


df.describe().T


# In[ ]:


def missing_value_table(df):
    missing_value = df.isna().sum().sort_values(ascending=False)
    missing_value_percent = 100 * df.isna().sum()//len(df)
    missing_value_table = pd.concat([missing_value, missing_value_percent], axis=1)
    missing_value_table_return = missing_value_table.rename(columns = {0 : 'Missing Values', 1 : '% Value'})
    cm = sns.light_palette("lightgreen", as_cmap=True)
    missing_value_table_return = missing_value_table_return.style.background_gradient(cmap=cm)
    return missing_value_table_return
  
missing_value_table(df)


# In[ ]:


df.region_2.unique().size


# In[ ]:


df.region_1.unique().size


# In[ ]:


df.designation.unique().size


# In[ ]:





# # Data Visualization
# 

# In[ ]:


plt.figure(figsize=(15,6))
df.dtypes.value_counts().plot.barh();


# In[ ]:


fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(15,7))
sns.kdeplot(df["points"],shade=True,ax=ax[0]);
sns.kdeplot(df["price"],shade=True,ax=ax[1]);


# In[ ]:


sns.pairplot(df)
sns.set(style="ticks", color_codes=True)


# In[ ]:


corr = df.corr()
plt.figure(figsize=(15,7))
sns.heatmap(corr, annot=True);


# In[ ]:


country = df.country.value_counts()[:10]
plt.figure(figsize=(15,7))
sns.barplot(x=country.index, y=country.values, palette="dark")
plt.xticks(rotation='vertical')
plt.ylabel('Frequency')
plt.xlabel('Country')
plt.title('top 10 countries',color = 'darkblue',fontsize=15);


# In[ ]:


f, ax = plt.subplots(figsize=(15,7))
sns.despine(f, left=True, bottom=True)

sns.scatterplot(x="points", y="price",
                hue="price",
                palette="ch:r=-.2,d=.3_r",
                sizes=(1, 8), linewidth=0,
                data=df, ax=ax);


# # Outlier Data

# In statistics, an outlier is a data point that differs significantly from other observations.
# 
# Outlier is smaller than Q1-1.5(Q3-Q1) and higher than Q3+1.5(Q3-Q1) .
# 
# (Q3-Q1) = IQR (INTER QUARTILE RANGE)
# 
# Q3 = Third Quartile(%75)
# Q1 = First Quartile(%25)
# train and test separation

# In[ ]:


num_data=pd.DataFrame(df.dtypes[df.dtypes!="object"]).index


# In[ ]:


count = 0
fig, ax =plt.subplots(nrows=1,ncols=2, figsize=(20,8))
sns.boxplot(df[num_data[0]],ax=ax[0])
sns.boxplot(df[num_data[1]],ax=ax[1]);


# In[ ]:


lower_and_upper = {}

for col in num_data:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = 1.5*(q3-q1)
    
    lower_bound = q1-iqr
    upper_bound = q3+iqr
    
    lower_and_upper[col] = (lower_bound, upper_bound)
    df.loc[(df.loc[:,col]<lower_bound),col]=lower_bound*0.75
    df.loc[(df.loc[:,col]>upper_bound),col]=upper_bound*1.25
    
    
lower_and_upper


# In[ ]:


count = 0
fig, ax =plt.subplots(nrows=1,ncols=2, figsize=(20,8))
sns.boxplot(df[num_data[0]],ax=ax[0])
sns.boxplot(df[num_data[1]],ax=ax[1]);


# In[ ]:





# # Missing Value

# ## Missing Value Visualization

# In[ ]:


import missingno as msno


# In[ ]:


df.isnull().sum()


# In[ ]:


msno.bar(df);


# In[ ]:


msno.matrix(df);


# In[ ]:


msno.heatmap(df);


# In[ ]:





# ## Fill Missing Value

# * Mode Imputation

# In[ ]:


df["country"].fillna(df["country"].mode()[0],inplace=True)
df["province"].fillna(df[df["country"]=="US"]["province"].mode()[0],inplace=True)


# In[ ]:


list_columns=["designation","region_1","region_2","variety","winery"]
for i in df[list_columns]:
    df[i].fillna(df[i].mode()[0],inplace=True)


# * KNN imputation

# In[ ]:


from sklearn.impute import KNNImputer


# In[ ]:


knn_imputer=KNNImputer()
df["price"]=knn_imputer.fit_transform(df[["price"]])


# In[ ]:


msno.bar(df);


# In[ ]:


df.isnull().sum()

