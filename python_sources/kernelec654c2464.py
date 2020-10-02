#!/usr/bin/env python
# coding: utf-8

# In[3]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

directory = "../input/"
print(os.listdir(directory))
os.getcwd()
# Any results you write to the current directory are saved as output.


# In[4]:


dfs = pd.read_excel(directory + "zillow_data_dictionary.xlsx", sheet_name=None)
dfs['Data Dictionary'].sort_values(by="Feature")


# ### Loading Training Data: Housing ID and Logerror 

# In[5]:


train17 = pd.read_csv(directory + "train_2017.csv")
train17.head(5)


# ### Loading House Property Data

# In[6]:


prop17=pd.read_csv(directory + "properties_2017.csv")
prop17.head(5)


# ### Merge the Training data with House Property Data

# In[7]:


df17 = pd.merge(train17, prop17, on=["parcelid"], how="left", indicator = True)


# In[8]:


print("Dataset Size is %d "%(df17.shape[0]))


# In[9]:


'''
Function to display Dataset Propety
'''
def DisplayT(DF, target_feature):
    counts=DF.count()
    nulls=DF.apply(lambda col: col.isnull().sum())
    pct_nulls = np.round(nulls / DF.shape[0],5)
    uniques=DF.apply(lambda col: [col.unique()])
#    count_uniques = DF.apply(lambda col: len([col.unique()]))
    count_uniques=[len(lst) for lst in uniques.iloc[0,:].values]

    dtypes=DF.dtypes
    #tbl=pd.DataFrame(list(zip(uniques.iloc[0,:].values)), columns=["uniqueness"])
    
    tbl=pd.DataFrame(list(zip(DF.columns,counts, nulls,pct_nulls*100 ,dtypes, uniques.iloc[0,:].values,count_uniques,  DF.corr()[target_feature])), 
                     columns=["feature","non_empty_counts","nulls","pct_nulls","dtypes", "uniqueness", "count_unique","corr_betweem_target" ])
    tbl=tbl.set_index("feature")
    return(tbl)


# ### This dataset contains 54 features with with 77613 instances.
# #### The following shows some information about the features
# - non_empty_counts: Number of Non-empty Cells of the specific feature
# - nulls: Number of Null cells of the specific feature
# - pct_nulls: Percent of null cells of the specific feature
# - dtype: Type of the feature
# - uniqueness: Sample of unique values of the specific feature
# - count_unique: Number of unique values of the specific feature
# - corr_betweem_target: Correlation value between the feature and Target Value

# In[10]:


adf=DisplayT(df17, "logerror")
print(adf.shape)
display(adf.sort_values(by="nulls", ascending = False))


# ### Missing Values
# #### There are quite amount of missing values in some of the features, the following features with over 50% of missing are going to exclude from the dataset

# In[11]:


display(adf[["pct_nulls", "dtypes"]][adf.pct_nulls>50].sort_values(by="pct_nulls", ascending = False))


# #### The following features (<50% Missing) is going to be stayed for further analysis

# In[12]:


display(adf[["pct_nulls", "dtypes"]][adf.pct_nulls<=50].sort_values(by="pct_nulls", ascending = False))


# ### Correlation on "float" features

# In[13]:


import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

features=adf[["pct_nulls", "dtypes"]][adf.pct_nulls<=50][adf["dtypes"] == "float64"].sort_values(by="pct_nulls", ascending = False).index

'''
Plotting Correlation 
'''
def plotCorr(DF,features,Target_Var,number_of_columns, width_of_plot, depth_of_plot):
    c=number_of_columns
    w=round(len(features)/c) 

    fig = plt.figure(figsize=(width_of_plot,depth_of_plot))
    cnt=0
    for row in range(w):
        for col in range(0,2):

            try:
                features[cnt]
                plt.subplot2grid((w,c),(row,col))
                sns.regplot(x=features[cnt], y=Target_Var, data=DF)
                plt.xticks(rotation=45)
                cnt+=1
            except:
                cnt+=1
                pass

plotCorr(df17, features,"logerror", 2,15,100)


# In[ ]:





# In[ ]:





# In[ ]:




