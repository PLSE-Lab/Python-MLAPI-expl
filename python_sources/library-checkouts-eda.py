#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))


# In[ ]:


# https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtypes
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        #else: df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


# In[ ]:


## Note: Kernel unstable when using all data
df = reduce_mem_usage(pd.read_csv("../input/checkouts-by-title.csv",
                 nrows=15123456,
                 keep_date_col=True,infer_datetime_format=True
                 ,parse_dates=[["CheckoutYear","CheckoutMonth"]]).rename(columns={"CheckoutYear_CheckoutMonth":"CheckoutDate"}).set_index("CheckoutDate"))

print(df.shape)
df.head()


# In[ ]:


### assuming all publication years are 4 digit years, we could parse with this. 
### It's not, so we get erros. We could coerce, but let's just leave them as numbers for now:
# df.PublicationYear = pd.to_datetime(df.PublicationYear.str.replace(r"[^0-9]",''),format="%Y")

df.PublicationYear = df.PublicationYear.str.replace(r"[^0-9]",'')#.astype(int) # astype int fails due to nans. Parsing as floats would give us "ugly" .0 per number


# In[ ]:


df.nunique()


# Let's get the most popular items over all the time-period covered:
# * Ideally we would also normalize titles' text and versions of the same product, but I'll ignore for now.
# * Could also look at distinct counts, (i.e not just # checkouts, but count of distinct months when it was checked out). 
# * We could also look at the quarterly level. Or even better, look at items as a % of total checkouts for that month per library. Or reaggregate per quarter

# In[ ]:


df["title_sum_total"] = df.groupby("Title")["Checkouts"].transform("sum")
df["title_MonthCounts_total"] = df.groupby("Title")["Checkouts"].transform("count")

df.describe()


# * Unsurprisingly, the distribution is very biased with a super long tail, and a "bump" for the very popular items! 

# In[ ]:


df["title_sum_total"].hist(bins=30)


# In[ ]:


df["title_MonthCounts_total"].hist()


# ## Add total library checkouts for the library per month.
# * Relevant: 
#     * Distinct checkouts (proxy for # visitors)
#     * Checkouts per MaterialType (books vs music..) 

# In[ ]:


df["total_checkouts_all_monthly_sum"] = df.groupby(["CheckoutYear","CheckoutMonth"])["Checkouts"].transform("sum")

df["total_checkouts_ByMaterial_monthly_sum"] = df.groupby(["MaterialType","CheckoutYear","CheckoutMonth"])["Checkouts"].transform("sum")


# In[ ]:


# we see a lot of overlap for some "mixed" categories
df.groupby("MaterialType")["total_checkouts_ByMaterial_monthly_sum"].plot()


# In[ ]:


df["total_checkouts_all_monthly_sum"].plot()


# # Let's filter out the "less popular" items. 
# * While interesting for many uses cases (and specific users, e.g. research textbook), we want to get at least moderately popular items
# 
# * We could also look at subsets by subject: i.e get total monthly checkouts per publisher (for popular subjects or publishers) , then  see which checked-out items have a larger proportional share of this. 
# 
# * the number of distinct months an item was checked out in, is also important for building a model to predict monthly checkouts per item: Our model will "see" each item only once per month, so # occurences is going to affect this strongly! 

# In[ ]:


print("Original shape: %i" %(df.shape[0]))
df = df.loc[(df["title_MonthCounts_total"]>8) & (df["title_sum_total"]>40)]
print("We keep items that were checked out at least once in 8+ different months, AND checked out at least 30 times in aggregate")
print("New shape: %i" %(df.shape[0]))


# In[ ]:


# Describe new counts/sum distribution for our "popular" subset
df.describe()


# # Let's look at some of the popular items
# 
# * We find a data issue: "<Unknown Title>" is very common (although we could probably create a new ID using the concatenation of subject+publisher, we may wish to drop it for now, as it creates outliers)
#     
#     * It seems that the popular titles are mostly not what we might expect! This could be a data issue in our process... or just the result of popular romance and historical novels being popular or taken out in batch :) 

# In[ ]:


df.nlargest(10, columns=["title_MonthCounts_total"])


# In[ ]:


df.nlargest(15, columns=["total_checkouts_all_monthly_sum"])


# * ## Drop *Unknown Title* for simplicity
#     * Ideally we would impute this or create a hummy ID. Easier to ignore for now

# In[ ]:


df = df.loc[df.Title !="<Unknown Title>"]
print(df.shape)


# In[ ]:


df.nlargest(10, columns=["title_MonthCounts_total"])


# In[ ]:


df.tail()


# In[ ]:


df.columns


# In[ ]:


df.to_csv("LibraryCheckouts_WithLeaks_v1.csv.gz",compression="gzip")
# df.drop([])

