#!/usr/bin/env python
# coding: utf-8

# I wanted to show you an idea I had about how to handle missing values. We'll use the porto seguro training data to see, how the idea works.

# In[ ]:


import pandas as pd
from sklearn.cluster import MiniBatchKMeans

X = pd.read_csv("../input/train.csv", na_values = -1)
X.drop(["id", "target"], axis = 1, inplace = True)

na_count = X.isnull().sum()
na_columns = list(na_count[na_count>0].index.values)

print("columns with missing values:")
print(na_columns)

na_count.plot(kind = "bar")


# As you can see, there are some features with a lot of missing values. So how do we handle them? Normally, I would replace nominal values with the median of the not-missing values and categorical/binary features with the most common value of the not-missing values. Below I try something more...

# In[ ]:


#create df only with columns with no missing values
X_no_missing = X.drop(na_columns, axis = 1)
 
#one hot encoding of categorical features
cat_columns_no_missing = list(filter(lambda x: x.endswith("cat"),
                                     X_no_missing.columns.values))
X_no_missing_oh = pd.get_dummies(X_no_missing, columns = cat_columns_no_missing)   


# So I drop all columns that contain missing values and then I use KMeans on the remaining columns to cluster the samples.

# In[ ]:


#train kmeans
kmeans = MiniBatchKMeans(n_clusters = 15, random_state = 0, batch_size = 2000)
kmeans.fit(X_no_missing_oh)
print("Clustersize: \n")
print(pd.Series(kmeans.labels_).value_counts())

#store cluster labels in df
X["cluster"] = kmeans.labels_


# We see that all clusters have approximately the same size. As a next step we loop over all columns containing missing values. For each column we drop the missing values and use the rest to calculate a replacement value. This would be the most common label for categorical/binary features and the median for nominal features. 

# In[ ]:


#for columns with missing values, drop missing values and find median or most common value - per cluster
Values_replace_missing = pd.DataFrame()

for i in na_columns:
    clean_df = X[["cluster", i]].dropna()
    if i.endswith("cat"):
        Values_replace_missing[i] = clean_df.groupby(["cluster"]).agg(lambda x:x.value_counts().index.values[0])
    else:
        Values_replace_missing[i] = clean_df.groupby(["cluster"]).median() 

print(Values_replace_missing)


# As you can see, different clusters have different replacement values. This is especially prominent for "ps_car_05_cat". Now we have to replace the missing values with the ones we calculated above. 

# In[ ]:


#replace missing values with median or most common value in the same cluster
for cl, cat in ((x, y) for x in range(15) for y in na_columns):
    X.loc[(X["cluster"] == cl) & pd.isnull(X[cat]), cat] = Values_replace_missing.loc[cl, cat]

#print remaining missing values (should be zero)
print("\n remaining missing values: " + str(X.isnull().sum().sum()))


# I have not tested the impact of this approach on the prediction quality but maybe this is interesting for you, too. So what do you think: does this approach make sense?
