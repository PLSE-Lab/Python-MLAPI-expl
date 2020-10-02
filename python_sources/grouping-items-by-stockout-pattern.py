#!/usr/bin/env python
# coding: utf-8

# For some competitions it can be helful to find patterns of NA values or outliers, and cluster observations by similar patterns. Here I'm wondering what value there is in looking at stockouts and grouping items by those patterns. In our data we have typical department store groupings - family and department. This makes sense for the most part although for this exercise, the groupings aren't always helpful. If you're planning a Sunday summer picnic for instance, you might go over to Household for a tablecloth, Hobbies for a frisbee, and Foods for a big pack of ground beef (don't judge..:). A lot of other people are doing the same thing and so maybe the store runs out of those items. Demand is event-driven in this case and crosses department lines.
# 
# We don't really know if an item is out of stock or just no one bought it. I'll start with items having higher median sales and assume that a drop to 0 is due to a stockout. This is sure to be wrong in some cases.

# In[ ]:


import numpy as np
np.set_printoptions(precision=2, suppress=True)
import pandas as pd
pd.options.display.max_columns = 2000

id_cols = ['store_id', 'item_id']
sales_cols = ['d_'+str(i) for i in range(1,1914)]

sales = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv',
                    usecols=id_cols+sales_cols, index_col=id_cols) \
            .astype(np.uint16).sort_index()
sales


# In[ ]:


# Filter items at one store 
sales_ca1 = sales.loc['CA_1']                  .assign(d_median=lambda x: x.median(axis=1))                  .query('d_median >= 4')                  .drop(columns='d_median')                  .iloc[:, -28*24:]  # use the last two years

sales_ca1


# In[ ]:


# Check dept counts
sales_ca1.groupby(sales_ca1.index.str[:-6]).size()


# The dendrogram from missingno is a quick way to see what sort of crossover there might be across departments and families. It looks like the lowest level of clusters (to the far right) are mostly within departments. Just a level or two up (leftward in the picture), the clusters become more cross-department. Maybe it's picnic time?

# In[ ]:


import missingno as msno
msno.dendrogram(sales_ca1.replace(0, np.nan).T, method='ward')


# Missingno uses [SciPy hierarchical clustering](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html#module-scipy.cluster.hierarchy) under the hood. You can use the scipy functions directly to return the cluster numbers and use those as a feature in a forecasting model. The reference also gives more detail on the specific clustering algorithms and distance measurements that are available. 

# In[ ]:


from scipy.cluster.hierarchy import linkage, fcluster

# Get matrix form of the dendrogram 
Z = linkage((sales_ca1>0).astype(int), method='ward')
print(Z)


# In[ ]:


# Map items to clusters
clust = fcluster(Z, t=12, criterion='maxclust')
pd.crosstab(clust, sales_ca1.index.str[:-6])


# Here there are 3 cross-family clusters with the rest specific to foods. Of course, different methods, distance metrics and clustering parameters will give different results. 
# 

# In[ ]:




