#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import scipy.stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# In[ ]:


data = pd.read_csv('../input/goodreadsbooks/books.csv', error_bad_lines=False)


# In[ ]:


data.head()


# In[ ]:


data = data.dropna()


# In[ ]:


df = data[["title", "average_rating","ratings_count", "text_reviews_count", "ratings_count"]]


# In[ ]:


df.head()


# In[ ]:


title_lst = list(df["title"].values)


# In[ ]:


value_arr = df[["average_rating", "ratings_count", "text_reviews_count", "ratings_count"]].values


# In[ ]:


sc = StandardScaler()


# In[ ]:


value_std = sc.fit_transform(value_arr)


# In[ ]:


kmeans = KMeans(n_clusters=100, random_state=0).fit(value_std)


# In[ ]:


kmeans.predict([value_std[0]])[0]


# In[ ]:


label_lst = kmeans.labels_ == kmeans.predict([value_std[0]])[0]


# In[ ]:


print("your_search:", title_lst[0])
print("")
for i in range(0, len(title_lst)):
    
    if label_lst[i] == True:
        print(title_lst[i])


# In[ ]:


label_lst = kmeans.labels_ == kmeans.predict([value_std[20]])[0]


# In[ ]:


print("your_search:", title_lst[20])
print("")
for i in range(0, len(title_lst)):
    
    if label_lst[i] == True:
        print(title_lst[i])


# In[ ]:




