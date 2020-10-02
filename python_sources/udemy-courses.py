#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
data = pd.read_csv("../input/udemy-courses/udemy_courses.csv")


# In[ ]:


delCols = ["course_title", "url"]
data.drop(labels=delCols, axis=1)

print( data.columns )
data.corr()


# In[ ]:


data = data[ ["course_id", "is_paid", "price", "num_subscribers", "num_reviews", "num_lectures", "content_duration"] ]
data.head()


# In[ ]:


data = data.groupby(by="course_id").mean()
data.head()


# **What are the best free courses by subject?**

# In[ ]:


data.sort_values(by="num_reviews")[data["price"]==0]["price"].head()


# **What are the most popular courses?**

# In[ ]:


data.sort_values(by="num_reviews").head()


# **What are the most engaging courses?**

# In[ ]:


print("List with id of the most engaging courses:")
print( list(data.sort_values(by="num_lectures").index[0:5]) )


# **How are courses related?**

# In[ ]:


print("Standard deviation for:")
print( "'price': "+str(data["price"].std()) )
print( "'num_subscribers': "+str(data["num_subscribers"].std()) )
print( "'num_reviews': "+str(data["num_reviews"].std()) )
print( "'num_lectures': "+str(data["num_lectures"].std()) )
print( "'content_duration': "+str(data["content_duration"].std()) )


# As we see standard deviation for price and for number of lectures is not so large.

# In[ ]:


print("Mean for:")
print( "'price': "+str(data["price"].mean()) )
print( "'num_lectures': "+str(data["num_lectures"].mean()) )


# As we see in main price for the courses  is in small range of values.

# **Which courses offer the best cost benefit?**

# In[ ]:


benefits = data["price"].div(data["content_duration"]).sort_values().head()

