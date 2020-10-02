#!/usr/bin/env python
# coding: utf-8

# **Hello world !**
# 
# It's my first kaggle's kernel so I hope you will find it nice !
# 
# Let's answer the question : *Wich are the LEGO's themes with the most sets* ?

# In[ ]:


# import packages that we need
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Loading sets we need
sets = pd.read_csv('../input/sets.csv')
themes = pd.read_csv('../input/themes.csv')


# Let's have a quick look on our data!

# In[ ]:


print(sets.head())


# In[ ]:


print(themes.head())


# So we can see that in the sets dataframe we can access the theme id as we can see on the schema : 
# ![](https://storage.googleapis.com/kaggle-datasets/1599/2846/downloads_schema.png?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1518525083&Signature=QzwZU1kuwmRl8M7ybA%2B4jhNrC0SDvYcPaKW%2B1v83QVn1YAL66ce5L3Pjo%2FV4KIe%2BLCc8xUw6j%2B5frjuldrlB0W%2BVIUJbv2NNUTcMg83cYgg4ZBjvoaTKeJGVGiGoJghvFEOFhbzJBZ3moW4pH5RJdTvMDqyJxnNbxjRCT9j8YIjFXyiCaMoRIz9ihbregxG6Bt2MufCR9%2FP1b7X1hJmCjBSgowvkKH0mI9dxsBnvFQheFR41uhIOsZu64rVRvpW5fqvlSuzuZye5Ien9fz1eo2Hg7tZlnh5YlV3s7SZgIOaXTVYAZReKLd2z3TMj7%2FokOGhoRZgNZO%2FCrB3sL7rfuw%3D%3D)
# 
# Also we can see that themes with parent are not really usefull here because parents theme count are the sum of each child count.
# 
# So let's start by counting the frequency of each theme :
# 

# In[ ]:


#Create pandas Series with id of theme as index and count of values as value
set_theme_count = sets["theme_id"].value_counts()
#Convert it to dataframe
set_theme_count = pd.DataFrame({'id':set_theme_count.index, 'count':set_theme_count.values})

print(set_theme_count.head())


# Now that we have our dataframe with the count we have to join it with the theme dataframe.

# In[ ]:


# Join name of theme
set_theme_count = pd.merge(set_theme_count, themes, on='id')

print(set_theme_count.head())


# We have now to remove rows with a parent_id because we are not interested in those

# In[ ]:


# Get only themes with no parent
set_theme_count_no_parent = set_theme_count[pd.isnull(set_theme_count['parent_id'])]

print(set_theme_count_no_parent.head())


# Great ! Now we have to get our top 10 and plot it !

# In[ ]:


# Get the top 10 and plot it
set_theme_count_top_10 = set_theme_count_no_parent.sort_values(by=["count"], ascending=False)[:10]
top_10 = set_theme_count_top_10["count"]
top_10.index = set_theme_count_top_10["name"]

top_10.plot.bar()
plt.show()


# And we found out that is the** Gear theme that have the most sets ** ! (Even if I'm a LEGO fan I didn't know this theme until now, my favorite theme is Star Wars and it's only the 9th).
# 
# I tried to find the Gear theme on the shop ([shop link](https://shop.lego.com/en-CA/Themes) but I can't, if you find that theme you comment comment the link there !
# 
# I hope that you enjoyed and **thanks for reading** ! :D 
# 
# ![](https://orig00.deviantart.net/fdd9/f/2013/011/2/5/25812eb3b3924fdb3c788aefe5635301-d5r6qyd.jpg)
# 
# 
