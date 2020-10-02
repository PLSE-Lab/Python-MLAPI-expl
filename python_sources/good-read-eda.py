#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn


# **Loading Data**

# In[ ]:


data = pd.read_csv("../input/books.csv", error_bad_lines=False)

data.info()


# **Visualize some data using histogram**

# In[ ]:


data_to_plot = data.drop(["title", "authors", "isbn", "isbn13", "bookID", "language_code"], axis = 1)
data_to_plot.columns


# In[ ]:


data_to_plot.hist(figsize = (20, 20))
plt.show()


# **Visualizing data from data_to_plot as  distributions 
# **

# In[ ]:


fig = plt.subplots(2,2, figsize = (20, 18))
for i in range (len(data_to_plot.columns)):
    plt.subplot(2, 2, i+1)
    sn.distplot(data_to_plot.iloc[:,i], color = "orange")


# **plotting the heatmap of the correlation in the data**

# In[ ]:


data_to_plot.corr()
plt.figure(figsize = (10, 10))
sn.heatmap(data_to_plot.corr(), linewidth = 0.5)


# **jointplot to examine correlation between two variables
# **
# 
# according to the heatmap above, the correlation between some variables are worth examining such as:
#     - text_reviews_count vs ratings_count
#     - average_rating vs num_pages

# In[ ]:


sn.jointplot(x = data["text_reviews_count"], y = data["ratings_count"], kind = "scatter")


# In[ ]:


sn.jointplot(x = data["average_rating"], y = data["# num_pages"], kind = "scatter")


# **barplot to show the top ten books that occurs in the dataset the most
# **

# In[ ]:


plt.figure(figsize=(20,15))
books = data['title'].value_counts()[:10]
rating = data["average_rating"][:10]
sn.barplot(x = books, y = books.index)
plt.xlabel("# Occurances", fontsize = 20)
plt.ylabel("Books", fontsize = 20)
plt.xticks(fontsize = 17)
plt.yticks(fontsize = 17)

plt.show()


# **barplot to show the number of books written in various different languages**

# In[ ]:


plt.figure(figsize = (12, 10))
language = data["language_code"].value_counts()
sn.barplot(x = language, y = language.index)
plt.yticks(fontsize = 15)
plt.xticks(fontsize = 12)
plt.ylabel("language code", fontsize = 15, color = "blue")
plt.xlabel("Count", fontsize = 15, color = "blue")


# **barplot to show the top ten books that have the most rating_counts**

# In[ ]:


plt.figure(figsize = (12, 12))
plotting = data.sort_values(["ratings_count"], ascending = False)
sn.barplot(x = plotting["ratings_count"][:10], y = plotting["title"][:10])
plt.yticks(fontsize = 10)
plt.xticks(fontsize = 10)
plt.ylabel("title", fontsize = 15)
plt.xlabel("ratings_count", fontsize = 15)


# **bar graph to show the top ten authors that published the most books
# **

# In[ ]:


fig = plt.figure(figsize = (15, 6))
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.xlabel("author", fontsize = 15, color = "green")
plt.ylabel("count", fontsize = 15, color = "green")
data["authors"].value_counts()[:10].plot.bar()


# **bar graph that shows the number of books published with respect to dfferent average_rating standard**

# In[ ]:


def graphing_rating(data):
    fig = plt.figure(figsize = (15, 6))
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.xlabel("author", fontsize = 15, color = "green")
    plt.ylabel("Book count", fontsize = 15, color = "green")
    data["authors"].value_counts()[:10].plot.bar()


# In[ ]:


condition1 = data[data["average_rating"] >= 4.3]
condition2 = data[data["average_rating"] >= 4.5]
condition3 = data[data["average_rating"] >= 4.7]


# In[ ]:


graphing_rating(condition1)


# In[ ]:


graphing_rating(condition2)


# In[ ]:


graphing_rating(condition3)

