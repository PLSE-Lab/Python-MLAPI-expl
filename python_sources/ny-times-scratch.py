#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests
from bs4 import BeautifulSoup


# In[ ]:


my_url = "https://www.nytimes.com/section/technology"


# In[ ]:


doc = requests.get(my_url)
doc


# In[ ]:


# doc.content


# In[ ]:


my_soup = BeautifulSoup(doc.content)
# my_soup


# In[ ]:


articles = my_soup.find_all("li", class_="css-ye6x8s")


# In[ ]:


for article in articles:
#     print(article.find("h2").string)
    print(article.find_all("div", class_="css-1lc2l26"))
    print(article.find_all("time"))


# In[ ]:


for article in articles:
#     print("article")
    childrens = list(article.find("a").children)
    child = childrens[2]
    print(child.text)


# In[ ]:


articles = my_soup.find_all("li", class_="css-ye6x8s")


# In[ ]:


type(articles)


# In[ ]:


len(articles)


# In[ ]:


my_list = list(articles[2].children)
my_list


# In[ ]:


def add_two(a, b):
    return a + b


# In[ ]:


def add_two_2(a, b):
    print(a + b)


# In[ ]:


my_var = add_two(1, 2)
print("test")
print(my_var)


# In[ ]:


add_two_2(1, 2)
print("test")


# In[ ]:


def my_attrs():
    return [1, 2, 3]

