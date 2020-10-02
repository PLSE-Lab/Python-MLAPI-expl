#!/usr/bin/env python
# coding: utf-8

# In this notebook, we will understand the steps of static web scrapping to extract a table from any website in simple steps. At first, we will see how to use pandas to extract tables from a website.
# 
# Python provides a powerful library BeautifulSoup for web scraping. As a second method, we will learn how to use BeautifulSoup to extract tables from a website. Apart from tables, we can extract list, texts, paragraphs, headers and all sort of things present on the website using BrautifulSoup.

# In[ ]:


import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


# ## Pandas

# In[ ]:


# using pandas
dfs = pd.read_html('https://www.zyxware.com/articles/4344/list-of-fortune-500-companies-and-their-websites')
for df in dfs:
    print(df)


# In[ ]:


dfs[0].to_csv('fortune_100.csv', index = False)


# ## BeautifulSoup

# In[ ]:


url = 'https://www.zyxware.com/articles/4344/list-of-fortune-500-companies-and-their-websites'
page = requests.get(url)


# In[ ]:


print(page.text[0:500]) # Looking at first 500 characters of the response which we got from the url


# #### Understanding the output
# The above response is in html format which is rendered by browsers to create the content on the website. Beautiful Soup gives us similar functionality of extracting useful informattion from the above content. 

# In[ ]:


soup = BeautifulSoup(page.content, 'html.parser')


# soup is a BeautifulSoup object from which we can extract any relevant data from the webpage. Following is the list of commoly used tags in html:
# 
# `Tags` | `Description`
# --- | ---
# !DOCTYPE | Defines the document type
# html | Defines an HTML document
# head | Contains metadata/information for the document
# title | Defines a title for the document
# body | Defines the document's body
# h1 to <h6 | Defines HTML headings
# p | Defines a paragraph
# br | Inserts a single line break
# hr | Defines a thematic change in the content
# !--...-- |	Defines a comment
# ul | Defines an unordered list
# ol | Defines an ordered list
# li | Defines a list item
# dir | Not supported in HTML5. Use ul instead. Defines a directory list
# dl | Defines a description list
# dt | Defines a term/name in a description list
# dd | Defines a description of a term/name in a description list
# table|Defines a table
# caption|Defines a table caption
# th|Defines a header cell in a table
# tr|Defines a row in a table
# td|Defines a cell in a table
# thead|Groups the header content in a table
# tbody|Groups the body content in a table
# tfoot|Groups the footer content in a table
# col|Specifies column properties for each column within a colgroup element
# colgroup | Specifies a group of one or more columns in a table for formatting
# 
# On our [target website](https://fortune.com/fortune500/2019/search/) we are trying to extract a table. So we will use table tag to extract the list.

# In[ ]:


table = soup.find('table')
table_rows = table.find_all('tr')
fortune = []
for tr in table_rows:
    td = tr.find_all('td')
    row = [i.text for i in td]
    fortune.append(row)


# In[ ]:


fortune = pd.DataFrame(fortune)
fortune.drop(0, axis = 0, inplace = True)
fortune.head()


# # Conclusion
# 
# That's it for now. In this notebook we learnt two methods to extract tables from a website - by using pandas and beautifulsoup. Please upvote this [notebook](https://www.kaggle.com/prasun2106/extracting-tables-from-any-website?scriptVersionId=33926509) if you find above code helpful, as it motivates me to write more tutorials for beginners. You can visit my website [datamaniac.tech](https://www.datamaniac.tech) for more such tutorials. Follow me on [Kaggle](https://www.kaggle.com/prasun2106), [Github](https://github.com/prasun2106) and contact me on [LinkedIn](https://www.linkedin.com/in/prasun-kumar-8250a5119/)
