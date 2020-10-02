#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from urllib.request import urlopen
from bs4 import BeautifulSoup


# After importing necessary modules, specify the URL containing the dataset and pass it to urlopen() to get the html of the page

# In[ ]:


url = "http://www.hubertiming.com/results/2017GPTR10K"
html = urlopen(url)


# creating beautifull soup object from html

# In[ ]:


soup = BeautifulSoup(html, 'lxml')
type(soup)


# soup object allows you to extract interesting information about the website you're scraping such as getting the title of the page as shown below

# In[ ]:


# Get the title
title = soup.title
print(title)


# In[ ]:


# Print out the text
text = soup.get_text()
print(soup.text)


# then use the find_all() method of soup to extract useful html tags within a webpage. Examples of useful tags include < a > for hyperlinks, < table > for tables, < tr > for table rows, < th > for table headers, and < td > for table cells. The code below shows how to extract all the hyperlinks within the webpage.

# In[ ]:


soup.find_all('a')


#  using a for loop and the get('"href") method to extract and print out only hyperlinks.

# In[ ]:


all_links = soup.find_all("a")
for link in all_links:
    print(link.get("href"))


# To print out table rows only, pass the 'tr' argument in soup.find_all().

# In[ ]:


# Print the first 10 rows for sanity check
rows = soup.find_all('tr')
print(rows[:10])


#  Below is a for loop that iterates through table rows and prints out the cells of the rows.

# In[ ]:


for row in rows:
    row_td = row.find_all('td')
print(row_td)
type(row_td)


# The easiest way to remove html tags is to use Beautiful Soup, and it takes just one line of code to do this. Pass the string of interest into BeautifulSoup() and use the get_text() method to extract the text without html tags.

# In[ ]:


str_cells = str(row_td)
cleantext = BeautifulSoup(str_cells, "lxml").get_text()
print(cleantext)


# The code below shows how to build a regular expression that finds all the characters inside the < td > html tags and replace them with an empty string for each table row
# 
#  After compiling a regular expression, you can use the re.sub() method to find all the substrings

# In[ ]:


import re

list_rows = []
for row in rows:
    cells = row.find_all('td')
    str_cells = str(cells)
    clean = re.compile('<.*?>')
    clean2 = (re.sub(clean, '',str_cells))
    list_rows.append(clean2)
print(clean2)
type(clean2)


# The next step is to convert the list into a dataframe and get a quick view of the first 10 rows using Pandas.

# In[ ]:


df = pd.DataFrame(list_rows)
df.head(10)


# To clean it up, you should split the "0" column into multiple columns at the comma position. This is accomplished by using the str.split() method.

# In[ ]:


df1 = df[0].str.split(',', expand=True)
df1.head(10)


# You can use the strip() method to remove the opening square bracket on column "0."

# In[ ]:


df1[0] = df1[0].str.strip('['and ']')
df1[1] = df1[1].str.strip('['and ']')
df1.head(10)


# In[ ]:


#use the find_all() method to get the table headers.

col_labels = soup.find_all('th')


# In[ ]:


#Similar to table rows, you can use Beautiful Soup to extract text in between html tags for table headers.
all_header = []
col_str = str(col_labels)
cleantext2 = BeautifulSoup(col_str, "lxml").get_text()
all_header.append(cleantext2)
print(all_header)


# In[ ]:


#convert the list of headers into a pandas dataframe.

df2 = pd.DataFrame(all_header)
df2.head()


# In[ ]:


#you can split column "0" into multiple columns at the comma position for all rows.

df3 = df2[0].str.split(',', expand=True)
df3.head()


# In[ ]:


#The two dataframes can be concatenated into one using the concat() method as illustrated below.

frames = [df3, df1]

df4 = pd.concat(frames)
df4.head(10)


# In[ ]:


#assign the first row to be the table header.

df5 = df4.rename(columns=df4.iloc[0])
df5.head()


# In[ ]:


#df5[0] = df5[0].str.strip('['and ']')
#df5.head(10)


# In[ ]:


df5.info()
df5.shape


# In[ ]:


#drop all rows with any missing values.

df6 = df5.dropna(axis=0, how='any')


# In[ ]:


df7 = df6.drop(df6.index[0])
df7.head()


# In[ ]:


# more data cleaning by renaming the '[Place' and ' Team]' columns. Python is very picky about space. Make sure you include space after the quotation mark in ' Team]'.

df7.rename(columns={'[Place': 'Place'},inplace=True)
df7.rename(columns={' Team]': 'Team'},inplace=True)
df7.head()


# In[ ]:


#removing the closing bracket for cells in the "Team" column.

df7['Team'] = df7['Team'].str.strip(']')
df7.head()


# Data Analysis and Visualization

# In[ ]:


#convert the column "Chip Time" into just minutes. One way to do this is to convert the column to a list first for manipulation.

time_list = df7[' Chip Time'].tolist()

# You can use a for loop to convert 'Chip Time' to minutes

time_mins = []
for i in time_list:
    h, m, s = i.split(':')
    math = (int(h) * 3600 + int(m) * 60 + int(s))/60
    time_mins.append(math)
#print(time_mins)


# In[ ]:


#convert the list back into a dataframe and make a new column ("Runner_mins") for runner chip times expressed in just minutes.

df7['Runner_mins'] = time_mins
df7.head()


# In[ ]:


df7.describe(include=[np.number])


# distribution plot of runners' chip times plotted using the seaborn library. The distribution looks almost normal.

# In[ ]:


#Visualizations
x = df7['Runner_mins']
ax = sns.distplot(x, hist=True, kde=True, rug=False, color='m', bins=25, hist_kws={'edgecolor':'black'})
plt.show()


# In[ ]:


#whether there were any performance differences between males and females of various age groups. Below is a distribution plot of chip times for males and females.

f_fuko = df7.loc[df7[' Gender']==' F']['Runner_mins']
m_fuko = df7.loc[df7[' Gender']==' M']['Runner_mins']
sns.distplot(f_fuko, hist=True, kde=True, rug=False, hist_kws={'edgecolor':'black'}, label='Female')
sns.distplot(m_fuko, hist=False, kde=True, rug=False, hist_kws={'edgecolor':'black'}, label='Male')
plt.legend()


# In[ ]:


#groupby() method to compute summary statistics for males and females separately as shown below.

g_stats = df7.groupby(" Gender", as_index=True).describe()
print(g_stats)


# In[ ]:




