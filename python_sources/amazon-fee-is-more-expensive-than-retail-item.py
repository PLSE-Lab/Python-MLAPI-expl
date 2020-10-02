#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import math
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)


from wordcloud import WordCloud, STOPWORDS

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import matplotlib 
import matplotlib.pyplot as plt
import sklearn
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 
plt.rcParams["figure.figsize"] = [16, 12]
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
filenames = check_output(["ls", "../input"]).decode("utf8").strip()


# In[ ]:


df = pd.read_excel("../input/"+filenames) 
print(df.dtypes)
df.head()


# In[ ]:


varnames = df.columns.values

for varname in varnames:
    if df[varname].dtype == 'object':
        lst = df[varname].unique()
        print(varname + " : " + str(len(lst)) + " values such as " + str(lst))


# We could introduce one more feature variable for continents (based on Country variable with 38 possible values).
# 
# We want to see relationship between unit price and quantity purchased.

# In[ ]:


sns.pairplot(x_vars = ['UnitPrice'],y_vars=['Quantity'], hue = 'Country', data = df, size = 10);


# InvoiceNo: Invoice number. Nominal, a 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'c', it indicates a cancellation. 
# StockCode: Product (item) code. Nominal, a 5-digit integral number uniquely assigned to each distinct product. 
# Description: Product (item) name. Nominal. 
# Quantity: The quantities of each product (item) per transaction. Numeric.	
# InvoiceDate: Invice Date and time. Numeric, the day and time when each transaction was generated. 
# UnitPrice: Unit price. Numeric, Product price per unit in sterling. 
# CustomerID: Customer number. Nominal, a 5-digit integral number uniquely assigned to each customer. 
# Country: Country name. Nominal, the name of the country where each customer resides.

# We want to better understand cancellation transactions. InvoiceNo code starts with 'c'. Which items tend to get cancelled?

# In[ ]:


df['InvoiceNo'].str


# In[ ]:


df['SalesAmount'] = df['UnitPrice'] * df['Quantity']


# In[ ]:


df.plot(x = 'InvoiceDate', y = 'SalesAmount')


# In[ ]:


df['Country'].value_counts()


# Let us draw time-series graph per country.

# In[ ]:


countryNames = df['Country'].unique()  


# In[ ]:


def tsplot(country):
    dg = df[df['Country'] == country]
    dg.plot(x = 'InvoiceDate', y = 'SalesAmount')


# In[ ]:


tsplot('United Kingdom')


# In[ ]:


tsplot('Germany')


# Occasionally there are really large purchases! Is it for some holiday, Christmas, wedding season?

# Let us look at the large purchases more closely.

# In[ ]:


df.sort_values(by='SalesAmount', ascending=False)


# AMAZONFEE? It is a large quantity with no CustomerID. There is also Adjust bad debt. POSTAGE (why are these amount positive? I would expect them to be cost, with negative value?) - It seems there are lots of "fee"like items with Quantity 1. Need to watch out. One could also make function to predict Amazon fees etc.

# 

# In[ ]:


df['QuantityAbs'] = df['Quantity'].apply(math.fabs)
df['UnitPriceAbs'] = df['UnitPrice'].apply(math.fabs)


# In[ ]:


df.sort_values(by='UnitPriceAbs', ascending=True)


# In[ ]:


df.sort_values(by='UnitPriceAbs', ascending=False)


# In[ ]:


df.sort_values(by='QuantityAbs', ascending=False)


# In[ ]:


df.sort_values(by='QuantityAbs', ascending=True)


# 

# From the item description, we want to cluster types of items using NLP (word2vector-like embedding).
# Mimick https://www.kaggle.com/jeffd23/visualizing-word-vectors-with-t-sne  and 
# https://medium.com/@aneesha/using-tsne-to-plot-a-subset-of-similar-words-from-word2vec-bb8eeaea6229
# to map each word into a vector
# 
# But each item description is made of multiple words, and the number of words vary a lot.
# 
# We could find a method to choose only 1 word from the description. 
# 
# Shall we care about the ordering of the words or not?
# 

# To get some idea of item description, and also for fun, let us make a word cloud with all the description texts.
# with help from https://www.kaggle.com/adiljadoon/word-cloud-with-python 
# 

# In[ ]:



stopwords = set(STOPWORDS) 

wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=200,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(df['Description']))

print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("itemDescriptionCloud.png", dpi=900)


# I could do all the cool plotting as https://pandas.pydata.org/pandas-docs/stable/visualization.html , but let us start with histogram of quantity and price (previous scatter plot didn't serve us well, since some values were too large) - perhaps we draw them in log scale
# https://seaborn.pydata.org/generated/seaborn.distplot.html 

# In[ ]:


sns.distplot(df['UnitPrice'])


# In[ ]:


sns.distplot(df['Quantity'])


# In[ ]:


sns.distplot(df['QuantityAbs'])


# In[ ]:




