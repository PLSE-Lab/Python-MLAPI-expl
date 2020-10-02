#!/usr/bin/env python
# coding: utf-8

# A household is said to be in fuel poverty when its members cannot afford to keep adequately warm at a reasonable cost, given their income. The term is mainly used in the UK, Ireland and New Zealand, although discussions on fuel poverty are increasing across Europe,and the concept also applies everywhere in the world where poverty may be present.https://en.wikipedia.org/wiki/Fuel_poverty

# In[ ]:


#codes from Rodrigo Lima  @rodrigolima82
from IPython.display import Image
Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSCfSVic6tmXi2E8eAoyRouDPNSeFoM6SPOh1oelSjB0AC9cdVZAg&s',width=400,height=400)


# Image newtonnews.co.uk

# In[ ]:


#codes from Rodrigo Lima  @rodrigolima82
from IPython.display import Image
Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRqHrhpTiD6kmKGyWBnQ4BMf3OGKGDdJZOq3OeQ5ytTQcLxTB0e&s',width=400,height=400)


# Image ashoka.org

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


#codes from Rodrigo Lima  @rodrigolima82
from IPython.display import Image
Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRcpmoMAg3qGqpNqHBCn0fJ1-m2wZF2axwPJY8DqWKWqBeSn_Iq&s',width=400,height=400)


# https://www.insidehousing.co.uk/news/news/consultation-on-governments-fuel-poverty-strategy-published-41003

# Some people can't even afford to heat their home. Nine percent of the EU population could not afford to heat their home sufficiently with Bulgaria scoring the highest of 39.2%.

# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling as pp
import plotly.graph_objs as go
from plotly.offline import iplot
import plotly.express as px
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.subplots import make_subplots
init_notebook_mode(connected=True)


# In[ ]:


nRowsRead = 1000 # specify 'None' if want to read whole file
df = pd.read_csv('../input/cusersmarildownloadsfuelpovertycsv/fuelpoverty.csv', delimiter=';', encoding = "ISO-8859-1", nrows = nRowsRead)
df.dataframeName = 'fuelpoverty.csv'
nRow, nCol = df.shape
print(f'There are {nRow} rows and {nCol} columns')


# In[ ]:


#codes from Rodrigo Lima  @rodrigolima82
from IPython.display import Image
Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ_iW0mtwSsmsaK9SaUqEqtHcGJnxbfSKgVwnmHrFc2FZ9MQsIP&s',width=400,height=400)


# Image independentage.org

# In[ ]:


df.head()


# In[ ]:


#codes from Rodrigo Lima  @rodrigolima82
from IPython.display import Image
Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQpHwhvGt5VK7CiYR1xtDVbfhBJTpFM6d4ERgsYjkq2SsCw6Tu6Zg&s',width=400,height=400)


# Image myutilitygenius.co.uk

# In[ ]:


categorical_cols = [cname for cname in df.columns if
                    df[cname].nunique() < 10 and 
                    df[cname].dtype == "object"]


# Select numerical columns
numerical_cols = [cname for cname in df.columns if 
                df[cname].dtype in ['int64', 'float64']]


# In[ ]:


print(numerical_cols)


# In[ ]:


print(categorical_cols)


# These codes below are from Fatih Bilgin. I'm trying to apply Lambda. Seems that I did't get it.

# In[ ]:


df.la_code = df.la_code.apply(lambda x: x+'0' if '.' in x[-1:0] else x)
df.la_code = df.la_code.apply(lambda x: x+'0' if '.' in x[-2:-1] else x)
df.la_code = df.la_code.apply(lambda x: x+'0' if '.' in x[-3:-2] else x)


# In[ ]:


df_la_code = df[df.la_name.str.len()==4]


# In[ ]:


#codes from Rodrigo Lima  @rodrigolima82
from IPython.display import Image
Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQNZwQSQJZbeUnn7EBImbHicfRfeFSF0pQcg6RX40S9ZphSlGA8vQ&s',width=400,height=400)


# Image boilerjuice.com

# EDA

# In[ ]:


df.describe()


# In[ ]:


#codes from Rodrigo Lima  @rodrigolima82
from IPython.display import Image
Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS1PBaotbXnkLD-cm6vxB06N4ZCOS5cXGn3Kuhd5DHgWkaa0xzHJw&s',width=400,height=400)


# Image geos.ed.ac.uk

# In[ ]:


#Missing values. Codes from my friend Caesar Lupum @caesarlupum
total = df.isnull().sum().sort_values(ascending = False)
percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(8)


# In[ ]:


#codes from Rodrigo Lima  @rodrigolima82
from IPython.display import Image
Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQGSCMqY59OBND41ml9AZDgPm67OxFu8wodMBDF1wvWnvlxaclY&s',width=400,height=400)


# Image poverty.org.uk

# In[ ]:


sns.scatterplot(x='2011_lsoa_code',y='10%_of_income_on_fuel_percentage_2011',data=df)


# In[ ]:


#codes from Rodrigo Lima  @rodrigolima82
from IPython.display import Image
Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRc9s8nTWUQ5UETEF3sB8mXJGYsQebP-u3WSzMwKvGsyqE_PHTF&s',width=400,height=400)


# Image thenorthernecho.co.uk

# In[ ]:


sns.countplot(df["10%_of_income_on_fuel_percentage_2011"])


# In[ ]:


import plotly.offline as pyo
import plotly.graph_objs as go
lowerdf = df.groupby('2011_lsoa_code').size()/df['10%_of_income_on_fuel_percentage_2011'].count()*100
labels = lowerdf.index
values = lowerdf.values

# Use `hole` to create a donut-like pie chart
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6)])
fig.show()


# In[ ]:


#codes from Rodrigo Lima  @rodrigolima82
from IPython.display import Image
Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQDw9WNal7HeYQ5Qu4wdmgUo5J5nLMsdmXo1dDcoJEn802I_S7_fA&s',width=400,height=400)


# Image tallbloke.wordpress.com

# In[ ]:


ng = df[df.high_cost_low_income_percentage_2011<250]
plt.figure(figsize=(8,4))
sns.boxplot(y="high_cost_low_income_percentage_2011",x ='2011_lsoa_code' ,data = ng)
plt.title("high_cost_low_income_percentage_2011 & 2011_lsoa_code < 250")
plt.show()


# Is that above a boxplot? I don't think so. Maybe a bunch of flies.

# In[ ]:


#catplot 2011 LSOA and High cost/low income 2011
plt.figure(figsize=(10,6))
sns.catplot(x="2011_lsoa_code", y="high_cost_low_income_percentage_2011", data=df);
plt.ioff()


# In[ ]:


#codes from Rodrigo Lima  @rodrigolima82
from IPython.display import Image
Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSavP0cCnMFJJcw7M8yVaEL0191_OrW74ainDqU2R03Plf5t7soWw&s',width=400,height=400)


# Image broadlandgroup.org

# In[ ]:


f, ax = plt.subplots(figsize=(8,6))
x = df['high_cost_low_income_percentage_2011']
ax = sns.distplot(x, bins=10, vertical = True)
plt.show()


# In[ ]:


datt =df[['2011_lsoa_code', 'high_cost_low_income_percentage_2011']]
g = sns.PairGrid(datt)
g = g.map(plt.scatter)


# In[ ]:


#codes from Rodrigo Lima  @rodrigolima82
from IPython.display import Image
Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR3ZEWV5JrNcc6fuaSm0jO1qg56MKlhM5PHjHTK9zX1PMwrwlxl8w&s',width=400,height=400)


# Image reelnews.co.uk

# In[ ]:


g = sns.PairGrid(datt, hue="2011_lsoa_code")
g = g.map_diag(plt.hist)
g = g.map_offdiag(plt.scatter)
g = g.add_legend()


# My first Word Cloud. Codes from Baval @bavalpreet26.

# In[ ]:


#word cloud
from wordcloud import WordCloud, ImageColorGenerator
text = " ".join(str(each) for each in df.la_name)
# Create and generate a word cloud image:
wordcloud = WordCloud(max_words=200, background_color="white").generate(text)
plt.figure(figsize=(10,6))
plt.figure(figsize=(15,10))
# Display the generated image:
plt.imshow(wordcloud, interpolation='Bilinear')
plt.axis("off")
plt.show()


# In[ ]:


#codes from Rodrigo Lima  @rodrigolima82
from IPython.display import Image
Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSdVLCVtti0pnWw6yMmQKGwssF2zPEazAfsTsTpDBmpJzRcYUsj&s',width=400,height=400)


# Image sustainableworkspaces.co.uk

# In[ ]:


#codes from Rodrigo Lima  @rodrigolima82
from IPython.display import Image
Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ-q3Zw4tPDYfO1SbPJHU1yiAM--9q9NOp3s-P1Nkm6xERtems&s',width=400,height=400)


# Image brysonenergy.org

# In[ ]:


#codes from Rodrigo Lima  @rodrigolima82
from IPython.display import Image
Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSrIqPvFTKjAAwtAVE42sjyEI5o3GMJgruf9sU7OnQM849Qlvv7SA&s',width=400,height=400)


# Images routledge.com

# While you're analysing, think about the people, all over the world, in such conditions of Poverty. They aren't only numbers.
