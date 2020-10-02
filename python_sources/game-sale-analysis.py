#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sales=pd.read_csv("../input/vgsales.csv")
sales.head()


# In[ ]:


sales.describe() 


# In[ ]:


sales.drop("Rank", axis=1,inplace=True)


# Let's identify all the genre present in the dataset and analyze how many games are published in each genre.

# In[ ]:


sales.groupby("Genre").size()


# A pie chart will show different genre as a part of the whole

# In[ ]:


sales.groupby("Genre").size().plot.pie(autopct="%1.1f%%", explode=(0.1,0,0,0,0,0,0,0,0,0,0.1,0), radius=2, startangle=90, shadow=True)


# As the pie chart depicts, the majority of games are published in action and sports genre.

# Now let's find which region contributed to the maximum sales of the video games

# In[ ]:


sales[["NA_Sales", "EU_Sales","JP_Sales","Other_Sales"]].sum().plot(kind='pie', autopct="%1.1f%%", explode=(0.1,0,0,0), startangle=270, radius=2)


# The above pie chart clearly shows that almost 50% of the total sales in video games is contributed by North American region which includes contries like Canada.

# Now lets identify in which year the sales were the maximum for all the three regions

# In[ ]:


sales.groupby("Year")["Global_Sales"].sum().plot(kind="line", grid=True, legend=True)


# The above line graph shows that maximum sales took place between 2005-2010 with a peak in the year 2008.

# Now let us identify the trends in sales in different regions.

# In[ ]:


sales.groupby("Year")["NA_Sales"].sum().plot(kind="line", grid=True, legend=True)
sales.groupby("Year")["EU_Sales"].sum().plot(kind="line", grid=True, legend=True)
sales.groupby("Year")["JP_Sales"].sum().plot(kind="line", grid=True, legend=True)
sales.groupby("Year")["Other_Sales"].sum().plot(kind="line", grid=True, legend=True)


# The sales peaked from 2005-2010 as shown by the above graphs.

# Now let us identify all the platforms and number of games published in the respective platforms.

# In[ ]:


sales.groupby("Platform").size().plot(kind="bar")


# This shows that maximum games were published on Nintendo DS and Sony PS2. Sony PS2 was a Great success during the early 2000.

# Lets identify all the major publisher.

# In[ ]:


print(sales.groupby("Publisher").size().idxmax())
print(sales.groupby("Publisher").size().max())


# Now let us identify the variability of regional sales upon global sales.

# In[ ]:


sns.lmplot(x="Global_Sales", y="NA_Sales", data=sales, fit_reg=True)
sns.lmplot(x="Global_Sales", y="EU_Sales", data=sales, fit_reg=True)
sns.lmplot(x="Global_Sales", y="JP_Sales", data=sales, fit_reg=True)


# The above graphs show the dependence of sales within an area on global sales.
