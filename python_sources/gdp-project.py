#!/usr/bin/env python
# coding: utf-8

# # PROJECT GDP
#  This project has two chapter and prepared for panda learners;
#        * First it shows how the money spread around the world.
#        * Second it shows how it is in your country
# ## Chapter-I

# ### importing libraries

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Loading Data File
# Data source: http://api.worldbank.org/v2/en/indicator/NY.GDP.PCAP.CD?downloadformat=excel

# In[2]:


#location=r"C:\Users\res\Desktop\projects\GDP\w_gdp.xls"
#location=r"http://api.worldbank.org/v2/en/indicator/NY.GDP.MKTP.CD?downloadformat=csv"


# In[3]:


df=pd.read_excel("../input/1960-2017-gdp/w_gdp.xls",
                 header=[3],index_col="Country Name")
df.head()


# In[4]:


df.iloc[10:20]


# ### Cleaning Data Frame

# In[8]:


df.info()


# In[9]:


#As we see there are some non-null values, lets clean them:


# In[10]:


df1=df.drop(df.iloc[:,:43],axis=1)


# In[11]:


df1=df1.drop("2017",axis=1)


# In[12]:


df1.head()


# In[13]:


df1.info()


# In[14]:


#There are still null values, lets get rid of them:


# In[15]:


df2=df1.dropna(thresh=2)


# In[16]:


df2=df2.interpolate(method='linear',axis=1)


# In[17]:


df2.info()


# In[18]:


df2=df2.fillna(method="bfill",axis=1)


# In[19]:


df2.info()


# In[20]:


df2.head(2)


# In[21]:


df3=df2.abs()


# In[22]:


df3=df2.astype(int)


# In[23]:


df3[:3]


# In[24]:


df3.columns
#len(df3.columns)


# In[25]:


df3.index


# In[26]:


df3.tail()


# ### Plotting DataFrame

# In[39]:


pd.set_option('display.precision', 3) # to adjust longness of the output number.
(df3.describe()/1000000).round()   # to round the long result numbers.
df3.describe()


# In[40]:


# transpose the df3 and plot it.


# In[41]:


df3.T.plot.line(figsize=(20,10))


# In[42]:


df4=df3["2016"].sort_values(ascending=False) # To sort 2016
df4.head(20)


# In[44]:


dfmean=(df3.mean(axis=1).sort_values(ascending=False))/1000000000
dfmean.head(20)


# In[45]:


dfmean.iloc[:20].plot(kind="bar",figsize=(15,10),color="orange")
plt.show()


# In[46]:


df4.iloc[0:20].T.plot.bar(figsize=(20,10),fontsize=18)
#plt.plot(dfmean,color="red")
plt.xlabel("COUNTRY NAME",fontsize=18)
plt.ylabel("10 X TRILLION DOLAR ($)",fontsize=18)
plt.title("COUNTRIES GROSS DOMESTIC PRODUCT (GDP)",fontsize=18)
plt.show()


# In[47]:


df4.iloc[0:20].T.plot.barh(figsize=(20,10),fontsize=18, color="red")
plt.xlabel("10 X TRILLION DOLAR ($)",fontsize=18)
plt.ylabel("COUNTRY NAME",fontsize=18)
plt.title("COUNTRIES GROSS DOMESTIC PRODUCT (GDP)",fontsize=18)
plt.show()


# ## Chapter-II
# > I chosed Turkey you can chose any country instead. Just change the name and run the codes.

# In[48]:


#dfc=df.loc["Turkey","1960":"2016"]
dfc=pd.DataFrame(df.loc["Turkey"],index=df.columns[3:-1])
dfc=dfc.astype(int)


# In[49]:


dfc.info()


# In[50]:


(dfc.describe()/1000000).round()


# In[51]:


# this is too small but just for the seeing the forest.
plt.plot(dfc, linestyle='dashed', linewidth=2,)
plt.show() 


# In[52]:


dfc.plot.bar(figsize=(20,10))
plt.xlabel('year', fontsize=20)
plt.ylabel('gdp', fontsize=20)
plt.title('TURKEY GDP 1960-2016')
plt.show()


# In[53]:


dfc.plot.barh(figsize=(20,10))
plt.xlabel('year', fontsize=20)
plt.ylabel('gdp', fontsize=20)
plt.title('TURKEY GDP 1960-2016')
plt.show()


# ## Bonus Chapter
# * This bonus chapter shows the growth ratio of countries between 2015-2020. Last 3 years expected growth ratio.

# In[54]:


growth=pd.read_excel(r"../input/gdp-growth/GDP_growth.xlsx")
growth.head()


# In[55]:


growth


# In[56]:


growth.info()


# In[58]:


growth=growth.sort_values(by=[2017],ascending=False)


# In[59]:


first20=growth[:20]
first20


# In[60]:


first20.plot(kind="bar",figsize=(10,5))
plt.show()


# In[61]:


# nice coding..

