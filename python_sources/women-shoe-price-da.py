#!/usr/bin/env python
# coding: utf-8

# **Importing Libraries**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sn
from wordcloud import WordCloud, STOPWORDS
from matplotlib import rcParams

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# **Read file data -" Datafiniti_Womens_Shoes.csv"and displaying its content**

# In[ ]:


df = pd.read_csv("../input/Datafiniti_Womens_Shoes.csv",low_memory=True) 
df.head()


# **Data Cleansing**

# In[ ]:


#Replacing Nan Values with 0

df.replace(np.nan, 0, inplace=True)  


# In[ ]:


df.info()


# In[ ]:


#Removing columns that are no where used in this analysis
df1 = df.drop(['prices.sourceURLs', 'upc','ean','asins','dimension'], axis=1)


# In[ ]:


#Dataframe detail with number of rows and columns count
df1.shape


# In[ ]:


#Renaming column names
df1.rename(columns={'prices.amountMax': 'amount_max'}, inplace=True)
df1.rename(columns={'prices.availability': 'availability'}, inplace=True)
df1.rename(columns={'prices.color': 'color'}, inplace=True)
df1.rename(columns={'prices.returnPolicy': 'returnpolicy'}, inplace=True)
df1.rename(columns={'prices.amountMin': 'amountMin'}, inplace=True)
df1.rename(columns={'prices.isSale': 'returnpolicy'}, inplace=True)


# In[ ]:


#Visualization of statistical Data Set value
df1.describe().plot(kind = "area",fontsize=20, figsize = (20,10), table = True,colormap="rocket")
plt.xlabel('Statistics')
plt.ylabel('Value')
plt.title("General Data Set Values")


# In[ ]:


#Correlation matrix
corrmat = df1.corr()
f, ax = plt.subplots(figsize=(15, 10))
sn.heatmap(corrmat, square=True);


# In[ ]:


#Graph Plot to display Shoe brands from data set
plt.figure(figsize=(25,15))
p = sn.countplot(df1['brand'])
p1 = plt.setp(p.get_xticklabels(), rotation=90)


# **Stock availability**

# In[ ]:


#Displaying unique values of stock availability 
df1.availability.unique()


# Assuming TRUE value here refers to the availability of the stock and updating the TRUE value to IN Stock.   
# Also 0 being  non availability of stock  i.e, Out of Stock

# In[ ]:


df1.loc[:, 'availability'].replace(['TRUE'], ['In Stock'], inplace=True)
df1.loc[:, 'availability'].replace([0], ['Out Of Stock'], inplace=True)


# In[ ]:


plt.figure(figsize=(15, 10))
x = sn.countplot(df1['availability'])
x1 = plt.setp(x.get_xticklabels(), rotation=90)
plt.title('Availability of Stock', fontsize = 25)


# In[ ]:


Avail = df1.loc[df1["availability"] == 'In Stock']; 
Avail.brand.value_counts()


# **Brands and Pricing**

# In[ ]:


print('There are', df['brand'].nunique(), 'unique values in Brand name column')
print ('Top 10 most common Brands are listed here ')
df['brand'].value_counts()[:10] #Top 10 most common Brands 


# In[ ]:


#Result of average price of each distinct brand listed 
average=df1.groupby('brand')['amount_max'].mean().sort_values(ascending=False)
average


# In[ ]:


#These are the top 10 brands with highest average price
brand_average=df1.groupby('brand')['amount_max'].mean().sort_values(ascending=False).head(10)
brand_average


# In[ ]:


#Plot graph from above  average list
 g = sn.distplot(brand_average, hist=False, color="Red", kde_kws={"shade": True})


# In[ ]:


fig = plt.figure(figsize=(20,10))
brand_average.plot(kind='bar', align='center', alpha=.8)
plt.title('Average Price of Brands')


# **Return Policy **

# **Colors**

# In[ ]:


# Using Wordcloud - displaying Colors
para_docs=(df1.color.astype('str'))
oc_cloud= WordCloud(background_color='white',stopwords= STOPWORDS, max_words=100,max_font_size=50,random_state=1).generate(str(para_docs))
 #Generate the wordcloud output
plt.imshow(oc_cloud)
plt.axis('off')


# In[ ]:


df1.color.value_counts()


# In[ ]:


# Top 5 Shoe colors 
Black= df1[df1.color =="Black"]
Grey= df1[df1.color =="Gray"]
Taupe= df1[df1.color =="Taupe"]
Navy= df1[df1.color =="Navy"]
Brown=df1[df1.color =="Brown"]


# In[ ]:


#Black
Black.brand.value_counts()


# In[ ]:


sn.distplot(df1[df1['color'] =='Black']['amount_max'], rug=True)


# In[ ]:


#Graph for Black color shoe collection - top 10 list
Black['brand'].value_counts()[:10].plot(kind='barh')
plt.title('Black color shoes data', fontsize = 25)
plt.xlabel('Count', fontsize = 10)
plt.ylabel('Brands', fontsize = 10)
plt.show()


# In[ ]:


#Grey
Grey.brand.value_counts()


# In[ ]:


sn.distplot(df1[df1['color'] =='Grey']['amount_max'], rug=True)


# In[ ]:


#Graph for Grey color shoe collection - top 10 list
Grey['brand'].value_counts()[:10].plot(kind='barh')
plt.title('Grey color shoes data', fontsize = 25)
plt.xlabel('Count', fontsize = 10)
plt.ylabel('Brands', fontsize = 10)
plt.show()


# In[ ]:


#Taupe
Taupe.brand.value_counts()


# In[ ]:


sn.distplot(df1[df1['color'] =='Taupe']['amount_max'], rug=True)


# In[ ]:


#Graph for Taupe color shoe collection - top 10 list
Taupe['brand'].value_counts()[:10].plot(kind='barh')
plt.title('Taupe color shoes data', fontsize = 25)
plt.xlabel('Count', fontsize = 10)
plt.ylabel('Brands', fontsize = 10)
plt.show()


# In[ ]:


#Navy
Navy.brand.value_counts()


# In[ ]:


sn.distplot(df1[df1['color'] =='Navy']['amount_max'], rug=True)


# In[ ]:


#Graph for Navycolor shoe collection - top 10 list
Navy['brand'].value_counts()[:10].plot(kind='barh')
plt.title('Navy color shoes data', fontsize = 25)
plt.xlabel('Count', fontsize = 10)
plt.ylabel('Brands', fontsize = 10)
plt.show()


# In[ ]:


#Brown
Brown.brand.value_counts().head(10)


# In[ ]:


sn.distplot(df1[df1['color'] =='Brown']['amount_max'], rug=True)


# In[ ]:


#Graph for Brown color shoe collection - top 10 list
Brown['brand'].value_counts()[:10].plot(kind='barh')
plt.title('Brown color shoes data', fontsize = 25)
plt.xlabel('Count', fontsize = 10)
plt.ylabel('Brands', fontsize = 10)
plt.show()


# In[ ]:


#Wordcloud on Brands
brands= ' '.join(df1['brand'].str.lower())
a = STOPWORDS
a.add('will')
wordcloud = WordCloud(stopwords=a, background_color="white", max_words=1000).generate(brands)
rcParams['figure.figsize'] = 20, 60
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# **Conclusion:**
# 
# Journee Collection and Lifestride Brands are under top 10 most common brands with huge collection of Shoes offering wide range of colors.
# 
# Red wing and Lowa brands are in the list of highest average price.
# 
# Salomon and Brinley Co.  brands has highest Stock availability.
# 
# Also to note no Return policy avail on the stocks from dataset.
# 
# Please upvote if you like this Kernel.
# Happy Kaggling!

# In[ ]:




