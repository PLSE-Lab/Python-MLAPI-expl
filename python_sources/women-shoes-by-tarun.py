#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)


# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


data = pd.read_csv('../input/womens-shoes-prices/7210_1.csv')


# In[ ]:





# In[ ]:





# In[ ]:


data.head(2)


# In[ ]:





# In[ ]:


data2 = pd.read_csv('../input/womens-shoes-prices/7210_1.csv')


# In[ ]:


data2.head(2)


# In[ ]:


data3 = pd.read_csv('../input/womens-shoes-prices/7210_1.csv')


# In[ ]:


data3.head(2)


# In[ ]:


shoes=data3


# In[ ]:


shoes.columns


# In[ ]:


shoes.shape


# In[ ]:


shoes.isnull().sum()


# In[ ]:


shoes.drop(["Unnamed: 48","Unnamed: 49","Unnamed: 50","Unnamed: 51","vin","websiteIDs","weight","prices.warranty","prices.returnPolicy","prices.flavor","flavors","count","prices.source","isbn","prices.count","asins"],axis=1,inplace=True)


# In[ ]:


shoes.shape


# In[ ]:


shoes.columns


# In[ ]:


shoes.isnull().sum()


# In[ ]:


shoes.head()


# In[ ]:


# finding the mean of price columns grouped by brands

shoes_avg=shoes.groupby(by="brand")["prices.amountMin","prices.amountMax"].mean()
shoes_avg.head()


# In[ ]:


shoes_avg.dtypes


# In[ ]:


# this will show all the rows of our dataset 
pd.set_option("display.max_rows",None)


# In[ ]:


shoes_avg.head()


# In[ ]:


shoes_avg["Average_Price"]=(shoes_avg["prices.amountMin"]+shoes_avg["prices.amountMax"])/2


# In[ ]:


shoes_avg.head()


# In[ ]:


# sorting the values so as to find the top ten brands with highest average price


# In[ ]:


shoes_avg.sort_values(by="Average_Price",ascending=False).head(10)


# In[ ]:


# this will help us to check the skewness of out price distribution and as we can see our dataset is right skewed

sns.distplot(shoes_avg["Average_Price"],kde=True)


# In[ ]:


sns.boxplot(shoes_avg["Average_Price"])


# In[ ]:


shoes_avg.describe()    # this will identify the mean of Average price column along with other details


# In[ ]:


# lets plot to show the brandwise no of shoes that have their average price above mean average price


# In[ ]:


sns.countplot(shoes_avg["Average_Price"]>95.62)    


# In[ ]:


shoes_avg.head()


# In[ ]:


shoes.head()


# In[ ]:


shoes.columns


# In[ ]:


shoes["prices.isSale"].value_counts()    


# In[ ]:





# In[ ]:


shoes["Average_Price"]=(shoes["prices.amountMin"]+shoes["prices.amountMax"])/2


# In[ ]:


shoes.head()


# In[ ]:


# lets found out the price gap between the lowest and highest prices grouped by brand


# In[ ]:


shoes_avg.head()


# In[ ]:


shoes_avg.columns


# In[ ]:


shoes_avg["Price Gap"]=shoes_avg["prices.amountMax"]-shoes_avg["prices.amountMin"]


# In[ ]:


# These are the top 10 brands that have the largest price gaps between their highest and lowest prices


# In[ ]:


shoes_avg.sort_values(by="Price Gap",ascending=False).head(10)


# In[ ]:


# comparing highest 10 entries of Price columns

shoes_avg.sort_values(by="Price Gap",ascending=False).head(10).plot(kind="bar",stacked=True)


# In[ ]:


shoes.head()


# In[ ]:


# applying Annova test to check if brand value has impact on Average Price on our data


# In[ ]:


import scipy.stats as stats
import statsmodels.api as sms
import statsmodels.formula.api as statsmodels
from statsmodels.formula.api import ols


# In[ ]:


shoes_ol=shoes[["brand","Average_Price"]]


# In[ ]:


shoes_ols=shoes_ol.sort_values(by="brand")
shoes_ols.head()


# In[ ]:


n = 3
dfn = n-1

dfd = shoes_ols["brand"].shape[0]-2
dfd


# In[ ]:


stats.f.ppf(1-0.05, dfn=dfn, dfd=dfd)


# In[ ]:


model_b = ols("Average_Price~brand",data=shoes).fit()


# In[ ]:


print(sms.stats.anova_lm(model_b))


# In[ ]:


# now we know our null hyp. was H0 : the brand of shoes has no impact on the average price
# likewise our alternte hyp. was H1 : the brand of shoes has impact on the average price 

# according to the pvalue we can say that that we reject our null hypothesis and hence brand has impact on the average price of our shoes


# In[ ]:





# In[ ]:


shoes["brand"].value_counts().shape


# In[ ]:


model_g = ols("Average_Price~colors",data=shoes).fit()


# In[ ]:


print(sms.stats.anova_lm(model_g))


# In[ ]:


shoes[["brand","dateAdded","Average_Price"]].head()


# In[ ]:


# now lets see how we can fill the null values of brand columns

# here's the idea , we can extract the brand name string from prices.sourceURLs but in order to do so we have to create a local data frame and then use split function


# In[ ]:


shoes["prices.sourceURLs"][12]


# In[ ]:


shoes[["brand","prices.sourceURLs"]].head()


# In[ ]:


shoes_strip=shoes[["brand","prices.sourceURLs"]]   # local dataframe


# In[ ]:


shoes_strip["Stripcol"]=shoes["prices.sourceURLs"].str.split("-").str[0]     # splitting the URL at "-" 


# In[ ]:


shoes_strip.head(15)


# In[ ]:


shoes_strip["Stripcol"]=shoes_strip["Stripcol"].str.split("/").str[-1]     # again splitting the URL ar "/" and -1 indicates we start from right


# In[ ]:


shoes_strip.head(50)


# In[ ]:


shoes_strip[shoes_strip["Stripcol"]=="SNEED"].count()


# In[ ]:


for i in range(0,len(shoes_strip["Stripcol"])):
    if shoes_strip["Stripcol"][i]=="SNEED" :
        shoes_strip["brand"][i]="SNEED"

        


# In[ ]:


shoes_strip["brand"].isnull().sum()


# In[ ]:


# grouping the brand=null dataframe on the basis of brand 
shoes_strip[shoes_strip["brand"].isnull()].groupby(by="Stripcol").first()


# In[ ]:


for i in range(0,len(shoes_strip["Stripcol"])):
    if shoes_strip["Stripcol"][i]=="Nine" :
        shoes_strip["brand"][i]="Nine West"


# In[ ]:


for i in range(0,len(shoes_strip["Stripcol"])):
    if shoes_strip["Stripcol"][i]=="Mirak" :
        shoes_strip["brand"][i]="Mirak Montana"


# In[ ]:


shoes_strip["brand"].isnull().sum()


# In[ ]:


shoes_strip["brand"].fillna(100,inplace=True)    # now in order to replace "NaN" null value with our brand name lets first fill the null value with a number so that it can be used in conditional loops


# In[ ]:


shoes_strip.head()


# In[ ]:


x=len(shoes_strip["brand"])     # replacing the 100 with our required brand name


# In[ ]:


for i in range(0,x):
    for j in shoes_strip["Stripcol"]:
        if shoes_strip["brand"][i]==100:
            shoes_strip["brand"][i]=j
        break
        
            


# In[ ]:


shoes_strip["brand"].isnull().sum()


# In[ ]:


shoes_strip.head()


# In[ ]:


shoes.head()


# In[ ]:


shoes.brand=shoes_strip.brand


# In[ ]:


shoes.brand.head()


# In[ ]:


shoes.brand.isnull().sum()


# In[ ]:


shoes.colors.isnull().sum()


# In[ ]:


model_g = ols("Average_Price~brand",data=shoes).fit()


# In[ ]:


print(sms.stats.anova_lm(model_g))     # without sorting


# In[ ]:


shoes_sort=shoes.sort_values(by="brand")


# In[ ]:


model_g=ols("Average_Price~brand",data=shoes_sort).fit()   # after sorting


# In[ ]:


print(sms.stats.anova_lm(model_g))    


# In[ ]:


# lets check the corelation between numeric columns
shoes_sort.corr()


# In[ ]:


# Word Cloud of brand names

import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np


# In[ ]:


list=[]
for i in shoes["brand"] :
    list.append(i)
slist = str(list)

wordcloud = WordCloud(width=400, height=200).generate(slist)
plt.figure(figsize=(12,12))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()


# In[ ]:


shoes_scatter=shoes_avg.sort_values(by="Average_Price",ascending=False).head(5)


# In[ ]:


shoes_scatter


# In[ ]:


# Top 10 Brands with highest Average Price
list1=[]
for i in shoes_scatter.index :
    list1.append(i)
slist1 = str(list1)

wordcloud = WordCloud(width=480, height=480, max_font_size=40, min_font_size=10).generate(slist1)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()


# In[ ]:


# Much work is still remaining


# In[ ]:





# In[ ]:




