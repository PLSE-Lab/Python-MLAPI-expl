#!/usr/bin/env python
# coding: utf-8

# ## My first Market Basket Analysis
# 
# I will use here some simple descriptive statistics and a market basket analysis to find out some things about the customers and the store. 
# 
# Let's import some important packages and read the input file. 
# 
# ![](https://d2fkddr0p2jbv6.cloudfront.net/render/standard/LPDqTkId2dNtSfoYXNORbMeVQcmGX8lLgLPdyAHpuXmdheQsxliHoFhBbzcU1vnQ/mug11oz-whi-z1-t-but-first-covfefe.jpg)

# In[ ]:


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv("../input/BreadBasket_DMS.csv")


# In[ ]:


df.head()


# ## Check the Data
# 
# Now you can see that we have some columns that contain the date, the time and the purchased product. The transaction indicates whether it was the same purchase, i.e. products with the same number were purchased together.
# 
# First, I will check if there are any NaN values. There are no missing values but if you look at the values in the "Item" column you can see that there are some items which are named "NONE". Since I don't know exactly what is meant by this, we will simply remove these purchases from the dataset.

# In[ ]:


df.isnull().any()


# In[ ]:


df.Item.unique()


# In[ ]:


df.loc[(df['Item']=="NONE")].head()


# In[ ]:


to_drop = df.loc[(df['Item']=="NONE")].index

df.drop(to_drop, axis = 0, inplace = True)

print ("Lines dropped")


# ## Feature Engineering
# 
# Our dataset is clean now. Now we will look at the other columns. Since the date, as well as the time can't be used in this shape, I will divide them into different columns.
# 
# But first, I use the "to_datetime" function of pandas to assign the weekdays to the date. Maybe sunday there are more customers than thursday. We will see...
# 
# In addition I make a rough division of the date into seasons, since I could imagine that other products sell well in winter than, for example, in summer. The biggest problem with the dataset, however, is that we only have transactions for a few months.

# In[ ]:


df["date"] = pd.to_datetime(df['Date'])
df["dayname"] = df["date"].dt.day_name()
df.drop("date", axis=1, inplace = True)
df.head()


# In[ ]:


df["year"], df["month"], df["day"] = df["Date"].str.split('-').str
df.drop("Date", axis=1, inplace = True)

df["hour"], df["minute"], df["second"] = df["Time"].str.split(':').str
df.drop("Time", axis=1, inplace = True)

df.head()


# In[ ]:


#Season
df["month"] = df["month"].astype(int)
df.loc[(df['month']==12),'season'] = "winter"
df.loc[(df['month']>=1) &  (df['month']<=3),'season'] = "winter"
df.loc[(df['month']>3) &  (df['month']<=6),'season'] = "spring"
df.loc[(df['month']>6) &  (df['month']<=9),'season'] = "summer"
df.loc[(df['month']>9) &  (df['month']<=11),'season'] = "fall"

df.head()


# # Descriptive Statistics
# 
# ## Do we have positive growth?
# 
# First we look at simple frequency distributions. It would be interesting to know whether the transactions have increased. So let's compare 2016 to 2017. It looks like we're right. 
# 
# But be careful! It could be that we simply have more months from 2017 in our dataset than from 2016. So let's take a look at the distribution over the months. 
# 
# As you can see we were right. There are simply more months from 2017 in the dataset and it even shows that most transactions took place in November 2016. But after December the sellings are slightly increasing. For October and April please note that these are not completely in the dataset. However, year-on-year, the average items per transaction increased slightly in 2017.
# 
# If we now divide this into categories, we see that more transactions take place simultaneously in winter than in autumn. This could be because more people are inside in winter and come to the bakery for coffee and cake instead of doing something outside. 

# In[ ]:


sns.countplot(df["year"].astype(int))
plt.show()


# In[ ]:


sns.countplot(df["month"].astype(int))
plt.show()


# In[ ]:


sns.barplot(df["year"].astype(int), df["Transaction"].value_counts())
plt.show()


# In[ ]:


sns.barplot(df["season"], df["Transaction"].value_counts())
plt.show()


# ## When are customers the most eager to buy?
# 
# Now it would be interesting to know when customers buy the most. This could be used, for example, to align offers or set certain incentives to support this behaviour or to increase sales during this time. 
# Let's check this!
# 
# You can see that the absolute transactions decrease towards the end of the month. However, the number of transactions remains largely the same. This indicates that the number of customers decreases towards the end of the month. 
# 
# Let's have a quick look to see if it's not due to the fact that some months are not completely included. 
# No, the picture is the same even then, but a little clearer. 
# 
# 
# 
# **How can this be interpreted? **
# 
# Since it is constantly going down it cannot be about effects of single weekdays e.g. that the end of the month often falls on a Saturday or similar. 
# 
# One could interpret it in such a way that towards the end of the month less budget is available with the customer and this then possibly less often a coffee goes or his bread rather in the Discounter buys. Perhaps we should examine the shopping baskets in the comparison of the days of the month.

# In[ ]:


sns.countplot(df["day"].astype(int))
plt.show()


# In[ ]:


sns.barplot(df["day"].astype(int), df["Transaction"].value_counts())
plt.show()


# In[ ]:


df_month = df[(df['month']>=1) & (df['month']<=3) | (df['month']>=11) & (df['month']<=12)]
sns.countplot(df_month["day"].astype(int))
plt.show()


# **Weekdays**
# 
# Let's take a look at the days of the week. In my experience, many people tend to buy fresh bread rolls or the like on weekends, so I can imagine that more sales will be made. On the other hand, many people buy something on their way to work in the morning. 
# 
# You can clearly see that more products are sold on weekends, as we suspected. You can see that the number of transactions remains the same, which suggests that simply more customers are coming. 

# In[ ]:


sns.barplot(df["dayname"], df["Transaction"].value_counts())
plt.show()


# In[ ]:


sns.barplot(df["dayname"], df["Transaction"].value_counts())
plt.show()


# **When should we use more employees?**
# 
# Now that we know that we have more customers at the weekend and fewer people at the end of the month, we should check when we have peak times. This is important e.g. for the personnel planning, that one distributes his capacities well and not one person is there alone for the rush hour.
# 
# We see that our peak is late in the morning and then decreases again in the early afternoon. We could check this for individual days to see if it changes on weekdays, for example, because people get up earlier.

# In[ ]:


sns.countplot(df["hour"].astype(int))
plt.show()


# In[ ]:


sns.barplot(df["hour"].astype(int), df["Transaction"].value_counts())
plt.show()


# ## Do we have differences in the seasons? 
# 
# Before we move on to the right market basket analysis, I get an overview of the products sold. I take a look at the top sellers, depending on the season, because I suspect that other types of products sell better around Christmas, like in autumn. 
# 
# At the other hand, I check for the worst selling products, maybe we can remove these products from our offer or see why they work so badly. Worst selling products can mean losses. For example, the brioche with salami sells very poorly, but if we consider that you still need fresh brioche and fresh salami there and these are only limited shelf life, such a product can quickly lead to losses.
# 
# As you can see, there are some products like Coffee, Bread or Tea that sell very well all year round and can be considered top sellers. On the other hand, there also seem to be seasonal products, such as hot chocolate, which breaks down in sales in spring when the temperature gets warmer.
# 
# It should also be noted that some products may not have been included in the bakery's range until later in the year and could not even be purchased, for example, in autumn. However, we have no information about this. 

# In[ ]:


df["Item"].value_counts()[:10].plot(kind="bar")
plt.show()


# In[ ]:


values = df.Item.loc[(df['season']== "fall")].value_counts()[:10]
labels = df.Item.loc[(df['season']== "fall")].value_counts().index[:10]

plt.pie(values, autopct='%1.1f%%', labels = labels,
        startangle=90)

plt.show()


# In[ ]:


values = df.Item.loc[(df['season']== "winter")].value_counts()[:10]
labels = df.Item.loc[(df['season']== "winter")].value_counts().index[:10]

plt.pie(values, autopct='%1.1f%%', labels = labels,
         startangle=90)

plt.show()


# In[ ]:


values = df.Item.loc[(df['season']== "spring")].value_counts()[:10]
labels = df.Item.loc[(df['season']== "spring")].value_counts().index[:10]

plt.pie(values, autopct='%1.1f%%', labels = labels, startangle=90)

plt.show()


# In[ ]:


top_spring = df.Item.loc[(df['season']== "spring")].value_counts()[:10] / sum(df.Item.loc[(df['season']== "spring")].value_counts()) *100 
top_fall = df.Item.loc[(df['season']== "fall")].value_counts()[:10] / sum(df.Item.loc[(df['season']== "fall")].value_counts()) *100 
top_winter = df.Item.loc[(df['season']== "winter")].value_counts()[:10] / sum(df.Item.loc[(df['season']== "winter")].value_counts()) *100 
top_overall = df.Item.value_counts()[:10] / sum(df.Item.value_counts()) *100 
topseller = pd.DataFrame([top_spring, top_fall, top_winter, top_overall], index = ["Spring", "Fall", "Winter", "Overall"]).transpose()
topseller


# In[ ]:


worst_spring = df.Item.loc[(df['season']== "spring")].value_counts()[-10:] / sum(df.Item.loc[(df['season']== "spring")].value_counts()) *100 
worst_fall = df.Item.loc[(df['season']== "fall")].value_counts()[-10:] / sum(df.Item.loc[(df['season']== "fall")].value_counts()) *100 
worst_winter = df.Item.loc[(df['season']== "winter")].value_counts()[-10:] / sum(df.Item.loc[(df['season']== "winter")].value_counts()) *100 
worst_overall = df.Item.value_counts()[-10:] / sum(df.Item.value_counts()) *100 
worst = pd.DataFrame([worst_spring, worst_fall, worst_winter, worst_overall], index = ["Spring", "Fall", "Winter", "Overall"]).transpose()
worst


# # Market Basket Analysis
# ## Have a look at the products
# 
# First you see that we have many customers who only buy one product. However, there are also many who buy two or more products. So it would be interesting to find out which products are bought together or not? We could bundle them into offers or improve our profits through price discrimination. 
# 
# 

# In[ ]:


sns.countplot(df["Transaction"].value_counts())
plt.show()


# **Combine all products from one transaction**
# 
# What I do now is to group all products that belong to the same shopping cart and then hotencode them. After that I use "Apriori" to create rules to analyse the market basket.

# In[ ]:


from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# **Results**
# 
# Now let's analyze the basket. Since I analyze the entire data set once and then the data set divided into seasons, I have written a few functions for it. In principle, the products of a related transaction are first written in a single row and then hot encoded. Finally, they are analyzed with "Apriori" or rules are created. 
# 
# As you can see, the most popular shopping baskets are all a combination with a coffee. This suggests that many people eat something on the spot or take a lunch with them and don't just come to buy bread or something similar. 

# In[ ]:


overall = df
fall = df[df["season"]=="fall"]
winter = df[df["season"]=="winter"]
spring = df[df["season"]=="spring"]

def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

def apri(data):
    encoding = data.groupby(['Transaction','Item'])['Item'].count().unstack().reset_index().fillna(0).set_index('Transaction').astype(int)
    encoding = encoding.applymap(encode_units)
    frequent_itemsets = apriori(encoding, min_support=0.01, use_colnames=True)
    rules_1 = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    output = rules_1.sort_values(by=['confidence'], ascending=False)
    return output

print("Model ready")


# In[ ]:


apri(overall).head(10)


# **Differences between the seasons**
# 
# Finally, I would like to see if there are differences between different times of the year in terms of the products purchased. For this I will display the product combinations which are not contained in both dataframes.
# 
# First of all, it can be said that there are some products that are likely to become more popular in specific seasons. For example, hot chocolate is particularly popular in winter, while in other seasons it is not so often bought together with coffee. But why buy two hot drinks? Usually this will be the case when several people meet there.
# 
# It should be noted, however, that we are only looking at the most popular combinations. In addition, there are some products that are probably not real products but have been created in the POS system such as "Keeping It Local". 

# In[ ]:


def compare(season_one, season_two): 
    dataframe =  pd.concat([apri(season_one).head(10), apri(season_two).head(10)]).drop_duplicates(subset = "antecedents", keep=False).sort_values(by=['confidence'], ascending=False)
    return dataframe


# In[ ]:


compare(winter, spring)


# In[ ]:


compare(winter, fall)


# In[ ]:


compare(spring, fall)


# # Conclusion
# 
# At the beginning our goal was to learn something about our company. After a few lines of code we can now make important statements. Now we know when most customers come to us, which products sell particularly well at which times of the year and in general what our top sellers are. In addition, we know important product combinations that we can use to create additional offers or incentives to increase our sales. Finally, we also know that there are some products that hardly anyone buys, but need some fresh raw materials, so we should make sure they don't have a negative impact on our profits. 
# 
