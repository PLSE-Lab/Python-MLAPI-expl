#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import files and Print 'Setup Complete' to verify import
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from pandas import Series, DataFrame

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
print("Setup Complete")


# In[ ]:


#Import file from Directory and check that it loaded properly
df = pd.read_csv('../input/chocolate-bar-ratings/flavors_of_cacao.csv')
df.head()


# In[ ]:


#rename columns for easier data manipulation
df.columns = ['company', 'origin', 'ref', 'review_date', 'cocoa_percent', 'company_location', 'rating', 'bean_type', 'bean_origin']
#check to confirm
df.head()


# In[ ]:


#remove percentage symbol from 'cocoa_percent' for easier python syntax and change the column to a float
df['cocoa_percent']=(df['cocoa_percent']).str.replace('%', '')
df['cocoa_percent']=(df['cocoa_percent']).astype(float)
#check to confirm
df.head()


# In[ ]:


df.head()


# In[ ]:


#Let's see a summary of of all the columns in the table
print("Table 1: Summary of Statistical Measurements")
df.describe(include='all').T


# We see that the company that produced most chocolate bars is Soma while **most of the cocoa beans are from Venezuela**. We also see that the **average chocolate bar contains 71% cocoa** while **consumers give an average rating of 3.1 to chocolate bars**.
# 

# In[ ]:


#Let's analyze some of these findings with visualizations
#To compare the number of users that rated chocolate bars with their ratings
sns.countplot(x='rating', data=df)
plt.xlabel('Rating')
plt.ylabel('Number of consumers')
plt.title('Number of consumers that rated Chocolate Bars')
print('Fig 1: Chocolate bar ratings')


# Out of a total 1795 chocolate bars that were rated from 63 countries. Most of the ratings given are between 3.0 and 3.5 with a rating of 3.5 given by about 380 consumers.
# 
# > *Most consumers considered the chocolate bars to be somewhere between Satisfactory(3.0) & Praiseworthy(3.75)*
# 

# In[ ]:


#Now, Let's find out how the percentage of cocoa affects the rating of a chocolate bar
sns.lmplot(x='rating', y='cocoa_percent', fit_reg=False,scatter_kws={"alpha":0.3,"s":100}, data=df)
plt.xlabel('Rating')
plt.ylabel('Percentage of Cocoa in Chocolate')
plt.title('Relationship between percentage of Cocoa and Chocolate rating')
print('Fig 2: Chocolate bar ratings and Cocoa Percentage')


# The Scatterplot shows that the density of the graph is highest between 65% and 80% of cocoa. Meaning chocolate bars with low cocoa percentage(less than 50%) and high cocoa percentage(above 90%) are less in number, but the most important fact is that most of these chocolate bars have a rating of less than 3. 
# 
# > *People prefer chocolate bars with not so much or not so little cocoa percentages. Usually around 70-75% cocoa composition*
# 

# In[ ]:


#Let's see how the chocolate bars are spread based on their cocoa percentage.
#sns.countplot(x='cocoa_percent', data=df)
sns.distplot(a=df['cocoa_percent'], hist=True, kde=False)
plt.xlabel('Percentage of Cocoa')
plt.ylabel('Number of chocolate bars')
plt.title('Count of chocolate bars and their percentage of cocoa')
print('Fig 3: Chocolate bar cocoa percentage')


# Close to 700 chocolate bars had a cocoa composition of 70%, while less than 100 bars had 60% or less cocoa. 

# # **Who Produces the best choclates?**

# In[ ]:


# Let's see the top 5 companies in terms of reviews
companies=df['company'].value_counts().index.tolist()[:5]
satisfactory={} # empty dictionary
for j in companies:
    c=0
    b=df[df['company']==j]
    br=b[b['rating']>=3] # rating more than 4
    for i in br['rating']:
        c+=1
        satisfactory[j]=c    
print(satisfactory)


# In[ ]:


#Visualizing the data above
li=satisfactory.keys()
plt.figure(figsize=(10,5))
plt.bar(range(len(satisfactory)), satisfactory.values())
plt.xticks(range(len(satisfactory)), list(li))
plt.xlabel('Company')
plt.ylabel('Number of chocolate bars')
plt.title("Top 5 Companies with Chocolate Bars Rating above 3.0")
print('Fig 4: Best chocolate companies')


# The company, Soma has the largest number of chocolate bars that have a rating above 3.0
# 

# In[ ]:


d2 = df.sort_values('cocoa_percent', ascending=False).head(6)
plt.figure(figsize=(15, 4))
sns.barplot(x='company', y='cocoa_percent', data=d2)
plt.xlabel("Chocolate Company")
plt.ylabel("Cocoa Percentage")
plt.title("Top 5 Companies in terms of Cocoa Percentage")
print('Fig 5: Top companies with Cocoa Percentage')


# We see that Domori, C-Amaro, Artisan du Chocolat, Sirene and Claudio Corallo are the top 5 companies with a 100% cocoa in their chocolate bars. 

# In[ ]:


plt.figure(figsize=(15, 4))
ax = sns.lineplot(x='review_date', y='cocoa_percent', data=df)
plt.xlabel("Year of Review")
plt.ylabel("Cocoa Percentage")
plt.title("Cocoa Percentage patterns over the years")
print('Fig 6: Pattern of Cocoa percentage over the years')


# 2008 had the highest number if cocoa percentages in chocolate bars at about 73% and it was followed by a steep reduction the very next year with the lowest percentage of cocoa in 2009 at a 69% cocoa composition. Between 2009 to 2013 the cocoa composition of chocolate bars rose to about 72.2% from 69%. From 2014, a steady decline in cocoa percentage in chocolate bars have been noticed and in 2017, it stands at just above 71.5%.
# 
# So how has this fluctuation in Cocoa percentage composition affected chocolate bar ratings?

# In[ ]:


plt.figure(figsize=(15, 4))
ax = sns.lineplot(x='review_date', y='rating', data=df)
plt.xlabel("Year of Review")
plt.ylabel("Ratings")
plt.title("Rating patterns over the years")
print('Fig 7: Pattern of Chocolate bar ratings over the years')


# We notice that the lowest ever rating was around 3 and it came in 2008. From 2008 to 2011, there was a steady increase in average ratings and in 2011 it was at 3.26. There has been several fluctuations in the ratings between 2011 & 2017 and in 2017, the rating lies at its highest at around 3.31.

# # So is there a relationship between the percentage of Cocoa in a chocolate bar and its rating?
# 
# The highest percentage of cocoa in choclolate bars came in 2008 and this was the year with the lowest ratings.
# We also notice two coreelations between 2008 and 2009:
# 1. There was a drastic reduction in the average percentage of cocoa in chocolate bars.
# 2. The average rating had a very steep increase to 3.08 in 2009 from 3.00 in 2008.
# 
# Is this an indication that the rate of a chocolate bar is proportional to the percentage of cocoa in it. It also looks like the percentage of cocoa that is considered a "tipping point" is 80% (I have a personal Experience on this).

# In[ ]:


#To confirm this speculation, let's see
cocoa_seventy=df[df['cocoa_percent' ] == 70.0]
cocoa_one_hundred=df[df['cocoa_percent' ] == 100.0] 
cocoa_seventy.count()
cocoa_one_hundred.count()
sns.countplot(x='rating', data=cocoa_seventy, color='orange')
sns.countplot(x='rating', data=cocoa_one_hundred, color='red')
print('Fig 8: Ratings of Chocolate Bars with 70% & 100% Cocoa')


# # Which country produces chocolates the most?

# In[ ]:



print ('Top Chocolate Producing Countries in the World\n')
country=list(df['company_location'].value_counts().head(10).index)
choco_bars=list(df['company_location'].value_counts().head(10))
prod_ctry=dict(zip(country,choco_bars))
print(df['company_location'].value_counts().head(10))


# In[ ]:


#Let's visualize this

plt.figure(figsize=(10,5))
plt.hlines(y=country,xmin=0,xmax=choco_bars)
plt.plot(choco_bars, country)
plt.xlabel('Number of chocolate bars')
plt.ylabel('Country')
plt.title("Top Chocolate Producing Countries in the World")
print('Fig 8: Countries with highest chocolate producing companies')


# With about 764 chocolate bar produced, the graph shows that the U.S.A has more chocolate companies than any other country. Closely followed by France and Canada.

# # Conclusion
# There is a direct correlation between the percentage of cocoa in a chocolate bar and the average rating of that bar. We also found that the U.S.A had the highest amount of Chocolate producing companies, this is presumed to be due to the high volume of demand for chocolate in the country. 

# In[ ]:




