#!/usr/bin/env python
# coding: utf-8

# **Exploratory Data Analysis**
# by [Iqrar Agalosi Nureyza](http://https://www.kaggle.com/iqrar99)

# Hello Everyone! I try my best to do data analysis. This Ramen Rating Dataset is a very good dataset for beginner and I hope you can understand my analysis. I am very open in accepting criticism and suggestions for perfecting this kernel.
# 
# If you find this notebook useful, feel free to **Upvote**.

# ## Basic Analysis

# In[ ]:


#importing all important packages
import numpy as np #linear algebra
import pandas as pd #data processing
import matplotlib.pyplot as plt #data visualisation
import seaborn as sns #data visualisation
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv('../input/ramen-ratings/ramen-ratings.csv') #reading the data and save it into a variable
data.head(10) #show the first 10 rows of the data


# In[ ]:


#checking total rows and column in our data
data.shape


# Alright, we know that we got 7 columns in our data. Which are:
# 1. *Review* : unique numbers that inform the review order from the latest
# 2. *Brand*  : Ramen brand
# 3. *Variety*: variation of ramen
# 4. *Style*  : style of ramen
# 5. *Country*: Where the ramen is available
# 6. *Stars*  : Ramen ratings
# 7. *Top Ten*: ramen achievement

# ## Data Cleaning
# Let's check if there are data missing in our data

# In[ ]:


data.isna().sum()


# We know that in the *Style* column we have 2 missing data. And *Top Ten* data tells us that if the ramen doesn't get 'Top Ten' achievement, then the data will be blank. Since we only have 2 missing data, we can drop it.

# In[ ]:


data = data.dropna(subset=['Style'])
print(data["Style"].isna().sum())


# ## Frequency
# Let's see all styles in *Style* column

# In[ ]:


data['Style'].unique()


# Ok, now we use *value_counts()* to count each unique Ramen Style.

# In[ ]:


data['Style'].value_counts()


# Pack Style is the most used style. Now let's see all countries in *Country* column.

# In[ ]:


print(data["Country"].unique())
print(len(data["Country"].unique()), 'Countries')


# And again, we use *value_counts()*

# In[ ]:


data['Country'].value_counts()


# Since Ramen is very popular in Japan, there is no doubt Japan will be the top one.

# ## The Top 10
# Let's see all top 10 ramen for each year.

# In[ ]:


top10 = data.dropna()
top10


# Take a look at the data. We found '\n' or *newline* in our data. Maybe it accidentally happened. So we easily filter the data using selection.

# In[ ]:


top10 = top10[top10['Top Ten'] != '\n'] #if the data in Top Ten column contains '\n' we can ignore it
top10 = top10.sort_values('Top Ten' ) #and we sort it by year
top10


# ## Top 10 Ramen Brand by total products
# Top 10 Ramen brands that have the most products.

# In[ ]:


data['Brand'].value_counts()[:10]


# ## Top 50 Ramen Ratings by brands
# We will calculate the average rating for each brand, and then we'll show who is in top 50.

# In[ ]:


#First, let's see how many ramen brands are in our data
print(len(data['Brand'].unique()))


# Let's see if *Stars* column has invalid value.

# In[ ]:


for s in data['Stars']:
    try:
        s = float(s)
    except:
        print(s)


# Okay, we found anomalies in our data. Some ramen don't have ratings. We can drop it from our data.

# In[ ]:


data = data[data['Stars'] != 'Unrated']
print(data[data['Stars'] == 'Unrated']['Stars'].sum()) #make sure if there are no 'Unrated'


# Ok, finally they're gone. It's time to have fun!

# In[ ]:


brands = list(data['Brand'].unique())
counter = [0.0]*355

brands_cnt = dict(zip(brands, counter)) #create dictionary to count all ratings and then save the averages

for brand in brands:
    brands_data = data[data['Brand'] == brand]
    for star in brands_data['Stars']:
        brands_cnt[brand] += float(star) #count all ratings
    brands_cnt[brand] /= len(brands_data) #average


# In[ ]:


top50ratings = [] #list for saving the brand name and its average rating
for key, values in brands_cnt.items():
    top50ratings.append([key,values])

#print the top 50 ramen ratings by brand
top50ratings = sorted(top50ratings, key = lambda x : x[1], reverse = True) #sorting values in descending order
top50ratings
for i in range(50):
    print('#{:<3}{:25} {}'.format(i+1, top50ratings[i][0], round(top50ratings[i][1],2)))


# _________________________

# ## Data Visualisation
# Now we move into interesting part. First, we make count plots to see value counts for each country
# #### Count PLot

# In[ ]:


sns.set(style = 'darkgrid')
f, ax = plt.subplots(1,1,figsize = (15,5))
sns.countplot(x = 'Country', data = data)
plt.xticks(rotation=90)

plt.show()


# #### Pie Plot
# If you want to plot something about percentage, then Pie Plot is a right choice. let's find out the percentage of total ramen based on style first.

# In[ ]:


labels = 'Pack', 'Bowl', 'Cup' , 'Tray', 'Box' #We can't include 'Bar' and 'Can' because they only appear once in our data.
size = [1531, 481, 450, 108, 6]

f, ax = plt.subplots(1,1, figsize= (10,10))

ax.pie(size, labels = labels, autopct = '%1.2f%%', startangle = 180)
ax.axis('equal')
ax.set_title("Style", size = 20)

plt.show()


# ___________________

# Ok, that's all my analysis. If you think I missed something, feel free to comment. Thank you for reading this notebook!
