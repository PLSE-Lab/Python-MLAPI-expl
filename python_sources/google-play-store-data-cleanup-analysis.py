#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


print(os.listdir('../input'))


# In[ ]:


df = pd.read_csv('../input/googleplaystore.csv')


# In[ ]:


df.info()


# In[ ]:


df.head()


# Hello everyone,
# 
# Welcome to one of my other early projects. Here we perform analysis on the Google Play store data.
# 
# Objectives:
# 
# Data cleanup - remove NA, filling in missing values, and change data types of some columns.
# 
# EDA - explore data to gain more knowledge and to derive insights.
# 
# How to fill in missing values?
# - search for it online if there's a couple of values missing in the column
# - Use the mean, median, or mode to fill in the missing values.
# - Or just drop them... if there's too many and we fill them with the mean, it's going to make the data look wonky.
# 
# 

# In[ ]:


df[df['Type'].isna()]


# A quick google search shows that C&C is a free to download game. It's just one row of data, but it was a quick fix.

# In[ ]:


df.replace({'Type': None}, 'Free', inplace=True)


# In[ ]:


df[df['Content Rating'].isna()]


# This whole entire row is a total mess. It looks like the columns shifted over to the left.

# In[ ]:


def shift_right(column):
    new_columns = [column[0]]
    counter = 1
    for i in range(1, len(column)):
        if i == 1:
            new_columns.append(None)
        else: 
            new_columns.append(column[counter])
            counter += 1
    return new_columns
        
    


# In[ ]:


x = df[df['Content Rating'].isna()].values


# In[ ]:


shifted = shift_right(x[0])


# In[ ]:


df[df['Content Rating'].isna()] = shifted


# In[ ]:


df[df['App'] == 'Life Made WI-Fi Touchscreen Photo Frame']


# We managed to shift the columns over in quite a contrived way. There's still missing data in there, and in hindsight, I should have just dropped this row, but now I'm rather invested in it, and it would be good practice to use some deduction to fill out the gaps.

# In[ ]:


df['Category'].unique()


# In[ ]:


df.replace({'Category': None}, 'PHOTOGRAPHY', inplace=True)


# Based on the name of the app, I assumed it's category as PHOTOGRAPHY.

# In[ ]:


df['Genres'].unique()


# In[ ]:


df.replace({'Genres': None}, 'Photography', inplace=True)


# In[ ]:


len(df[(df['Current Ver'].isna()) | (df['Android Ver'].isna())])


# It seems like we got quite a few rows with no Current Ver or Android ver. Finding these data will require us to research them individually, which could take too much time. Our dataset has over 10,000 rows so it's no major loss if we drop them (even though I went through some trouble finding data for 2 rows)
# 

# In[ ]:


df.dropna(inplace=True)


# In[ ]:


df['Size'].unique()


# Our next step is to make all the data in 'Size' the same type. It seems like the best way to do this is to convert them all to M without the letter 'M' so it can be an integer (and it's more scalable)

# In[ ]:


def size_convert(size):
    
   return "".join(list(size)[:-1])
    
    


# In[ ]:


k = df[df['Size'].apply(lambda x : list(x)[-1] == 'k')]


# In[ ]:


k['Size'] = k['Size'].apply(size_convert)


# In[ ]:


k


# In[ ]:


k['Size'] = k['Size'].astype(float)


# In[ ]:


k['Size'] = k['Size'] / 1000


# In[ ]:


m = df[df['Size'].apply(lambda x : list(x)[-1] == 'M')]
m['Size'] = m['Size'].apply(lambda x : "".join(list(x)[:-1]))


# Okay, so we managed to get them all converted to M with no letters behind them. We can convert them to int in a bit, but right now I want to fill in the 'Sizes' with 'Varies with device'. My plan is to find the average of the apps sizes by 'Category', and then use that to fill in the missing values.

# In[ ]:


m['Size'] = m['Size'].astype(float)


# In[ ]:


size_df = pd.concat([k, m ])


# In[ ]:


size_df['Size'] = size_df['Size'].astype(float)


# In[ ]:


sns.set_style('darkgrid')

plt.figure(figsize=(16, 4))
plt.xticks(rotation=90)
sns.boxplot(y='Size', x='Category', data=size_df)


# It appears the majority of the apps are under 50M, except for Games. We will be grabbing the mean of each category and replae "varies with device" sizes.

# In[ ]:


size_dict = size_df.groupby('Category').mean().to_dict()['Size']


# In[ ]:


vwd = df[df['Size'] == 'Varies with device']


# In[ ]:


vwd['new_size'] = vwd['Category'].map(size_dict)


# In[ ]:


vwd['Size'] = vwd['new_size']


# In[ ]:


vwd.drop('new_size', axis=1, inplace=True)


# In[ ]:


new_df = pd.concat([vwd, k, m])


# In[ ]:


new_df = new_df.reset_index().sort_values(by='index', ascending=True)


# In[ ]:


new_df.index = new_df['index']


# In[ ]:


new_df.head()


# In[ ]:


new_df.drop('index', axis=1, inplace=True)


# In[ ]:


new_df.head()


# In[ ]:


new_df['Size'] = new_df['Size'].round(2)


# In[ ]:


plt.figure(figsize=(16, 4))
plt.xticks(rotation=90)

sns.boxplot(y='Size', x='Category', data=new_df)


# Neat, the 'varies with device' is filled out with the average size. It should be noted that the boxplot is skewed lower.

# In[ ]:


new_df.info()


# In[ ]:


new_df['Installs'].unique()


# In[ ]:


def remove_mark (installs):
    numbers = list(installs)[:-1]
    
    while ',' in numbers:
        numbers.remove(',')
    return "".join(numbers)
    


# In[ ]:


new_df['Installs'] = new_df['Installs'].apply(remove_mark)


# In[ ]:


new_df['Installs'] = new_df['Installs'].astype(int)


# In[ ]:


order = new_df.groupby('Installs').count().sort_values(by='App',ascending=True).index.values

plt.xticks(rotation=90)

sns.countplot(x='Installs', data=new_df, order=order)


# We managed to convert the number of installations into integers. Values that could be fed into the machine. These could also be an important variable because apps that are commonly installed are most likely to be well rated.

# In[ ]:


def remove_dollar_sign(price):
    if price == '0':
        return price
    else:
        return "".join(list(price[1:]))


# In[ ]:


new_df['Price'] = new_df['Price'].apply(remove_dollar_sign)


# In[ ]:


new_df['Price'] = new_df['Price'].astype(float)


# In[ ]:


new_df.info()


# In[ ]:


new_df['Reviews'] = new_df['Reviews'].astype(int)
new_df['Rating'] = new_df['Rating'].astype(float)


# Okay, so we managed to make all numberical values truly numeric (no symbols or commas). At this point before we do some dummy transformations, I'm doing to do some EDA to generate more insights.

# In[ ]:


sns.pairplot(new_df[['Reviews', 'Size','Installs', 'Price', 'Rating']])


# Using the pair plot is a quick way to find corrleation between away numerical variables. 
# - Most high rating apps have more reviews. High rating apps gets more attention and reviews.
# - Highly rated apps have more installs. High ratings would definitely make any app popular.
# - Apps with poor ratings are more likely to be small in size. Larger size could mean the app is more fleshed out.
# - Apps with low price are more likely to be installed. Cheap and free apps seem to be a safe investment.
# 

# In[ ]:


sns.heatmap(new_df.corr(), annot=True)


# In[ ]:


sns.countplot(x='Type', data=new_df)


# In[ ]:


sns.barplot(x='Type', y='Rating', data=new_df)


# Seems to be very little difference between 'free' and 'paid' app ratings. The paid apps are rated slighly better.

# In[ ]:


new_df.sort_values(by='Price',ascending=False)[new_df['Price'] > 200]


# It's pretty ridiculous to see that people actually bought these useless app just to merely flaunt their wealth. 

# In[ ]:


sns.scatterplot(y='Rating', x='Price', data=new_df, hue='Type')


# We have seen this graph before, but I added a hue to help see the paid apps more clearly. It seems generally that paid apps  do a little better and rarely have very low ratings. Price doesn't seem to indicate good ratings.

# In[ ]:


new_df.head()


# In[ ]:


plt.figure(figsize=(12, 4))
sns.scatterplot(x='Installs', y='Rating', data=new_df)
plt.xlim(0, 1000000000)


# It seems like apps with more installations have higher ratings. This doesn't come as too much of a surprise.

# In[ ]:


sns.barplot(y='Rating', x='Content Rating', data=new_df)


# Apps have the same ratings regardless of content ratings. For Adults only 18+, it seems to be a hit or miss.

# In[ ]:


new_df['Reviews per Install'] = new_df['Reviews'] / new_df['Installs']


# In[ ]:


sns.scatterplot(x='Reviews per Install', y='Rating', data=new_df)


# The only interesting observation here is the outlier with 4 x more reviews than installs! Time to investigate further.

# In[ ]:


new_df[new_df['Reviews per Install'] > 1].sort_values(by='Reviews per Install', ascending=False)


# So I checked to see if you can review without installing the app. As far as I can tell, you can't.
# This lead me to believe that this could be:
# - a glitch or some kind of incorrect input.
# - people reviewing and then uninstalling. I'm not sure if the data took account of that.
# 
# 

# In[ ]:


plt.figure(figsize=(16, 4))
plt.xticks(rotation=90)


sns.boxplot(y='Rating', x='Category', data=new_df)


# Most app categories do well overall. A couple of them have strong outliers such as business, finance, health & fitness, and family.

# In[ ]:



plt.figure(figsize=(16, 4))
plt.xticks(rotation=90)


sns.barplot(y='Installs', x='Category', data=new_df.groupby('Category').sum().reset_index())


# The two most commonly installed apps are communication and game.

# In[ ]:


new_df['Last Updated'] = pd.to_datetime(new_df['Last Updated'], yearfirst=True )


# In[ ]:


plt.figure(figsize=(16, 4))
sns.lineplot(x='Last Updated', y='App', data=new_df.groupby('Last Updated').count().reset_index().sort_values(by='Last Updated', ascending=True))


# Most apps are currently being updated as of this data set.

# In[ ]:


plt.figure(figsize=(16, 4))
sns.lineplot(x='Last Updated', y='Rating', data=new_df.groupby('Last Updated').mean().reset_index().sort_values(by='Last Updated', ascending=True))


# Apps that are last updated recently seem to be more consistenly well rated. The further back you go, the more jumpy the line graph shows. There is somewhat a pattern that shows that apps last updated further back tend to have lower ratings.
# 
# These low ratings in the past could be an indicator that an app didn't do too well so it was last updated years ago. This implies that its development halted out of lack of interest from phone users.

# In[ ]:


new_df.head()


# In[ ]:


new_df['Rounded Rating'] = new_df['Rating'].round()


# In[ ]:


sns.swarmplot(y='Rating', x='Type', hue='Content Rating', data=new_df[new_df['Category'] == 'GAME'])


# Paid games are usually rated better than free games. None of the paid games are rated below 3.0. It seems like games are more likely to be worth your money.

# In[ ]:


new_df['Revenue'] = new_df['Installs'] * new_df['Price']


# In[ ]:



plt.figure(figsize=(16, 4))
plt.xticks(rotation=90)

rev = new_df.groupby('Category').sum().sort_values(by='Revenue', ascending=False).reset_index()
sns.barplot(x='Category', y='Revenue', data=rev)


# Want to make bank? FAMILY category is hot right now in profits.

# In[ ]:


new_df.sort_values(by='Revenue', ascending=False).head(10)


# Is minecraft miscategorized as a FAMILY app? This certainly does inflate the amount of money family apps make, and could rightfully make GAMES the true money maker (which we all know is one of the most profitable businesses out there). Lets change that because I'm sure we can all agree that Minecraft is a game.

# In[ ]:


new_df.drop(new_df[new_df.duplicated('App')].index.get_values(), inplace=True)


# In[ ]:


df_length = new_df.count()


# In[ ]:


new_df.index = np.arange(0, df_length[0])


# After deleting a bunch of rows, I should readjust the index ordering

# In[ ]:


new_df.replace({'Category': {1661: 'GAME'}}, inplace=True)


# In[ ]:


new_df[new_df['App'] == 'Minecraft']['Category'] = 'GAME'


# In[ ]:


new_df.loc[1661, 'Category'] = 'GAME'


# In[ ]:



plt.figure(figsize=(16, 4))
plt.xticks(rotation=90)

rev = new_df.groupby('Category').sum().sort_values(by='Revenue', ascending=False).reset_index()
sns.barplot(x='Category', y='Revenue', data=rev)


# FAMILY got knocked by two places and GAME takes the title of the most profitable category.

# In[ ]:



plt.figure(figsize=(16,4))
plt.xticks(rotation=90)

sns.violinplot(x='Category', y='Rating', data=new_df)


# In[ ]:


new_df.head()


# In[ ]:



plt.figure(figsize=(16,4))
plt.xticks(rotation=90)

sns.barplot(x='Category', y='Reviews', data=new_df.groupby('Category').sum().sort_values(by='Reviews', ascending=False).reset_index())


# Apps that get the most reviews are communication and games. 

# # Thank you for your time!
# 
# More will be included in the futre!

# In[ ]:




