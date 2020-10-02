#!/usr/bin/env python
# coding: utf-8

# <h3>Introduction</h3>

# Hi all, This is a beginners' work. Feel Free to upvote it if you like. My goal is to learn and contriburte to the data science community.

# <h4> Work done in this Kernel </h4>

# - Importing and cleaning of data
# - Analysis of Restaurants based on - 
#     1. Rating
#     2. Online Delivery
#     3. Type
#     4. Table booking facility
#     5. Cuisine Type
#     6. Average Price for two
#     7. Restaurant Chains
#     8. Location and many more
# - Regression taking rating as Depedent variable using Random Forest Regressor

# <h4> Let's get started! </h4>

# <b>Importing required libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
get_ipython().run_line_magic('matplotlib', 'inline')


# <b> Load the data and make a copy of it 

# In[ ]:


zomato_data = pd.read_csv('../input/zomato-bangalore-restaurants/zomato.csv')


# In[ ]:


df = zomato_data.copy()


# In[ ]:


df.head()


# <b> Performing some EDA to better understand the data

# In[ ]:


df.info()


# Based on the above info, we can consider address as unique identifier for distinct restaurants

# In[ ]:


add_group = df.groupby('address')
group_1 = add_group.filter(lambda x: len(x) > 1)[add_group.filter(lambda x: len(x) > 1)['name'] == 'Onesta']


# In[ ]:


group_1['address'][7]


# We can further filter out group_1 on the basis of adderss

# In[ ]:


group_1[add_group.filter(lambda x: len(x) > 1)[add_group.filter(lambda x: len(x) > 1)['name'] == 'Onesta'].address == '2469, 3rd Floor, 24th Cross, Opposite BDA Complex, 2nd Stage, Banashankari, Bangalore']


# From the above data we can see that for the same restaurant we have different url, listed_in(city), also different vote counts
# We can remove the url, listed_in(type) and listed_in(city) and won't lose any information as these are quite generalize form of rest_type
# 
# Let's do some data cleaning before our analysis

# In[ ]:


df = df.drop(['url', 'listed_in(type)', 'listed_in(city)'], axis=1).reset_index().drop(['index'], axis=1).drop_duplicates()


# We can also remove phone number as it will not help us in our analysis

# In[ ]:


df = df.drop(['phone'],axis=1)


# <h3>Now let's start analyzing the data with help of visualizations

# <b> Top restaurant chains in Banglore

# In[ ]:


plt.figure(figsize=(12, 8))
sns.set_style('white')
rest = df.groupby(['address', 'name'])
rest_chains = rest.name.nunique().index.to_frame()['name'].value_counts()[:15] # first 15

ax = sns.barplot(x=rest_chains, y=rest_chains.index, palette='Greys_d')
sns.despine()
plt.title('Top 15 Restaurant chains in Bangalore')
plt.xlabel('Number of Outlets')
plt.ylabel('Name')
for p in ax.patches:
    width = p.get_width()
    ax.text(width+0.007, p.get_y()+p.get_height()/2. + 0.2, format(width), ha="left", color="black")
plt.show()


# CCD has the highest number of outlets in Banglore, followed by Domino's

# <b> Top restaurant types in Bangalore

# In[ ]:


#Preprocessing restaurant types
rest_type_df = df.groupby(['address', 'rest_type']).rest_type.nunique().index.to_frame()
temp_df = pd.DataFrame()
temp_df = rest_type_df.rest_type.str.strip().str.split(',', expand=True)


# In[ ]:


rest_type_temp_df = pd.DataFrame()
rest_type_temp_df = pd.concat([temp_df.iloc[:,0].str.strip(), temp_df.iloc[:, 1].str.strip()]).value_counts()
rest_type_temp_df


# In[ ]:


plt.figure(figsize=(12,8))
sns.set_style('white')
restaurant_types = rest_type_temp_df[:15]
ax = sns.barplot(x=restaurant_types, y=restaurant_types.index, palette="Greys_d")
sns.despine()
plt.title('Top 15 restaurant types in Bangalore')
plt.xlabel('Number of restaurants')
plt.ylabel('Restaurant types')
total_rest = len(rest_type_df)     #restaurants after grouping by address and rest_type
for p in ax.patches:
    percentage='{:.1f}%'.format(100*p.get_width()/total_rest)    # find % of rest types
    x = p.get_x() + p.get_width() + 0.02
    y = p.get_y() + p.get_height()/2
    ax.annotate(percentage, (x,y), ha="left", color='black')
plt.show()


# 1. Quick Bites, Casual Dining and Delivery are common restaurant types in bangalore
# 2. Bhojanalya being the rarest rest_type

# <b> Top Cuisines served in Bangalore

# In[ ]:


#Preprocessing of Cuisines
cuisines_df = df.groupby(['address', 'cuisines']).cuisines.nunique().index.to_frame()
cuisines_temp_df = pd.DataFrame()
cuisines_temp_df = cuisines_df.cuisines.str.strip().str.split(',', expand=True)


# In[ ]:


cuisines = pd.DataFrame()
cuisines = pd.concat([cuisines_temp_df.iloc[:,0].str.strip(), cuisines_temp_df.iloc[:,1].str.strip(), cuisines_temp_df.iloc[:,2].str.strip(),
                     cuisines_temp_df.iloc[:,3].str.strip(),cuisines_temp_df.iloc[:,4].str.strip(),cuisines_temp_df.iloc[:,5].str.strip(),
                     cuisines_temp_df.iloc[:,6].str.strip(),cuisines_temp_df.iloc[:,7].str.strip()]).value_counts()


# In[ ]:


plt.figure(figsize=(12,8))
sns.set_style('white')
cuisine = cuisines[:15]
ax = sns.barplot(x=cuisine, y=cuisine.index, palette="Greys_d")
sns.despine()
plt.title('Top 15 cuisines served in Bangalore')
plt.xlabel('Number of restaurants')
plt.ylabel('Name of cuisines')
total_rest = len(cuisines_df)     #restaurants after grouping by address and rest_type
for p in ax.patches:
    percentage='{:.1f}%'.format(100*p.get_width()/total_rest)    # find % of rest types
    x = p.get_x() + p.get_width() + 0.02
    y = p.get_y() + p.get_height()/2
    ax.annotate(percentage, (x,y), ha="left", color='black')
plt.show()


# In[ ]:


cuisines[cuisines.index == "Healthy Food"]


# 1. Around 40% restaurants serves North Indian cuisines, followed by Chinese and South Indian
# 2. There are around 200+ restaurants in Bangalore serving healthy food

# <b> Top locations for foodies in Bangalore

# In[ ]:


location_df = df.groupby(['address', 'location']).location.nunique().index.to_frame()


# In[ ]:


print(location_df['location'].value_counts()[location_df['location'].value_counts().index.str.contains('Koramangala')])


# In[ ]:


print("Number of restaurants in Koramangala:",
    sum(location_df['location'].value_counts()[location_df['location'].value_counts().index.str.contains('Koramangala')]))


# In[ ]:


plt.figure(figsize=(12, 8))
sns.set_style('white')
locations= location_df['location'].value_counts()[:15]
ax = sns.barplot(x= locations, y = locations.index, palette='Greys_d')
sns.despine()
plt.title('Top 15 locations for foodies in Bangalore')
plt.xlabel('Number of restaurants')
plt.ylabel('Name of Location')
for p in ax.patches:
    width = p.get_width()
    ax.text(width+0.007, p.get_y() + p.get_height() / 2. + 0.2, format(width), 
            ha="left", color='black')
plt.show()


# 1. Whitefield, BTM, Electronic City, Marathahali and HSR has the most number of restaurants
# 2. Koramangala (combining all blocks) has 868 restaurants

# <b> Top dishes served in Bangalore

# In[ ]:


#Preprocessing dish liked
dish_liked_df = df.groupby(['address', 'dish_liked']).dish_liked.nunique().index.to_frame()
dish_liked_temp_df = pd.DataFrame()
dish_liked_temp_df = dish_liked_df.dish_liked.str.strip().str.split(',', expand=True)


# In[ ]:


dish_liked = pd.DataFrame()
dish_liked = pd.concat([dish_liked_temp_df.iloc[:,0].str.strip(), dish_liked_temp_df.iloc[:,1].str.strip()]).value_counts()
dish_liked


# In[ ]:


plt.figure(figsize=(12, 8))
sns.set_style('white')
dishes = dish_liked[:15]
ax = sns.barplot(x= dishes, y = dishes.index, palette='Greys_d')
sns.despine()
plt.title('Top 15 commonly served dishes in Bangalore')
plt.xlabel('Number of restaurants')
plt.ylabel('Name of dishes')
total = len(dish_liked_df)

for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_width()/total)
        x = p.get_x() + p.get_width() + 0.02
        y = p.get_y() + p.get_height()/2
        ax.annotate(percentage, (x, y),
        ha="left", color='black')
plt.show()


# 1. As there are more quick bites restaurants in Bangalore, we can see Burgers and Pasta being served more.
# 2. Biryani is the most famous Indian dish being served

# <b> Some data cleaning for rating column

# In[ ]:


#Replacing restaurants with their ratings given as New to NAN and dropping them finally
rating_df = df.copy()
rating_df['rate'] = rating_df['rate'].replace('NEW',np.NaN)
rating_df['rate'] = rating_df['rate'].replace('-',np.NaN)
rating_df.dropna(how = 'any', inplace = True)


# In[ ]:


rating_df['rate'] = rating_df.loc[:,'rate'].replace('[ ]','',regex = True)
rating_df['rate'] = rating_df['rate'].astype(str)
rating_df['rate'] = rating_df['rate'].apply(lambda r: r.replace('/5',''))
rating_df['rate'] = rating_df['rate'].apply(lambda r: float(r))


# <h3> Let's see some relation between different variables

# <b> Table Booking vs Rating

# In[ ]:


plt.figure(figsize=(20, 8))
sns.set_style('white')
ax = sns.countplot(x='rate', hue='book_table', data=rating_df, palette='Greys_d')
sns.despine()
plt.title('Rating of Restaurants vs Table Booking')
plt.xlabel('Rating')
plt.ylabel('Number of restaurants')
plt.show()


# <b> Online Delivery vs Rating

# In[ ]:


plt.figure(figsize=(20, 8))
sns.set_style('white')
ax = sns.countplot(x='rate', hue='online_order', data=rating_df, palette='Greys_d')
sns.despine()
plt.title('Rating of Restaurants vs Online Delivery')
plt.xlabel('Rating')
plt.ylabel('Number of restaurants')
plt.show()


# In[ ]:


rating_df.loc[rating_df['rate'] >= 4, 'rating_category'] = 'Above 4'
rating_df.loc[(rating_df['rate'] >= 3) & (rating_df['rate'] < 4), 'rating_category'] = 'Above 3'
rating_df.loc[(rating_df['rate'] >= 2) & (rating_df['rate'] < 3), 'rating_category'] = 'Above 2'
rating_df.loc[rating_df['rate'] < 2, 'rating_category'] = 'Above 1'


# <b>Let's see the trend restaurant type wise

# In[ ]:


def dish_served_per_rest_type(rest_type):
    dishLiked_restType_df = df.groupby(['address', 'dish_liked', 'rest_type']).dish_liked.nunique().index.to_frame()
    dishLiked_restType_temp_df = pd.DataFrame()
    dishLiked_restType_temp_df = dishLiked_restType_df[dishLiked_restType_df['rest_type'] == rest_type].dish_liked.str.strip().str.split(',', expand=True)
    dish_liked_temp_df = pd.DataFrame()
    dish_liked_temp_df = pd.concat([dishLiked_restType_temp_df.iloc[:,0].str.strip(), dishLiked_restType_temp_df.iloc[:,1].str.strip()]).value_counts()
    temp_df = pd.DataFrame({'dishes':dish_liked_temp_df[:10], 'group':dish_liked_temp_df[:10].index})
    norm = matplotlib.colors.Normalize(vmin=min(dish_liked_temp_df[:10]), vmax=max(dish_liked_temp_df[:10]))
    colors = [matplotlib.cm.Blues(norm(value)) for value in dish_liked_temp_df[:10]]
    squarify.plot(sizes=temp_df['dishes'], label=temp_df['group'], alpha=0.8, color=colors)
    plt.title("Top 10 dishes served in "+rest_type, fontsize=15, fontweight='bold')
    plt.axis('off')
    plt.show()


# In[ ]:


dish_served_per_rest_type('Quick Bites')


# In[ ]:


dish_served_per_rest_type('Casual Dining')


# In[ ]:


dish_served_per_rest_type('Cafe')


# In[ ]:


dish_served_per_rest_type('Delivery')


# In[ ]:


dish_served_per_rest_type('Dessert Parlor')


# In[ ]:


dish_served_per_rest_type('Takeaway')


# In[ ]:


dish_served_per_rest_type('Bar')


# Burgers are quite famous in most of the Restaurant types

# <b> Distribution of cost for two people

# In[ ]:


costfortwo_df = df.groupby(['address', 'approx_cost(for two people)'])
cost_df = costfortwo_df['approx_cost(for two people)'].nunique().index.to_frame()


# In[ ]:


cost_df['approx_cost(for two people)'] = cost_df['approx_cost(for two people)'].str.replace(',', '').astype(float)


# In[ ]:


plt.figure(figsize=(8,8))
cost_dist = cost_df['approx_cost(for two people)'].dropna()
sns.distplot(cost_dist, bins=20, kde_kws={'color':'k', 'lw':3, 'label':'KDE'})
plt.show()


# There are very few restaurants with cost more than 1000. Overall distribution of cost in right skewed

# <b> Distribution of Rating

# In[ ]:


rating_df_temp = rating_df.groupby(['address', 'rate'])
plt.figure(figsize=(8,8))
rating = rating_df_temp.rate.nunique().index.to_frame()['rate']
sns.distplot(rating,bins=20,kde_kws={"color": "k", "lw": 3, "label": "KDE"})
plt.show()


# In[ ]:


rating.describe()


# Majority of restaurants are rated more than 3.5, the distribution tends to be negative/left skew

# <b> Relation of Rating and Average cost (for two) for both Table booking and Online order

# In[ ]:


cost_rating_temp_df = df.groupby(['address', 'rate', 'approx_cost(for two people)', 'book_table', 'online_order'])
cost_rating_df = cost_rating_temp_df[['rate', 'approx_cost(for two people)', 'book_table', 'online_order']].nunique().index.to_frame()


# In[ ]:


cost_rating_df['approx_cost(for two people)'] = cost_rating_df['approx_cost(for two people)'].str.replace(',','').astype(float)
cost_rating_df['rate'] = cost_rating_df['rate'].apply(lambda x: float(x.split('/')[0]) if len(x) > 3 else np.nan).dropna()


# In[ ]:


fig, axis = plt.subplots(nrows=1, ncols=2, figsize=(16,6))
sns.scatterplot(x='rate', y='approx_cost(for two people)', hue='book_table', data=cost_rating_df, ax=axis[0])
sns.scatterplot(x='rate', y='approx_cost(for two people)', hue='online_order', data=cost_rating_df, ax=axis[1])
plt.show()


# 1. There is no trend between cost and rating 
# 2. Generally we see restaurants with table booking facitlity tends to get higher rating
# 3. There is not much relationship between cost of restaurants accepting online orders vs. not accepting orders, but mostly costly restaurants doesnot accept online orders 

# <h3> Basic Regression Analysis using Random Forest Regressor

# <b> Some transformations before modelling

# In[ ]:


regression_df = zomato_data.copy()


# In[ ]:


regression_df=regression_df.drop(['url','dish_liked','phone'],axis=1)
regression_df.dropna(how='any', inplace=True)


# In[ ]:


#Preprocessing rate
regression_df = regression_df.loc[regression_df['rate'] !='NEW']
regression_df = regression_df.loc[regression_df['rate'] !='-'].reset_index(drop=True)
remove_slash = lambda x: x.replace('/5', '') if type(x) == np.str else x
regression_df['rate'] = regression_df['rate'].apply(remove_slash).str.strip().astype('float')


# In[ ]:


#Preprocessing approx_cost(for two people)
regression_df['approx_cost(for two people)'] = regression_df['approx_cost(for two people)'].dropna()
regression_df['approx_cost(for two people)'] = regression_df['approx_cost(for two people)'].astype(str)
regression_df['approx_cost(for two people)'] = regression_df['approx_cost(for two people)'].apply(lambda x: x.replace(',','.'))
regression_df['approx_cost(for two people)'] = regression_df['approx_cost(for two people)'].astype(float)


# In[ ]:


regression_df['name'] = regression_df['name'].apply(lambda x: x.title())
regression_df['online_order'].replace(('Yes', 'No'), (True, False), inplace=True)
regression_df['book_table'].replace(('Yes', 'No'), (True, False), inplace=True)


# <b> Encoding

# In[ ]:


def encoding(regression_df):
    for column in regression_df.columns[~regression_df.columns.isin(['rate', 'approx_cost(for two people)', 'votes'])]:
        regression_df[column] = regression_df[column].factorize()[0]
    return regression_df

regression_df_en = encoding(regression_df)


# <b> Correlation Heatmap

# In[ ]:


plt.figure(figsize=(15,10))
sns.heatmap(regression_df_en.corr(), annot=True, cmap='Greys')
plt.title('Correlation Heatmap', size=20)
plt.show()


# <b> Modelling

# In[ ]:


#Independent variables and dependent variable
x = regression_df_en.iloc[:, [2,3,5,6,7,8,9,11]]
y = regression_df_en['rate']


# In[ ]:


#Splitting dataset into training and testing dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)


# In[ ]:


#Random Forest Regression
model = RandomForestRegressor(n_estimators=500)
model.fit(x_train, y_train)


# In[ ]:


print('Accuracy: {}'.format(model.score(x_test, y_test)))


# - RandomForest regressor shows a 92% accuracy score. We can test this using other models as well and compare different algorithms

# This kernel was made for learning purpose and was referenced from some other kernels as well. 

# In[ ]:




