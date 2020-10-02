#!/usr/bin/env python
# coding: utf-8

# # Seattle Airbnb Open Data
# 
# Before the whole COVID-19 start, I was looking into booking a few places to visit around Washington state incluing Seattle and Whibdey island. As I was looking for place I could see that prices vary significantly from city to city and from neighborhood to another.
# 
# The current [data](https://www.kaggle.com/airbnb/seattle) at hand, only contains information about the city of Seattle. Let's see what information we can find about the data and how we can predict the price of a rental based on its description and any data available in this set.
# 
# * Host experience: As I was talking to a friend of mine we were thinking that perhaps more experienced hosts charge more for their listings. Is that true? It seems intuitive that as you get more customer and expand, you can also charge more. What does the data tell us?
# 
# * Is there a correlation with the date of a reservation and the price?
# 
# * Is there any useful information in the neighborhood and municipality about the unit price?
# 
# * What are the most determining factors in the price of a unit?
# 

# Let's import the necessary libraries

# In[ ]:


from glob import glob
import pandas as pd
import numpy as np
import seaborn
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor


# Load data

# In[ ]:


path = '../input/seattle/'
df = pd.read_csv(path + 'listings.csv')
calendar = pd.read_csv(path + 'calendar.csv')
reviews = pd.read_csv(path + 'reviews.csv')


# ## Clean the data, part I
# 
# Here are the steps for cleaning the data.
# 
# 1. Drop nan values
# 2. Extract feature from dates, such as day of the week and week of the year
# 3. Turn date strings into numbers for year, month, and day
# 4. Turn price (string with dollar sign) to number

# In[ ]:


def convert_time(dataframe, name):
    """
    This function takes a dataframe as an input, plus the name of the date column. Then splits the date by '-', 
    and converts to year, month, and day. It also extracts day of the week and week of the year as features.
    
    Args:
    dataframe (pandas.DataFrame): the input dataframe
    name (str): the name of the column to be treated as date string
    
    Returns:
    pd.DataFrame: with new columns for year, month, day, dayofweek, and weekofyear
    """
    to_numeric = lambda x: float(x.lstrip('$').replace(',', ''))
    suffixes = ['y', 'm', 'd']
    date = pd.to_datetime(dataframe[name])
    dataframe['dayofweek'] = date.dt.dayofweek
    dataframe['weekofyear'] = date.dt.week
    split_df = dataframe[name].str.split('-', expand=True).rename(columns={idx: '%s_%s' % (name, suffix) for idx, suffix in enumerate(suffixes)}).applymap(float)
    dataframe = pd.concat((dataframe.drop([name], axis=1), split_df), axis=1)
    dataframe['price'] = dataframe.price.apply(to_numeric)
    return dataframe
    
calendar.dropna(inplace=True)
df = df.applymap(lambda x: str.lower(x) if isinstance(x, str) else x)

calendar = convert_time(calendar, 'date')
df = convert_time(df, 'host_since')


# ## Neighborhood and Municipality
# Before we go any further, let's answer the first question, is there any information about the city and municipality?
# 
# Looking at the data below you can see that there are three columns with the name seattle. One was capitalized, and in the cell above as I lower-cased each string, they become the same. The third one is in Chinese characters. Despite being the same city, the mean values for price are wildly different. Perhaps the Seattle value in Chinese character is a special type of unit for visiting students? I have no idea. I don't think much information can be drawn from the city field.

# In[ ]:


city_data = df.groupby('city')['price']
city_data.mean()


# Let's look at neighborhood instead. Let's pull all columns that has neighborhood in them.

# In[ ]:


df[[x for x in df.columns if 'neighb' in x]]


# It appears that the best column to look at is "neighbourhood_group_cleansed"

# In[ ]:


ax = seaborn.boxplot(data=df, x='neighbourhood_group_cleansed', y='price', fliersize=0.1)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_ylim(0, 400);


# In[ ]:


relevant_data = df.groupby('neighbourhood_group_cleansed')[['price', 'longitude', 'latitude']].mean()
plt.figure(figsize=(10, 10))
seaborn.scatterplot(data=df, x='longitude', y='latitude', size='price', marker='o', color='skyblue')
seaborn.scatterplot(data=relevant_data, x='longitude', y='latitude', size='price', marker='o', color='red')


# ## Host experience
# 
# How does the host experience correlate with listing price?
# 
# By inspecting the heatmap below, we can see that there is a small correlation with the year the host started renting Airbnb unit and the unit price. This correlation is slightly negative, which means that if the host joined sooner (smaller year) the price is slightly higher.

# In[ ]:


data = df[[x for x in df.columns if 'host' in x] + ['price']].drop(['host_id'], axis=1)
seaborn.heatmap(data.corr(), fmt='.2f', annot=True)


# Below in the bar plot, we can see that hosts who joined in 2008-2009 have relatively cheaper units, which explains why the overall correlation is small. Plus, the standard error on these averages is high, which also diminishes correlations.

# In[ ]:


df.groupby('host_since_y')['price'].mean().plot.bar(title='Price by year since the host joined', yerr=df.groupby('host_since_y')['price'].std())


# ## Date of reservation
# 
# By looking at the correlation of we can see, not surprisingly, that the reservation price has a slight correlation the day of week and month. It has little dependence on the year of the reservation. Given the symmetric nature of the demand that peaks in the middle of summer, there is little difference between month and week of the year. In other words, month is the stronger indicator, and week of the year just follows month. If there was strong week-to-week variation within a month, then week would have been an important feature.

# In[ ]:


seaborn.heatmap(calendar.drop('listing_id', axis=1).corr(), fmt='.2f', annot=True)


# In[ ]:


fig, axs = plt.subplots(1, 3, figsize=(16, 5))
seaborn.boxplot(data=calendar, x='date_m', y='price', ax=axs[0], fliersize=0)
seaborn.boxplot(data=calendar, x='dayofweek', y='price', ax=axs[1], fliersize=0)
seaborn.boxplot(data=calendar, x='date_y', y='price', ax=axs[2], fliersize=0)
for ax in axs:
    ax.set_ylim(0, 350)


# ## Clean the data, part II
# 
# Let's put the columns in different categories. 
# 
# * Remove columns with 'id' and 'url' in their names.
# * Keep the numeric columns.
# * If columns are objects, see how many unique values exist. If the number of unique elements is limited, we can treat it as a categorical variable.
# * If only one values exist, there is no information, drop it.
# * Create dummy one-hot variables from categorical ones.
# * If there are more than 20% null values, drop the column
# * If there are less than 20%, fill with mean value

# In[ ]:


def group_columns(df):
    """
    This function takes a dataframe and groups the columns by their characteristics.
    This is relatively a general function and reusable on other datasets
    
    Args:
    df (pandas.DataFrame): The input data frame
    
    Returns:
    dict: A dictionary of column types and list of columns
    """
    columns = df.columns
    columns_list = {'categorical': [], 'binary': [], 'text': [], 'number': [], 'drop': []}
    for col in columns:
        if col == 'id':
            continue
        elif 'id' in col or 'url' in col:
            columns_list['drop'].append(col)
        else:
            if df[col].dtype == np.dtype('O'):
                n = len(np.unique(df[col][df[col].notnull()]))
                if n == 1:
                    columns_list['drop'].append(col)
                elif n < 20:
                    columns_list['categorical'].append(col)
                else:
                    columns_list['text'].append(col)
            else:
                columns_list['number'].append(col)
    return columns_list

def clean_data(df):
    """
    This function cleans the input data frame by removing unwanted columns, turning categorical variables to 
    one-hot numbers, and dropping columns or filling NAN values based on the frequency of NAN in the columns.
    """
    df.drop(['city'], axis=1, inplace=True)
    columns_list = group_columns(df)

    df = df.drop(columns_list['drop'], axis=1)
    for col in columns_list['categorical']:
        dummies = pd.get_dummies(df[col], prefix=col, prefix_sep='_', drop_first=True, dummy_na=True)
        df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
        columns_list['number'].extend(dummies.columns.tolist())
        assert(col not in df.columns)
    for col in columns_list['text']:
        df[col] = df[col].fillna('na')

    df_new = df[columns_list['number'] + ['id']]
    
    df_nan = df.isnull().mean()
    for col in df_new.columns:
        if df_nan[col] > 0.2:
            df_new.drop(col, axis=1, inplace=True)
        elif col in df_nan.index:
            df_new.fillna({col: df_new[col].mean()}, inplace=True)
    return df_new

df_new = clean_data(df)


# * Merge the data two datasets
# 

# In[ ]:


df = pd.merge(calendar.drop('available', axis=1), df_new, left_on='listing_id', right_on='id')


# ## Machine learning
# 
# Now that we have merged the data. Let's look at the variablity of the some of the remaining columns, and drop them if there is no information in them.
# 
# Then, we will, as usual, split the data into train and test to be able to evaluate our model. I'll simply pick random forest as a starting point with default values.

# In[ ]:


X = df.drop(['price_x', 'price_y', 'listing_id', 'id'], axis=1)
drop_more = [x for x in X.columns if X[x].std() <= 1e-6]
X.drop(drop_more, axis=1, inplace=True)
y = df['price_x']
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


model = RandomForestRegressor()
model.fit(xtrain, ytrain)

ypred = model.predict(xtrain)
ypredtest = model.predict(xtest)

r2 = r2_score(ytrain, ypred)
r2test = r2_score(ytest, ypredtest)

print("Training r2-score: %.4f,  Test r2-score: %.4f" % (r2, r2test))


# Since the result is pretty good on our first try, I'll avoid further study of different models for now.

# ## The most determining factors of reservation price
# 
# In order to figure out the answer to this question, one way is to look at the correlation of the top 10 important features with the unit price. In order to do that, we are going to sort the feature importance of our trained model and pick the data column that corresponds to the feature.

# In[ ]:


inds = np.argsort(-model.feature_importances_)
importance_ds = pd.Series(model.feature_importances_[inds][:20], index=X.columns[inds][:20])
cols = X.columns[inds][:10]
new_data = pd.concat((X[cols], y), axis=1)
fig, axs = plt.subplots(1, 2, figsize=(20, 10))
seaborn.heatmap(new_data.corr(), fmt='.2f', annot=True, ax=axs[0])
importance_ds.plot.bar(title='Top 20 important features', ax=axs[1])


# ## Does this make sense?
# 
# Well the short answer is yes and no!
# 
# Looking at quantities like number of bedrooms and bathrooms, it makes sense that units with more are more expensive. It's also logical to think that units that allow accomodating more are offered at a higher rate.
# 
# How about the feature 'reviews_per_month', why does that have negative correlation with price and relatively a strong effect on the regression? Well, the first thing comes to mind is that maybe cheaper units have more review per month, but why? Because they are visited more frequently? or people who visit cheaper places tend to leave reviews more often? I don't think that is clear from this data. Perhaps if the frequency of visit to each unit was available, we could normalize the number of review by the number of visits and find out if that's the case.
# 
# A few variables, e.g. `room_type`, have few categories and are turned to binary values 0, 1 indicating other options or this option respectively. For example, room_type_private_room=1, means the unit is a private room. If it's equal to zero, it means other options are chosen (share & nan).
# 
# A few varaibles with `_x` and `_y` exist which comes from merging the two data sets `listings` and `calendar`. Price is positively and negatively correlated to apparently the same varialbes. The classifier thinks that the two are good indicators for the unit price. Is this overfitting? In other words, is there a non repeatable pattern that deep tree classifiers have found?
# 
# <style type="text/css">
# .tg  {border-collapse:collapse;border-spacing:0;}
# .tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
#   overflow:hidden;padding:10px 5px;word-break:normal;}
# .tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
#   font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
# .tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
# </style>
# <table class="tg">
#   <tr>
#     <th class="tg-0pky">Feature</th>
#     <th class="tg-0pky">Correlation</th>
#     <th class="tg-0pky">Meaning</th>
#   </tr>
#   <tr>
#     <td class="tg-0pky">Bedrooms</td>
#     <td class="tg-0pky">+0.63</td>
#     <td class="tg-0pky">Large, expensive unit</td>
#   </tr>
#   <tr>
#     <td class="tg-0pky">Review per month</td>
#     <td class="tg-0pky">-0.18</td>
#     <td class="tg-0pky">Negative correlation due to infrequency of expensive units</td>
#   </tr>
#   <tr>
#     <td class="tg-0pky">Longitude & Latitude</td>
#     <td class="tg-0pky">-0.11 -> -0.03</td>
#     <td class="tg-0pky">Price depends on location</td>
#   </tr>
#   <tr>
#     <td class="tg-0pky">room_type (private_room)</td>
#     <td class="tg-0pky">-0.39</td>
#     <td class="tg-0pky">1 indicate private room -> cheaper compared with nan </td>
#   </tr>
#   <tr>
#     <td class="tg-0pky">room_type (shared_room)</td>
#     <td class="tg-0pky">-0.17</td>
#     <td class="tg-0pky">1 indicate private room -> cheaper compared with nan </td>
#   </tr>
#   <tr>
#     <td class="tg-0pky">weekofyear_x & _y</td>
#     <td class="tg-0pky">+ and -</td>
#     <td class="tg-0pky">Due to merge, there are two of them. Overfitting?</td>
#   </tr>
#   
# </table>

# # Conclusion
# 
# We looked at this data rather carefully. So far I have ignore all the usefull information in the text, but that's for another day. From the quantitative date, it looks like the price of a unit is related to very intuitive features such as number of bedrooms and bathrooms, location, number of reviews, room type and date of the reservation.
# 
# * Host experience: Except for the years 2008-2009 (hosts who joined during recession), hosts who have joined Airbnb earlier charge higher on average.
# 
# * Is there a correlation with the date of a reservation and the price? Well, yes! Units tend to be more expensive on Friday-Satureday and in Summer.
# 
# * Is there any useful information in the neighborhood and municipality about the unit price? There are expensive and cheap neighborhoods. Magnolia and west Seattle are very expensive. If I want to visit there, and I'm certainly going to rent a car, I'll rent somewhere else!
# 
# * What are the most determining factors in the price of a unit? The table above summarize that information. Size of the unit and the location are the biggest indicators. It may not be so helpful to a traveller as an end user, because usually size is as a constraint. If you have to get a big unit for group of friends or children, there is only so much room for flexibility.
# 
# Depending on the audience and who the end user of this analysis is, we may or may not have a solid conclusion.
# 
# 
# 
