#!/usr/bin/env python
# coding: utf-8

# # Table of Contents
# 1. Data Cleaning
# 2. EDA
# 3. Preprocessing
# 4. Modeling
# 5. Summary
# Let's begin by importing what we need for now and read the file

# # Data Cleaning

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        df = pd.read_csv(os.path.join(dirname, filename))
        
df.head()


# Nice. Now let's check for any good ol'  ![image](https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/supernannyjofrostreturns-1573063913.jpg?crop=1.00xw:0.751xh;0,0.0523xh&resize=768:*)

# In[ ]:


df.isnull().sum()


# In[ ]:


df['Year'].fillna(df['Year'].mode()[0], inplace=True)

df['Publisher'].replace(np.nan, df['Publisher'].mode()[0], inplace=True)

df.isnull().sum()


# Great. Now we can move on!  
# While looking through the data, I noticed this:

# In[ ]:


np.unique(df['Platform'])


# 2600? A.K.A Atari

# In[ ]:


df['Platform'].replace('2600', 'Atari', inplace=True)


# In[ ]:


np.unique(df['Platform'])


# While graphing, I noticed there was a lack of Years in 2017+. So I removed anything from 2017 and up.

# In[ ]:


df = df[df.Year < 2017]
df.head()


# Ok. That is all the data cleaning I'll be doing.

# # EDA
# Work in progress.  
#   
# Let's import what we need

# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
import pandas as pd


# ## Global Sales Over the Years

# In[ ]:


g_sales_over_years = df.groupby(['Year'])['Global_Sales'].sum()

box_plot_df = pd.DataFrame(columns=[str(int(i)) for i in g_sales_over_years.index])

for i in g_sales_over_years.index:
    box_plot_df.at[0, str(int(i))] = g_sales_over_years[i]

plt.figure(figsize=(13.7, 9))

sns.barplot(x='variable', y='value', data=pd.melt(box_plot_df), palette='plasma', ec='Black')

plt.ylabel('Global Sales (in Millions)', fontsize=22)
plt.xlabel('Year', fontsize=22)
plt.title('Global Sales (in Millions) throughout the Years', fontsize=24, fontweight='bold')
plt.xticks(rotation=45, fontsize=16)
plt.yticks(fontsize=16)
plt.show()


# Personally, I found it very interesting that the peak was in 2007-2009 considering that nowadays you can download copies of the game instead of having to go to GameStop or something of the sort. (Note that the peak is a bit taller because we filled the nas with the mode)

# ## Platform Releases

# In[ ]:


style.use('seaborn-poster')

f, ax = plt.subplots()
platform_releases = df['Platform'].value_counts()

sns.barplot(x=platform_releases.values, y=platform_releases.index, ec='Black')
ax.set_title('Platforms with the Most Releases', fontweight='bold', fontsize=23)
ax.set_xlabel('Releases', fontsize=18)
ax.set_xlim(0, max(platform_releases.values)+130)
ax.set_ylabel('Platform', fontsize=18)

for p in ax.patches:
   width = p.get_width()
   ax.text(width + 62,
           p.get_y() + p.get_height() / 2. + 0.28,
           int(width),
           ha="center", fontsize=14)

plt.show()


# I found it interesting that PS2 had the most releases over all of the previous consoles -- and that the Nintendo DS had the most. 

# ## Most Popular Genres

# In[ ]:


style.use('seaborn-poster')
genre_global_sales = df.groupby(['Genre'])['Global_Sales'].sum().sort_values(ascending=False)
print(genre_global_sales)
sns.barplot(x=genre_global_sales.index, y=genre_global_sales.values, ec='Black', palette='twilight')
plt.xticks(rotation=20, fontsize=12)
plt.xlabel('Genre', fontsize=18)
plt.ylabel('Global Sales (in Millions)', fontsize=18)
plt.title('Global Sales of Genres from 1980-2016', fontweight='bold', fontsize=22)
plt.tight_layout()
plt.show()


# I feel like Action is a meta genre, as all genres here can be action games.

# # Top 5 Genres and their Sales Over the Years Respective of their Countries  
# Title is a mouthfull, but basically just showing the sales of the top 5 genres (classified by most sales) separated by the individual country.

# In[ ]:


top5_genres_list = df.groupby(['Genre'])['Global_Sales'].sum().sort_values(ascending=False).head(5).index

top5_genre_df = df[df.Genre.isin(top5_genres_list)]
fig, (ax0,ax1) = plt.subplots(2,2, figsize=(17,10))

fig.suptitle('Top 5 Genres and their Sales (in Millions) Respective to their Country', fontsize=20, fontweight = 'bold')

sns.lineplot(x='Year', y='NA_Sales', hue='Genre', data=top5_genre_df, ci=None, ax=ax0[0], palette='Set1')

sns.lineplot(x='Year', y='EU_Sales', hue='Genre', data=top5_genre_df, ci=None, ax=ax0[1], palette='Set1')

sns.lineplot(x='Year', y='JP_Sales', hue='Genre', data=top5_genre_df, ci=None, ax=ax1[0], palette='Set1')

sns.lineplot(x='Year', y='Other_Sales', hue='Genre', data=top5_genre_df, ci=None, ax=ax1[1], palette='Set1')

ax0[0].legend(loc='upper right')
ax0[1].legend(loc='upper right')
ax1[0].legend(loc='upper right')
ax1[1].legend(loc='upper right')

ax1[1].set_ylim(-0.1,1.6)

ax0[0].set_ylabel('NA Sales (in Millions)', fontsize=16)
ax0[1].set_ylabel('EU Sales (in Millions)', fontsize=16)
ax1[0].set_ylabel('Japan Sales (in Millions)', fontsize=16)
ax1[1].set_ylabel('Other Sales (in Millions)', fontsize=16)

ax0[0].set_xlabel('Year', fontsize=16)
ax0[1].set_xlabel('Year', fontsize=16)
ax1[0].set_xlabel('Year', fontsize=16)
ax1[1].set_xlabel('Year', fontsize=16)


plt.show()


# Due to missing values, I believe this is why Role-Playing has really weird behavior in the Other Sales graph.

# # Publishers' Global Sales 
# I was curious to see how the individual publishers did over time.  
# I took the 10 publishers with the most global sales and spit them into two graphs.

# In[ ]:


style.use('seaborn-poster')

top10_publishers_list = df.groupby(['Publisher'])['Global_Sales']    .sum()    .sort_values(ascending=False).head(10).index

zero_to_five_publishers_list = top10_publishers_list[0:5]
five_to_ten_publishers_list = top10_publishers_list[5:]
zero_to_five_publishers_df = df[df.Publisher.isin(zero_to_five_publishers_list)]
five_to_ten_publishers_df = df[df.Publisher.isin(five_to_ten_publishers_list)]


fig, (ax0, ax1) = plt.subplots(2,1)
plt.subplots_adjust(hspace=0.33, top=.95)

# 1 - 5 in Global Sales
sns.lineplot(x='Year', y='Global_Sales',
             data=zero_to_five_publishers_df, hue='Publisher',
             ci=None, ax=ax0, palette='Set1')

ax0.legend(prop={'size':11.5})

# 5-10 in Global Sales
sns.lineplot(x='Year', y='Global_Sales',
             data=five_to_ten_publishers_df, hue='Publisher',
             ci=None, ax=ax1, palette='Set1')

ax0.set_title('Top 1-5 Publishers by Global Sales')
ax0.set_ylabel('Global Sales (in Millions)')

ax0.spines['right'].set_visible(False)
ax0.spines['top'].set_visible(False)

ax1.set_title('Top 5-10 Publishers by Global Sales')
ax1.set_ylabel('Global Sales (in Millions)')
ax1.legend(loc='upper center', prop={'size': 11.5})
ax1.set_ylim(-0.5, 5)

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

plt.show()


# Pretty interesting timeline for the Sega genesis. Similar behavior to that of Activisions. 

# # Predictive Modeling
# 
# So I plan to have different types of predicting going on, but as of now this is what I have:  
# Linear Regression: Predicts the global sales of a product based on Publisher, Genre, and Platform
# 
# ## Linear Regression
# 

# ### Preprocessing
# I store in memory the labels that I am going to later use to build my dataframe solely from the features and label that I need.  
# My plan is to label encode all the categorical labels  
# Let's import what we need:  

# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


categorical_labels = ['Platform', 'Genre', 'Publisher']
numerical_lables = ['Global_Sales']
enc = LabelEncoder()
encoded_df = pd.DataFrame(columns=['Platform', 'Genre', 'Publisher', 'Global_Sales'])

for label in categorical_labels:
    temp_column = df[label]

    encoded_temp_col = enc.fit_transform(temp_column)

    encoded_df[label] = encoded_temp_col

for label in numerical_lables:
    encoded_df[label] = df[label].values

encoded_df.head()


# ### Without Cross Validation

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import train_test_split


# In[ ]:


x = encoded_df.iloc[:, 0:3]
y = encoded_df.iloc[:,3:]

scalar = StandardScaler()

x = scalar.fit_transform(x)

linear_reg = LinearRegression()

linear_reg.fit(x, y)

y_pred = linear_reg.predict(x)

r2 = r2_score(y, y_pred)

print('\nLinear Regression Results without Cross Validation:')

print(f'MAE in $ (Millions): {mean_absolute_error(y_pred, y)}')

print(f'MSE in $ (Millions): {mean_squared_error(y_pred, y)}')

print(f'R2 Coeff: {r2}')


# Not bad,but not good. The MAE here incidicts how off I am from the correct answer by 1 being equal to 1 million in sales.

# ### With Cross Validation

# In[ ]:


linear_reg = LinearRegression()

y_pred = cross_val_predict(linear_reg, x, y, cv=5)

r2 = r2_score(y, y_pred)

print(f'MAE in Sales (Millions): {mean_absolute_error(y_pred, y)}')

print(f'MSE in Sales (Millions): {mean_squared_error(y_pred, y)}')

print(f'R2 Coeff: {r2}')


# ### Ridge Regression with Cross Validation

# In[ ]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


# In[ ]:


x = scalar.fit_transform(x)

ridge = Ridge()

grid = GridSearchCV(ridge, param_grid={'alpha':range(0,10)}, refit=True)

y_pred = cross_val_predict(grid, x,y, cv=5)

r2 = r2_score(y, y_pred)

print(f'MAE in Sales (Millions): {mean_absolute_error(y_pred, y)}')

print(f'MSE in Sales (Millions): {mean_squared_error(y_pred, y)}')

print(f'R2 Coeff: {r2}')


# ### Lasso Regression with Cross Validation

# In[ ]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV


# In[ ]:


lasso = Lasso()

grid = GridSearchCV(lasso, param_grid={'alpha': range(1, 10)}, refit=True)

y_pred = cross_val_predict(grid, x, y, cv=5)

r2 = r2_score(y, y_pred)

print(f'MAE in Sales (Millions): {mean_absolute_error(y_pred, y)}')

print(f'MSE in Sales (Millions): {mean_squared_error(y_pred, y)}')

print(f'R2 Coeff: {r2}')


# ## Results from Different Models

# It seems that Lasso Regression was able to perform the best in terms of MAE and MSE. I also optimized Alpha with GridSearch and found it to be 1. So in my actual model, that is the alpha parameter I will be using. 

# ## Predictive Modeling with Lasso Regression
# I run the model through 10 different test train splits to see how much the model results variate.

# In[ ]:


def linear_regression_lasso_model(x, y, scalar):
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    x_train = scalar.fit_transform(x_train)
    x_test = scalar.transform(x_test)

    lasso = Lasso(alpha=1)

    lasso.fit(x_train, y_train)

    y_pred = lasso.predict(x_test)

    return {'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred)}


# In[ ]:


lasso_result_list = list()
for i in range(0,10):
    lasso_result_list.append(linear_regression_lasso_model(x,y,scalar))

mae = list()
mse = list()

for result in lasso_result_list:
    mae.append(result['MAE'])
    mse.append(result['MSE'])

print(f'MAE in Sales (Millions): {np.mean(mae)}')
print(f'MSE in Sales (Millions): {np.mean(mse)}')
print('\nTest by Test Results:')

for counter in range(0,len(lasso_result_list)):
    print(f'Test {counter+1}:\n\tMAE: {mae[counter]}\n\tMSE: {mse[counter]}')


# # Summary
# 
# By using cross validation, I was able to get a good idea of which model would perform best. In this case, Lasso Regression outperformed the models by around 10,000 sales. When it came to actually working against the test data set, it performed really well--averaging a MAE of about 580,000 sales. This isn't too bad, considering that the scale of Global Sales is in the millions for this data set. So while it is not pinpoint accurate, it can still give some indication of how a game will do depending on the features I used.

# Thank you for taking the time to explore my kernel!
# If you have any tips, criticisms, or suggestions, please feel free to tell me.
