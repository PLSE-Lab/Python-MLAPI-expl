#!/usr/bin/env python
# coding: utf-8

# ![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQg5EKdN0yjeDitN4XVZkS3jb86I7Lu0UBcrop0ViQwq-w9uRN-fQ)

#   **1. Introduction**
#   
#   In this kernal, I am investigating the dataset "Countries of the World" by Fernando Lasso. I will be focusing on the factors that affecting a country's GDP per capita and try to make a model using the data of 227 countries from the dataset. I will also briefly discuss the total GDPs. For me, this is a practise of EDA and virsualization, if you have any suggestion on how to make more extensive analysis please kindly let me know, Any feedback is greatly appreciated!

# **2. Load libraries**

# In[ ]:


import numpy as np #linear algebra
import pandas as pd #data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from matplotlib import pyplot as plt

#data are available in the "../input/" directory

import os
print(os.listdir("../input/")) #for example, running this (click Shift+Enter) will list the files in the input directory

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_squared_log_error


# **3. Table overview**

# In[ ]:


data = pd.read_csv('../input/countries of the world.csv', decimal=',')
print('number of missing data:')
print(data.isnull().sum())
data.describe(include='all')


# **4. Data preparation - fill in missing data**
# 
# We noticed that there are some missing data in the table. For simplicity, I will just fill the missing data using the median of the region that a country belongs, as countries that are close geologically are often similar in many ways. For example, lets check the region median of 'GDP ($ per capita)', 'Literacy (%)' and 'Agriculture'. Note that for 'climate' we use the mode instead of median as it seems that 'climate' is a categorical feature here.
# 

# In[ ]:


data.groupby('Region')[['GDP ($ per capita)', 'Literacy (%)', 'Agriculture']].median()


# In[ ]:


for col in data.columns.values:
    if data[col].isnull().sum() == 0:
        continue
    if col == 'Climate':
        guess_values = data.groupby('Region')['Climate'].apply(lambda x: x.mode().max())
    else:
        guess_values = data.groupby('Region')[col].median()
    for region in data['Region'].unique():
        data[col].loc[(data[col].isnull())&(data['Region']==region)] = guess_values[region]


# In[ ]:


print(data.isnull().sum()) #check if we filled all missing values


# **5. Data exploration**
# 
# *Correlation heatmap show the correlation between all numericall columns*
# 

# In[ ]:


plt.figure(figsize=(16,12))
sns.heatmap(data=data.iloc[:,2:].corr(),annot=True,fmt='.2f',cmap='coolwarm')
plt.show()


# Look at the top 20 countries with hioghest GDP per capita, check the first one :)

# In[ ]:


fig, ax = plt.subplots(figsize=(16,6))
top_gdp_countries = data.sort_values('GDP ($ per capita)',ascending=False).head(20)
mean = pd.DataFrame({'Country':['World mean'], 'GDP ($ per capita)':[data['GDP ($ per capita)'].mean()]})
gdps = pd.concat([top_gdp_countries[['Country','GDP ($ per capita)']],mean],ignore_index=True)
sns.barplot(x='Country', y='GDP ($ per capita)', data=gdps, palette='Set1')
ax.set_xlabel(ax.get_xlabel(), labelpad=15)
ax.set_ylabel(ax.get_ylabel(), labelpad=30)
ax.xaxis.label.set_fontsize(16)
ax.yaxis.label.set_fontsize(16)
plt.xticks(rotation=90)
plt.show()


# In[ ]:


fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20,20))
plt.subplots_adjust(hspace=0.4)

corr_to_gdp = pd.Series()
for col in data.columns.values[2:]:
    if ((col!='GDP ($ per capita)')&(col!='Climate')):
        corr_to_gdp[col] = data['GDP ($ per capita)'].corr(data[col])
abs_corr_to_gdp = corr_to_gdp.abs().sort_values(ascending=False)
corr_to_gdp = corr_to_gdp.loc[abs_corr_to_gdp.index]

for i in range(3):
    for j in range(3):
        sns.regplot(x=corr_to_gdp.index.values[i*3+j], y='GDP ($ per capita)', data=data,
                   ax=axes[i,j], fit_reg=False, marker='.')
        title = 'correlation='+str(corr_to_gdp[i*3+j])
        axes[i,j].set_title(title)
axes[1,2].set_xlim(0,102)
plt.show()


# **Countries with low birthrate and low GDP per capita**
# 
# Some features, like phones, are related to the everage GDP more linearlz, while others ano not. For example, high birthrate usuallz means low GDP per capita, but average  GDP in low birthrate countries can vary a  lot.
# 

# In[ ]:


data.loc[(data['Birthrate']<14) & (data['GDP ($ per capita)'] < 10000)]


# **6. Modeling**
# 
# **Training and testing**
# 
# First label encode the categorical features 'Regrion' and 'Climate', and while just use all features given in the dataset without further enginerring.

# In[ ]:


LE = LabelEncoder()
data['Regional_label'] = LE.fit_transform(data['Region'])
data['Climate_label'] = LE.fit_transform(data['Climate'])
data.head(10)


# In[ ]:


train, test = train_test_split(data, test_size=0.3, shuffle=True)
training_features = ['Population', 'Area (sq. mi.)',
       'Pop. Density (per sq. mi.)', 'Coastline (coast/area ratio)',
       'Net migration', 'Infant mortality (per 1000 births)',
       'Literacy (%)', 'Phones (per 1000)',
       'Arable (%)', 'Crops (%)', 'Other (%)', 'Birthrate',
       'Deathrate', 'Agriculture', 'Industry', 'Service', 'Regional_label',
       'Climate_label','Service']
target = 'GDP ($ per capita)'
train_X = train[training_features]
train_Y = train[target]
test_X = test[training_features]
test_Y = test[target]


# In[ ]:


model = LinearRegression()
model.fit(train_X, train_Y)
train_pred_Y = model.predict(train_X)
test_pred_Y = model.predict(test_X)
train_pred_Y = pd.Series(train_pred_Y.clip(0, train_pred_Y.max()), index=train_Y.index)
test_pred_Y = pd.Series(test_pred_Y.clip(0, test_pred_Y.max()), index=test_Y.index)

rmse_train = np.sqrt(mean_squared_error(train_pred_Y, train_Y))
msle_train = mean_squared_log_error(train_pred_Y, train_Y)
rmse_test = np.sqrt(mean_squared_error(test_pred_Y, test_Y))
msle_test = mean_squared_log_error(test_pred_Y, test_Y)

print('rmse_train:',rmse_train,'msle_train:',msle_train)
print('rmse_test:',rmse_test,'msle_test:',msle_test)

