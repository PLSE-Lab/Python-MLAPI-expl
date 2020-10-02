#!/usr/bin/env python
# coding: utf-8

# This kernel is an attempt to explain my approach on Imarticus Car price prediction Hackathon.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_excel('../input/Data_Train.xlsx')
test = pd.read_excel('../input/Data_Test.xlsx')
example = pd.read_excel('../input/Sample_submission.xlsx')


# ## Data Inspection[](http://)

# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


example.head()


# In[ ]:


print('Training dataset has {} row and {} columns'.format(train.shape[0], train.shape[1]))
print('Testing dataset has {} row and {} columns'.format(test.shape[0], test.shape[1]))


# In[ ]:


train.describe()


# In[ ]:


train.info()


# Most of the entries in New_Price columns are empty

# In[ ]:


test.info()


# In[ ]:


combined = train.append(test, ignore_index=True, sort=False)


# In[ ]:


combined.shape


# In[ ]:


combined.head()


# In[ ]:


combined.tail()


# ### Observations
# 
# - We have to predict the Price
# - Train and test datasets has New Price column which has missing values.
# 

# In[ ]:


train_locations = train.Location.unique()
print(train_locations)


# In[ ]:


all_locations = combined.Location.unique()
print(all_locations)


# In[ ]:


train_car_names = train.Name.unique()
print('There are {} car models in our dataset'.format(len(train_car_names)))


# In[ ]:


all_car_names = combined.Name.unique()
print('There are {} car models in our dataset'.format(len(all_car_names)))


# In[ ]:


car_counts = {}
for car in combined.Name:
    if car not in car_counts:
        car_counts[car] = 1
    else:
        car_counts[car] += 1
        
print(len(car_counts))


# In[ ]:


car_count_df = pd.DataFrame.from_dict(car_counts, orient='index', columns=['Count'])
car_count_df.head()


# In[ ]:


car_count_df = car_count_df.sort_values(by='Count', ascending=False)
car_count_df.head()


# In[ ]:


plt.figure(figsize=(15,8))
ax = sns.barplot(data=car_count_df.iloc[:50], x=car_count_df.iloc[:50].index, y='Count')
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
plt.show()


# In[ ]:


location_price = train.groupby(by='Location').mean()['Price']
location_price


# In[ ]:


plt.figure(figsize=(12,6))
location_price.plot(kind='bar', grid=True, rot=0, )
plt.ylabel('Average Price (in lakhs)')
plt.show()


# In[ ]:


all_location_count = combined.groupby(by='Location').count()['Name']
all_location_count


# In[ ]:


train_location_count = train.groupby(by='Location').count()['Name']
train_location_count


# In[ ]:


plt.figure(figsize=(12,6))
all_location_count.plot(kind='bar', grid=True, rot=0, )
plt.ylabel('No. of cars in complete dataset')
plt.show()


# In[ ]:


plt.figure(figsize=(12,6))
train_location_count.plot(kind='bar', grid=True, rot=0, )
plt.ylabel('No. of cars in train dataset')
plt.show()


# In[ ]:


train.groupby(['Fuel_Type'])['Price'].mean().plot.bar(title = 'Average price vs Fuel type', rot=0)
plt.show()


# In[ ]:


train.groupby(['Year'])['Price'].mean().plot.bar(title = 'Average price vs Year')
plt.show()


# In[ ]:


train.groupby(['Owner_Type'])['Price'].mean().plot.bar(title = 'Average price vs Owner Type', rot=0)
plt.show()


# In[ ]:


combined.insert(1, 'Company', combined.Name.apply(lambda x : str.title(x).split()[0]))


# In[ ]:


combined.head()


# In[ ]:


plt.figure(figsize=(15,4))
combined.groupby(['Company'])['Price'].mean().plot.bar(title = 'Average price vs Company', rot=45)
plt.show()


# In[ ]:


plt.figure(figsize=(15,4))
combined.groupby(['Company'])['Name'].count().plot.bar(title = 'Number of Cars per Company', rot=45)
plt.show()


# In[ ]:


sns.distplot(train.Price)
plt.show()


# In[ ]:


train.plot.scatter(y = 'Price', x='Kilometers_Driven', rot=45)
plt.show()


# In[ ]:


combined[combined.Kilometers_Driven > 6000000]


# Kilometers driven shown for above car seems to be insane and probably impossible. So we set to a more likely value like 65000 kms

# In[ ]:


combined.loc[2328, 'Kilometers_Driven'] = 65000


# In[ ]:


combined.loc[2328]


# In[ ]:


combined[combined.Kilometers_Driven > 400000]


# In[ ]:


train.New_Price.unique()


# In[ ]:


train.dtypes


# Before starting data pre-processing let us combine both train and test data sets.

# In[ ]:


combined.New_Price = combined.New_Price.fillna(0)


# In[ ]:


def clean_price(price):
    
    '''Converts Prices to floats'''
    
    if price == 0:
        return price
    
    elif 'Cr' in price:
        price = price.split()[0]
        new_price = float(price)*1000
        return new_price
    
    else:
        new_price = float(price.split()[0])
        return new_price

combined.New_Price = combined.New_Price.apply(lambda x : clean_price(x))


# In[ ]:


combined[combined.New_Price != 0]


# In[ ]:


combined[combined.New_Price == 0]


# In[ ]:


combined[combined.New_Price == 0].shape


# There are lots of missing values in New Price column.

# We can see that in Mileage column there are two units of measurement kmpl and km/kg. Let us have a look into it.

# In[ ]:


combined.Mileage.unique()


# In[ ]:


combined[combined.Mileage.str.contains('km/kg') == True]


# It is clear that km/kg unit is for cars with CNG/LPG as fuel. Now we have to do an effective conversion of these to a common metric

# Most simplest one I could think of was to convert Mileage to Cost per km. For this I used prices of CNG, LPG, Petrol and Diesel in Mumbai, since as per our training dataset Mumbai has most number of cars.

# In[ ]:


cng_per_kg = 51.57
lpg_per_kg = 49.96 #(709.50/14.2)
diesel_per_ltr = 67.40
petrol_per_ltr = 76.15

def convert_mileage(data):
    #print(data)
    #print(type(data))
    if data.Fuel_Type == 'CNG':
        data.Mileage = float(data.Mileage.split()[0]) / cng_per_kg

    elif data.Fuel_Type == 'LPG':
        data.Mileage = float(data.Mileage.split()[0]) / lpg_per_kg
        
    elif data.Fuel_Type == 'Petrol':
        data.Mileage = float(data.Mileage.split()[0]) / petrol_per_ltr
        
    elif data.Fuel_Type == 'Diesel':
        data.Mileage = float(data.Mileage.split()[0]) / diesel_per_ltr
        
    return data


# In[ ]:


combined = combined.apply(lambda x:convert_mileage(x), axis=1)


# In[ ]:


combined.Mileage.unique()


# In[ ]:


combined[combined.Mileage == 0]


# In[ ]:


combined[combined.Mileage == 0].index


# In[ ]:


train.iloc[[  14,   67,   79,  194,  229,  262,  307,  424,  443,  544,  631,
             647,  707,  749,  915,  962,  996, 1059, 1259, 1271, 1308, 1345,
            1354, 1385, 1419, 1460, 1764, 1857, 2053, 2096, 2130, 2267, 2343,
            2542, 2597, 2681, 2780, 2842, 3033, 3044, 3061, 3093, 3189, 3210,
            3271, 3516, 3522, 3645, 4152, 4234, 4302, 4412, 4629, 4687, 4704,
            5016, 5022, 5119, 5270, 5311, 5374, 5426, 5529, 5647, 5875, 5943,
            5972, 6011]]


# In[ ]:


combined[combined.Mileage == 0].shape


# In[ ]:


combined.Mileage = combined.Mileage.fillna(0)


# In[ ]:


combined[combined.Mileage == 0].shape


# In[ ]:


combined[combined.Mileage == 0]


# In[ ]:


combined.head(10)


# In[ ]:


def clean_engine_power(data):
    
    #print(data.Engine, data.Power)
    data.Engine = data.Engine.split()[0]
    data.Power = data.Power.split()[0]
    
    return data


# In[ ]:


combined.Engine = combined.Engine.fillna('0 CC')
combined.Power = combined.Power.fillna('0 bhp')


# In[ ]:


combined[combined.Engine == '0 CC']


# In[ ]:


combined = combined.apply(lambda x: clean_engine_power(x), axis=1)


# In[ ]:


combined.Engine.unique()


# In[ ]:


len(combined.Engine.unique())


# In[ ]:


combined[combined.Engine == '0']


# In[ ]:


combined[combined.Engine == '0'].shape


# In[ ]:


combined.Power.unique()


# In[ ]:


combined[combined.Power == 'null'].shape


# In[ ]:


combined[combined.Power == '0'].shape


# In[ ]:


combined.Power = combined.Power.replace('null', '0')


# In[ ]:


combined[combined.Power == '0'].shape


# In[ ]:


combined.Power.unique()


# In[ ]:


combined.head()


# In[ ]:


combined.Power.unique()


# In[ ]:


combined[combined.Power == '0']


# In[ ]:


combined[combined.Power == '0'].shape


# In[ ]:


combined.Seats.unique()


# In[ ]:


combined.Seats = combined.Seats.fillna(99)


# In[ ]:


combined[combined.Seats == 99 ].index


# In[ ]:


combined.Seats = combined.Seats.replace(99, np.NaN)


# In[ ]:


combined.Seats.unique()


# In[ ]:


combined.Seats = combined.Seats.interpolate('nearest')


# In[ ]:


combined.Seats.unique()


# In[ ]:


combined[combined.Seats == 10.]


# In[ ]:


train.iloc[[ 194,  208,  229,  733,  749, 1294, 1327, 1385, 1460, 1917, 2074,
            2096, 2264, 2325, 2335, 2369, 2530, 2542, 2623, 2668, 2737, 2780,
            2842, 3272, 3404, 3520, 3522, 3800, 3810, 3882, 4011, 4152, 4229,
            4577, 4604, 4697, 4712, 4952, 5015, 5185, 5270, 5893]]


# In[ ]:


combined[combined.Seats == 0.]


# Audi A4 is a 5 seater car

# In[ ]:


combined.iloc[3999, combined.columns.get_loc('Seats')] = 5.0


# In[ ]:


combined.iloc[3999]


# In[ ]:


combined[['Engine', 'Power']] = combined[['Engine', 'Power']].apply(pd.to_numeric)


# In[ ]:


combined.dtypes


# In[ ]:


categories = ['Location', 'Company', 'Year', 'Fuel_Type', 'Transmission', 
              'Owner_Type', 'Seats',]

for cols in categories:
    combined[cols] = combined[cols].astype('category')


# In[ ]:


combined.dtypes


# In[ ]:


combined.describe()


# In[ ]:


combined.info()


# In[ ]:


combined.isnull().sum(axis=0)


# In[ ]:


combined = combined.apply(lambda x : x.replace(0, np.NaN))


# In[ ]:


combined.isnull().sum(axis=0)


# In[ ]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')

cols_impute = ['Mileage', 'Engine', 'Power']
combined[cols_impute] = imputer.fit_transform(combined[cols_impute])


# In[ ]:


combined.isnull().sum()


# In[ ]:


combined.columns


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[ ]:


cols_clean = ['Company', 'Location', 'Year', 'Kilometers_Driven', 'Fuel_Type',
       'Transmission', 'Owner_Type', 'Mileage', 'Engine', 'Power', 'Seats',
       'New_Price', 'Price']
combined_clean = combined[cols_clean]


# In[ ]:


combined_clean[['Kilometers_Driven', 'Mileage', 'Engine', ]] = 


# In[ ]:


combined_clean = pd.get_dummies(combined_clean, drop_first=True)


# In[ ]:


combined_clean.shape


# In[ ]:


combined_clean.describe()


# In[ ]:


combined_clean.info()


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[ ]:


combined_clean = scaler.fit_transform(combined_clean)


# ### Modelling to predict New Price

# In[ ]:


df_new_price = combined_clean.drop(['Price'], axis=1).sort_values(by='New_Price', ascending=False).reset_index(drop=True)
df_new_price.head()


# In[ ]:


df_new_price.tail()


# In[ ]:


df_new_price.isnull().sum()


# In[ ]:


7253-6247


# In[ ]:


df_new_price.iloc[1005]


# In[ ]:


df_new_price.iloc[1006]


# In[ ]:


train_new_price = df_new_price.iloc[:1006]
test_new_price = df_new_price.iloc[1006:]


# In[ ]:


x_train_new_price = train_new_price.drop(['New_Price'], axis=1)
y_train_new_price = train_new_price['New_Price']


# In[ ]:


print(x_train_new_price.shape, y_train_new_price.shape)


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
lr = LinearRegression()


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x_train_new_price, y_train_new_price, test_size=0.3, random_state = 11)


# In[ ]:


lr.fit(x_train, y_train)


# In[ ]:



lr = LinearRegression()


# In[ ]:





# In[ ]:


#pd.get_dummies(train_new)
train_new.isnu


# In[ ]:


train_columns = ['Name', 'Location', 'Company', 'Year', 'Kilometers_Driven', 'Fuel_Type',
                 'Transmission', 'Owner_Type', 'Mileage', 'Engine', 'Power', 'Seats']
#df_train = train[train_columns]
#y_new_price = train['New_Price']
#y_price = train['Price']


# In[ ]:


df_train.Location.unique()


# In[ ]:


df_train.Year.unique()


# In[ ]:


df_train.Fuel_Type.unique()


# In[ ]:


df_train.Transmission.unique()


# In[ ]:


df_train.Owner_Type.unique()


# In[ ]:


df_train.Mileage.unique()


# In[ ]:





# In[ ]:





# In[ ]:


df_train.Engine.unique()


# In[ ]:


df_train.Power.unique()


# In[ ]:


df_train[df_train.Seats == df_train.Seats.unique()[-1]]


# In[ ]:


df_train.isnull().sum()


# In[ ]:


print(train.shape)
print(y_new_price.shape)
print(y_price.shape)


# In[ ]:


df_train.head()


# In[ ]:


y_new_price.isnull().sum()

