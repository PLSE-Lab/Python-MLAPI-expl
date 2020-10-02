#!/usr/bin/env python
# coding: utf-8

# # **DATA SCIENCE PROJECT FOR GLOBAL FOOD PRICES**
# 
# ## **Turkey and its Eastern Neighbors**

# My motivation for choosing this data is to analyze the differences in product prices of the country I am living in and its eastern neighbors. I thought that this data was a good starting point to step into the data science. 

# ### **1. IMPORTING LIBRARIES**

# In[ ]:


# for linear algebra
import numpy as np 

# for data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd 

# for visualizations
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# Importing Available files
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# MEMORY REDUCTION FOR BETTER PERFORMANCE
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# Reading the data for analysis.

# In[ ]:


get_ipython().run_line_magic('time', 'd1 = pd.read_csv("/kaggle/input/global-food-prices/wfp_market_food_prices.csv",encoding="ISO-8859-1")')
print(d1.shape)


# In[ ]:


d1.info


# In[ ]:


# improving performance by reducing memory usage to work comfortably on the data.

reduce_mem_usage(d1)


# In[ ]:


# Number of rows for each country

country_unique, country_freq = np.unique(d1['adm0_name'], return_counts = True)
listrows = []
for i in range(country_unique.shape[0]):
    
    print(country_unique[i], ': ', country_freq[i])
    listrows.append( [country_unique[i], country_freq[i]])


# In[ ]:


listrows


# In[ ]:


df = pd.DataFrame(listrows, columns = ['Country', 'RowCount'])
df= df.sort_values(by=['RowCount'],ascending=False)
ax= df.plot(kind='bar', y = 'RowCount',x ='Country',    
    legend = False,figsize=(64,32), fontsize=24)
ax.set_xlabel("Country",fontsize=24)
ax.set_ylabel("RowCount",fontsize=24)
plt.show()


# Above, I plotted bar chart for visualizing the available data number with respect to the countries. Rwanda seems to have way higher than the others and this looks a little bit interesting as I would not have thought there would be this much data coming out of this country.

# ### **2. DATA INSPECTION AND PREPARATION**

# **    IDENTIFICATION OF CUR_NAME COLUMN ( CURRENCY CONVERSION)**

# I would like to understand what is the cur_name column, I guess that it is currency and to check this argument:

# In[ ]:


turkey = d1.loc[d1['adm0_name'] == 'Turkey', 'cur_name']
turkey_unique = np.unique(turkey)
print(turkey)


# Thus, it should be currency abbreviation for Turkish lira.
# 
# In order to compare prices of different countries, we need to represent them in a single currency. I prefer using dollars as it is mostly used.
# 
#  In order to do that, I got currency rates for all countries in 1th January for years 2013 to 2017 and took average of them to convert them to USD.
# https://www.xe.com/currencytables/?from=USD&date=2017-01-01

# In[ ]:


get_ipython().run_line_magic('time', 'cur = pd.read_excel("/kaggle/input/currency/currency.xlsx")')


# In[ ]:


cur.info


# In[ ]:


cur['USD per Unit_Avg']


# #### Below, I renamed the column names of global price data to clearly understand the meaning of them.

# In[ ]:



d1.rename(columns={'adm0_id':'country_id',
                   'adm0_name':'country',
                   'adm1_id':'province_id',
                   'adm1_name':'province',
                   'mkt_id':'city_id',
                   'mkt_name':'city',
                   'cm_id':'food_id',
                   'cm_name':'food',
                   'mp_month':'month',
                   'mp_year':'year',
                   'mp_price':'price',
                   'mp_commoditysource':'source',
                   'um_name':'unit',
                   'cur_name':'currency',
                   'cur_id':'currency_id',
                   'um_id':'unit_id',
                   'pt_name':'purchase_type',
                   'pt_id':'purchase_type_id'},
                   inplace=True)


# In[ ]:


# getting unique currencies

currency_ = d1['currency'].unique()
currency_


# In[ ]:


# turkey, iran, syria, iraq
f1= d1[d1['country'] == 'Turkey'].head(1) 
f2= d1[d1['country'] == 'Iran'].head(1)
f3= d1[d1['country'] == 'Iraq'].head(1)
f4= d1[d1['country'] == 'Syria'].head(1)
result = pd.concat([f1,f2,f3,f4])
result


# In[ ]:


# Looking for Syria if it exists in the data
d1[d1['country'].str.contains('Syria')]


# In[ ]:


# Looking for Iran if it exists in the data
d1[d1['country'].str.contains('Iran')]


# In[ ]:


# Rename our filters as correct names for countries
f1= d1[d1['country'] == 'Turkey'].head(1) 
f2= d1[d1['country'].str.contains('Iran')].head(1)
f3= d1[d1['country'] == 'Iraq'].head(1)
f4= d1[d1['country'].str.contains('Syria')].head(1)
result = pd.concat([f1,f2,f3,f4])
result


# In[ ]:


result['price']


# #### We have spotted the currencies of these four countries. Lets put their respective USD values into new frame.
# 

# In[ ]:



currency_try = cur[cur['Currency_Code'] == 'TRY']
currency_try = currency_try['USD per Unit_Avg']
currency_try


# In[ ]:


currency_irr = cur[cur['Currency_Code'] == 'IRR']
currency_irr = currency_irr['USD per Unit_Avg']
currency_irr


# In[ ]:


currency_iqd = cur[cur['Currency_Code'] == 'IQD']
currency_iqd = currency_iqd['USD per Unit_Avg']
currency_iqd


# In[ ]:


currency_syp = cur[cur['Currency_Code'] == 'SYP']
currency_syp = currency_syp['USD per Unit_Avg']


# In[ ]:


# correlation matrix for the attributes of the data (it is meaningless for categorical data. 
#I am leaving it like this in need of a further use )

#corrmat = d1.corr()
#mask = np.zeros_like(corrmat, dtype=np.bool)
#mask[np.triu_indices_from(mask)] = True

#f, ax = plt.subplots(figsize=(12, 9))
#sns.heatmap(corrmat,mask=mask, vmax=1, square=True);


# In[ ]:


d1.describe()


# **FOOD TYPE COUNT AND HIGHEST PRICE**

# In[ ]:


product_types = d1['food'].unique()
count = 0
for i in product_types:
     count+=1
count


# In[ ]:


# Calculating the average price of all products
items =pd.DataFrame(d1.groupby("food")["price"].agg("mean").sort_values(ascending=False))[:321]
items.head(5)


# Because of the currency difference, it was meaningless to take the average prices of all these products. ( I will create a dataframe from the original one to calculate the respective prices in USD according to all countries)

# **DATA ATTRIBUTES ANALYSIS**

# In[ ]:


d1.columns # Columns of the data.


# In[ ]:


d1.head() # Getting the first 5 rows of the data.


# In[ ]:


d1.tail() # Getting the first 5 rows of the data.


# In[ ]:


country = d1['country'].unique()
count = 0
for i in country: 
    count += 1
    
print("Unique Country Number:",count)

food = d1['food'].unique()
count = 0
for i in food: 
    count += 1
    
print("Unique Food Number:",count)

source = d1['source'].unique()
count = 0
for i in source: 
    count += 1
    
print("Unique Source Number:",count)

purchase_type = d1['purchase_type'].unique()
count = 0
for i in purchase_type: 
    count += 1
    
print("Purchase Type Number:",count)


# **CITIES FOR EACH COUNTRY**

# In[ ]:


# Unique City Count for each Country

dups_color_and_shape = d1.pivot_table(index=['country','city'], aggfunc='size')
dups = d1.drop_duplicates(['country', 'city'], keep='last')
city_count = dups.groupby('country').count()
#city_count
a = city_count.sort_values('country_id',ascending=False)
a.reset_index()
a['country_id']


# In[ ]:


plt.style.use("ggplot")

plt.figure(figsize=(8,10))
plt.gcf().subplots_adjust(left=.3)
fig = sns.barplot(x=a.country_id,y=a.index,data=a)
plt.gca().set_title("City Count For Each Country")
fig.set(xlabel='City Count', ylabel='Country')
plt.show()


# Some countries does not have city names, they have national average prices like Turkey.

# **PREPARING DATA FOR FOUR COUNTRIES(TURKEY,IRAQ,IRAN,SYRIA)**

# In[ ]:


#Turkey
turkey1 = d1.loc[d1['country'] == 'Turkey' , ['country','city','purchase_type','food','unit','price','month','year']]
turkey1


# In[ ]:


#Iran
iran1 = d1.loc[d1['country'].str.contains('Iran') , ['country','city','purchase_type','food','unit','price','month','year']]
iran1


# In[ ]:


#Iraq
iraq1 = d1.loc[d1['country'].str.contains('Iraq') , ['country','city','purchase_type','food','unit','price','month','year']]
iraq1


# In[ ]:


#Syria
syria1 = d1.loc[d1['country'].str.contains('Syria') , ['country','city','purchase_type','food','unit','price','month','year']]
syria1


# # **PRICE CONVERSION TO USD FOR TURKEY, IRAQ, IRAN AND SYRIA**

# ## I have converted Turkey's product prices into USD.

# In[ ]:


turkey = turkey1.copy()
turkey['price'] = turkey['price'] * float(currency_try)
turkey


# ## I have converted Iran's product prices into USD.

# In[ ]:



iran = iran1.copy()
iran['price'] = iran['price'] * float(currency_irr)
iran


# ## I have converted Iraq's product prices into USD.

# In[ ]:



iraq = iraq1.copy()
iraq['price'] = iraq['price'] * float(currency_iqd)
iraq


# ## I have converted Syria's product prices into USD.

# In[ ]:


syria = syria1.copy()
syria['price'] = syria['price'] * float(currency_syp)
syria


# ## **Top Food Prices for Turkey**

# In[ ]:


turkey_top10=pd.DataFrame(turkey.groupby("food")["price","unit"].agg("mean").sort_values(by="price",ascending=False))[:10]
turkey_top = turkey.copy()
turkey_top.sort_values("food", inplace = True) 
turkey_top.drop_duplicates(subset = "food", 
                     keep = 'first', inplace = True)
turkey_top10


# Fuel (gas) has unit in 12 KG, we need to convert it to one KG to better compare it with other products.

# In[ ]:


turkey.loc[turkey['food'] == 'Fuel (gas)', ['price']] = turkey.loc[turkey['food'] == 'Fuel (gas)', ['price']] / 12


# In[ ]:


turkey[turkey['food'].isin(['Fuel (gas)'])]


# As seen above, the prices are converted into unit of one KG

# In[ ]:


plt.style.use("ggplot")
turkey_top10 = turkey.copy()
turkey_top10=pd.DataFrame(turkey.groupby("food")["price"].agg("mean").sort_values(ascending=False))[:10]
plt.figure(figsize=(8,10))
plt.gcf().subplots_adjust(left=.3)
sns.barplot(x=turkey_top10.price,y=turkey_top10.index,data=turkey_top10)
plt.gca().set_title("Top 10 items produced in Turkey")
plt.show()


# After modifying Fuel(gas), it dropped out from the top 10. Milk powder seems to be the most expensive product in Turkey with just under 25 dollars. We can ignore wage as it is not a product, second most expensive is cocoa with 13 dollars.

# **Finding Common Products for These Four Countries**

# In[ ]:


# Taking array of unique products for each country
tu = turkey['food'].unique()
ira = iran['food'].unique()
irq = iraq['food'].unique()
syr = syria['food'].unique()

# Joining arrays to identify these products
all_countries = np.concatenate([tu,ira,irq,syr])

# printing count of products to spot common products
food_unique, food_frequency = np.unique(all_countries, return_counts = True)

for i in range(food_unique.shape[0]):
    
    print(food_unique[i], ': ', food_frequency[i])


# After looking at the result above, I decided to compare the prices of sugar, Wheat flour,Rice, Lentils, Fuel (gas), Yogurt, and Tomatoes.

# In[ ]:


# Finding which countries owns these products

turkey_temp = turkey[turkey['food'].isin(['Wheat flour', 'Sugar','Rice','Lentils','Yogurt','Fuel (gas)', 'Tomatoes'])]
turkey_temp = turkey_temp.copy()
turkey_temp.sort_values("food", inplace = True) 
turkey_temp.drop_duplicates(subset = "food", 
                     keep = 'first', inplace = True)
turkey_temp


# In[ ]:


iran_temp = iran[iran['food'].isin(['Wheat flour', 'Sugar','Rice','Lentils','Yogurt','Fuel (gas)', 'Tomatoes'])]
iran_temp = iran_temp.copy()
iran_temp.sort_values("food", inplace = True) 
iran_temp.drop_duplicates(subset ="food", 
                     keep = 'first', inplace = True)
iran_temp


# In[ ]:


iraq_temp = iraq[iraq['food'].isin(['Wheat flour', 'Sugar','Rice','Lentils','Yogurt','Fuel (gas)', 'Tomatoes'])]
iraq_temp = iraq_temp.copy()
iraq_temp.sort_values("food", inplace = True) 
iraq_temp.drop_duplicates(subset ="food", 
                     keep = 'first', inplace = True)
iraq_temp


# In[ ]:


syria_temp = syria[syria['food'].isin(['Wheat flour', 'Sugar','Rice','Lentils','Yogurt','Fuel (gas)', 'Tomatoes'])]
syria_temp = syria_temp.copy()
syria_temp.sort_values("food", inplace = True) 
syria_temp.drop_duplicates(subset ="food", 
                     keep = 'first', inplace = True)
syria_temp


# Thus all of countries have sugar,
# 
# Turkey, Syria and Iraq have rice,
# 
# Turkey and Syria have Tomatoes,
# 
# Turkey, Syria and Iran have lentils,
# 
# Turkey and Syria have Fuel (gas),
# 
# Turkey, Syria and Iraq have Wheat flour,
# 
# and finally Turkey and Syria have yogurt.
# 
# 
# As there will be too much graphs, I will just look into the distributions of sugar,lentils and rice in these countries.

# In[ ]:


t3 = d1[(d1['country'] == 'Turkey')]
t3['city'].unique()


# In[ ]:


#sns.distplot(a=d1["price"],rug=True)


# **Rice Price Distribution in Turkey, Iraq, Syria**

# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution Price of Rice in Turkey")
turkey_rice = turkey[turkey['food'].isin(['Rice'])]
ax = sns.distplot(turkey_rice["price"], color = 'r')

plt.figure(figsize=(12,5))
plt.title("Distribution Price of Rice in Iraq")
iraq_rice = iraq[iraq['food'].isin(['Rice'])]
ax = sns.distplot(iraq_rice["price"], color = 'b')

plt.figure(figsize=(12,5))
plt.title("Distribution Price of Rice in Syria")
syria_rice = syria[syria['food'].isin(['Rice'])]
ax = sns.distplot(syria_rice["price"], color = 'y')


# **Lentil Price Distribution in Turkey, Iran, Syria**

# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution Price of Lentils in Turkey")
turkey_lentils = turkey[turkey['food'].isin(['Lentils'])]
ax = sns.distplot(turkey_lentils["price"], color = 'r')

plt.figure(figsize=(12,5))
plt.title("Distribution Price of Lentils in Iran")
iran_lentils = iran[iran['food'].isin(['Lentils'])]
ax = sns.distplot(iran_lentils["price"], color = 'g')

plt.figure(figsize=(12,5))
plt.title("Distribution Price of Lentils in Syria")
syria_lentils = syria[syria['food'].isin(['Lentils'])]
ax = sns.distplot(syria_lentils["price"], color = 'y')


# **Sugar Price Distribution in Turkey, Iran, Syria, Iraq**

# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution Price of Sugar in Turkey")
turkey_sugar = turkey[turkey['food'].isin(['Sugar'])]
ax = sns.distplot(turkey_sugar["price"], color = 'r')

plt.figure(figsize=(12,5))
plt.title("Distribution Price of Sugar in Iraq")
iran_sugar = iran[iran['food'].isin(['Sugar'])]
ax = sns.distplot(iran_sugar["price"], color = 'g')

plt.figure(figsize=(12,5))
plt.title("Distribution Price of Sugar in Syria")
syria_sugar = syria[syria['food'].isin(['Sugar'])]
ax = sns.distplot(syria_sugar["price"], color = 'y')

plt.figure(figsize=(12,5))
plt.title("Distribution Price of Sugar in Syria")
iraq_sugar = iraq[iraq['food'].isin(['Sugar'])]
ax = sns.distplot(iraq_sugar["price"], color = 'b')


# There seems to be outlier values for lentil,sugar and rice prices in Syria. Maybe it is due to some monopolies managing the price in certain locations or because of the ongoing war in that country. Supply and demand balance may be changing according to these situations, thus price of lentils may have high variation. 

# **Descriptive Statistics for Sugar,Lentil and Rice Price in Syria**

# Due to outlier values spotted in the plots of Syria, lets have a look into the descriptive statistics of these products in that country.

# In[ ]:


syria_lentils.describe()


# In[ ]:


syria_sugar.describe()


# In[ ]:


syria_rice.describe()


# St.deviations appear too high and max value is way much higher than the mean. Lets look at Turkey's product descriptives.

# **Descriptive Statistics for Lentil Price in Turkey**

# In[ ]:


turkey_lentils.describe()


# As we can see in the statistics of both countries for lentil, standard deviation of lentils in Syria(4,09) is way higher than in Turkey(0,52). Moreover, the highest price is 57.75 USD in Syria in contrast to 2.99 USD in Turkey.

# In[ ]:


turkey_sugar.describe()


# In[ ]:


turkey_rice.describe()


# Same comments are valid for rice and sugar for both countries.

# **Finding Outlier Values for Lentil,Sugar and Rice prices in Syria**

# In[ ]:


q1 = syria_lentils['price'].quantile(.25)
print('Lentils'' q1 is {}'.format(q1))
q3 = syria_lentils['price'].quantile(.75)
print('Lentils'' q3 is {}'.format(q3))
IQR = q3- q1
syria_lentils_v2 = syria_lentils.copy()
syria_lentils_v2.drop(syria_lentils_v2[(syria_lentils_v2['price'] < 0) | (syria_lentils_v2['price'] > (1.5 * IQR)) ].index , inplace=True)
print(syria_lentils_v2['price'].value_counts())


# In[ ]:


q1 = syria_sugar['price'].quantile(.25)
print('Sugar'' q1 is {}'.format(q1))
q3 = syria_lentils['price'].quantile(.75)
print('Sugar'' q3 is {}'.format(q3))
IQR = q3- q1
syria_sugar_v2 = syria_sugar.copy()
syria_sugar_v2.drop(syria_sugar_v2[(syria_sugar_v2['price'] < 0) | (syria_sugar_v2['price'] > (1.5 * IQR)) ].index , inplace=True)
print(syria_sugar_v2['price'].value_counts())


# In[ ]:


q1 = syria_rice['price'].quantile(.25)
print('Rice'' q1 is {}'.format(q1))
q3 = syria_rice['price'].quantile(.75)
print('Rice'' q3 is {}'.format(q3))
IQR = q3- q1
syria_rice_v2 = syria_lentils.copy()
syria_rice_v2.drop(syria_rice_v2[(syria_rice_v2['price'] < 0) | (syria_rice_v2['price'] > (1.5 * IQR)) ].index , inplace=True)
print(syria_rice_v2['price'].value_counts())


# In[ ]:


syria_lentils_v2.describe()


# In[ ]:


syria_sugar_v2.describe()


# In[ ]:


syria_rice_v2.describe()


# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution Price of Lentils in Syria")
ax = sns.distplot(syria_lentils_v2["price"], color = 'y')


# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution Price of Sugar in Syria")
ax = sns.distplot(syria_sugar_v2["price"], color = 'y')


# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution Price of Rice in Syria")
ax = sns.distplot(syria_rice_v2["price"], color = 'y')


# After discarding the outliers we get more accurate price distribution for representing lentil,rice and sugar prices in Syria. There seems to be a binomial distribution in sugar prices and approximately normal distributions in lentil and rice prices. Lets look at the skewness of these distributions before and after removing the outliers.

# In[ ]:


syria_lentils['price'].skew()


# In[ ]:


syria_lentils_v2['price'].skew()


# Before removing outliers, 9.11 value of skewness indicates that the distribution is highly skewed to the left. After removing them, it turns out that the skewness is 0.54 which means that it is on the limit of being assymetric, but we can say that it is nearly symmetric.

# In[ ]:


syria_sugar['price'].skew()


# In[ ]:


syria_sugar_v2['price'].skew()


# There is still high skewness for sugar after removing the outliers. But i guess it is due to the binomial distribution of the sugar prices in Syria.

# In[ ]:


syria_rice['price'].skew()


# In[ ]:


syria_rice_v2['price'].skew()


# After removing the outliers, we have a good value indicating that the distribution is normal.
