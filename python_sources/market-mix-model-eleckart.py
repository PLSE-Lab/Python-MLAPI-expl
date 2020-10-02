#!/usr/bin/env python
# coding: utf-8

# # Problem Statement

# ElecKart is an e-commerce firm based out of Ontario, Canada specialising in electronic products. Over the last one year, they had spent a significant amount of money on marketing. Occasionally, they had also offered big-ticket promotions (similar to the Big Billion Day). They are about to create a marketing budget for the next year, which includes spending on commercials, online campaigns, and pricing & promotion strategies. The CFO feels that the money spent over the last 12 months on marketing was not sufficiently impactful, and, that they can either cut on the budget or reallocate it optimally across marketing levers to improve the revenue response.
# 
#  
# 
# Imagine that you are a part of the marketing team working on budget optimisation. You need to develop a market mix model to observe the actual impact of different marketing variables over the last year. Using your understanding of the model, you have to recommend the optimal budget allocation for different marketing levers for the next year.

# # 1. Data Reading And Understanding

# In[ ]:


import warnings
warnings.filterwarnings('ignore')

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
pd.set_option('display.float_format', '{:0.3f}'.format)


# In[ ]:


# Set ipython's max row display
pd.set_option('display.max_row', 500)

# Set iPython's max column width to 100
pd.set_option('display.max_columns', 100)


# In[ ]:


# Read the file
consumer = pd.read_csv('../input/eleckartdata/ConsumerElectronics.csv')


# In[ ]:


# Take a look at the first 5 rows
consumer.head()


# In[ ]:


# Get info about the dataset
consumer.info()


# - There are no null values in the dataset!
# - Columns like order_date, gmv, deliverybdays, deliverycdays, pincode have incorrect data types and need to be changed.

# In[ ]:


# Let's take a look at the statistical info of the dataset
consumer.describe(percentiles = [0.25, 0.5, 0.75, 0.90, 0.99, 0.999])


# - All the numeric columns are almost normally distributed!
# - We need to cap the SLAs to max and min values.

# # 2. Data Cleaning And Preparation

# In[ ]:


consumer.columns


# In[ ]:


consumer.replace(r'^\s+$', np.nan, regex=True, inplace = True)
consumer.replace('\\N', np.nan, inplace = True)


# In[ ]:


# let's check the null percentage for each column
round(100*(consumer.isnull().sum()/len(consumer.index)), 2)


# In[ ]:


#removing null valued GMV
consumer = consumer.loc[~(consumer.gmv.isnull())]


# In[ ]:


# let's check the null percentage for each column again
round(100*(consumer.isnull().sum()/len(consumer.index)), 2)


# #### 1. product_analytic_super_category, product_analytic_category, product_analytic_sub_category, product_analytic_vertical

# In[ ]:


# Let's drop the rows that have product analytic vertical as null.
consumer = consumer[~pd.isnull(consumer.product_analytic_vertical)]


# In[ ]:


# Let's now check the product_analytic_super_category unique values
consumer.product_analytic_super_category.unique()


# - There is only one value for this column. Hence, we can remove the column.

# In[ ]:


consumer.drop('product_analytic_super_category',1, inplace = True)


# In[ ]:


consumer.product_analytic_category.unique()


# In[ ]:


consumer.product_analytic_sub_category.unique()


# In[ ]:


#The three product sub categories for the MMM are - camera accessory, home audio and gaming accessory.
#Removing the rows with other sub categories

consumer = consumer.loc[(consumer.product_analytic_sub_category=='CameraAccessory') |
                       (consumer.product_analytic_sub_category=='GamingAccessory')|
                       (consumer.product_analytic_sub_category=='HomeAudio')]


# In[ ]:


consumer.product_analytic_vertical.unique()


# #### 2. gmv

# In[ ]:


#Let's convert the data type of GMV

consumer['gmv'] = pd.to_numeric(consumer['gmv'])


# In[ ]:


#Checking the minimum and maximum values of GMV
print(consumer.gmv.min())
print(consumer.gmv.max())


# gmv (Gross Merchendising Value - The cost price at which the item is sold multiplied by number of units) - Value at 0.0 seems odd. 
# 
# Assumption : It could be because of any promotional offers, hence not deleting them
# 

# Looks like a duplicated order. Let's check for duplicates

# In[ ]:


consumer[consumer.duplicated(['fsn_id','order_date','order_id','order_item_id',
                              'gmv','cust_id','pincode',
                              'product_analytic_category','product_analytic_sub_category',
                             'product_analytic_vertical'])]
#consumer.loc[consumer.duplicated()]


# In[ ]:


len(consumer[consumer.duplicated(['fsn_id','order_date','order_id','order_item_id',
                              'gmv','cust_id','pincode',
                              'product_analytic_category','product_analytic_sub_category',
                             'product_analytic_vertical'])])


# In[ ]:


#Removing duplicated values
consumer = consumer[~consumer.duplicated(['fsn_id','order_date','order_id','order_item_id',
                              'gmv','cust_id','pincode',
                              'product_analytic_category','product_analytic_sub_category',
                             'product_analytic_vertical'])]


# In[ ]:


consumer.loc[consumer.duplicated()]


# In[ ]:


#Checking nulls in gmv value
consumer.gmv.isnull().sum()


# In[ ]:


consumer.shape


# #### 3. deliverybdays and deliverycdays

# In[ ]:


# The columns deliverybdays and deliverycdays are populated with \N, which is incorrect.
# Let's replace them with null.
print(consumer.deliverybdays.value_counts().head())
print(consumer.deliverycdays.value_counts().head())


# In[ ]:


print(consumer.deliverybdays.isnull().sum()/len(consumer))
print(consumer.deliverycdays.isnull().sum()/len(consumer))


# In[ ]:


# We can drop delivercdays and deliverybdays column as it has 79% null values.
consumer.drop(['deliverybdays', 'deliverycdays'],1, inplace = True)


# #### 4. order_date

# In[ ]:


# Befor dealing with null values, let's first correct the data type of order_date
consumer['order_date'] = pd.to_datetime(consumer['order_date'])


# In[ ]:


# We now need to check if the dates are not outside July 2015 and June 2016.
consumer.loc[(consumer.order_date < '2015-07-01') | (consumer.order_date >= '2016-07-01')]


# - There is 608 records that lie outside the range. Let's delete those rows.

# In[ ]:


consumer = consumer.loc[(consumer.order_date >= '2015-07-01')]
consumer = consumer.loc[(consumer.order_date < '2016-07-01')]


# #### 5. s1_fact.order_payment_type

# In[ ]:


#Changing the name of the column s1_fact.order_payment_type
consumer.rename(columns={'s1_fact.order_payment_type':'order_payment_type'}, inplace=True)


# In[ ]:


consumer.order_payment_type.value_counts()


# Clearly COD is preferred more than Prepaid order payment type.

# #### 6. pincode, custid

# In[ ]:


#Converting the datatype
consumer['pincode'] = pd.to_numeric(consumer['pincode'])


# In[ ]:


#Let's see the values of pincode field
consumer.pincode.min()


# In[ ]:


consumer.pincode.isnull().sum()


# In[ ]:


# Before handling null values, there are negative values for pincode which we need to handle.
# Let's make all the negative values as positive.
consumer.pincode = consumer.pincode.abs()


# In[ ]:


# Let's now check the frequency of pincodes to decide whether we can impute the missing pincodes with the highest frequency one.
consumer.pincode.value_counts()


# In[ ]:


#pincode and cust_id doesn't seem to be of any use


# In[ ]:


consumer.drop(['cust_id','pincode'], axis = 1, inplace = True)


# #### 7. product_mrp

# In[ ]:


consumer[(consumer.product_mrp == 0)].head()


# In[ ]:


len(consumer[(consumer.product_mrp == 0)])


# In[ ]:


#Removing values with 0 MRP, since that is not possible at all
consumer = consumer.loc[~(consumer.product_mrp==0)]


# In[ ]:


consumer['gmv_per_unit'] = consumer.gmv/consumer.units


# In[ ]:


#Replacing the values of MRP with GMV per unit where the values of GMV/unit is greater than MRP
consumer['product_mrp'].loc[consumer.gmv_per_unit>consumer.product_mrp] = consumer['gmv_per_unit']


# In[ ]:


consumer.loc[consumer.gmv_per_unit>consumer.product_mrp]


# In[ ]:


consumer.drop(['gmv_per_unit'],1,inplace=True)


# #### 8. sla and product_procurement_sla

# In[ ]:


consumer.shape


# In[ ]:


plt.figure(figsize=(20,8))

plt.subplot(1,2,1)
sns.boxplot(y=consumer.sla, palette=("cubehelix"))

plt.subplot(1,2,2)
sns.boxplot(y=consumer.product_procurement_sla, palette=("cubehelix"))


# In[ ]:


plt.figure(figsize=(20,8))

plt.subplot(1,2,1)
sns.distplot(consumer.sla)

plt.subplot(1,2,2)
sns.distplot(consumer.product_procurement_sla)


# In[ ]:


consumer.sla.describe(percentiles=[0.0,0.25,0.5,0.75,0.9,0.95,0.99,1.0])


# In[ ]:


consumer.product_procurement_sla.describe(percentiles=[0.0,0.25,0.5,0.75,0.9,0.95,0.99,1.0])


# In[ ]:


#Converting negative values to the positive
len(consumer.loc[consumer.product_procurement_sla<0])


# In[ ]:


consumer.product_procurement_sla = abs(consumer.product_procurement_sla)


# In[ ]:


consumer.sla.std()


# In[ ]:


#Taking three sigma values for outliers treatment
print(consumer.sla.mean()+(3*(consumer.sla.std())))
print(consumer.sla.mean()-(3*(consumer.sla.std())))


# In[ ]:


consumer.product_procurement_sla.std()


# In[ ]:


#Taking three sigma values for outliers treatment
print(consumer.product_procurement_sla.mean()+(3*(consumer.product_procurement_sla.std())))
print(consumer.product_procurement_sla.mean()-(3*(consumer.product_procurement_sla.std())))


# In[ ]:


# Capping the values at three sigma value
len(consumer[consumer.sla > 14])


# In[ ]:


# Let's cap the SLAs.
consumer.loc[consumer.sla > 14,'sla'] = 14


# In[ ]:


# Similarly, the min value of product procurement sla is 0 and the max value is 15. However, three sigma value is 7. 
print(len(consumer[consumer.product_procurement_sla > 7]))


# In[ ]:


# Let's cap the product procuremtn SLAs.
consumer.loc[consumer.product_procurement_sla > 7,'product_procurement_sla'] = 7


# In[ ]:


consumer.shape


# In[ ]:


consumer.loc[consumer.duplicated()]


# #### Duplicates removal

# In[ ]:


len(consumer[consumer.duplicated(['order_id','order_item_id'])])


# - Clearly, there can't be two orders with the same combination of order id and order item id that were ordered at the same timestamp.
# - We can hence, drop the duplicates.

# In[ ]:


consumer = consumer[~consumer.duplicated(['order_id','order_item_id'])]


# In[ ]:


consumer.describe()


# In[ ]:


plt.figure(figsize=(20,8))

plt.subplot(1,2,1)
sns.distplot(consumer.gmv)

plt.subplot(1,2,2)
sns.distplot(consumer.product_mrp)

plt.show()


# # 3. Feature Engineering and KPI

# ### Pricing KPI

# #### Listed Price

# In[ ]:


#2. gmv (Gross Merchendising Value - The cost price at which the item is sold multiplied by number of units)

# Let's derive listing price, which is nothing but gmv/units

consumer['listing_price'] = round((consumer.gmv/consumer.units),2)


# In[ ]:


#Let's check if there are any rows with listing price > MRP

len(consumer.loc[consumer.listing_price>consumer.product_mrp])


# ### Discount and Promotion related KPI

# #### Dicount

# In[ ]:


# Let's now calculate the discount %, which is nothing but (mrp-list price)/mrp
consumer['discount'] = round(((consumer.product_mrp - consumer.listing_price)/(consumer.product_mrp)),2)


# In[ ]:


consumer['discount'].describe()


# #### Total Price

# In[ ]:


consumer['Order_Item_Value'] = consumer['product_mrp'] * consumer['units']


# ### Mapping Week into the Data

# In[ ]:


# We can create the week number
consumer['week'] = np.where(consumer.Year == 2015, (consumer.order_date.dt.week - pd.to_datetime('2015-07-01').week + 1), consumer.order_date.dt.week+27)

# Dates like 2016-01-01 will be 53rd week as per ISO standard, hence the week value would be 53+27=80.
# We can make those values as week 27
consumer.week.values[(consumer.Year == 2016) & (consumer.week == 80)] = 27


# ### Product assortment and quality related KPI

# #### Payment type

# In[ ]:


### Prepaid = '1' or COD = '0'
consumer['order_payment_type'] = np.where(consumer['order_payment_type'] == "Prepaid",1,0)


# ### Seasonality and Trend related KPI

# #### Calendar

# In[ ]:


### Creating Calendar for the period
calendar = pd.DataFrame(pd.date_range('2015-07-01','2016-06-30').tolist(), columns = ['Date'])
### Mapping week in the calendar
calendar['week'] = calendar.Date.dt.week
### Jan 2016 should be week 54 ,not week 1.
calendar['week'] = np.where((calendar['week'] <= 26) & (calendar.Date.dt.year == 2016), calendar['week']+53, calendar['week'])


# #### Special Sale

# In[ ]:


### Special Sales List

special_sales_list = ["2015-07-18","2015-07-19","2015-08-15","2015-08-16","2015-08-17","2015-08-28","2015-08-29",
                      "2015-08-30","2015-10-15","2015-10-16","2015-10-17","2015-11-07","2015-11-08","2015-11-09",
                      "2015-11-10","2015-11-11","2015-11-12","2015-11-13","2015-11-14","2015-12-25","2015-12-26",
                      "2015-12-27","2015-12-28","2015-12-29","2015-12-30","2015-12-31","2016-01-01","2016-01-02",
                      "2016-01-03","2016-01-20","2016-01-21","2016-01-22","2016-02-01","2016-02-02","2016-02-14",
                      "2016-02-15","2016-02-20","2016-02-21","2016-03-07","2016-03-08","2016-03-09","2016-05-25",
                      "2016-05-26","2016-05-27"]

ss_list = pd.DataFrame(special_sales_list,columns = ['Date'])
ss_list['Date'] = pd.to_datetime(ss_list['Date'])
ss_list['Special_sales'] = True


# In[ ]:


calendar = calendar.merge(ss_list, 'left')
calendar.fillna(False, inplace = True)


# In[ ]:


calendar['Special_sales'] = calendar['Special_sales'].astype(int)


# In[ ]:


calendar.head()


# #### Payday

# In[ ]:


calendar['Payday'] = ((calendar['Date'].dt.day == 1) | (calendar['Date'].dt.day == 15)).astype(int)


# #### Climate Data

# In[ ]:


### Ontario Climate data of year 2015-2016 
ontario_climate_2015 = pd.DataFrame(pd.read_csv('../input/eleckartdata/ONTARIO-2015.csv',encoding="ISO-8859-1",skiprows=24))
ontario_climate_2016 = pd.DataFrame(pd.read_csv('../input/eleckartdata/ONTARIO-2016.csv',encoding="ISO-8859-1",skiprows=24))


# In[ ]:


### Merge Calendar with dataset on week

ontario_climate = ontario_climate_2015.append(ontario_climate_2016)
ontario_climate = ontario_climate.reset_index()
ontario_climate.head()


# In[ ]:


### Checking for any nan values

round((ontario_climate.isnull().sum()/len(ontario_climate.index))*100,2)


# In[ ]:


### Dropping columns we do not require in the analysis.
ontario_climate.drop(['index','Data Quality','Max Temp Flag','Min Temp Flag','Mean Temp Flag',
                      'Heat Deg Days Flag','Cool Deg Days Flag','Total Rain Flag','Total Snow Flag',
                      'Total Precip Flag','Snow on Grnd Flag','Dir of Max Gust (10s deg)','Dir of Max Gust Flag',
                      'Spd of Max Gust (km/h)','Spd of Max Gust Flag'], axis = 1, inplace = True)


# In[ ]:


ontario_climate.columns = ['Date','Year','Month','Day','max_temp_C','min_temp_C','mean_temp_C','heat_deg_days','cool_deg_days','total_rain_mm','total_snow_cm','total_precip_mm','snow_on_grnd_cm']


# In[ ]:


ontario_climate['Date'] = ontario_climate['Date'].apply(pd.to_datetime)


# In[ ]:


### Keeping Climate data from July 15 to June 16

ontario_climate=ontario_climate[(ontario_climate['Month'] >= 7) & (ontario_climate['Year'] == 2015) 
                               |(ontario_climate['Month'] <= 6) & (ontario_climate['Year'] == 2016)]


# In[ ]:


### Mapping week in the Climate data
ontario_climate['week'] = ontario_climate.Date.dt.week

### Jan 2016 should be week 54 ,not week 1.
ontario_climate['week'] = np.where((ontario_climate['week'] <= 26) & (ontario_climate['Year'] == 2016), ontario_climate['week']+53, ontario_climate['week'])

ontario_climate = ontario_climate.reset_index()
ontario_climate.drop('index',axis=1,inplace=True)
ontario_climate.head()


# In[ ]:


### Checking for any nan values

round((ontario_climate.isnull().sum()/len(ontario_climate.index))*100,2)


# In[ ]:


### Replacing Nan with mean value
ontario_climate['max_temp_C'] = ontario_climate['max_temp_C'].fillna(ontario_climate['max_temp_C'].mean())
ontario_climate['min_temp_C'] = ontario_climate['min_temp_C'].fillna(ontario_climate['min_temp_C'].mean())
ontario_climate['mean_temp_C'] = ontario_climate['mean_temp_C'].fillna(ontario_climate['mean_temp_C'].mean())
ontario_climate['heat_deg_days'] = ontario_climate['heat_deg_days'].fillna(ontario_climate['heat_deg_days'].mean())
ontario_climate['cool_deg_days'] = ontario_climate['cool_deg_days'].fillna(ontario_climate['cool_deg_days'].mean())
ontario_climate['total_rain_mm'] = ontario_climate['total_rain_mm'].fillna(ontario_climate['total_rain_mm'].mean())
ontario_climate['total_snow_cm'] = ontario_climate['total_snow_cm'].fillna(ontario_climate['total_snow_cm'].mean())
ontario_climate['total_precip_mm'] = ontario_climate['total_precip_mm'].fillna(ontario_climate['total_precip_mm'].mean())
ontario_climate['snow_on_grnd_cm'] = ontario_climate['snow_on_grnd_cm'].fillna(ontario_climate['snow_on_grnd_cm'].mean())


# In[ ]:


ontario_climate.head()


# ### Other KPI

# #### Net Promoters Score & Stock_Index

# In[ ]:


nps_score = pd.read_excel("../input/eleckartdata/Media data and other information.xlsx", sheet_name='Monthly NPS Score', skiprows=1)


# In[ ]:


### Transforming NPS and Stock_index
nps_score = nps_score.T.reset_index(drop=True)
nps_score.columns = ['NPS','Stock_Index']
nps_score = nps_score.drop(nps_score.index[[0]]).reset_index(drop=True)


# In[ ]:


### Adding Month and Year
nps_score['Month'] = pd.Series([7,8,9,10,11,12,1,2,3,4,5,6])
nps_score['Year'] = pd.Series([2015,2015,2015,2015,2015,2015,2016,2016,2016,2016,2016,2016])


# In[ ]:


nps_score['NPS'] = nps_score['NPS'].astype(float)
nps_score['Stock_Index'] = nps_score['Stock_Index'].astype(float)


# In[ ]:


nps_score.head()


# ### Mapping KPI on Calendar

# In[ ]:


calendar = calendar.merge(ontario_climate, 'left')


# In[ ]:


calendar = calendar.merge(nps_score, 'left')


# In[ ]:


# We can create the week number
calendar['week'] = np.where(calendar.Date.dt.year == 2015, (calendar.Date.dt.week - pd.to_datetime('2015-07-01').week + 1), calendar.Date.dt.week+27)

# Dates like 2016-01-01 will be 53rd week as per ISO standard, hence the week value would be 53+27=80.
# We can make those values as week 27
calendar.week.values[(calendar.Date.dt.year == 2016) & (calendar.week == 80)] = 27


# In[ ]:


calendar.head()


# In[ ]:


calendar = pd.DataFrame(calendar.groupby('week').agg({'NPS':'mean','Stock_Index':'mean',
                                                             'Special_sales':'mean','Payday':'mean',
                                                             'max_temp_C':'mean','min_temp_C':'mean',
                                                             'mean_temp_C':'mean','heat_deg_days':'mean',
                                                             'cool_deg_days':'mean','total_rain_mm':'mean',
                                                             'total_snow_cm':'mean','total_precip_mm':'mean',
                                                             'snow_on_grnd_cm':'mean'}))


# In[ ]:


calendar.reset_index(inplace = True)


# In[ ]:


calendar.head()


# ### Advertisement Related KPI

# #### Marketing 

# In[ ]:


### Marketing Investment Data
marketing = pd.read_excel("../input/eleckartdata/Media data and other information.xlsx", sheet_name='Media Investment', skipfooter = 4, skiprows=2)


# In[ ]:


marketing.drop('Unnamed: 0', axis = 1, inplace = True)
marketing.replace(np.nan,0,inplace = True)
marketing['Date'] = pd.to_datetime(marketing[['Year', 'Month']].assign(DAY=1))
marketing.set_index('Date', inplace = True)
marketing


# In[ ]:


### Renaming the columns

marketing.columns = ['Year','Month','Total_Investment','TV','Digital','Sponsorship','Content_marketing',
                     'Online_marketing','Affiliates','SEM','Radio','Other']


# In[ ]:


### convert to datetimeindex
marketing.index = pd.to_datetime(marketing.index)


# In[ ]:


marketing


# In[ ]:


### add new next month for correct resample
idx = marketing.index[-1] + pd.offsets.MonthBegin(1)
idx


# In[ ]:


marketing = marketing.append(marketing.iloc[[-1]].rename({marketing.index[-1]: idx}))
marketing


# In[ ]:


#Resampling the data on weekly frequency
marketing = marketing.resample('W').ffill().iloc[:-1]
marketing


# In[ ]:


### divide by size of months
marketing['Total_Investment'] /= marketing.resample('MS')['Total_Investment'].transform('size')
marketing['TV'] /= marketing.resample('MS')['TV'].transform('size')
marketing['Digital'] /= marketing.resample('MS')['Digital'].transform('size')
marketing['Sponsorship'] /= marketing.resample('MS')['Sponsorship'].transform('size')
marketing['Content_marketing'] /= marketing.resample('MS')['Content_marketing'].transform('size')
marketing['Online_marketing'] /= marketing.resample('MS')['Online_marketing'].transform('size')
marketing['Affiliates'] /= marketing.resample('MS')['Affiliates'].transform('size')
marketing['SEM'] /= marketing.resample('MS')['SEM'].transform('size')
marketing['Radio'] /= marketing.resample('MS')['Radio'].transform('size')
marketing['Other'] /= marketing.resample('MS')['Other'].transform('size')


# In[ ]:


marketing.head()


# In[ ]:


marketing.reset_index(inplace = True)

###  Mapping week in the marketing

marketing['Date'] = pd.to_datetime(marketing['Date'])
# We can create the week number
marketing['week'] = np.where(marketing.Date.dt.year == 2015, (marketing.Date.dt.week - pd.to_datetime('2015-07-01').week + 1), marketing.Date.dt.week+27)

marketing.week.values[(marketing.Date.dt.year == 2016) & (marketing.week == 80)] = 27
marketing.sort_values('week', inplace = True)


# In[ ]:


marketing.head()


# #### Adstock

# In[ ]:


def adstocked_advertising(adstock_rate=0.5, advertising = marketing):
    
    adstocked_advertising = []
    for i in range(len(advertising)):
        if i == 0: 
            adstocked_advertising.append(advertising.iloc[i])
        else:
            adstocked_advertising.append(advertising.iloc[i] + adstock_rate * advertising.iloc[i-1])            
    return adstocked_advertising
   


# In[ ]:


adstock = pd.DataFrame()


# In[ ]:


adstock['TV_ads'] = adstocked_advertising(adstock_rate=0.5, advertising = marketing['TV'])

adstock['Digital_ads'] = adstocked_advertising(adstock_rate=0.5, advertising = marketing['Digital'])

adstock['Sponsorship_ads'] = adstocked_advertising(adstock_rate=0.5, advertising = marketing['Sponsorship'])

adstock['Content_marketing_ads'] = adstocked_advertising(adstock_rate=0.5, advertising = marketing['Content_marketing'])

adstock['Online_marketing_ads'] = adstocked_advertising(adstock_rate=0.5, advertising = marketing['Online_marketing'])

adstock['Affiliates_ads'] = adstocked_advertising(adstock_rate=0.5, advertising = marketing['Affiliates'])

adstock['SEM_ads'] = adstocked_advertising(adstock_rate=0.5, advertising = marketing['SEM'])

adstock['Radio_ads'] = adstocked_advertising(adstock_rate=0.5, advertising = marketing['Radio'])

adstock['Other_ads'] = adstocked_advertising(adstock_rate=0.5, advertising = marketing['Other'])


# In[ ]:


adstock.head()


# #### Mapping marketing and adstock

# In[ ]:


marketing = pd.concat([marketing,adstock] ,axis=1)


# In[ ]:


marketing.head()


# ### Product premium-ness

# In[ ]:


# The premium-ness of the product depends on the MRP. Higher the MRP, more premium is the product.
# Let's check the percentiles of MRP in the dataset.

consumer.product_mrp.describe(percentiles=[0.25,0.5,0.75,0.8,0.9,0.95,0.99])


# In[ ]:


# Let's assume that products with MRP greater than 90 percentile to be premium products.
# Create a dataframe with mrp, number of units sold and gmv against each product vertical to analyse better.
prod_cat = pd.DataFrame(pd.pivot_table(consumer, values = ['units','product_mrp', 'gmv'], index = ['product_analytic_vertical'], 
               aggfunc={'units':np.sum, 'product_mrp':np.mean, 'gmv':np.sum}).to_records())


# In[ ]:


# Marking products with MRP greater than 90th percentile with 1 and rest with 0
prod_cat['premium_product'] = np.where((prod_cat.product_mrp>consumer.product_mrp.quantile(0.9)),1,0)


# In[ ]:


prod_cat.loc[prod_cat.premium_product==1]


# - Clearly, Teleconverter, SoundMixer, SlingBox, MotionController, KaraokePlayer, DJController are premium products. All other products are mass products.
# - Let's visualise how the premium products contribute towards the GMV.

# In[ ]:


plt.figure(figsize=(15,5))
sns.barplot(x = prod_cat.product_analytic_vertical, y=prod_cat.gmv, hue=prod_cat.premium_product)
plt.xticks(rotation=90)
plt.show()


# - We can clearly see that, maximum revenue is generated through mass producs like HomeAudioSpeaker, Lens, GamingPad, etc and not premium products that contibute quite less towards revenue.
# - The company hence should focus more on mass products than premium products.

# In[ ]:


consumer = consumer.merge(prod_cat[['product_analytic_vertical', 'premium_product']] , left_on='product_analytic_vertical', 
            right_on='product_analytic_vertical',
                   how = 'inner')


# In[ ]:


sales = consumer.copy()


# In[ ]:


consumer.drop(['product_analytic_vertical'],1,inplace=True)


# In[ ]:


consumer.head()


# # 4. Aggregation

# ## 1. Camera Sub Category

# In[ ]:


camera_df = consumer[consumer['product_analytic_sub_category'] == 'CameraAccessory']


# In[ ]:


###  Removing outliers is important as
###  1. There may be some garbage value.
###  2. Bulk orders can skew the analysis


# In[ ]:


### Outlier Analysis
fig, axs = plt.subplots(1,3, figsize = (20,4))
plt1 = sns.boxplot(camera_df['gmv'], ax = axs[0])
plt2 = sns.boxplot(camera_df['units'], ax = axs[2])
plt4 = sns.boxplot(camera_df['product_mrp'], ax = axs[1])
plt.tight_layout()


# In[ ]:


### Treating outliers
### Outlier treatment for gmv & product_mrp
Q1 = camera_df.gmv.quantile(0.25)
Q3 = camera_df.gmv.quantile(0.75)
IQR = Q3 - Q1
camera_df = camera_df[(camera_df.gmv >= Q1 - 1.5*IQR) & (camera_df.gmv <= Q3 + 1.5*IQR)]
Q1 = camera_df.product_mrp.quantile(0.25)
Q3 = camera_df.product_mrp.quantile(0.75)
IQR = Q3 - Q1
camera_df = camera_df[(camera_df.product_mrp >= Q1 - 1.5*IQR) & (camera_df.product_mrp <= Q3 + 1.5*IQR)]


# In[ ]:


### Outlier Analysis
fig, axs = plt.subplots(1,3, figsize = (20,4))
plt1 = sns.boxplot(camera_df['gmv'], ax = axs[0])
plt2 = sns.boxplot(camera_df['units'], ax = axs[2])
plt4 = sns.boxplot(camera_df['product_mrp'], ax = axs[1])
plt.tight_layout()


# In[ ]:


camera_df.columns


# In[ ]:


camera_df.head()


# In[ ]:


### Aggregating dataset on weekly level

ca_week = pd.DataFrame(camera_df.groupby('week').agg({'gmv':'sum','listing_price':'mean',
                                                             'product_mrp':'mean','discount':'mean',
                                                             'sla':'mean','product_procurement_sla':'mean',
                                                             'fsn_id':pd.Series.nunique,'order_item_id':pd.Series.nunique,
                                                             'order_id': pd.Series.nunique,
                                                             'order_payment_type':'sum',
                                                            'premium_product':'sum'}))

ca_week.reset_index( inplace = True)


# In[ ]:


ca_week.head()


# In[ ]:


### Sum of GMV / No of unique Orders

ca_week['AOV'] = ca_week['gmv']/ca_week['order_id']


# In[ ]:


ca_week['online_order_perc'] = ca_week['order_payment_type']*100/ca_week['order_item_id']


# In[ ]:


ca_week.week.unique()


# In[ ]:


calendar.week.unique()


# In[ ]:


ca_week['week'] = ca_week['week'].astype(int)
calendar['week'] = calendar['week'].astype(int)


# In[ ]:


ca_week = ca_week.merge(marketing, how = 'left', on = 'week')


# In[ ]:


ca_week = ca_week.merge(calendar, how = 'left', on = 'week')


# In[ ]:


ca_week.head()


# ### Exploratory Data Analysis

# In[ ]:


ca_week_viz = ca_week.round(2)


# #### Univariate Analysis

# Target Variable

# In[ ]:


sns.distplot(ca_week_viz['gmv'],kde=True)


# In[ ]:


plt.figure(figsize=(15, 5))
sns.barplot(ca_week_viz['week'],ca_week_viz['gmv'])


# Marketing KPI

# In[ ]:


ca_week_viz.columns


# In[ ]:


fig, axs = plt.subplots(2,4,figsize=(16,8))

plt1 = sns.scatterplot(x = 'Total_Investment', y = 'gmv', data = ca_week_viz, ax = axs[0,0])

plt2 = sns.scatterplot(x = 'TV', y = 'gmv', data = ca_week_viz, ax = axs[0,1])

plt3 = sns.scatterplot(x = 'Digital', y = 'gmv', data = ca_week_viz, ax = axs[0,2])

plt4 = sns.scatterplot(x = 'Sponsorship', y = 'gmv', data = ca_week_viz, ax = axs[0,3])

plt5 = sns.scatterplot(x = 'Content_marketing', y = 'gmv', data = ca_week_viz, ax = axs[1,0])

plt6 = sns.scatterplot(x = 'Online_marketing', y = 'gmv', data = ca_week_viz, ax = axs[1,1])

plt7 = sns.scatterplot(x = 'Affiliates', y = 'gmv', data = ca_week_viz, ax = axs[1,2])

plt8 = sns.scatterplot(x = 'SEM', y = 'gmv', data = ca_week_viz, ax = axs[1,3])

plt.tight_layout()


# #### Bivariate Analysis

# GMV and Holiday weekly

# In[ ]:


plt.figure(figsize=(20, 5))
sns.barplot(x= ca_week_viz['week'], y =ca_week_viz['gmv'], hue = ca_week_viz['Special_sales'], dodge = False)
plt.show()


# GMV and Dicount weekly

# In[ ]:


plt.figure(figsize=(20, 5))
sns.barplot(x= ca_week_viz['week'], y =ca_week_viz['gmv'], hue = pd.cut(ca_week_viz['discount'],3), dodge = False)
plt.show()


# #### Camera Accessory - Moving Average

# In[ ]:


### ca_week

### Moving Average for listing_price and discount

### ca_week = ca_week.sort_values('order_date')

ca_week[['MA2_LP','MA2_Discount']] = ca_week[['listing_price','discount']].rolling(window=2,min_periods=1).mean()
ca_week[['MA3_LP','MA3_Discount']] = ca_week[['listing_price','discount']].rolling(window=3,min_periods=1).mean()
ca_week[['MA4_LP','MA4_Discount']] = ca_week[['listing_price','discount']].rolling(window=4,min_periods=1).mean()

### Reference listed price Inflation 

ca_week['MA2_listed_price'] = (ca_week['listing_price']-ca_week['MA2_LP'])/ca_week['MA2_LP']
ca_week['MA3_listed_price'] = (ca_week['listing_price']-ca_week['MA3_LP'])/ca_week['MA3_LP']
ca_week['MA4_listed_price'] = (ca_week['listing_price']-ca_week['MA4_LP'])/ca_week['MA4_LP']

### Reference discount Inflation

ca_week['MA2_discount_offer'] = (ca_week['discount']-ca_week['MA2_Discount'])/ca_week['MA2_Discount']
ca_week['MA3_discount_offer'] = (ca_week['discount']-ca_week['MA3_Discount'])/ca_week['MA3_Discount']
ca_week['MA4_discount_offer'] = (ca_week['discount']-ca_week['MA4_Discount'])/ca_week['MA4_Discount']


ca_week.drop(['MA2_LP','MA3_LP','MA4_LP','MA2_Discount','MA3_Discount','MA4_Discount'], axis = 1, inplace = True)  
ca_week.head()


# #### Camera Accessory - Data Profiling to see multicollinearity and variable distributions

# In[ ]:


# ### To identify multicollinearity between variable
plt.figure(figsize=(20,20))
sns.heatmap(ca_week.corr(),annot = True, cmap="YlGnBu")
plt.show()


# In[ ]:


### Highly Correlated Columns should be dropped

ca_week.drop(['TV', 'Digital', 'Sponsorship', 'Content_marketing','Online_marketing', 'Affiliates', 'SEM','Radio',
              'Other'], axis = 1, inplace = True)


# In[ ]:


plt.figure(figsize=(25,20))
sns.heatmap(ca_week.corr(), cmap="coolwarm", annot=True)
plt.show()


# In[ ]:


ca_week.drop(['Affiliates_ads','SEM_ads','Digital_ads','Radio_ads','Other_ads','mean_temp_C','min_temp_C',
              'order_id','order_item_id','total_precip_mm','Total_Investment','MA3_discount_offer',
               'MA3_listed_price','AOV','max_temp_C','MA2_listed_price','MA4_discount_offer'],1,inplace=True)


# In[ ]:


#Successfully removed more than 90% correlation


# #### Camera Accessory -  Lag Variable Functions

# In[ ]:


### Lag of listed_price, discount_offer, NPS, Special_sales

ca_week['lag_1_listed_price'] = ca_week['listing_price'].shift(-1).fillna(0)
ca_week['lag_2_listed_price'] = ca_week['listing_price'].shift(-2).fillna(0)
ca_week['lag_3_listed_price'] = ca_week['listing_price'].shift(-3).fillna(0)

ca_week['lag_1_discount'] = ca_week['discount'].shift(-1).fillna(0)
ca_week['lag_2_discount'] = ca_week['discount'].shift(-2).fillna(0)
ca_week['lag_3_discount'] = ca_week['discount'].shift(-3).fillna(0)

ca_week['lag_1_Stock_Index'] = ca_week['Stock_Index'].shift(-1).fillna(0)
ca_week['lag_2_Stock_Index'] = ca_week['Stock_Index'].shift(-2).fillna(0)
ca_week['lag_3_Stock_Index'] = ca_week['Stock_Index'].shift(-3).fillna(0)

ca_week['lag_1_Special_sales'] = ca_week['Special_sales'].shift(-1).fillna(0)
ca_week['lag_2_Special_sales'] = ca_week['Special_sales'].shift(-2).fillna(0)
ca_week['lag_3_Special_sales'] = ca_week['Special_sales'].shift(-3).fillna(0)

ca_week['lag_1_Payday'] = ca_week['Payday'].shift(-1).fillna(0)
ca_week['lag_2_Payday'] = ca_week['Payday'].shift(-2).fillna(0)
ca_week['lag_3_Payday'] = ca_week['Payday'].shift(-3).fillna(0)

ca_week['lag_1_NPS'] = ca_week['NPS'].shift(-1).fillna(0)
ca_week['lag_2_NPS'] = ca_week['NPS'].shift(-2).fillna(0)
ca_week['lag_3_NPS'] = ca_week['NPS'].shift(-3).fillna(0)


# In[ ]:


ca_week.head()


# ## 2. Gaming Sub Category

# In[ ]:


gaming_accessory = consumer[consumer['product_analytic_sub_category'] == 'GamingAccessory']


# In[ ]:


###  Removing outliers is important as
###  1. There may be some garbage value.
###  2. Bulk orders can skew the analysis


# In[ ]:


### Outlier Analysis
fig, axs = plt.subplots(1,3, figsize = (20,4))
plt1 = sns.boxplot(gaming_accessory['gmv'], ax = axs[0])
plt2 = sns.boxplot(gaming_accessory['units'], ax = axs[2])
plt4 = sns.boxplot(gaming_accessory['product_mrp'], ax = axs[1])
plt.tight_layout()


# In[ ]:


### Treating outliers
### Outlier treatment for gmv & product_mrp
Q1 = gaming_accessory.gmv.quantile(0.25)
Q3 = gaming_accessory.gmv.quantile(0.75)
IQR = Q3 - Q1
gaming_accessory = gaming_accessory[(gaming_accessory.gmv >= Q1 - 1.5*IQR) & (gaming_accessory.gmv <= Q3 + 1.5*IQR)]
Q1 = gaming_accessory.product_mrp.quantile(0.25)
Q3 = gaming_accessory.product_mrp.quantile(0.75)
IQR = Q3 - Q1
gaming_accessory = gaming_accessory[(gaming_accessory.product_mrp >= Q1 - 1.5*IQR) & (gaming_accessory.product_mrp <= Q3 + 1.5*IQR)]


# In[ ]:


### Outlier Analysis
fig, axs = plt.subplots(1,3, figsize = (20,4))
plt1 = sns.boxplot(gaming_accessory['gmv'], ax = axs[0])
plt2 = sns.boxplot(gaming_accessory['units'], ax = axs[2])
plt4 = sns.boxplot(gaming_accessory['product_mrp'], ax = axs[1])
plt.tight_layout()


# In[ ]:


gaming_accessory.columns


# In[ ]:


### Aggregating dataset on weekly level

ga_week = pd.DataFrame(gaming_accessory.groupby('week').agg({'gmv':'sum','listing_price':'mean',
                                                             'product_mrp':'mean','discount':'mean',
                                                             'sla':'mean','product_procurement_sla':'mean',
                                                             'fsn_id':pd.Series.nunique,'order_item_id':pd.Series.nunique,
                                                             'order_id': pd.Series.nunique,
                                                             'order_payment_type':'sum'}))

ga_week.reset_index( inplace = True)


# In[ ]:


ga_week.head()


# In[ ]:


### Sum of GMV / No of unique Orders

ga_week['AOV'] = ga_week['gmv']/ga_week['order_id']


# In[ ]:


ga_week['online_order_perc'] = ga_week['order_payment_type']*100/ga_week['order_item_id']


# In[ ]:


ga_week.head()


# In[ ]:


ga_week = ga_week.merge(marketing, how = 'left', on = 'week')


# In[ ]:


ga_week = ga_week.merge(calendar, how = 'left', on = 'week')


# In[ ]:


ga_week.head()


# ### Exploratory Data Analysis

# In[ ]:


ga_week_viz = ga_week.round(2)


# #### Univariate Analysis

# Target Variable

# In[ ]:


sns.distplot(ga_week_viz['gmv'],kde=True)


# In[ ]:


plt.figure(figsize=(15, 5))
sns.barplot(ga_week_viz['week'],ga_week_viz['gmv'])


# Marketing KPI

# In[ ]:


ga_week_viz.columns


# In[ ]:


fig, axs = plt.subplots(2,4,figsize=(16,8))

plt1 = sns.scatterplot(x = 'Total_Investment', y = 'gmv', data = ga_week_viz, ax = axs[0,0])

plt2 = sns.scatterplot(x = 'TV', y = 'gmv', data = ga_week_viz, ax = axs[0,1])

plt3 = sns.scatterplot(x = 'Digital', y = 'gmv', data = ga_week_viz, ax = axs[0,2])

plt4 = sns.scatterplot(x = 'Sponsorship', y = 'gmv', data = ga_week_viz, ax = axs[0,3])

plt5 = sns.scatterplot(x = 'Content_marketing', y = 'gmv', data = ga_week_viz, ax = axs[1,0])

plt6 = sns.scatterplot(x = 'Online_marketing', y = 'gmv', data = ga_week_viz, ax = axs[1,1])

plt7 = sns.scatterplot(x = 'Affiliates', y = 'gmv', data = ga_week_viz, ax = axs[1,2])

plt8 = sns.scatterplot(x = 'SEM', y = 'gmv', data = ga_week_viz, ax = axs[1,3])

plt.tight_layout()


# #### Bivariate Analysis

# GMV and Holiday weekly

# In[ ]:


plt.figure(figsize=(20, 5))
sns.barplot(x= ga_week_viz['week'], y =ga_week_viz['gmv'], hue = ga_week_viz['Special_sales'], dodge = False)
plt.show()


# GMV and Holiday weekly

# In[ ]:


plt.figure(figsize=(20, 5))
sns.barplot(x= ga_week_viz['week'], y =ga_week_viz['gmv'], hue = pd.cut(ga_week_viz['discount'],3), dodge = False)
plt.show()


# #### Gaming Accessory - Moving Average

# In[ ]:


### ga_week

### Moving Average for listed_price and discount_offer

### ga_week = ga_week.sort_values('order_date')

ga_week[['MA2_LP','MA2_Discount']] = ga_week[['listing_price','discount']].rolling(window=2,min_periods=1).mean()
ga_week[['MA3_LP','MA3_Discount']] = ga_week[['listing_price','discount']].rolling(window=3,min_periods=1).mean()
ga_week[['MA4_LP','MA4_Discount']] = ga_week[['listing_price','discount']].rolling(window=4,min_periods=1).mean()

### Reference listed price Inflation 

ga_week['MA2_listed_price'] = (ga_week['listing_price']-ga_week['MA2_LP'])/ga_week['MA2_LP']
ga_week['MA3_listed_price'] = (ga_week['listing_price']-ga_week['MA3_LP'])/ga_week['MA3_LP']
ga_week['MA4_listed_price'] = (ga_week['listing_price']-ga_week['MA4_LP'])/ga_week['MA4_LP']

### Reference discount Inflation

ga_week['MA2_discount'] = (ga_week['discount']-ga_week['MA2_Discount'])/ga_week['MA2_Discount']
ga_week['MA3_discount'] = (ga_week['discount']-ga_week['MA3_Discount'])/ga_week['MA3_Discount']
ga_week['MA4_discount'] = (ga_week['discount']-ga_week['MA4_Discount'])/ga_week['MA4_Discount']


ga_week.drop(['MA2_LP','MA3_LP','MA4_LP','MA2_Discount','MA3_Discount','MA4_Discount'], axis = 1, inplace = True)  
ga_week


# #### Heatmap to see multicollinearity

# In[ ]:


plt.figure(figsize=(25,20))

### Heatmap
sns.heatmap(ga_week.corr(), cmap="coolwarm", annot=True)
plt.show()


# In[ ]:



ga_week.drop(['TV', 'Digital', 'Sponsorship', 'Content_marketing','Online_marketing', 'Affiliates', 'SEM','Radio',
              'Other','Affiliates_ads','SEM_ads','Digital_ads','Radio_ads','Other_ads','mean_temp_C','min_temp_C',
              'order_id','order_item_id','total_precip_mm','Total_Investment','MA3_discount',
              'MA3_listed_price','AOV','MA4_listed_price'], axis = 1, inplace = True)


# In[ ]:


ga_week.drop(['max_temp_C'], axis = 1, inplace = True)


# In[ ]:


###  Successfully removed more than 90% highly correlated variables from dataset.


# #### Gaming Accessory - Lag Variable Functions

# In[ ]:


### Lag of listed_price, discount_offer, NPS, Special_sales

ga_week['lag_1_listed_price'] = ga_week['listing_price'].shift(-1).fillna(0)
ga_week['lag_2_listed_price'] = ga_week['listing_price'].shift(-2).fillna(0)
ga_week['lag_3_listed_price'] = ga_week['listing_price'].shift(-3).fillna(0)

ga_week['lag_1_discount_offer'] = ga_week['discount'].shift(-1).fillna(0)
ga_week['lag_2_discount_offer'] = ga_week['discount'].shift(-2).fillna(0)
ga_week['lag_3_discount_offer'] = ga_week['discount'].shift(-3).fillna(0)

ga_week['lag_1_NPS'] = ga_week['NPS'].shift(-1).fillna(0)
ga_week['lag_2_NPS'] = ga_week['NPS'].shift(-2).fillna(0)
ga_week['lag_3_NPS'] = ga_week['NPS'].shift(-3).fillna(0)

ga_week['lag_1_Stock_Index'] = ga_week['Stock_Index'].shift(-1).fillna(0)
ga_week['lag_2_Stock_Index'] = ga_week['Stock_Index'].shift(-2).fillna(0)
ga_week['lag_3_Stock_Index'] = ga_week['Stock_Index'].shift(-3).fillna(0)

ga_week['lag_1_Special_sales'] = ga_week['Special_sales'].shift(-1).fillna(0)
ga_week['lag_2_Special_sales'] = ga_week['Special_sales'].shift(-2).fillna(0)
ga_week['lag_3_Special_sales'] = ga_week['Special_sales'].shift(-3).fillna(0)

ga_week['lag_1_Payday'] = ga_week['Payday'].shift(-1).fillna(0)
ga_week['lag_2_Payday'] = ga_week['Payday'].shift(-2).fillna(0)
ga_week['lag_3_Payday'] = ga_week['Payday'].shift(-3).fillna(0)


# In[ ]:


ga_week.head()


# ## 3. Home Audio Sub Category

# In[ ]:


home_audio = consumer[consumer['product_analytic_sub_category'] == 'HomeAudio']


# In[ ]:


###  Removing outliers is important as
###  1. There may be some garbage value.
###  2. Bulk orders can skew the analysis


# In[ ]:


### Outlier Analysis
fig, axs = plt.subplots(1,3, figsize = (20,4))
plt1 = sns.boxplot(home_audio['gmv'], ax = axs[0])
plt2 = sns.boxplot(home_audio['units'], ax = axs[2])
plt4 = sns.boxplot(home_audio['product_mrp'], ax = axs[1])
plt.tight_layout()


# In[ ]:


### Treating outliers
### Outlier treatment for gmv & product_mrp
Q1 = home_audio.gmv.quantile(0.25)
Q3 = home_audio.gmv.quantile(0.75)
IQR = Q3 - Q1
home_audio = home_audio[(home_audio.gmv >= Q1 - 1.5*IQR) & (home_audio.gmv <= Q3 + 1.5*IQR)]
Q1 = home_audio.product_mrp.quantile(0.25)
Q3 = home_audio.product_mrp.quantile(0.75)
IQR = Q3 - Q1
home_audio = home_audio[(home_audio.product_mrp >= Q1 - 1.5*IQR) & (home_audio.product_mrp <= Q3 + 1.5*IQR)]


# In[ ]:


### Outlier Analysis
fig, axs = plt.subplots(1,3, figsize = (20,4))
plt1 = sns.boxplot(home_audio['gmv'], ax = axs[0])
plt2 = sns.boxplot(home_audio['units'], ax = axs[2])
plt4 = sns.boxplot(home_audio['product_mrp'], ax = axs[1])
plt.tight_layout()


# In[ ]:


home_audio.columns


# In[ ]:


### Aggregating dataset on weekly level

ha_week = pd.DataFrame(home_audio.groupby('week').agg({'gmv':'sum','listing_price':'mean',
                                                             'product_mrp':'mean','discount':'mean',
                                                             'sla':'mean','product_procurement_sla':'mean',
                                                             'fsn_id':pd.Series.nunique,'order_item_id':pd.Series.nunique,
                                                             'order_id': pd.Series.nunique,
                                                             'order_payment_type':'sum'}))

ha_week.reset_index( inplace = True)


# In[ ]:


ha_week.head()


# In[ ]:


### Sum of GMV / No of unique Orders

ha_week['AOV'] = ha_week['gmv']/ha_week['order_id']


# In[ ]:


ha_week['online_order_perc'] = ha_week['order_payment_type']*100/ha_week['order_item_id']


# In[ ]:


ha_week.head()


# In[ ]:


ha_week = ha_week.merge(marketing, how = 'left', on = 'week')


# In[ ]:


ha_week = ha_week.merge(calendar, how = 'left', on = 'week')


# In[ ]:


ha_week.head()


# ### Exploratory Data Analysis

# In[ ]:


ha_week_viz = ha_week.round(2)


# #### Univariate Analysis

# Target Variable

# In[ ]:


sns.distplot(ha_week_viz['gmv'],kde=True)


# In[ ]:


plt.figure(figsize=(15, 5))
sns.barplot(ha_week_viz['week'],ha_week_viz['gmv'])


# Marketing KPI

# In[ ]:


ha_week_viz.columns


# In[ ]:


fig, axs = plt.subplots(2,4,figsize=(16,8))

plt1 = sns.scatterplot(x = 'Total_Investment', y = 'gmv', data = ha_week_viz, ax = axs[0,0])

plt2 = sns.scatterplot(x = 'TV', y = 'gmv', data = ha_week_viz, ax = axs[0,1])

plt3 = sns.scatterplot(x = 'Digital', y = 'gmv', data = ha_week_viz, ax = axs[0,2])

plt4 = sns.scatterplot(x = 'Sponsorship', y = 'gmv', data = ha_week_viz, ax = axs[0,3])

plt5 = sns.scatterplot(x = 'Content_marketing', y = 'gmv', data = ha_week_viz, ax = axs[1,0])

plt6 = sns.scatterplot(x = 'Online_marketing', y = 'gmv', data = ha_week_viz, ax = axs[1,1])

plt7 = sns.scatterplot(x = 'Affiliates', y = 'gmv', data = ha_week_viz, ax = axs[1,2])

plt8 = sns.scatterplot(x = 'SEM', y = 'gmv', data = ha_week_viz, ax = axs[1,3])

plt.tight_layout()


# #### Bivariate Analysis

# GMV and Holiday weekly

# In[ ]:


plt.figure(figsize=(20, 5))
sns.barplot(x= ha_week_viz['week'], y =ha_week_viz['gmv'], hue = ha_week_viz['Special_sales'], dodge = False)
plt.show()


# GMV and discount weekly

# In[ ]:


plt.figure(figsize=(20, 5))
sns.barplot(x= ha_week_viz['week'], y =ha_week_viz['gmv'], hue = pd.cut(ha_week_viz['discount'],3), dodge = False)
plt.show()


# #### Home Audio - Moving Average

# In[ ]:


### ha_week

### Moving Average for listed_price and discount_offer

### ha_week = ha_week.sort_values('order_date')

ha_week[['MA2_LP','MA2_Discount']] = ha_week[['listing_price','discount']].rolling(window=2,min_periods=1).mean()
ha_week[['MA3_LP','MA3_Discount']] = ha_week[['listing_price','discount']].rolling(window=3,min_periods=1).mean()
ha_week[['MA4_LP','MA4_Discount']] = ha_week[['listing_price','discount']].rolling(window=4,min_periods=1).mean()

### Reference listed price Inflation 

ha_week['MA2_listed_price'] = (ha_week['listing_price']-ha_week['MA2_LP'])/ha_week['MA2_LP']
ha_week['MA3_listed_price'] = (ha_week['listing_price']-ha_week['MA3_LP'])/ha_week['MA3_LP']
ha_week['MA4_listed_price'] = (ha_week['listing_price']-ha_week['MA4_LP'])/ha_week['MA4_LP']

### Reference discount Inflation

ha_week['MA2_discount'] = (ha_week['discount']-ha_week['MA2_Discount'])/ha_week['MA2_Discount']
ha_week['MA3_discount'] = (ha_week['discount']-ha_week['MA3_Discount'])/ha_week['MA3_Discount']
ha_week['MA4_discount'] = (ha_week['discount']-ha_week['MA4_Discount'])/ha_week['MA4_Discount']


ha_week.drop(['MA2_LP','MA3_LP','MA4_LP','MA2_Discount','MA3_Discount','MA4_Discount'], axis = 1, inplace = True)  
ha_week


# #### Heatmap to see multicollinearity

# In[ ]:


plt.figure(figsize=(25,20))

### Heatmap
sns.heatmap(ha_week.corr(), cmap="coolwarm", annot=True)
plt.show()


# In[ ]:


ha_week.drop(['TV', 'Digital', 'Sponsorship', 'Content_marketing','Online_marketing', 'Affiliates', 'SEM','Radio',
              'Other','Affiliates_ads','SEM_ads','Digital_ads','Radio_ads','Other_ads','mean_temp_C','min_temp_C',
              'order_id','order_item_id','total_precip_mm','Total_Investment','MA3_discount',
              'MA3_listed_price','AOV'], axis = 1, inplace = True)


# In[ ]:


ha_week.drop(['max_temp_C'], axis = 1, inplace = True)


# In[ ]:


###  Successfully removed more than 90% highly correlated variables from dataset.


# #### Home Audio - Lag Variable Functions

# In[ ]:


### Lag of listed_price, discount_offer, NPS, Special_sales

ha_week['lag_1_listed_price'] = ha_week['listing_price'].shift(-1).fillna(0)
ha_week['lag_2_listed_price'] = ha_week['listing_price'].shift(-2).fillna(0)
ha_week['lag_3_listed_price'] = ha_week['listing_price'].shift(-3).fillna(0)

ha_week['lag_1_discount_offer'] = ha_week['discount'].shift(-1).fillna(0)
ha_week['lag_2_discount_offer'] = ha_week['discount'].shift(-2).fillna(0)
ha_week['lag_3_discount_offer'] = ha_week['discount'].shift(-3).fillna(0)

ha_week['lag_1_NPS'] = ha_week['NPS'].shift(-1).fillna(0)
ha_week['lag_2_NPS'] = ha_week['NPS'].shift(-2).fillna(0)
ha_week['lag_3_NPS'] = ha_week['NPS'].shift(-3).fillna(0)

ha_week['lag_1_Stock_Index'] = ha_week['Stock_Index'].shift(-1).fillna(0)
ha_week['lag_2_Stock_Index'] = ha_week['Stock_Index'].shift(-2).fillna(0)
ha_week['lag_3_Stock_Index'] = ha_week['Stock_Index'].shift(-3).fillna(0)

ha_week['lag_1_Special_sales'] = ha_week['Special_sales'].shift(-1).fillna(0)
ha_week['lag_2_Special_sales'] = ha_week['Special_sales'].shift(-2).fillna(0)
ha_week['lag_3_Special_sales'] = ha_week['Special_sales'].shift(-3).fillna(0)

ha_week['lag_1_Payday'] = ha_week['Payday'].shift(-1).fillna(0)
ha_week['lag_2_Payday'] = ha_week['Payday'].shift(-2).fillna(0)
ha_week['lag_3_Payday'] = ha_week['Payday'].shift(-3).fillna(0)


# In[ ]:


ha_week.head(10)


# # 5. Modeling - Camera Accessory

# ### 1. Linear Model

# In[ ]:


###  Importing RFE and LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


# In[ ]:


ca_week.columns


# In[ ]:


camera_lm = ca_week[['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',
       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',
       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 
       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA4_listed_price',
       'MA2_discount_offer','premium_product']]
                            
    
camera_lm.head()


# In[ ]:


### Checking NaN

camera_lm.isnull().sum()


# In[ ]:


camera_lm.fillna(0, inplace = True)


# In[ ]:


from sklearn.model_selection import train_test_split


np.random.seed(0)
df_train, df_test = train_test_split(camera_lm, train_size = 0.7, test_size = 0.3, random_state = 100)


# In[ ]:


### Import the StandardScaler()
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

### Create a scaling object
# scaler = StandardScaler()
scaler = MinMaxScaler()


### Create a list of the variables that you need to scale
varlist = ['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',
       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',
       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 
       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA4_listed_price',
       'MA2_discount_offer','premium_product']
                                      

### Scale these variables using 'fit_transform'
df_train[varlist] = scaler.fit_transform(df_train[varlist])


# In[ ]:


df_train.head()


# In[ ]:


### Split the train dataset into X and y
from sklearn.model_selection import train_test_split
X_train = df_train.drop('gmv',axis=1)
y_train = df_train['gmv']


# In[ ]:


#RFE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm 
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[ ]:


lm = LinearRegression()
lm.fit(X_train,y_train)
rfe = RFE(lm, 10)
rfe = rfe.fit(X_train, y_train)


# In[ ]:


list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# In[ ]:


X_train.columns[rfe.support_]


# #### Building model using statsmodel, for the detailed statistics

# In[ ]:


X_train_rfe = X_train[X_train.columns[rfe.support_]]
X_train_rfe.head()


# In[ ]:


def build_model(X,y):
    X = sm.add_constant(X) #Adding the constant
    lm = sm.OLS(y,X).fit() # fitting the model
    print(lm.summary()) # model summary
    return X
    
def checkVIF(X):
    vif = pd.DataFrame()
    vif['Features'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by = "VIF", ascending = False)
    return(vif)


# #### MODEL 1

# In[ ]:


X_train_new = build_model(X_train_rfe,y_train)


# In[ ]:


checkVIF(X_train_new)


# In[ ]:


X_train_new = X_train_rfe.drop(["discount"], axis = 1)


# #### MODEL 2

# In[ ]:


X_train_new = build_model(X_train_new,y_train)


# In[ ]:


checkVIF(X_train_new)


# In[ ]:


X_train_new = X_train_new.drop(["heat_deg_days"], axis = 1)


# #### MODEL 3

# In[ ]:


X_train_new = build_model(X_train_new,y_train)


# In[ ]:


X_train_new = X_train_new.drop(["snow_on_grnd_cm"], axis = 1)


# #### MODEL 4

# In[ ]:


X_train_new = build_model(X_train_new,y_train)


# In[ ]:


checkVIF(X_train_new)


# In[ ]:


X_train_new = X_train_new.drop(["sla"], axis = 1)


# #### MODEL 5

# In[ ]:


X_train_new = build_model(X_train_new,y_train)


# In[ ]:


checkVIF(X_train_new)


# In[ ]:


X_train_new = X_train_new.drop(["MA2_discount_offer"], axis = 1)


# #### MODEL 6

# In[ ]:


X_train_new = build_model(X_train_new,y_train)


# In[ ]:


checkVIF(X_train_new)


# In[ ]:


X_train_new = X_train_new.drop(["product_procurement_sla"], axis = 1)


# #### MODEL 7

# In[ ]:


X_train_new = build_model(X_train_new,y_train)


# In[ ]:


checkVIF(X_train_new)


# In[ ]:


X_train_new = X_train_new.drop(["MA4_listed_price"], axis = 1)


# #### MODEL 8

# In[ ]:


X_train_new = build_model(X_train_new,y_train)


# In[ ]:


checkVIF(X_train_new)


# #### Residual Analysis of Model

# In[ ]:


lm = sm.OLS(y_train,X_train_new).fit()
y_train_price = lm.predict(X_train_new)


# In[ ]:


# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((y_train - y_train_price), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)   


# Error terms seem to be approximately normally distributed, so the assumption on the linear modeling seems to be fulfilled.

# #### Prediction and Evaluation

# In[ ]:


#Scaling the test set
num_vars = ['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',
       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',
       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 
       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA4_listed_price',
       'MA2_discount_offer','premium_product']
df_test[num_vars] = scaler.fit_transform(df_test[num_vars])


# In[ ]:


#Dividing into X and y
y_test = df_test.pop('gmv')
X_test = df_test


# In[ ]:


# Now let's use our model to make predictions.
X_train_new = X_train_new.drop('const',axis=1)
# Creating X_test_new dataframe by dropping variables from X_test
X_test_new = X_test[X_train_new.columns]

# Adding a constant variable 
X_test_new = sm.add_constant(X_test_new)


# In[ ]:


# Making predictions
y_pred = lm.predict(X_test_new)


# #### Evaluation of test via comparison of y_pred and y_test

# In[ ]:


from sklearn.metrics import r2_score 
r2_score(y_test, y_pred)


# In[ ]:


#EVALUATION OF THE MODEL
# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_pred', fontsize=16)   


# #### Evaluation of the model using Statistics

# In[ ]:


print(lm.summary())


# ### 1.1 Linear Model - Stepwise Selection for feature selection

# In[ ]:


###  Importing RFE and LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


# In[ ]:


ca_week.columns


# In[ ]:


camera_lm = ca_week[['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',
       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',
       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 
       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA4_listed_price',
       'MA2_discount_offer']]
                            
    
camera_lm.head()


# In[ ]:


### Checking NaN
camera_lm.isnull().sum()


# In[ ]:


camera_lm.fillna(0, inplace = True)


# In[ ]:


### Import the StandardScaler()
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

### Create a scaling object
# scaler = StandardScaler()
scaler = MinMaxScaler()


### Create a list of the variables that you need to scale
varlist = ['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',
       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',
       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 
       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA4_listed_price',
       'MA2_discount_offer']
                                      

### Scale these variables using 'fit_transform'
camera_lm[varlist] = scaler.fit_transform(camera_lm[varlist])


# In[ ]:


camera_lm.head()


# In[ ]:


### Split the train dataset into X and y
from sklearn.model_selection import train_test_split
x = camera_lm.drop('gmv',axis=1)
y = camera_lm['gmv']

camera_train_lm = camera_lm


# In[ ]:


print("x dataset: ",x.shape)
print("y dataset: ",y.shape)


# In[ ]:


###  Instantiate
lm = LinearRegression()

###  Fit a line
lm.fit(x,y)


# In[ ]:


###  Coefficient values

coef = pd.DataFrame(x.columns)
coef['Coefficient'] = pd.Series(lm.coef_)
coef


# In[ ]:


col = x.columns
col


# #### Model Building - Stepwise selection for feature selection

# In[ ]:


import statsmodels.api as sm  
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[ ]:


def stepwise_selection(x, y,
                       initial_list=[ 'discount', 'sla','product_procurement_sla', 'order_payment_type',
       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',
       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 
       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA4_listed_price',
       'MA2_discount_offer'],
                       threshold_in=0.01,threshold_out = 0.05, verbose=True):
    
    included = list(initial_list)
    while True:
        changed=False
        ###forward step
        excluded = list(set(x.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))
                
                
        ###backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included]))).fit()
        ###use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() ###null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included


# In[ ]:


import statsmodels.api as sm  

final_features = stepwise_selection(x, y)

print("\n","final_selected_features:",final_features)


# In[ ]:


###  Import statsmodels
import statsmodels.api as sm  

###  Subsetting training data for 15 selected columns
x_rfe1 = x[final_features]

x_rfe1 = sm.add_constant(x_rfe1)

###  Fitting the model with selected variables
lm1 = sm.OLS(y, x_rfe1).fit() 

print(lm1.params)


# In[ ]:


print(lm1.summary())


# In[ ]:


###  Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()

vif['Features'] = x_rfe1.columns
vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


features = list(x_rfe1.columns)
features.remove('const')
features


# In[ ]:


###  Model Evaluation on testing data
x_2 = x[features]


###  Predicition with selected features on the test data
y_pred = lm1.predict(sm.add_constant(x_2))


# In[ ]:


###  Mean square error (MSE)

mse = np.mean((y_pred - y)**2)
mse


# In[ ]:


###  Coefficient values

coef = pd.DataFrame(x_rfe1.columns)
coef['Coefficient'] = pd.Series(lm.coef_)
coef


# In[ ]:


### Mean Square Error 
###  Using K-Fold Cross validation evaluating on selected dataset

# lm = LinearRegression()
fold = KFold(10,shuffle = True, random_state = 100)

cv_scores = -cross_val_score(lm,x[features],y,cv=fold,scoring='neg_mean_squared_error')

print("Neg. of MSE:",cv_scores,"\n")
print("Mean of 5 KFold CV - MSE:",cv_scores.mean())


# In[ ]:


def elasticity(model,x):
    
    features_df = pd.DataFrame(model.params)
    features_df = features_df.rename(columns={0:'coef'})
    
    features_df['imp_feature'] = model.params.index
    features_df = features_df[features_df.imp_feature != 'const'][['imp_feature','coef']]
    features_df.index = range(len(features_df))
#    features

    elasticity_list = list()
    
    for i in range(len(features_df)):
        elasticity_list.append(((features_df.coef[i] * np.mean(x[features_df.imp_feature[i]])) / np.mean(x['gmv'])))

    features_df['elasticity'] = np.round(elasticity_list,3)
    
    sns.barplot(x='elasticity',y='imp_feature',data=features_df)
    plt.show()
    
    return features_df
    


# In[ ]:


elasticity(lm1,camera_train_lm)


# In[ ]:


# Plotting y and y_pred to understand the spread

fig = plt.figure()
plt.scatter(y, y_pred)
fig.suptitle('y vs y_pred', fontsize = 20)              # Plot heading 
plt.xlabel('y', fontsize = 18)                          # X-label
plt.ylabel('y_pred', fontsize = 16)  


# In[ ]:


# Figure size
plt.figure(figsize=(8,5))

# Heatmap
sns.heatmap(camera_train_lm[features].corr(), cmap="YlGnBu", annot=True)
plt.show()


# ### 2. Multiplicative Model

# In[ ]:


ca_week.columns


# In[ ]:


camera_mm = ca_week[['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',
       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',
       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 
       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA4_listed_price',
       'MA2_discount_offer']]         

camera_mm.head()


# In[ ]:


### Applying Log 
camera_mm=np.log(camera_mm)

camera_mm = camera_mm.fillna(0)
camera_mm = camera_mm.replace([np.inf, -np.inf], 0)


# In[ ]:


camera_mm.head()


# In[ ]:


###  Import the StandardScaler()
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

###  Create a scaling object
# scaler = StandardScaler()
scaler = MinMaxScaler()

###  Create a list of the variables that you need to scale
varlist = ['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',
       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',
       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 
       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA4_listed_price',
       'MA2_discount_offer']      



### Scale these variables using 'fit_transform'
camera_mm[varlist] = scaler.fit_transform(camera_mm[varlist])


# In[ ]:


camera_mm.head()


# In[ ]:


### Split the train dataset into X and y
from sklearn.model_selection import train_test_split

x = camera_mm.drop('gmv',axis=1)
y = camera_mm['gmv']

camera_train_mm = camera_mm


# In[ ]:


print("X = Independent variable & Y = Target variable")
print(x.shape,y.shape)


# In[ ]:


### Instantiate
mm = LinearRegression()

### Fit a line
mm.fit(x,y)


# In[ ]:


### Coefficient values

coef = pd.DataFrame(x.columns)
coef['Coefficient'] = pd.Series(mm.coef_)
coef


# In[ ]:


col = x.columns
col


# #### Model Building - Stepwise selection for feature selection

# In[ ]:


import statsmodels.api as sm  
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[ ]:


def stepwise_selection(x, y,
                       initial_list=['discount', 'sla','product_procurement_sla', 'order_payment_type',
       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',
       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 
       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA4_listed_price',
       'MA2_discount_offer'],
                       threshold_in=0.01,threshold_out = 0.05, verbose=True):
    
    included = list(initial_list)
    while True:
        changed=False
        ###forward step
        excluded = list(set(x.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))
                
                
        ###backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included]))).fit()
        ###use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() ###null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included


# In[ ]:


import statsmodels.api as sm  

final_features = stepwise_selection(x, y)

print("\n","final_selected_features:",final_features)


# In[ ]:


### Import statsmodels
import statsmodels.api as sm  

### Subsetting training data for 15 selected columns
x_rfe1 = x[final_features]

x_rfe1 = sm.add_constant(x_rfe1)


### Fitting the model with selected variables
mm1 = sm.OLS(y, x_rfe1).fit() 

print(mm1.params)


# In[ ]:


print(mm1.summary())


# In[ ]:


x_rfe1.drop('TV_ads',1,inplace=True)

x_rfe1 = sm.add_constant(x_rfe1)


### Fitting the model with selected variables
mm1 = sm.OLS(y, x_rfe1).fit() 

print(mm1.params)


# In[ ]:


print(mm1.summary())


# In[ ]:


### Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()

vif['Features'] = x_rfe1.columns
vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


features = list(x_rfe1.columns)
features.remove('const')
features


# In[ ]:


### Model Evaluation on testing data
x_2 = x[features]


### Predicition with selected features on the test data
y_pred = mm1.predict(sm.add_constant(x_2))


# In[ ]:


### Mean square error (MSE)
mse = np.mean((y_pred - y)**2)
mse


# In[ ]:


### Coefficient values

coef = pd.DataFrame(x_rfe1.columns)
coef['Coefficient'] = pd.Series(mm.coef_)
coef


# In[ ]:


### Mean Square Error 
###  Using K-Fold Cross validation evaluating on selected dataset

# lm = LinearRegression()
fold = KFold(10,shuffle = True, random_state = 100)

cv_scores = -cross_val_score(mm,x[features],y,cv=fold,scoring='neg_mean_squared_error')

print("Neg. of MSE:",cv_scores,"\n")
print("Mean of 5 KFold CV - MSE:",cv_scores.mean())


# In[ ]:


def elasticity(model,x):
    
    features_df = pd.DataFrame(model.params)
    features_df = features_df.rename(columns={0:'coef'})
    
    features_df['imp_feature'] = model.params.index
    features_df = features_df[features_df.imp_feature != 'const'][['imp_feature','coef']]
    features_df.index = range(len(features_df))
#     features

    elasticity_list = list()
    
    for i in range(len(features_df)):
        elasticity_list.append(((features_df.coef[i] * np.mean(x[features_df.imp_feature[i]])) / np.mean(x['gmv'])))

    features_df['elasticity'] = np.round(elasticity_list,3)
    
    sns.barplot(x='elasticity',y='imp_feature',data=features_df)
    plt.show()
    
    return features_df
    


# In[ ]:


elasticity(mm1,camera_train_mm)


# In[ ]:


# Plotting y and y_pred to understand the spread

fig = plt.figure()
plt.scatter(y, y_pred)
fig.suptitle('y vs y_pred', fontsize = 20)              # Plot heading 
plt.xlabel('y', fontsize = 18)                          # X-label
plt.ylabel('y_pred', fontsize = 16)  


# ### 3. Kyock Model

# In[ ]:


ca_week.columns


# In[ ]:


camera_km = ca_week[['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',
       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',
       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 
       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA4_listed_price',
       'MA2_discount_offer']]           


camera_km.head()


# In[ ]:


camera_km['lag_1_gmv'] = camera_km['gmv'].shift(-1)


# In[ ]:


### Checking NaN

camera_km.isnull().sum()


# In[ ]:


camera_km = camera_km.fillna(0)


# In[ ]:


camera_km.head()


# In[ ]:


###  Import the StandardScaler()
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

### Create a scaling object
# scaler = StandardScaler()
scaler = MinMaxScaler()

### Create a list of the variables that you need to scale
varlist = ['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',
       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',
       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 
       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA4_listed_price',
       'MA2_discount_offer','lag_1_gmv']

### Scale these variables using 'fit_transform'
camera_km[varlist] = scaler.fit_transform(camera_km[varlist])


# In[ ]:


camera_km.head()


# In[ ]:


### Split the train dataset into X and y
from sklearn.model_selection import train_test_split
x = camera_km.drop('gmv',axis=1)
y = camera_km['gmv']

camera_train_km = camera_km


# In[ ]:


print("X = Independent variable & Y = Target variable")
print(x.shape,y.shape)


# In[ ]:


###  Instantiate
km = LinearRegression()

###  Fit a line
km.fit(x,y)


# In[ ]:


###  Coefficient values

coef = pd.DataFrame(x.columns)
coef['Coefficient'] = pd.Series(km.coef_)
coef


# In[ ]:


col = x.columns
col


# #### Model Building - Stepwise selection for feature selection

# In[ ]:


import statsmodels.api as sm  
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[ ]:


def stepwise_selection(x, y,
                       initial_list=['discount', 'sla','product_procurement_sla', 'order_payment_type',
       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',
       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 
       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA4_listed_price',
       'MA2_discount_offer', 'lag_1_gmv'],
                       threshold_in=0.01,threshold_out = 0.05, verbose=True):
    
    included = list(initial_list)
    while True:
        changed=False
        ### forward step
        excluded = list(set(x.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))
                
                
        ###backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included]))).fit()
        ###use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() ###null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included


# In[ ]:


import statsmodels.api as sm  

final_features = stepwise_selection(x, y)

print("\n","final_selected_features:",final_features)


# In[ ]:


### Import statsmodels
import statsmodels.api as sm  

### Subsetting training data for 15 selected columns
x_rfe1 = x[final_features]

x_rfe1 = sm.add_constant(x_rfe1)

### Fitting the model with selected variables
km1 = sm.OLS(y, x_rfe1).fit() 

print(km1.params)


# In[ ]:


print(km1.summary())


# In[ ]:


### Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()

vif['Features'] = x_rfe1.columns
vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


features = list(x_rfe1.columns)
features.remove('const')
features


# In[ ]:


### Model Evaluation on testing data
x_2 = x[features]


### Predicition with selected features on the test data
y_pred = km1.predict(sm.add_constant(x_2))


# In[ ]:


### Mean square error (MSE)
mse = np.mean((y_pred - y)**2)
mse


# In[ ]:


### Coefficient values

coef = pd.DataFrame(x_rfe1.columns)
coef['Coefficient'] = pd.Series(km.coef_)
coef


# In[ ]:


### Using K-Fold Cross validation evaluating on whole dataset

# lm = LinearRegression()
fold = KFold(10,shuffle = True, random_state = 100)

cv_scores = -cross_val_score(km,x[features],y,cv=fold,scoring='neg_mean_squared_error')

print("Neg. of MSE:",cv_scores,"\n")
print("Mean of 5 KFold CV - MSE:",cv_scores.mean())


# In[ ]:


def elasticity(model,x):
    
    features_df = pd.DataFrame(model.params)
    features_df = features_df.rename(columns={0:'coef'})
    
    features_df['imp_feature'] = model.params.index
    features_df = features_df[features_df.imp_feature != 'const'][['imp_feature','coef']]
    features_df.index = range(len(features_df))
#      features

    elasticity_list = list()
    
    for i in range(len(features_df)):
        elasticity_list.append(((features_df.coef[i] * np.mean(x[features_df.imp_feature[i]])) / np.mean(x['gmv'])))

    features_df['elasticity'] = np.round(elasticity_list,3)
    
    sns.barplot(x='elasticity',y='imp_feature',data=features_df)
    plt.show()
    
    return features_df
    


# In[ ]:


elasticity(km1,camera_train_km)


# In[ ]:


# Plotting y and y_pred to understand the spread

fig = plt.figure()
plt.scatter(y, y_pred)
fig.suptitle('y vs y_pred', fontsize = 20)              # Plot heading 
plt.xlabel('y', fontsize = 18)                          # X-label
plt.ylabel('y_pred', fontsize = 16)  


# In[ ]:


# Figure size
plt.figure(figsize=(8,5))

# Heatmap
sns.heatmap(camera_train_km[features].corr(), cmap="YlGnBu", annot=True)
plt.show()


# ### 4. Distributed Lag Model

# In[ ]:


ca_week.columns


# In[ ]:


camera_dlm = ca_week[['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',
       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',
       'NPS', 'Stock_Index', 'Special_sales', 'Payday','heat_deg_days', 'cool_deg_days', 
       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm',  'MA4_listed_price',
       'MA2_discount_offer', 'lag_1_listed_price','lag_1_discount',
       'lag_2_discount','lag_3_discount','lag_2_NPS','lag_3_NPS','lag_1_Stock_Index',
       'lag_2_Stock_Index','lag_3_Stock_Index','lag_1_Special_sales','lag_2_Special_sales','lag_3_Special_sales',
       'lag_1_Payday','lag_2_Payday','lag_3_Payday']]           


camera_dlm.head()


# In[ ]:


camera_dlm['lag_1_gmv'] = camera_dlm['gmv'].shift(-1)
camera_dlm['lag_2_gmv'] = camera_dlm['gmv'].shift(-2)
camera_dlm['lag_3_gmv'] = camera_dlm['gmv'].shift(-3)


# In[ ]:


### Checking NaN

camera_dlm.isnull().sum()


# In[ ]:


camera_dlm = camera_dlm.fillna(0)


# In[ ]:


camera_dlm.head()


# In[ ]:


camera_dlm.columns


# In[ ]:


###  Import the StandardScaler()
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

###  Create a scaling object
# scaler = StandardScaler()
scaler = MinMaxScaler()


###  Create a list of the variables that you need to scale
varlist = ['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',
       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',
       'NPS', 'Stock_Index', 'Special_sales', 'Payday','heat_deg_days', 'cool_deg_days', 
       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm',  'MA4_listed_price',
       'MA2_discount_offer', 'lag_1_listed_price','lag_1_discount',
       'lag_2_discount','lag_3_discount','lag_2_NPS','lag_3_NPS','lag_1_Stock_Index',
       'lag_2_Stock_Index','lag_3_Stock_Index','lag_1_Special_sales','lag_2_Special_sales','lag_3_Special_sales',
       'lag_1_Payday','lag_2_Payday','lag_3_Payday']


###  Scale these variables using 'fit_transform'
camera_dlm[varlist] = scaler.fit_transform(camera_dlm[varlist])


# In[ ]:


camera_dlm.head()


# In[ ]:


###  Split the train dataset into X and y
from sklearn.model_selection import train_test_split
x = camera_dlm.drop('gmv',axis=1)
y = camera_dlm['gmv']

camera_train_dlm = camera_dlm


# In[ ]:


print("X = Independent variable & Y = Target variable")
print(x.shape,y.shape)


# In[ ]:


###  Instantiate
dlm = LinearRegression()

###  Fit a line
dlm.fit(x,y)


# In[ ]:


###  Coefficient values

coef = pd.DataFrame(x.columns)
coef['Coefficient'] = pd.Series(dlm.coef_)
coef


# In[ ]:


col = x.columns
col


# #### Model Building - Stepwise selection for feature selection

# In[ ]:


import statsmodels.api as sm  
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[ ]:


def stepwise_selection(x, y,
                       initial_list=['discount', 'sla','product_procurement_sla', 'order_payment_type',
       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',
       'NPS', 'Stock_Index', 'Special_sales', 'Payday','heat_deg_days', 'cool_deg_days', 
       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm',  'MA4_listed_price',
       'MA2_discount_offer', 'lag_1_listed_price','lag_1_discount',
       'lag_2_discount','lag_3_discount','lag_2_NPS','lag_3_NPS','lag_1_Stock_Index',
       'lag_2_Stock_Index','lag_3_Stock_Index','lag_1_Special_sales','lag_2_Special_sales','lag_3_Special_sales',
       'lag_1_Payday','lag_2_Payday','lag_3_Payday'],
                     threshold_in=0.01,threshold_out = 0.05, verbose=True):
    
    included = list(initial_list)
    while True:
        changed=False
        ###  forward step
        excluded = list(set(x.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))
                
                
        ###  backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included]))).fit()
        ###  use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() ###  null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included


# In[ ]:


import statsmodels.api as sm  

final_features = stepwise_selection(x, y)

print("\n","final_selected_features:",final_features)


# In[ ]:


###  Import statsmodels
import statsmodels.api as sm  

###  Subsetting training data for 15 selected columns
x_rfe1 = x[final_features]

x_rfe1 = sm.add_constant(x_rfe1)

###  Fitting the model with selected variables
dlm1 = sm.OLS(y, x_rfe1).fit() 

print(dlm1.params)


# In[ ]:


print(dlm1.summary())


# In[ ]:


###  Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()

vif['Features'] = x_rfe1.columns
vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


x_rfe1.drop('discount', axis = 1, inplace = True)


# In[ ]:


### 2


# In[ ]:


# Refitting with final selected variables
x_rfe1 = sm.add_constant(x_rfe1)

# Fitting the model with final selected variables
dlm1 = sm.OLS(y, x_rfe1).fit()   
print(dlm1.summary())


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()

vif['Features'] = x_rfe1.columns
vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


x_rfe1.drop('product_procurement_sla', axis = 1, inplace = True)


# In[ ]:


### 3


# In[ ]:


# Refitting with final selected variables
x_rfe1 = sm.add_constant(x_rfe1)

# Fitting the model with final selected variables
dlm1 = sm.OLS(y, x_rfe1).fit()   
print(dlm1.summary())


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()

vif['Features'] = x_rfe1.columns
vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


x_rfe1.drop('lag_1_listed_price', axis = 1, inplace = True)


# In[ ]:


### 4


# In[ ]:


# Refitting with final selected variables
x_rfe1 = sm.add_constant(x_rfe1)

# Fitting the model with final selected variables
dlm1 = sm.OLS(y, x_rfe1).fit()   
print(dlm1.summary())


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()

vif['Features'] = x_rfe1.columns
vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


features = list(x_rfe1.columns)
features.remove('const')
features


# In[ ]:


###  Model Evaluation on testing data
x_2 = x[features]

###  Predicition with selected features on the test data
y_pred = dlm1.predict(sm.add_constant(x_2))


# In[ ]:


###  Mean square error (MSE)
mse = np.mean((y_pred - y)**2)
mse


# In[ ]:


###  Coefficient values

coef = pd.DataFrame(x_rfe1.columns)
coef['Coefficient'] = pd.Series(dlm.coef_)
coef


# In[ ]:


###  Using K-Fold Cross validation evaluating on whole dataset

# lm = LinearRegression()
fold = KFold(10,shuffle = True, random_state = 100)

cv_scores = -cross_val_score(dlm,x[features],y,cv=fold,scoring='neg_mean_squared_error')

print("Neg. of MSE:",cv_scores,"\n")
print("Mean of 5 KFold CV - MSE:",cv_scores.mean())


# In[ ]:


def elasticity(model,x):
    
    features_df = pd.DataFrame(model.params)
    features_df = features_df.rename(columns={0:'coef'})
    
    features_df['imp_feature'] = model.params.index
    features_df = features_df[features_df.imp_feature != 'const'][['imp_feature','coef']]
    features_df.index = range(len(features_df))
#      features

    elasticity_list = list()
    
    for i in range(len(features_df)):
        elasticity_list.append(((features_df.coef[i] * np.mean(x[features_df.imp_feature[i]])) / np.mean(x['gmv'])))

    features_df['elasticity'] = np.round(elasticity_list,3)
    
    sns.barplot(x='elasticity',y='imp_feature',data=features_df)
    plt.show()
    
    return features_df
    


# In[ ]:


elasticity(dlm1,camera_train_dlm)


# In[ ]:


# Plotting y and y_pred to understand the spread

fig = plt.figure()
plt.scatter(y, y_pred)
fig.suptitle('y vs y_pred', fontsize = 20)              # Plot heading 
plt.xlabel('y', fontsize = 18)                          # X-label
plt.ylabel('y_pred', fontsize = 16)  


# In[ ]:


# Figure size
plt.figure(figsize=(8,5))

# Heatmap
sns.heatmap(camera_train_dlm[features].corr(), cmap="YlGnBu", annot=True)
plt.show()


# ### 5. Distributed Lag + Multiplicative Model

# In[ ]:


ca_week.columns


# In[ ]:


camera_dlmm = ca_week[['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',
       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',
       'NPS', 'Stock_Index', 'Special_sales', 'Payday','heat_deg_days', 'cool_deg_days', 
       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm',  'MA4_listed_price',
       'MA2_discount_offer', 'lag_1_listed_price','lag_1_discount',
       'lag_2_discount','lag_3_discount','lag_2_NPS','lag_3_NPS','lag_1_Stock_Index',
       'lag_2_Stock_Index','lag_3_Stock_Index','lag_1_Special_sales','lag_2_Special_sales','lag_3_Special_sales',
       'lag_1_Payday','lag_2_Payday','lag_3_Payday']]           


camera_dlmm.head()


# In[ ]:


camera_dlmm['lag_1_gmv'] = camera_dlmm['gmv'].shift(-1)
camera_dlmm['lag_2_gmv'] = camera_dlmm['gmv'].shift(-2)
camera_dlmm['lag_3_gmv'] = camera_dlmm['gmv'].shift(-3)


# In[ ]:


### Checking NaN

camera_dlmm.isnull().sum()


# In[ ]:


camera_dlmm = camera_dlmm.fillna(0)


# In[ ]:


### Applying Log 
camera_dlmm=np.log(camera_dlmm)

camera_dlmm = camera_dlmm.fillna(0)
camera_dlmm = camera_dlmm.replace([np.inf, -np.inf], 0)


# In[ ]:


camera_dlmm.head()


# In[ ]:


camera_dlmm.columns


# In[ ]:


###  Import the StandardScaler()
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

###  Create a scaling object
# scaler = StandardScaler()
scaler = MinMaxScaler()


###  Create a list of the variables that you need to scale
varlist = ['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',
       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',
       'NPS', 'Stock_Index', 'Special_sales', 'Payday','heat_deg_days', 'cool_deg_days', 
       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm',  'MA4_listed_price',
       'MA2_discount_offer', 'lag_1_listed_price','lag_1_discount',
       'lag_2_discount','lag_3_discount','lag_2_NPS','lag_3_NPS','lag_1_Stock_Index',
       'lag_2_Stock_Index','lag_3_Stock_Index','lag_1_Special_sales','lag_2_Special_sales','lag_3_Special_sales',
       'lag_1_Payday','lag_2_Payday','lag_3_Payday']


###  Scale these variables using 'fit_transform'
camera_dlmm[varlist] = scaler.fit_transform(camera_dlmm[varlist])


# In[ ]:


camera_dlmm.head()


# In[ ]:


###  Split the train dataset into X and y
from sklearn.model_selection import train_test_split
x = camera_dlmm.drop('gmv',axis=1)
y = camera_dlmm['gmv']

camera_train_dlmm = camera_dlmm


# In[ ]:


print("X = Independent variable & Y = Target variable")
print(x.shape,y.shape)


# In[ ]:


###  Instantiate
dlm = LinearRegression()

###  Fit a line
dlm.fit(x,y)


# In[ ]:


###  Coefficient values

coef = pd.DataFrame(x.columns)
coef['Coefficient'] = pd.Series(dlm.coef_)
coef


# In[ ]:


col = x.columns
col


# #### Model Building - Stepwise selection for feature selection

# In[ ]:


import statsmodels.api as sm  
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[ ]:


def stepwise_selection(x, y,
                       initial_list=['discount', 'sla','product_procurement_sla', 'order_payment_type',
       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',
       'NPS', 'Stock_Index', 'Special_sales', 'Payday','heat_deg_days', 'cool_deg_days', 
       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm',  'MA4_listed_price',
       'MA2_discount_offer', 'lag_1_listed_price','lag_1_discount',
       'lag_2_discount','lag_3_discount','lag_2_NPS','lag_3_NPS','lag_1_Stock_Index',
       'lag_2_Stock_Index','lag_3_Stock_Index','lag_1_Special_sales','lag_2_Special_sales','lag_3_Special_sales',
       'lag_1_Payday','lag_2_Payday','lag_3_Payday'],
                     threshold_in=0.01,threshold_out = 0.05, verbose=True):
    
    included = list(initial_list)
    while True:
        changed=False
        ###  forward step
        excluded = list(set(x.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))
                
                
        ###  backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included]))).fit()
        ###  use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() ###  null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included


# In[ ]:


import statsmodels.api as sm  

final_features = stepwise_selection(x, y)

print("\n","final_selected_features:",final_features)


# In[ ]:


###  Import statsmodels
import statsmodels.api as sm  

###  Subsetting training data for 15 selected columns
x_rfe1 = x[final_features]

x_rfe1 = sm.add_constant(x_rfe1)

###  Fitting the model with selected variables
dlm1 = sm.OLS(y, x_rfe1).fit() 

print(dlm1.params)


# In[ ]:


print(dlm1.summary())


# In[ ]:


###  Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()

vif['Features'] = x_rfe1.columns
vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


x_rfe1.drop('lag_1_gmv', axis = 1, inplace = True)


# In[ ]:


### 2


# In[ ]:


# Refitting with final selected variables
x_rfe1 = sm.add_constant(x_rfe1)

# Fitting the model with final selected variables
dlm1 = sm.OLS(y, x_rfe1).fit()   
print(dlm1.summary())


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()

vif['Features'] = x_rfe1.columns
vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


x_rfe1.drop('lag_2_discount', axis = 1, inplace = True)


# In[ ]:


### 3


# In[ ]:


# Refitting with final selected variables
x_rfe1 = sm.add_constant(x_rfe1)

# Fitting the model with final selected variables
dlm1 = sm.OLS(y, x_rfe1).fit()   
print(dlm1.summary())


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()

vif['Features'] = x_rfe1.columns
vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


x_rfe1.drop('Special_sales', axis = 1, inplace = True)


# In[ ]:


### 4


# In[ ]:


# Refitting with final selected variables
x_rfe1 = sm.add_constant(x_rfe1)

# Fitting the model with final selected variables
dlm1 = sm.OLS(y, x_rfe1).fit()   
print(dlm1.summary())


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()

vif['Features'] = x_rfe1.columns
vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


x_rfe1.drop('lag_1_listed_price', axis = 1, inplace = True)


# In[ ]:


### 5


# In[ ]:


# Refitting with final selected variables
x_rfe1 = sm.add_constant(x_rfe1)

# Fitting the model with final selected variables
dlm1 = sm.OLS(y, x_rfe1).fit()   
print(dlm1.summary())


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()

vif['Features'] = x_rfe1.columns
vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


x_rfe1.drop('lag_1_discount', axis = 1, inplace = True)


# In[ ]:


### 6


# In[ ]:


# Refitting with final selected variables
x_rfe1 = sm.add_constant(x_rfe1)

# Fitting the model with final selected variables
dlm1 = sm.OLS(y, x_rfe1).fit()   
print(dlm1.summary())


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()

vif['Features'] = x_rfe1.columns
vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


x_rfe1.drop('lag_2_gmv', axis = 1, inplace = True)


# In[ ]:


### 7


# In[ ]:


# Refitting with final selected variables
x_rfe1 = sm.add_constant(x_rfe1)

# Fitting the model with final selected variables
dlm1 = sm.OLS(y, x_rfe1).fit()   
print(dlm1.summary())


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()

vif['Features'] = x_rfe1.columns
vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


x_rfe1.drop('lag_2_NPS', axis = 1, inplace = True)


# In[ ]:


### 8


# In[ ]:


# Refitting with final selected variables
x_rfe1 = sm.add_constant(x_rfe1)

# Fitting the model with final selected variables
dlm1 = sm.OLS(y, x_rfe1).fit()   
print(dlm1.summary())


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()

vif['Features'] = x_rfe1.columns
vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


x_rfe1.drop('sla', axis = 1, inplace = True)


# In[ ]:


### 9


# In[ ]:


# Refitting with final selected variables
x_rfe1 = sm.add_constant(x_rfe1)

# Fitting the model with final selected variables
dlm1 = sm.OLS(y, x_rfe1).fit()   
print(dlm1.summary())


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()

vif['Features'] = x_rfe1.columns
vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


features = list(x_rfe1.columns)
features.remove('const')
features


# In[ ]:


###  Model Evaluation on testing data
x_2 = x[features]

###  Predicition with selected features on the test data
y_pred = dlm1.predict(sm.add_constant(x_2))


# In[ ]:


###  Mean square error (MSE)
mse = np.mean((y_pred - y)**2)
mse


# In[ ]:


###  Coefficient values

coef = pd.DataFrame(x_rfe1.columns)
coef['Coefficient'] = pd.Series(dlm.coef_)
coef


# In[ ]:


###  Using K-Fold Cross validation evaluating on whole dataset

# lm = LinearRegression()
fold = KFold(10,shuffle = True, random_state = 100)

cv_scores = -cross_val_score(dlm,x[features],y,cv=fold,scoring='neg_mean_squared_error')

print("Neg. of MSE:",cv_scores,"\n")
print("Mean of 5 KFold CV - MSE:",cv_scores.mean())


# In[ ]:


def elasticity(model,x):
    
    features_df = pd.DataFrame(model.params)
    features_df = features_df.rename(columns={0:'coef'})
    
    features_df['imp_feature'] = model.params.index
    features_df = features_df[features_df.imp_feature != 'const'][['imp_feature','coef']]
    features_df.index = range(len(features_df))
#      features

    elasticity_list = list()
    
    for i in range(len(features_df)):
        elasticity_list.append(((features_df.coef[i] * np.mean(x[features_df.imp_feature[i]])) / np.mean(x['gmv'])))

    features_df['elasticity'] = np.round(elasticity_list,3)
    
    sns.barplot(x='elasticity',y='imp_feature',data=features_df)
    plt.show()
    
    return features_df
    


# In[ ]:


elasticity(dlm1,camera_train_dlm)


# In[ ]:


# Plotting y and y_pred to understand the spread

fig = plt.figure()
plt.scatter(y, y_pred)
fig.suptitle('y vs y_pred', fontsize = 20)              # Plot heading 
plt.xlabel('y', fontsize = 18)                          # X-label
plt.ylabel('y_pred', fontsize = 16)  


# In[ ]:


# Figure size
plt.figure(figsize=(8,5))

# Heatmap
sns.heatmap(camera_train_dlm[features].corr(), cmap="YlGnBu", annot=True)
plt.show()


# # 6. Modeling - Gaming Accessory

# ### Linear Model

# In[ ]:


###  Importing RFE and LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


# In[ ]:


ga_week.columns


# In[ ]:


gaming_lm = ga_week[['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',
       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',
       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 
       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price']]
                            
    
gaming_lm.head()


# In[ ]:


### Checking NaN
gaming_lm.isnull().sum()


# In[ ]:


gaming_lm.fillna(0, inplace = True)


# In[ ]:


### Import the StandardScaler()
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

### Create a scaling object
# scaler = StandardScaler()
scaler = MinMaxScaler()


### Create a list of the variables that you need to scale
varlist = ['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',
       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',
       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 
       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price']
                                      

### Scale these variables using 'fit_transform'
gaming_lm[varlist] = scaler.fit_transform(gaming_lm[varlist])


# In[ ]:


gaming_lm.head()


# In[ ]:


### Split the train dataset into X and y
from sklearn.model_selection import train_test_split
x = gaming_lm.drop('gmv',axis=1)
y = gaming_lm['gmv']

gaming_train_lm = gaming_lm


# In[ ]:


print("x dataset: ",x.shape)
print("y dataset: ",y.shape)


# In[ ]:


###  Instantiate
lm = LinearRegression()

###  Fit a line
lm.fit(x,y)


# In[ ]:


###  Coefficient values

coef = pd.DataFrame(x.columns)
coef['Coefficient'] = pd.Series(lm.coef_)
coef


# In[ ]:


col = x.columns
col


# #### Model Building - Stepwise selection for feature selection

# In[ ]:


import statsmodels.api as sm  
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[ ]:


def stepwise_selection(x, y,
                       initial_list=['discount', 'sla','product_procurement_sla', 'order_payment_type',
       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',
       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 
       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price'],
                       threshold_in=0.01,threshold_out = 0.05, verbose=True):
    
    included = list(initial_list)
    while True:
        changed=False
        ###forward step
        excluded = list(set(x.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))
                
                
        ###backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included]))).fit()
        ###use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() ###null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included


# In[ ]:


import statsmodels.api as sm  

final_features = stepwise_selection(x, y)

print("\n","final_selected_features:",final_features)


# In[ ]:


###  Import statsmodels
import statsmodels.api as sm  

###  Subsetting training data for 15 selected columns
x_rfe1 = x[final_features]

x_rfe1 = sm.add_constant(x_rfe1)

###  Fitting the model with selected variables
lm1 = sm.OLS(y, x_rfe1).fit() 

print(lm1.params)


# In[ ]:


print(lm1.summary())


# In[ ]:


###  Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()

vif['Features'] = x_rfe1.columns
vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


features = list(x_rfe1.columns)
features.remove('const')
features


# In[ ]:


###  Model Evaluation on testing data
x_2 = x[features]


###  Predicition with selected features on the test data
y_pred = lm1.predict(sm.add_constant(x_2))


# In[ ]:


###  Mean square error (MSE)

mse = np.mean((y_pred - y)**2)
mse


# In[ ]:


###  Coefficient values

coef = pd.DataFrame(x_rfe1.columns)
coef['Coefficient'] = pd.Series(lm.coef_)
coef


# In[ ]:


### Mean Square Error 
###  Using K-Fold Cross validation evaluating on selected dataset

# lm = LinearRegression()
fold = KFold(10,shuffle = True, random_state = 100)

cv_scores = -cross_val_score(lm,x[features],y,cv=fold,scoring='neg_mean_squared_error')

print("Neg. of MSE:",cv_scores,"\n")
print("Mean of 5 KFold CV - MSE:",cv_scores.mean())


# In[ ]:


def elasticity(model,x):
    
    features_df = pd.DataFrame(model.params)
    features_df = features_df.rename(columns={0:'coef'})
    
    features_df['imp_feature'] = model.params.index
    features_df = features_df[features_df.imp_feature != 'const'][['imp_feature','coef']]
    features_df.index = range(len(features_df))
#    features

    elasticity_list = list()
    
    for i in range(len(features_df)):
        elasticity_list.append(((features_df.coef[i] * np.mean(x[features_df.imp_feature[i]])) / np.mean(x['gmv'])))

    features_df['elasticity'] = np.round(elasticity_list,3)
    
    sns.barplot(x='elasticity',y='imp_feature',data=features_df)
    plt.show()
    
    return features_df
    


# In[ ]:


elasticity(lm1,gaming_train_lm)


# In[ ]:


# Plotting y and y_pred to understand the spread

fig = plt.figure()
plt.scatter(y, y_pred)
fig.suptitle('y vs y_pred', fontsize = 20)              # Plot heading 
plt.xlabel('y', fontsize = 18)                          # X-label
plt.ylabel('y_pred', fontsize = 16)  


# In[ ]:


# Figure size
plt.figure(figsize=(8,5))

# Heatmap
sns.heatmap(gaming_train_lm[features].corr(), cmap="YlGnBu", annot=True)
plt.show()


# ### Multiplicative Model

# In[ ]:


ga_week.columns


# In[ ]:


gaming_mm = ga_week[['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',
       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',
       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 
       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price']]         

gaming_mm.head()


# In[ ]:


### Applying Log 
gaming_mm=np.log(gaming_mm)

gaming_mm = gaming_mm.fillna(0)
gaming_mm = gaming_mm.replace([np.inf, -np.inf], 0)


# In[ ]:


gaming_mm.head()


# In[ ]:


###  Import the StandardScaler()
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

###  Create a scaling object
# scaler = StandardScaler()
scaler = MinMaxScaler()

###  Create a list of the variables that you need to scale
varlist = ['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',
       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',
       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 
       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price']      



### Scale these variables using 'fit_transform'
gaming_mm[varlist] = scaler.fit_transform(gaming_mm[varlist])


# In[ ]:


gaming_mm.head()


# In[ ]:


### Split the train dataset into X and y
from sklearn.model_selection import train_test_split

x = gaming_mm.drop('gmv',axis=1)
y = gaming_mm['gmv']

gaming_train_mm = gaming_mm


# In[ ]:


print("X = Independent variable & Y = Target variable")
print(x.shape,y.shape)


# In[ ]:


### Instantiate
mm = LinearRegression()

### Fit a line
mm.fit(x,y)


# In[ ]:


### Coefficient values

coef = pd.DataFrame(x.columns)
coef['Coefficient'] = pd.Series(mm.coef_)
coef


# In[ ]:


col = x.columns
col


# #### Model Building - Stepwise selection for feature selection

# In[ ]:


import statsmodels.api as sm  
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[ ]:


def stepwise_selection(x, y,
                       initial_list=['discount', 'sla','product_procurement_sla', 'order_payment_type',
       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',
       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 
       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price'],
                       threshold_in=0.01,threshold_out = 0.05, verbose=True):
    
    included = list(initial_list)
    while True:
        changed=False
        ###forward step
        excluded = list(set(x.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))
                
                
        ###backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included]))).fit()
        ###use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() ###null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included


# In[ ]:


import statsmodels.api as sm  

final_features = stepwise_selection(x, y)

print("\n","final_selected_features:",final_features)


# In[ ]:


### Import statsmodels
import statsmodels.api as sm  

### Subsetting training data for 15 selected columns
x_rfe1 = x[final_features]

x_rfe1 = sm.add_constant(x_rfe1)


### Fitting the model with selected variables
mm1 = sm.OLS(y, x_rfe1).fit() 

print(mm1.params)


# In[ ]:


print(mm1.summary())


# In[ ]:


### Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()

vif['Features'] = x_rfe1.columns
vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


x_rfe1.drop('order_payment_type', axis = 1, inplace = True)


# In[ ]:


### 2


# In[ ]:


# Refitting with final selected variables
x_rfe1 = sm.add_constant(x_rfe1)

# Fitting the model with final selected variables
mm1 = sm.OLS(y, x_rfe1).fit()   
print(mm1.summary())


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()

vif['Features'] = x_rfe1.columns
vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


x_rfe1.drop('snow_on_grnd_cm', axis = 1, inplace = True)


# In[ ]:


### 3


# In[ ]:


# Refitting with final selected variables
x_rfe1 = sm.add_constant(x_rfe1)

# Fitting the model with final selected variables
mm1 = sm.OLS(y, x_rfe1).fit()   
print(mm1.summary())


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()

vif['Features'] = x_rfe1.columns
vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


features = list(x_rfe1.columns)
features.remove('const')
features


# In[ ]:


### Model Evaluation on testing data
x_2 = x[features]


### Predicition with selected features on the test data
y_pred = mm1.predict(sm.add_constant(x_2))


# In[ ]:


### Mean square error (MSE)
mse = np.mean((y_pred - y)**2)
mse


# In[ ]:


### Coefficient values

coef = pd.DataFrame(x_rfe1.columns)
coef['Coefficient'] = pd.Series(mm.coef_)
coef


# In[ ]:


### Mean Square Error 
###  Using K-Fold Cross validation evaluating on selected dataset

# lm = LinearRegression()
fold = KFold(10,shuffle = True, random_state = 100)

cv_scores = -cross_val_score(mm,x[features],y,cv=fold,scoring='neg_mean_squared_error')

print("Neg. of MSE:",cv_scores,"\n")
print("Mean of 5 KFold CV - MSE:",cv_scores.mean())


# In[ ]:


def elasticity(model,x):
    
    features_df = pd.DataFrame(model.params)
    features_df = features_df.rename(columns={0:'coef'})
    
    features_df['imp_feature'] = model.params.index
    features_df = features_df[features_df.imp_feature != 'const'][['imp_feature','coef']]
    features_df.index = range(len(features_df))
#     features

    elasticity_list = list()
    
    for i in range(len(features_df)):
        elasticity_list.append(((features_df.coef[i] * np.mean(x[features_df.imp_feature[i]])) / np.mean(x['gmv'])))

    features_df['elasticity'] = np.round(elasticity_list,3)
    
    sns.barplot(x='elasticity',y='imp_feature',data=features_df)
    plt.show()
    
    return features_df
    


# In[ ]:


elasticity(mm1,gaming_train_mm)


# In[ ]:


# Plotting y and y_pred to understand the spread

fig = plt.figure()
plt.scatter(y, y_pred)
fig.suptitle('y vs y_pred', fontsize = 20)              # Plot heading 
plt.xlabel('y', fontsize = 18)                          # X-label
plt.ylabel('y_pred', fontsize = 16)  


# In[ ]:


# Figure size
plt.figure(figsize=(8,5))

# Heatmap
sns.heatmap(gaming_train_mm[features].corr(), cmap="YlGnBu", annot=True)
plt.show()


# ### Kyock Model

# In[ ]:


ga_week.columns


# In[ ]:


gaming_km = ga_week[['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',
       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',
       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 
       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price']]           


gaming_km.head()


# In[ ]:


gaming_km['lag_1_gmv'] = gaming_km['gmv'].shift(-1)


# In[ ]:


### Checking NaN

gaming_km.isnull().sum()


# In[ ]:


gaming_km = gaming_km.fillna(0)


# In[ ]:


gaming_km.head()


# In[ ]:


###  Import the StandardScaler()
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

### Create a scaling object
# scaler = StandardScaler()
scaler = MinMaxScaler()

### Create a list of the variables that you need to scale
varlist = ['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',
       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',
       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 
       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price','lag_1_gmv']

### Scale these variables using 'fit_transform'
gaming_km[varlist] = scaler.fit_transform(gaming_km[varlist])


# In[ ]:


gaming_km.head()


# In[ ]:


### Split the train dataset into X and y
from sklearn.model_selection import train_test_split
x = gaming_km.drop('gmv',axis=1)
y = gaming_km['gmv']

gaming_train_km = gaming_km


# In[ ]:


print("X = Independent variable & Y = Target variable")
print(x.shape,y.shape)


# In[ ]:


###  Instantiate
km = LinearRegression()

###  Fit a line
km.fit(x,y)


# In[ ]:


###  Coefficient values

coef = pd.DataFrame(x.columns)
coef['Coefficient'] = pd.Series(km.coef_)
coef


# In[ ]:


col = x.columns
col


# #### Model Building - Stepwise selection for feature selection

# In[ ]:


import statsmodels.api as sm  
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[ ]:


def stepwise_selection(x, y,
                       initial_list=['discount', 'sla','product_procurement_sla', 'order_payment_type',
       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',
       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 
       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price', 'lag_1_gmv'],
                       threshold_in=0.01,threshold_out = 0.05, verbose=True):
    
    included = list(initial_list)
    while True:
        changed=False
        ### forward step
        excluded = list(set(x.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))
                
                
        ###backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included]))).fit()
        ###use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() ###null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included


# In[ ]:


import statsmodels.api as sm  

final_features = stepwise_selection(x, y)

print("\n","final_selected_features:",final_features)


# In[ ]:


### Import statsmodels
import statsmodels.api as sm  

### Subsetting training data for 15 selected columns
x_rfe1 = x[final_features]

x_rfe1 = sm.add_constant(x_rfe1)

### Fitting the model with selected variables
km1 = sm.OLS(y, x_rfe1).fit() 

print(km1.params)


# In[ ]:


print(km1.summary())


# In[ ]:


### Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()

vif['Features'] = x_rfe1.columns
vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


features = list(x_rfe1.columns)
features.remove('const')
features


# In[ ]:


### Model Evaluation on testing data
x_2 = x[features]


### Predicition with selected features on the test data
y_pred = km1.predict(sm.add_constant(x_2))


# In[ ]:


### Mean square error (MSE)
mse = np.mean((y_pred - y)**2)
mse


# In[ ]:


### Coefficient values

coef = pd.DataFrame(x_rfe1.columns)
coef['Coefficient'] = pd.Series(km.coef_)
coef


# In[ ]:


### Using K-Fold Cross validation evaluating on whole dataset

# lm = LinearRegression()
fold = KFold(10,shuffle = True, random_state = 100)

cv_scores = -cross_val_score(km,x[features],y,cv=fold,scoring='neg_mean_squared_error')

print("Neg. of MSE:",cv_scores,"\n")
print("Mean of 5 KFold CV - MSE:",cv_scores.mean())


# In[ ]:


def elasticity(model,x):
    
    features_df = pd.DataFrame(model.params)
    features_df = features_df.rename(columns={0:'coef'})
    
    features_df['imp_feature'] = model.params.index
    features_df = features_df[features_df.imp_feature != 'const'][['imp_feature','coef']]
    features_df.index = range(len(features_df))
#      features

    elasticity_list = list()
    
    for i in range(len(features_df)):
        elasticity_list.append(((features_df.coef[i] * np.mean(x[features_df.imp_feature[i]])) / np.mean(x['gmv'])))

    features_df['elasticity'] = np.round(elasticity_list,3)
    
    sns.barplot(x='elasticity',y='imp_feature',data=features_df)
    plt.show()
    
    return features_df
    


# In[ ]:


elasticity(km1,gaming_train_km)


# In[ ]:


# Plotting y and y_pred to understand the spread

fig = plt.figure()
plt.scatter(y, y_pred)
fig.suptitle('y vs y_pred', fontsize = 20)              # Plot heading 
plt.xlabel('y', fontsize = 18)                          # X-label
plt.ylabel('y_pred', fontsize = 16)  


# In[ ]:


# Figure size
plt.figure(figsize=(8,5))

# Heatmap
sns.heatmap(gaming_train_km[features].corr(), cmap="YlGnBu", annot=True)
plt.show()


# ### Distributed Lag Model

# In[ ]:


gaming_dlm = ga_week[['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',
       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',
       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 
       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price','lag_1_listed_price','lag_1_discount_offer',
       'lag_2_discount_offer','lag_3_discount_offer','lag_2_NPS','lag_3_NPS','lag_1_Stock_Index',
       'lag_2_Stock_Index','lag_3_Stock_Index','lag_1_Special_sales','lag_2_Special_sales','lag_3_Special_sales',
       'lag_1_Payday','lag_2_Payday','lag_3_Payday']]           


gaming_dlm.head()


# In[ ]:


gaming_dlm['lag_1_gmv'] = gaming_dlm['gmv'].shift(-1)
gaming_dlm['lag_2_gmv'] = gaming_dlm['gmv'].shift(-2)
gaming_dlm['lag_3_gmv'] = gaming_dlm['gmv'].shift(-3)


# In[ ]:


### Checking NaN

# gaming_dlm.isnull().sum()


# In[ ]:


gaming_dlm = gaming_dlm.fillna(0)


# In[ ]:


gaming_dlm.head()


# In[ ]:


gaming_dlm.columns


# In[ ]:


###  Import the StandardScaler()
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

###  Create a scaling object
# scaler = StandardScaler()
scaler = MinMaxScaler()


###  Create a list of the variables that you need to scale
varlist = ['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',
       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',
       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 
       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price','lag_1_listed_price','lag_1_discount_offer',
       'lag_2_discount_offer','lag_3_discount_offer','lag_2_NPS','lag_3_NPS','lag_1_Stock_Index',
       'lag_2_Stock_Index','lag_3_Stock_Index','lag_1_Special_sales','lag_2_Special_sales','lag_3_Special_sales',
       'lag_1_Payday','lag_2_Payday','lag_3_Payday']


###  Scale these variables using 'fit_transform'
gaming_dlm[varlist] = scaler.fit_transform(gaming_dlm[varlist])


# In[ ]:


gaming_dlm.head()


# In[ ]:


###  Split the train dataset into X and y
from sklearn.model_selection import train_test_split
x = gaming_dlm.drop('gmv',axis=1)
y = gaming_dlm['gmv']

gaming_train_dlm = gaming_dlm


# In[ ]:


print("X = Independent variable & Y = Target variable")
print(x.shape,y.shape)


# In[ ]:


###  Instantiate
dlm = LinearRegression()

###  Fit a line
dlm.fit(x,y)


# In[ ]:


###  Coefficient values

coef = pd.DataFrame(x.columns)
coef['Coefficient'] = pd.Series(dlm.coef_)
coef


# In[ ]:


col = x.columns
col


# #### Model Building - Stepwise selection for feature selection

# In[ ]:


import statsmodels.api as sm  
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[ ]:


def stepwise_selection(x, y,
                       initial_list=['discount', 'sla','product_procurement_sla', 'order_payment_type',
       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',
       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 
       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price','lag_1_listed_price','lag_1_discount_offer',
       'lag_2_discount_offer','lag_3_discount_offer','lag_2_NPS','lag_3_NPS','lag_1_Stock_Index',
       'lag_2_Stock_Index','lag_3_Stock_Index','lag_1_Special_sales','lag_2_Special_sales','lag_3_Special_sales',
       'lag_1_Payday','lag_2_Payday','lag_3_Payday'],
                     threshold_in=0.01,threshold_out = 0.05, verbose=True):
    
    included = list(initial_list)
    while True:
        changed=False
        ###  forward step
        excluded = list(set(x.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))
                
                
        ###  backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included]))).fit()
        ###  use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() ###  null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included


# In[ ]:


import statsmodels.api as sm  

final_features = stepwise_selection(x, y)

print("\n","final_selected_features:",final_features)


# In[ ]:


###  Import statsmodels
import statsmodels.api as sm  

###  Subsetting training data for 15 selected columns
x_rfe1 = x[final_features]

x_rfe1 = sm.add_constant(x_rfe1)

###  Fitting the model with selected variables
dlm1 = sm.OLS(y, x_rfe1).fit() 

print(dlm1.params)


# In[ ]:


print(dlm1.summary())


# In[ ]:


###  Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()

vif['Features'] = x_rfe1.columns
vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


x_rfe1.drop('lag_3_Stock_Index', axis = 1, inplace = True)


# In[ ]:


### 2


# In[ ]:


# Refitting with final selected variables
x_rfe1 = sm.add_constant(x_rfe1)

# Fitting the model with final selected variables
dlm1 = sm.OLS(y, x_rfe1).fit()   
print(dlm1.summary())


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()

vif['Features'] = x_rfe1.columns
vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


x_rfe1.drop('lag_3_NPS', axis = 1, inplace = True)


# In[ ]:


### 3


# In[ ]:


# Refitting with final selected variables
x_rfe1 = sm.add_constant(x_rfe1)

# Fitting the model with final selected variables
dlm1 = sm.OLS(y, x_rfe1).fit()   
print(dlm1.summary())


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()

vif['Features'] = x_rfe1.columns
vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


x_rfe1.drop('sla', axis = 1, inplace = True)


# In[ ]:


### 4


# In[ ]:


# Refitting with final selected variables
x_rfe1 = sm.add_constant(x_rfe1)

# Fitting the model with final selected variables
dlm1 = sm.OLS(y, x_rfe1).fit()   
print(dlm1.summary())


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()

vif['Features'] = x_rfe1.columns
vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


features = list(x_rfe1.columns)
features.remove('const')
features


# In[ ]:


###  Model Evaluation on testing data
x_2 = x[features]

###  Predicition with selected features on the test data
y_pred = dlm1.predict(sm.add_constant(x_2))


# In[ ]:


###  Mean square error (MSE)
mse = np.mean((y_pred - y)**2)
mse


# In[ ]:


###  Coefficient values

coef = pd.DataFrame(x_rfe1.columns)
coef['Coefficient'] = pd.Series(dlm.coef_)
coef


# In[ ]:


###  Using K-Fold Cross validation evaluating on whole dataset

# lm = LinearRegression()
fold = KFold(10,shuffle = True, random_state = 100)

cv_scores = -cross_val_score(dlm,x[features],y,cv=fold,scoring='neg_mean_squared_error')

print("Neg. of MSE:",cv_scores,"\n")
print("Mean of 5 KFold CV - MSE:",cv_scores.mean())


# In[ ]:


def elasticity(model,x):
    
    features_df = pd.DataFrame(model.params)
    features_df = features_df.rename(columns={0:'coef'})
    
    features_df['imp_feature'] = model.params.index
    features_df = features_df[features_df.imp_feature != 'const'][['imp_feature','coef']]
    features_df.index = range(len(features_df))
#      features

    elasticity_list = list()
    
    for i in range(len(features_df)):
        elasticity_list.append(((features_df.coef[i] * np.mean(x[features_df.imp_feature[i]])) / np.mean(x['gmv'])))

    features_df['elasticity'] = np.round(elasticity_list,3)
    
    sns.barplot(x='elasticity',y='imp_feature',data=features_df)
    plt.show()
    
    return features_df
    


# In[ ]:


elasticity(dlm1,gaming_train_dlm)


# In[ ]:


# Plotting y and y_pred to understand the spread

fig = plt.figure()
plt.scatter(y, y_pred)
fig.suptitle('y vs y_pred', fontsize = 20)              # Plot heading 
plt.xlabel('y', fontsize = 18)                          # X-label
plt.ylabel('y_pred', fontsize = 16)  


# In[ ]:


# Figure size
plt.figure(figsize=(8,5))

# Heatmap
sns.heatmap(gaming_train_dlm[features].corr(), cmap="YlGnBu", annot=True)
plt.show()


# ### Distributed + Multiplicative Lag Model

# In[ ]:


gaming_dlmm = ga_week[['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',
       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',
       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 
       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price','lag_1_listed_price','lag_1_discount_offer',
       'lag_2_discount_offer','lag_3_discount_offer','lag_2_NPS','lag_3_NPS','lag_1_Stock_Index',
       'lag_2_Stock_Index','lag_3_Stock_Index','lag_1_Special_sales','lag_2_Special_sales','lag_3_Special_sales',
       'lag_1_Payday','lag_2_Payday','lag_3_Payday']]           


gaming_dlmm.head()


# In[ ]:


gaming_dlmm['lag_1_gmv'] = gaming_dlmm['gmv'].shift(-1)
gaming_dlmm['lag_2_gmv'] = gaming_dlmm['gmv'].shift(-2)
gaming_dlmm['lag_3_gmv'] = gaming_dlmm['gmv'].shift(-3)


# In[ ]:


### Checking NaN

gaming_dlmm.isnull().sum()


# In[ ]:


gaming_dlmm = gaming_dlmm.fillna(0)


# In[ ]:


### Applying Log 
gaming_dlmm=np.log(gaming_dlmm)

gaming_dlmm = gaming_dlmm.fillna(0)
gaming_dlmm = gaming_dlmm.replace([np.inf, -np.inf], 0)


# In[ ]:


gaming_dlmm.head()


# In[ ]:


gaming_dlmm.columns


# In[ ]:


###  Import the StandardScaler()
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

###  Create a scaling object
# scaler = StandardScaler()
scaler = MinMaxScaler()


###  Create a list of the variables that you need to scale
varlist = ['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',
       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',
       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 
       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price','lag_1_listed_price','lag_1_discount_offer',
       'lag_2_discount_offer','lag_3_discount_offer','lag_2_NPS','lag_3_NPS','lag_1_Stock_Index',
       'lag_2_Stock_Index','lag_3_Stock_Index','lag_1_Special_sales','lag_2_Special_sales','lag_3_Special_sales',
       'lag_1_Payday','lag_2_Payday','lag_3_Payday']


###  Scale these variables using 'fit_transform'
gaming_dlmm[varlist] = scaler.fit_transform(gaming_dlmm[varlist])


# In[ ]:


gaming_dlmm.head()


# In[ ]:


###  Split the train dataset into X and y
from sklearn.model_selection import train_test_split
x = gaming_dlmm.drop('gmv',axis=1)
y = gaming_dlmm['gmv']

gaming_train_dlmm = gaming_dlmm


# In[ ]:


print("X = Independent variable & Y = Target variable")
print(x.shape,y.shape)


# In[ ]:


###  Instantiate
dlm = LinearRegression()

###  Fit a line
dlm.fit(x,y)


# In[ ]:


###  Coefficient values

coef = pd.DataFrame(x.columns)
coef['Coefficient'] = pd.Series(dlm.coef_)
coef


# In[ ]:


col = x.columns
col


# #### Model Building - Stepwise selection for feature selection

# In[ ]:


import statsmodels.api as sm  
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[ ]:


def stepwise_selection(x, y,
                       initial_list=['discount', 'sla','product_procurement_sla', 'order_payment_type',
       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',
       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 
       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price','lag_1_listed_price','lag_1_discount_offer',
       'lag_2_discount_offer','lag_3_discount_offer','lag_2_NPS','lag_3_NPS','lag_1_Stock_Index',
       'lag_2_Stock_Index','lag_3_Stock_Index','lag_1_Special_sales','lag_2_Special_sales','lag_3_Special_sales',
       'lag_1_Payday','lag_2_Payday','lag_3_Payday'],
                     threshold_in=0.01,threshold_out = 0.05, verbose=True):
    
    included = list(initial_list)
    while True:
        changed=False
        ###  forward step
        excluded = list(set(x.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))
                
                
        ###  backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included]))).fit()
        ###  use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() ###  null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included


# In[ ]:


import statsmodels.api as sm  

final_features = stepwise_selection(x, y)

print("\n","final_selected_features:",final_features)


# In[ ]:


###  Import statsmodels
import statsmodels.api as sm  

###  Subsetting training data for 15 selected columns
x_rfe1 = x[final_features]

x_rfe1 = sm.add_constant(x_rfe1)

###  Fitting the model with selected variables
dlm1 = sm.OLS(y, x_rfe1).fit() 

print(dlm1.params)


# In[ ]:


print(dlm1.summary())


# In[ ]:


###  Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()

vif['Features'] = x_rfe1.columns
vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


x_rfe1.drop('lag_3_Stock_Index', axis = 1, inplace = True)


# In[ ]:


### 2


# In[ ]:


# Refitting with final selected variables
x_rfe1 = sm.add_constant(x_rfe1)

# Fitting the model with final selected variables
dlm1 = sm.OLS(y, x_rfe1).fit()   
print(dlm1.summary())


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()

vif['Features'] = x_rfe1.columns
vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


x_rfe1.drop('lag_3_discount_offer', axis = 1, inplace = True)


# In[ ]:


### 3


# In[ ]:


# Refitting with final selected variables
x_rfe1 = sm.add_constant(x_rfe1)

# Fitting the model with final selected variables
dlm1 = sm.OLS(y, x_rfe1).fit()   
print(dlm1.summary())


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()

vif['Features'] = x_rfe1.columns
vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


x_rfe1.drop('lag_2_NPS', axis = 1, inplace = True)


# In[ ]:


### 4


# In[ ]:


# Refitting with final selected variables
x_rfe1 = sm.add_constant(x_rfe1)

# Fitting the model with final selected variables
dlm1 = sm.OLS(y, x_rfe1).fit()   
print(dlm1.summary())


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()

vif['Features'] = x_rfe1.columns
vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


x_rfe1.drop('lag_2_discount_offer', axis = 1, inplace = True)


# In[ ]:


### 5


# In[ ]:


# Refitting with final selected variables
x_rfe1 = sm.add_constant(x_rfe1)

# Fitting the model with final selected variables
dlm1 = sm.OLS(y, x_rfe1).fit()   
print(dlm1.summary())


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()

vif['Features'] = x_rfe1.columns
vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


x_rfe1.drop('Online_marketing_ads', axis = 1, inplace = True)


# In[ ]:


### 6


# In[ ]:


# Refitting with final selected variables
x_rfe1 = sm.add_constant(x_rfe1)

# Fitting the model with final selected variables
dlm1 = sm.OLS(y, x_rfe1).fit()   
print(dlm1.summary())


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()

vif['Features'] = x_rfe1.columns
vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


x_rfe1.drop('lag_1_listed_price', axis = 1, inplace = True)


# In[ ]:


### 7


# In[ ]:


# Refitting with final selected variables
x_rfe1 = sm.add_constant(x_rfe1)

# Fitting the model with final selected variables
dlm1 = sm.OLS(y, x_rfe1).fit()   
print(dlm1.summary())


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()

vif['Features'] = x_rfe1.columns
vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


x_rfe1.drop('lag_1_discount_offer', axis = 1, inplace = True)


# In[ ]:


### 8


# In[ ]:


# Refitting with final selected variables
x_rfe1 = sm.add_constant(x_rfe1)

# Fitting the model with final selected variables
dlm1 = sm.OLS(y, x_rfe1).fit()   
print(dlm1.summary())


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()

vif['Features'] = x_rfe1.columns
vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


x_rfe1.drop('lag_3_NPS', axis = 1, inplace = True)


# In[ ]:


### 9


# In[ ]:


# Refitting with final selected variables
x_rfe1 = sm.add_constant(x_rfe1)

# Fitting the model with final selected variables
dlm1 = sm.OLS(y, x_rfe1).fit()   
print(dlm1.summary())


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()

vif['Features'] = x_rfe1.columns
vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


x_rfe1.drop('Stock_Index', axis = 1, inplace = True)


# In[ ]:


### 10


# In[ ]:


# Refitting with final selected variables
x_rfe1 = sm.add_constant(x_rfe1)

# Fitting the model with final selected variables
dlm1 = sm.OLS(y, x_rfe1).fit()   
print(dlm1.summary())


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()

vif['Features'] = x_rfe1.columns
vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


x_rfe1.drop('order_payment_type', axis = 1, inplace = True)


# In[ ]:


### 11


# In[ ]:


# Refitting with final selected variables
x_rfe1 = sm.add_constant(x_rfe1)

# Fitting the model with final selected variables
dlm1 = sm.OLS(y, x_rfe1).fit()   
print(dlm1.summary())


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()

vif['Features'] = x_rfe1.columns
vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


x_rfe1.drop('lag_1_gmv', axis = 1, inplace = True)


# In[ ]:


### 12


# In[ ]:


# Refitting with final selected variables
x_rfe1 = sm.add_constant(x_rfe1)

# Fitting the model with final selected variables
dlm1 = sm.OLS(y, x_rfe1).fit()   
print(dlm1.summary())


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()

vif['Features'] = x_rfe1.columns
vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


x_rfe1.drop('lag_1_Stock_Index', axis = 1, inplace = True)


# In[ ]:


### 13


# In[ ]:


# Refitting with final selected variables
x_rfe1 = sm.add_constant(x_rfe1)

# Fitting the model with final selected variables
dlm1 = sm.OLS(y, x_rfe1).fit()   
print(dlm1.summary())


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()

vif['Features'] = x_rfe1.columns
vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


x_rfe1.drop('TV_ads', axis = 1, inplace = True)


# In[ ]:


### 14


# In[ ]:


# Refitting with final selected variables
x_rfe1 = sm.add_constant(x_rfe1)

# Fitting the model with final selected variables
dlm1 = sm.OLS(y, x_rfe1).fit()   
print(dlm1.summary())


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()

vif['Features'] = x_rfe1.columns
vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


features = list(x_rfe1.columns)
features.remove('const')
features


# In[ ]:


###  Model Evaluation on testing data
x_2 = x[features]

###  Predicition with selected features on the test data
y_pred = dlm1.predict(sm.add_constant(x_2))


# In[ ]:


###  Mean square error (MSE)
mse = np.mean((y_pred - y)**2)
mse


# In[ ]:


###  Coefficient values

coef = pd.DataFrame(x_rfe1.columns)
coef['Coefficient'] = pd.Series(dlm.coef_)
coef


# In[ ]:


###  Using K-Fold Cross validation evaluating on whole dataset

# lm = LinearRegression()
fold = KFold(10,shuffle = True, random_state = 100)

cv_scores = -cross_val_score(dlm,x[features],y,cv=fold,scoring='neg_mean_squared_error')

print("Neg. of MSE:",cv_scores,"\n")
print("Mean of 5 KFold CV - MSE:",cv_scores.mean())


# In[ ]:


def elasticity(model,x):
    
    features_df = pd.DataFrame(model.params)
    features_df = features_df.rename(columns={0:'coef'})
    
    features_df['imp_feature'] = model.params.index
    features_df = features_df[features_df.imp_feature != 'const'][['imp_feature','coef']]
    features_df.index = range(len(features_df))
#      features

    elasticity_list = list()
    
    for i in range(len(features_df)):
        elasticity_list.append(((features_df.coef[i] * np.mean(x[features_df.imp_feature[i]])) / np.mean(x['gmv'])))

    features_df['elasticity'] = np.round(elasticity_list,3)
    
    sns.barplot(x='elasticity',y='imp_feature',data=features_df)
    plt.show()
    
    return features_df
    


# In[ ]:


elasticity(dlm1,gaming_train_dlm)


# In[ ]:


# Plotting y and y_pred to understand the spread

fig = plt.figure()
plt.scatter(y, y_pred)
fig.suptitle('y vs y_pred', fontsize = 20)              # Plot heading 
plt.xlabel('y', fontsize = 18)                          # X-label
plt.ylabel('y_pred', fontsize = 16)  


# In[ ]:


# Figure size
plt.figure(figsize=(8,5))

# Heatmap
sns.heatmap(gaming_train_dlm[features].corr(), cmap="YlGnBu", annot=True)
plt.show()


# # 7. Modeling - Home Audio

# ### Linear Model

# In[ ]:


###  Importing RFE and LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


# In[ ]:


ha_week.columns


# In[ ]:


home_lm = ha_week[['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',
       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',
       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 
       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price', 'MA4_listed_price']]
                            
    
home_lm.head()


# In[ ]:


### Checking NaN

home_lm.isnull().sum()


# In[ ]:


home_lm.fillna(0, inplace = True)


# In[ ]:


### Import the StandardScaler()
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

### Create a scaling object
# scaler = StandardScaler()
scaler = MinMaxScaler()


### Create a list of the variables that you need to scale
varlist = ['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',
       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',
       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 
       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price', 'MA4_listed_price']
                                      

### Scale these variables using 'fit_transform'
home_lm[varlist] = scaler.fit_transform(home_lm[varlist])


# In[ ]:


home_lm.head()


# In[ ]:


### Split the train dataset into X and y
from sklearn.model_selection import train_test_split
x = home_lm.drop('gmv',axis=1)
y = home_lm['gmv']

home_train_lm = home_lm


# In[ ]:


print("x dataset: ",x.shape)
print("y dataset: ",y.shape)


# In[ ]:


###  Instantiate
lm = LinearRegression()

###  Fit a line
lm.fit(x,y)


# In[ ]:


###  Coefficient values

coef = pd.DataFrame(x.columns)
coef['Coefficient'] = pd.Series(lm.coef_)
coef


# In[ ]:


col = x.columns
col


# #### Model Building - Stepwise selection for feature selection

# In[ ]:


import statsmodels.api as sm  
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[ ]:


def stepwise_selection(x, y,
                       initial_list=['discount', 'sla','product_procurement_sla', 'order_payment_type',
       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',
       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 
       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price', 'MA4_listed_price'],
                       threshold_in=0.01,threshold_out = 0.05, verbose=True):
    
    included = list(initial_list)
    while True:
        changed=False
        ###forward step
        excluded = list(set(x.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))
                
                
        ###backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included]))).fit()
        ###use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() ###null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included


# In[ ]:


import statsmodels.api as sm  

final_features = stepwise_selection(x, y)

print("\n","final_selected_features:",final_features)


# In[ ]:


### Import statsmodels
import statsmodels.api as sm  

### Subsetting training data for 15 selected columns
x_rfe1 = x[final_features]

x_rfe1 = sm.add_constant(x_rfe1)

### Fitting the model with selected variables
lm1 = sm.OLS(y, x_rfe1).fit() 

print(lm1.params)


# In[ ]:


print(lm1.summary())


# In[ ]:


###  Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()

vif['Features'] = x_rfe1.columns
vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


features = list(x_rfe1.columns)
features.remove('const')
features


# In[ ]:


###  Model Evaluation on testing data
x_2 = x[features]

###  Predicition with selected features on the test data
y_pred = lm1.predict(sm.add_constant(x_2))


# In[ ]:


###  Mean square error (MSE)

mse = np.mean((y_pred - y)**2)
mse


# In[ ]:


###  Coefficient values

coef = pd.DataFrame(x_rfe1.columns)
coef['Coefficient'] = pd.Series(lm.coef_)
coef


# In[ ]:


### Mean Square Error 
###  Using K-Fold Cross validation evaluating on selected dataset

# lm = LinearRegression()
fold = KFold(10,shuffle = True, random_state = 100)

cv_scores = -cross_val_score(lm,x[features],y,cv=fold,scoring='neg_mean_squared_error')

print("Neg. of MSE:",cv_scores,"\n")
print("Mean of 5 KFold CV - MSE:",cv_scores.mean())


# In[ ]:


def elasticity(model,x):
    
    features_df = pd.DataFrame(model.params)
    features_df = features_df.rename(columns={0:'coef'})
    
    features_df['imp_feature'] = model.params.index
    features_df = features_df[features_df.imp_feature != 'const'][['imp_feature','coef']]
    features_df.index = range(len(features_df))
#    features

    elasticity_list = list()
    
    for i in range(len(features_df)):
        elasticity_list.append(((features_df.coef[i] * np.mean(x[features_df.imp_feature[i]])) / np.mean(x['gmv'])))

    features_df['elasticity'] = np.round(elasticity_list,3)
    
    sns.barplot(x='elasticity',y='imp_feature',data=features_df)
    plt.show()
    
    return features_df
    


# In[ ]:


elasticity(lm1,home_train_lm)


# In[ ]:


# Plotting y and y_pred to understand the spread

fig = plt.figure()
plt.scatter(y, y_pred)
fig.suptitle('y vs y_pred', fontsize = 20)              # Plot heading 
plt.xlabel('y', fontsize = 18)                          # X-label
plt.ylabel('y_pred', fontsize = 16)  


# In[ ]:


# Figure size
plt.figure(figsize=(8,5))

# Heatmap
sns.heatmap(home_train_lm[features].corr(), cmap="YlGnBu", annot=True)
plt.show()


# ### Multiplicative Model

# In[ ]:


ha_week.columns


# In[ ]:


home_mm = ha_week[['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',
       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',
       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 
       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price', 'MA4_listed_price']]         

home_mm.head()


# In[ ]:


### Applying Log 
home_mm=np.log(home_mm)

home_mm = home_mm.fillna(0)
home_mm = home_mm.replace([np.inf, -np.inf], 0)


# In[ ]:


home_mm.head()


# In[ ]:


###  Import the StandardScaler()
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

###  Create a scaling object
# scaler = StandardScaler()
scaler = MinMaxScaler()

###  Create a list of the variables that you need to scale
varlist = ['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',
       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',
       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 
       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price', 'MA4_listed_price']      



### Scale these variables using 'fit_transform'
home_mm[varlist] = scaler.fit_transform(home_mm[varlist])


# In[ ]:


home_mm.head()


# In[ ]:


### Split the train dataset into X and y
from sklearn.model_selection import train_test_split

x = home_mm.drop('gmv',axis=1)
y = home_mm['gmv']

home_train_mm = home_mm


# In[ ]:


print("X = Independent variable & Y = Target variable")
print(x.shape,y.shape)


# In[ ]:


### Instantiate
mm = LinearRegression()

### Fit a line
mm.fit(x,y)


# In[ ]:


### Coefficient values

coef = pd.DataFrame(x.columns)
coef['Coefficient'] = pd.Series(mm.coef_)
coef


# In[ ]:


col = x.columns
col


# #### Model Building - Stepwise selection for feature selection

# In[ ]:


import statsmodels.api as sm  
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[ ]:


def stepwise_selection(x, y,
                       initial_list=['discount', 'sla','product_procurement_sla', 'order_payment_type',
       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',
       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 
       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price', 'MA4_listed_price'],
                       threshold_in=0.01,threshold_out = 0.05, verbose=True):
    
    included = list(initial_list)
    while True:
        changed=False
        ###forward step
        excluded = list(set(x.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))
                
                
        ###backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included]))).fit()
        ###use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() ###null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included


# In[ ]:


import statsmodels.api as sm  

final_features = stepwise_selection(x, y)

print("\n","final_selected_features:",final_features)


# In[ ]:


### Import statsmodels
import statsmodels.api as sm  

### Subsetting training data for 15 selected columns
x_rfe1 = x[final_features]

x_rfe1 = sm.add_constant(x_rfe1)


### Fitting the model with selected variables
mm1 = sm.OLS(y, x_rfe1).fit() 

print(mm1.params)


# In[ ]:


print(mm1.summary())


# In[ ]:


### Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()

vif['Features'] = x_rfe1.columns
vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


features = list(x_rfe1.columns)
features.remove('const')
features


# In[ ]:


### Model Evaluation on testing data
x_2 = x[features]


### Predicition with selected features on the test data
y_pred = mm1.predict(sm.add_constant(x_2))


# In[ ]:


### Mean square error (MSE)
mse = np.mean((y_pred - y)**2)
mse


# In[ ]:


### Coefficient values

coef = pd.DataFrame(x_rfe1.columns)
coef['Coefficient'] = pd.Series(mm.coef_)
coef


# In[ ]:


### Mean Square Error 
###  Using K-Fold Cross validation evaluating on selected dataset

# lm = LinearRegression()
fold = KFold(10,shuffle = True, random_state = 100)

cv_scores = -cross_val_score(mm,x[features],y,cv=fold,scoring='neg_mean_squared_error')

print("Neg. of MSE:",cv_scores,"\n")
print("Mean of 5 KFold CV - MSE:",cv_scores.mean())


# In[ ]:


def elasticity(model,x):
    
    features_df = pd.DataFrame(model.params)
    features_df = features_df.rename(columns={0:'coef'})
    
    features_df['imp_feature'] = model.params.index
    features_df = features_df[features_df.imp_feature != 'const'][['imp_feature','coef']]
    features_df.index = range(len(features_df))
#     features

    elasticity_list = list()
    
    for i in range(len(features_df)):
        elasticity_list.append(((features_df.coef[i] * np.mean(x[features_df.imp_feature[i]])) / np.mean(x['gmv'])))

    features_df['elasticity'] = np.round(elasticity_list,3)
    
    sns.barplot(x='elasticity',y='imp_feature',data=features_df)
    plt.show()
    
    return features_df
    


# In[ ]:


elasticity(mm1,home_train_mm)


# In[ ]:


# Plotting y and y_pred to understand the spread

fig = plt.figure()
plt.scatter(y, y_pred)
fig.suptitle('y vs y_pred', fontsize = 20)              # Plot heading 
plt.xlabel('y', fontsize = 18)                          # X-label
plt.ylabel('y_pred', fontsize = 16)  


# In[ ]:


# Figure size
plt.figure(figsize=(8,5))

# Heatmap
sns.heatmap(home_train_mm[features].corr(), cmap="YlGnBu", annot=True)
plt.show()


# ### Kyock Model

# In[ ]:


ha_week.columns


# In[ ]:


home_km = ha_week[['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',
       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',
       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 
       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price', 'MA4_listed_price']]           


home_km.head()


# In[ ]:


home_km['lag_1_gmv'] = home_km['gmv'].shift(-1)


# In[ ]:


### Checking NaN

home_km.isnull().sum()


# In[ ]:


home_km = home_km.fillna(0)


# In[ ]:


home_km.head()


# In[ ]:


###  Import the StandardScaler()
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

### Create a scaling object
# scaler = StandardScaler()
scaler = MinMaxScaler()

### Create a list of the variables that you need to scale
varlist = ['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',
       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',
       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 
       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price', 'MA4_listed_price','lag_1_gmv']

### Scale these variables using 'fit_transform'
home_km[varlist] = scaler.fit_transform(home_km[varlist])


# In[ ]:


home_km.head()


# In[ ]:


### Split the train dataset into X and y
from sklearn.model_selection import train_test_split
x = home_km.drop('gmv',axis=1)
y = home_km['gmv']

home_train_km = home_km


# In[ ]:


print("X = Independent variable & Y = Target variable")
print(x.shape,y.shape)


# In[ ]:


###  Instantiate
km = LinearRegression()

###  Fit a line
km.fit(x,y)


# In[ ]:


###  Coefficient values

coef = pd.DataFrame(x.columns)
coef['Coefficient'] = pd.Series(km.coef_)
coef


# In[ ]:


col = x.columns
col


# #### Model Building - Stepwise selection for feature selection

# In[ ]:


import statsmodels.api as sm  
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[ ]:


def stepwise_selection(x, y,
                       initial_list=['discount', 'sla','product_procurement_sla', 'order_payment_type',
       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',
       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 
       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price', 'MA4_listed_price', 'lag_1_gmv'],
                       threshold_in=0.01,threshold_out = 0.05, verbose=True):
    
    included = list(initial_list)
    while True:
        changed=False
        ### forward step
        excluded = list(set(x.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))
                
                
        ###backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included]))).fit()
        ###use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() ###null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included


# In[ ]:


import statsmodels.api as sm  

final_features = stepwise_selection(x, y)

print("\n","final_selected_features:",final_features)


# In[ ]:


### Import statsmodels
import statsmodels.api as sm  

### Subsetting training data for 15 selected columns
x_rfe1 = x[final_features]

x_rfe1 = sm.add_constant(x_rfe1)

### Fitting the model with selected variables
km1 = sm.OLS(y, x_rfe1).fit() 

print(km1.params)


# In[ ]:


print(km1.summary())


# In[ ]:


### Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()

vif['Features'] = x_rfe1.columns
vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


features = list(x_rfe1.columns)
features.remove('const')
features


# In[ ]:


### Model Evaluation on testing data
x_2 = x[features]


### Predicition with selected features on the test data
y_pred = km1.predict(sm.add_constant(x_2))


# In[ ]:


### Mean square error (MSE)
mse = np.mean((y_pred - y)**2)
mse


# In[ ]:


### Coefficient values

coef = pd.DataFrame(x_rfe1.columns)
coef['Coefficient'] = pd.Series(km.coef_)
coef


# In[ ]:


### Using K-Fold Cross validation evaluating on whole dataset

# lm = LinearRegression()
fold = KFold(10,shuffle = True, random_state = 100)

cv_scores = -cross_val_score(km,x[features],y,cv=fold,scoring='neg_mean_squared_error')

print("Neg. of MSE:",cv_scores,"\n")
print("Mean of 5 KFold CV - MSE:",cv_scores.mean())


# In[ ]:


def elasticity(model,x):
    
    features_df = pd.DataFrame(model.params)
    features_df = features_df.rename(columns={0:'coef'})
    
    features_df['imp_feature'] = model.params.index
    features_df = features_df[features_df.imp_feature != 'const'][['imp_feature','coef']]
    features_df.index = range(len(features_df))
#      features

    elasticity_list = list()
    
    for i in range(len(features_df)):
        elasticity_list.append(((features_df.coef[i] * np.mean(x[features_df.imp_feature[i]])) / np.mean(x['gmv'])))

    features_df['elasticity'] = np.round(elasticity_list,3)
    
    sns.barplot(x='elasticity',y='imp_feature',data=features_df)
    plt.show()
    
    return features_df
    


# In[ ]:


elasticity(km1,home_train_km)


# In[ ]:


# Plotting y and y_pred to understand the spread

fig = plt.figure()
plt.scatter(y, y_pred)
fig.suptitle('y vs y_pred', fontsize = 20)              # Plot heading 
plt.xlabel('y', fontsize = 18)                          # X-label
plt.ylabel('y_pred', fontsize = 16)  


# In[ ]:


# Figure size
plt.figure(figsize=(8,5))

# Heatmap
sns.heatmap(home_train_km[features].corr(), cmap="YlGnBu", annot=True)
plt.show()


# ### Distributed Lag Model

# In[ ]:


home_dlm = ha_week[['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',
       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',
       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 
       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price', 'MA4_listed_price','lag_1_listed_price','lag_1_discount_offer',
       'lag_2_discount_offer','lag_3_discount_offer','lag_2_NPS','lag_3_NPS','lag_1_Stock_Index',
       'lag_2_Stock_Index','lag_3_Stock_Index','lag_1_Special_sales','lag_2_Special_sales','lag_3_Special_sales',
       'lag_1_Payday','lag_2_Payday','lag_3_Payday']]           


home_dlm.head()


# In[ ]:


home_dlm['lag_1_gmv'] = home_dlm['gmv'].shift(-1)
home_dlm['lag_2_gmv'] = home_dlm['gmv'].shift(-2)
home_dlm['lag_3_gmv'] = home_dlm['gmv'].shift(-3)


# In[ ]:


### Checking NaN

home_dlm.isnull().sum()


# In[ ]:


home_dlm = home_dlm.fillna(0)


# In[ ]:


home_dlm.head()


# In[ ]:


###  Import the StandardScaler()
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

###  Create a scaling object
# scaler = StandardScaler()
scaler = MinMaxScaler()


###  Create a list of the variables that you need to scale
varlist = ['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',
       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',
       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 
       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price', 'MA4_listed_price','lag_1_listed_price','lag_1_discount_offer',
       'lag_2_discount_offer','lag_3_discount_offer','lag_2_NPS','lag_3_NPS','lag_1_Stock_Index',
       'lag_2_Stock_Index','lag_3_Stock_Index','lag_1_Special_sales','lag_2_Special_sales','lag_3_Special_sales',
       'lag_1_Payday','lag_2_Payday','lag_3_Payday']


###  Scale these variables using 'fit_transform'
home_dlm[varlist] = scaler.fit_transform(home_dlm[varlist])


# In[ ]:


home_dlm.head()


# In[ ]:


###  Split the train dataset into X and y
from sklearn.model_selection import train_test_split
x = home_dlm.drop('gmv',axis=1)
y = home_dlm['gmv']

home_train_dlm = home_dlm


# In[ ]:


print("X = Independent variable & Y = Target variable")
print(x.shape,y.shape)


# In[ ]:


###  Instantiate
dlm = LinearRegression()

###  Fit a line
dlm.fit(x,y)


# In[ ]:


###  Coefficient values

coef = pd.DataFrame(x.columns)
coef['Coefficient'] = pd.Series(dlm.coef_)
coef


# In[ ]:


col = x.columns
col


# #### Model Building - Stepwise selection for feature selection

# In[ ]:


import statsmodels.api as sm  
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[ ]:


def stepwise_selection(x, y,
                       initial_list=['discount', 'sla','product_procurement_sla', 'order_payment_type',
       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',
       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 
       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price', 'MA4_listed_price','lag_1_listed_price','lag_1_discount_offer',
       'lag_2_discount_offer','lag_3_discount_offer','lag_2_NPS','lag_3_NPS','lag_1_Stock_Index',
       'lag_2_Stock_Index','lag_3_Stock_Index','lag_1_Special_sales','lag_2_Special_sales','lag_3_Special_sales',
       'lag_1_Payday','lag_2_Payday','lag_3_Payday'],
                     threshold_in=0.01,threshold_out = 0.05, verbose=True):
    
    included = list(initial_list)
    while True:
        changed=False
        ###  forward step
        excluded = list(set(x.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))
                
                
        ###  backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included]))).fit()
        ###  use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() ###  null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included


# In[ ]:


import statsmodels.api as sm  

final_features = stepwise_selection(x, y)

print("\n","final_selected_features:",final_features)


# In[ ]:


###  Import statsmodels
import statsmodels.api as sm  

###  Subsetting training data for 15 selected columns
x_rfe1 = x[final_features]

x_rfe1 = sm.add_constant(x_rfe1)

###  Fitting the model with selected variables
dlm1 = sm.OLS(y, x_rfe1).fit() 

print(dlm1.params)


# In[ ]:


print(dlm1.summary())


# In[ ]:


###  Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()

vif['Features'] = x_rfe1.columns
vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


x_rfe1.drop('sla', axis = 1, inplace = True)


# In[ ]:


# Refitting with final selected variables
x_rfe1 = sm.add_constant(x_rfe1)

# Fitting the model with final selected variables
dlm1 = sm.OLS(y, x_rfe1).fit()   
print(dlm1.summary())


# In[ ]:


x_rfe1.drop('lag_1_discount_offer', axis = 1, inplace = True)


# In[ ]:


# Refitting with final selected variables
x_rfe1 = sm.add_constant(x_rfe1)

# Fitting the model with final selected variables
dlm1 = sm.OLS(y, x_rfe1).fit()   
print(dlm1.summary())


# In[ ]:


x_rfe1.drop('product_procurement_sla', axis = 1, inplace = True)


# In[ ]:


# Refitting with final selected variables
x_rfe1 = sm.add_constant(x_rfe1)

# Fitting the model with final selected variables
dlm1 = sm.OLS(y, x_rfe1).fit()   
print(dlm1.summary())


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()

vif['Features'] = x_rfe1.columns
vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


features = list(x_rfe1.columns)
features.remove('const')
features


# In[ ]:


###  Model Evaluation on testing data
x_2 = x[features]

###  Predicition with selected features on the test data
y_pred = dlm1.predict(sm.add_constant(x_2))


# In[ ]:


###  Mean square error (MSE)
mse = np.mean((y_pred - y)**2)
mse


# In[ ]:


###  Coefficient values

coef = pd.DataFrame(x_rfe1.columns)
coef['Coefficient'] = pd.Series(dlm.coef_)
coef


# In[ ]:


###  Using K-Fold Cross validation evaluating on whole dataset

# lm = LinearRegression()
fold = KFold(10,shuffle = True, random_state = 100)

cv_scores = -cross_val_score(dlm,x[features],y,cv=fold,scoring='neg_mean_squared_error')

print("Neg. of MSE:",cv_scores,"\n")
print("Mean of 5 KFold CV - MSE:",cv_scores.mean())


# In[ ]:


def elasticity(model,x):
    
    features_df = pd.DataFrame(model.params)
    features_df = features_df.rename(columns={0:'coef'})
    
    features_df['imp_feature'] = model.params.index
    features_df = features_df[features_df.imp_feature != 'const'][['imp_feature','coef']]
    features_df.index = range(len(features_df))
#      features

    elasticity_list = list()
    
    for i in range(len(features_df)):
        elasticity_list.append(((features_df.coef[i] * np.mean(x[features_df.imp_feature[i]])) / np.mean(x['gmv'])))

    features_df['elasticity'] = np.round(elasticity_list,3)
    
    sns.barplot(x='elasticity',y='imp_feature',data=features_df)
    plt.show()
    
    return features_df
    


# In[ ]:


elasticity(dlm1,home_train_dlm)


# In[ ]:


# Plotting y and y_pred to understand the spread

fig = plt.figure()
plt.scatter(y, y_pred)
fig.suptitle('y vs y_pred', fontsize = 20)              # Plot heading 
plt.xlabel('y', fontsize = 18)                          # X-label
plt.ylabel('y_pred', fontsize = 16)  


# In[ ]:


# Figure size
plt.figure(figsize=(8,5))

# Heatmap
sns.heatmap(home_train_dlm[features].corr(), cmap="YlGnBu", annot=True)
plt.show()


# ### Distributed + Multiplicated Model

# In[ ]:


home_dlmm = ha_week[['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',
       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',
       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 
       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price', 'MA4_listed_price','lag_1_listed_price','lag_1_discount_offer',
       'lag_2_discount_offer','lag_3_discount_offer','lag_2_NPS','lag_3_NPS','lag_1_Stock_Index',
       'lag_2_Stock_Index','lag_3_Stock_Index','lag_1_Special_sales','lag_2_Special_sales','lag_3_Special_sales',
       'lag_1_Payday','lag_2_Payday','lag_3_Payday']]           


home_dlmm.head()


# In[ ]:


home_dlmm['lag_1_gmv'] = home_dlmm['gmv'].shift(-1)
home_dlmm['lag_2_gmv'] = home_dlmm['gmv'].shift(-2)
home_dlmm['lag_3_gmv'] = home_dlmm['gmv'].shift(-3)


# In[ ]:


### Checking NaN

home_dlmm.isnull().sum()


# In[ ]:


home_dlmm = home_dlmm.fillna(0)


# In[ ]:


### Applying Log 
home_dlmm=np.log(home_dlmm)

home_dlmm = home_dlmm.fillna(0)
home_dlmm = home_dlmm.replace([np.inf, -np.inf], 0)


# In[ ]:


home_dlmm.head()


# In[ ]:


###  Import the StandardScaler()
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

###  Create a scaling object
# scaler = StandardScaler()
scaler = MinMaxScaler()


###  Create a list of the variables that you need to scale
varlist = ['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',
       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',
       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 
       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price', 'MA4_listed_price','lag_1_listed_price','lag_1_discount_offer',
       'lag_2_discount_offer','lag_3_discount_offer','lag_2_NPS','lag_3_NPS','lag_1_Stock_Index',
       'lag_2_Stock_Index','lag_3_Stock_Index','lag_1_Special_sales','lag_2_Special_sales','lag_3_Special_sales',
       'lag_1_Payday','lag_2_Payday','lag_3_Payday']


###  Scale these variables using 'fit_transform'
home_dlmm[varlist] = scaler.fit_transform(home_dlmm[varlist])


# In[ ]:


home_dlmm.head()


# In[ ]:


###  Split the train dataset into X and y
from sklearn.model_selection import train_test_split
x = home_dlmm.drop('gmv',axis=1)
y = home_dlmm['gmv']

home_train_dlmm = home_dlmm


# In[ ]:


print("X = Independent variable & Y = Target variable")
print(x.shape,y.shape)


# In[ ]:


###  Instantiate
dlmm = LinearRegression()

###  Fit a line
dlmm.fit(x,y)


# In[ ]:


###  Coefficient values

coef = pd.DataFrame(x.columns)
coef['Coefficient'] = pd.Series(dlmm.coef_)
coef


# In[ ]:


col = x.columns
col


# #### Model Building - Stepwise selection for feature selection

# In[ ]:


import statsmodels.api as sm  
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[ ]:


def stepwise_selection(x, y,
                       initial_list=['discount', 'sla','product_procurement_sla', 'order_payment_type',
       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',
       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 
       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price', 'MA4_listed_price','lag_1_listed_price','lag_1_discount_offer',
       'lag_2_discount_offer','lag_3_discount_offer','lag_2_NPS','lag_3_NPS','lag_1_Stock_Index',
       'lag_2_Stock_Index','lag_3_Stock_Index','lag_1_Special_sales','lag_2_Special_sales','lag_3_Special_sales',
       'lag_1_Payday','lag_2_Payday','lag_3_Payday'],
                     threshold_in=0.01,threshold_out = 0.05, verbose=True):
    
    included = list(initial_list)
    while True:
        changed=False
        ###  forward step
        excluded = list(set(x.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))
                
                
        ###  backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included]))).fit()
        ###  use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() ###  null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included


# In[ ]:


import statsmodels.api as sm  

final_features = stepwise_selection(x, y)

print("\n","final_selected_features:",final_features)


# In[ ]:


###  Import statsmodels
import statsmodels.api as sm  

###  Subsetting training data for 15 selected columns
x_rfe1 = x[final_features]

x_rfe1 = sm.add_constant(x_rfe1)

###  Fitting the model with selected variables
dlmm1 = sm.OLS(y, x_rfe1).fit() 

print(dlmm1.params)


# In[ ]:


print(dlmm1.summary())


# In[ ]:


###  Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()

vif['Features'] = x_rfe1.columns
vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


x_rfe1.drop('lag_2_Stock_Index', axis = 1, inplace = True)


# In[ ]:


# Refitting with final selected variables
x_rfe1 = sm.add_constant(x_rfe1)

# Fitting the model with final selected variables
dlmm1 = sm.OLS(y, x_rfe1).fit()   
print(dlmm1.summary())


# In[ ]:


x_rfe1.drop('lag_2_NPS', axis = 1, inplace = True)


# In[ ]:


# Refitting with final selected variables
x_rfe1 = sm.add_constant(x_rfe1)

# Fitting the model with final selected variables
dlmm1 = sm.OLS(y, x_rfe1).fit()   
print(dlmm1.summary())


# In[ ]:


###  Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()

vif['Features'] = x_rfe1.columns
vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


features = list(x_rfe1.columns)
features.remove('const')
features


# In[ ]:


###  Model Evaluation on testing data
x_2 = x[features]

###  Predicition with selected features on the test data
y_pred = dlmm1.predict(sm.add_constant(x_2))


# In[ ]:


###  Mean square error (MSE)
mse = np.mean((y_pred - y)**2)
mse


# In[ ]:


###  Coefficient values

coef = pd.DataFrame(x_rfe1.columns)
coef['Coefficient'] = pd.Series(dlmm.coef_)
coef


# In[ ]:


###  Using K-Fold Cross validation evaluating on whole dataset

# lm = LinearRegression()
fold = KFold(10,shuffle = True, random_state = 100)

cv_scores = -cross_val_score(dlm,x[features],y,cv=fold,scoring='neg_mean_squared_error')

print("Neg. of MSE:",cv_scores,"\n")
print("Mean of 5 KFold CV - MSE:",cv_scores.mean())


# In[ ]:


def elasticity(model,x):
    
    features_df = pd.DataFrame(model.params)
    features_df = features_df.rename(columns={0:'coef'})
    
    features_df['imp_feature'] = model.params.index
    features_df = features_df[features_df.imp_feature != 'const'][['imp_feature','coef']]
    features_df.index = range(len(features_df))
#      features

    elasticity_list = list()
    
    for i in range(len(features_df)):
        elasticity_list.append(((features_df.coef[i] * np.mean(x[features_df.imp_feature[i]])) / np.mean(x['gmv'])))

    features_df['elasticity'] = np.round(elasticity_list,3)
    
    sns.barplot(x='elasticity',y='imp_feature',data=features_df)
    plt.show()
    
    return features_df
    


# In[ ]:


elasticity(dlmm1,home_train_dlm)


# In[ ]:


# Plotting y and y_pred to understand the spread

fig = plt.figure()
plt.scatter(y, y_pred)
fig.suptitle('y vs y_pred', fontsize = 20)              # Plot heading 
plt.xlabel('y', fontsize = 18)                          # X-label
plt.ylabel('y_pred', fontsize = 16)  


# In[ ]:


# Figure size
plt.figure(figsize=(8,5))

# Heatmap
sns.heatmap(home_train_dlm[features].corr(), cmap="YlGnBu", annot=True)
plt.show()

