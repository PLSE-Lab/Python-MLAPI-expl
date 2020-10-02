#!/usr/bin/env python
# coding: utf-8

# # Melboune house price prediction,growth area identification and modeling (linear, random tree forest and logistic) Layout

# ### Team members: Mehul Haria, Naureen Pethani and Raymond (shanhua) Huang

# ### 1. Data structure. It inclues data cleaning against data type, outliers, null value.
# 
# ##### 1.1 Exploring data, such select_dtypes().columns, describe(), info() and shape()
# 
# ##### 1.2 Changing data type, date type and catogory type
# 
# ##### 1.3 Dealing with null value, mainly replacing null in price with mode using fillna()
# 
# ##### 1.4 Finding outliers
# 
# 
# ### 2. Data presentation and prediction 7 section. (27239 records)
# 
# ##### 2.1 Year vs price per type
# 
# ##### 2.2 Price prediction for 2019
# 
# ###### 2.2.1 Predict price for unit in S Metro 
# 
# ###### 2.2.1 Predict price for unit in E Metro 
# 
# ##### 2.3 Season vs price 
# 
# ##### 2.4 Region vs price change and growth rate vs year
# 
# ###### 2.4.1 Region price change vs year per type
# 
# ###### 2.4.2 Feature engineering to get count growth and price growth against year per region and type
# 
# ###### 2.4.2.1 Price growth rate over 0.05 table per region and type
# 
# ###### 2.4.2.2 Top 20 region per type with largest count
# 
# ###### 2.4.2.3 Count growth rate over 0.2 table per region and type
# 
# ###### 2.4.2.4 Actual count over 1000 for each year pet type and region
# 
# ##### 2.5 Other features method, distance, rooms, car vs price
# 
# ###### 2.5.1 Method vs Price
# 
# ###### 2.5.2 Rooms # impact on Price VS Year
# 
# ###### 2.5.3 Distance vs Price
# 
# ###### 2.5.4 Car spot vs Price
# 
# ##### 2.6 Ideal house type for top 10 region per type max count of sales
# 
# ##### 2.6.1 Top 10 house type in S Metro with different rooms and bathrooms by count
# 
# ##### 2.6.2 Top 10 house type in all regions per type with different rooms and bathrooms by count
# 
# ##### 2.6.3 Top 10 units in E Metro with different rooms and bathrooms by count
# 
# ##### 2.7 Heat map for relationships
# 
# ### 3. Data linear modeling to see which variable contribute most (20394 records)
# 
# ##### 3.1 Prepare the dataset and label for training models, include removing all null value, get_dummies of region, change type into numeric
# 
# ##### 3.2 Getting coefficient
# 
# ##### 3.3 Random forest model
# 
# ##### 3.4 Model comparison get score function
# 
# 
# ### 4. Performance evaluation  sample size vs machine learning
# 
# ##### 4.1 Define fuction to get correlation, refine results and storage value
# 
# ##### 4.2 Learning curve : Plot training scores against validation score
# 
# ### 5. Conclusion
# 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection  import train_test_split
import numpy as np
from scipy.stats import norm # for scientific Computing
from scipy import stats, integrate
import matplotlib.pyplot as plt


# In[ ]:


melbourne_data  = pd.read_csv("../input/melbourne-housing-market/Melbourne_housing_FULL.csv")


# In[ ]:


melbourne_data.head(10)


# # 1.Exploring Data and data cleaning

# # 1.1 Exploring data

# In[ ]:


melbourne_data.shape


# In[ ]:


melbourne_data.info()


# Looking at the data information above, we can see that non-numerical data is being considered as object. The list included following columns: 'Suburb', 'Address', 'Type', 'Method', 'SellerG', 'Date', 'CouncilArea','Regionname'
# In next few steps we will be changing object data types to categorical and date data types

# In[ ]:


## verifying columns with object data type
print(melbourne_data.select_dtypes(["object"]).columns)


# # 1.2 Changing data type

# In[ ]:


##changing all object data types to category - This step is necessary to be able to plot categorical data for our analysis
objdtype_cols = melbourne_data.select_dtypes(["object"]).columns
melbourne_data[objdtype_cols] = melbourne_data[objdtype_cols].astype('category')


# In[ ]:


melbourne_data.info()


# In[ ]:


## looking at data information above, we can notice that "Data" is also converted to category. 
## In this step we will cast date to datetime
melbourne_data['Date'] =  pd.to_datetime( melbourne_data['Date'])


# In[ ]:


## the following command suggests that all our data types are now in required format
melbourne_data.info()


# 
# For next few steps we will be doing data prepration for Numerical Feature Variables

# In[ ]:


## describe command will give us all statstical information about our numeric variables
melbourne_data.describe().T


# Observing above information abount numerical data, it can be noticed that Postcode is also being treated as numerical data. Since we know that Postcode is a catergorical data, we will be casting it to category

# In[ ]:


melbourne_data["Postcode"] = melbourne_data["Postcode"].astype('category')


# In[ ]:


melbourne_data.describe().T


# 
# After carefully evaluating data, it can be noticed that variables "Rooms" and "Bedroom2" are pretty much similar and one of the columns should be removed to avoid duplication of data

# In[ ]:


## in this step we will first confirm our above statemnt by obesrving "Rooms" and "Bedroom2"

melbourne_data['b 2 r'] = melbourne_data["Bedroom2"] - melbourne_data["Rooms"]
melbourne_data[['b 2 r', 'Bedroom2', 'Rooms']].head()


# In[ ]:


## We can see that the difference is very minimal here that is will be wise to remove one of the 2 columns
melbourne_data = melbourne_data.drop(['b 2 r', 'Bedroom2'], 1)


# # 1.3 Working with missing data
# 

# 
# There are multiple ways that we can use to explore missisng data. Here we will be using a visual way first to get some hint. In later step we will do some calculations to get exact number of missing data from each variables. Based on data, our experience and business need we will either fill in missing values or we will drop rows or columns having null values

# In[ ]:


## visualizing missing values
fig, ax = plt.subplots(figsize=(15,7))
sns.heatmap(melbourne_data.isnull(), yticklabels=False,cmap='viridis')


# From the visual above, it can be concluded that we have few missing data in Price, Bathroom, Car and Landsize, Longitude and Latitude columns. There are so many missing values in Building Area and Year Built features. In next step let's explore the count of missing values

# In[ ]:


# Percentage of missing values
melbourne_data.isnull().sum()/len(melbourne_data)*100


# 
# From the information above, we can notice that few feature varaibles still have large percentage of missing values. At this point we are ignoring it, but at later state if we will take those as our feature variables for oour model, we will explore ways to fill in those information or to remove those from our data.

# In[ ]:


melbourne_data = melbourne_data.drop(["Landsize", "BuildingArea", "YearBuilt"], axis=1)


# In[ ]:


## Also since our target variable is price, it makes sense to drop rows for Price columns wher price values are missing
melbourne_data.dropna(subset=["Price"], inplace=True)


# In[ ]:


#from sklearn.preprocessing import Imputer
#X=melbourne_data[['Bathroom','Car']]
#imp=Imputer(missing_values='NaN',strategy='median',axis=0)
#imp.fit(X)
#X=pd.DataFrame(data=imp.transform(X),columns=X.columns)
#melbourne_data[['Bathroom','Car']]=X


# In[ ]:


melbourne_data['Car']=melbourne_data['Car'].fillna(melbourne_data['Car'].mode()[0])
melbourne_data['Bathroom']=melbourne_data['Bathroom'].fillna(melbourne_data['Bathroom'].mode()[0])


# In[ ]:


melbourne_data.shape


# In[ ]:


# Percentage of missing values
melbourne_data.isnull().sum()/len(melbourne_data)*100


# # 1.4 Finding Outliers
# 

# Outliers can significantly impact data analysis and can also impact normalization of data. It is very important in during data prepration to indentify them remove them. In next few steps we will work in our data to get rid of outliers (if any)

# In[ ]:


melbourne_data.describe().T


# From the statstical summary above we can see that max price in our data is nearly $11.2 million. That looks like a clear outlier. But before removing it, lets first ensure that we have very few values in that range.

# In[ ]:


## to findout outliers lets divide data into different price ranges to identify number of occurences of data in different price ranges
melbourne_data['PriceRange'] = np.where(melbourne_data['Price'] <= 100000, '0-100,000',  
                                       np.where ((melbourne_data['Price'] > 100000) & (melbourne_data['Price'] <= 1000000), '100,001 - 1M',
                                                np.where((melbourne_data['Price'] > 1000000) & (melbourne_data['Price'] <= 3000000), '1M - 3M',
                                                        np.where((melbourne_data['Price']>3000000) & (melbourne_data['Price']<=5000000), '3M - 5M',
                                                                np.where((melbourne_data['Price']>5000000) & (melbourne_data['Price']<=6000000), '5M - 6M',
                                                                        np.where((melbourne_data['Price']>6000000) & (melbourne_data['Price']<=7000000), '6M - 7M',
                                                                                np.where((melbourne_data['Price']>7000000) & (melbourne_data['Price']<=8000000), '7M-8M', 
                                                                                         np.where((melbourne_data['Price']>8000000) & (melbourne_data['Price']<=9000000), '8M-9M', 
                                                                                                 np.where((melbourne_data['Price']>9000000) & (melbourne_data['Price']<=10000000), '9M-10M', 
                                                                                                         np.where((melbourne_data['Price']>10000000) & (melbourne_data['Price']<=11000000), '10M-11M', 
                                                                                                                 np.where((melbourne_data['Price']>11000000) & (melbourne_data['Price']<=12000000), '11M-12M', '')
                                                                                                                 ))))))))))


# In[ ]:


melbourne_data.groupby(['PriceRange']).agg({'PriceRange': ['count']})


# By exploring above table, it can be concluded that:
# - 1 data item in the range 0-100,00 
# - 2 data item in range 7M - 8M
# - 1 data item in range 8M - 9M
# - 1 data item in range 11M - 12M
# For the purpose of this study, let us drop rows that match above mentioned conditions

# In[ ]:


melbourne_data.info()


# In[ ]:


melbourne_data.describe().T


# In[ ]:


melbourne_data.drop(melbourne_data[(melbourne_data['PriceRange'] == '0-100,000') |
                                   (melbourne_data['PriceRange'] == '7M-8M') |
                                   (melbourne_data['PriceRange'] == '8M-9M') |
                                   (melbourne_data['PriceRange'] == '11M-12M')].index, inplace=True)


# In[ ]:


melbourne_data.describe().T


# In[ ]:


melbourne_data.groupby(['Rooms'])['Rooms'].count()


# In[ ]:


melbourne_data.drop(melbourne_data[(melbourne_data['Rooms'] == 12) | 
                                   (melbourne_data['Rooms'] == 16)].index, inplace=True)


# In[ ]:


melbourne_data.describe().T


# In[ ]:


##sns.distplot(melbourne_data, kde=False, bins=20).set(xlabel='Price');
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
##melbourne_data.select_dtypes(include = numerics)
melbourne_data.select_dtypes(include = numerics).hist(bins=15, figsize=(15, 6), layout=(2, 4))


# In[ ]:


melbourne_data['Distance'] = round(melbourne_data['Distance'])


# In[ ]:


melbourne_data.shape


# 27239 record will be used for presenting

# # 2.0 Data presentation and relationship

# # 2.1 The first factor we look at is the price versus year and season. Then predict price using linear function for 2019 and 2020.

# # 2.1.1 Price trend against year per house type

# In[ ]:


## extract year from date
melbourne_data['Year']=melbourne_data['Date'].apply(lambda x:x.year)
melbourne_data.head(5)


# In[ ]:


#data subset by type
#house price
melbourne_data_h=melbourne_data[melbourne_data['Type']=='h']
#condo price
melbourne_data_u=melbourne_data[melbourne_data['Type']=='u']
#townhouse price
melbourne_data_t=melbourne_data[melbourne_data['Type']=='t']
#house,condo and town house price groupby year and mean
melbourne_data_h_y=melbourne_data_h.groupby('Year').mean()
melbourne_data_u_y=melbourne_data_u.groupby('Year').mean()
melbourne_data_t_y=melbourne_data_t.groupby('Year').mean()
melbourne_data_h_y.head()


# In[ ]:


#sns.lmplot(x="Year", y="Price", hue="Type", data=melbourne_data,  x_estimator=np.mean);
melbourne_data_h_y['Price'].plot(kind='line', color='r',label='House')
melbourne_data_u_y['Price'].plot(kind='line', color='g',label='Condo')
melbourne_data_t_y['Price'].plot(kind='line', color='b',label='Townhouse')
year_xticks=[2016,2017,2018]
plt.ylabel('Price')
plt.xticks( year_xticks)
plt.title('Melboune price trend Vs Year per type')
plt.legend()


# House price was going down dramatically by 100,000 units
# Condo price climb up slowly while Townhouse price kept steady.
# From this graph, it is anticipated that house price will keptgoing down but less slope
# Townhouse price will kepte unchanged
# Condo price will increase.
# To developer, it is time to built more condos in 2019.House budget need to be cut
# For home buyers, it is time to buy house in 2019.

# ###### 2.2 Predicting house price for all types in South Metro, units in South Metro and units in East Melbourne for 2019 and 2020

# In[ ]:


melbourne_data.shape


# In[ ]:


melbourne_data.columns


# In[ ]:



melbourne_data_South_M=melbourne_data[melbourne_data['Regionname']=='Southern Metropolitan']
melbourne_data_South_M_average=melbourne_data_South_M.groupby(['Year'])['Price'].mean()
# Series.to_frame()


# In[ ]:


# create X and y

X = melbourne_data_South_M[[ 'Year']]
y = melbourne_data_South_M[['Price']]

# instantiate and fit
lm2 = LinearRegression()
lm2.fit(X, y)

# print the coefficients
print (lm2.intercept_)
print (lm2.coef_)


# In[ ]:



### STATSMODELS ###

# you have to create a DataFrame since the Statsmodels formula interface expects it
X_new = pd.DataFrame({'Year': [2019,2020,2021]})

# predict for a new observation
lm2.predict(X_new)


# From this rough approximation, 2019 average price will be 1557639, 1630019 for 2020 for all types in melboune

# # 2.2.1 Predict prcie for S Metro unit

# In[ ]:


melbourne_data_SM=melbourne_data[melbourne_data['Regionname']=='Southern Metropolitan']
melbourne_data_SM_u=melbourne_data_SM[melbourne_data_SM['Type']=='u']
melbourne_data_SM_u.shape


# In[ ]:



### STATSMODELS ###

# create a fitted model
lm1 = smf.ols(formula='Price ~ Year', data=melbourne_data_SM_u).fit()

# print the coefficients
lm1.params


# In[ ]:



# you have to create a DataFrame since the Statsmodels formula interface expects it
X_new = pd.DataFrame({'Year': [2016,2017,2018,2019,2020,2021]})

# predict for a new observation
lm1.predict(X_new)


# In[ ]:


lm1.rsquared


# For Condo in S Metro region, price will be aroud 795272 dollars with lower R2

# ###### 2.2.2 Predicting prcie for E Metro unit

# In[ ]:



melbourne_data_E=melbourne_data[melbourne_data['Regionname']=='Eastern Metropolitan']
melbourne_data_E_u=melbourne_data_E[melbourne_data_E['Type']=='u']

lme = smf.ols(formula='Price ~ Year', data=melbourne_data_E_u).fit()

# print the coefficients
lme.params


# In[ ]:


melbourne_data_E_u.shape


# In[ ]:


X_new = pd.DataFrame({'Year': [2016,2017,2018,2019,2020,2021]})

# predict for a new observation
lme.predict(X_new)


# For E Metro, price grow 10% from 2016 to 2017 and 7.7% from 2018 to 2019 can be expected. though count is less compared to southern region
# 

# ##### 2.3 Seasonal performance

# In[ ]:


#get month information from date 
#df['year_month']=df.datetime_column.apply(lambda x: str(x)[:7])
#per = df.Date.dt.to_period("M")
# How many calls, sms, and data entries are in each month?
#data.groupby(['month', 'item'])
#df['birthdate'].groupby([df.birthdate.dt.year, df.birthdate.dt.month]).agg('count')
melbourne_data['Month']=pd.DatetimeIndex(melbourne_data['Date']).month

#lois[_y_m]=lois['Price'].groupby(['Month']).mean()
#Prepare data for pie chart to check sales based on month in order to see which month sell most.
melbourne_data_2016=melbourne_data[melbourne_data['Year']==2016]
melbourne_data_2017=melbourne_data[melbourne_data['Year']==2017]
melbourne_data_2018=melbourne_data[melbourne_data['Year']==2018]
melbourne_data_2016_count=melbourne_data_2016.groupby(['Month']).count()
melbourne_data_2017_count=melbourne_data_2017.groupby(['Month']).count()
melbourne_data_2018_count=melbourne_data_2018.groupby(['Month']).count()
Comparison={2016:melbourne_data_2016.shape,2017:melbourne_data_2017.shape,2018:melbourne_data_2018.shape}
Comparison


# In[ ]:


label_2016=['January','March','April','May','June','July','August','September','October','November','December']
plt.pie(melbourne_data_2016_count['Price'],labels=label_2016,autopct='%.1f %%')
plt.title('Year 2016')
plt.show()


# In[ ]:


label_2017=['January','February','March','April','May','June','July','August','September','October','November','December']
plt.pie(melbourne_data_2017_count['Price'],labels=label_2017,autopct='%.1f %%')
plt.title('Year 2017')


# In[ ]:


label_2018=['January','February','March','June','October']
plt.pie(melbourne_data_2018_count['Price'],labels=label_2018,autopct='%.1f %%')
plt.title('Year 2018')


# In general, it looks like that winters in 2016 an 2017 have the least sales count. that means house sales will favors more from May to November. Year 2018 seems a lot of missing data and date shape only one third compard to the others thus it is hard to make conclusion. 

# # 2.4 Region versus Price

# In[ ]:



# Abbreviate Regionname categories for presentation
melbourne_data['Regionabb'] = melbourne_data['Regionname'].map({'Northern Metropolitan':'N Metro',
                                            'Western Metropolitan':'W Metro', 
                                            'Southern Metropolitan':'S Metro', 
                                            'Eastern Metropolitan':'E Metro', 
                                            'South-Eastern Metropolitan':'SE Metro', 
                                            'Northern Victoria':'N Vic',
                                            'Eastern Victoria':'E Vic',
                                            'Western Victoria':'W Vic'})


# # 2.4.1 Region price change vs year per type
# 
# In General, East, north, south and west Metro are popular area based on sales count.
# 
# Next look at price against region per type
# 

# In[ ]:



sns.lmplot(x="Year", y="Price",hue="Type", data=melbourne_data,col='Regionabb', x_estimator=np.mean,col_wrap=2)
plt.ylim(200000, 2000000)
plt.xlim(2015,2020)


# In[ ]:


#South region price change vs year per type
sns.lmplot(x="Year", y="Price",hue="Type", data=melbourne_data[melbourne_data['Regionabb']=='S Metro'], x_estimator=np.mean);


# In general, Townhouse in E Metro, Condo in East Metro, House in S Metro, Condo in S Metro and Condo in N Metro are growing each year.

# In[ ]:


# East region price change vs year one type
melbourne_data_S=melbourne_data[melbourne_data['Regionabb']=='S Metro']
sns.lmplot(x="Year", y="Price", data=melbourne_data_S[melbourne_data_S['Type']=='u'], x_estimator=np.mean);


# Get mean value for price for each year, region per type

# ##### 2.4.2 Feature engineering to get count growth and price growth against year per region and type

# In[ ]:


Pct_change=melbourne_data.groupby(['Year','Regionabb','Type'],as_index=False)['Price'].mean()
Pct_change = Pct_change.sort_values(['Regionabb', 'Type','Year']).set_index(np.arange(len(Pct_change.index)))

Pct_change.info()


# In[ ]:


melbourne_data_count_region_y=melbourne_data.groupby(['Year','Regionabb','Type'],as_index=False)['Price'].count()
melbourne_data_count_region_y = melbourne_data_count_region_y.sort_values(['Regionabb', 'Type','Year']).set_index(np.arange(len(melbourne_data_count_region_y.index)))
melbourne_data_count_region_y.rename(columns={'Price':'Count'}, inplace=True)


# In[ ]:


# define fucntion to get year growth rate again price per region and type
def PCTM(gg):
    df=pd.DataFrame(gg['Price'].pct_change())
    df['Year']=gg['Year']
    df['region']=gg['Regionabb']
    df['Type']=gg['Type']
    df=df[df['Year']!=2016]
    return df


# ##### 2.4.2.1 Price growth rate over 0.05 table per region and type

# In[ ]:


#df2[df2['id'].isin(['SP.POP.TOTL','NY.GNP.PCAP.CD'])]
melboune_growthrate_y_t=PCTM(Pct_change)
melboune_growthrate_y_t1=melboune_growthrate_y_t[melboune_growthrate_y_t['region'].isin(['N Metro','S Metro','E Metro','SE Metro','W Metro','S Metro'])]
melboune_growthrate_y_t1.rename(columns={'Price':'Price Growth Rate'}, inplace=True)
melboune_growthrate_y_t1[melboune_growthrate_y_t1['Price Growth Rate']>0.05]


# Due to a lot of missing data in 2018, S Metro units has 8.7% in 2017 and 2.7% in 2018. If 2017 was chozen to look at growth in price, units and townhouse in E Metro,SE Metro and S Metro, townhouse in W Metro and house in S Metro present positive growth over 5%. looking at 2018, it seems people shift to buy more from SE or E Metro from S Metro. But this shift was not that significant yet in count as it showed in later table. Trend is there.

# In[ ]:


Sales_count=melbourne_data.groupby(['Regionabb'])['Price'].count()
Sales_count.head(10)


# ##### 2.4.2.2 Top 20 region per type with largest count

# In[ ]:


Sales_count=melbourne_data.groupby(['Regionabb','Type'])['Price'].count()
Sales_count.nlargest(20)


# In[ ]:



            # define fucntion to get year growth rate again count per region and type
def PCTMC(gg):
    df=pd.DataFrame(gg['Count'].pct_change())
    df['Year']=gg['Year']
    df['region']=gg['Regionabb']
    df['Type']=gg['Type']
    df=df[df['Year']!=2016]
    return df


# ##### 2.4.2.3 Count growth rate over 0.2 table per region and type

# In[ ]:


#df2[df2['id'].isin(['SP.POP.TOTL','NY.GNP.PCAP.CD'])]
melboune_growthrate_y_c=PCTMC(melbourne_data_count_region_y)
melboune_growthrate_y_c1=melboune_growthrate_y_c[melboune_growthrate_y_c['region'].isin(['N Metro','S Metro','E Metro','SE Metro','W Metro','S Metro'])]

melboune_growthrate_y_c1.rename(columns={'Count':'Count Growth Rate'}, inplace=True)
melboune_growthrate_y_c1[melboune_growthrate_y_c1['Count Growth Rate']>0.2]


# ##### 2.4.2.4 Actual count over 1000 for each year pet type and region 

# In[ ]:


melboune_count1=melbourne_data_count_region_y[melbourne_data_count_region_y['Regionabb'].isin(['S Metro','E Metro','SE Metro','W Metro','S Metro','N Metro'])]
melboune_count1[melboune_count1['Count']>1000]


# From above information count growth percentage and acutal sales count by year, South metro and N Metro seems to be the area where people tend to pay more and buy more, but as price kept going up, those live in south try to move to E and SE Metro

# To Conclude this section:
# 
# 1.Regarding house, S Metro has over 5 % in price growth in 2017 and 4718 sales count in three years, ranking second place among all.In terms of Units/Condo, S Metro has 2782 in sales count ranking No.4 and 8.7 % in price growth in 2017.
# 
# 2.Units and townhouses in E Metro and SE MEtro has great potential tough now they dont have such attraction as S Metro. They have count growth over 100% and price growth rate over 8%.

# ##### 2.5 Suplots of other numeric features v price

# ###### 2.5.1 Method vs Price

# In[ ]:


sns.boxplot(x = 'Method', y = 'Price', data = melbourne_data)
plt.show()
#Sold method did not affect price


# ###### 2.5.2 Rooms # impact on Prcie VS Year

# In[ ]:


sns.lmplot(x="Year", y="Price", hue="Rooms", data=melbourne_data,  x_estimator=np.mean);


# 2.5.3 Distance vs Price

# Insights: Increase distance reduce price

# In[ ]:


sns.lmplot(x="Distance", y="Price", data=melbourne_data, x_estimator=np.mean);


# ###### 2.5.4 Car spot vs Price

# In[ ]:


sns.lmplot(x="Car", y="Price", data=melbourne_data, x_estimator=np.mean);


# # 2.6 Ideal house type

# In this section, Count will be used to find out leads about the best sales type in all region.

# ###### 2.6.1 Top 10 house type in S Metro with different rooms and bathrooms by count

# In[ ]:




Ideal_House=melbourne_data.groupby(['Regionabb','Type','Rooms','Bathroom'])['Price'].count()


Ideal_House.loc[['S Metro'],'h'].nlargest(10)


# Insights: In S Metro, house with 3 rooms and 1 or 2 bathroom and house with 4 rooms and 2 bathrooms have the most sales among all.

# ###### 2.6.2 Top 10 house type in all regions per type with different rooms and bathrooms by count

# In[ ]:


Ideal_House.nlargest(10)


# Insights:North metro house was the most favorable types among those and W Metro follow behind. In general, house is more favorable than other types. South Metro condo with 2 bed rooms and 1 bathroom was listed as top 3 in sales count.

# ###### 2.6.3 Top 10 units in E Metro with different rooms and bathrooms by count

# In[ ]:


Ideal_House.loc[['E Metro'],'u'].nlargest(10)


# The best Condo type in East region Metropolitan is 2 rooms with one bed room.

# ###### 2.7 Heat map for presenting relationship

# In[ ]:


corrmat=melbourne_data.corr()


# In[ ]:


fig,ax=plt.subplots(figsize=(12,10))
sns.heatmap(corrmat,annot=True,annot_kws={'size': 12})


# Define function to refine those correlation more than 0.3 with abs value

# In[ ]:


#define function to refine those correlation more than 0.3 with abs value
def getCorrelatedFeature(corrdata,threshold):
    feature=[]
    value=[]
    
    for i, index in enumerate(corrdata.index):
        if abs(corrdata[index])>threshold:
            feature.append(index)
            value.append(corrdata[index])
    df=pd.DataFrame(data=value,index=feature,columns=['Corr Value'])
    return df


# In[ ]:


threshold=0.4
corr_value=getCorrelatedFeature(corrmat['Price'],threshold)
corr_value


# Rooms and Bathroom has highest correlation with house price compared to other factors.

# # 3.0 Linear model for coefficient to present impact for 'Rooms','Distance', 'Bathroom', 'Car', 'Year', 'Propertycount','Type_Code',region

# # 3.1 Prepare the dataset and label for training models, include removing all null value, get_dummies of region, change type into numeric

# In[ ]:


melbourne_data.isnull().sum()


# In[ ]:


melbourne_data['Type_Code'] = melbourne_data['Type'].map({'h':3,
                                            't':2, 
                                            'u':1, 
                                            'dev site':0, 
                                            'o res':0, 
                                            'br':0})


# Used get_dummies to change category data into numeric date. Region will be expressed in numerics

# In[ ]:


# Group Regionname categories 
melbourne_data1 = pd.get_dummies(melbourne_data['Regionabb'],drop_first=False)
melbourne_data=pd.concat([melbourne_data,melbourne_data1],axis=1)
melbourne_data.columns.values


# In[ ]:



#fig,ax=plt.subplots(figsize=(12,10))
#df=melbourne_data[['Price','Rooms','Distance', 'Bathroom',  'Year', 'Type_Code','RegionCode']]
#sns.heatmap(df,annot=True)
#dff=melbourne_data[['Price','Rooms','Distance', 'Bathroom', 'Car', 'Year', 'Propertycount','Type_Code',]].groupby('RegionCode')
#dff.head()


# In[ ]:


melbourne_data_NN=melbourne_data[['Rooms','Distance', 'Bathroom', 'Car', 'Year', 'Propertycount','Type_Code','N Metro','W Metro','S Metro','E Metro','SE Metro','N Vic','E Vic','W Vic','Price']].dropna()
melbourne_data_NN[['Rooms','Distance', 'Bathroom', 'Car', 'Year', 'Propertycount','Type_Code','N Metro','W Metro','S Metro','E Metro','SE Metro','N Vic','E Vic','W Vic','Price']].isnull().sum()


# In[ ]:


melbourne_data_NN.shape


# 27236 records will applied to the model

# In[ ]:


#Finding coefficient

X=melbourne_data_NN[['Rooms','Distance', 'Bathroom', 'Car', 'Year', 'Propertycount','Type_Code','N Metro','W Metro','S Metro','E Metro','SE Metro','N Vic','E Vic','W Vic']]
y=melbourne_data_NN['Price']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .20, random_state=5)


# In[ ]:


# Fit
# Import model
from sklearn.linear_model import LinearRegression

# Create linear regression object
regressor = LinearRegression()

# Fit model to training data
regressor.fit(X_train,y_train)


# In[ ]:



# Predict
# Predicting test set results
y_pred = regressor.predict(X_test)


# In[ ]:


regressor.score(X_test,y_test)


# In[ ]:



from sklearn import metrics
print('MAE:',metrics.mean_absolute_error(y_test,y_pred))
print('MSE:',metrics.mean_squared_error(y_test,y_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# In[ ]:


print('R^2 =',metrics.explained_variance_score(y_test,y_pred))


# In[ ]:


plt.scatter(y_test, y_pred)


# In[ ]:


# Histogram of the distribution of residuals
sns.distplot((y_test - y_pred))


# ##### 3.2 Getting coefficient

# In[ ]:


cdf = pd.DataFrame(data = regressor.coef_, index = X.columns, columns = ['Coefficients'])
cdf


# Before using fillna.mode
# Every unit increase in those features will:
# Rooms is linked to an increase in Price by 170624.6
# Distance is linked to an decrease in Price by 44797
# Bathroom is linked to an increase in Price by 185170
# Car space is linked to an increase in Price by 38890
# Year is linked to an increase in Price by 44305
# Type code is associated with increase in Price by 271838
# 
# After using fillna
# Every unit increase in those features will:
# Rooms is linked to an increase in Price by 217253.4
# Distance is linked to an decrease in Price by 42651.8
# Bathroom is linked to an increase in Price by 141730.5
# Car space is linked to an increase in Price by 37952
# Year is linked to an increase in Price by 29458.7
# Type code is associated with increase in Price by 235465
# 
# Type code, Rooms and bathroom are very important in house price. With limited landsize, the more rooms and bathroom, the higher the price is.
#  
# S Metro, SE Metro seems to be linked to an increase in price

# In[ ]:





# # 3.3 Random tree forest model for coefficient to present impact for 'Rooms','Distance', 'Bathroom', 'Car', 'Year', 'Propertycount','Type_Code',region

# In[ ]:


X.head()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
#model=RandomForestClassifier(n_estimators=20)
#model.fit(X_train,y_train)


# In[ ]:


clf=RandomForestClassifier(n_jobs=2,random_state=0)
clf.fit(X,y)


# In[ ]:


clf.predict(X)


# In[ ]:


clf.score(X_test,y_test)


# In[ ]:





# In[ ]:





# In[ ]:





# # 3.4 Model comparison

# In[ ]:


from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
folds=StratifiedKFold(n_splits=3)


# In[ ]:



def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


# In[ ]:


print(get_score(LogisticRegression(solver='liblinear',multi_class='ovr'), X_train, X_test, y_train, y_test))


# In[ ]:





# In[ ]:


print(get_score(LinearRegression(), X_train, X_test, y_train, y_test))


# Linear has better score

# In[ ]:





# In[ ]:





# In[ ]:





# # 4.0 Performance evaluation

# Using define function to evaluate different input threshold for correlation value
# 
# Explore relationships between input sample size and machine learning scores
# 
# Only correlation more than 0.4 was selected below. so Rooms and Bathroom fall into this category

# ##### 4.1 Define fuction to get correlation, refine results and storage value
# 

# In[ ]:


correlated_data=melbourne_data_NN[corr_value.index]
correlated_data.head()


# In[ ]:


corr_value.index


# In[ ]:


sns.pairplot(correlated_data)
plt.tight_layout()


# In[ ]:


sns.heatmap(correlated_data.corr(),annot=True,annot_kws={'size':12})


# Only extract data with high relations to price

# In[ ]:


X1=correlated_data.drop(labels=['Price'],axis=1)
y1=correlated_data['Price']
X1.head()


# In[ ]:


X1_train,X1_test,y1_train,y1_test=train_test_split(X1,y1,test_size=0.2,random_state=0)


# In[ ]:


X1_train.shape,X1_test.shape


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error


# In[ ]:


model = LinearRegression()
model.fit(X1_train,y1_train)


# In[ ]:


y1_predict=model.predict(X1_test)


# In[ ]:


y1_predict,y1_test


# In[ ]:


df=pd.DataFrame(data=[y1_predict,y1_test])
df.T.head(5)


# In[ ]:


from sklearn.metrics import r2_score


# In[ ]:


score=r2_score(y1_test,y1_predict)
mae=mean_absolute_error(y1_test,y1_predict)
mse=mean_squared_error(y1_test,y1_predict)
print("r2_score", score)
print("mae", mae)
print("mse", mse)


# In[ ]:


#store feature performance
total_features=[]
total_features_name=[]
selected_correlation_value=[]
r3_score=[]
mae_value=[]
mse_value=[]


# In[ ]:


def performance_metrics(features, th, y_true,y_pred):
    score=r2_score(y_true,y_pred)
    mae=mean_absolute_error(y_true,y_pred)
    mse=mean_squared_error(y_true,y_pred)
    
    total_features.append(len(features)-1)
    total_features_name.append(str(features))
    selected_correlation_value.append(th)
    r3_score.append(score)
    mae_value.append(mae)
    mse_value.append(mse)
    
    metrics_dataframe=pd.DataFrame(data=[total_features_name, total_features,selected_correlation_value,r3_score,mae_value,mse_value],index=['Features name','Total features','corre value','r2 score','mae','mse'])
    return metrics_dataframe.T


# In[ ]:


def get_y_predict(corrdata):
    X=corrdata.drop(labels=['Price'],axis=1)
    y=corrdata['Price']
    
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
    model=LinearRegression()
    model.fit(X_train,y_train)
    y_predict=model.predict(X_test)
    return y_predict
    


# In[ ]:





# In[ ]:


th5=0.4
corr_value=getCorrelatedFeature(corrmat['Price'],th5)
correlated_data=melbourne_data_NN[corr_value.index]
y_predict=get_y_predict(correlated_data)
performance_metrics(correlated_data.columns,th5,y_test,y_predict)


# In[ ]:





# 
# ##### 4.2 Learning curve : Plot training scores against validation score

# In[ ]:


#Ploting learning curves
from sklearn.model_selection import learning_curve, ShuffleSplit


# In[ ]:


def plot_learning_curve(estimator,title,X,y,ylim=None,cv=None,n_jobs=None,train_sizes=np.linspace(0.1,1.0,10)):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes,train_scores,test_scores=learning_curve(estimator,X,y,cv=cv,n_jobs=n_jobs,train_sizes=train_sizes)
    
    train_scores_mean=np.mean(train_scores,axis=1)
    train_scores_std=np.std(train_scores,axis=1)
    test_scores_mean=np.mean(test_scores,axis=1)
    test_scores_std=np.std(test_scores,axis=1)
    
    plt.grid()
    
    plt.fill_between(train_sizes,train_scores_mean - train_scores_std,train_scores_mean+train_scores_std,alpha=0.1,color="r")
    plt.fill_between(train_sizes,test_scores_mean - test_scores_std,test_scores_mean+test_scores_std,alpha=0.1,color="g")
    plt.plot(train_sizes,train_scores_mean,'o-',color="r",label="Training score")
    plt.plot(train_sizes,test_scores_mean,'o-',color="g",label="Cross-validation score")
    
    plt.legend(loc="best")
    return plt

X=correlated_data.drop(labels=['Price'],axis=1)
y=correlated_data['Price']

title="learning curves (linear regression)" + str(X.columns.values)
cv=ShuffleSplit(n_splits=100,test_size=0.2,random_state=0)

estimator=LinearRegression()
plot_learning_curve(estimator,title,X1,y1,ylim=(0.7,1.01),cv=cv,n_jobs=-1)

plt.show()


# It looks like training examples between 9 000 and 15 000 will give less change to the model.

# # 5.CONCLUSION

# 
# 
# Based on all 4 sentions, some finding and conclusion can be found as follow:
# 
# 1.From 2.1.1 section, house price in general will go down and Condo/units price will go up. that means investing in Condo will be better.
# 
# 
# 2.Regarding house, S Metro has over 5 % in price growth in 2017 and 4718 sales count in three years, ranking second place among all.In terms of Units/Condo, S Metro has 2782 in sales count ranking No.4 and 8.7 % in price growth in 2017 from Section 2.4.
# 
# 3.Units and townhouses in E Metro and SE MEtro has great potential tough now they dont have such attraction as S Metro. They have count growth over 100% and price growth rate over 8% from Section 2.4.
# 
# 4.From 2.6 ideal house type, we can see uints/condo in Southern Metro was ranked no.3. Though house in North and West Metro has the most count but their price is going down as showed in section 2.4 sns.lmplot. In terms of counts, Southern condo/units has great market potential as its count rank No.3 and price kept increasing by years presented in section 2.4. Southen house also has great market and rank no.4 after its unit in same region. But the price kept dropping compared to units in same region.
# 
# 5.From section 3 and 4, though the score is low due to low coefficient values, Southern Metro show No.2 biggest coefficient compared to other regions. E vic coefficient is high but is low in count thus can be neglected. That means Southern Metro has the great market in price.
# 
# 
# 6.Linear model has some limitations as if more rooms does equal to higher price and more sales count. But its coefficient can help to understand which factor has great impact. If price was only output dependent variables, the conclusion will be biased and not looking at the whole pictures. But if slaes count and price be presented as output variables, a clear picture will be clear. On the other hand, data is data, the perfect r2 sometimes doesn't mean the great insights of a business. The combination of business unstanding and data can present more real insights of business. The dataset miss great amount of value in 2018 and it results in that count cannot be used as model output. Only price can be output variable. So Cout in past three years and count in 2016 to 2017 will provide more leads for market.
# 
# 7.Continue improvement: 
# Due to limited timeframe, some work need to done to improve linear relationship and reduce r2 value:
# 
# a.replace null value with median to see if r2 drops (Yes, it drops by 0.01 and reduce effect of increasing unit against price regarding bathroom, car spots; reduce training score and vailidation score by 0.02 and validation score is always under training score by 0.01 throughout 1400 samples). so the code is there in 1.3 but not in effect and more investigation using different strategy will be applied.
# 
# b.different regressor or K-Fold model will be applied to reduced r2; 
# 
# c.more feature engineering against sales count will be explored, such as count growth in region against year.
# 
# 8.On the basis of conclusion 1-4, Units with 2 room and one bathroom in Southern Metro will be recommended to investor or home developer as it has 955 count in 3 years, ranking No.3 in counts and price kept increaing steadily 6 %. It is safe area for conservative investment agaist the unstable market. 
# 
# 
# 
# Great Thanks to the team work!

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




