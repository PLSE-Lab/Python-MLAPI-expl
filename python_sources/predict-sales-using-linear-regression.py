#!/usr/bin/env python
# coding: utf-8

# # Getting Started

# In[ ]:


#import required libraries
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#reading data files
store_df=pd.read_csv("../input/rossmann-store-sales/store.csv")
train_df=pd.read_csv("../input/rossmann-store-sales/train.csv")


# # Getting to Know your Data

# Data fields
# 
# Most of the fields are self-explanatory. The following are descriptions for those that aren't.
# 
#     Id - an Id that represents a (Store, Date) duple within the test set
#     Store - a unique Id for each store
#     Sales - the turnover for any given day (this is what you are predicting)
#     Customers - the number of customers on a given day
#     Open - an indicator for whether the store was open: 0 = closed, 1 = open
#     StateHoliday - indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. Note that all schools are closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None
#     SchoolHoliday - indicates if the (Store, Date) was affected by the closure of public schools
#     StoreType - differentiates between 4 different store models: a, b, c, d
#     Assortment - describes an assortment level: a = basic, b = extra, c = extended
#     CompetitionDistance - distance in meters to the nearest competitor store
#     CompetitionOpenSince[Month/Year] - gives the approximate year and month of the time the nearest competitor was opened
#     Promo - indicates whether a store is running a promo on that day
#     Promo2 - Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating
#     Promo2Since[Year/Week] - describes the year and calendar week when the store started participating in Promo2
#     PromoInterval - describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store
# 

# In[ ]:


store_df.head()


# In[ ]:


store_df.describe()


# In[ ]:


#Checking the no. of NaN vales
store_df.isna().sum()


# In[ ]:


train_df.head()


# In[ ]:


train_df.describe()


# In[ ]:


#Checking the no. of NaN values
train_df.isna().sum()


# # Data Cleaning

# In[ ]:


#Merging both the Dataframes into one based on the "Store" ID
df=store_df.merge(train_df,on=["Store"],how="inner")
df.head()


# In[ ]:


#(rowsxcolumns) of the merged DataFrame
df.shape


# In[ ]:


#Checking the no. of NaN values
df.isna().sum()


# The columns - CompetitionOpenSinceMonth, CompetitionOpenSinceYear, Promo2SinceWeek, Promo2SinceYear, PromoInterval have too many values as NaN (roughly 30% or above).
# Whereas, the column CompetitionDistance has very few values missing, and these values can be substituted by the 'mode' of the very same column. 

# In[ ]:


#Dropping columns
df=df.drop(columns=["PromoInterval","Promo2SinceWeek","Promo2SinceYear","CompetitionOpenSinceMonth","CompetitionOpenSinceYear"])


# In[ ]:


#Handling NaN
df["CompetitionDistance"]=df["CompetitionDistance"].fillna(df["CompetitionDistance"].mode()[0])


# # Handling Outliers

# In[ ]:


#Find the range of data
plt.figure(figsize=(5,10))
sns.set(style="whitegrid")
sns.distplot(df["Sales"])


# In[ ]:


#Find the range of the data
plt.figure(figsize=(5,10))
sns.set(style="whitegrid")
sns.distplot(df["Customers"])


# In[ ]:


plt.figure(figsize=(10,10))
sns.set(style="whitegrid")
sns.boxenplot(data=df,scale="linear",x="DayOfWeek",y="Sales",color="orange")


# In[ ]:


plt.figure(figsize=(10,10))
sns.set(style="whitegrid")
sns.boxenplot(y="Customers", x="DayOfWeek",data=df, scale="linear",color="orange")


# This data, contains many outliers, but these might have been caused to the surge of customers during a festival or Holiday, or due to an effective promo.
# However I will cap off, the Customers at 3000, and Sales at 20,000.

# In[ ]:


df["Sales"]=df["Sales"].apply(lambda x: 20000 if x>20000 else x)
df["Customers"]=df["Customers"].apply(lambda y: 3000 if y>3000 else y)
print(max(df["Sales"]))
print(max(df["Customers"]))


# # Working With 'TIME'

# In[ ]:


df["Date"]=pd.to_datetime(df["Date"])
df["Year"]=df["Date"].dt.year
df["Month"]=df["Date"].dt.month
df["Day"]=df["Date"].dt.day
df["Week"]=df["Date"].dt.week%4
df["Season"] = np.where(df["Month"].isin([3,4]),"Spring",np.where(df["Month"].isin([5,6,7,8]), "Summer",np.where(df["Month"].isin ([9,10,11]),"Fall",np.where(df["Month"].isin ([12,1,2]),"Winter","None"))))
df


# Adding an additional feature, that records the no. of Holidays per week.

# In[ ]:


Holiday_Year_Month_Week_df=pd.DataFrame({"Holiday per week":df["SchoolHoliday"],"Week":df["Week"],"Month":df["Month"],"Year":df["Year"],"Date":df["Date"]})
Holiday_Year_Month_Week_df=Holiday_Year_Month_Week_df.drop_duplicates(subset=['Date'])
Holiday_Year_Month_Week_df=Holiday_Year_Month_Week_df.groupby(["Year","Month","Week"]).sum()
Holiday_Year_Month_Week_df


# In[ ]:


df=df.merge(Holiday_Year_Month_Week_df, on=["Year","Month","Week"],how="inner")


# Adding additional features, that records the avg. no. of Customers per month and avg. no. of Customers per week

# In[ ]:


customer_time_df=pd.DataFrame({"Avg CustomersPerMonth":df["Customers"],"Month":df["Month"]})
AvgCustomerperMonth=customer_time_df.groupby("Month").mean()
AvgCustomerperMonth


# In[ ]:


customer_time_df=pd.DataFrame({"Avg CustomersPerWeek":df["Customers"],"Week":df["Week"],"Year":df["Year"],"Month":df["Month"]})
AvgCustomerperWeek=customer_time_df.groupby(["Year","Month","Week"]).mean()
AvgCustomerperWeek


# In[ ]:


df=df.merge(AvgCustomerperMonth,on="Month",how="inner")
df=df.merge(AvgCustomerperWeek,on=["Year","Month","Week"],how="inner")


# adding an additional feature that records the no. of promo per week

# In[ ]:


promo_time_df=pd.DataFrame({"PromoCountperWeek":df["Promo"],"Year":df["Year"],"Month":df["Month"],"Week":df["Week"],"Date":df["Date"]})
promo_time_df=promo_time_df.drop_duplicates(subset=['Date'])
promo_time_df=promo_time_df.groupby(["Year","Month","Week"]).sum()
promo_time_df


# In[ ]:


df=df.merge(promo_time_df,on=["Year","Month","Week"], how="inner")


# # Handling Categorical Data

# The columns StoreType, Assortment, Season have char type or String type values, all of this need to converted to a numerical value

# In[ ]:


numerical_data_col=["Store","Competition Distance","Promo2","DayOfWeek","Sales","Customers","Open","SchoolHoliday","Year","Month","Day","Week"]
categorical_data_col=["StoreType","Assortment","Season"]


# In[ ]:


for i in categorical_data_col:
    p=0
    for j in df[i].unique():
        df[i]=np.where(df[i]==j,p,df[i])
        p=p+1

    df[i]=df[i].astype(int)


# In[ ]:


#The column StateHoliday contains 0,'0',a and b. This needs to be conerted to a pure numerical data column
df["StateHoliday"].unique()


# In[ ]:


df["StateHoliday"]=np.where(df["StateHoliday"] == '0' ,0,1)
df["StateHoliday"]=df["StateHoliday"].astype(int)


# # EDA

# ## Are the promos effective?

# In[ ]:


plt.figure(figsize=(10,10))
sns.set(style="whitegrid",palette="pastel",color_codes=True)
sns.violinplot(x="DayOfWeek",y="Sales",hue="Promo",split=True, data=df)


# The days promos were present have indeed shown a slight improvement in Sales.
# The plot above also shows that there was no promo offered on 6th and the 7th day of the week (Saturday and Sunday), and stores didn't suffer for doing so either, as it can be seen the no. of customers on the weekends, were more that that during the weekdays.

# In[ ]:


plt.figure(figsize=(10,10))
sns.set(style="whitegrid",palette="pastel",color_codes=True)
sns.violinplot(x="DayOfWeek",y="Customers",hue="Promo",split=True, data=df)


# ## Does competition distance matter?

# In[ ]:


sns.set(style="whitegrid")
g=sns.relplot(x="CompetitionDistance", y="Sales", hue="Promo", data=df)
g.fig.set_size_inches(15,15)


# Most of the stores, that have very less Competition Distance have still managed to make big Sales, by appling promo, as can be seen.

# ## Is there a surge of customers during SchoolHolidays?

# In[ ]:


sns.set(style="whitegrid")
g=sns.relplot(y="Avg CustomersPerWeek", x="Week", hue="Holiday per week", data=df)
g.fig.set_size_inches(10,10)


# It doesn't look like there is a big difference in the no. of customers even if there were 4 School Holidays that week

# ## Is there an increase in promo if it is a School Holiday?

# In[ ]:


sns.set(style="whitegrid")
g=sns.relplot(y="Holiday per week", x="Week", hue="PromoCountperWeek", data=df)
g.fig.set_size_inches(10,10)


# It doesn't seem like the Holidays had any effect on promo and Customers.

# # Final Check

# In[ ]:


df.head()


# In[ ]:


#Find Correlation between the data columns
plt.figure(figsize=(15,15))
sns.heatmap((df.drop(columns=["Date"]).corr()))


# the heatmap shows all our hypothesis were true, there is very little correlation between School Holiday, Customers and Promo, but there is a strong correlation between Promo and Sales

# # Using Linear Regression to predict 'Sales'

# In[ ]:


df=df.drop(columns=["Date"])
df.shape


# In[ ]:


#Splitting of data
features=df[["Customers","Open","Promo","Assortment","PromoCountperWeek","SchoolHoliday","StoreType","Week","Month"]]
features=preprocessing.scale(features)
target=df["Sales"]
X_train,X_test,Y_train,Y_test=train_test_split(features,target)


# In[ ]:


model1=LinearRegression()
model1.fit(X_train,Y_train)
print(model1.score(X_test,Y_test))

