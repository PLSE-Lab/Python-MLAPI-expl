#!/usr/bin/env python
# coding: utf-8

# ## DataSet

#  Features of the Melbourne House Price Data:
#  
# * Suburb: Suburb
# * Address: Address
# * Rooms: Number of rooms
# * Price: Price in Australian dollars
# * Method: S - property sold; SP - property sold prior; PI - property passed in; PN - sold prior not disclosed; SN - sold not disclosed; NB - no bid; VB - vendor bid; W - withdrawn prior to auction; SA - sold after auction; SS - sold after auction price not disclosed. N/A - price or highest bid not available.
# * Type:  h - house,cottage,villa, semi,terrace; u - unit, duplex; t - townhouse; 
# * SellerG: Real Estate Agent
# * Date: Date sold
# * Distance: Distance from CBD in Kilometres
# * Regionname: General Region (West, North West, North, North east ...etc)
# * Propertycount: Number of properties that exist in the suburb.
# * Bedroom2 : Scraped # of Bedrooms (from different source)
# * Bathroom: Number of Bathrooms
# * Car: Number of carspots
# * Landsize: Land Size in Metres
# * BuildingArea: Building Size in Metres
# * YearBuilt: Year the house was built
# * CouncilArea: Governing council for the area
# * Lattitude: Self explanitory
# * Longtitude: Self explanitory
# 

# ## Context

# The aim of this notebook is to apply some exploratory analysis and Visualize the data to find out the features which affect the house price. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


# Import the rquired library

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

import cufflinks as cf
cf.go_offline()


# In[ ]:


# load the required data

home_data=pd.read_csv("../input/melbourne-housing-snapshot/melb_data.csv")
home_data.head() # observations


# In[ ]:


# shape(Number of rows and columns) of the data

home_data.shape


# ### Basic features of the data

# Based on the information extracted from the below code, there are total 21 attributes included in this dataset, out of which 8 attributes are categorical and remaining are numerical and float type.

# In[ ]:


home_data.info()


# ### Categorical attributes

# In[ ]:


print("Categorical variable : ", home_data.select_dtypes(include=["O"]).columns.to_list())


# ### Numerical Attributes

# In[ ]:


print("Numerical variable : ", home_data.select_dtypes(exclude=["O"]).columns.to_list())


# ### Explore the data

# From the description below, we can see that the average price of the houses is $1075684 and  average  3 rooms are available in the house.
# Based on the data we can say that the oldest house was built in 1196 while the newest house built in 2018.In the summary we also have description of postcode which should be a categorical variable, so we need to convert postcode into categorical variable.

# In[ ]:


# Lets describe the datasets
home_data.describe().T


# In[ ]:


# Convert the "Postcode" into categorical variable

home_data["Postcode"]=home_data["Postcode"].astype("category")

home_data.info()


# In[ ]:



# check for duplicate columns.Datasets have "Rooms" and "Bedroom2" features which could be duplicate.

home_data["room"]=home_data["Rooms"]-home_data["Bedroom2"]

home_data


# We can see there is little difference between "Rooms" and "Bedroom2". Thus, we can remove "Bedroom2" feature from the datasets.

# In[ ]:


# drop the selected columns

home_data.drop(["Bedroom2","room"],axis=1,inplace=True)
home_data.columns


# ## Missing Data

# From the below code, we learn that there are four variable whose values are missing. We can drop those features whose values are missing but droping the variable might removes the lot of useful information,  so we can drop the missing values.

# In[ ]:


# Check for missing value(counts)

home_data.isnull().sum().sort_values(ascending=False)


# In[ ]:


# percentage of missing value

home_data.isnull().sum()/len(home_data)*100


# In "BuildingArea" and "YearBuilt" features, many values are missing, so we need to remove those values so that it will not affect in our downstream analysis

# In[ ]:


# Handle the missing value

# fill the missing value in "Car" variable

home_data.fillna(value={"Car": 0},inplace=True)

# Drop the missing value in other column

home_data.dropna(inplace=True)
home_data.info()


# Based on the description after removing missing values reveals that minimum of Landsize and BuildingArea is 0 which seems odd. We need further analysis. 

# In[ ]:


home_data.describe().T


# In[ ]:


#  house with BuildingArea==0 

home_data[home_data["BuildingArea"] ==0]


# In[ ]:


# Drop the rows where BuildingArea is 0 which is not possible for any house to be 0 size. It acts as an outlier. 

home_data=home_data[home_data["BuildingArea"] !=0]


# In[ ]:


# House with Landsize = 0
home_data[home_data["Landsize"]==0]


# As there are lots of houses with landsizes 0.It could be indicative of that the houses are very near to edge of the property line so might want to keep this data.

# ## Exploratory Analysis

# In this dataset we are trying to understand the price of the houses and how other features are affecting the price, so we will look for variables on which the price is dependent.  

# ### Univariate Analysis

# The house price is seems to be normally distributed but positively skewed.Based on the observation we can say that majority of the houses are around 1M dollar, while some are around 8M-9M dollar, which could be outliers.

# In[ ]:


# Analyse the price variable

plt.style.use("ggplot")
plt.figure(figsize=(10,6))

sns.distplot(home_data["Price"],kde=False,hist_kws=dict(edgecolor="k"))
plt.title("House Price Distribution In Melbourne",size=16);


# From the below plot we can see that now price is normally distributed.

# In[ ]:


## Log transformation of price variable 
plt.style.use("classic")
plt.figure(figsize=(10,6))
sns.distplot(np.log(home_data["Price"]),kde=False)

plt.title("Distribution of Log Tranformed Price ", size=16);


# ### Bivariate Analysis

# In[ ]:


home_data.columns


# ### Analysis of Price VS Suburb

# In[ ]:


# Price analysis based on suburb 
# Prepare the data

suburb_Price=home_data.groupby("Suburb",as_index=False)["Price"].mean().sort_values(by="Price",ascending=False).reset_index(drop=True)
suburb_Price.rename(columns={"Price":"AveragePrice"},inplace=True)
suburb_Price.head(20)


# Suburb "Canterbury", "Malvern", "Middle Park", "Brighton", "Ivanhoe East", "Princes Hill", these are the costliest Suburb. 

# In[ ]:


# Top 20 costliest Suburb


fig=px.bar(suburb_Price.head(20),x="AveragePrice",y="Suburb",color="Suburb",title="Top 20 Costliest Suburb",text="AveragePrice",orientation="h",height=800,width=900)
fig.update_traces(textposition="inside")
fig.update_layout(plot_bgcolor='rgb(193,255,193)')
fig.show()


# In[ ]:


# Top 20 least costlier Suburb

fig=px.bar(suburb_Price.tail(20),x="AveragePrice",y="Suburb",color="Suburb",title="Top 20 Least Costliest Suburb",text="AveragePrice",orientation="h",height=800,width=900)
fig.update_traces(textposition="inside")
fig.update_layout(plot_bgcolor='rgb(275, 270, 273)')
fig.show()


# Average number of rooms available in houses based on Costliest Suburb.By observing the data, i can  say that in average 3-4 rooms availble per house in most of the costliest suburb region. 

# In[ ]:


# Rooms vs Suburb
# Prepare the data

rooms_suburb=home_data.groupby("Suburb")[["Rooms","Price"]].mean().sort_values(by="Price",ascending=False).reset_index()
rooms_suburb.head()


# In[ ]:


# Average number of rooms available per house in Costliest Suburb

rooms_suburb[["Suburb","Rooms"]].head(20).iplot(kind="bar",x="Suburb",title="Average Number of Rooms ")


# PropertyCount is same as Suburb, It tells us the number of houses in each suburb.

# In[ ]:


# Property count Vs Price

fig=px.scatter(home_data,x="Propertycount",y="Price",color="Type",title="Price Distribution Vs PropertyCount")
fig.show()


# ### Price analysis Vs Rooms

# Price of the houses is also depends on the availability of rooms in the house. so lets see what our data says.

# In[ ]:


# Analyse the price as per Availability of the number of rooms
# Prepare the data

room_data=home_data.groupby("Rooms")["Price"].mean().sort_values(ascending=False).reset_index()
room_data


# Observations reveals that prices of the house is depend on the number of rooms available in the house, means variable price increases when the number of room increases. We can say that there is linear relationship between price and the rooms. 
# Observations:
# 
# * Price is highest at 5 rooms.
# * Price increases as number of room increases
# * There are some outliers
# 

# In[ ]:



plt.figure(figsize=(14,6))
sns.boxplot(x="Rooms",y="Price",data=home_data)
plt.title('Price analysis Vs Rooms',size=16);


# We can clearly see that, as the number of rooms increases in the house, price variable also increases.

# In[ ]:


# Rooms Vs Price along with regression 

sns.lmplot(x="Rooms",y="Price",hue="Type",data=home_data)
plt.title("Price Vs Rooms",size=15);


# ### Average Price trend By date

# In[ ]:


# Convert the Date variable into datetime object

home_data["Date"]=pd.to_datetime(home_data["Date"])

# add Year column to home_data
home_data["Year"]=home_data.Date.dt.year


# #### Lets see Price distribution over the year.
#  Price Analysis over last 10 year, 
# since the newest home built in 2018 so we will look for the price since 2008. We can cleary see that there is gradual increase in house price from 2013 to 2016. Since 2016, melbourne housing has been cooled off. 

# In[ ]:


# Price Distibution Over last 10 year

year_data=home_data[home_data["YearBuilt"]>2008]

fig=px.box(year_data,x="YearBuilt",y="Price",title="Price Distribution Over Last 10 Years")
fig.show()


# There could be number of possible reasons, why the prices of the house has been down. Lets further explore some other variable

# ### Price Vs Regionname

# Data reveals that most of people Bought house in  region named "Southern Metropolitan", "Northern Metropolitan","Western Metropolitan", and "Eastern Metropolitan". Out of this "Southern Metropolitan" region is costlier than others. Among most preferred region,  "Northern Metropolitan" and "Western Metropolitan" region are less expensive. 

# In[ ]:


# Different Regionname and which region are preferred more 

home_data.Regionname.value_counts()


# In[ ]:


# Price of the house based on the different region

home_data.groupby("Regionname")["Price"].mean().sort_values(ascending=False)


# In[ ]:


# 
plt.figure(figsize=(14,7))

sns.boxplot(x="Regionname",y="Price",data=home_data)
plt.xticks(rotation=45)
plt.title("Price Distribution Over Different Region",size=15);


# ### Price Vs Type of House

# In[ ]:


# Analyse Price against Type of house
print("Type of the houses in melbourne : ",home_data.Type.value_counts().count())


# In[ ]:


# 
print("Count of h type house  : ",home_data.Type.value_counts()[0])
print("Count of u type house  : ",home_data.Type.value_counts()[1])
print("Count of t type house  : ",home_data.Type.value_counts()[2])


# In[ ]:


plt.figure(figsize=(8,6))
sns.set(style="darkgrid")
sns.countplot(x="Type",data=home_data)
plt.title("Type of House in Melbourne",size=15);


# In[ ]:


# Lets explore how price varyies with different type of house
print("Average Price for the h - house,cottage,villa, semi,terrace : $%.f "  % home_data.groupby("Type").Price.mean()[0])
print("Average Price for the u - unit, duplex  : $%.f " % home_data.groupby("Type").Price.mean()[2])
print("Average Price for the t - townhouse : $%.f " % home_data.groupby("Type").Price.mean()[1])


# Based on the observations we can say that, most of people liked cottage and villa type house, and average price of the "h" type house is also high compare to other two. While Buyers also preferred to buy duplex type of houses, whose Avearge Price is less compare to "h" and "t" type of house. House Price is cleary dependednt on type of house.

# In[ ]:


# price Distribution across Type of house

fig=px.box(home_data,y="Type",x="Price",title="Price Distribution Vs Type of House",orientation="h")
fig.show()


#  Based on the finding, in last 2-3 year the avearge price of the cottage and villa type house prices drop quite low compared to their prices.There are ups and downs in prices in 2018 especially for township and villa properties. 

# In[ ]:


# Price Distribution Over last 10 Year Based On Type of House

fig=px.box(year_data,y="Price",x="YearBuilt",color="Type",title="Price Distribution Over last 10 Year Based On Type of House")
fig.show()


# ## Analysis Based on the Property Sold or Unsold

# Type of method we in our data:
# 
# Method: 
# * S - property sold; 
# * SP - property sold prior; 
# * PI - property passed in;
# * VB - vendor bid; 
# * SA - sold after auction; 
# 
# There is not much difference between price distribution and the properties sold by each of the methods as we can in the graph below. But there is extreme Price observe in VB and PI. VB is vendor bid properties, which is at their highest Price, so no one can bid for higher. 

# In[ ]:


print("Number of property Sold: ",home_data.Method.value_counts()[0])
print("Number of property Sold prior: ",home_data.Method.value_counts()[1])
print("Number of property Passed in: ",home_data.Method.value_counts()[2])
print("Number of Vendor Bid: ",home_data.Method.value_counts()[3])
print("Number of property Sold after auction: ",home_data.Method.value_counts()[4])


# In[ ]:


# Analysis based on the property sold
 # Prepare the data
df=home_data[(home_data["Method"]!="PI")]
df1=df[(df["Method"]!="VB")]


# In[ ]:


# Number of property sold 
plt.figure(figsize=(8,6))
sns.set(style="darkgrid")
sns.countplot(x="Method",data=df1)
plt.title("Number of Property Sold in Melbourne",size=15);


# In[ ]:


# Number of property Unsold

unsold=home_data[home_data["Method"]!="S"]
unsold1=unsold[unsold["Method"]!="SP"]
unsold2=unsold1[unsold1["Method"]!="SA"]

# Count plot

plt.figure(figsize=(8,6))
sns.set(style="darkgrid")
sns.countplot(x="Method",data=unsold2)
plt.title("Number of Property Unsold in Melbourne",size=15);


# In[ ]:


# Distribution of Price against the Method

fig,ax=plt.subplots(figsize=(12,6))
sns.violinplot(y="Price",x="Method",data=home_data,ax=ax);
plt.title("Distribution of Price Vs Property Sold or Unsold",size=19);


# In[ ]:


# Distribution of price over the last 10 year across all the method

fig,ax=plt.subplots(figsize=(14,8))
sns.violinplot(x="YearBuilt",y="Price",hue="Method",data=year_data,ax=ax);
plt.title("Distribution of Price Over Last 10 years Vs Method ", size=16);


# ## Analysis of Price Vs Distance

# This is clear from the data that  house Price is depends on the Distance From CBD, which means that if the distance from CBD is less, Price of the house will be more. I can say that there is negative correlation between price and CBD distance.

# In[ ]:


# Relationship between Price and distnace From CBD
dist=home_data[home_data["Distance"]>0]
plt.figure(figsize=(12,8))
sns.scatterplot(x="Distance",y="Price",data=dist)
plt.title("Distance From CBD and Price Anlaysis",size=16);


# Observations based on the below graph, we can say that why the house price of the specific region such as "Southern Metropolitan", "Northern Metropolitan", "Western Metropolitan" and "Eastern Metropolitan" region is high. All these region, whose price is higher compare to other region, are very near to CBD and this is why price is high for those region. Price depends on the distance from CBD. 

# In[ ]:


# Price analysis based on distance from CBD and regionname
fig= px.scatter(home_data,x="Distance",y="Price",color="Regionname",title="Price Vs Distance")
fig.show()


# From regression line, it is clear that as the Distnace from CBD increases, house price decreases.

# In[ ]:


sns.lmplot(x="Distance",y="Price",data=home_data, x_estimator=np.mean)
plt.title("Price Vs Distance",size=14);


# ### Price Vs CouncilArea

# Based on the following observation, Price depends on CouncilArea. House Price in some of CouncilArea such as "Port Philip","Stonnington","Boroondara","Whitehorse","GlenEira","Bayside","Manningham", is high compare to other council area. These area also comes under "Southern Metropolitan" and "Eastern MetroPolitan" region and these two region are near to CBD. Councilare, regionname, distance from CBD, all together are the reason for the price in this Area. 

# In[ ]:


# Distribution of Price Vs CouncilArea

fig,ax=plt.subplots(figsize=(14,9))
ax=sns.boxplot(y="CouncilArea",x="Price",data=home_data,whis=np.inf)
ax.set_title("CouncilArea Vs Price Distribution",size=18);


# In[ ]:


# CouncilArea and Region

home_data.groupby(["Regionname","CouncilArea"])["Price"].mean().reset_index()


# ### Analyse Building Area

# The House Prices increases with increase in Building Area.

# In[ ]:


## BuildingArea Vs Price

sns.lmplot(x="BuildingArea",y="Price",hue="Regionname",data=home_data)
plt.title("Building Area and Price Analysis",size=14);


# ## Landsize Vs Price 

# Price is lineraly dependent on Landsize of cottage and villa properties, while for township, price is less likely to increase with increase in Landsize. Here exactly we can not say that Price is strongly correlated with Landsize but it actually depends on the type of houses and region.

# In[ ]:


# Landsize Vs Price Analysis
sns.lmplot(x="Landsize",y="Price",hue="Type",data=home_data)
plt.title("Price Vs Landsize",size=15);


# In[ ]:


# Landsize Vs Price across different region

sns.scatterplot(x="Landsize",y="Price",hue="Regionname",data=home_data)
plt.title("Price Vs Landsize",size=15);


# In[ ]:


# Analysis of Price Vs  selected features
data=home_data[["Rooms","Distance","Bathroom","Landsize","YearBuilt","Car","Date","Propertycount"]]
price=home_data["Price"]
fig=plt.figure(figsize=(15,20))
for i in range(len(data.columns)):
    fig.add_subplot(3,3,i+1)
    sns.scatterplot(x=data.iloc[:,i],y=price)
plt.tight_layout()
plt.show()


# ### Correlation Among Different Attributes

# Lets explore how variable are correlated with one another

# In[ ]:


# Prepare the data for correlation

corr=home_data.corr()

# Relation between different variable
fig,ax=plt.subplots(figsize=(14,9))
sns.heatmap(corr,annot=True,cmap = 'coolwarm',linewidth = 1,annot_kws={"size": 11})
plt.title("Correlation Among Different Variable",size=15);


# #### Insights:
# * Positive Correlation between price and Number of rooms in house
# * House Price moderately increases with increase in Building Area
# * Negative correlation between Price and Distance From CBD
# * moderate relationship between Bathroom and Price
# * Bathroom, rooms, and BuildingArea are all related to one another
# 
# 
#     

# ### Common type of house and rooms and bathroom available Regionwise

# #### Southern Metropolitan
# In Southern Metropolitan region, duplex and villa house with 2 ,3 or 4 rooms and 1 or 2 bathrooms are more common.

# In[ ]:


common=home_data.groupby(['Regionname','Type','Rooms','Bathroom'])['Price'].count().reset_index()
SM=common[common["Regionname"]=="Southern Metropolitan"].sort_values(by="Price",ascending=False)


# #### Northern Metropolitan:
# 
# In Northern Metropolitan region, duplex and villa properties with 2,3 or 4 rooms and 1 or 2 bathrooms are more common type of house.

# In[ ]:


NM=common[common["Regionname"]=="Northern Metropolitan"].sort_values(by="Price",ascending=False)
NM.head(10)


# In[ ]:


# Common type of house, rooms and number of bathroom in Eastern Region

EM=common[common["Regionname"]=="Eastern Metropolitan"].sort_values(by="Price",ascending=False)
EM.head(10)


# In[ ]:


# Reasonable Price for 2 Bedroom unit 

two_rooms=home_data.groupby(["Regionname","Type","Rooms","Bathroom"]).Price.median().reset_index()
two_rooms[two_rooms["Rooms"]==2].sort_values(by="Price").reset_index(drop=True)


# # Summary

# ##### Type:
# * h= House cottage or villa type, category has the costliest Property. Since this is cottage or villa so it makes sense that this type is the costliest one.
# * t= township, category has very less number of houses 
# * u=duplex, this category has lowest price of the house, and this type of houses are also located at the prime location of melbourne.
# 
# ##### Distance:
# * Properties with high prices lie in low distance from Central District.
# * There is no correlation between Landsize and Price but Price for specific type of houses depends on the landsize such as for villa propertes, house price increased with increased Landsize. 
# 
# ##### Rooms:
# * There is moderate relationship between Number ofRooms available in the house with House Price and it also makes sense that if number of rooms increases, price also increases along with it.
# 
# ##### Regionname/CouncilArea/Suburb:
# * Since each suburb comes under Council Area and council area comes under Region. Metropolitan Region are costlier region compare to all victoria region.
# 
# ##### Price:
# * Over the last 2-3 year the house price has been cooled down, even the maximum price for villa property in 2017 was $1.1M.
# 
# ##### Factors affecting the House Price based on the overall analysis:
# * Rooms
# * Distance
# * Suburb
# * CouncilArea
# * Type
# * Bathroom
# * BuildingArea
# * Regionname
# * Landsize based on the type of house
# 
# All these features can be used for predicting House price in Melbourne.

# #### Thankyou!!
