#!/usr/bin/env python
# coding: utf-8

# <h1 style="text-align:center;color:red;">Melbourne Housing Dataset</h1>

# # Introduction

# ## Hello Everyone

# * This notebook is based on **Melbourne Housing Snapshot dataset**.
# * This dataset is a snapshot of a dataset created by `Tony Pino` which was scraped from publicly available results posted every week from Domain.com.au.
# * Without wasting any time , let's start this exercise.

# <img src="https://i.ytimg.com/vi/7lUM3DVKqhc/maxresdefault.jpg" style="width:800px;height:400px;">

# # Import Libraries

# In[ ]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns

#Plotly
import plotly.express as px
import plotly.graph_objs as go

#Some styling
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

import plotly.io as pio
pio.templates.default = 'presentation'

#Subplots
from plotly.subplots import make_subplots


#Showing full path of datasets
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
# Disable warnings 
import warnings
warnings.filterwarnings('ignore')


# # Import Dataset

# In[ ]:


df = pd.read_csv("/kaggle/input/melbourne-housing-snapshot/melb_data.csv")


# In[ ]:


df.head()


# # Data Description

# Features of the Melbourne House Price Data:
# 
# * **Suburb**: An outlying district of a city
# * **Address**: Address of the property.
# * **Rooms**: Number of rooms
# * **Method**: S - property sold; SP - property sold prior; PI - property passed in; PN - sold prior not disclosed; SN - sold not disclosed; NB - no bid; VB - vendor bid; W - withdrawn prior to auction; SA - sold after auction; SS - sold after auction price not disclosed. N/A - price or highest bid not available.
# * **Type**: h - house,cottage,villa, semi,terrace; u - unit, duplex; t - townhouse;
# * **SellerG**: Real Estate Agent
# * **Date**: Date sold
# * **Distance**: Distance from CBD(Central Business District)
# * **Regionname**: General Region (West, North West, North, North east ...etc)
# * **Propertycount**: Number of properties that exist in the suburb.
# * **Bedroom2** : Scraped # of Bedrooms (from different source)
# * **Bathroom**: Number of Bathrooms
# * **Car**: Number of carspots
# * **Landsize**: Land Size
# * **BuildingArea**: Building Size
# * **YearBuilt**: Year the house was built
# * **CouncilArea**: Governing council for the area
# * **Lattitude**: Self explanatory
# * **Longtitude**: Self explanatory

# ### Shape

# In[ ]:


df.shape


# * Our dataset consists of 13580 rows and 21 columns
# * Let's look at the columns.

# ### Columns

# In[ ]:


df.columns


# In[ ]:


#Firstly I will change some column names to make them more meaningful
# It is not required but it helps me in better analysis.

df.rename(columns={"Type":"Property_Type",
                  "Method":"Method_Sold",
                  "Distance":"Distance_CBD",
                  "Car":"Carspots",
                  "Date":"Date_Sold"},inplace=True)


# In[ ]:


#After changing few column names
df.columns

#Now we have some meaningful names


# ### Info

# In[ ]:


#Let's look at the info again

df.info()


# * Above information tells us that we have 
# <ul>
#     <li>Float :- 12</li>
#     <li>Int :- 1</li>
#     <li>Object :- 8</li>
# </ul>
# * We also have some datetime features `Date_Sold` and `YearBuilt`.
# * `YearBuilt` is in float datatype , that needs to be converted to int datatype as year cannot have float values.
# * Total 13 numerical variables and 8 categorical variables
# * We can see `BuildingArea`,`YearBuilt`,`Carspots` and `CouncilArea` have some missing values. 

# ### Description

# In[ ]:


df.describe().T


# * Our target feature `Price` has average value 1075684.
# * One thing to notice here is some features have minimum value as zero.
# * For discrete features ,like `Bedroom2`,`Bathroom` and `Carspots` we can have minimum value as zero.
# * But can `Landsize` and `Building area` be zero? Let's find out
# 

# ## Can Landsize be zero?

# * Land size being zero mean it is a **Zero-Lot-Line house**.
# * A zero-lot-line house is a piece of residential real estate in which the structure comes up to, or very near to, the edge of the property line. 
# * Rowhouses, garden homes, patio homes, and townhomes are all types of properties that may be zero-lot-line homes. 
# * They may be attached (as in a townhouse) or a detached single story or multistory residence.
# * So yes , landsize of a house/property can be zero .

# ### Skewness

# In[ ]:


#Let's check the skewness of our features

df.skew()


# * `Landsize` and `BuildingArea` are highly skewed features in our dataset.
# * `YearBuilt` , `Lattitude`,`Longitude` are negatively skewed variables as their skewness is less than zero.
# * Rest of the features are positively skewed.
# * Skewness may also indicate that there are outliers in highly skewed data , but the data may also be skewed with no outliers .
# * We'll have to check.

# ### Missing Values

# In[ ]:


df.isna().sum().sum()


# * There are total 13256 missing values in our dataset

# ### Missing values in each feature

# In[ ]:


df.isna().sum()


# * We have four features with missing values 
# * Carspots
# * BuildingArea
# * YearBuilt
# * CouncilArea

# #### Carspots

# In[ ]:


df['Carspots'].isna().sum()


# In[ ]:


df["Carspots"].median()


# In[ ]:


df['Carspots'].value_counts()


# * Carspots is a numerical discrete feature with float values.
# * There are 12 unique values including NaN.
# * The missing values may be because of no carspots.
# * It is better to replace the missing values with one of the unique value in the feature i.e 0.0.

# In[ ]:


df['Carspots'] = df['Carspots'].fillna(0.0)


# #### BuildingArea

# In[ ]:


df['BuildingArea'].isna().sum()


# * BuildingArea is a continuous numerical variable.
# * We see that there are 6459 missing values in BuildingArea that is half of the total count in the dataset i.e 13580.
# * As there are a lot of missing values , and to fill these nan values with any statistic may affect our analysis.
# * We will drop BuildingArea feature from our dataset.

# In[ ]:


df.drop(columns=['BuildingArea'],axis=1,inplace=True)


# ### YearBuilt

# In[ ]:


df['YearBuilt'].isna().sum()


# * There are 5375 missing values in YearBuilt out of the total 13580 data.
# * YearBuilt feature tells us on which year the house was built.
# * As it is a year feature , it can be any year that the house may have built and it will be inappropriate to fill the missing values with one median value.
# * We will fill the nan values with some random year and exclude this year while doing analysis.
# 

# In[ ]:


df['YearBuilt'] = df['YearBuilt'].fillna(1190)


# * Changing YearBuilt datatype to int as year cannot be in float datatype.

# In[ ]:


df['YearBuilt'] = df['YearBuilt'].astype(int)


# * Cast to period array

# In[ ]:


pd.to_datetime(df['YearBuilt'], format='%Y',errors = 'coerce').dt.to_period('Y')


# ### Council Area

# In[ ]:


df['CouncilArea'].isna().sum()


# * As per the description,Council Area tells us the Governing council of the area
# * There are 1369 missing values in CouncilArea feature.
# * Council Area is a categorical feature,let's have a look at the most repeated value in this feature.

# In[ ]:


df['CouncilArea'].value_counts()


# * We can see there is one Unavailable label in Council Area feature.It means there is no governing council for that area.
# * But it is not necessary that the areas with missing values in Council Area do not have any governing council.
# * As we don't have much information , we'll go with filling the missing values with Unavailable.

# In[ ]:


df['CouncilArea'] = df['CouncilArea'].fillna('Unavailable')


# #### We are done with missing values of the dataset.

# # Finding outliers in data

# <img src = "https://miro.medium.com/max/1280/1*2c21SkzJMf3frPXPAR_gZA.png" style="width:600px;height:300px;" >

# * The above image gives us a proper explanation of a box plot.
# * The points more than the maximum point and less than the minimum point in the box plot are considered as outliers.
# * Outlier Affect on variance, and standard deviation of a data distribution. 
# * In a data distribution, with extreme outliers, the distribution is skewed in the direction of the outliers which makes it difficult to analyze the data.

# In[ ]:


df['Landsize'].mean()


# In[ ]:


df['Landsize'].median()


# In[ ]:


#First step is to find outliers in our data
# Then decide whether to remove the outliers or to cap them
# We exclude here categorical and discrete features.
# And datetime related features ,Lattitude and Logtitude
# With this , we are left with only one feature i,e Landsize

col ='Landsize'


IQR = df[col].quantile(0.75) - df[col].quantile(0.25)
Lower_Bound = df[col].quantile(0.25) - (IQR*1.5)
Upper_Bound = df[col].quantile(0.25) + (IQR*1.5)

print("The outliers in {} feature are values << {} and >> {}\n".format(col,Lower_Bound,Upper_Bound))
minimum=df[col].min()
maximum=df[col].max()
print("The minimum value in {} is {} and maximum value is {}".format(col,minimum,maximum))

print("\nMaximum value is greater than the Upper_Bound limit")
print("Thus , outliers are values greater than Upper_Bound")

number_of_out = len(df[df['Landsize']>Upper_Bound])

print('\nThere are {} outliers in Landsize feature'.format(number_of_out))

        
    


# * This is a huge number 
# * Also we don't have much information about Landsize .
# * Landsize can also be zero as we saw above , but landsize of 4 lakhs looks strange.Thus , instead of removing these outliers we will cap them to Upper Bound limit of Landsize i.e 888.

# ## Cap the outliers

# In[ ]:


# We'll look at the box plot of Landsize before and after capping the outliers to show the difference.

fig = px.box(df,y='Landsize',width=800,title='Before capping the outliers')
fig.show()


# In[ ]:


df['Landsize'] = np.where(df[col]>Upper_Bound,Upper_Bound,df[col])


# In[ ]:


fig = px.box(df,y='Landsize',width=800,title='After capping the outliers')
fig.show()


# * Now we can see the difference in both the box plots.

# # Target Feature Analysis

# ## Price

# * Our target feature **`Price`** is a continuous variable that tells us about the price of the house sold.
# * Following two plots will give us a better look.

# In[ ]:


fig=plt.figure(figsize=(15,10))

plt.subplot(2,1,1)
fig1 = sns.distplot(df['Price'],color='red')

plt.subplot(2,1,2)
fig2 = sns.boxplot(data=df,x='Price',color='aqua')


# * Distribution of Price is right skewed with median 903000.
# * The highest price in our data is 9000000 and the least price is only 85000 so that's a vast difference that we are looking at.

# * Let's look at the rest of the features in our dataset
# * And we'll closely analyze each feature with Price and which feature plays a major role in prediction of price.

# * As there are two types of datatypes , we'll analyze the categorical and numerical features separately.

# # Categorical Features

# In[ ]:


#Extracting the categorical features from the dataset
skip_features=['Date_Sold','YearBuilt']  #Analyze both features later
cat = [col for col in df.columns if df[col].dtype=="O" and col not in skip_features]


# In[ ]:


#Display categorical features
print(cat)


# ## Suburb

# In[ ]:


df['Suburb'].describe()


# In[ ]:


df['Suburb'].value_counts()


# * There are a total 314 unique suburbs in our dataset.
# * As it will be very difficult to visualize each suburb , it's better to plot them on a map with the help of lattitude and longtitude.

# In[ ]:


df.groupby(['Suburb','Regionname'],as_index=False)['Lattitude','Longtitude','Price'].median()


# In[ ]:


df.columns


# In[ ]:


map_suburb = df.groupby(['Suburb','Regionname'],as_index=False)['Lattitude','Longtitude','Price'].median()
map_suburb

fig = px.scatter_mapbox(map_suburb,
                        lat="Lattitude",
                        lon="Longtitude",
                        color='Price',
                        mapbox_style='open-street-map',
                        hover_name='Suburb',
                        color_continuous_scale=px.colors.cyclical.IceFire,
                        size='Price',
                        center={"lat": -37.8136, "lon": 144.9631},
                        zoom=12,
                        hover_data=['Regionname','Suburb','Price'],
                       title='Average Price in different suburbs and regions')
fig.update_geos(fitbounds="locations", visible=True)
fig.update_geos(projection_type="orthographic")
fig.update_layout(template='plotly_dark',margin=dict(l=20,r=20,t=40,b=20))
fig.show()


# * We can see 314 points on the map plotted in Melbourne.
# * Based on the colorscale , the points with red dots and higher , are the suburbs with properties sold at higher price on average and the blue one's are with less price .
# * There is one black dot  i.e the `Kooyong surburb` where properties are sold on average price of 2185000.
# * And we can see more orange dots in southern metropolitan region and few in eastern metropolitan region.

# ## Property_Type

# In[ ]:


df['Property_Type'].unique()


# * There are three unique property types in our dataset.
# * As given in the description,
# * h refers to house,cottage,villa, semi,terrace.
# * u refers to a unit or duplex
# * And t refers to townhouse

# In[ ]:


df['Property_Type'].value_counts()


# In[ ]:


sns.countplot(data=df,x='Property_Type')


# * We have almost more than 8000 houses with h type and less than 1500 data for t type.
# * We don't have uniformity in data for Property_Type feature.

# #### Which property type on average is having highest price in our dataset?

# In[ ]:


#We groupby the data with Property Type and take median of Sale Price

temp = df.groupby(['Property_Type'],as_index=False)['Price'].median()
fig = px.bar(temp.sort_values(by='Price'),y='Property_Type',x='Price',orientation='h',color='Price',text='Price')
fig.update_traces(textposition='outside')
fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide',height=400,width=850)

fig.show()


# * `Houses,cottages,Villas or terraces` are having higher price than other property types.
# * If property is a `unit or duplex` type , then it's having less price on average around 560000.
# * As `t` type data is very less , it's hard to refer to average price.
# * Let's analyze these property types further.

# * A townhouse or town home is a row of houses attached to other houses. 
# * A duplex is a building having two units close to each other.Thus even though townhouse are less in the data , their average price is more than a unit or house or even duplex.
# * We'll analyze townhouse types further.

# ### Townhouse

# In[ ]:


townhouse = df[df['Property_Type']=='t']
townhouse = townhouse.reset_index(drop=True)


# ### In which regions are these townhouses located ?

# In[ ]:


px.bar(townhouse.groupby(['Regionname'],as_index=False)['Price'].median().sort_values(by='Price',ascending=False),y='Price',x='Regionname',color='Price',
       color_continuous_scale='Rainbow',height=500,width=800)


# * Townhouse are only located in metropolitan regions.
# * The highest average price of a townhouse sold is in Southern and South-Eastern Metropolitan regions and followed by eastern and other metropolitan regions.

# ## Method_Sold

# In[ ]:


df['Method_Sold'].unique()


# In[ ]:


df['Method_Sold'].value_counts()


# * 9022 properties have already been sold.
# * Only 92 properties have been sold after auction.
# * And the one's not sold in auction are the values for PI i.e 1564.
# * SP refers to property being sold prior to auction which generally happens if there is only one buyer and there are 1703 houses that have been sold by this method.
# * And last category VB refers to property being sold by bidding.
# * Let's see which method has the most effect on the price.
# * For this , we'll take the median price of each method which will be done easily by box plot.

# In[ ]:


fig1 = px.histogram(df,x="Price",color='Method_Sold',barmode='overlay')
fig1.show()

fig2 = px.box(df,x='Method_Sold',y='Price',color='Method_Sold')
fig2.show()


# * We don't see any variation in price based on which method the house has been sold.
# * Among the five methods, on average if the house has been sold prior auction , it's sold at lower price which is expected.

# ## SellerG

# * We can expect variation in price in this feature.
# * As it highly depends on the real estate agent on how he sells the house.
# * Let's look  at the number of sellers in our data.

# In[ ]:


df['SellerG'].nunique()


# * There are total 268 sellers in our data situated in melbourne.
# * Out of these sellers,which sellers have sold houses on average at a higher rate and which sellers have sold at a lower rate.

# In[ ]:


sellers = df.groupby(['SellerG'],as_index=False)['Price'].median()
sellers = sellers.sort_values(by='Price',ascending=False).reset_index(drop=True)
sellers


# * We can see the variation in price depending on each seller.
# * One thing to note is that , the reason some having sold at higher price or less price majorly depends on the property also.
# * Thus we'll look at the top sellers and sellers who have sold at lower price w.r.t type of property they have sold.

# In[ ]:


sellers_pt = df.groupby(['SellerG','Property_Type'],as_index=False)['Price','Lattitude','Longtitude'].median()
sellers_pt


# In[ ]:


# Now we separate the sellers into three property types,
sellers_pt_h = sellers_pt[sellers_pt['Property_Type']=='h']
sellers_pt_u = sellers_pt[sellers_pt['Property_Type']=='u']
sellers_pt_t = sellers_pt[sellers_pt['Property_Type']=='t']


# ### Average Price of House sold by each seller

# In[ ]:


fig = px.scatter_mapbox(sellers_pt_h,
                        lat="Lattitude",
                        lon="Longtitude",
                        color='Price',
                        mapbox_style='open-street-map',
                        hover_name='SellerG',
                        color_continuous_scale=px.colors.cyclical.IceFire,
                        size='Price',
                        center={"lat": -37.8136, "lon": 144.9631},
                        zoom=12,
                        hover_data=['SellerG','Property_Type','Price'],
                       title='Average price of houses sold by sellers')
fig.update_geos(fitbounds="locations", visible=True)
fig.update_geos(projection_type="orthographic")
fig.update_layout(margin=dict(l=20,r=20,t=40,b=20))
fig.show()


# * There are total 234 houses sold by sellers , and among them Caine,Weast have sold at highest price.
# * We can see many houses are sold at lower price by sellers.

# ### Average Price of Unit sold by each seller

# In[ ]:


fig = px.scatter_mapbox(sellers_pt_u,
                        lat="Lattitude",
                        lon="Longtitude",
                        color='Price',
                        mapbox_style='open-street-map',
                        hover_name='SellerG',
                        color_continuous_scale=px.colors.cyclical.IceFire,
                        size='Price',
                        center={"lat": -37.8136, "lon": 144.9631},
                        zoom=12,
                        hover_data=['SellerG','Property_Type','Price'],
                       title='Average price of units sold by sellers')
fig.update_geos(fitbounds="locations", visible=True)
fig.update_geos(projection_type="orthographic")
fig.update_layout(margin=dict(l=20,r=20,t=40,b=20))
fig.show()


# * There is only one black dot in the entire map , and we don't see any orange dots also.
# * One unit that is sold at a price of 1100000 is by David.

# ### Average Price of Townhouses sold by each seller

# In[ ]:


fig = px.scatter_mapbox(sellers_pt_t,
                        lat="Lattitude",
                        lon="Longtitude",
                        color='Price',
                        mapbox_style='open-street-map',
                        hover_name='SellerG',
                        color_continuous_scale=px.colors.cyclical.IceFire,
                        size='Price',
                        center={"lat": -37.8136, "lon": 144.9631},
                        zoom=12,
                        hover_data=['SellerG','Property_Type','Price'],
                       title='Average price of towhouses sold by sellers')
fig.update_geos(fitbounds="locations", visible=True)
fig.update_geos(projection_type="orthographic")
fig.update_layout(margin=dict(l=20,r=20,t=40,b=20))
fig.show()


# * In townhouses, Castran has sold at highest price of 2175000 followed by LJ agent.

# ### Now we look at top 5 sellers in our data depending upon their average price sold.

# In[ ]:


top_5 = df.query('SellerG in ["Weast","VICProp","Darras","Lucas","Kelly"]')
top_5


# * Weast is at the top who has sold only one house at highest price.
# * Darras and Lucas have also sold only one house at a price of 29,50,000 and 24,50,000 repectively.
# * The prices are calculated as average , and the highest price of a house sold is 9,00,0000.Let's see which seller has sold this house at 9000000.
# 

# In[ ]:


df[df['Price']==9000000]['SellerG']


# * Hall has sold a house at highest price in ur entire dataset at the price of 900000.
# * Let's look at his average price.
# * On average , hall has sold houses around 750000.

# ### Sellers who sold properties at a very low price on average

# * Following are the sellers with lowest selling price in the data and also look at these sellers .

# In[ ]:


temp[-5:]


# In[ ]:


bottom_5 = df.query('SellerG in ["Wood","hockingstuart/Village","hockingstuart/Advantage","Rosin","PRDNationwide"]')
bottom_5


# * PRDNationwide is a leading property research team , But in this data ,it has sold three houses on lowest price compared to rest of them.
# * We can see the sellers at the bottom of the table are small agencies that have sold houses at less price.
# * Rosin and Wood each have sold one house at 330000 and 370000 respectively.
# * The lowest price of a house sold is 85000,similarly we'll see which seller has sold at lowest price.

# In[ ]:


df[df['Price']==85000]['SellerG']


# * Burnham has sold at lowest price.

# ## Council Area

# In[ ]:


df['CouncilArea'].value_counts()


# * Council Area refers to the governing council area of that area.
# * Note that ,we see most of the data for Unavailable category because we filled the missing values with Unavailable.
# * So, the properties in melbourne are mostly from Moreland or Boroondara area.
# * And there is only one house that is in Moorabool Area.

# In[ ]:


fig=plt.figure(figsize=(20,25))

table = df.groupby(['CouncilArea'],as_index=False)['Price'].median().sort_values(by='Price',ascending=False)

sns.boxplot(data=df,x='Price',y='CouncilArea',order=table['CouncilArea'].to_list());
plt.yticks(fontsize=18);
plt.xticks(fontsize=18);
plt.ylabel("Council Areas",fontsize=22);
plt.xlabel("Price",fontsize=22);


# * We can see some variation in Council Areas w.r.t Price.
# * Boroondara and Bayside are the council areas in which houses have been sold at higher price.
# * But still the median price of top areas is less than 2 lakhs which is less than the average price of our entire dataset.
# * Properties in areas below Moreland in the above box plot have price less than 7 lakhs.
# 

# ## Region Name

# In[ ]:


df['Regionname'].value_counts()


# * So there are 8 regions in Melbourne with most of the houses are from Southern Metropolitan and least from Western Victoria.

# #### Does Region affect Price of the house? Let's see

# In[ ]:


fig=px.histogram(df,x='Price',color='Regionname',barmode="overlay")
fig.show()

fig=plt.figure(figsize=(15,10))
fig=sns.boxplot(data=df,x='Price',y='Regionname',palette='Dark2')
plt.yticks(fontsize=18);
plt.xticks(fontsize=18);
plt.ylabel("",fontsize=22);
plt.xlabel("Price",fontsize=22);
plt.show()


# * In Southern Metropolitan region , houses have been sold at higher rate.
# * If we look closely , in Victoria regions houses have been sold at lower rate than metropolitan regions.
# * But we also know that houses in victoria regions are less in this data.
# * It is obvious that metropolitan areas will have more demands. 

# ### Property Types in each region

# In[ ]:


fig = px.box(df,y='Price',x='Regionname',color='Property_Type',height=800,width=1500)

fig.update_xaxes(
    showgrid=True,
    tickson="boundaries",
    ticklen=10
)


# * We can see houses type are in every region and highest average price for houses type is in Southern Metropolitan region.
# * Townhouse we have already analyzed above.Unit type properties are not present in Northern and Western Victoria regions.
# * Unit types are having comparatively less average selling price in each region.
# 

# # Numerical Features

# In[ ]:


num=[col for col in df.columns if df[col].dtype!="O"]


# In[ ]:


print(num)


# * Now, numerical features are divided into two categories.

# # Discrete

# In[ ]:


discrete=[]
for col in df.columns:
    if df[col].dtype!="O" and len(df[col].unique()) < 15:
        discrete.append(col)


# In[ ]:


print(discrete)


# ## Let's analyze discrete features

# In[ ]:


fig = plt.figure(figsize=(15,5))
fig = sns.boxplot(data=df,x='Rooms',y='Price',palette='Purples')
plt.show()

fig = plt.figure(figsize=(15,5))
fig = sns.boxplot(data=df,x='Bedroom2',y='Price',palette='Reds')
plt.show()

fig = plt.figure(figsize=(15,5))
fig = sns.boxplot(data=df,x='Bathroom',y='Price',palette='Greens')
plt.show()

fig = plt.figure(figsize=(15,5))
fig = sns.boxplot(data=df,x='Carspots',y='Price',palette='Blues')
plt.show()


# * Here we compare all the categories in each discrete feature with Price.
# * `Rooms` :- In Rooms , as number of rooms increases , price also increases.But after 6 rooms ,  we see a drop in average price.This tells us that if the house has more than 6 rooms , it most probably will be sold at the same price as 4 or 5 rooms.
# * `Bedroom2` :- As per the description , the number of bedrooms are scraped from a different source.Similar to rooms , here also after 6 bedrooms the price stops increasing
# * `Bathroom` :- Wheareas in bathroom , If there are 7 bathrooms the price is the highest and we can see a sudden change in the price.And in houses with 8 bathrooms , the price is again less.
# * `Carspots` :- The price higher in zero number of carspots than one carspot is because of missing values filled with 0.Here we don't see any change, the slope keeps on increasing with the number of carspots.
#   

# # Continuous

# In[ ]:


continuous = [col for col in df.columns if df[col].dtype!="O" and col not in discrete]

print(continuous)


# In[ ]:


continuous = ['Distance_CBD', 'Postcode', 'Landsize', 'Propertycount','Price']


# * Firstly we'll look at the correlation of these continuous variables with price using Heatmap.

# In[ ]:


corr = df[continuous].corr()
fig = plt.figure(figsize=(15,10))

sns.heatmap(corr,annot=True,linewidths=.5,cmap='coolwarm',vmin=-1,vmax=1,center=0);


# * Distance to Central Business District has negative correlation , which is obvious as distance  to CBD decreases , Price of the property will increase.
# * Landsize is having a positive correlation with price but still less than 0.5.
# * We expected Landsize to have more correlation with price.
# * Property count has negative correlation with price of -0.042.

# ## Distribution of each continuous variable

# In[ ]:


#Separate Price as we have already been analyzed
#And also separate Postcode as it is not useful
continuous = ['Distance_CBD', 'Landsize', 'Propertycount']

plt.figure(figsize=(15,5))
plt.subplots_adjust(hspace=0.2)

i=1
colors = ['indianred','chocolate','yellowgreen','indigo']
j=0
for col in continuous:
    plt.subplot(1,3,i)
    a1 = sns.distplot(df[col],color=colors[j])
    i+=1
    j+=1


# * All the distribution have some skewness.Distance_CBD and Propertycount have positive skewness and Landsize has negative skewness.
# * Most of the houses have almost 0-20 distance to central business district.
# * Property count ranges from 249 to 21650.Average count is around 6555.That tells us on average 6500 properties are there in each suburb.

# ### Scatterplots will tell us the extent of correlation of these above features with Price.

# In[ ]:


continuous = ['Distance_CBD', 'Landsize', 'Propertycount']


plt.figure(figsize=(15,5))
i=1
colors = ['indianred','chocolate','yellowgreen']
j=0
for col in continuous:
    plt.subplot(1,3,i)
    a1 = sns.scatterplot(data=df,x=col,y='Price',color=colors[j])
    i+=1
    j+=1


# * Scatterplot also shows that as distance increases from central business districe , price decreases.
# * We can see a upward slope in Landsize , as landsize increases , price is also increasing.

# # DateTime Features

# * We have two timeseries features in our dataset.
# * Date_Sold :- Date at which the house was sold.
# * YearBuilt :- Year at which the house was built.

# ## DateSold

# * Instead of looking at the dates on which the house was sold , we take the year from it and analyze average price in each year.

# In[ ]:


df['Date_Sold'] = pd.to_datetime(df['Date_Sold'])
df['Year_Sold'] = df['Date_Sold'].dt.year


# In[ ]:


year_sold_grouped = df.groupby(['Year_Sold'],as_index=False)['Price'].median()
year_sold_grouped


# * We don't see any difference in both the year.
# * Houses sold in 2016 and 2017 have on average almost same price.

# * We'll analyze each month in both these years and compare the average price.

# In[ ]:


df['Month_Sold'] = df['Date_Sold'].dt.month
df['Month_Sold']


# In[ ]:


year_sold = df.groupby(['Year_Sold','Month_Sold'],as_index=False)['Price'].median()
year_sold


# In[ ]:


year_sold_2016 = year_sold[year_sold['Year_Sold']==2016]
year_sold_2017 = year_sold[year_sold['Year_Sold']==2017]


# In[ ]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=year_sold_2016['Month_Sold'], y=year_sold_2016['Price'],
                    mode='lines+markers',
                    name='House sold in 2016'))

fig.add_trace(go.Scatter(x=year_sold_2017['Month_Sold'], y=year_sold_2017['Price'],
                    mode='lines+markers',
                    name='House sold in 2017'))
fig.show()


# * We can see there is variation in each month in 2016 and 2017.
# * From April to June and August to September in the year 2017 , average price is more than in year 2016.
# * In both the year , April to August , there is downfall in price in both the years.

# ## YearBuilt

# In[ ]:


temp = df.groupby(['YearBuilt'],as_index=False)['Price'].median()

#We drop first two rows as one is the missing values and other is an outlier in our YearBuilt feature.
temp = temp.drop([0,1],axis=0).reset_index(drop=True)
temp


# In[ ]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=temp['YearBuilt'], y=temp['Price'],
                    mode='lines+markers',
                    name='Average Price over the years'))

fig.show()


# * This is strange, we see that over the year average price has come down from 1830 to 2018.

# ## That's it

# * We analyzed Price , our target feature and other features related to it.
# * Divided the data into different categories , and analyzed each feature with Price.
# * Detected outliers in our data.
# * There's more to come in this exercise.
# * We'll build an effective model for predicting Price feature based on the above analysis.
# * That will include Feature Scaling techniques,Feature Selection techniques.
# * Use different machine learning models for prediction.
# 

# **I'll be updating this notebook with some more interesting analysis , visualizations and predictive modelling.**
# 
# **Stay Tuned**
# 
# **Till then, Stay Home Stay Safe.**

# ## And If you like this notebook , please give an upvote :) :)
