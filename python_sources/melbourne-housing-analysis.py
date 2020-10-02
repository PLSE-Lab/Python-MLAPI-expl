#!/usr/bin/env python
# coding: utf-8

# # Introduction

# The dataset given is related to Melbourne house prices.Our aim is to find relation between saleprice and other variables given.The dataset includes Address, Type of Real estate, Suburb, Method of Selling, Rooms, Price, Real Estate Agent, Date of Sale,Distance etc. Using this analysis we can predict the house price in the future. 

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas.api.types import is_numeric_dtype
from scipy import stats
# Plotting Tools
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
sns.set_style("darkgrid")
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go
#Subplots
from plotly.subplots import make_subplots
#search for missing data
import missingno as msno
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# Bring test data into the environment
md= pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')


# In[ ]:


#fuction to show more rows and columns
def show_all(df):
    #This fuction lets us view the full dataframe
    with pd.option_context('display.max_rows', 400, 'display.max_columns', 100):
        display(df)


# In[ ]:


show_all(md)


# In[ ]:


md.info()


# In[ ]:


md.describe().T


# # Finding Missing Values

# In[ ]:


# Plot missing values of each column in the given dataset 
def plot_missing(df):
    # Find columns having missing values and count
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    
    # Plot missing values by count 
    missing.plot.bar(figsize=(16,5))
    plt.xlabel('Columns with missing values')
    plt.ylabel('Count')
    msno.matrix(df=df, figsize=(16,5), color=(0,0.2,1))
plot_missing(md)
    


# Using the heatmap below we are trying to find correlation between missing values in the data

# In[ ]:


msno.heatmap(md,figsize=(16,8));


# By analysing the missing values heatmap and dendrogram we can see the relationship between different missing values, that is if there is any relationship.

# In[ ]:


msno.dendrogram(md,figsize=(16,8));


# # Filling Missing Values

# Its really complicated to fill the missing values, any wrong assumption will lead to wrong analysis. I am filling the missing values based on CouncilArea, using the median of the selected column related to each CouncilArea to fill missing values of the selected column. This is done because i found that deviation from mean is more than deviation from median.

# **CouncilArea**

# The method followed by me is to fill the missing values of CouncilArea using Postcodes. As there is no missing values in Postcode and as postcodes are closely related to CouncilArea i tried to extract Postcodes from dataframe ca(has all CouncilArea which is null) then use it to find corresponding CouncilArea of the same Postcode which may have been filled in other rows. After this we were able to fill almost all the missing council areas except 5 rows. 

# In[ ]:


ca = md[md['CouncilArea'].isnull()]
cb = md[md['CouncilArea'].notnull()]

for i in list(ca.index):
    if (ca['Postcode'][i] in list(cb['Postcode'])):
        x = cb[cb['Postcode']== ca['Postcode'][i]].index[0]
        ca['CouncilArea'][i] = cb.iloc[x]['CouncilArea']


# In[ ]:


md1=pd.merge(ca,cb, how ='outer') 
md1=md1[md1["CouncilArea"].notnull()]#3rows are deleted
md1=md1[md1["CouncilArea"] != 'Unavailable']#1 row is deleted
md1=md1[md1["CouncilArea"] != 'Moorabool']#1 row is deleted


# **BuildingArea**

# In case of Building area i am using groupby to group the data by CouncilArea having median value of BuildingArea as a column.Then i used these median values to fill the missing values.

# In[ ]:


a=md1.groupby(['CouncilArea'])['BuildingArea'].median()
ba= md1[md1['BuildingArea'].isnull()]
bb= md1[md1['BuildingArea'].notnull()]
for i in list(ba.index):
    j= ba['CouncilArea'][i]
    ba['BuildingArea'][i] = a[j]
md1=pd.merge(ba,bb, how ='outer') 


# **YearBuilt**

# In case of YearBuilt i am using groupby to group the data by CouncilArea having median value of YearBuilt as a column.Then i used these median values to fill the missing values.It still maynot be accurate but this is the best approximation i could find.

# In[ ]:


year=md1.groupby(['CouncilArea'])['YearBuilt'].median()
yeara= md1[md1['YearBuilt'].isnull()]
yearb= md1[md1['YearBuilt'].notnull()]
for i in list(yeara.index):
    j= yeara['CouncilArea'][i]
    yeara['YearBuilt'][i] = year[j]
md1=pd.merge(yeara,yearb, how ='outer')


# **Car**

# In case of car i am using groupby to group the data by CouncilArea having median value of car as a column.Then i used these median values to fill the missing values.

# In[ ]:


car=md1.groupby(['CouncilArea'])['Car'].median()
cara= md1[md1['Car'].isnull()]
carb= md1[md1['Car'].notnull()]
for i in list(cara.index):
    j= cara['CouncilArea'][i]
    cara['Car'][i] = car[j]
md=pd.merge(cara,carb, how ='outer') 


# In[ ]:


import missingno as msno
msno.matrix(df=md, figsize=(16,5), color=(0,0.2,1));


# we can see data now has no more missing values.

# # Target variable

# In[ ]:


fig = make_subplots(1,2)

fig.add_trace(go.Histogram(x=md1['Price']),1,1)
fig.add_trace(go.Box(y=md['Price'],boxpoints='all',line_color='purple'),1,2)

fig.update_layout(height=500, showlegend=False,title_text="SalePrice Distribution and Box Plot")


# In[ ]:


md['Price'].skew()


# The house sellingprice data from the data we got is skewd but normaly we expect the the number of houses with comparatively lower prices to be of large number. Majority of the population wont be able afford the higher houseprices thus as the demand is more for comparatively lower priced houses the houses are constructed and priced that way, The data also reflects our assumptions.

# # Removing 'Extreme' outliers

# In this step we are removing some of the outliers as removing them will result in a better analysis.

# In[ ]:


for name in list(md.columns):
    if is_numeric_dtype(md[name]):
        y = md[name]
        removed_outliers = y.between(y.quantile(.001), y.quantile(.999))
        #removed_outliers.value_counts()
        index_names = md[~removed_outliers].index # invert removed outliers
        md.drop(index_names, inplace=True)


# In[ ]:


show_all(md)


# By removing the 'Extreme' outliers we have removed 212 rows from the dataframe.

# Now we are going to convert the date(date the house was sold) column into datetime format and create two new columns namely Yr_sold and Mth_sold which will help in our analysis.

# In[ ]:


md['Date'] = pd.to_datetime(md['Date'])
md['Yr_sold'] = md['Date'].dt.year
md['Mth_sold'] = md['Date'].dt.month
date= md.groupby(['Yr_sold','Mth_sold'],as_index=False)['Price'].median()
yr_2016 = date[date['Yr_sold']==2016]
yr_2017 = date[date['Yr_sold']==2017]
fig = go.Figure()
fig.add_trace(go.Scatter(x=yr_2016['Mth_sold'], y=yr_2016['Price'],
                    mode='lines+markers',
                    name='House price in 2016'))
fig.add_trace(go.Scatter(x=yr_2017['Mth_sold'], y=yr_2017['Price'],
                    mode='lines+markers',
                    name='House price in 2017'))
fig.show()


# All the houses we consider here have selling year as 2016 or 2017. In the above graph we can see the trend of house price throughout 
# the year and side by side comparison between 2016 and 2017.

# # Relationship Between Salesprice And Other Variables

# In[ ]:


#here we are checking the correlation between saleprice of house and other variables
corr_mat = md[['Price','Rooms','Distance','Bedroom2','Bathroom','Car','Landsize','Propertycount']].corr()
f, ax = plt.subplots(figsize=(30, 15))
sns.heatmap(corr_mat, vmax=1 , square=True,annot=True,linewidths=.5);


# The heatmap given above is not showing any clear correlation between salesprice and other variables.The highest correlation can be found between salesprice and bathroom,bedroom2,room. In case of correlation between other variables we can find very high correlation between bedroom2 and rooms.

# In[ ]:


yrblt= md.groupby(['YearBuilt'],as_index=False)['Price'].median()
fig = go.Figure()
fig.add_trace(go.Scatter(x=yrblt['YearBuilt'], y=yrblt['Price'],
                    mode='lines+markers'))


fig.show()


# By looking at the above graph we see that the price of houses have reduced over the years, but when we consider consecutive years variance in median price has reduced, 

# In[ ]:


#we can observe the level of influence of each variable in the following graphs.
cat1=['Rooms','Bedroom2','Bathroom','Car']

plt.figure(figsize=(25,15))
plt.subplots_adjust(hspace=0.5)

i = 1
for j in cat1:
    plt.subplot(1,4,i)
    sns.boxplot(x=md[j],y=md['Price'])
    plt.xlabel(j)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel('SalePrice', fontsize=18)
    plt.xlabel(j, fontsize=18)
    i+=1


# **General:-**
# The above boxplots clearly show a relationship between the variables and salesprice only exception being number of cars where we can see only a slow increase in salesprice as number of cars increases.It has to be noted that we are not considering the outliers while drawing the above conclusions.
# 

# **Rooms:-**
# The houseprice increases as the number of rooms increases this continues till almost 6 number of rooms then for 7,8,10 number of rooms the sellingprice is comparatively decreasing this is when we consider the median of the respective data.
# 

# **Bedroom**:-The price of houses with 0 bedrooms is higher when compared to 1 bedroom houses but after that the selling price of houses increases with the number of bedrooms till 6 bedrooms then the selling price of houses is show a gradual decrease with the number of bedrooms.

# **Bathroom**:-In case of bathrooms we can clearly see a increase in selling price of house with respect to increase in number of bathrooms till 4 then the selling price of house gradualy decreases after number of bathrooms crosses 4, Only exception to this is number of bathrooms of value 7 where selling price of house is highest.

# **Cars**:-The relation between number of cars and selling price of house is not clear from the given data as no strong upward or downward correlation is seen.But we can say that houses with car value as 7 has highest selling price and houses with car value as 1 has lowest selling price.  

# In[ ]:


fig = px.histogram(md, x=md.Price, y=md.Landsize, color=md.Type,marginal="box", hover_data=md.columns)
fig.show()


# From the above plot we can see that h type of houses have higher selling price compared to others followed by t and then u. Most of the house price values are between 0 and 3 million.

# In[ ]:


fig=plt.figure(figsize=(15,8))

fig = sns.scatterplot(x='Propertycount', y='Price', data=md);


# we can see that as property counts increases the sales price of houses also increases till 11000 then the sales price decreases as decreases.

# In[ ]:


cat2=['Distance','Landsize','Propertycount']
sns.lmplot(x='Distance', y='Price', data=md,scatter=False,aspect=4,height=15);
plt.xticks(fontsize=50)
plt.yticks(fontsize=50)
plt.ylabel('SalePrice', fontsize=50)
plt.xlabel('Distance', fontsize=50);
sns.lmplot(x="Landsize", y="Price", data=md,aspect=4);
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ylabel('SalePrice', fontsize=22)
plt.xlabel("Landsize", fontsize=22);


# **Distance:**-From the graph we can see that the sales price decreases as the distance increases and is 
# inversely proportional.

# **Landsize**:-The graph shows a positive correlation between landsize and salesprice, ie the house selling price is more for larger landsize.

# In[ ]:


cat3=['Type','SellerG','CouncilArea','Regionname']
plt.figure(figsize=(26,40))
plt.subplots_adjust(hspace=0.5)
plt.subplot(2,1,1)
sns.barplot(data=md,x='Regionname',y='Price',hue="Type");
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(prop={'size': 15})
plt.ylabel('SalePrice', fontsize=18)
plt.xlabel('Regionname', fontsize=18)
plt.xticks(rotation=90)
plt.xticks(rotation=90);
plt.subplot(2,1,2)
sns.barplot(data=md,x='CouncilArea',y='Price',hue="Type");
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(prop={'size': 15})
plt.ylabel('SalePrice', fontsize=18)
plt.xlabel('CouncilArea', fontsize=18)
plt.xticks(rotation=90);


# **Regionname**:-we can see that generaly southern metropolitan has highest house selling price and when we consider the types houses has the highest number among all, followed by town house but we have note that northern and western victoria has no townhouse or unit type houses.   

# **Councilarea**:-we can see a similar trend as seen in region here but house selling price gradually decreases from left to right.

# In[ ]:


plt.figure(figsize=(26,40))
plt.subplots_adjust(hspace=0.5)
plt.subplot(2,1,1)
sns.swarmplot(data=md,x='Regionname',y='Price',hue="Method");
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(prop={'size': 15})
plt.ylabel('SalePrice', fontsize=18)
plt.xlabel('Regionname', fontsize=18)
plt.xticks(rotation=90);
plt.subplot(2,1,2)
sns.swarmplot(data=md,x='CouncilArea',y='Price',hue="Method");
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(prop={'size': 15})
plt.ylabel('SalePrice', fontsize=18)
plt.xlabel('CouncilArea', fontsize=18)
plt.xticks(rotation=90);


# By the above plot we can see that method of selling is highest for 'S - property' sold when we consider both region and councilarea followed by 'SP - property sold prior' 'PI - property passed in' which has similar number of values.Number house sold is decreasing from left to right in case of council area and region in the above plot.The highest selling price is seen in Boroondara in case of council area and southern metropolitan in case of region.  

# In[ ]:


a=md.groupby(['Suburb','Lattitude','Longtitude'],as_index=False)['Price'].median()
fig = px.scatter_mapbox(a,
                        lat="Lattitude",
                        lon="Longtitude",
                        color='Price',
                        mapbox_style='open-street-map',
                        hover_name='Suburb',
                        size='Price',
                        center={'lat': -37.8136, 'lon': 144.9631},
                        zoom=13,
                        hover_data=['Suburb','Price'],
                        title= 'SalesPrice In Each Suburb')
fig.update_geos(fitbounds="locations", visible=True)
fig.update_geos(projection_type="orthographic")
fig.update_layout(template='plotly_dark',margin=dict(l=20,r=20,t=40,b=20))
fig.show()


# By above plot we can see the selling price of houses region wise.

# # Conclution

# From the data we have,the sales price increases as number of rooms,bedrooms,bathrooms increases to a point after which selling price might decrease.When in case of increase in property size we can see a increase in sales price but when we consider the distance factor we see the house sales price decreasing as distance increasing.Region and council wise difference is also clearly vissible in the data as Boroondara has highest house selling price in council and southern metropolitan has highest house selling price in region.We can use the information extracted to predict the house selling price in the future. 

# **Do upvote if u find my analysis usefull...**
