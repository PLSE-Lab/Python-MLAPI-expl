#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing the basic libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly
import plotly.express as px


# In[ ]:


#loading the dataset
df= pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')
df.head()


# # Data Cleaning

# In[ ]:


#Null Values in dataframe
df.isnull().sum()


# In[ ]:


# As their are very few null values in Type,Content Rating, Current Ver and Android Ver thus we can drop the null values
df.dropna(subset=['Current Ver','Android Ver','Content Rating','Type'],inplace=True)


# In[ ]:


# In the ratings column as their are 1474 null values so we can't remove all of them or replace all of them with a single value.
df['Rating'].describe()



# In[ ]:


# The average of Ratings columns is 4.2 but as the number of null values are high it won't be a good step to fill all the null values with a single value.
# We will check the null values distribution under differnt categories of app and fill the null values with the average of the respective category.
# We can find out distribution of null values under various categories using the excel pivot function.
Category              Null Values Count   Avg Rating
BUSINESS              157                 4.1
TOOLS                 109                 4
PERSONALIZATION       78                  4.3
LIFESTYLE             68                  4.1
BOOKS_AND_REFERENCE   53                  4.3
NEWS_AND_MAGAZINES    50                  4.1
GAME                  47                  4.3
HEALTH_AND_FITNESS    44                  4.3
FINANCE               43                  4.1
DATING                39                  4             
# For the categories mentioned above we will fill the null values with respective categories and for others we will fill null values with the overall average i.e 4.2


# In[ ]:


#filling null values in rating column

df["Rating"] = df["Rating"].fillna(-1)     #filling null values with any random number that will be replaced by values mentioned below
for i in range(10829):
    if df['Rating'].iloc[i]==-1:
        if df['Category'].iloc[i]=='EVENTS' or df['Category'].iloc[i]=='ART_AND_DESIGN' or df['Category'].iloc[i]=='EDUCATION' :
            df['Rating'].iloc[i]=4.4
        elif df['Category'].iloc[i]=='PERSONALIZATION' or df['Category'].iloc[i]=='BOOKS_AND_REFERENCE' or df['Category'].iloc[i]=='GAME' :
            df['Rating'].iloc[i]=4.3
        elif df['Category'].iloc[i]=='BUSINESS' or df['Category'].iloc[i]=='LIFESTYLE' or df['Category'].iloc[i]=='NEWS_AND_MAGAZINES' or df['Category'].iloc[i]=='FINANCE' :
            df['Rating'].iloc[i]=4.1
        elif df['Category'].iloc[i]=='TOOLS' or df['Category'].iloc[i]=='DATING' :
            df['Rating'].iloc[i]=4.0
        else :
            df['Rating'].iloc[i]=4.2
        i=i+1


# In[ ]:


df['Reviews'] = df['Reviews'].astype(int)  #convert Reviews column to int type


# In[ ]:


#Editing the size column
#In size column some values are in kb and some are in kb so we can convert all values in kb
#I have added a new column size_in_kb that will contain app size in kb
df['size_in_kb']=df['Rating']*0
for i in range(10829):
    if df['Size'].iloc[i][-1]=='M':
        df['size_in_kb'].iloc[i]=float(df['Size'].iloc[i][0:-1])*1024
    elif df['Size'].iloc[i][-1]=='k':
        df['size_in_kb'].iloc[i]=float(df['Size'].iloc[i][0:-1])
    else :
        df['size_in_kb'].iloc[i]=df['Size'].iloc[i]
    i=i+1
#We can drop the size column
df.drop(['Size'], axis = 1,inplace=True)


# In[ ]:


#Editing the Installs Column
for i in range(10829):
        df['Installs'].iloc[i]=df['Installs'].iloc[i][0:-1]
        i=i+1


# In[ ]:


df.shape


# In[ ]:


# In Installs column the values are in form of 10,000 or 500,000 so before converting it into integer we need to remove the commas in values
for i in range(10829):
    df['Installs'].iloc[i] = df['Installs'].iloc[i].replace(',', '')
    i=i+1


# In[ ]:


df['Installs'] = df['Installs'].astype(int)


# In[ ]:


#In size_in_kb column their are some rows with entry 'varies with device'. we can delete these rows
df1=df.copy()
df = df1[df1['size_in_kb'] != 'Varies with device'] 
df['size_in_kb'].unique()   # Now we can check that all values are either int or float


# In[ ]:


# drop Current ver and Android Ver columns
df.drop(['Current Ver'], axis = 1,inplace=True)
df.drop(['Android Ver'], axis = 1,inplace=True)


# In[ ]:


df.shape


# In[ ]:


#Editing the price column.
for i in range(9135):
    if df['Price'].iloc[i]!= '0':
        df['Price'].iloc[i]=df['Price'].iloc[i][1::]
    i=i+1


# In[ ]:


df.rename(columns = {'Price':'Price_in_dollar'}, inplace = True)   #Renaming the price column
df['Price_in_dollar'] = df['Price_in_dollar'].astype(float)  #convert price column to int type


# In[ ]:


# Now we have cleaned up all the data and we check how the final dataset looks like
df.head()


# # * Data visualization using Plotly

# In[ ]:


#Paid Vs Free Apps
fig = px.histogram(df, x="Type",height=400)
fig.show()


# In[ ]:


#Distribution of Ratings in various categories
fig = px.violin(df,x='Category', y="Rating",color='Category',width=2000, height=600)
fig.show()
plotly.offline.plot(fig, filename='violin0.html')


# In[ ]:


#Rating of Paid and free apps
g1=df.groupby(['Type'],as_index=False)[['Rating']].mean()
g1.rename(columns={'Rating':'Avg_rating'},inplace=True)
fig = px.histogram(g1, x="Type",y='Avg_rating',height=800)
fig.show()

#from the table we can say that paid apps have higher ratings than free apps


# In[ ]:


#Analysing paid apps
df_paid=df[df['Type']=='Paid']
df_paid.head()


# In[ ]:


fig = px.histogram(df_paid, x="Category",y="Installs",labels={'Category':'Category wise paid apps', 'Installs':'Number of Installs'},height=600,width=1000)
fig.show()


# In[ ]:


g4=df_paid.groupby(['Category'],as_index=False)[['Price_in_dollar']].mean()
g4.rename(columns={'Price_in_dollar':'Avg_Price'},inplace=True)
fig = px.bar(g4, x="Category",y='Avg_Price',color='Avg_Price')
fig.show()
#from the barplot we can conclude that finance apps have highest average price


# In[ ]:



fig = px.histogram(df, x="Content Rating",color="Type",height=600)
fig.show()


# In[ ]:


#Most popular category
fig = px.histogram(df, x="Category",y="Installs",color='Type',labels={'Category':'Category', 'Installs':'Number of Installs'},height=600,width=1000)
fig.show()

#From the plot we can say that Game category has highest number of installs so it is the most popular category


# In[ ]:


#Largest app in terms of size
df1=df.copy()
df1.sort_values(by=['size_in_kb'], inplace=True,ascending=False)
df1.head()
#from the dataset we can check that the app named 'Gangster Town: Vice District	' is largest in terms of size.


# In[ ]:


#Distribution of Ratings
fig = px.histogram(df, x="Rating",height=600,width=1000)
fig.show()


# In[ ]:


#Average number of Installs in Various Categories
g2=df.groupby(['Category'],as_index=False)[['Installs']].mean()
g2
g2.rename(columns={'Installs':'Avg_Installs'},inplace=True)
fig = px.bar(g2, x="Category",y='Avg_Installs',color='Avg_Installs',height=800)
fig.show()


# In[ ]:


# From the bar plot we can say that the Average Installs for free apps is higher than that of paid apps
g3=df.groupby(['Type'],as_index=False)[['Installs']].mean()
g3.rename(columns={'Installs':'Avg_Installs'},inplace=True)
fig = px.bar(g3, x="Type",y='Avg_Installs',color='Avg_Installs')
fig.show()


# In[ ]:




