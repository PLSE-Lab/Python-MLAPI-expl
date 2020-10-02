#!/usr/bin/env python
# coding: utf-8

# # Lets import the things

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Lets load the dataset

# In[ ]:


df = pd.read_csv("/kaggle/input/suicides-in-india/Suicides in India 2001-2012.csv")


# # 1. Data Wrangling

# In[ ]:


df.describe()


# # 1.1Getting Info about dataset

# In[ ]:





# In[ ]:


df.info()


# # 1.2Getting sample about Dataset

# In[ ]:


df.head()


# # 1.3 Checking for null value

# In[ ]:


df.isna().sum()


# # 1.4 Shape of the Dataset

# In[ ]:


df.shape


# # 1.5 Copying the dataset for future use

# In[ ]:


df_copy = df.copy()


# # 1.6 Droping the duplicates

# In[ ]:


df.drop_duplicates(inplace=True)


# # 1.7 Shape After Droping the Duplicates

# In[ ]:


df.shape


# # 2 . Data Visualization

# In[ ]:


df.head()


# In[ ]:


df.plot()


# In[ ]:


df["Total"].plot()


# # From the above Graph
# ## we can see that this df["Total"] is not a going good..
# ## let's see what's going on there

# In[ ]:


df.Total.max()


# ### So The max value comes from

# In[ ]:


df[df["Total"] == 63343]


# ### We can see that the data is all over india but it should be a State of india  , However let's explore df["Total"] more

# In[ ]:


df[df["Year"] == 2012].tail()


# ### so the data is sorted by State name , we can't find All india total there  , so let's go deep

# In[ ]:


df.query("State == 'Total (All India)'").sample(n=5)


# ### so this proves that in all year The data of All India Total is stored right , no  more doubts in df["Total"]

# In[ ]:


get_ipython().run_cell_magic('html', '', '<h1> "Let\'s Finish Data Visualization and move to Exploratory Data Analysis "</h1>')


# # 3. Exploratory Data Analysis

# ## 3.1 . Exploring Single Variables

# In[ ]:


df.head()


# ### 3.1.0 We don't want to care about Total == 0 right because they are like No susides right

# In[ ]:


df = df[df["Total"] > 0 ]


# In[ ]:


df.shape


# In[ ]:


df_copy.shape


# In[ ]:


print(f"No of columns contains Target == 0 is :{df_copy.shape[0] - df.shape[0]}")


# In[ ]:


df.head()


# ## 3.2 Get Count of age Group

# In[ ]:


df["Age_group"].value_counts().plot()


# ## 3.3 Get Counts by Gender

# In[ ]:


df["Age_group"].value_counts()


# In[ ]:


df["Gender"].value_counts(normalize=True)


# ## 3.4 Gets counts of TYPE

# In[ ]:


df["Type"].value_counts(normalize=True)


# ## 3.5 Gets count of Type_code

# In[ ]:


df["Type_code"].value_counts()


# In[ ]:


df["Type_code"].value_counts(normalize=True)


# # Need to do 2 variable EDA

# In[ ]:


df.head()


# # So lets Get Year and total

# In[ ]:


df_year_total = df.groupby(df["Year"]).agg({"Total":"sum"})


# In[ ]:


df_year_total.reset_index(inplace=True)


# In[ ]:


df_year_total.head()


# In[ ]:


plt.title("Suicides in india ")
sns.barplot(x="Year" , y="Total" , data=df_year_total , palette="viridis" )


# # Male vs Female in Total suicides rate

# In[ ]:


df_gender = df.groupby("Gender").agg({"Total":"sum"})


# In[ ]:


df_gender.reset_index(inplace=True)


# In[ ]:


df_gender


# In[ ]:


plt.figure(figsize=(10,10))
plt.title("Male vs Female Suicides in India")
g = sns.barplot(x="Gender" , y="Total" , data=df_gender , palette="muted" )


# # Types and Total values

# In[ ]:


df_types = df.groupby(df["Type"]).agg({"Total":"sum"})


# In[ ]:


df_types.sort_values(by="Total" , inplace=True ,ascending=False)


# In[ ]:


df_types.reset_index(inplace=True)


# ## Top 5 Types of Suicides

# In[ ]:


plt.figure(figsize=(10,5))
plt.title("Top 5 Types of Suicides in India")
sns.barplot(x="Type" , y="Total" , data=df_types[:5])


# # Type code vs total

# In[ ]:


df_types = df.groupby(df["Type_code"]).agg({"Total":"sum"})


# In[ ]:


df_types.reset_index(inplace=True)


# In[ ]:


plt.title("Type_code and Suicides count in india")
sns.barplot(y="Type_code" , x="Total" , data=df_types , orient="h")


# In[ ]:





# In[ ]:


df.head()


# ## Finally State vs Total

# In[ ]:


df_state = df.groupby("State").agg({"Total":"sum"}).sort_values(by="Total" , ascending=False)


# In[ ]:


df_state.reset_index(inplace=True)


# In[ ]:


df_state.head()


# In[ ]:


df_state.drop(df_state.index[[0,1]] , inplace=True)


# In[ ]:


df_state.head()


# In[ ]:


plt.title("Top 7 States Suicides list in Inida 2001 - 2012")
sns.barplot(y="State" , x="Total" , data=df_state[:7])


# In[ ]:


plt.title("Bottom 7 States Suicides list in Inida 2001 - 2012")
sns.barplot(y="State" , x="Total" , data=df_state[-7:])


# # we're not finished in double variables we're going on

# ## we missed Age_group vs total

# In[ ]:


df_age_group = df.groupby(df["Age_group"]).agg({"Total":"sum"})


# In[ ]:


df_age_group = df_age_group[1:]


# In[ ]:


df_age_group


# In[ ]:


df_age_group.reset_index(inplace=True)


# In[ ]:


plt.title("Age group vs Suicides in India 2001 - 2012")
sns.barplot(x="Age_group" , y="Total" , data=df_age_group)


# In[ ]:





# # Muliti variable EDA

# In[ ]:


df.head()


# # State vs Year and Total

# In[ ]:


df_year_state = df.groupby(["State" , "Year"]).agg({"Total":"sum"})


# In[ ]:


df_year_state.reset_index(inplace=True)


# In[ ]:


df_year_state.head(15)


# In[ ]:


df_temp  = df_year_state.query("State == 'A & N Islands' or State == 'Andhra Pradesh' ")


# In[ ]:


df_temp


# ## Suicides in  Andhra vs A & N island over year's

# In[ ]:


g = sns.FacetGrid(col="State" , data=df_temp)
g.map(sns.lineplot , "Year" , "Total")
# plt.xticks(rotation="vertical")
g.set_xticklabels(rotation=90)
# g.set_title("AP vs A&N island")
plt.show()


# In[ ]:


plt.figure(figsize=(20,10))
g = sns.FacetGrid(col="State" , data=df_temp)
g.map(sns.barplot , "Year" , "Total")
# plt.xticks(rotation="vertical")
g.set_xticklabels(rotation=90)
plt.show()


# # State vs Type code

# In[ ]:


df_temp = df.groupby(["State" , "Type_code"]).agg({"Total":"sum"})


# In[ ]:


df_temp.reset_index(inplace=True)


# In[ ]:


df_temp = df_temp.query("State=='Andhra Pradesh' or State == 'Tamil Nadu'")


# In[ ]:


g = sns.FacetGrid(col="State" , data = df_temp)
g.map(sns.barplot , "Type_code" , "Total")
g.set_xticklabels(rotation=90)


# In[ ]:


g = sns.FacetGrid(col="State" , data = df_temp)
g.map(sns.lineplot , "Type_code" , "Total")
g.set_xticklabels(rotation=90)


# ## State vs Types

# In[ ]:


df_temp = df.groupby(["State" , "Type"]).agg({"Total":"sum"})


# In[ ]:


df_temp.reset_index(inplace=True)


# In[ ]:


df_temp.Type.unique()


# In[ ]:


df_temp = df_temp.query("State=='Andhra Pradesh' or State == 'Tamil Nadu'").query("Type == 'By Jumping off Moving Vehicles/Trains' or Type == 'Dowry Dispute' or Type == 'Failure in Examination'")


# In[ ]:


df_temp


# In[ ]:


g = sns.FacetGrid(col = "State" , data = df_temp )
g.map(sns.lineplot , "Type" , "Total")
g.map(sns.barplot , "Type" , "Total")
g.set_xticklabels(rotation=90)


# In[ ]:


df.head()


# # Year ,  State , Type_code  , Total

# In[ ]:


df_temp = df.groupby(["Year" , "Type_code" , "State"]).agg({"Total":"sum" ,} , ["State"]).reset_index()


# In[ ]:


df_temp


# In[ ]:


df_temp = df_temp.query("State == 'Andhra Pradesh' or State == 'Tamil Nadu'")


# In[ ]:


df_temp


# In[ ]:


g = sns.FacetGrid(data=df_temp , col="Type_code" , col_wrap=2 , hue="State" , palette="summer"  ,)
g.map(sns.barplot , "Year" , "Total" , alpha=0.9).add_legend()
[plt.setp(ax.get_xticklabels(), rotation=70) for ax in g.axes.flat]
plt.show()


# In[ ]:


g = sns.FacetGrid(data=df_temp , col="Type_code" , col_wrap=2 , hue="State" , palette="twilight_shifted_r"  ,)
g.map(sns.lineplot , "Year" , "Total" ).add_legend()
g.set_ylabels("Total Sucides")
[plt.setp(ax.get_xticklabels(), rotation=70) for ax in g.axes.flat]

plt.show()


# In[ ]:


df.head()


# In[ ]:


df.groupby("Type").agg({"Total":"sum"}).reset_index().query("Type == 'Love Affairs' or Type == 'Failure in Examination'")


# # Year , State , Gender  , total

# In[ ]:


df_temp = df.groupby(["Year" , "State" , "Gender"]).agg({"Total":"sum"})


# In[ ]:


df_temp.reset_index(inplace=True)


# In[ ]:


df_temp = df_temp.query("State == 'Andhra Pradesh' or State == 'Tamil Nadu'")


# In[ ]:


df_temp.head()


# In[ ]:


g = sns.FacetGrid(data = df_temp , col="Gender" , hue = "State" )
g.map(sns.lineplot , "Year" , "Total").add_legend()


# In[ ]:





# # **So End is here  , Thanks and it's my 3rd EDA Practice and 1st on kaggle**

# 

# Hey i'm writing Blog.
# medium.com/@sanjaykhanssk
# 
# instagram.com/i_ssk16

# In[ ]:




