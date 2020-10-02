#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pylab as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("/kaggle/input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv")
df_regions = pd.read_csv("/kaggle/input/120-years-of-olympic-history-athletes-and-results/noc_regions.csv")
df.head()


# # checking all columns names and adding them to a list 

# In[ ]:


cols = df.columns.values
df["Age"]= df["Age"].fillna(0)
df["Age"] = df["Age"].astype("int64")
df["Height"] = df["Height"]/100
cols


# In[ ]:


df.info()


# # Start exploring our dataset and checking total numbers of male and female

# In[ ]:


sns.countplot("Sex", data = df)


# # distribution plot for percentage of age in our dataset
# the most comon age group is between 18-27

# In[ ]:


tickss = np.arange(df["Age"].min(),df["Age"].max(),5)
plt.figure(figsize=(10,10))
g=sns.distplot(df["Age"])
plt.xticks(tickss)
plt.show()


# In[ ]:


sns.distplot(df["Height"])


# # -Merging to two data frame to able to handle our dataset through region(Country)

# In[ ]:


df_merged= pd.merge(df, df_regions, on= "NOC")
df_merged.columns


# the most contires participating in olympics

# In[ ]:


df_countries =df_merged["region"].value_counts().to_frame().reset_index().sort_values(by= "region",ascending = False)
df_countries.head()


# In[ ]:


sns.countplot(df["Medal"], hue= df["Sex"])
plt.title("Each medal type distibution between Male and Females", fontsize=20)


# In[ ]:


plt.figure(figsize=(15,15))
sns.countplot(df["Sport"], hue= df["Sex"])
plt.xticks(rotation=90)
plt.title("Number of player in each sport for males and females")


# # - Exploring all player from Egypt team

# In[ ]:


df_egypt= df_merged[df_merged["region"]=="Egypt"]
df_egypt_medal = df_egypt["Medal"].value_counts()
print("List of total medals achived by Egyptian player: \n", df_egypt_medal)


# In[ ]:


df_egypt.shape


# # -Distribution of Egyptian players over years

# In[ ]:


df_year_count_egypt = df_egypt.groupby("Year")["ID"].count().reset_index()
# df_year_count_egypt
plt.figure(figsize=(10,10))
sns.barplot(data= df_year_count_egypt, x= "Year", y = "ID")
plt.xticks(rotation = 90, fontsize=12)
plt.title("Bar chart to number of Egyptian player in each year", fontsize= 18)
plt.show()


# # -Adding new column(Age Group) showing player age condition(Young-Old)

# In[ ]:


df_merged["Age_group"] = ["Young" if item < 40 else "Old" for item in df_merged["Age"]]
df_merged["Age_group"].value_counts()


# # -Adding new column(BMI) showing each player body mass index ratio
# # -Adding another column(Body fit) showing each player body codition(Thin-Obese_Fit)

# In[ ]:


df_merged["BMI"] = df_merged["Weight"]/np.power(df_merged["Height"],2)
df_merged["Body_Fit"] = ["Thin" if x <18.5 else "Obese" if x > 25 else "Fit" for x in df_merged["BMI"]]


# In[ ]:


df_merged["Body_Fit"].value_counts()


# # -Exploring the obese player list to know what kind of sport compitions can include obese player

# In[ ]:


df_body_Obese = df_merged[df_merged["Body_Fit"]== "Obese"]
df_body_Obese.head()


# In[ ]:


obese_sport_list = df_body_Obese["Sport"].value_counts()
print("List of all sports which required obese or high Body mass index: \n", obese_sport_list)


# # -Same as pervious step we will check what kind of sport can by done by thin people

# In[ ]:


df_body_thin = df_merged[df_merged["Body_Fit"]== "Thin"]
df_body_thin.shape


# In[ ]:


thin_sport_list = df_body_thin["Sport"].value_counts()
print("List of all sports which can be done by low Body mass index athletes: \n", thin_sport_list)


# # By logic all fit people can participate in all sports and they are the majority of our dataset

# In[ ]:


def find(year,sport,Medal="Gold"):
    """
    parameter: Year,Sport type
    return: Athletes who receive a gold medal according to the sport type and the year.
    """
    data_find=df_merged[(df_merged['Year']==year) & (df_merged['Sport']==sport) & (df_merged['Medal']==Medal)]
    return data_find
# find(2000,"Wrestling") # list of all player with gold medal in wrestling in 2000


# # now we are going to explore the gold medalest player over 40 years
# And how is it posible to achieve a gold medal after 40s

# In[ ]:


df_Old_Medalest = df_merged[np.logical_and(df_merged["Age_group"] == "Old",df_merged["Medal"]== "Gold")]
df_Old_Medalest.head(3)


# In[ ]:


old_medalest_list = df_Old_Medalest.groupby("Sport")["Team"].count().sort_values(ascending = False).to_frame().reset_index()

print("List with gold medalest older than 40 year: \n", old_medalest_list)


# In[ ]:


plt.figure(figsize=(13,13))
plt.tight_layout()
sns.countplot(df_Old_Medalest["Sport"])
plt.xticks(rotation= 90, fontsize= 12)
plt.title("Gold Medal for Athletes over 40Y", fontsize= 18)
plt.show()


# # Data frame for all women with gold medal in Olympics

# In[ ]:


womenInOlympics = df_merged[(df_merged["Sex"]=="F") & (df_merged["Medal"] == "Gold")]
womenInOlympics.shape
print("Total of women with gold medal in Olympics is: ",womenInOlympics.shape[0])


# In[ ]:


df_merged.columns


# # Exploring the most contries participating with ladies in olympics games

# In[ ]:


women_ByCountry = womenInOlympics.groupby("region")["ID"].count().sort_values(ascending= False).to_frame().reset_index()
women_ByCountry.loc[0:20,]


# In[ ]:


plt.figure(figsize=(15,15))
sns.barplot(women_ByCountry["region"], women_ByCountry["ID"])
plt.xticks(rotation= 90, fontsize = 12)


# # -Exploring total number of player over years for both males and females

# In[ ]:


df_year_grouped = df_merged.groupby(["Year", "Sex"])["ID"].count().to_frame().reset_index()
plt.figure(figsize=(10,10))
sns.pointplot(data = df_year_grouped, x= "Year",y="ID", hue="Sex" )
plt.xticks(rotation = 90)


# In[ ]:





# In[ ]:




