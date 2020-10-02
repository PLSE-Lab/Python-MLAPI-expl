#!/usr/bin/env python
# coding: utf-8

# ### Introduction
# 
# The purpose of this document is to perform Exploratory Data Analysis on zomato restaurants located in banglore. 
# 
# <font color="blue">Data Set</font> : Data set is collected from the kaggle data set.
# 
# <font color="blue">Busines Objective</font> : Understand the data and inspect the extract insights such as 

# In[ ]:


# basic libraries used

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

pd.options.display.max_columns = None

# code presenting customer functions

from IPython.display import Markdown, display
def printmd(string):
    display(Markdown(string))
    
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[ ]:


#reading Data 
data = pd.read_csv("../input/zomato.csv")

printmd("### initial look into data")

print(data.info())

data.head()


# ### Review Remaks 
# 
# Data Set is having 51,717 observations with 17 fields. in the intial look observed that all fields are string type except votes field. 
# 
# noted for the EDA, we don't need these URL, Address, Phone so desided to remove these columns 
# 
# menu items are in the form of string convert string to list

# In[ ]:


#reorganizing items in menu items to list

data["menu_item"]=data["menu_item"].str.strip("'[]'").str.split()

# Split the rates to the seperate
data["rate"]=data["rate"].str.split("/")
#data["rate"] = data.rate.str[0]

# Reorgnizing Rating field

data["rating"]=pd.to_numeric(data.rate.str[0],errors='coerce') 
data["rating_outof"]=pd.to_numeric(data.rate.str[1],errors='coerce') 

#Deleting unwanted columns 
data.drop(["address","url","phone"],axis=1,inplace=True)

#reattanging cost field 
data["costfortwo"]=data["approx_cost(for two people)"].str.replace(",",'').astype(float)


# In[ ]:


## Missing values analysis

_missing_counts=pd.DataFrame(round(data.isna().sum() / data.shape[0] ,2))

_missing_counts.columns =["Count"]
_missing_counts=_missing_counts.reset_index()

_missing_counts.sort_values(by="Count",ascending = False)


# In[ ]:


# from the rating understanded that if rating is like NEW can be considered as NEW or else existing restaurant

data["Category"] = np.where((data.rating.isna()==True) & (data.rate.isna()==False),"New","Existing")


# In[ ]:


data.groupby("name")["name"].count().sort_values(ascending =False)[:20]


# In[ ]:


# identified some estaurants indicating resturent chains. and these type of resturents represnts mostly same kind of standards hence. 

# hence for analysis on rating and votes we will prepare another data set by deleting all resturents with keeping only one Restaurants of any specfic chain

data_unique_Restaurants=(data.drop_duplicates(subset=["name"],keep="first")).copy()

data_unique_Restaurants.shape


# ### Performing univariate analysis 

# In[ ]:


# analysing Category of restaurants 

def dis_pie(field,num):
    #plt.subplot(num)
    _temp = pd.DataFrame(data.groupby(field)[field].count())
    #plt.title(field+" Descriptions")
    #plt.pie(_temp,labels = _temp.index,autopct='%1.1f%%', shadow=True, startangle=90)
    print(_temp)
    
plt.figure(figsize=(10,10))
dis_pie("Category",221)
dis_pie("online_order",222)
dis_pie("book_table",223)
plt.show()

def dis_barChart(field):
    plt.figure(figsize=(20,10))
    _temp = pd.DataFrame(data.groupby(field)[field].count())
    _temp.columns =["count"]
    _temp=_temp.reset_index()
    _temp.columns =[field,"count"]  
    _temp=_temp.sort_values(by="count",ascending =False)
    plt.title(field+" Descriptions")
    g=sns.barplot(data =_temp,x=field,y="count")
    plt.xticks(rotation=90)
    plt.show()
    
    
dis_barChart("listed_in(city)")
dis_barChart("rest_type")
dis_barChart('listed_in(type)')

dis_barChart('rating')
#_temp = pd.DataFrame(data.groupby())


# ### Review Remaks 
# 
# a. Restaurants in banglore rating indicates perfect normalized graph. i.e major resturents are having rating between 3.6 and 4.1
# 
# b. major number of Restaurants are in BTS (around 10%), HSR, kormangala 5th block JP nagar 
# 
# c. Restaurants classified majorly described as  quick bites type food

# In[ ]:


#Derived columns for cuisines

#covert list variables as list
data_unique_Restaurants["cuisines_ls"]=data_unique_Restaurants.cuisines.str.split(",")

#find out distient elements of cuisines
_temp=data_unique_Restaurants['cuisines_ls'].apply(pd.Series)
for i in _temp.columns:
    _temp[i] = _temp[i].str.strip().str.lower()    
    
data_unique_Restaurants["cuisines_ls1"]=_temp.apply(lambda x:list(x),axis=1)
list_of_cuisines = _temp.stack().value_counts()


# In[ ]:


for i in list_of_cuisines[:10].index:
    data_unique_Restaurants[i] =data_unique_Restaurants.cuisines_ls1.astype(str).str.contains(i)


# In[ ]:


#display Rare cuisines surving restaruents
rare_cuisines = list_of_cuisines[-40:-1]
rare_cuisines


# In[ ]:


# identified restaurants serving where rare foods

data_unique_Restaurants["rare_cuisines"] =data_unique_Restaurants.cuisines_ls1.astype(str).str.contains("hot dogs")

for i in rare_cuisines.index:
    data_unique_Restaurants["rare_cuisines"] =(data_unique_Restaurants.cuisines_ls1.astype(str).str.contains(i)) | (data_unique_Restaurants["rare_cuisines"])


# In[ ]:


# derived colum defining type of cuisines surved by the restaurant
data_unique_Restaurants["number_of_different_cuisines"] = data_unique_Restaurants.cuisines_ls.str.len()


# In[ ]:


# attemped to identify how many of Restaurants serving each popular type of cuisnies + rare_cuisines

_temp=pd.DataFrame(data_unique_Restaurants[list_of_cuisines[:10].index.tolist() +["rare_cuisines"]].stack())
_temp=_temp.reset_index(level=1)
_temp.columns=["name","values"]


(_temp.groupby(["name","values"])["values"].count() / data_unique_Restaurants.shape[0]).unstack().plot(kind ="bar",stacked=True,figsize=(20,5),title="restaurants serving specific cuisine comparing with number of restaurants")
plt.show()


# ### Review Remaks 
# 
# Based on above graph Even Most of Restaurants serving "North Indian Food" more than 60 of total Restaurants will not have a choice of north indian dishes
# similarly chinese serving Restaurants closely about 25+% only and 75% Restaurants not serving chainees dishes

# In[ ]:


# multi cuisine Restaurants

_temp =data_unique_Restaurants.groupby('number_of_different_cuisines')["number_of_different_cuisines"].count() / data_unique_Restaurants.shape[0]

_temp =pd.DataFrame(_temp)
_temp.columns =["Per"]
_temp=_temp.reset_index()
_temp.columns =["noof_multi_cuisine","Per"]

plt.figure(figsize=(10,5))
ax=sns.barplot(x="noof_multi_cuisine",y="Per",data=_temp)
plt.title("Multi Cuisine Resturents in Banglore")

for index, row in _temp.iterrows():
    ax.text(index ,row.Per+0.001,str(round(row.Per,4)*100)+"%", color='black', ha="center")
    
    
plt.show()


# ### Review remaks 
# 
# 1. arround 24% of Restaurants serving  only one type of cuisine 
# 2. around 35% of Restaurants serving  two type of cuisine 
# 3. around 24% of Restaurants serving  three type of cuisine 
# 
# 4. only about 3% of Restaurants serving more than 6 cuisine

# In[ ]:


## Review number of votes 

#data.votes.describe()
data_unique_Restaurants["votes_range"]=pd.cut(data_unique_Restaurants.votes,[0,2,5,7,10,15,20,50,100,200,1000,10000])

_temp =data_unique_Restaurants.groupby("votes_range")["name"].count() / data_unique_Restaurants.shape[0]
_temp=pd.DataFrame(_temp)
_temp=_temp.reset_index()
_temp.columns=["votes_range","Per"]

plt.figure(figsize=(20,5))
ax=sns.barplot(x="votes_range",y="Per",data =_temp)
plt.title("Resturent by votes")

for index, row in _temp.iterrows():
    ax.text(index ,row.Per+0.001,str(round(row.Per,4)*100)+"%", color='black', ha="center")
    
plt.show()
##data.head()


# ### Review Remaks 
# 
# 1. maximum votes for the segment of 20 to 50 votes of about 14.13%
# 2. around 12% of bangnore Restaurants got the votes in the range of 200-1000
# 3. opnly 3% of Restaurants are got the votes above 1000

# In[ ]:


#_temp=data.pivot_table(index="votes_range",columns=["north indian"],values=["north indian"],aggfunc="count")

#_temp=_temp.reset_index(level =0)
_temp=pd.concat([data_unique_Restaurants[list_of_cuisines[:10].index.tolist() +["rare_cuisines"]].astype(int) , data_unique_Restaurants["votes_range"]],axis=1)

_temp=_temp.groupby("votes_range").sum() 
#_temp=_temp.reset_index()
plt.figure(figsize=(20,7))
plt.subplot(1,2,1)
plt.title("Votes by cusine type for all resturents (# Count)")
sns.heatmap(_temp,cmap="YlGnBu",vmin=0.4)
#plt.show()

_temp=_temp.apply(lambda x: x/sum(x),axis=0)

#_temp=_temp.reset_index()
#plt.figure(figsize=(10,7))
plt.subplot(1,2,2)
plt.title("% Votes by cusine type on the same segment of serving cuisine")
sns.heatmap(_temp,cmap="YlGnBu",vmin=0.01,annot=True)
plt.show()
#data[list_of_cuisines[:10].index.tolist() +["rare_cuisines"]].apply(lambda x: 1 if x else 0,axis=0)


# ### Remaks 
# 
# ##### Graph-1 
# 
# since north indian /  chinese cuisine serving Restaurants are high in bangalore, it is +ve correlated with the number Restaurants attracting more votes.
# followed by continental and cafes and italian 
# 
# eventhough <b> south  indian Restaurants </b> are in high in numbers number of votes attracted above 200 is very less.
# 
# 
# ##### Graph-2
# 
# if Restaurants is serving continential /  italian / rare cuisine  the are majorly attract higher votes above 200+ votes.  fro example on continential serving resturents closly 49% of resturents attract more than 200 votes.
# 
# south indian serving Restaurants's otherhand seems having several classes of resturent i.e mostly spreaded accross 20 votes to 1000 votes
# 
# 

# In[ ]:


#_temp=data.pivot_table(index="votes_range",columns=["north indian"],values=["north indian"],aggfunc="count")

#_temp=_temp.reset_index(level =0)
_temp=pd.concat([data_unique_Restaurants[list_of_cuisines[:10].index.tolist() +["rare_cuisines"]].astype(int) , data_unique_Restaurants["rating"]],axis=1)

_temp=_temp.groupby("rating").sum() 
#_temp=_temp.reset_index()
plt.figure(figsize=(20,7))
plt.subplot(1,2,1)
plt.title("Votes by cusine type for all resturents (# count)")
sns.heatmap(_temp,cmap="YlGnBu",vmin=0.4)

_temp=_temp.apply(lambda x: x/sum(x),axis=0)

#_temp=_temp.reset_index()
plt.subplot(1,2,2)
plt.title("Votes by cusine type on the same segment of serving cuisine (%)")
sns.heatmap(_temp,cmap="YlGnBu",vmin=0.01)
plt.show()
#data[list_of_cuisines[:10].index.tolist() +["rare_cuisines"]].apply(lambda x: 1 if x else 0,axis=0)


# ### Review Remaks 
# 
# Similar to previous analysis on voting. rating's also appeared to be showing same indication.
# 
# 
# #### Graph1
# Overall major Restaurants offering north indian and chines Restaurants number of resturents getting higher rating i.e above 3 is more on this type serving resturents.
# 
# #### Graph 2
# 
# similar to the previous analysis, if resturents is serving rare cuisnines or continental  /  cafe indicating probability of attract higher rating

# In[ ]:


_temp=data_unique_Restaurants.loc[ data_unique_Restaurants["costfortwo"].isnull() ==False ,["votes_range","costfortwo","rating"]]

#_temp["costfortwo"]=_temp["approx_cost(for two people)"].str.replace(',', '').astype(float) 

plt.figure(figsize=(20,5))
plt.title("Number of votes vs cost for two")
sns.boxplot(data =_temp,x="votes_range",y="costfortwo")
plt.show()

plt.figure(figsize=(20,5))
plt.title("Rating vs cost for two")
sns.boxplot(data =_temp,x="rating",y="costfortwo")
plt.show()

#_temp["approx_cost(for two people)"]=_temp["approx_cost(for two people)"].str.replace(',', '').astype(float) 

#_temp[_temp["approx_cost(for two people)"].isnull()]

#data.iloc[1662]


# ### Review Remaks 
# 
# when the number of votes are high, indicating that cost for food also relatively high.
# 
# when the ratung higher than 4 indicating ,that price for 2 expected to be mean of 800+ and it keep increased with the increase of resrurent rating
# 
# <font color="red"> there are some Restaurants with the rating lesser than 2.5 and indicating price for the is closely about 1000 Rs. </font>
#     
#     

# In[ ]:


# online orders vs votes
data_unique_Restaurants.groupby(["votes_range","online_order"])["name"].count().unstack().plot(kind="bar",stacked=True,figsize=(15,5),title="Option of Online Order vs Votes")
plt.show()

#online order vs rating
data_unique_Restaurants.groupby(["rating","online_order"])["name"].count().unstack().plot(kind="bar",stacked=True,figsize=(15,5),title="Option of Online Order vs rating")
plt.show()


# #### Review Remaks 
# 
# not found evidence of online order facility attracting higher number of rating or votes

# In[ ]:


# online orders vs votes
data_unique_Restaurants.groupby(["votes_range","book_table"])["name"].count().unstack().plot(kind="bar",stacked=True,figsize=(15,5),title="Option of Online Order vs Votes")
plt.show()

#online order vs rating
data_unique_Restaurants.groupby(["rating","book_table"])["name"].count().unstack().plot(kind="bar",stacked=True,figsize=(15,5),title="Option of Online Order vs rating")
plt.show()


# ### Review Remaks 
# 
# if Restaurants offering table booking option indicating to be attract higher number of votes and higher number of ratings

# In[ ]:


_temp=data_unique_Restaurants.loc[ data_unique_Restaurants["approx_cost(for two people)"].isnull() ==False ,["book_table","approx_cost(for two people)","online_order"]]
_temp["costfortwo"]=_temp["approx_cost(for two people)"].str.replace(',', '').astype(float) 


plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
plt.title("Booking Table option vs cost for two")
sns.boxplot(data =_temp,x="book_table",y="costfortwo")


plt.subplot(1,2,2)
plt.title("Online Order option vs cost for two")
sns.boxplot(data =_temp,x="online_order",y="costfortwo")
plt.show()


# ### Review remaks 
# 
# booking table option indicating to be Restaurants with the good standards. it represent highest customer expereince by means of attracting higher votes and rating. and if option offering online or not does't indicate any coorlation with the customer expereince

# In[ ]:



# Top of Restaurants of choice can be tried

data_unique_Restaurants.loc[(data_unique_Restaurants.rating>4.5) &(data_unique_Restaurants.
                                                                 costfortwo <=800),["name","cuisines","listed_in(city)","listed_in(type)","votes","rating",
                                                                                    "costfortwo"]].sort_values(by="costfortwo")




