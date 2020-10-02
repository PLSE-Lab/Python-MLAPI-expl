#!/usr/bin/env python
# coding: utf-8

# ## This Dataset is related to Zomato Bangalore Restaurant's

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Breakdown of this notebook:
# Loading the dataset: Load the data and import the libraries.
# 
# Data Cleaning:
# 1. Dropping Uncessary col's.
# 2. Dropping duplicates.
# 3. Checking Missing Value Percent with respect to each column.
# 4. Dropping row's on having atleast one null value.
# 5. Reanaming Column's
# 7. Feature Transfomation 
# ---------------------
# Data Visualization : 
# 1. Univariate analysis.
# 2. Bivariate  analysis.
# -------------------
# Regression Analysis
# 1. Linear Regression.
# 2. Random Forest Regression.
# -------------
# Regression metrics Used 
# 1. r_squared value.
# 2. adjusted_r_squared value.
# 

# In[ ]:


import pandas as pd
import numpy  as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("darkgrid")


# In[ ]:


zomato = pd.read_csv("/kaggle/input/zomato-bangalore-restaurants/zomato.csv")


# In[ ]:


zomato = pd.DataFrame(zomato)
zomato.columns = [i.lower() for i in zomato.columns]
print("Original DataSet : ",zomato.shape)


# In[ ]:


zomato.columns


# In[ ]:


# let's drop uncessary columns
#url,phone,address,reviews_list,menu_item
zomato.drop(["url","phone","address","reviews_list","menu_item","location"],axis = 1,inplace = True)


# In[ ]:


#let's drop duplicate values from the dataset
zomato = zomato.drop_duplicates()
#after removing the duplicate's from the dataset
zomato.shape


# In[ ]:


# let's see null percentage respect to each column
Null_Value_Percent = pd.DataFrame(zomato.isnull().sum()/len(zomato)*100)
Null_Value_Percent.rename(columns = {0:"Missing Value Percent"},inplace = True)
Null_Value_Percent


# In[ ]:


# let's drop null value records from the dataset
zomato.dropna(how = 'any',inplace = True,axis = 0)
# how = 'any', means remove the record if it has atleast one null value .
zomato.shape


# In[ ]:


#let's rename some column's
zomato.rename(columns = {'approx_cost(for two people)':"cost","listed_in(type)":"service_type","listed_in(city)":"city"},inplace = True)
zomato.columns


# In[ ]:


zomato.info()


# # Let's Preprocess Our Feature's

# In[ ]:


# let's convert type of cost feature in to int type
zomato["cost"] = zomato["cost"].apply(lambda cost : cost.replace(",",""))
zomato["cost"] = zomato["cost"].astype(int)
#zomato["cost"].sample(10)


# In[ ]:


#cleaning rate column

# let's remove records related to newly opened shop's
new_rating_index = zomato[zomato["rate"]=="NEW"].index
zomato.drop(new_rating_index,axis = 0,inplace = True)
print("Dataset After droping records related to newly opened shop's : ",zomato.shape)

# let's remove /5 from rating(rate) column
zomato["rate"] = zomato["rate"].astype(str)
zomato["rate"] = zomato["rate"].apply(lambda x : x[0:3])
zomato["rate"] = zomato["rate"].astype(float)


# ## Let's Explore

# # univariate analysis

# ## 1. top 10 Restaurants Type's
#  insight : Most of the  restaurant's offering Casual Dinning Service.

# In[ ]:


plt.rcParams['figure.figsize'] = (16,6)
top_ten_restaurant_types = zomato.rest_type.value_counts().head(10)
sns.barplot(top_ten_restaurant_types.index,top_ten_restaurant_types)
plt.title("Top 10 Restaurants Type's",fontsize = 20)
plt.xlabel("Restaurants Type's",fontsize = 20)
plt.ylabel("count")


# ## 2. Top 10 liked dish's 
# insight : Biryani is the Most Liked food among all,

# In[ ]:


top_ten_liked_dish = zomato.dish_liked.value_counts().head(10)
sns.barplot(top_ten_liked_dish.index,top_ten_liked_dish)
plt.title("Top ten liked dish's",fontsize = 20)
plt.xlabel("Dish's ",fontsize = 20)
plt.ylabel("count")


# ## 3.Which Food Service Type is  Preferred  by Most Of the Restaurant ?
# insight : Most of the Restaurant's prefer's "Delivery" and "Dine-out" as thier Service "

# In[ ]:


food_service_type = zomato.service_type.value_counts()
sns.barplot(food_service_type.index,food_service_type)
plt.title("Food Service Type's ",fontsize = 20)


# ## 4.Top Five Cooking Style's Preferred By Restaurant's  ( cuisines )
# insight : North indian  (cuisines) cooking style is Prefferred By Most of the Restaurant's .

# In[ ]:


top_five_cooking_style = zomato.cuisines.value_counts().head(5)
sns.barplot(top_five_cooking_style.index,top_five_cooking_style)
plt.title("Top Five Cooking Style's",fontsize = 20)
plt.xlabel("Cooking Style's ",fontsize = 20)
plt.ylabel("count")


# ## 5.Which City Got More Number of Restaurant's ? (top 5)
# insight : Koramangala got more Number of Restaurant's 

# In[ ]:


# get the city
zomato.city = zomato.city.str.split(" ").str[0] 


# In[ ]:


city_rest_count = zomato.city.value_counts().head()
sns.barplot(city_rest_count.index,city_rest_count)
plt.title("City with More Number of Restaurant's",fontsize = 20)


# ## 6.  Let's see the Distribution Of Cost Column 
# insight : Most of the Food Ordered Cost's is in between the range of 1 to 1000

# In[ ]:


sns.kdeplot(zomato["cost"],shade = True)
plt.title("Cost Distribution ")
plt.xlabel("Cost Value's")
print("Highest Food Ordered Cost :",zomato["cost"].max())
print("Lowest  Food Ordered Cost :",zomato["cost"].min())


# # Bivariate Analysis

# ## 1. what is the  average rating given to the  Restaurant's in a Particular City (top 5  cities with highest rated Restaurant's ) 
# insight : Restaurant's present in Chruch Road Location rated highest comapred to others,

# In[ ]:


avg_city_rest_rate = pd.pivot_table(zomato,index = ["city"],values = ["rate"])
avg_city_rest_rate = avg_city_rest_rate.sort_values(by = 'rate',ascending = False).head(5)
sns.barplot(avg_city_rest_rate.index,avg_city_rest_rate["rate"])
plt.title("Top 5 cities with highest rated Restaurant's ",fontsize = 20)


# ## 2. Best Restaurant's in best  Cities (top 5  cities with highest rated Restaurant's )

# In[ ]:


plt.rcParams['figure.figsize'] = (8,4)
top_rest_in_chruch_road = zomato[zomato["city"] == "Church"].sort_values(by = "rate",ascending = False).head(5)
sns.barplot(top_rest_in_chruch_road.name,top_rest_in_chruch_road.rate)
plt.title("Best Restaurant's in Chruch Road")
plt.rcParams['figure.figsize'] = (8,4)


# In[ ]:


top_rest_in_Residency = zomato[zomato["city"] == "Residency"].sort_values(by = "rate",ascending = False).head(5)
sns.barplot(top_rest_in_Residency.name,top_rest_in_Residency.rate)
plt.title("Best Restaurant's in Residency Road")


# In[ ]:


top_rest_in_Brigade = zomato[zomato["city"] == "Brigade"].sort_values(by = "rate",ascending = False).head(5)
sns.barplot(top_rest_in_Brigade.name,top_rest_in_Brigade.rate)
plt.title("Best Restaurant's in Brigade Road")


# In[ ]:


top_rest_in_MG = zomato[zomato["city"] == "MG"].sort_values(by = "rate",ascending = False).head(5)
sns.barplot(top_rest_in_MG.name,top_rest_in_MG.rate)
plt.title("Best Restaurant's in MG Road")


# ## 3.Top 5 Food price's 

# In[ ]:


food_with_highest_price = zomato[["dish_liked","cost"]]
food_with_highest_price = food_with_highest_price.drop_duplicates().sort_values(by = "cost",ascending = False ).head(5)
food_with_highest_price


# ## 4.What type of Order's do Most of Restaurant's Get (offline v/s online )
# insight  : Most of Restaurant's Get online order's 

# In[ ]:


order_type = zomato.online_order.value_counts()
plt.pie(order_type,labels = order_type.index,autopct='%1.1f%%', shadow=True)
plt.show()


# ## 5. Restaurant with Highest Votes
# 

# In[ ]:


rest_with_highest_votes = zomato[["name","votes"]].drop_duplicates().sort_values(by="votes",ascending = False).head(5)
sns.barplot(rest_with_highest_votes.name,rest_with_highest_votes.votes)
plt.title("Top 3 Restaurant's of Banglore ")
plt.xlabel("Top Voted Restaurant's")
plt.ylabel("Number of Votes ")


# ## 6.is there any relationship between (votes v/s cost) ?
# insight : There is no relationship (votes v/s cost)

# In[ ]:


sns.jointplot(zomato["votes"],zomato["cost"])


# ## 7.let's find best  restaurants
# The criteria must be: 
# 1.cheapest,
# 2.highly rated,
# 3.reliable(large number of votes) options

# In[ ]:


# eg : let's find best restuarant under given criteria for eating 
# Paneer Peri Peri, Pancakes, Fajitas, Sweet Crepe, Peri Peri Chicken, Chicken Breast, Nutella Crepe


# In[ ]:


#assume restaurant's with greater than 4.5 ratings are best one
high_rated_rest = zomato[zomato["rate"]>4.5]


# In[ ]:


#assume any one most_liked_dished and compare the result's in order to find cheapest with better quailty
#high_rated_rest.dish_liked.value_counts() #uncomment this to know better
high_rated_cheapprice_rest = high_rated_rest[high_rated_rest["dish_liked"]=="Paneer Peri Peri, Pancakes, Fajitas, Sweet Crepe, Peri Peri Chicken, Chicken Breast, Nutella Crepe"]


# In[ ]:


# consider Restaurant with Highest vote's 
# assume restaurant's with votes greater than 700 are all good once's,
high_rated_cheapprice_highvotes_rest = high_rated_cheapprice_rest[high_rated_cheapprice_rest["votes"]>700]
best_restuarant = high_rated_cheapprice_highvotes_rest["cost"].min()
high_rated_cheapprice_highvotes_rest[high_rated_cheapprice_highvotes_rest["cost"]==best_restuarant].iloc[:,:]

# below are the best restaurant's for eating ,
# Paneer Peri Peri, Pancakes, Fajitas, Sweet Crepe, Peri Peri Chicken, Chicken Breast, Nutella Crepe


# ## let's see the correlation between the features

# In[ ]:


sns.heatmap(zomato.corr(),annot = True)
# feature's are highly correlated


# In[ ]:


def get_num_for_cat(zomato):
    for column in zomato.columns[~zomato.columns.isin(['rate', 'cost', 'votes'])]:
        zomato[column] = zomato[column].factorize()[0]
    return zomato

z_fact = get_num_for_cat(zomato.copy())


# In[ ]:


import numpy as np
from sklearn.model_selection import train_test_split
x = z_fact.loc[:,z_fact.columns!="rate"]
y = np.array(z_fact.loc[:,"rate"])


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


# In[ ]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.3,random_state=1)


# # Linear Regression Model

# In[ ]:


len(xtest)


# In[ ]:


lr_model =LinearRegression()
lr_model.fit(xtrain,ytrain)
ypred=lr_model.predict(xtest)


# In[ ]:


print("r2_score for Linear Reg model : ",r2_score(ytest,ypred))   
r2 = r2_score(ytest,ypred)
k = xtest.shape[1]
n = len(xtest)
adjusted_r2 = 1 - (((1-r2)*(n- 1))/(n - k - 1)) 
print("adjusted_r2_score for Linear Reg model : ",adjusted_r2)


# ## RandomForestRegressor

# In[ ]:


rf_model = RandomForestRegressor(n_estimators=200,min_samples_split=20,random_state=43)
rf_model.fit(xtrain,ytrain)
ypred = rf_model.predict(xtest)


# In[ ]:


print("r2_score for RandomForestRegressor model : ",r2_score(ytest,ypred))   
r2 = r2_score(ytest,ypred)
k = xtest.shape[1]
n = len(xtest)
adjusted_r2 = 1 - (((1-r2)*(n- 1))/(n - k - 1)) 
print("adjusted_r2_score for RandomForestRegressor model : ",adjusted_r2)


# Thank You :) ,Please Upvote if you find this kernel helpful ,and i am happy to get any kind of suggestions and concern's related my work .

# In[ ]:




