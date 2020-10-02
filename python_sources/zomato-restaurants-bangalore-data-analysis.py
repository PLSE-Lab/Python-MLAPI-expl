#!/usr/bin/env python
# coding: utf-8

# ### Objective of the notebook

# To do Exploratory Data Analysis on data of more than 50000 restaurants across the Bangalore City.<br>
# Following are some of the outcomes we wish to obtain from the analysis:<br>
# - Identify the food trends in Bangalore
# - Areas with huge number of outlets
# - How much a meal for two costs in Bangalore
# - What is the quality of outlets and food
# - Is there any correlation between the ratings of an outlet and the cost of meal.
# - Which are the largest food chains in Bangalore
# - Top rated outlets
# - Which category is witnessing influx of new outlets.

# ### Importing libraries and loading Data set

# In[ ]:


# Import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import plotly.express as px
import math
import re
import warnings


# In[ ]:


#loading the data set
restaurannt_data = pd.read_csv("../input/zomato-bangalore-restaurants/zomato.csv")


# In[ ]:


pd.set_option("display.max_rows",500)
pd.set_option("display.max_columns", 200)


# In[ ]:


main_df = pd.DataFrame(restaurannt_data)
main_df.head(100)


# In[ ]:


main_df.info()


# ###  Data Preprocessing (Cleaning and Handling Missing Values)

# #### Dropping Irrelevant Columns

# In[ ]:


#Droping irrelevant columns
main_df.drop(columns = ["url","phone"], inplace = True)


# In[ ]:


#bringing the names column to the front
main_df.insert(loc = 0, column = "Name", value = main_df["name"])


# In[ ]:


#dropping the old name column
main_df.drop(columns = "name", inplace = True)


# In[ ]:


main_df.info()


# #### Handling non-float values of ratings column

# In[ ]:


#Modify the ratings column
main_df.rename(columns = {"rate" :"ratings"}, inplace = True)
main_df["ratings"] = main_df["ratings"].replace(to_replace = ["/5"], value = '', regex = True).str.strip()


# In[ ]:


#Lets check the unique values of ratings column
main_df["ratings"].unique()


# In[ ]:


#Replace all non digit values to 0
main_df["ratings"].replace(to_replace = ["-","NEW",np.NaN], value = 0, inplace = True)
main_df.ratings.unique()


# In[ ]:


#Finally convert the data type of the ratings column to float
main_df["ratings"] = main_df["ratings"].astype('float')

main_df.ratings.dtype


# #### Cleaning columns "menu_item" and "reviews_list"

# In[ ]:


main_df["menu_item"].replace(to_replace = "[]", value = np.NaN, inplace = True)
main_df["reviews_list"].replace(to_replace = "[]", value = np.NaN, inplace = True)


# In[ ]:


main_df.isna().sum()


# In[ ]:


#We will drop the rows where value of approx cost and rest_type is not mentioned
main_df.dropna(subset = ["rest_type","approx_cost(for two people)","cuisines"], inplace = True)
main_df.isna().sum()


# #### Dropping the "menu_item" , "reviews_list" and "dish_liked" columns

# In[ ]:


#Now we will drop the menu_item and dish liked columns as approximately 50% of the values in these columns are missing
#Also we will drop the reviews list column which we forgot to drop earlier as the column is of no use to us in our analysis
main_df.drop(["menu_item","dish_liked","reviews_list"],axis = 1, inplace = True)


# In[ ]:


#We have handled the missing values in the data frame
main_df.isna().sum()


# #### Modifying the data type of "approx_cost" column

# In[ ]:


#Now we will convert the approx cost column values to integer data type
main_df["approx_cost(for two people)"] = main_df["approx_cost(for two people)"].replace(to_replace = "[,]", value = "", regex = True)
main_df["approx_cost(for two people)"] = main_df["approx_cost(for two people)"].astype("int")


# #### Inserting a new column with categorized values of cost_column

# In[ ]:


#Funtion to categorize the cost for two values
def cost_for_two(value):
    if value < 200:
        return "<200"
    elif value < 500:
        return "200-500"
    elif value < 800:
        return "500-800"
    elif value < 1500:
        return "800-1500"
    elif value <3000:
        return "1500-3000"
    else:
        return ">3000"


# In[ ]:


#Categorizing the value of the "approx cost for two people"
main_df["cost_for_two"] = main_df["approx_cost(for two people)"].apply(lambda x:cost_for_two(x))


# #### Creating separate data frames for old and new restaurants (with 0 ratings)

# In[ ]:


#We will remove all rows with zero ratings and store then in another data frame for future separate analysis

new_rest_df = main_df[main_df["ratings"] == 0]
res_df = main_df[main_df["ratings"] != 0]


# In[ ]:


#Shape of two data frames we created
res_df.shape, new_rest_df.shape, main_df.shape


# #### Reviewing remaining columns for data irregularities

# In[ ]:


main_df["listed_in(city)"].unique()


# In[ ]:


main_df["online_order"].unique(), main_df["book_table"].unique()


# Now our data is clean and can be used for data visualization and analysis

# ### Exploratory Data Analysis

# #### Outlets accepting online orders

# In[ ]:


online_orders = ((main_df["online_order"].value_counts()/main_df["online_order"].count())*100).round(2)
print(online_orders)
sn.barplot(x = online_orders.index, y = online_orders)
plt.title("Percentage of outlets accepting online orders", fontsize = 16)
plt.ylabel("Percentage %", fontsize = 14)
plt.xticks(fontsize = 14)
plt.show()


# Majority or approximately 60% of the outlets did accept online orders

# #### Outlets providing table booking service

# In[ ]:


book_table = (main_df['book_table'].value_counts() * 100 /main_df['book_table'].count()).round(2)
print(book_table)
sn.barplot(x = book_table.index, y = book_table)
plt.title("Percentage of outlets with Table booking service", fontsize = 16)
plt.ylabel("Percentage %", fontsize = 14)
plt.xticks(fontsize = 14)
plt.show()


# More than 85% of the outlets did not provide any table booking service

# #### Observations on the ratings of the outlets

# In[ ]:


print(res_df["ratings"].describe())
plt.figure(figsize = (12,4))
plt.subplot(1,2,1)
sn.distplot(res_df.ratings, hist = False)
plt.subplot(1,2,2)
sn.boxplot(res_df["ratings"])
plt.show()
potential_outliers_count = res_df[res_df["ratings"]<2.5]["ratings"].count()
print(f"Total number of potential outliers with ratings less than 2.5 are {potential_outliers_count}")


# 1. The mean rating of all the outlets on zomato is 3.7 which can be considered a decent average
# 2. We can say that the quality of the restraunts is good in Bangalore
# 2. On the basis of this metric zomato can drop such outlets from its platform.

# #### Percentage of Top rated outlets for each type of Outlet

# In[ ]:


top_rated = res_df[res_df.sort_values(by = ["listed_in(type)","ratings"], ascending = [True, False])["ratings"]>4.0]
all_outlets = res_df.sort_values(by = ["listed_in(type)"])
all_outlets_count = all_outlets["listed_in(type)"].value_counts()
top_rated_count = top_rated["listed_in(type)"].value_counts()
perc_by_each_type = (top_rated_count*100/all_outlets_count).round(2)
perc_by_each_type.sort_values(ascending = False, inplace = True)
perc_by_each_type


# In[ ]:


print(all_outlets_count)
sn.barplot(y = perc_by_each_type.index, x = perc_by_each_type,color = "blue", alpha = 0.5)
plt.title("Percentage of Top (more than 4) rated outlets in respective categories ", fontsize = 16)
plt.xlabel("Outlet Types", fontsize = 14)
plt.ylabel("Percentage", fontsize = 14)
plt.show()


# 1. High Percentage of outlets in Pubs and Bars, Drinks & Nightlife, Buffet and cafe categories received better ratings.
# 2. Less than 30% of outlets in categories Delivery, Dine-out and Desserts received high ratings.
# 3. Performace of Pubs and Bars, Drinks & Nightlife, Buffets and cafes is better than Delivery, Dine-out and Dessert Outlets.

# #### Number of Outlets in each location

# In[ ]:


number_of_outlets = main_df["listed_in(city)"].value_counts()
plt.figure(figsize = (16,10))
sn.barplot(y = number_of_outlets.index, x = number_of_outlets, color = "red", alpha = 0.5)
plt.title("Total number of outlets in each location", fontsize = 18)
plt.xlabel("Locations in city", fontsize = 15)
plt.ylabel("Number of Outlets", fontsize = 15)
plt.show()


# Maximum number of outlets are established in BTM and Kormangla (Block 4-7) which states that majority of the foodies are in these areas. 

# ##### Location wise distribution of Outlets with average ratings

# In[ ]:


fig = px.treemap(res_df, path = ["listed_in(city)","listed_in(type)"], 
                 color = "ratings",
                 height = 800,
                 title = "Plot showing Number of outlets in each category for each location ")
fig.show()


# #### Distribution of cost for two column

# In[ ]:


#Plotting a distribution graph for the cost for two column

plt.figure(figsize = (18,14))
temp_df = main_df["approx_cost(for two people)"].value_counts().sort_index()
plt.subplot(2,2,1)
sn.distplot(temp_df)
plt.xticks(rotation = 30 , fontsize = 12)
plt.xlabel("Cost", fontsize = 14)
plt.title("Cost for two - Distribution", fontsize = 16)

#Plotting a bar plot for the cost for two column 

temp_df2 = main_df["cost_for_two"].value_counts()
plt.subplot(2,2,2)
sn.barplot(x = temp_df2.index, y = temp_df2, color = "lightseagreen")
plt.xticks(rotation = 30 , fontsize = 12)
plt.xlabel("Cost", fontsize = 14)
plt.ylabel("Number of Outlets", fontsize = 14)
plt.title("Cost for two - Number of Outlets", fontsize = 16)

plt.subplot(2,2,3)
sn.boxplot(main_df["approx_cost(for two people)"])
plt.xlabel("Cost", fontsize = 14)
plt.title("Cost for two", fontsize = 16)
plt.show()

print("The Median cost for Meal for two in Bangalore is ", main_df["approx_cost(for two people)"].median())


# 1. As the distribution is right skewed we say that the majority of restaurants offer meals for two in the range of 0 to 1500.
# 2. Very few outlets cost more than 2000 for two people.

# #### Cost for two depending on the type of outlet

# In[ ]:


fig = px.box(main_df, x = "listed_in(type)", y = "approx_cost(for two people)")
fig.update_layout(title = dict(text = "Cost for two for different types of outlets",
                                font = dict(size = 16, color = "black"),
                              x = 0.5),
                 xaxis = dict(title = "Type of Outlet"),
                 yaxis = dict(title = "Cost for Two"))
fig.show()


# - Outlets in categories - Buffet, Dine-out, Drinks & nightlife and Pubs & Bars are the only ones where the cost for two people goes beyond 3000. 
# - Median Cost for two people for home delivery, desserts and Dine is least in comparison to other outlet types.
# - Meal for two costs above 5000 only in dineout restaurants given the hospitality, service and ambience in such outlets.
# - Median cost for two is highest for outlets listed in Drinks & Nightlife and Pubs and Bar due to the premium charged on the drinks (potentially).
# - Cost for two in Desserts outlets lies generally between 200 - 500.

# In[ ]:


fig = px.box(main_df, x = "listed_in(city)", y = "approx_cost(for two people)")
fig.update_layout(title = dict(text = "Cost for two people in different locations",
                              font = dict(size = 16, color = "black"),
                              x = 0.5),
                 xaxis = dict(title = "Locations"),
                 yaxis = dict(title = 'Cost for two people'))
fig.add_shape( # add a horizontal "target" line
    type="line", line_color="purple", line_width=2, opacity=0.5, line_dash="dot",
    x0=0, x1=1, xref="paper", y0=3000, y1=3000, yref="y")
fig.show()


# - Majority of the high end outlets are in areas such as Brigade Road, Church street, Indiranagar, Malleshwaram, Lavelle Road, etc.
# - Surprisingly high end outlets in Kormangala are None.
# - Areas such as Jayanagar, JP nagar, Banashankari, kalyan nagar are swarmed with budget restaurants.

# #### Distribution of Outlets accepting online orders

# In[ ]:


fig = px.histogram(main_df, x = "listed_in(type)",color = "online_order", 
                   log_y = False, color_discrete_sequence= ["darkcyan",'lightseagreen'])
fig.update_layout(title = dict(text = "Distribution of outlets accepting online orders",
                  x = 0.5,
                  font = dict(size = 16)),
                  yaxis = dict(title = "Count",type = "log", nticks = 3),
                 xaxis = dict(title = 'Outlet Types'))
fig.show()


# - Majority of the Delivery, Cafes, Dine out and Dessert outlets accept online orders
# - Very few outlets listed in Drinks and Nightlife and Pubs & Bars accept online order.

# #### Outlets offering Table Booking Service

# In[ ]:


fig = px.histogram(main_df, x = "listed_in(type)", color = "book_table", log_y= True, 
                   color_discrete_sequence=["indigo","darkorchid"])
fig.update_layout(title = dict(text = "Outlets offering Table booking Serive",
                              font = dict(size = 16),
                              x = 0.5),
                               yaxis = dict(title = "Count",
                                           type = "log",
                                           nticks = 3),
                               xaxis = dict(title = "Type of Outlet"))
fig.show()


# Majority of outlets in Buffet, Drinks & Nightlife and Pubs & Bars category offer table booking service. On the other end majority of cafes and dessert outlets do not offer such service.

# #### Analysis of different types of outlets based on ratings

# In[ ]:


fig = px.box(res_df, x = "listed_in(type)", y = 'ratings', color_discrete_sequence=["darkcyan"])
fig.update_layout(title = dict(text = "Analysis of different types of outlets based on Ratings",
                              x = 0.5,
                              font = dict(size = 18)),
                 xaxis = dict(title = "Type of Outlet"),
                 yaxis = dict(title = "Rating"))
fig.show()


# - Median rating of outlets in Buffet, Drinks & Nightlife and Pubs and Bars is above 4.

# #### Analysis Cost for two Vs Ratings

# In[ ]:


fig = px.box(res_df , x = "cost_for_two", y = "ratings")
fig.update_layout(title = dict(text = "Analysis of Cost for two VS Ratings",
                              x = 0.5,
                              font = dict(size = 18)),
                 xaxis = dict(title = "Cost for Two"),
                 yaxis = dict(title = "Rating"))
fig.show()


# ##### Spearman correlation Between Cost for two and Ratings<br>
# It is useful in evaluating correlation between a rank or ordinal variable (cost_for_two) and a continuous variable (ratings)

# In[ ]:


df = res_df.groupby("cost_for_two")["ratings"].median()
df.sort_values(ascending = True, inplace = True)
df = pd.DataFrame(df)
df["Rank"] = [2,1,3,4,5,6]
df


# In[ ]:


sn.regplot(df["Rank"], df["ratings"])


# In[ ]:


from scipy import stats


# In[ ]:


stats.spearmanr(df)


# - There is a strong correlation between Median rating and the cost of meal for two. Median rating for outlets is increasing with increase in cost of meal for two people.
# - Hence, we can say that if the cost for two in a restaurant is more than 3000 then the probability of having a good meal is high.

# #### Analysis based on Cuisines

# ##### Extracting unique cuisine types

# In[ ]:


cuisine = " "
for i in main_df["cuisines"]:
    for j in i.split(","):
        cuisine += j + ","
    cuisine += " "  
cuisine = list(set(cuisine.split(",")))
cuisine = [i.strip() for i in cuisine]
cuisine.remove("")
print("Total number of different cuisines is",len(cuisine))


# ##### Number of outlets serving each cuisine

# In[ ]:


df = {}
for i in cuisine:
    df[i] = len(main_df[main_df["cuisines"].str.contains(i)==True]["cuisines"])
df = pd.DataFrame(data = [list(df.keys()), list(df.values())], index = ["cuisine_type","num_of_outlets"]).T
df.sort_values("num_of_outlets", ascending = False, inplace = True)
df


# ##### Popularity of Cuisines displayed using Graph

# In[ ]:


fig = px.bar(df, "cuisine_type", "num_of_outlets", log_y= True)
fig.update_layout(title = dict(text = "Popularity of Different Cuisines",
                             font = dict(size = 18),
                             x = 0.5),
                xaxis = dict(title = "Cuisine Type"),
                yaxis = dict(title = "Number of Outlets", type = "log", nticks = 5))
fig.show()


# - Most famous cuisines in Bangalore are Indian, North Indian and Chinese with more than 15K outlets serving these cuisines.
# - Even though Bangalore is situated in South India, huge number of outlets serving North Indian and Chinese cuisine signifies that Bangalore with swarmed with people from Northern India given that it is the Silicon Valley of India.
# - There are many outlets serving various International and regional cuisines for people from different parts of the world.It indicates that Bangalore attracts people from across the world.

# #### Famous restaurant chains in different categories

# In[ ]:


famous_chains = main_df.groupby("listed_in(type)")
famous_chains = famous_chains.apply( lambda x: x["Name"].value_counts()).reset_index(drop = False)
famous_chains.rename(columns = {"level_1" : "name", "Name" : "total_outlets"}, inplace = True)
famous_chains = famous_chains.groupby("listed_in(type)").head(5)
px.treemap(famous_chains, path = ["listed_in(type)","name"], color = "total_outlets")


# #### Analysis of New Outlets

# In[ ]:


fig = px.histogram(new_rest_df,"listed_in(type)",color = "cost_for_two", 
                   log_y = True,
                  color_discrete_sequence= ["darkcyan","lightseagreen", "darkturquoise", "cadetblue","mediumturquoise"])
fig.update_layout(title = dict(text = "New Outlets Opened in different Categories",
                              font = dict(size = 18),
                              x = 0.5),
                 yaxis = dict(title = "Number of Outlets",
                             type = "log",
                             nticks = 5),
                 xaxis = dict(title = "Outlet Type"))
fig.show()


# - Maximum number of outlets have opened under Delivery type. This may be due to increase in the number of young crowd of students and working professionals.
# - Least number of outlets have opened in Buffets category.
# - Also maximum outlets serve meals within the price range 200-500 for two people.
# - No new outlet costs more than 3000 for an average meal for two people.

# #### Names of 5 Most Famous outlets in each catgories

# In[ ]:


famous = res_df.sort_values(["listed_in(type)","votes"], ascending = [True,False]).groupby("listed_in(type)")

#Removing all the multiple outlets of different chains
famous = famous.apply(lambda x: x.drop_duplicates(subset = "Name")).reset_index(drop = True)


# In[ ]:


famous.groupby("listed_in(type)").head(5)[["Name","ratings","votes","listed_in(type)","listed_in(city)"]]


# #### Names of 5 Most Famous outlets in each catgories sorted by location

# In[ ]:


famous = famous.groupby("listed_in(type)")
famous_df = famous.head(5)[["Name","ratings","votes","listed_in(type)","listed_in(city)"]].sort_values("listed_in(city)")
famous_df


# In[ ]:


px.treemap(famous_df, path = ["listed_in(city)","listed_in(type)","Name"], hover_data = ["ratings","Name"], color = "votes")


# ### Key Takeaways

# - More than 60% of the restaurants provided online ordering service. Delivery, Dine Out, Desserts and cafes were main categories in which restaurants provided this facility.
# - Near about only 13% outlets provided table booking service which includes majorly restaurants listed under Pubs and Bars, Nightlife & Drinks, Fine Dining and Buffets.
# - Huge supply of Indian, North Indian and chinese cuisine shows that Bangalore is swarmed with large number of migrants from other parts of India.
# - Outlets in Bangalore have a decent average rating which shows that food is not an issue in Bangalore.
# - Median cost for meal for two is 400 which is quite inexpensive.
# - BTM, Kormangala (Block 4-7), Jayanagar have high number of outlets, which gives us an insight into the demographics of people living in these areas, real estate prices, traffic, infrastructure and other facilities.
# - The median cost for two people is minimum for delivery outlets and maximum for outlets in Drinks and Nightlife.
# - There is a strong correlation between Median rating and the cost of meal for two. Median rating for outlets is increasing with increase in cost of meal for two people.
# - Maximum number of new outlets have opened under Delivery type. This may be due to the increasing young crowd of students and working professionals.
# 
# 

# In[ ]:




