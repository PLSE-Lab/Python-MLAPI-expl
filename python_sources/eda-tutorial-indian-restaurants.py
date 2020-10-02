#!/usr/bin/env python
# coding: utf-8

# # EDA Tutorial - Indian Restaurants 
# 
# ## Introduction
# 
# Exploratory Data Analysis (EDA) is a preliminary step of Machine Learning and is used extensively in this field. Although it is not necessary to perform EDA to build models, but it is definitely recommended as it helps to know the data better. If performed correctly, it gives us insights which are not easy to witness directly. 
# 
# In this notebook, I have performed a detailed analysis on Indian Restaurants Dataset from Zomato(<a href="https://www.kaggle.com/rabhar/zomato-restaurants-in-india">link</a>). This notebook can be used as a manual to perform basic to intermediate EDA on any dataset. Following are the things that you will learn from this project :-
# 1. Knowing basic composition of data
# 2. Removing duplicates
# 3. Dealing with missing values
# 4. Understanding features
# 5. Plotting horizontal bar charts (multicolor)
# 6. Using groupby, apply, and unique functions 
# 7. Scatter plot
# 8. Word Cloud
# 9. Box plot
# 10. Density plot
# 11. Bar Charts
# 12. Drawing insights and conclusions from data
# 
# Don't forget to upvote if you find this useful! :)
# 
# So without further ado, let's get started!
# 
# ## Project outline

# - Importing 
# - Preprocessing
# - - Exploring data
# - - Removing duplicates
# - - Dealing with missing values
# - - Omitting not useful features
# - EDA
# - - Restaurant Chains
# - - - Chains vs Outlets
# - - - Top Restaurant Chains (by number of outlets)
# - - - Top Restaurant Chains (by average ratings)
# - - Establishment Types
# - - - Number of Restaurants 
# - - - Average Rating, Votes, and Photo count
# - - Cities
# - - - Number of Restaurants 
# - - - Average Rating, Votes, and Photo count
# - - Cuisine
# - - - Total number of unique cuisines
# - - - Number of Restaurants
# - - - Highest rated cuisines
# - - Highlights 
# - - - Number of Restaurants
# - - - Highest rated features
# - - - Highlights wordcloud
# - - Rating and cost
# - - - Rating Distribution
# - - - Average Cost for two distribution
# - - - Price range count
# - - - Relation between Average price for two and Rating
# - - - Relation between Price Range and Rating
# - - - Relation between Votes and Rating
# - Conclusions

# ## Importing necessary libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import random
from wordcloud import WordCloud


# ## Preprocessing
# ### Exploring data

# In[ ]:


data = pd.read_csv("../input/zomato-restaurants-in-india/zomato_restaurants_in_India.csv")


# In[ ]:


data.head(3)


# In[ ]:


data.shape


# Our dataset has 26 features and 0.2 million plus rows. Let's find out more about these features using the <b>info( )</b> function

# In[ ]:


data.info()


# We have many interesting features which can be great for analysis and also some which we will omit. The difference in count of some features gives us hint of some missing values. 
# While <b>info( )</b> is used to know about count, null and type properties, <b>describe( )</b> gives us statistical information about numerical data.

# In[ ]:


data.describe()


# ### Removing duplicates
# Its important to remove duplicate rows to avoid biasness in our analysis. Since res_id is unique identifier of our restaurants, we can use it to remove duplicates.

# In[ ]:


data.drop_duplicates(["res_id"],keep='first',inplace=True)
data.shape


# Oops! Looks like almost 75% of our data had duplicate rows. Its good that we got that out before getting started. Even though we are left with 1/4th of our original dataset, about 55000+ restaurants is still good enough to perform analysis.
# ### Dealing with missing values
# Now let's see how many variables have missing values.

# In[ ]:


data.isnull().sum()


# We have 5 variables with some kind of missing values. 
# Since zipcode has ~80% missing data, its better to not consider it at all. The other 4 features can be delt with some kind of imputation, but before going through the trouble, its better to look and decide whether they would be beneficial for our analysis or we can simply omit them.
# ### Omitting not useful features
# Here we will look at each feature and decide to consider them for our analysis or not:- 
# 1. <b>res_id</b> - Unique ID for each restaurant
# 2. <b>name</b> - Name is useful since we will use it to find top restaurants
# 3. <b>establishment</b> - Let's see what type of values we have in establishment

# In[ ]:


data["establishment"].unique()


# In[ ]:


print(data["establishment"].unique()[0])
print(type(data["establishment"].unique()[0]))


# Establishment looks like a nice feature to perform EDA, however each value has an unwanted square brackets and quotes which seems noisy. Let's remove them with <b>apply( )</b> function.
# Also, we have one value which is an empty string, let's rename it to "NA" to avoid confusion.

# In[ ]:


# Removing [' '] from each value
print(data["establishment"].unique()[0])
data["establishment"] = data["establishment"].apply(lambda x:x[2:-2])
print(data["establishment"].unique()[0])

# Changing ''  to 'NA'
print(data["establishment"].unique())
data["establishment"] = data["establishment"].apply(lambda x : np.where(x=="", "NA", x))
print(data["establishment"].unique())


# 4. <b>url</b> - URL is the link to restaurant's page which is not useful for us
# 5. <b>address</b> - Not useful since it has long strings and its difficult to classify
# 6. <b>city</b> - Let's check unique cities

# In[ ]:


data["city"].unique()


# Look's good. 
# 7. <b>city_id</b> - We can uniquely use city name or id. So one feature is enough
# 8. <b>locality</b> - Let's see number of unique values

# In[ ]:


data["locality"].nunique()


# Although it can be an interesting feature, but since this feature has so many unique classes, we will avoid it.
# 9. <b>latitude</b> - Can be helpful while using geographic maps, but we won't be doing that here
# 10. <b>longitude</b> - Same as above
# 11. <b>zipcode</b> - Approx 80% missing values
# 12. <b>country_id</b> - Since this dataset is for Indian restaurants, there should be just one unique id here. Let's check.

# In[ ]:


data["country_id"].unique()


# 13. <b>locality_verbose</b> - Same as locality 

# In[ ]:


data["locality_verbose"].nunique()


# 14. <b>cuisines</b> - This feature has some missing values. Even though this has 9382 unique classes, we can see that each restaurant has a list of cusinies and the composition of the list is the reason why we have so many different cuisine classes. Let's check actual number of unique cuisine classes. But first we need to replace null values with a label.

# In[ ]:


print(data["cuisines"].nunique())
print(data["cuisines"].unique())


# In[ ]:


data["cuisines"] = data["cuisines"].fillna("No cuisine")


# In[ ]:


cuisines = []
data["cuisines"].apply(lambda x : cuisines.extend(x.split(", ")))
cuisines = pd.Series(cuisines)
print("Total number of unique cuisines = ", cuisines.nunique())


# 15. <b>timings</b> - This also has missing data, however it has 7740 unique classes. Also, it is not structured even if we try to reduce the number classes like we did in cuisines. Its better to omit it altogether.

# In[ ]:


print(data["timings"].nunique())
print(data["timings"].unique())


# 16. <b>average_cost_for_two</b> - This is an interesting feature for our analysis, although the value "0" is strange and should be an outlier

# In[ ]:


data["average_cost_for_two"].unique()


# 17. <b>price_range</b> - Average prices automatically characterized into bins

# In[ ]:


data["price_range"].unique()


# 18. <b>currency</b> - Only one class. Not useful

# In[ ]:


data["currency"].unique()


# 19. <b>highlights</b> - They represent certain features that the restaurant specializes in and wants to highlight to their customers. Each restaurant has a list of highlights which makes the composition different for each one. We can, filter this and find total unique highlights from all restaurants.

# In[ ]:


print(data["highlights"].nunique())
print(data["highlights"].unique())


# In[ ]:


hl = []
data["highlights"].apply(lambda x : hl.extend(x[2:-2].split("', '")))
hl = pd.Series(hl)
print("Total number of unique highlights = ", hl.nunique())


# 20. <b>aggregate_rating</b> - Rating given to the restaurant
# 21. <b>rating_text</b> - Characterisation of numeric rating into bins by using labels. We will be using direct ratings in our analysis, so we can ignore this.
# 22. <b>votes</b> - Number of votes contributing to the rating
# 23. <b>photo_count</b> - Photo uploads in reviews
# 
# Let's check the mean and range of above features

# In[ ]:


data[["aggregate_rating","votes","photo_count"]].describe().loc[["mean","min","max"]]


# Rating ranges between 0 and 5 while 42539 are the maximum votes given to a restaurant. The negative value in votes might be an outlier.
# 24. <b>opentable_support</b> - Not useful since no restaurant has True value for this
# 25. <b>delivery</b> - This feature has 3 classes but there is no explanation for those classes. We can consider -1 and 0 to be one class or ignore this feature for now
# 26. <b>takeaway</b> - Again not useful since it only has one class

# In[ ]:


data["opentable_support"].unique()


# In[ ]:


data["delivery"].unique()


# In[ ]:


data["takeaway"].unique()


# Now that we have taken a deep look at our data, let's start with some EDA!

# ## Exploratory Data Analysis (EDA)
# ### Restaurant chains
# Here chains represent restaurants with more than one outlet
# #### Chains vs Outlets

# In[ ]:


outlets = data["name"].value_counts()


# In[ ]:


chains = outlets[outlets >= 2]
single = outlets[outlets == 1]


# In[ ]:


print("Total Restaurants = ", data.shape[0])
print("Total Restaurants that are part of some chain = ", data.shape[0] - single.shape[0])
print("Percentage of Restaurants that are part of a chain = ", np.round((data.shape[0] - single.shape[0]) / data.shape[0],2)*100, "%")


# <b>35%</b> of total restaurants are part of some kind of restaurant chain. Here, we should account for cases where two different retaurants might have exact same name but are not related to each other.
# #### Top restaurant chains (by number of outlets)
# Let's plot a horizontal bar graph to look at Top 10 restaurant chains. For the color scheme, we are using a list of pre-defined and selected colours to make the chart more appealing. If you want your analysis to look good visually, you should customize each and every element of your graph.

# In[ ]:


top10_chains = data["name"].value_counts()[:10].sort_values(ascending=True)


# In[ ]:


height = top10_chains.values
bars = top10_chains.index
y_pos = np.arange(len(bars))

fig = plt.figure(figsize=[11,7], frameon=False)
ax = fig.gca()
ax.spines["top"].set_visible("#424242")
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#424242")
ax.spines["bottom"].set_color("#424242")

#colors = ["green","blue","magenta","cyan","gray","yellow","purple","violet","orange","red","maroon"]
#random.shuffle(colors)
colors = ["#f9cdac","#f2a49f","#ec7c92","#e65586","#bc438b","#933291","#692398","#551c7b","#41155e","#2d0f41"]
plt.barh(y_pos, height, color=colors)
 
plt.xticks(color="#424242")

plt.yticks(y_pos, bars, color="#424242")
plt.xlabel("Number of outlets in India")

for i, v in enumerate(height):
    ax.text(v+3, i, str(v), color='#424242')
plt.title("Top 10 Restaurant chain in India (by number of outlets)")


plt.show()


# This chart is majorly dominaed by big fast food chains
# #### Top restaurant chains (by average rating)
# Here we will look at top chains by their ratings. I have set the criteria of number of outlets to greater than 4 to remove some outliers.

# In[ ]:


outlets = data["name"].value_counts()


# In[ ]:


atleast_5_outlets = outlets[outlets > 4]


# In[ ]:


top10_chains2 = data[data["name"].isin(atleast_5_outlets.index)].groupby("name").mean()["aggregate_rating"].sort_values(ascending=False)[:10].sort_values(ascending=True)


# In[ ]:


height = pd.Series(top10_chains2.values).map(lambda x : np.round(x, 2))
bars = top10_chains2.index
y_pos = np.arange(len(bars))

fig = plt.figure(figsize=[11,7], frameon=False)
ax = fig.gca()
ax.spines["top"].set_visible("#424242")
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#424242")
ax.spines["bottom"].set_color("#424242")

#colors = ["green","blue","magenta","cyan","gray","yellow","purple","violet","orange","red","maroon"]
#random.shuffle(colors)
colors = ['#fded86', '#fce36b', '#f7c65d', '#f1a84f', '#ec8c41', '#e76f34', '#e25328', '#b04829', '#7e3e2b', '#4c3430']
plt.barh(y_pos, height, color=colors)

plt.xlim(3)
plt.xticks(color="#424242")
plt.yticks(y_pos, bars, color="#424242")
plt.xlabel("Number of outlets in India")

for i, v in enumerate(height):
    ax.text(v + 0.01, i, str(v), color='#424242')
plt.title("Top 10 Restaurant chain in India (by average Rating)")


plt.show()


# Interestingly, no fast food chain appears in this chart. To maintain a high rating, restaurants needs to provide superior service which becomes impossible with booming fast food restaurant in every street.
# 
# ### Establishment Types
# #### Number of restaurants (by establishment type)

# In[ ]:


est_count = data.groupby("establishment").count()["res_id"].sort_values(ascending=False)[:5]

fig = plt.figure(figsize=[8,5], frameon=False)
ax = fig.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#424242")
ax.spines["bottom"].set_color("#424242")

#colors = ["green","blue","magenta","cyan","gray","yellow","purple","violet","orange","red","maroon"]
#random.shuffle(colors)
colors = ["#2d0f41",'#933291',"#e65586","#f2a49f","#f9cdac"]
plt.bar(est_count.index, est_count.values, color=colors)

plt.xticks(range(0, 6), color="#424242")
plt.yticks(range(0, 25000, 5000), color="#424242")
plt.xlabel("Top 5 establishment types")

for i, v in enumerate(est_count):
    ax.text(i-0.2, v+500, str(v), color='#424242')
plt.title("Number of restaurants (by establishment type)")


plt.show()


# Top 3 represents more casual and quick service restaurants, then from 4-6 we have dessert based shops.
# #### Average rating, votes and photos (by Establishment)
# Here, we will not plot each graph since it will make this notebook filled with horizontal bar charts. I see horizontal bar charts the only option to display results of this kind when we have lots of classes to compare (here 10 classes). Let's look at <b>value_counts( )</b> directly

# In[ ]:


rating_by_est = data.groupby("establishment").mean()["aggregate_rating"].sort_values(ascending=False)[:10]
rating_by_est


# In[ ]:


# To check the number of outlets in each of the above establishment type, uncomment to following code

#est_count = data.groupby("establishment").count()["name"].sort_values(ascending=False)
#rating_by_est_map = est_count.index.isin(rating_by_est.index)
#est_count = est_count[rating_by_est_map][rating_by_est.index]
#est_count


# In[ ]:


data.groupby("establishment").mean()["votes"].sort_values(ascending=False)[:10]


# In[ ]:


data.groupby("establishment").mean()["photo_count"].sort_values(ascending=False)[:10]


# It can be concluded that establishments with alcohol availability have highest average ratings, votes and photo uploads.
# 
# ### Cities
# #### Number of restaurants (by city)

# In[ ]:


city_counts = data.groupby("city").count()["res_id"].sort_values(ascending=True)[-10:]

height = pd.Series(city_counts.values)
bars = city_counts.index
y_pos = np.arange(len(bars))

fig = plt.figure(figsize=[11,7], frameon=False)
ax = fig.gca()
ax.spines["top"].set_visible("#424242")
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#424242")
ax.spines["bottom"].set_color("#424242")

#colors = ["green","blue","magenta","cyan","gray","yellow","purple","violet","orange","red","maroon"]
#random.shuffle(colors)
colors = ['#dcecc9', '#aadacc', '#78c6d0', '#48b3d3', '#3e94c0', '#3474ac', '#2a5599', '#203686', '#18216b', '#11174b']
plt.barh(y_pos, height, color=colors)

plt.xlim(3)
plt.xticks(color="#424242")
plt.yticks(y_pos, bars, color="#424242")
plt.xlabel("Number of outlets")

for i, v in enumerate(height):
    ax.text(v + 20, i, str(v), color='#424242')
plt.title("Number of restaurants (by city)")


plt.show()


# As expected, metro cities have more number of restaurants than others with South India dominating the Top 4
# 
# #### Average rating, votes and photos (by city)

# In[ ]:


rating_by_city = data.groupby("city").mean()["aggregate_rating"].sort_values(ascending=False)[:10]
rating_by_city


# In[ ]:


# To check the number of outlets in each of the above establishment type

#city_count = data.groupby("city").count()["name"].sort_values(ascending=False)
#rating_by_city_map = city_count.index.isin(rating_by_city.index)
#city_count = city_count[rating_by_city_map][rating_by_city.index]
#city_count


# In[ ]:


data.groupby("city").mean()["votes"].sort_values(ascending=False)[:10]


# In[ ]:


data.groupby("city").mean()["photo_count"].sort_values(ascending=False)[:10]


# Gurgaon has highest rated restaurants whereas Hyderabad has more number of critics. Mumbai and New Delhi dominates for most photo uploads per outlet.
# 
# ### Cuisine
# #### Unique cuisines

# In[ ]:


print("Total number of unique cuisines = ", cuisines.nunique())


# #### Number of restaurants (by cuisine)

# In[ ]:


c_count = cuisines.value_counts()[:5]

fig = plt.figure(figsize=[8,5], frameon=False)
ax = fig.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#424242")
ax.spines["bottom"].set_color("#424242")

#colors = ["green","blue","magenta","cyan","gray","yellow","purple","violet","orange","red","maroon"]
#random.shuffle(colors)
colors = ['#4c3430', '#b04829', '#ec8c41', '#f7c65d','#fded86']
plt.bar(c_count.index, c_count.values, color=colors)

plt.xticks(range(0, 6), color="#424242")
plt.yticks(range(0, 30000, 5000), color="#424242")
plt.xlabel("Top 5 cuisines")

for i, v in enumerate(c_count):
    ax.text(i-0.2, v+500, str(v), color='#424242')
plt.title("Number of restaurants (by cuisine type)")


plt.show()


# Surprisingly, Chinese food comes second in the list of cuisines that Indians prefer, even more than fast food, desserts and South Indian food.
# #### Highest rated cuisines

# In[ ]:


data["cuisines2"] = data['cuisines'].apply(lambda x : x.split(", "))

cuisines_list = cuisines.unique().tolist()
zeros = np.zeros(shape=(len(cuisines_list),2))
c_and_r = pd.DataFrame(zeros, index=cuisines_list, columns=["Sum","Total"])


# In[ ]:


for i, x in data.iterrows():
    for j in x["cuisines2"]:
        c_and_r.loc[j]["Sum"] += x["aggregate_rating"]  
        c_and_r.loc[j]["Total"] += 1


# In[ ]:


c_and_r["Mean"] = c_and_r["Sum"] / c_and_r["Total"]
c_and_r


# In[ ]:


c_and_r[["Mean","Total"]].sort_values(by="Mean", ascending=False)[:10]


# We can ignore a few cuisines in this list since they are available in less number. But the overall conclusion which can be drawn is that International (and rarely available) cuisines are rated higher than local cuisines.
# ### Highlights/Features of restaurants
# #### Unique highlights

# In[ ]:


print("Total number of unique cuisines = ", hl.nunique())


# #### Number of restaurants (by highlights)

# In[ ]:


h_count = hl.value_counts()[:5]

fig = plt.figure(figsize=[10,6], frameon=False)
ax = fig.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#424242")
ax.spines["bottom"].set_color("#424242")

#colors = ["green","blue","magenta","cyan","gray","yellow","purple","violet","orange","red","maroon"]
#random.shuffle(colors)
colors = ['#11174b', '#2a5599', '#3e94c0', '#78c6d0', '#dcecc9']
plt.bar(h_count.index, h_count.values, color=colors)

plt.xticks(range(0, 6), color="#424242")
plt.yticks(range(0, 70000, 10000), color="#424242")
plt.xlabel("Top 5 highlights")

for i, v in enumerate(h_count):
    ax.text(i-0.2, v+500, str(v), color='#424242')
plt.title("Number of restaurants (by highlights)")


plt.show()


# Top 5 highlights doesn't convey much information since they are very trivial to almost every restaurant. Let's look at uncommon highlights that matter more to the customers.
# #### Highest rated highlights

# In[ ]:


data["highlights"][0]


# In[ ]:


data["highlights2"] = data['highlights'].apply(lambda x : x[2:-2].split("', '"))

hl_list = hl.unique().tolist()
zeros = np.zeros(shape=(len(hl_list),2))
h_and_r = pd.DataFrame(zeros, index=hl_list, columns=["Sum","Total"])


# In[ ]:


for i, x in data.iterrows():
    for j in x["highlights2"]:
        h_and_r.loc[j]["Sum"] += x["aggregate_rating"]  
        h_and_r.loc[j]["Total"] += 1


# In[ ]:


h_and_r["Mean"] = h_and_r["Sum"] / h_and_r["Total"]
h_and_r


# In[ ]:


h_and_r[["Mean","Total"]].sort_values(by="Mean", ascending=False)[:10]


# We can safely ignore highlights which have a frequency of less than 10 since they can be considered as outliers. Features like Gastro pub, Craft beer, Romantic dining and Sneakpeek are well received among customers.
# 
# #### Highlights wordcloud
# Here we will create a wordcloud of top 30 highlights

# In[ ]:


# https://www.geeksforgeeks.org/generating-word-cloud-python/

hl_str = ""
for i in hl:
    hl_str += str(i) + " "
wordcloud = WordCloud(width = 800, height = 500, 
                background_color ='white', 
                min_font_size = 10, max_words=30).generate(hl_str) 
                         
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show()


# ### Ratings and cost
# #### Ratings distribution
# Let's see how the ratings are distributes

# In[ ]:


sns.kdeplot(data['aggregate_rating'], shade=True)
plt.title("Ratings distribution")
plt.show()


# There is a huge spike at 0 which might account for newly opened or unrated restaurants. On average, majority of restaurants have rating between 3 to 4 with fewer restaurants managing to go beyond 4.
# #### Avergae cost for two distribution

# In[ ]:


sns.kdeplot(data['average_cost_for_two'], shade=True)
plt.title("Average cost for 2 distribution")
plt.show()


# With few restaurants charging average of Rs.25000+ for two, this graph is extremely skewed. Let's take a closer look at a lower range of 0 to 60000.

# In[ ]:


sns.kdeplot(data['average_cost_for_two'], shade=True)
plt.xlim([0, 6000])
plt.xticks(range(0,6000,500))
plt.title("Average cost for 2 distribution")
plt.show()


# Majority of restaurants are budget friendly with an average cost between Rs.250 to Rs.800 for two.
# 
# #### Price range count

# In[ ]:


pr_count = data.groupby("price_range").count()["name"]

fig = plt.figure(figsize=[8,5], frameon=False)
ax = fig.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#424242")
ax.spines["bottom"].set_color("#424242")

#colors = ["green","blue","magenta","cyan","gray","yellow","purple","violet","orange","red","maroon"]
#random.shuffle(colors)
colors = ["#2d0f41",'#933291',"#f2a49f","#f9cdac"]
plt.bar(pr_count.index, pr_count.values, color=colors)

plt.xticks(range(0, 5), color="#424242")
plt.yticks(range(0, 40000, 5000), color="#424242")
plt.xlabel("Price Ranges")

for i, v in enumerate(pr_count):
    ax.text(i+0.85, v+700, str(v), color='#424242')
plt.title("Number of restaurants (by price ranges)")


plt.show()


# Price range chart supports our previous observation from the Average cost chart. Number of restaurant decreases with increase in price range.
# 
# #### Relation between Average price for two and Rating

# In[ ]:


np.round(data[["average_cost_for_two","aggregate_rating"]].corr()["average_cost_for_two"][1],2)


# A correlation can be seen between restaurant average cost and rating

# In[ ]:


plt.plot("average_cost_for_two","aggregate_rating", data=data, linestyle="none", marker="o")
plt.xlim([0,6000])
plt.title("Relationship between Average cost and Rating")
plt.xlabel("Average cost for two")
plt.ylabel("Ratings")
plt.show()


# There is definetely a direct relation between the two. Let's take a smaller sample to draw a clearer scatter plot.

# In[ ]:


plt.plot("average_cost_for_two","aggregate_rating", data=data.sample(1000), linestyle="none", marker="o")
plt.xlim([0,3000])
plt.show()


# This relation concludes that that as average cost for two increases, there is a better chance that the restaurant will be rated highly. Let's look at price range for a better comparison.
# 
# #### Relation between Price range and Rating

# In[ ]:


np.round(data[["price_range","aggregate_rating"]].corr()["price_range"][1],2)


# In[ ]:


sns.boxplot(x='price_range', y='aggregate_rating', data=data)
plt.ylim(1)
plt.title("Relationship between Price range and Ratings")
plt.show()


# Now, it is clear. The higher the price a restaurant charges, more services they provide and hence more chances of getting good ratings from their customers.

# ## Conclusions
# 
# After working on this data, we can conclude the following things:-
# 1. Approx. 35% of restaurants in India are part of some chain
# 2. Domino's Pizza, Cafe Coffee Day, KFC are the biggest fast food chains in the country with most number of outlets
# 3. Barbecues and Grill food chains have highest average ratings than other type of restaurants 
# 4. Quick bites and casual dining type of establishment have most number of outlets
# 5. Establishments with alcohol availability have highest average ratings, votes and photo uploads
# 6. Banglore has most number of restaurants 
# 7. Gurgaon has highest rated restaurants (average 3.83) whereas Hyderabad has more number of critics (votes). Mumbai and New Delhi dominates for most photo uploads per outlet
# 8. After North Indian, Chinese is the most prefered cuisine in India
# 9. International cuisines are better rated than local cuisines
# 10. Gastro pub, Romantic Dining and Craft Beer features are well rated by customers
# 11. Most restaurants are rated between 3 and 4
# 12. Majority of restaurants are budget friendly with average cost of two between Rs.250 to Rs.800
# 13. There are less number of restaurants at higher price ranges
# 14. As the average cost of two increases, the chance of a restaurant having higher rating increases
# 
# Now we have come to the end of this project, I hope you learned some new tricks. 
# 
# <b>Please give this notebook an upvote if you find it useful!</b>
