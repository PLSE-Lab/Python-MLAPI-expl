#!/usr/bin/env python
# coding: utf-8

# # Executive Summary
# This Exploratory Data Analysis starts with some data cleaning and then visualisation of the playstore app data.
# 
# Implications for potential [app development](http://localhost:8888/notebooks/projects/google_play_store/google_play_store.ipynb#Possible-Takeaways) and opportunities for [fraud detection](http://localhost:8888/notebooks/projects/google_play_store/google_play_store.ipynb#Fishiness?) were discussed. Specifically, some of the findings suggest there may be fake reviews or some other gaming of the system for some highly rated apps
# 
# As always, please provide candid and honest feedback :)

# # Import

# In[ ]:


import pandas as pd
import numpy as np
import time #is this required?
from re import sub
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

playstore = pd.read_csv("../input/googleplaystore.csv")
reviews = pd.read_csv("../input/googleplaystore_user_reviews.csv")


# # Quick Look Around
# I prefer printing the head of the dataframes as it gives me a better idea of the terrain. I might do a describe later.

# In[ ]:


playstore.head(2)


# In[ ]:


reviews.head(2)


# # Interesting questions
# 
# * What factors might predict a higher rating?
#     * More reviews?
#     * More installs?
#     * Genre?
#     * Category?
#     * Price?
#     * Free / Paid?
#     * Review Sentiment_Polarity? (or some kind of measure balanced against subjectivity?)
# 

# # Data Cleaning
# Before creating a scatterplot, I can see that I need to convert the string values in the "Price" variable to float to better represent cash

# In[ ]:


unique_values = playstore["Price"].unique()
unique_values.sort()
unique_values


# It appears that we have a few unusual values over and above the expected "$<dollar amount>" format.
# * '0'
# * 'Everyone'

# ## "Price" == "Everyone"
# 
# ### Investigation
# 
# Firstly, I'll take a look at the rows that contain "Everyone" as the value for "Price"

# In[ ]:


everyone = playstore[playstore['Price'] == 'Everyone']
print(everyone)


# ### Cleaning
# 
# It looks like the scraper wasn't quite on target with this app. So I'll fix that here:

# In[ ]:


broken = everyone.iloc[0]
app_name = broken["App"]
fixed = broken.shift(1)
fixed["App"] = app_name
#these value were found after finding the app on the live Playstore
fixed["Category"] = "Lifestyle"
fixed["Genres"] = "Lifestyle"
playstore.iloc[10472] = fixed
playstore.iloc[10472]


# Index 10472 is looking much nicer.
# 
# ## "Price" == "0"
# ### Investigation
# I have a strong hunch that this is the value given to free apps; I'll do a quick check here:

# In[ ]:


playstore[(playstore["Price"] == "0") & (playstore["Type"] != "Free")]


# It looks like there is one (see above) exception to the rule. I'll fix that here
# 
# (It also gives an important heads-up for another candidate for cleaning: apps with 0 reviews and, therefore, a rating of NaN)

# In[ ]:


playstore.iloc[9148] = playstore.iloc[9148].set_value("Type", "Free")


# ### Conversion of "Price" values to float

# In[ ]:


playstore["Price"] = playstore["Price"].apply(lambda x: float(sub(r'[^\d\-.]', '', x)))
unique_values = playstore["Price"].unique()
unique_values.sort()
print(unique_values)


# Alright, that's looking a lot more like currency.
# 
# Now, to take a look at the Rating data.

# In[ ]:


unique_values = playstore["Rating"].unique()
#unique_values.sort()
unique_values


# # "Rating" == NaN
# Before plotting commences, I want to deal with this funky data.   

# In[ ]:


nan_rating = playstore[playstore["Rating"].isnull()]
nan_rating.shape


# I suspect this is the case when Reviews == 0. 

# In[ ]:


nan_rating = playstore[(playstore["Rating" ].isnull()) & (playstore["Reviews"] != "0")]
nan_rating.shape


# Incorrect; there are _878_ instances of apps with a null rating but at least one review.
# 
# What to do with these? I'm just going to set them (and other NaN ratings) to "0" (and later turn all the ratings into a float).

# In[ ]:


#create a loop that iterates through nan_rating and sets the viewed to 0 and pushes them back into playstore
nan_rating_idx = list(nan_rating.index.values)

for idx  in nan_rating_idx:
    playstore.iloc[idx] = playstore.iloc[idx].set_value("Rating", 0)

nan_rating = playstore[playstore["Rating"].isnull()]
print(nan_rating.shape)

playstore["Rating"] = playstore["Rating"].astype(float)
playstore["Price"] = playstore["Price"].astype(float)


# And that has taken care of those 1,474 "Ratings" == NaN records.
# # Visualisations
# # Scatterplot of "Price" vs "Rating"

# In[ ]:


plt.figure()
plt.scatter(playstore["Price"], playstore["Rating"])
plt.xlabel("Price ($)")
plt.ylabel("Rating (Stars)")
plt.show()


# All that tells me is that there are a lot of apps below the \$30 mark
# 
# What if I plotted the average price per rating?
# ## Scatterplot of Ratings vs Average Price

# In[ ]:


average_price = playstore["Price"].groupby(playstore["Rating"]).mean()
plt.figure()
plt.scatter(average_price.index.values, average_price.values)
plt.xlabel("Rating (Stars)")
plt.ylabel("Average Price ($)")
plt.show()


# This reveals some surprising insights:
# * the average price of a 0 star (unrated) app is almost \$2 (maybe because they can't entice anyone to try it)
# * the average price of a 1 star (lowest rated) app is \$0.28
# * the average price of a 5 star (highest rated) app is \$0.37

# In[ ]:


average_price.head(5)


# In[ ]:


average_price.tail(5)


# ## Scatterplot of Price vs Average Rating
# While snooping around on the data, I noticed that there were a bunch of apps that were priced above \$50 and still rated reasonably high (above 3.5 stars). I have highlighted these in the figure below.

# In[ ]:


average_rating = playstore["Rating"].groupby(playstore["Price"]).mean()
average_rating = pd.DataFrame(data=average_rating)
average_rating = average_rating.reset_index()
average_rating.loc[(average_rating["Price"] > 50) & (average_rating["Rating"] > 3.5), "Expensive_App_Highly_Rated"] = 1
average_rating["Expensive_App_Highly_Rated"].fillna(0, inplace=True)


# In[ ]:


plt.figure(figsize=(25,12))
#sns.scatterplot(average_rating.index.values, average_rating.values, hue="Expensive_App_Highly_Rated", data=df)
sns.scatterplot(x="Price", y="Rating", hue="Expensive_App_Highly_Rated", data=average_rating)
plt.xlabel("Price Point ($)", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel("Average Rating (Stars)", fontsize=18)
plt.legend().remove()
plt.show()


# ### What are these really expensive apps?

# In[ ]:


expensive_apps = playstore[(playstore["Price"] > 50) & (playstore["Rating"] > 3.5)]
expensive_apps[["App", "Category", "Genres", "Price", "Rating", "Reviews", "Installs"]].sort_values("Price", ascending=False)


# #### Fishiness?
# Scouting out one of these "I Am Rich" type apps, from the reviews, it seems like they introduce the app as initially free then pump up the price after they have got a certain number of downloads.
# 
# Is there a way to classify the frauds from the legit apps? 
# 
# Would probably need some information on their price history.
# 
# Perhaps the subjectivity might be able to say more about this.

# In[ ]:


reviews.head(2)


# # Per Genre Analysis

# ## Preparation
# The first step is to separate the genres from each other and create dummy variables for easier querying. The dummy variables will be prefixed with "gen_".
# 
# Not sure if iterrows is the right way to do this but I couldn't work out how to do it with apply.

# In[ ]:


for name, row in playstore.iterrows():
    genres_list = row["Genres"].split(";")
    for genre in genres_list:
        playstore.loc[name, str("gen_" + genre)] = 1


# I then created a separate dataframe with the aggregate figures for price and rating for each genre.

# In[ ]:


genre_dummies = [col for col in playstore if col.startswith('gen_')]
genre_dummies.sort()

aggregate_figures_per_genre_data = {}
for genre_dummy in genre_dummies:
    average_rating = playstore[playstore[genre_dummy] == 1]["Rating"].mean()
    average_price = playstore[playstore[genre_dummy] == 1]["Price"].mean()
    count = playstore[playstore[genre_dummy] == 1]["Rating"].count()
    aggregate_figures_per_genre_data[genre_dummy] = {
        "average_rating" : average_rating, 
        "average_price" : average_price, 
        "count" : count
    }
aggregate_figures_per_genre = pd.DataFrame(index=list(aggregate_figures_per_genre_data.keys()),data=list(aggregate_figures_per_genre_data.values()))
aggregate_figures_per_genre.head(5)


# ## Visualisation
# 
# ### Average Ratings per Genre
# The "Trivia" and "Books & Reference" genres are suffering low average ratings. Whether it is a particularly bad app or the genre underperforming as a whole might be worth further investigation for app developers wondering what genre they should target.

# In[ ]:


rating_plot = plt.figure(figsize=(25,12))
sns.barplot(x=aggregate_figures_per_genre.index.values, y=aggregate_figures_per_genre["average_rating"])
plt.xlabel("Genre", fontsize=18)
plt.xticks(fontsize=14, rotation=90)
plt.yticks(fontsize=14)
plt.ylabel("Average Rating (Stars)", fontsize=18)
plt.show()


# ### Average Price per Genre
# 
# Interestingly, "Medical" is commanding a high price despite the relatively low average ratings seen above. Further investigation *might* reveal whether this is a genre that has a dissatisfied customer base that could be disrupted by a low cost entrant.

# In[ ]:


rating_plot = plt.figure(figsize=(25,12))
sns.barplot(x=aggregate_figures_per_genre.index.values, y=aggregate_figures_per_genre["average_price"])
plt.xlabel("Genre", fontsize=18)
plt.xticks(fontsize=14, rotation=90)
plt.yticks(fontsize=14)
plt.ylabel("Average Price ($)", fontsize=18)
plt.show()


# # Per Category Analysis
# Fortunately, category didn't require much preparation to get it ready for analysis.
# ## Visualisation
# ### Average Price Per Category
# Interestingly, the "Lifestyle" *category* seems to suffer lower ratings than its *genre* counterpart. However, there appears to be a "LIFESTYLE" category that is doing a lot better. Does it make sense to merge these? I might do this in another version.
# 
# Apart from this, there is nothing particularly compelling about the display.

# In[ ]:


average_rating_by_category = playstore["Rating"].groupby(playstore["Category"]).mean()
rating_plot = plt.figure(figsize=(25,12))
sns.barplot(x=average_rating_by_category.index.values, y=average_rating_by_category.values)
plt.xlabel("Price Point ($)", fontsize=18)
plt.xticks(fontsize=14, rotation=90)
plt.yticks(fontsize=14)
plt.ylabel("Average Rating (Stars)", fontsize=18)
plt.show()


# ### Average Price per Category
# This display tells a story similar to the per Genre analysis with the "FINANCE", "LIFESTYLE" and "MEDICAL" achieving a similar price performance to its genre counterparts.

# In[ ]:


average_price_by_category = playstore["Price"].groupby(playstore["Category"]).mean()
price_plot = plt.figure(figsize=(25,12))
sns.barplot(x=average_price_by_category.index.values, y=average_price_by_category.values)
plt.ylabel("Price Point ($)", fontsize=18)
plt.xticks(fontsize=14, rotation=90)
plt.yticks(fontsize=14)
plt.xlabel("Category", fontsize=18)
plt.show()


# # Possible Takeaways
# 
# These preliminary results suggest that a potential app developer might be able to maximise revenue by developing a finance, medical or lifestyle app.
# 
# If a developer is concerned with _revenue_ rather than net profit, this might provide a reasonable rule of thumb. However, some categories may have higher development costs; for example, developing a medical app might be more costly than developing an entertainment app.
# 
# In addition, some of the high performers in terms of price and rating in these categories seem dubious with some reviews pointing to practices that [might artifically inflate ratings](http://localhost:8888/notebooks/projects/google_play_store/google_play_store.ipynb#Fishiness?). Mining the review data might be a useful follow up to understand if there is actually a problem here.
