#!/usr/bin/env python
# coding: utf-8

# # Predicting Price of Airbnb Listings in NYC - Part 1
# 
# 
# 
# The following notebook showcases my analysis of AirBnB listings dataset originally posted on Kaggle by Dgomonov. The listings were scrapped on July 8th 2019 and are specific to NYC, NY.
# 
# Link to the dataset: https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data
# 
# The objective of the analysis is to:
# 
# - estimate listing price based on provided information
# - derive additional useful and interesting insights
# 
# 
# Part 1 deals with taking the existing dataset, performing data cleaning, feature engineering and running preliminary analysis. The product of this part is a data file for Machnie Learning analysis.
# 
# Part 2 - The Machine Learning part of the project applies machine learning algorithms to predict price of lisings based on various input variables. 

# ## Data Cleaning
# 
# Before applying any machine learning algorithms I will prepare the dataset by turning any potentially useful data into ML accessible format.

# In[ ]:


#import modules:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Open the data file and convert to a Data Frame
#Here I also separated randomly selected 20% of the data for later validation

data = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
#randomize the data
data.sample(frac=1)
df=data.copy()
df.shape


# ### Following are the dataset columns:
# 
# I grouped the columns into column categories for enhanced understanding of the dataset
# 
# #### Host descriptors:
# - __host_id:__ host ID
# - __host_name:__ name of the host
# - __calculated_host_listings_count:__ amount of listing per host
# 
# #### Listing descriptors:
# - __id:__ listing ID
# - __name:__ name of the listing
# - __room_type:__ listing space type
# - __minimum_nights:__ amount of nights minimum
# - __availability_365:__ number of days when listing is available for booking
# - __price:__ price in dollars
# 
# #### Review descriptors:
# - __number_of_reviews:__ number of reviews
# - __last_review:__ latest review
# - __reviews_per_month:__ number of reviews per month
# 
# #### Location descriptors:
# - __neighbourhood_group:__ location
# - __neighbourhood:__ area
# - __latitude:__ latitude coordinates
# - __longitude:__ longitude coordinates
# 
# 
# ### General approach
# 
# Prior to diving into the data it is worth to perform a thought experiment in which we ask ourselves what factors may be driving the lisiting prices for the given dataset, considering that the listings were most likely scrapped on July 8th 2019 (Monday), right after the Independence Day in New York City. 
# 
# I will also limit my analysis to lisitngs that are confined to 31 minimum nights at most, in order to exclude long term rentals. Therefore, my analysis will target short term renters, most likley vistors to the city, majority of whom may be tourists.
# 
# Here are a few items I brainstormed:
# - neighborhood
# - proximity to landmark objects
# - proximity to public transportation (subways mostly)
# - size of the dwelling (shared v. private v. apartment)
# - prior reviews
# - quality of the listing (ie. is the room pretty/tidy etc)
# 
# #### With that let's dive into data!
# 
# This is a snapshot of our data:

# In[ ]:


df.head().iloc[:,0:8]


# In[ ]:


df.head().iloc[:,8:]


# ### Taking care of dataset missing values
# Replace missing values
# 
# - listings without reviews have missing values for last_review and reviews_per_month. For these lisitngs the missing values will be replaced by 0
# 
# - Some listings are missing a name or the host name is missing. These will be replaced with "None"

# In[ ]:


#fill missing values for last review and reviews per month with 0
df[["last_review", "reviews_per_month"]] = df[["last_review", "reviews_per_month"]].fillna(0)

#if there is no host name or listing name fill in None
df[["name", "host_name"]] = df[["name", "host_name"]].fillna("None")

#Drop rows were price of the listing is 0. We are not intersted in "free" 
#listings as they are most likely an error.
free = len(df[df.price == 0])
df = df[df.price != 0].copy()

#Print initial insights:
print("The initial dataset contained " + str(free)+ " listings with price of 0 USD, that had been removed")
print("There are " + str(len(df["id"].unique()))+" listings")
print("There are "+str(len(df.host_id.unique()))
      +" unique and indentifiable "+ "hosts.")
print("There are "+str(len(df[df["host_name"]=="None"]))
      +" unindentifiable "+ "hosts.")
print("Dataframe shape: "+str(df.shape))


# inital analysis indicates that there are 11,492 more listings than hosts (identifiable and unidentifiable). This means that some hosts may list several properties. Let's verify that.
# 
# 
# Each listing contains "Calculated_host_listings_count", which is a count of total listing by a specific host in the provided data. The logic below derives value of the calculated host listings count for a specific listing (36485609 in this example) and checks if it is equal to the total number of listings by that host (whose host ID is 30985759) in the dataset
# 
# The comparison yields true, meaning that "calculated_host_listings_count"  for the specific host indicated gives an accurate number of listings posted by the same host. Properties could then easily related by the host ID.

# In[ ]:


(len(df[df["host_id"]==30985759]) == df[df["id"]==36485609]["calculated_host_listings_count"]).tolist()


# In[ ]:


df[(df["calculated_host_listings_count"]>1)][["host_id","calculated_host_listings_count"]].sort_values(by=['host_id']).head(10)


# #### Next, lets shift focus to the minimum nights 
# 
# Here I am plotting a histogram of minimum nights of rental required in the listing:

# In[ ]:


df_old=df.copy()
df = df[df["minimum_nights"] <=31].copy()
removed_listings = len(df_old)-len(df)

fig = plt.figure(figsize=(14,3))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)


ax1.hist(df_old.minimum_nights, bins=100, log=True)
ax1.set_ylabel("Frequency")
ax1.set_title("No limit on minimum nights")

ax2.hist(df.minimum_nights, bins=31, log=True)
ax2.set_ylabel("Frequency")
ax2.set_title("Maximum 31 minimum nights")

plt.show()

print("As a result of imposing minimum nights limit, " + str(removed_listings)+" listings were removed.")


# Looking at the left-hand side histogram of minimum nights required to rent the property, there appears to be many listings targeted to rent for a minimum period of over 31 days.
# 
# As mentioned previously, in order to narrow the scope of this analysis I will focus on a potential tourist market, assuming that tourists will not want to stay longer than 31 days. Hence, I will remove any listings with required minimum nights greater than 31.
# 
# The updated distribution is plotted in the right-hand side histogram. It appears that the distribution is bimodal. There are two peaks of minimum nights. First peak maximizes at 1 night, while the other at 30 nights. This indicates, that the limit of 31 days that I set, captures renters interested in at least a month long rental period.

# At this point there are no null values in the dataframe:

# In[ ]:


df.isnull().sum()


# Below is a statistical summary of the columns in the dataframe. We can conclude that:
# - there are 38708 rows of data being considered
# - all listings are between 40.499 and 40.913 latitude and -74.244 and -73.712 longitude squarely fitting into NYC's geographic location
# - the listing price ranges between 10USD and 10,000USD with mean price of 149USD and standard deviation of 219USD suggesting a very broad price range distribution
# - the minimum nights limit ranges betweeen 1 and 31 as determined above. It averages at 5.65, but as we have seen it is not normally distributed
# - on average the listings are available 110 days per year. It would be interesting to understand if the listings are available in short intervals (such as weekends) or long intervals (several months at a time). Perhaphs this can be understood by looking at the correlation between minimum nights and availability.

# In[ ]:


df.describe().iloc[:,0:8]


# In[ ]:


df.describe().iloc[:,8:]


# At this point we can also take a closer look into number based variables to see if any of them should be turned into categorical variables:

# In[ ]:


df.dtypes


# #### Let's start by looking at the distributions:

# In[ ]:


#separate out numerical variables
a=pd.DataFrame(df.dtypes.copy())
b= a[a[0] != 'object'].reset_index()
#drop id and host id:
numeric_vars=b["index"].tolist()[2:]

fig = plt.figure(figsize=(14,14))
ax1 = fig.add_subplot(3, 3, 1)
ax2 = fig.add_subplot(3, 3, 2)
ax3 = fig.add_subplot(3, 3, 3)
ax4 = fig.add_subplot(3, 3, 4)
ax5 = fig.add_subplot(3, 3, 5)
ax6 = fig.add_subplot(3, 3, 6)
ax7 = fig.add_subplot(3, 3, 7)
ax8 = fig.add_subplot(3, 3, 8)

ax1.hist(df[numeric_vars[0]], bins=30)
ax1.set_ylabel("Frequency")
ax1.set_title(numeric_vars[0])

ax2.hist(df[numeric_vars[1]], bins=30)
ax2.set_ylabel("Frequency")
ax2.set_title(numeric_vars[1])

ax3.hist((df[numeric_vars[2]]), bins=30)
ax3.set_ylabel("Frequency")
ax3.set_title('price')

ax4.hist(df[numeric_vars[3]], bins=31)
ax4.set_ylabel("Frequency")
ax4.set_title(numeric_vars[3])

ax5.hist(df[numeric_vars[4]], bins=30)
ax5.set_ylabel("Frequency")
ax5.set_title("number of reviews")

ax6.hist(df[numeric_vars[5]], bins=30)
ax6.set_ylabel("Frequency")
ax6.set_title("last review")

ax7.hist(df[numeric_vars[6]], bins=30)
ax7.set_ylabel("Frequency")
ax7.set_title(numeric_vars[6])

ax8.hist(df[numeric_vars[7]])
ax8.set_ylabel("Frequency")
ax8.set_title(numeric_vars[7])
plt.show()


# Looking at the distributions, clearly the following are heavily right skewed:
# - price
# - minimum_nights
# - number of reviews
# - last review
# - calculated_host_lising_count
# 
# One way to reduce the skeweness is to logarithmically transform the distributions:

# In[ ]:


numeric_vars


# In[ ]:


fig = plt.figure(figsize=(14,14))
ax1 = fig.add_subplot(3, 3, 1)
ax2 = fig.add_subplot(3, 3, 2)
ax3 = fig.add_subplot(3, 3, 3)
ax4 = fig.add_subplot(3, 3, 4)
ax5 = fig.add_subplot(3, 3, 5)
ax6 = fig.add_subplot(3, 3, 6)
ax7 = fig.add_subplot(3, 3, 7)
ax8 = fig.add_subplot(3, 3, 8)

ax1.hist(df[numeric_vars[0]], bins=30)
ax1.set_ylabel("Frequency")
ax1.set_title(numeric_vars[0])

ax2.hist(df[numeric_vars[1]], bins=30)
ax2.set_ylabel("Frequency")
ax2.set_title(numeric_vars[1])

ax3.hist(np.log((df[numeric_vars[2]])), bins=30)
ax3.set_ylabel("Frequency")
ax3.set_title('log(price)')

ax4.hist(np.log((df[numeric_vars[3]])), bins=31)
ax4.set_ylabel("Frequency")
ax4.set_title("log(minimum nights + 1)")

ax5.hist(np.log(df[numeric_vars[4]]+1), bins=30)
ax5.set_ylabel("Frequency")
ax5.set_title("log(number of reviews + 1)")

ax6.hist(np.log(df[numeric_vars[5]]+1), bins=30)
ax6.set_ylabel("Frequency")
ax6.set_title("log(last review + 1)")

ax7.hist(np.log(df[numeric_vars[6]]+1), bins=30)
ax7.set_ylabel("Frequency")
ax7.set_title("log(calculated host listing count) + 1)")

ax8.hist(np.log(df[numeric_vars[7]]+1), bins=30)
ax8.set_ylabel("Frequency")
ax8.set_title("log(availability 365 + 1)")

plt.show()


# Transform the variables

# In[ ]:


for num in numeric_vars[3:]:
    df["log_("+num+" +1)"] = np.log(df[num]+1)
df["log_price"] = np.log(df.price)
df=df.drop(columns = numeric_vars[2:]).copy()


# In[ ]:


df.columns.tolist()


# In[ ]:


df.shape


# Note that logarithmic data transformation was used to smooth out the distributions. 
# 
# #### Let's see if any correlations between variables start to emerge:

# In[ ]:


numeric_vars = df.columns.tolist()[6:8]+df.columns.tolist()[10:]


# In[ ]:


numeric_vars


# In[ ]:


import seaborn as sns
x=df[numeric_vars].apply(lambda x: np.log(np.abs(x+1))).corr(method='pearson')
sns.heatmap(x, annot=True)
plt.show()


# #### Observations on price:
# - price seems to be positively correlated with longitude meaning that one can expect higher prices as position in NYC moves West. This is expected because Manhattan, which is the most expensive borough of the city, is located on the west side of the city
# - latitude seems to have lesser effect on the price. However, there is a slight indication of higher prices located in the northern parts of the city
# - price is also positively correlated with: increasing availability, the fact that the property is rented by a host who lists other properties, and increasing number of minimum nights
# - price is negatively correlated with number of reviews and reviews per month, indicating that it is possible that the prior reviews could depress the prices to some extent
# 
# #### Other interesting observations:
# - calculated host lisitng count is positively correlated with minimum nights and availability_365 indicating that hosts who list more than one property may be more strategic rather than opportunisitic about their rentals. That may attempt to maximize the amount of time a single renter stays at their property to minimize turnover cost. They also tend to maximize the amount of time the property is being rented.

# #### Next let's turn the attention to non numerical variables

# In[ ]:


#separate out numerical variables
a=pd.DataFrame(df.dtypes.copy())
b= a[a[0] == 'object'].reset_index()
#drop id and host id:
non_num=b["index"].tolist()
print(non_num)


# These belong to:
# 
# #### Host descriptors:
# - __host_name:__ name of the host
# 
# #### Listing descriptors:
# - __name:__ name of the listing
# - __room_type:__ listing space type
# 
# #### Review descriptors:
# - __last_review:__ latest review
# 
# #### Location descriptors:
# - __neighbourhood_group:__ location
# - __neighbourhood:__ area
# 

# We already know that location is an important price determinant. Hence let's dig a bit deeper to see how mean prices vary by neighborhood

# In[ ]:


y = df.latitude
x = df.longitude
p = df.log_price
plt.figure(figsize=(16,9))
plt.scatter(x,y,c=p,cmap='viridis')
plt.colorbar()
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Distribution of listing prices")
plt.show()


# The scatter plot above visualizes the geographical distribution of listings along with the relative pricing (increasing with brightening color).
# 
# High prices appear to be concentrated around Manhattan starting with neighborhoods around Central Park going south, as well as around portions of Brooklyn and Queens close to Manhattan.
# 
# Let's group the data by neighborhood, deriving mean pricing for each neighborhood:

# In[ ]:


grouped = df.groupby("neighbourhood")
price_grouped = grouped["log_price"]
price = price_grouped.agg([np.mean,np.median,np.max, np.std]).sort_values("mean")


fig = plt.figure(figsize=(14,4))
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)

ax1.barh(price.index,price["mean"])
ax1.set_yticklabels([])
ax1.set_ylabel("Neighborhood")
ax1.set_xlabel("Mean Price")
ax1.set_title("Mean Listing Price per Neighborhood, Sorted")
ax1.set_xlim(3,7)

ax2.barh(price.index,price["median"])
ax2.set_yticklabels([])
ax2.set_ylabel("Neighborhood")
ax2.set_xlabel("Median Price")
ax2.set_title("Median Listing Price per Neighborhood")
ax2.set_xlim(3,7)

ax3.barh(price.index,price["std"])
ax3.set_yticklabels([])
ax3.set_ylabel("Neighborhood")
ax3.set_xlabel("Standard Deviation of Price")
ax3.set_title("StDev of Listing Prices per Neighborhood")
plt.show()


# Based on current findings it is worth to one-hot encode the neighborhood variable:

# In[ ]:


#One hot encoding
df = pd.concat([df, pd.get_dummies(df["neighbourhood"], drop_first=False)], axis=1)
#save neighborhoods into a list for further analysis:
neighborhoods = df.neighbourhood.values.tolist()
boroughs = df.neighbourhood_group.unique().tolist()
#drop the neighbourhood column from the database
df.drop(['neighbourhood'],axis=1, inplace=True)


# In[ ]:


df.shape


# #### The next obvious variable to evaluate is Room type. This variable provides information about relative privacy that comes with the listing as well as its size

# In[ ]:


grouped = df.groupby("room_type")
room_type_price_grouped = grouped["log_price"]
room_type_price = room_type_price_grouped.agg([np.mean,np.median,np.max, np.std]).sort_values("mean")
room_type_price


# ### Extreme Outliers
# - The data most likely includes extreme outliers, which will be difficult to model. 
# - The following code will remove extreme outliers by borough.
# - An outlier is defined as 3 x IQR below 25th quantile and above 75th quantile
# - The code will also treat each room type differently to avoid data bias

# In[ ]:


sns.boxplot(x="room_type",y="log_price", data=df)
plt.show()


# In[ ]:


def removal_of_outliers(df,room_t, nhood, distance):
    '''Function removes outliers that are above 3rd quartile and below 1st quartile'''
    '''The exact cutoff distance above and below can be adjusted'''

    new_piece = df[(df["room_type"]==room_t)&(df["neighbourhood_group"]==nhood)]["log_price"]
    #defining quartiles and interquartile range
    q1 = new_piece.quantile(0.25)
    q3 = new_piece.quantile(0.75)
    IQR=q3-q1

    trimmed = df[(df.room_type==room_t)&(df["neighbourhood_group"]==nhood) &(df.log_price>(q1-distance*IQR))&(df.log_price<(q3+distance*IQR))]
    return trimmed

#apply the function
df_private = pd.DataFrame()
for neighborhood in boroughs:
    a = removal_of_outliers(df, "Private room",neighborhood,3)
    df_private = df_private.append(a)

df_shared = pd.DataFrame()
for neighborhood in boroughs:
    a = removal_of_outliers(df, "Shared room",neighborhood,3)
    df_shared = df_shared.append(a)
    
df_apt = pd.DataFrame()
for neighborhood in boroughs:
    a = removal_of_outliers(df, "Entire home/apt",neighborhood,3)
    df_apt = df_apt.append(a)
    
# Create new dataframe to absorb newly produced data    
df_old=df.copy()    
df = pd.DataFrame()
df = df.append([df_private,df_shared,df_apt])

#plot the results
fig = plt.figure(figsize=(14,4))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

ax1.hist(df_old.log_price)
ax1.set_xlim(2,7)
ax1.set_ylabel("Frequency")
ax1.set_xlabel("Log Price")
ax1.set_title("Original price distribution")

ax2.hist(df.log_price)
ax2.set_xlim(2,7)
ax2.set_ylabel("Frequency")
ax2.set_xlabel("Log Price")
ax2.set_title("Price distribution after removal of extreme outliers")
plt.show()

print("As a result of oulier removal " + str(df_old.shape[0]-df.shape[0]) + " rows of data were removed.")


# In[ ]:


df.shape


# In[ ]:


grouped = df.groupby("room_type")
room_type_price_grouped = grouped["log_price"]
room_type_price = room_type_price_grouped.agg([np.mean,np.median,np.max, np.std]).sort_values("mean")
room_type_price


# #### Further analysis

# In[ ]:


#convert room types to dummies
df = pd.concat([df, pd.get_dummies(df["room_type"], drop_first=False)], axis=1)
df.drop(['room_type'],axis=1, inplace=True)


# In[ ]:


df.shape


# In[ ]:


y = df[(df["SoHo"]==1) & (df["Private room"]==1)].latitude
x = df[(df["SoHo"]==1) & (df["Private room"]==1)].longitude
p = df[(df["SoHo"]==1) & (df["Private room"]==1)].log_price
plt.scatter(x,y,c=p,cmap='viridis')
plt.xlim(-74.01,-73.995)
plt.ylim(40.718,40.73)
plt.colorbar()
plt.show()


# ####  Last review is a data when last review has been posted. Perhaps the most effective way of dealing with this variable is to
# - covert it to number of days since last review counting down from the data the data was scraped off the web
# - categorize

# In[ ]:


import datetime as dt
#convert object to datetime:
df["last_review"] = pd.to_datetime(df["last_review"])
#Check the latest review date in the datebase:
print(df["last_review"].max())


# In[ ]:


df.shape


# The last review in the database dates to July 8th 2019, which will be used as time zero for analysis:

# In[ ]:


df["last_review"]=df["last_review"].apply(lambda x: dt.datetime(2019,7,8)-x)
df["last_review"]=df["last_review"].dt.days.astype("int").replace(18085, 1900)
plt.hist(df["last_review"], bins=100)
plt.ylabel("Frequency")
plt.xlabel("Days since last review")
plt.ylabel("Frequency")
plt.title("Histogram of days since last review")
plt.show()


# Replace with the following categories for simplification:
# - last month
# - last 6 months
# - last year
# - last 5 years
# - never

# In[ ]:


def date_replacement(date):
    if date <=3:
        return "Last_review_last_three_day"
    elif date <= 7:
        return "Last_review_last_week"
    elif date <= 30:
        return "Last_review_last_month"
    elif date <= 183:
        return "Last_review_last_half_year"
    elif date <= 365:
        return "Last_review_last year"
    elif date <= 1825:
        return "Last_review_last_5_years"
    else:
        return "Last_review_never" 

    
df["last_review"]=df["last_review"].apply(lambda x: date_replacement(x))
sns.boxplot(x="last_review", y=df.log_price, data=df)
plt.show()


# Time since last review does not show any clear trend on the sale of entire dataset. Yet it may be important to keep the categorical values included for later machine learning experiments.

# In[ ]:


grouped = df.groupby("last_review")
last_review_price_grouped = grouped["log_price"]
last_review_price = last_review_price_grouped.agg([np.mean,np.median,np.max, np.std]).sort_values("mean")
last_review_price


# In[ ]:


#convert last review to dummies
df = pd.concat([df, pd.get_dummies(df["last_review"], drop_first=False)], axis=1)
df.drop(["last_review"],axis=1, inplace=True)


# There is no readily apparent trend of recency of reviews with respect to room price visible. Further analysis may reveal more insight.

# #### Natural Language Processing

# In[ ]:


#import necessary libraries
import nltk
import os
import nltk.corpus
from nltk import ne_chunk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('maxent_ne_chunker')
nltk.download('words')


# In[ ]:


#initiate stopwords
a = set(stopwords.words('english'))
#obtain text
text = df["name"].iloc[10]
#tokenize text
text1 = word_tokenize(text.lower())
#create a list free of stopwords
no_stopwords = [x for x in text1 if x not in a]
#lemmatize the words
lemmatizer = WordNetLemmatizer() 
lemmatized = [lemmatizer.lemmatize(x) for x in no_stopwords]


# In[ ]:


def unique_words1(dwelling):

    apt = df[df[dwelling]==1]["name"]
    a = set(stopwords.words('english'))
    words = []
    # append each to a list
    for lis in range(0, len(apt)):
        listing = apt.reset_index().iloc[lis,1]
        #tokenize text
        text1 = word_tokenize(listing.lower())
        #create a list free of stopwords
        no_stopwords = [x for x in text1 if x not in a]
        #lemmatize the words
        lemmatized = [lemmatizer.lemmatize(x) for x in no_stopwords]
        no_punctuation = [x.translate(str.maketrans('','',string.punctuation)) for x in lemmatized]
        no_digits = [x.translate(str.maketrans('','',"0123456789")) for x in no_punctuation ]
        for item in no_digits:
            words.append(item)


    #create a dictionary
    unique={}
    for word in words:
        if word in unique:
            unique[word] +=1
        else:
            unique[word] = 1

    #sort the dictionary
    a=[]
    b=[]

    for key, value in unique.items():
        a.append(key)
        b.append(value)

    aa=pd.Series(a)
    bb=pd.Series(b)    

    comb=pd.concat([aa,bb],axis=1).sort_values(by=1, ascending=False).copy()

    return comb

#apply the function
private = unique_words1("Private room")
home = unique_words1("Entire home/apt")
shared = unique_words1("Shared room")

words_private = private.iloc[1:,1]
words_home = home.iloc[1:,1] 
words_shared = shared.iloc[1:,1] 

#plot the results
plt.plot(words_shared.reset_index()[1], label="shared")
plt.plot(words_private.reset_index()[1], label ="private")
plt.plot(words_home.reset_index()[1], label="Entire home/apt")
plt.xlim(0,200)
plt.ylabel("WordFrequency")
plt.xlabel("Word position on the list")
plt.legend()
plt.show()


# In[ ]:


home_new = home.reset_index().iloc[1:50,1:3].copy()
private_new = private.reset_index().iloc[1:50,1:3].copy()
shared_new = shared.reset_index().iloc[1:50,1:3].copy()

all_words = pd.concat([home_new, private_new, shared_new], axis=1, sort=False)
all_words


# In[ ]:


#see how many listing there are for each type of room:
print("Numer of shared room listings: "+str(len(df[df["Shared room"]==1])))
print("Numer of private room listings: "+str(len(df[df["Private room"]==1])))
print("Numer of entire home/apt listings: "+str(len(df[df["Entire home/apt"]==1])))


# In[ ]:


#Create a list of the most popular words common for all room types:
most_popular_words = home_new.iloc[:,0].tolist()+private_new.iloc[:,0].tolist()+shared_new.iloc[:,0].tolist()
most_popular = pd.Series(most_popular_words)
popular_descriptors=most_popular.unique().tolist()


# In[ ]:


def unique_words2(name, word):
    '''This function takes individual name and looks for a matching word in it'''
    a = set(stopwords.words('english'))
    #tokenize the name
    text1 = word_tokenize(str(name).lower())
    #create a list free of stopwords
    no_stopwords = [x for x in text1 if x not in a]
    #lemmatize the words
    lemmatized = [lemmatizer.lemmatize(x) for x in no_stopwords]
    no_punctuation = [x.translate(str.maketrans('','',string.punctuation)) for x in lemmatized]
    no_digits = [x.translate(str.maketrans('','',"0123456789")) for x in no_punctuation ]
    counter = 0
    for item in no_digits:
        if str(item) == str(word):
            counter += 1
        else:
            continue

    if counter != 0:
        return 1
    else:
        return 0
    
#Apply the function 
for item in popular_descriptors:
    df[item]= df["name"].apply(lambda x: unique_words2(x,item))


# In[ ]:


#convert last review to dummies
df = pd.concat([df, pd.get_dummies(df['neighbourhood_group'], drop_first=False)], axis=1)
df.drop(['neighbourhood_group'],axis=1, inplace=True)


# In[ ]:


#drop unnecessary columns
df = df.drop(['id','name','host_id','host_name'], axis=1).copy()
#copy for later
df2 = df.copy()
df.shape


# In[ ]:


len(popular_descriptors)


# In[ ]:


def plot_by_word(word):
    '''creates a plot of price for listings matching given word'''
    y = df[(df[word]==1)].latitude
    x = df[(df[word]==1)].longitude
    p = df[(df[word]==1)].log_price
    plt.figure(figsize=(16,9))
    plt.scatter(x,y,c=p,cmap='viridis')
    plt.xlabel
    plt.colorbar()
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Word 'Luxury' in the name of the listing\nColormap indicates price")
    plt.show()
    
plot_by_word("Manhattan")


#  ### Proceed to Machine Learning
# 
# With data cleaning and feature engineering complete, it is time now to apply machine learning algorithms to develop an approprie price prediction model and derive futher insights from the data.

# In[ ]:


#import modules:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from sklearn.tree            import DecisionTreeRegressor
from sklearn.neural_network  import MLPRegressor
from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics         import mean_squared_error
from sklearn.metrics         import r2_score


# In[ ]:


target = df['log_price'].copy()
#drop unnecessary columns
df = df.drop(['log_price'], axis=1).copy()
#strip the target column from input columns and put it in front
df = pd.concat([target, df], axis=1).copy()
#select input variable columns
nums = df.iloc[:,1:]


# Let's first review the final dataframe that will be used for the analysis:

# In[ ]:


#first few rows of the dataframe:
df.head()


# In[ ]:


#dataframe shape
print(df.shape)


# In[ ]:


#column names in the dataframe
df.columns.tolist()


# With one hot encoding, feature engineering and natural language processing the shape of the dataframe grew substantially. 
# - Several numeric columns have been log transformed
# - Individual neighborhoods have been one hot encoded along with boroughs and categorized times since the last review
# - Lastly columns were created to document use of popular words in the listing name
# 
# 
# Next, let's use the entire dataset and feed it into some of the most common regression models to see what sort of root mean square error we get:

# ## Experiment with several ML approaches

# In[ ]:


y= target
x = nums
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.20, random_state=1)


# ### Decision Tree Regression - kfold

# In[ ]:


rmse_dt=[]
dt = DecisionTreeRegressor()
kf = KFold(5, shuffle = True, random_state=1)
mse = cross_val_score(dt ,x,y, scoring = "neg_mean_squared_error", cv=kf) 
rmse = np.sqrt(np.absolute(mse))
avg_rmse = np.sum(rmse)/len(rmse)
rmse_dt.append(avg_rmse)
print("Root mean square error: " +str(round(rmse_dt[0],2)))


# ### XG Boost - kfold

# In[ ]:


rmse_xg = []
data_dmatrix = xgb.DMatrix(data=x,label=y)
params = {
              'colsample_bytree': 0.9,
              'learning_rate': 0.1,
              'max_depth': 1, 
              'alpha': 10}
cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=5, num_boost_round=300,
                        early_stopping_rounds=10, metrics="rmse", as_pandas=True, 
                        seed=123)
    
rmse_xg.append(cv_results["test-rmse-mean"].tolist()[-1])
print("Root mean square error: " +str(round(rmse_xg[0],2)))


# ### Random Forest Regression - kfold

# In[ ]:


rmse_rf=[]
rf=RandomForestRegressor(n_estimators = 100, random_state=1,  min_samples_leaf=2)
kf = KFold(5, shuffle = True, random_state=1)
mse = cross_val_score(rf ,x,y, scoring = "neg_mean_squared_error", cv=kf) 
rmse = np.sqrt(np.absolute(mse))
avg_rmse = np.sum(rmse)/len(rmse)
rmse_rf.append(avg_rmse)
print(rmse_rf)


# ### Neural Network  - kfold

# In[ ]:


rmse_nndf=[]
mlp = MLPRegressor(activation='relu', max_iter=1000)
kf = KFold(5, shuffle = True, random_state=1)
mse = cross_val_score(mlp ,x,y, scoring = "neg_mean_squared_error", cv=kf) 
rmse = np.sqrt(np.absolute(mse))
avg_rmse = np.sum(rmse)/len(rmse)
rmse_nndf.append(avg_rmse)
print(rmse_nndf)


# In[ ]:


dt = pd.Series(rmse_dt, name ="Decision Tree")
rand = pd.Series(rmse_rf, name ="Random Forest")
xgb = pd.Series(rmse_xg, name ="XG Boost")
nn = pd.Series(rmse_nndf, name="Neural Network")
pd.concat([dt,rand,xgb,nn],axis=1)


# ### Comments:
# 
# Random Forest model yields the lowest RMSE, followed by Neural Netork, XGBoost and Decision Tree
# 
# Since random forest yielded the best performance, let's keep optimizing it:

# In[ ]:


#optimizing number of estimators
train_results = []
test_results = []
n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]
for estimator in n_estimators:
    rf = RandomForestRegressor(n_estimators=estimator, n_jobs=-1, random_state=1)
    rf.fit(X_train, y_train)
    train_pred = rf.predict(X_train)
    rmse = round(np.sqrt(mean_squared_error(y_train, train_pred)),2)
    train_results.append(rmse)
    y_pred = rf.predict(X_test)
    rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)),2)
    test_results.append(rmse)
    
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(n_estimators, train_results, 'b', label='Train RMSE')
line2, = plt.plot(n_estimators, test_results, 'r', label='Test RMSE')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('RMSE')
plt.xlabel('n_estimators')
plt.show()


# - Clearly, the model overfits since the RMSE for train dataset is much lower then the test dataset regardless of the number of estimators
# - For the best model performance on unseen data, a minimum of 100 estimators appears to be sufficient

# In[ ]:


#optimizing max_features
train_results = []
test_results = []
max_features = ['auto','sqrt','log2']
for feature in max_features:
    rf = RandomForestRegressor(max_features=feature, n_estimators=100, n_jobs=-1, random_state=1)
    rf.fit(X_train, y_train)
    train_pred = rf.predict(X_train)
    rmse = round(np.sqrt(mean_squared_error(y_train, train_pred)),2)
    train_results.append(rmse)
    y_pred = rf.predict(X_test)
    rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)),2)
    test_results.append(rmse)
    
plt.bar(max_features,test_results)
plt.bar(max_features,train_results)
plt.ylabel('RMSE')
plt.xlabel('max_features, test, train')
plt.show()


# In[ ]:


#optimizing min_sample_leaf
train_results = []
test_results = []
min_samples_leaf = [1,2,10,50,70,100]
for leaf in min_samples_leaf:
    rf = RandomForestRegressor(min_samples_leaf = leaf, max_features='auto', n_estimators=100, n_jobs=-1, random_state=1)
    rf.fit(X_train, y_train)
    train_pred = rf.predict(X_train)
    rmse = round(np.sqrt(mean_squared_error(y_train, train_pred)),3)
    train_results.append(rmse)
    y_pred = rf.predict(X_test)
    rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)),3)
    test_results.append(rmse)
    
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(min_samples_leaf, train_results, 'b', label='Train RMSE')
line2, = plt.plot(min_samples_leaf, test_results, 'r', label='Test RMSE')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('RMSE')
plt.xlabel('min_sample_leaf')
plt.show()


# In[ ]:


test_results 


# The graph above inticates that the model provides the best performance if the minimum samples on a leaf is at least 1
# 
# Based on the hyperparameter optimization the best hyperparameter conditions are:
# - n_estimator = 100
# - max_features = 'auto'
# - min_samples_leaf = 2
# 
# Next, apply these parameters to see how the overall model performs:

# In[ ]:


#apply the hyperparameter optmized model:
rf=RandomForestRegressor(n_estimators = 300, max_features = 'auto', min_samples_leaf=2, random_state=1)
rf.fit(X_train,y_train)
predicted = rf.predict(X_test)


# In[ ]:


#plot the results of the model:
plt.figure(figsize=(8,4.5))
plt.scatter(y_test,predicted, label="Model Results")
plt.plot([2,7],[2,7], color="red", label = "Equality Line")
plt.title("Predictions for test portion of the dataset")
plt.xlim(2,7)
plt.ylim(2,7)
plt.legend()
plt.ylabel("Predicted log_price")
plt.xlabel("Actual log_price")
plt.show()


# In[ ]:


print("Model accuracy measures for withheld data:\nR2: "+str(round(r2_score(y_test,predicted),2)))
print("Root mean square error: "+str(round(np.sqrt(mean_squared_error(y_test,predicted)),3)))


# ### Comments:
# 
# On the low end of the actual log_price the results tend to cluster above the line, while on the high end they tend to cluster below the line. 
# 
# This has consequences of underpredicting high prices and overpredicting low prices.
# 
# Next, let's take a look at what features are important in the model:

# In[ ]:


#derive important features
feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance', ascending=False)

print("Number of important features: "+str(feature_importances[feature_importances["importance"]!=0].shape[0]))
print("\nTop fifteen features by importance:")
feature_importances[feature_importances["importance"]!=0].head(15)


# The most important features that factor into the price of a listing are:
# 
# - listing type (if it is a home/apartment)
# - location, which is very intuitive considering that in real estate location is often a decided factor for price
# - availability and review related factors
# - certain listing descriptor words indicating the character or location of a listing
# 
# ### Further visualization of results
# 
# The three graphs below illustrate the pricing in USD:
# - first graph illustrates actual and predicted price for the test dataset in order of growing price
# - second graph illustrates associated % error
# - finally the third graph visualizes the distribution of errors with respect to latitude and longitude

# In[ ]:


predicted_ = rf.predict(X_test)
pred = pd.DataFrame({'Predicted log_price':predicted_,'log_price':y_test})
df_with_predictions = pd.concat([X_test, pred], axis=1).copy()
df_with_predictions["price"]=df_with_predictions["log_price"].apply(lambda x: np.exp(x))
df_with_predictions["predicted_price"]=df_with_predictions["Predicted log_price"].apply(lambda x: round(np.exp(x),1))


# In[ ]:


prices=df_with_predictions.sort_values(by="price").reset_index()
prices["error"]=np.abs(prices.price-prices.predicted_price)/prices.price*100


# In[ ]:


plt.figure(figsize=(15,4.5))
plt.plot(prices["predicted_price"], label="Predicted Price")
plt.plot(prices["price"], label = "Actual Price")
plt.title("Price prediction vs. actual price for listings in the test dataset sorted")
plt.xlim(0,10000)
plt.ylim(0,800)
plt.legend()
plt.ylabel("Price USD")
plt.xlabel("Lisitng")
plt.show()


# In[ ]:


plt.figure(figsize=(15,4.5))
plt.plot(prices["error"], label="Price Error")
plt.title("Absolute price error % sorted")
plt.xlim(0,10000)
plt.ylim(0,400)
plt.legend()
plt.ylabel("Price Error (%)")
plt.xlabel("Lisitng")
plt.show()


# In[ ]:


small_error = prices[prices.error<20].copy()
y = small_error["latitude"]
x = small_error["longitude"]
p = small_error.error
plt.figure(figsize=(16,9))
plt.scatter(x,y,c=p,cmap='viridis')
plt.colorbar()
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Distribution of errors")
plt.show()


# There is no apparent geographic concentration of error visible in the graph above.
# 
# 
# The following are mean price errors for each borough in USD.

# In[ ]:


print(prices[prices.Manhattan==1]["error"].mean())
print(prices[prices.Brooklyn==1]["error"].mean())
print(prices[prices.Bronx==1]["error"].mean())
print(prices[prices.Queens==1]["error"].mean())
print(prices[prices["Staten Island"]==1]["error"].mean())


# ### Impact of Private v. Shared v. Home on the model's accuracy
# 
# Let's look at model efficiency differences when considering different listing types (shared, private, apartment):

# In[ ]:


#define random forest function with kfold cross-validation
def random_forest(df):
    target = df['log_price'].copy()
    #select input variable columns
    nums = df.iloc[:,1:]

    #split the data into test and train
    y= target
    x = nums
    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.20, random_state=1)

    rmse_rf=[]
    rf=RandomForestRegressor(n_estimators = 300, max_features = 'auto', min_samples_leaf=1, random_state=1)
    kf = KFold(5, shuffle = True, random_state=1)
    mse = cross_val_score(rf ,x,y, scoring = "neg_mean_squared_error", cv=kf) 
    rmse = np.sqrt(np.absolute(mse))
    avg_rmse = np.sum(rmse)/len(rmse)
    rmse_rf.append(avg_rmse)
    return rmse_rf


# In[ ]:


#separate datasets
private = df[df["Private room"]==1].copy()
shared = df[df["Shared room"]==1].copy()
homes = df[df["Entire home/apt"]==1].copy()

private_rmse = random_forest(private)
shared_rmse = random_forest(shared)
home_rmse = random_forest(homes)

print("\nShared RMSE: "+str(round(shared_rmse[0],3)))
print("Private RMSE: "+str(round(private_rmse[0],3)))
print("Home RMSE: "+str(round(home_rmse[0],3)))


# - Private room dataset yielded the best model accuracy followed by home and shared. 
# - The difference in accuracy between shared and the other two datasets is most likely derived from fewer datapoints available to train the shared model, resulting in lower accuracy.
# - It is clear that the model predicts with better accuracy for private listings than for the home/apartment listings. It is likely driven by larger spread of prices within the home/apartments listings relative to private listings. Look at standard deviations for the two population samples below:

# In[ ]:


print(private.log_price.std())
print(homes.log_price.std())


# ### Impact of Borough on the model's accuracy

# In[ ]:


#separate datasets
manhattan = df[(df["Manhattan"]==1)].copy()
brooklyn = df[(df["Brooklyn"]==1)].copy()
queens = df[(df["Queens"]==1)].copy()
bronx = df[(df["Bronx"]==1)].copy()
staten_island = df[(df["Staten Island"]==1)].copy()

manhattan_rmse = random_forest(manhattan)
brooklyn_rmse = random_forest(brooklyn)
queens_rmse = random_forest(queens)
bronx_rmse = random_forest(bronx)
staten_island_rmse = random_forest(staten_island)

print("\nManhattan RMSE: "+str(round(manhattan_rmse[0],3)))
print("Brooklyn RMSE: "+str(round(brooklyn_rmse[0],3)))
print("Queens RMSE: "+str(round(queens_rmse[0],3)))
print("Bronx RMSE: "+str(round(bronx_rmse[0],3)))
print("Staten Island RMSE: "+str(round(staten_island_rmse[0],3)))


# In[ ]:


print("Number of listings in Manhattan: "+str(len(manhattan)))
print("Number of listings in Brooklyn: "+str(len(brooklyn)))
print("Number of listings in Queens: "+str(len(queens)))
print("Number of listings in Bronx: "+str(len(bronx)))
print("Number of listings in Staten Island: "+str(len(staten_island)))


# In[ ]:


plt.scatter(x=[0.384,0.369,0.369,0.419,0.458],y=[21192,19801,5592,1071,370])
plt.xlabel("RMSE")
plt.ylabel("Number of listings in a borough")
plt.title("Listing number vs. Model RMSE")
plt.show()


# There are also large differences in model performance from borough to borough. The graph above illustrates that as the number of listings in a given borough decreases, the model accuracy decreases as well. The only exception is Queens where in spite of a realtively small amount of listings, the model delivered a very good performance.
# 
# As a next step let's check model accuracy for each borough considering only private rooms.

# In[ ]:


#separate datasets
manhattan = df[(df["Manhattan"]==1)&(df["Private room"]==1)].copy()
brooklyn = df[(df["Brooklyn"]==1)&(df["Private room"]==1)].copy()
queens = df[(df["Queens"]==1)&(df["Private room"]==1)].copy()
bronx = df[(df["Bronx"]==1)&(df["Private room"]==1)].copy()
staten_island = df[(df["Staten Island"]==1)&(df["Private room"]==1)].copy()

manhattan_rmse = random_forest(manhattan)
brooklyn_rmse = random_forest(brooklyn)
queens_rmse = random_forest(queens)
bronx_rmse = random_forest(bronx)
staten_island_rmse = random_forest(staten_island)

print("\nManhattan RMSE: "+str(round(manhattan_rmse[0],3)))
print("Brooklyn RMSE: "+str(round(brooklyn_rmse[0],3)))
print("Queens RMSE: "+str(round(queens_rmse[0],3)))
print("Bronx RMSE: "+str(round(bronx_rmse[0],3)))
print("Staten Island RMSE: "+str(round(staten_island_rmse[0],3)))


# Here see if the global model and deliver the same performance as the specialized model for the same set of data. For instance only Manhattand and only private room.

# In[ ]:



target = df2['log_price'].copy()
#drop unnecessary columns
df = df2.drop(['log_price'], axis=1).copy()
#strip the target column from input columns and put it in front
df = pd.concat([target, df], axis=1).copy()
#select input variable columns
nums = df.iloc[:,1:]


# In[ ]:


# RMSE for the global model considering Queens and Private room only
qns_priv_price=df_with_predictions[(df_with_predictions["Queens"]==1)&(df_with_predictions["Private room"]==1)].log_price
qns_priv_predprice=df_with_predictions[(df_with_predictions["Queens"]==1)&(df_with_predictions["Private room"]==1)]["Predicted log_price"]
rmse_global = round(np.sqrt(mean_squared_error(qns_priv_price, qns_priv_predprice)),3)
print(rmse_global)


# In[ ]:


#separating the data to make it specific to the borough of Queens and the listing type
df=df[(df["Queens"]==1)&(df["Private room"]==1)].copy()

target = df['log_price'].copy()
#drop unnecessary columns
df = df.drop(['log_price'], axis=1).copy()
#strip the target column from input columns and put it in front
df = pd.concat([target, df], axis=1).copy()
#select input variable columns
nums = df.iloc[:,1:]

y= target
x = nums
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.20, random_state=1)

rf=RandomForestRegressor(n_estimators = 300, max_features = 'auto', min_samples_leaf=2, random_state=1)
rf.fit(X_train,y_train)
predicted = rf.predict(X_test)


# In[ ]:


round(np.sqrt(mean_squared_error(y_test, predicted)),3)


# Based on the Queens/Private bedroom example it appears to be more beneficial to train several models specific to the listing type and location (such as Borough).
# 
# Making a model specific improved RMSE from 0.334 to 0.309 or by 7.5%.
# 
# Let's see if taking the approach of using a model specifc to borough and listing type can improve the overall RMSE for the entire data set.

# In[ ]:


def final_model(borough, room_type):
    '''Build a function specifc to a borough and room_type'''
    #read the cleaned data from a file:
    df = df2.copy()
    #filter the data
    df=df[(df[borough]==1)&(df[room_type]==1)].copy()    
    target = df['log_price'].copy()
    #drop unnecessary columns
    df = df.drop(['log_price'], axis=1).copy()    
    #strip the target column from input columns and put it in front
    df = pd.concat([target, df], axis=1).copy()
    #select input variable columns
    nums = df.iloc[:,1:]
    
    y= target
    x = nums
    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.20, random_state=1)

    rf=RandomForestRegressor(n_estimators = 300, max_features = 'auto', min_samples_leaf=2, random_state=1)
    rf.fit(X_train,y_train)
    predicted = rf.predict(X_test)
    y_test = y_test.values.tolist()
    predicted = predicted.tolist()
    return y_test, predicted 


# Run individual models and combine the predictions

# In[ ]:


boroughs = ["Manhattan", "Brooklyn", "Bronx","Queens","Staten Island"]
listings= ["Private room","Shared room","Entire home/apt"]

actual=[]
predicted=[]
for borough in boroughs:
    for listing in listings:
        a,b = final_model(borough, listing)
        actual +=a
        predicted +=b
        
round(np.sqrt(mean_squared_error(actual, predicted)),3)


# In[ ]:


#plot the results of the model:
plt.figure(figsize=(8,4.5))
plt.scatter(actual, predicted, label="Model Results")
plt.plot([2,7],[2,7], color="red", label = "Equality Line")
plt.title("Predictions for test portion of the dataset")
plt.xlim(2,7)
plt.ylim(2,7)
plt.legend()
plt.ylabel("Predicted log_price")
plt.xlabel("Actual log_price")
plt.show()


# In[ ]:


print("Model accuracy measures for withheld data:\nR2: "+str(round(r2_score(actual,predicted),3)))
print("Root mean square error: "+str(round(np.sqrt(mean_squared_error(actual,predicted)),3)))


# While the approach of building individual models worked well for certain cases, overall the RMSE of 0.378 was higher than the one achieved using the entire dataset (0.372)
# 
# ### In summary:
# 
# - Random Forest regression model provided best accuracy for prediction of listing price based on variables generated from the initial data
# - the model as is tends to underpredict listings priced relatively high
# - the model tends to underpredict listings priced relatively low
# - the model importances can be used to further understand what drives the price of an Airbnb listing in NYC
# - RMSE are given based on log_price values
