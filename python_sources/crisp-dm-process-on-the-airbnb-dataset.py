#!/usr/bin/env python
# coding: utf-8

# This is the project of Udacity's data scientist nanodegree program (Term2, Project1).  
# In this project, we are required to analysis data with **CRISP-DM** process. The CRISP-DM process is below.
# 
# #### CRISP-DM (Cross-Industry Standard Process for Data Mining)
# - Business Understanding
# - Data Understanding
# - Data Preparation
# - Modeling
# - Evaluation
# - Deployment
# 
# So, in this kernel I analysis data with following this process.

# ## Business Understanding
# 
# Airbnb is a platform of accommodation which match the needs of staying and of lending.  
# Their main source of income is **fee for host**. Basically, as the number of transactions between the host and the guest increases, their profit also increases.  
# So, It is important to their business and I expect it to be one of their KPIs.
# 
# 
# <img src="https://bmtoolbox.net/wp-content/uploads/2016/06/airbnb.jpg" width=700>
# 
# ref: https://bmtoolbox.net/stories/airbnb/
# 
# #### What can we do to increase the transactions?
# I considered three below questions to explore its way.
# 
# * **How long is the period available for lending by rooms?**  
# Is there rooms which is available all for years? or almost rooms are available on spot like one day or one week?  
# Here, I want to know the trend in the outline of the data.
# 
# * **Is there a busy season?**  
# If the demand for accommodation is more than the number of rooms available for lending, it leads to the loss of business opportunities.  
# So, I want to know whether is there the busy season. If this is true, we must create a mechanism to increase the number of rooms available for lending during the busy season.
# 
# * **Are there any trends of popular rooms?**  
# If this question's answer is true, we can suggest host to ways make the room easier to rent.  
# In this part, I'll use machine learning technique.

# ## Data Understanding
# 
# We have three data.
# 
# * `listings`: including full descriptions and average review score
# * `calendar`: including listing id and the price and availability for that day
# * `reviews`: including unique id for each reviewer and detailed comments
# 
# In this part, I'll make some visualization and aggregation to understand the charactoristics of the data.

# In[1]:


# import necessary package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score


get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# load data
seattle_calendar = pd.read_csv('../input/calendar.csv')
seattle_listing = pd.read_csv('../input/listings.csv')
seattle_review = pd.read_csv('../input/reviews.csv')


# ### calendar

# Let's look first 5 row of the data and column information.

# In[3]:


seattle_calendar.head()


# In[4]:


seattle_calendar.info()


# There are 4 columns.  
# Here, I found some charactoristics of the data.
# 
# * Not only available days are stored in data, it seems to be stored not available days.
# * If the `available` values are `f`, the `price` values seems to be `NaN`. 
# * The `price` values are stored as object, not integer. This is caused the value stored like `$xx.xx`, and it is necessary to transform this column.

# In response to the result, now I have two question .
# 
# 1. If the available values are f, the price values seems to be NaN. But it is only a hypothesis, is it true all data?
# 2. How many rows per each listing_id?
# 
# Let's answer these questions with exploring data.

# In[5]:


#  If the available values are f, the price values seems to be NaN. But it is only a hypothesis, is it true all data?
calendar_q1_df = seattle_calendar.groupby('available')['price'].count().reset_index()
calendar_q1_df.columns = ['available', 'price_nonnull_count']
calendar_q1_df


# In[6]:


#  How many rows per each listing_id?
calendar_q2_df = seattle_calendar.groupby('listing_id')['date'].count().reset_index()
calendar_q2_df['date'].value_counts()


# Above, I can answer my question. The answer is
# 
# ***If the available values are f, the price values seems to be NaN. But it is only a hypothesis, is it true all data?***  
# -> true !!
# 
# ***How many rows per each listing_id?***  
# -> 365 days record. This is equal a year.

# Now, I almost understood the features of the data.  
# Finally, I'll research is there any trend of the listings price.

# In[47]:


# process data
calendar_q3_df = seattle_calendar.copy(deep=True)
calendar_q3_df.dropna(inplace=True)
calendar_q3_df['date'] = pd.to_datetime(calendar_q3_df['date'])
calendar_q3_df['price'] = calendar_q3_df['price'].map(lambda x: float(x[1:].replace(",", "")))

# apply aggregation
calendar_q3_df = calendar_q3_df.groupby('date')['price'].mean().reset_index()

# plot avg listings prices over time.
plt.figure(figsize=(15, 8))
plt.plot(calendar_q3_df.date, calendar_q3_df.price, color='b', marker='.', linewidth=0.9)
plt.title("Average listing price by date")
plt.xlabel('date')
plt.ylabel('average listing price')
plt.grid()


# This is interesting.  
# There are two trend of the data.
# 
# 1. The average price rise from 2016/1 to 2016/7, and reach peak for three months, and getting lower. And the average proce of 2017/1 is higher than 1 years ago.
# 2. There is periodic small peak.

# The first trend can be split into two foctors. One is seasonal factor, and the other is overall factor.  
# The second trend looks like a weekly trend, so let's close look at!!

# In[48]:


# plot more narrow range
plt.figure(figsize=(15, 8))
plt.plot(calendar_q3_df.date.values[:15], calendar_q3_df.price.values[:15], color='b', marker='o', linewidth=1.5)
plt.title("Average listing price by date")
plt.xlabel('date')
plt.ylabel('average listing price')
plt.grid()


# It looks like a weekly trend as I thought.  
# Then, which does weekday have high price? 

# In[9]:


# create weekday column
calendar_q3_df["weekday"] = calendar_q3_df["date"].dt.weekday_name

# boxplot to see price distribution
plt.figure(figsize=(15, 8))
sns.boxplot(x = 'weekday',  y = 'price', data = calendar_q3_df, palette="Blues", width=0.6)
plt.show()


# The weekend, Friday and Saturday has high prices. 

# #### Summary
# 
# * Each listings has `365` days record in this data.
# * If `available` values are `f`, the `price` values are `NaN`.
# * There is the weekly trend which the listing prices in weekend are higher than other weekday.

# ### listings

# Let's begin with looking at first 5 row of the data and columns information.

# In[10]:


seattle_listing.head()


# In[11]:


print(list(seattle_listing.columns.values))


# There are many columns, so I can't explore each columns here.  
# Here I'll look at some columns of my interest.

# First, I'll investigate how many listings are in the data.

# In[12]:


print("Num of listings: ", seattle_listing.id.count())
print("Num of rows: ", seattle_listing.shape[0])


# This shows the each rows represents unique listings.

# Next, I am interested in below columns.
# 
# * review_scores_rating
# * price
# * maximum_nights
# 
# What is the distribution of these values in each columns? Is the distribution skewed or normal?  
# Let's look at!

# #### review_scores_rating

# In[13]:


seattle_listing['review_scores_rating'].describe().reset_index()


# In[14]:


# cleaning data
listings_q1_df = seattle_listing['review_scores_rating'].dropna()

# plot histgram
plt.figure(figsize=(15, 8))
plt.hist(listings_q1_df.values, bins=80, color='b')
plt.grid()


# This is very right skewed distribution.  
# The 75% or more values are 90 points. And the most common thing is 100 points.  
# I can say the low score listings are minolity.

# #### price

# In[15]:


# cleaning data
listings_q2_df = seattle_listing.copy(deep=True)
listings_q2_df = listings_q2_df['price'].dropna().reset_index()
listings_q2_df['price'] = listings_q2_df['price'].map(lambda x: float(x[1:].replace(',', '')))

listings_q2_df['price'].describe().reset_index()


# In[16]:


plt.figure(figsize=(15, 8))
plt.hist(listings_q2_df.price, bins=100, color='b')
plt.grid()


# This is long tail distribution.  
# Almost values are from 0 to 200.

# #### maximum_nights

# In[17]:


seattle_listing['maximum_nights'].describe().reset_index()


# In[49]:


# eliminate outliers because maximum values are very large.
listings_q3_df = seattle_listing[seattle_listing['maximum_nights'] <= 1500]

plt.figure(figsize=(15, 8))
plt.hist(listings_q3_df.maximum_nights, bins=100, color='b')
plt.xlabel('maximum nights')
plt.ylabel('listings count')
plt.grid()


# This is very surprising because I expect it would be a week at most.  
# In fact, almost `maxmum_night` values are setted 1125.   
# I have not used Airbnb so I don't know, but maybe there may be something like the default value.  
# Or there maybe two segments, one is `spot available listings`, the other is `long term listings like normal rent`. 

# #### Summary
# 
# * The listings data has 92 columns.
# * The `review_scores_rating` has right skewed distribution, and almost values are over 90 points.
# * The `price` has long tail distribution, almost values are around 100\$ but some values are much higher than other values.  
# * The `maximum_nights` has very special distribution. Their are two segments, one is about 3 years, the other is around 1week.
# 
# OK, let's look at last data.

# ### reviews

# Let's begin with looking at first 5 row of the data and columns information.

# In[19]:


seattle_review.head()


# In[20]:


seattle_review.info()


# There are six columns, such as listing_id that received review, id of reviews, when review submitted, and so on.  
# I'm concerned that there are no review scores here. I think it might be in comments, so let's confirm this.

# In[21]:


print("sample 1: ", seattle_review.comments.values[0], "\n")
print("sample 2: ", seattle_review.comments.values[3])


# From the above, the review score seems not to be included.

# Next, I want to see the time series change of the number of comments.

# In[50]:


# convert date column's data type to date from object
review_q1_df = seattle_review.copy(deep=True)
review_q1_df.date = pd.to_datetime(review_q1_df.date)

review_q1_df = review_q1_df.groupby('date')['id'].count().reset_index()

# plot avg listings prices over time.
plt.figure(figsize=(15, 8))
plt.plot(review_q1_df.date, review_q1_df.id, color='b', linewidth=0.9)
plt.title("Number of reviews by date")
plt.xlabel('date')
plt.ylabel('number of reviews')
plt.grid()


# It is little noisy, but we can see an increase in the number of Airbnb users. (and the date range is wide than calendar data)  
# And I realize it seems to have a peak at about the same time of each year.  
# So, let's use moving averages to smooth the graph.

# In[51]:


# create rolling mean column
review_q1_df["rolling_mean_30"] = review_q1_df.id.rolling(window=30).mean()

# plot avg listings prices over time.
plt.figure(figsize=(15, 8))
plt.plot(review_q1_df.date, review_q1_df.rolling_mean_30, color='b', linewidth=2.0)
plt.title("Number of reviews by date")
plt.xlabel('date')
plt.ylabel('number of reviews')
plt.grid()


# I tried thirty days (about 1 month) window.  
# The graph became smooth and the trend became clear, and my belief that the peaks were in the same place became stronger.  
# Next, I extract when the peak comes in each year.

# In[24]:


review_q1_df["year"] = review_q1_df.date.dt.year
years = review_q1_df.year.unique()

for year in years:
    if year >= 2010 and year < 2016:
        year_df = review_q1_df[review_q1_df.year == year]
        max_value = year_df.rolling_mean_30.max()
        max_date = year_df[year_df.rolling_mean_30 == max_value].date.dt.date.values[0]
        print(year, max_date, np.round(max_value, 1))


# My hypothesis is correct.  
# The peak seems to be towards the beginning of September!!
# Is this summer vacation?

# ### Answer my Question

# Up to this point, I can answer two of the three questions mentioned at the beginning. 
# First of all, let's answer from the question.
# **How long is the period available for lending by rooms?**
# 
# This was shown when investigating listing data. There were two groups in listings. It's a listing available at spots with a maximum nights less than a week and a listing of a sense of renting available for up to three years. 
# 
# For a further discussion, plot a scatter plot of maximum nights and minimum nights.

# In[52]:


listings_q3_df["min_max_night_diff"] = listings_q3_df.maximum_nights - listings_q3_df.minimum_nights

plt.figure(figsize=(15, 8))
plt.plot(listings_q3_df.maximum_nights, listings_q3_df.minimum_nights, color='b', marker='o', linewidth=0, alpha=0.25)
plt.xlabel('maximum nights')
plt.ylabel('minimum nights')
plt.grid()


# From here, it can be seen that the minimum nights is almost constant regardless of the maximum nights.
# In other words, it can be seen that listings with a long maximum nights are not rented exclusively for rental, but are widely handled from spot use to long-term stay.

# Let's answer the second question.  
# **Is there a busy season?**
# 
# It can not be said exactly because the actual duration of the user's stay are not included in the data, but the number of reviews is considered to be a guide.
# In addition, since periodical peaks appear in the number of reviews annually, it may be considered that the neighborhood is a busy season.
# 
# We found that the biggest busy season was the beginning of September, but how long will it be the busy season? Let's look in more detail!

# In[26]:


review_q2_df = review_q1_df[review_q1_df.year == 2015]

plt.figure(figsize=(15, 8))
plt.plot(review_q2_df.date, review_q2_df.rolling_mean_30, color='b', linewidth=2.0)
plt.title("Number of reviews by date")
plt.grid()


# It's hard to say clearly when to begin and when to end.  
# But, from here it may be able to say the busy season is One month before and after from September.

# ## Data Preparation

# From here, I want to answer the last question, **"Are there any trends of popular rooms?"**.  
# So, Let's begin to clean and process the listing data.

# First, we need to define the target variable 'Popular of listings'. I thought this could be defined as follows.
# 
# [Actual number of times rent] / ([Available days from 2016 to 2017] * (2017 - [Year of the listings open]))
# 
# Since the number of times the listings are actually rent is considered to be proportional to the number of days the property is available, it needs to be scaled, and Furthermore, since the number is considered to increase the earlier the listings is released to the public, we need to scale there as well.
# 
# One thing I can not consider here is the period when the listings was actually rented. The longer the period of rending, the smaller the number of times of rending. However, there are no data to consider this, so I will assume that most of Airbnb users are short-term use. 

# First, we start with a check for null values. This is because we can not use columns that has too null values.

# In[27]:


prepare_df = seattle_listing.copy(deep=True)


# In[28]:


# check null count
df_length = prepare_df.shape[0]

for col in prepare_df.columns:
    null_count = prepare_df[col].isnull().sum()
    if null_count == 0:
        continue
        
    null_ratio = np.round(null_count/df_length * 100, 2)
    print("{} has {} null values ({}%)".format(col, null_count, null_ratio))


# As you can see, it seems that there are columns 90% or more of the values are null, but most of the columns have between 0-30% of null_ratio. Therefore, here, I decided to excluded from analysis the columns with 30% or more null ratio.
# 
# Also, it seems that there are only two null values in `host_since` used to calculate the target variable. This needs to be removed. 

# In[29]:


# detect need drop columns
drop_cols = [col for col in prepare_df.columns if prepare_df[col].isnull().sum()/df_length >= 0.3]

# drop null
prepare_df.drop(drop_cols, axis=1, inplace=True)
prepare_df.dropna(subset=['host_since'], inplace=True)

# check after
for col in prepare_df.columns:
    null_count = prepare_df[col].isnull().sum()
    if null_count == 0:
        continue
        
    null_ratio = np.round(null_count/df_length * 100, 2)
    print("{} has {} null values ({}%)".format(col, null_count, null_ratio))


# Good.  
# Next, we will delete the column that seems not to be a feature of ease of borrowing. Since we do not include natural language processing this time, we will also delete comment-based columns.

# In[30]:


drop_cols = ['listing_url', 'scrape_id', 'last_scraped', 'name', 'summary', 'space', 'description', 'neighborhood_overview',
                'transit', 'medium_url', 'picture_url', 'xl_picture_url', 'host_id', 'host_url', 'host_name', 'host_about', 'host_thumbnail_url',
                'host_picture_url', 'street', 'city', 'state', 'zipcode', 'market', 'smart_location', 'country_code', 'country', 'latitude', 'longitude',
                'calendar_updated', 'calendar_last_scraped', 'first_review', 'last_review', 'amenities', 'host_verifications']

prepare_df.drop(drop_cols, axis=1, inplace=True)


# In[31]:


prepare_df.columns


# Furthermore, delete columns that have only a single value, because the feature value does not make sense.

# In[32]:


drop_cols = []
for col in prepare_df.columns:
    if prepare_df[col].nunique() == 1:
        drop_cols.append(col)
        
prepare_df.drop(drop_cols, axis=1, inplace=True)
prepare_df.columns


# Good. Now I almost finished remove columns that is not used for anallysis. So next, I create target valuable.

# In[33]:


# available days count each listings
listing_avalilable = seattle_calendar.groupby('listing_id')['price'].count().reset_index()
listing_avalilable.columns = ["id", "available_count"]

# merge
prepare_df = prepare_df.merge(listing_avalilable, how='left', on='id')

# create target column
prepare_df['host_since_year'] = pd.to_datetime(prepare_df['host_since']).dt.year
prepare_df["easily_accomodated"] = prepare_df.accommodates / (prepare_df.available_count+1) / (2017 - prepare_df.host_since_year)


# Next, delete the column that seems to be directly related to the objective variable (ex. num_review)

# In[34]:


print("Before: {} columns".format(prepare_df.shape[1]))

drop_cols = ['host_since', 'accommodates', 'availability_30', 'availability_60', 'availability_90', 'availability_365',
                'number_of_reviews', 'review_scores_rating', 'available_count', 'reviews_per_month', 'host_since_year', 'review_scores_value']

prepare_df.drop(drop_cols, axis=1, inplace=True)
print("After: {} columns".format(prepare_df.shape[1]))


# From here, I will turn the data into a form that the model can learn.  
# First, I will convert the category valuables to dummy valuables.

# In[35]:


# convert true or false value to 1 or 0
dummy_cols = ['host_is_superhost', 'require_guest_phone_verification', 'require_guest_profile_picture', 'instant_bookable', 
              'host_has_profile_pic', 'host_identity_verified', 'is_location_exact']

for col in dummy_cols:
    prepare_df[col] = prepare_df[col].map(lambda x: 1 if x == 't' else 0)

# create dummy valuables
dummy_cols = ['host_location', 'host_neighbourhood', 'neighbourhood', 'neighbourhood_cleansed', 'neighbourhood_group_cleansed',
             'property_type', 'room_type', 'bed_type', 'cancellation_policy', 'host_response_time']

prepare_df = pd.get_dummies(prepare_df, columns=dummy_cols, dummy_na=True)


# In[36]:


df_length = prepare_df.shape[0]

for col in prepare_df.columns:
    null_count = prepare_df[col].isnull().sum()
    if null_count == 0:
        continue
        
    null_ratio = np.round(null_count/df_length * 100, 2)
    print("{} has {} null values ({}%)".format(col, null_count, null_ratio))


# Next, handle any remaining null values.

# In[37]:


prepare_df["is_thumbnail_setted"] = 1 - prepare_df.thumbnail_url.isnull()
prepare_df.drop('thumbnail_url', axis=1, inplace=True)
prepare_df.host_response_rate = prepare_df.host_response_rate.fillna('0%').map(lambda x: float(x[:-1]))
prepare_df.host_acceptance_rate = prepare_df.host_acceptance_rate.fillna('0%').map(lambda x: float(x[:-1]))
prepare_df.bathrooms.fillna(0, inplace=True)
prepare_df.bedrooms.fillna(0, inplace=True)
prepare_df.beds.fillna(0, inplace=True)
prepare_df.cleaning_fee.fillna('$0', inplace=True)
prepare_df.review_scores_accuracy.fillna(0, inplace=True)
prepare_df.review_scores_cleanliness.fillna(0, inplace=True)
prepare_df.review_scores_checkin.fillna(0, inplace=True)
prepare_df.review_scores_communication.fillna(0, inplace=True)
prepare_df.review_scores_location.fillna(0, inplace=True)


# Finally, since the value to be acted as a numeric value is recognized as a string, so let's convert it. 

# In[38]:


for col in prepare_df.columns:
    if prepare_df[col].dtypes == 'object':
        print(col)


# In[39]:


prepare_df.price = prepare_df.price.map(lambda x: float(x[1:].replace(',', '')))
prepare_df.cleaning_fee = prepare_df.cleaning_fee.map(lambda x: float(x[1:].replace(',', '')))
prepare_df.extra_people = prepare_df.extra_people.map(lambda x: float(x[1:].replace(',', '')))


# ## Modeling, Evaluation

# This problem is regression.
# I'll use random forest regressor here, and use cross validation to evaluation.

# In[40]:


X = prepare_df.drop(['id', 'easily_accomodated'], axis=1)
y = prepare_df.easily_accomodated.values

rf = RandomForestRegressor(n_estimators=100, max_depth=5)
scores = cross_val_score(rf, X, y, cv=5)


# In[41]:


scores


# I can not predict at all. Why?  
# To consider why, let's make a scatter plot of predicted and actual values.

# In[42]:


rf.fit(X, y)
predictions = rf.predict(X)


# In[43]:


plt.figure(figsize=(8, 8))

plt.plot((0, 4), (0, 4), color='gray')
plt.plot(y, predictions, linewidth=0, marker='o', alpha=0.5)
plt.grid()
plt.xlim((-0.2, 4.2))
plt.ylim((-0.2, 4.2))
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.show()


# The closer to the gray line, the more accurate the predicted value is.  
# Then, at first glance I can understand the predocted values are not accurate.  
# In addition, it is also understood that the prediction is pulled to a small value near 0, and a small prediction is made for the larger value.

# Let's try logarithmic transformation of the target variable.

# In[44]:


X = prepare_df.drop(['id', 'easily_accomodated'], axis=1)
y = np.log(prepare_df.easily_accomodated.values)

rf = RandomForestRegressor(n_estimators=100, max_depth=5)
scores = cross_val_score(rf, X, y, cv=5)
print(scores)


# In[45]:


rf.fit(X, y)
predictions = rf.predict(X)


# In[46]:


plt.figure(figsize=(8, 8))

plt.plot((-10, 10), (-10, 10), color='gray')
plt.plot(y, predictions, linewidth=0, marker='o', alpha=0.5)
plt.grid()
plt.xlim((-8.2, 2))
plt.ylim((-8.2, 2))
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.show()


# It has improved somewhat, but the values are still too dense and many predicted values have been pulled into that part.
# To improve this, I think two method are effective.
# 
# * Review the logic of the objective variable; this definition maybe too complex, it maybe good with more simple definition of target valuables.  
# * Do downsampling in dense areas.
# 
# This analysis doesn't cover that much, but I'm going to add it if there is something in the future.

# ## Summary

# This notebook uses data from the Seattle area of Airbnb and has been analyzed to answer the following questions.
# Here we summarize the answers to those questions.
# 
# * **How long is the period available for lending by rooms?**  
# The histogram of maximum nights shows that there are two groups.    
# One is a listing that can be used at spots such as the maximum number of nights within a week.  
# The other is a listing that supports a wide range of stay from the super long-term stay of the maximum number of stays for three years or more and the minimum number of nights for around two days to the spot use. 
# 
# 
# * **Is there a busy season?**   
# The answer is **Yes**.  
# Apart from the increase in the number of Airbnb users, there was definitely a timely increase in the number of reviews at the same time each year.  
# It is thought that it is about one month around early September and overlaps with the summer vacation time. It is important that the number of properties that can be provided at this time exceeds demand.
# 
# 
# * **Are there any trends of popular rooms?**  
# I could not derive it from my analysis.  
# However, I learned that the score improves by logarithmic transformation. I will find time in the future and try to improve.

# In[ ]:




