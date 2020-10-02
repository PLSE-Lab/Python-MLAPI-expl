#!/usr/bin/env python
# coding: utf-8

# <h2>This project will follow CRISP-DM methdology which provides a structed process to approach data science problems</h2>
# <br>
# <h2>It's constructed of 5 steps:</h2>
# <ol>
# <li>Business understanding</li>
# <li>Data understanding and preparing</li>
# <li>Modeling</li>
# <li>Evaluation</li>
# <li>Conclusion</li>
# </ol>

# <h2>1. Business Understanding</h2>
# 

# <pr>To understand how it works sa well as asking a right question I looked at a business model canvas for airbnb. From my understanding my questions will be about:</pr>
# <br>
# <ol>
#     <li>What are the factors that affect the price of rooms?</li>
#     <li>In which period of the year is there a high number of listings?</li>
#     <li>how to choose a suitable room at low cost?</li>
#     <li>What kind of room are most requested?</li>
# </ol>    
# 
# <pr>We can be able to answer these questions by analysing publicly accessible <a href="https://www.airbnb.com/">AirBnB</a> data, available on <a href="https://www.kaggle.com/airbnb/seattle">Kaggle.</pr>

# <h2>2. Data Understanding and Preparing</h2>
# <pr>We will do some steps to work with this dataset<pr>
# <ul>
#     <li>What data we have?</li>
#     <li>What is the missing values we have?</li>
#     <li>How will we deal with the missing values?</li>
#     <ol>
#         <li>Reformat the information and imputing missing values</li>
#         <li>Delete data that we will not need</li>
#     </ol><br>
#     <li>Find out correlations between our interesting variables</li>
#     <li>Use machine learning for further analysis</li>
# </ul>

# In[ ]:


#Import linear algebra and data manipulation
import numpy as np
import pandas as pd

#Import plotting packages
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error 

#Import machine learning
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


#To show more rows and columns
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 20)


# <pr>We have 3 datasets:</pr>
# <ul>
#     <li>calendar.csv - calendar data for the listings: availability dates, price for each date.</li>
#     <li>listings.csv - summary information on listing in Seattle such as: location, host information, cleaning and guest fees, amenities etc.</li>
#     <li>reviews.csv - summary review data for the listings.</li>
# </ul>  
# <pr>But we will use first two.</pr>

# In[ ]:


bnb_l = pd.read_csv('../input/seattle/listings.csv')
bnb_c = pd.read_csv('../input/seattle/calendar.csv')


# <pr>I want take a look at the names of the columns for each one</pr>

# In[ ]:


bnb_l.columns


# In[ ]:


bnb_c.columns


# <pr>Now it's the time to select our interesting columns from each data set, which will help us answering to our questions</pr>

# In[ ]:


reviews = bnb_l[['review_scores_rating',
                'review_scores_accuracy', 'review_scores_cleanliness',
                'review_scores_checkin', 'review_scores_communication',
                'review_scores_location', 'review_scores_value']]


# In[ ]:


rooms = bnb_l[['room_type', 'accommodates',
               'bathrooms', 'bedrooms', 'beds', 'bed_type']]


# In[ ]:


hosts = bnb_l[['host_since', 'host_response_rate', 'host_acceptance_rate', 'host_is_superhost','extra_people']]


# In[ ]:


bnb_c['month'] = pd.to_datetime(bnb_c['date']).dt.month
bnb_c['year'] = pd.to_datetime(bnb_c['date']).dt.year


# In[ ]:


airbnb_l = pd.concat((reviews,rooms,hosts),axis=1)


# In[ ]:


airbnb_l.head()


# In[ ]:


airbnb_l['host_is_superhost'] = airbnb_l['host_is_superhost'].map({'f':0 , 't':1})


# In[ ]:


airbnb_l.info()


# In[ ]:


airbnb_l = pd.concat((airbnb_l,bnb_l['id']),axis=1)


# In[ ]:


airbnb_l.rename(index=str, columns={"id": "listing_id"}, inplace= True)


# In[ ]:


airbnb_df = pd.merge(bnb_c, airbnb_l , on= 'listing_id')


# In[ ]:


airbnb_df.drop(['date','available'], axis=1, inplace=True)


# In[ ]:


airbnb_df_1 = airbnb_df.copy()
airbnb_df_1.dropna(subset=['price'],inplace=True)


# In[ ]:


#Convert 'price' to float
airbnb_df_1['price'] = airbnb_df_1['price'].str.replace("[$, ]", "").astype("float")


# In[ ]:


listings_per_month = pd.Series([12])
for i in range(1, 13):
    listings_per_month[i] = len(airbnb_df_1[(airbnb_df_1['month'] == i) & (airbnb_df_1['year'] == 2016)]['listing_id'].unique())
    
listings_per_month = listings_per_month.drop(0)


months = ['Jan', 'Feb', 'Mar', 'Apr', 'May' , 'Jun' , 'Jul' , 'Aug' , 'Sep' , 'Oct' , 'Nov' , 'Dec']
plt.subplots(figsize = (15,10))
ax = sns.pointplot(x = months, y = listings_per_month)
plt.ylabel('Number of listings')
plt.xlabel('Months')
plt.title('Number of listings per month in 2016')

plt.savefig('number of available listings.png')


# <pr>The graph above determines the number of listings per month. So, It's clear that in April there are a high number of listings on the opposite in July</pr>
# <br>
# <pr>Because of this graph, I suppose that the price in July is very high. Let's prove that by plot a graph between the price and the number of months</pr>

# In[ ]:


price_per_month = airbnb_df_1.groupby('month')['price'].mean()

plt.subplots(figsize = (15,10))
ax = sns.pointplot(x = months, y = price_per_month)
plt.ylabel('Number of listings')
plt.xlabel('Months')
plt.title('The price per month in 2016')

plt.savefig('The price per month.png')


# <pr>Ok .. my assumption was right</pr>

# In[ ]:


room = [airbnb_df_1['room_type'].value_counts()[0],
        airbnb_df_1['room_type'].value_counts()[1],
        airbnb_df_1['room_type'].value_counts()[2]]

room_type = ['Entire home/apt', 'Private room', 'Shared room']

plt.figure(figsize=(15,10))
plt.ylabel('Number of room requests')
plt.xlabel('Room Type')
plt.title('The kind of room which are most requested')
plt.bar(room_type,room)

plt.savefig('The kind of room which are most requested.png')


# <pr>Most of the people here as shown preferred to an apartment or an entire home.</pr>

# In[ ]:


price = airbnb_df_1.groupby('listing_id')['price'].mean()
plt.figure(figsize=(15,10))
plt.hist(price,bins=100)

plt.ylabel('Number of listings')
plt.xlabel('Price')
plt.title('Number of listings depending on price')

plt.savefig('Price distrubution.png')

plt.show()


# <pr>We can say that the price of listings is between 25 USD to 350 USD.</pr>

# In[ ]:


reviews = ['review_scores_rating',
           'review_scores_accuracy', 'review_scores_cleanliness',
           'review_scores_checkin', 'review_scores_communication',
           'review_scores_location', 'review_scores_value']
for review in reviews:
    airbnb_df_1[review].fillna(airbnb_df_1[review].mean(),inplace=True) 


# In[ ]:


room_features = ['accommodates', 'bathrooms', 'bedrooms', 'beds']
for feature in room_features:
    airbnb_df_1[feature].fillna(airbnb_df_1[feature].mode()[0],inplace=True)


# In[ ]:


airbnb_df_1.info()


# In[ ]:


extra_cols = ['host_since', 'host_is_superhost', 'extra_people']
for col in extra_cols:
    airbnb_df_1[col].fillna(airbnb_df_1[col].mode()[0],inplace=True)


# In[ ]:


airbnb_df_1['host_since'] = pd.to_datetime(airbnb_df_1['host_since']).dt.year


# In[ ]:


def get_extra_people_fee(row):
    ''' Return 1 when the is fee for exatra people '''
    if row['extra_people'] == '$0.00':
        return 0.0
    else:
        return 1.0


# In[ ]:


airbnb_df_1['extra_people'] = airbnb_df_1.apply(lambda row: get_extra_people_fee(row),axis=1)


# In[ ]:


airbnb_df_1.info()


# <h4>Correlation between Price and Other Features</h4>

# In[ ]:


corr_cols = ['review_scores_rating',
             'review_scores_accuracy', 'review_scores_cleanliness',
             'review_scores_checkin', 'review_scores_communication',
             'review_scores_location', 'review_scores_value','accommodates', 'bathrooms', 'bedrooms', 'beds',
             'host_since', 'host_is_superhost', 'extra_people', 'price']

corrs = np.corrcoef(airbnb_df_1[corr_cols].values.T)
hm=sns.heatmap(corrs, cbar = True, annot=True, square = True, fmt = '.2f',
              yticklabels = corr_cols, xticklabels = corr_cols).set_title('Correlations heatmap')

sns.set(rc={'figure.figsize':(20,20)})
fig = hm.get_figure()
fig.savefig('correlations.png')


# <pr>From the correlations heatmap diagram we can see that price is correlated with number of accomodates, bathrooms, bedrooms and beds. And it's reasonable because the more area is, the high price is </pr>
# 
# <br>
# <br>
# <pr>by this step, I answered all my questions. but, I want to use a machine learning model to predict the price of the room. </pr>
# <br>
# <h2>3. Modeling</h2>
# <h3>Machine Learning</h3>

# In[ ]:


categorical_cols = list(airbnb_df_1.select_dtypes(include=['object']).columns)
for col in  categorical_cols:
    airbnb_df_1 = pd.concat([airbnb_df_1.drop(col, axis=1), pd.get_dummies(airbnb_df_1[col], prefix=col, prefix_sep='_',
                            drop_first=True, dummy_na=True)], axis=1)
#drop listing_id and year columns
airbnb_df_1 = airbnb_df_1.drop(columns = ['listing_id', 'year'])


# In[ ]:


airbnb_df_1.columns


# In[ ]:


airbnb_df_1.drop(['host_response_rate_17%',
       'host_response_rate_25%', 'host_response_rate_30%',
       'host_response_rate_31%', 'host_response_rate_33%',
       'host_response_rate_38%', 'host_response_rate_40%',
       'host_response_rate_43%', 'host_response_rate_50%',
       'host_response_rate_53%', 'host_response_rate_55%',
       'host_response_rate_56%', 'host_response_rate_57%',
       'host_response_rate_58%', 'host_response_rate_60%',
       'host_response_rate_63%', 'host_response_rate_64%',
       'host_response_rate_65%', 'host_response_rate_67%',
       'host_response_rate_68%', 'host_response_rate_69%',
       'host_response_rate_70%', 'host_response_rate_71%',
       'host_response_rate_75%', 'host_response_rate_76%',
       'host_response_rate_78%', 'host_response_rate_80%',
       'host_response_rate_81%', 'host_response_rate_82%',
       'host_response_rate_83%', 'host_response_rate_86%',
       'host_response_rate_87%', 'host_response_rate_88%',
       'host_response_rate_89%', 'host_response_rate_90%',
       'host_response_rate_91%', 'host_response_rate_92%',
       'host_response_rate_93%', 'host_response_rate_94%',
       'host_response_rate_95%', 'host_response_rate_96%',
       'host_response_rate_97%', 'host_response_rate_98%',
       'host_response_rate_99%', 'host_response_rate_nan'],axis=1,inplace=True)


# In[ ]:


airbnb_df_1.columns


# In[ ]:


airbnb_df_1.info()


# In[ ]:


airbnb_df_1.head()


# In[ ]:


#Prepare train and test
X = airbnb_df_1.drop(['price'],axis=1)
y = airbnb_df_1['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=42)


#I will train Random Forest Regressor model
forest = RandomForestRegressor(n_estimators=100, 
                               criterion='mse', 
                               random_state=42, 
                               n_jobs=-1)
forest.fit(X_train, y_train.squeeze())

#calculate scores for the model
y_train_preds = forest.predict(X_train)
y_test_preds = forest.predict(X_test)

print('Random Forest MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_preds),
        mean_squared_error(y_test, y_test_preds)))
print('Random Forest R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_preds),
        r2_score(y_test, y_test_preds)))


# In[ ]:




