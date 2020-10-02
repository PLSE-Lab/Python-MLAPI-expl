#!/usr/bin/env python
# coding: utf-8

# ## Analysis of hotel booking data
# 
# 
# **Hotel bookings EDA**
# 1. Inspect and clean the data
# 2. Discern the timeframe of bookings
# 3. Find the average pricer per person per night
# 4. Variation of price per room type
# 5. Fraction of bookings that result in cancellation
# 6. Relationship between current cancellation and number of previous cancellations
# 7. Cancellations per month
# 8. Busiest months by number of hotel guests
# 9. Meal types per month per hotel
# 10. Bookings and cancellations by market segment
# 
# **Preprocessing and feature selection**
# 1. Categorical encoding and feature analysis
# 2. Numeric feature analysis
# 
# **Model evalution and cross validations scores**
# 
# **Final points about the dataset**
# 
# 

# 

# Import libraries to begin with.

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# **Hotel bookings EDA

# Prepare the data for analysis.
# 
# 1) Look for NaNs/NULLs
# 
# 2) Check for 'unknown' values and blanks
# 
# 3) Inspect the data for any other anonmalies that may affect analysis and act as unecessary outliers
# 
# 4) Replace all values (meaningfully)

# In[ ]:


path = 'C:/Users/Damie/OneDrive/Desktop/'

hotel_bookings_path = pd.read_csv('../input/hotel-booking-demand/hotel_bookings.csv')
bk = hotel_bookings_path


# In[ ]:


### Prepare and clean the data ready for analysis, feature selection and ML classification algorithm
### Print a table with the sum of NaNs and NULLs for each column
a = bk.isna().sum()
print (a)


# In[ ]:


### Columns with NaNs and/or NULLs have been identified, now we must decide what to do with them
### Arrival_month - as it is 1 unknown month we shall ignore this
### Children - Find the mean and round to the nearest integer and replace
### Country - Replace with 'Not known'
### Agents and Companies - Replace with 0, not sure if this means a third party made the booking
### Meal - Has some undefined for Resort hotel, as these are categorical I won't make any changes
print ('Average number of children' ,bk['children'].mean())


# In[ ]:



### As the rounded average of children is 0 we will replace the NaNs with 0
bk_replace = {'children' : 0, 'country':'Not Known', 'agent':0,'company':0}
bk_clean = bk.fillna(bk_replace)
bk = bk_clean
### Print again to ensure replacement occurred
a = bk.isna().sum()
print (a)


# Next, we create our connection to sqlite3 which returns a connection object that represents a database, 
# then run a quick column query to see what kind of information we are interrogating.

# In[ ]:


### Create SQLite connection
e = pd.read_sql_query
conn = sqlite3.connect('bookings.db')
bk.to_sql('hotel_bookings', conn, if_exists='replace', index=False)



bk_columns = bk.columns

print (bk_columns)


# We can see the data is taken from two separate hotels.

# In[ ]:


print (pd.unique(bk.hotel))


# There is data from the following years

# In[ ]:


print (pd.unique(bk.arrival_date_year))


# The data is taken from July 2015 - August 2017

# In[ ]:


_hotel = 'Resort Hotel'
_year = 2015

def _checks(hotel, year):
    dates = bk
    dates = dates.loc[dates['hotel'] == hotel]
    dates = dates.loc[dates['arrival_date_year'] == year]
    dates = dates[['arrival_date_month']]
    dates = dates.drop_duplicates()
    print (dates)

_checks(hotel = _hotel, year = _year)


# In[ ]:


_hotel = 'Resort Hotel'
_year = 2017

def _checks(hotel, year):
    dates = bk
    dates = dates.loc[dates['hotel'] == hotel]
    dates = dates.loc[dates['arrival_date_year'] == year]
    dates = dates[['arrival_date_month']]
    dates = dates.drop_duplicates()
    print (dates)

_checks(hotel = _hotel, year = _year)


# Next we will create a new column where we find the average price per person by divind the daily rate by the number of guests

# In[ ]:


### find the average price per person for each hotel
avg = bk
avg['avg'] = bk['adr'] / (bk['adults'] + bk['children'])
avg = avg.loc[avg['is_canceled'] == 0 ]


# Lets consider how the average price per person of the hotels changes over the time frame of the data

# In[ ]:


### Finding the the Price chnages over the course of the year
avg_ppp = avg[['hotel', 'arrival_date_month', 'arrival_date_year', 'avg']]
avg_ppp = avg_ppp.replace([np.inf], np.nan)
avg_ppp = avg_ppp.dropna()
avg_ppp = avg_ppp.sort_values(by='avg', ascending=True)
ordered_months = ["January", "February", "March", "April", "May", "June", 
          "July", "August", "September", "October", "November", "December"]
avg_ppp["arrival_date_month"] = pd.Categorical(avg_ppp["arrival_date_month"], categories=ordered_months, ordered=True)


# In[ ]:


chart = sns.relplot(x = 'arrival_date_month', y = 'avg', col='arrival_date_year', hue = 'hotel', kind = 'line',data=avg_ppp)
chart.set_xticklabels(rotation=45, horizontalalignment='right')
plt.subplots_adjust(top = 0.7)
plt.suptitle('Average Price Per Person For Each Year of Data')
chart.set_axis_labels('Month','Price (Euro)')


# Next, let us consider the price variation by the room types for each hotel.

# In[ ]:


### Prices per room type per hotel
room_type = avg
room_type = room_type[['hotel', 'avg','reserved_room_type']].sort_values(by='reserved_room_type')


# In[ ]:


plt.figure(figsize=(16,6))
chart = sns.boxplot(x='reserved_room_type', y='avg',hue='hotel', data=room_type)
plt.suptitle('Price Per Room Type')
plt.xlabel('Room Type')
plt.ylabel('Price (Euro)')
plt.ylim(0,360)


# Now lets make an observation of the fraction of total bookings that result in cancellations

# In[ ]:


### Hotel cancellations
cancellations = bk[['hotel', 'is_canceled']]


# In[ ]:


### Plot of hotel cancellations

chart = sns.barplot(x='is_canceled', y='hotel', data=cancellations)
plt.title('Fraction of Bookings that Result in Cancellations')
plt.xlabel('Fraction of Total Bookings That Are Cancellations')


# We can see that cancellations at the resort hotel constitute just over 25% of the total bookings, whereas, city hotel constitutes just over 40%

# Investigating the cancellations further, considering only bookings that resulted in cancellations of what proportion of those bookings were first time cancellations and what proportion had 1 or more previous cancellations (repeat offenders)

# In[ ]:


tot_cancel =        e('''
                      
                          SELECT hotel, COUNT(previous_cancellations) AS Prev, previous_cancellations
                          FROM hotel_bookings
                          WHERE is_canceled = 1
                          GROUP BY previous_cancellations, hotel
                    
                      ''',conn)


# In[ ]:


### Plot of count of cancellations on frequency of previous cancellation
chart = sns.catplot(x = 'previous_cancellations', y ='Prev', hue='hotel',kind='bar', data=tot_cancel)
chart.set_axis_labels('Number of Previous Cancellations', 'Total Cancellations')


# From the figure above we see the majority of cancellations for both hotels were first time cancellations, with ever decreasing number of total cancellations as the number of previous cancellations increases.

# What is the percentage of cancellations for each hotel per month?

# In[ ]:


### Percentage of cancellations per month
cancel_month =      e('''
                      
                          SELECT hotel, arrivaL_date_month, COUNT(is_canceled)
                          FROM hotel_bookings
                          WHERE arrival_date_month NOT LIKE 'None' AND is_canceled = 1
                          GROUP BY arrival_date_month, hotel, is_canceled
                          
                          
                    ''', conn)

total_bk_month =    e('''
                      
                          SELECT hotel, COUNT(arrival_date_month), arrival_date_month
                          FROM hotel_bookings
                          WHERE arrival_date_month NOT LIKE 'None'
                          GROUP BY arrival_date_month, hotel
                          
                    ''', conn)
                    
percent_cancel_month = pd.concat([cancel_month,total_bk_month['COUNT(arrival_date_month)']], axis=1, sort=False)
percent_cancel_month['percentile'] = (percent_cancel_month['COUNT(is_canceled)'] / percent_cancel_month['COUNT(arrival_date_month)']) * 100
percent_cancel_month = percent_cancel_month[['hotel', 'arrival_date_month','percentile']]


# In[ ]:


### Plot of percentile cancellations by month per hotel
chart = sns.catplot(x='arrival_date_month',y='percentile', hue='hotel', kind = 'bar', data=percent_cancel_month, height=6, aspect=2)
chart.set_xticklabels(rotation=45, horizontalalignment='right')
plt.subplots_adjust(top = 0.7)
plt.suptitle('Cancellations per Month For City and Resort Hotel')
chart.set_axis_labels(' Month', 'Percentage of Cancellations (%)')


# What are the busiest months per month for each hotel?

# In[ ]:


### Total guests per month per hotel per year
guest =        e(''' 
                        
                         SELECT hotel, arrival_date_year, arrival_date_month, SUM(adults + children) AS total_guests FROM hotel_bookings 
                         WHERE is_canceled = 0
                         GROUP BY hotel, arrival_date_month, arrival_date_year
                         
                         
                ''',conn)

guest["arrival_date_month"] = pd.Categorical(guest["arrival_date_month"], categories=ordered_months, ordered=True)


# In[ ]:


chart = sns.catplot(x = 'arrival_date_month', y = 'total_guests', col='arrival_date_year', hue = 'hotel', kind='bar',data=guest)
chart.set_xticklabels(rotation=45, horizontalalignment='right')
plt.subplots_adjust(top = 0.7)
plt.suptitle('Total number of guests per hotel per month')
chart.set_axis_labels(' Month', 'Total Guests')


# Some more benign analysis here, however, we provide some figures below on the meal type popularity and variation relative to the hotel offering the meal packages

# In[ ]:


### Meal types
meal_types =        e(''' 
                        
                         SELECT hotel, arrival_date_month, COUNT(meal) as meal_count, meal FROM hotel_bookings 
                         WHERE is_canceled = 0
                         GROUP BY hotel, arrival_date_month, meal
                         
                         
                ''',conn)
meal_types["arrival_date_month"] = pd.Categorical(meal_types["arrival_date_month"], categories=ordered_months, ordered=True)


# In[ ]:


### Plot of meal-types by each hotel per month
chart = sns.catplot(x = 'arrival_date_month', y = 'meal_count', col='hotel', hue = 'meal', kind = 'bar', data=meal_types, height = 5, aspect = 2 )
chart.set_xticklabels(rotation=45, horizontalalignment='right')
plt.subplots_adjust(top = 0.7)
plt.suptitle('Meal Types Per Month For Each Hotel')
chart.set_axis_labels('Month', 'Number of Meals')


# Let's quickly take a look at where the customers are travelling treating each hotel as a separate case

# In[ ]:


### Country of origin of guests
countries =         e('''
                      
                          SELECT hotel, COUNT(country), country
                          FROM hotel_bookings
                          GROUP BY hotel, country
                          
                          
                          ''',conn)


# In[ ]:


get_ipython().run_cell_magic('HTML', '', "<div class='tableauPlaceholder' id='viz1586788844804' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;5B&#47;5BPN38X7N&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='path' value='shared&#47;5BPN38X7N' /> <param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;5B&#47;5BPN38X7N&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1586788844804');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# Interact with the figure above using the panel in the top left. Overall city hotel has the most bookings every month, naturally the biggest proportion of people travelling to the hotels tends to be city hotel. The exceptions being African countries and northern European countries. This be indicative of business/leisure purpose for travelling. 

# Investigation into the effect of the length of the waiting list on cancellations. 

# In[ ]:


### Days in waiting list relationship to cancellations
dinwl =              e('''
                      
                          SELECT days_in_waiting_list, COUNT(is_canceled) AS Total
                          FROM hotel_bookings
                          WHERE is_canceled = 0
                          GROUP BY days_in_waiting_list
                          
                          ''',conn)

print (dinwl.sort_values(by='Total'))


# In[ ]:


### Days in waiting list relationship to cancellations
dinwl =              e('''
                      
                          SELECT days_in_waiting_list, COUNT(is_canceled) AS Total
                          FROM hotel_bookings
                          WHERE is_canceled = 1
                          GROUP BY days_in_waiting_list
                          
                          ''',conn)

print (dinwl.sort_values(by='Total'))


# For cancelled and not cancelled, by far the number of cancellations with zero waiting days is the most populous. Sorting the number of cancellations by ascending order we can see that the number of days elapsed on the waiting list neither decreases nor increases in correlation with the number of cancellations.

# In[ ]:


### Total bookings by market
market_total = bk['market_segment'].value_counts()


# In[ ]:


### Total bookings made by market segment
plt.figure(figsize=(16,7))
chart = sns.barplot(market_total.index, market_total.values)
plt.title('Total Bookings by Different Markets')
plt.xlabel('Market')
plt.ylabel('Number of Bookings')


# In[ ]:


### Market segment with associated cancelations

market =            e('''
                      
                          SELECT market_segment, COUNT(is_canceled) AS Total
                          FROM hotel_bookings
                          WHERE is_canceled == 1
                          GROUP BY market_segment
                          
                          ''',conn)
market_order = market.sort_values(by='Total', ascending=False)


# In[ ]:


### Number of cancelations by market segment
plt.figure(figsize=(16,7))
chart = sns.barplot(x='market_segment', y='Total', data=market_order)
plt.title('Number of Cancellations by Each Market')
plt.xlabel('Market')
plt.ylabel('Number of Cancellations')

**Pre-processing and feature selection**
# Now we can look at the weight of the categorical features on the target variable, first define our categorical features and the target variable. We have ignored country and reservation status date data as one hot encoding will produce many columns. Actually I didn't ignore these first time around and it took 10 minutes to print the figure with the new columns on. Woops.
#  Also, reservation status has been taken out, if that's included the canceled status far outweighs all over features, obviously.

# In[ ]:


categ = bk[['hotel','arrival_date_month','meal','market_segment', 
            'distribution_channel','deposit_type','customer_type','reserved_room_type',
              'assigned_room_type']]

y = y = bk['is_canceled']


# We will use get_dummies to one hot encode the categorical variables, we label train and test data with the suffix cat to distinguish it from the full traning and testing data later

# In[ ]:


X_cat = pd.get_dummies(categ, prefix=['H','Mn', 'Me', 'Mk', 'Dst', 'Dep', 'Cust','Rm','At'])
                                    
X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X_cat, y, test_size=0.25, random_state=1)


# Use chi2 from SelectKBest to determine the statistical weight of each category. We have alot of unique categories to consider in the data so we will change default (k=10) to k=20.

# In[ ]:


fs = SelectKBest(score_func=chi2, k=20).fit(X_train_cat, y_train_cat)
X_train_fs = fs.transform(X_train_cat)
X_test_fs = fs.transform(X_test_cat)

num_scores = fs.scores_

score_labels = X_train_cat.columns.transpose()
feature_scores = zip(fs.get_support(),score_labels)

for i, j in feature_scores:
    if i == True:
        print (i,j)


# We can see we have some arrival_months and not others, for the sake of generality we will keep all the arrival_months in for the actual model. The only category where all sub-categories were not selected is Meal. Everything else will be kept for generality.

# In[ ]:


### Plot Chi2 scores
plt.figure(figsize=(24,9))
plt.bar(score_labels, num_scores)
plt.xticks(rotation=90)
plt.xlabel('Categories')
plt.ylabel('Category Chi2 Weighting')
plt.show()


# We will perform pairwise column calculations using pearsons correlation for the numeric inputs.

# In[ ]:


### Numeric Data
num_feat = bk[['is_canceled','lead_time','arrival_date_year','arrival_date_week_number','arrival_date_day_of_month',
               'stays_in_weekend_nights','stays_in_week_nights','adults','children','babies','is_repeated_guest','previous_cancellations','previous_bookings_not_canceled',
               'booking_changes','agent','company','adr','required_car_parking_spaces','total_of_special_requests',
                'reserved_room_type','avg','days_in_waiting_list']]

num_feat = num_feat.corr(method='pearson')


# In[ ]:


### Plot pearson correlation
plt.figure(figsize=(16,14))
sns.heatmap(num_feat, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[ ]:


nume = num_feat['is_canceled']
print (nume.abs().sort_values(ascending = False))


# This gives us an ordered list of correlation weighting.

# **Model evaltuation and cross validation scores**
# 
# 
# Now we develop the full model for training and testing the data bearing in mind the results from the feature selection method.

# In[ ]:


### Build full training and testing data based on the previous results
### from the feature selection analysis
bk = bk.drop(['reservation_status_date','country','reservation_status','avg'], axis=1)


cat = ['hotel','arrival_date_month','meal','market_segment', 
            'distribution_channel','deposit_type','customer_type','reserved_room_type',
              'assigned_room_type']

num = ['lead_time','arrival_date_year','arrival_date_week_number','arrival_date_day_of_month',
               'stays_in_weekend_nights','stays_in_week_nights','adults','children','babies','is_repeated_guest','previous_cancellations','previous_bookings_not_canceled',
               'booking_changes','agent','company','adr','required_car_parking_spaces','total_of_special_requests','days_in_waiting_list']

all_features = cat + num

X = bk.drop(['is_canceled'], axis=1)[all_features]
y = bk['is_canceled']

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=1)


# Build the pipepline for the numeric and categorical data. 

# In[ ]:


cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("onehot", OneHotEncoder(handle_unknown='ignore'))])


transform_num = StandardScaler()

process_features = ColumnTransformer(transformers=[('num', transform_num, num),
                                                   ('cat', cat_transformer, cat )])


# Next we choose some classifer models to test on our data.

# In[ ]:


models = []
models.append(('LR', LogisticRegression(max_iter=500)))
models.append(('DTC', DecisionTreeClassifier()))
models.append(('RF', RandomForestClassifier()))


# A solver warning usually appears for the LogisticRegression model, after many attempts at testing and fine tuning I was sick of seeing the message so I set a default number of iterations to prevent the warning displaying any more. I tested all the way to 4000 iterations but it had very little impact on the final result, therefore, we will keep the run time low

# In[ ]:


results = []
names = []
for name, model in models:
    pl = Pipeline([('process', process_features), ('models', model)])
    kfold = StratifiedKFold(n_splits=10)
    cv_results = cross_val_score(pl, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


# The random forest model gives the highest cross validation score.

# **Final points about the dataset**

# If we look back at the feature selection data we can see that the deposit type 'Non Refundable' has the greatest correlation with a cancellation being made which is the opposite of what I would have expected. 
# This relationship has been confirmed by Susmit Vengurlekar and Marcus Wingen in their discussions also. This paper https://www.researchgate.net/publication/310504011_Predicting_Hotel_Booking_Cancellation_to_Decrease_Uncertainty_and_Increase_Revenue dated before the paper given in this dataset is produced by the same authors and relates to the same hotel dataset and determines the correlation, in regards to booking type and cancellation, to be the opposite of what we have been finding in our dataset which leads me to believe that the data has been labelled incorrectly by the authors in the most recent paper.
# 
# Finally, this is my first notebook and my first attempt at machine learning, so, if you have time, please voice any constructive critisicms in the comments as I would appreciate any and all input.
# 
