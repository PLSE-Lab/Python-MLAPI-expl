#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


os.chdir("/kaggle/input/seattle/")
listing = pd.read_csv("listings.csv")
calendar = pd.read_csv("calendar.csv")


# # Data Cleaning

# In[ ]:


listing.head()


# In[ ]:


calendar.head()


# In[ ]:


listing.info()


# ## 1. Manually examine interesting columns to examine further.
# 
# Here I chose columns that are roughly grouped into these categories:
# 
#  - Information regarding host;
#      - review; performance; experience
#  - Information regarding house;
#      - location; amenities; furniture; policy; affordability

# In[ ]:


listing_col = ['id', 'host_id', 'host_since', 'host_response_time', 'host_response_rate',
               'host_acceptance_rate', 'host_is_superhost', 'host_neighbourhood', 
               'host_listings_count', 'host_total_listings_count', 'host_has_profile_pic', 
               'host_identity_verified', 'neighbourhood_group_cleansed', 'latitude', 
               'longitude', 'property_type', 'room_type', 'accommodates', 'bathrooms', 
               'beds', 'bed_type', 'amenities', 'square_feet', 'guests_included', 
               'extra_people', 'minimum_nights', 'maximum_nights', 'calendar_updated',
               'has_availability', 'number_of_reviews', 'first_review', 'last_review', 
               'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 
               'review_scores_checkin', 'review_scores_checkin', 'review_scores_communication', 
               'review_scores_location', 'review_scores_value', 'instant_bookable', 
               'cancellation_policy', 'require_guest_profile_picture', 
               'require_guest_phone_verification', 'calculated_host_listings_count', 
               'reviews_per_month',
               'price']
listing = listing[listing_col]


# ## 2. Group columns into types; correct for inconsistent data types

# In[ ]:


datetime_cols = ['host_since', 'first_review', 'last_review']
numeric_cols = ['host_response_rate', 'host_acceptance_rate', 'host_listings_count', 
                'host_total_listings_count', 'latitude', 'longitude', 
                'accommodates', 'bathrooms', 'beds', 'square_feet', 'guests_included', 
                'extra_people', 'minimum_nights', 'maximum_nights', 'number_of_reviews', 
                'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 
                'review_scores_checkin', 'review_scores_checkin', 'review_scores_communication', 
                'review_scores_location', 'review_scores_value', 'calculated_host_listings_count', 
                'reviews_per_month', 'price']

nominal_cols = ['host_is_superhost', 'host_has_profile_pic', 
                'host_identity_verified', 'neighbourhood_group_cleansed', 
                'has_availability', 'instant_bookable', 'require_guest_profile_picture', 
                'require_guest_phone_verification']

ordinal_cols = ['host_response_time', 'property_type', 'room_type', 'bed_type', 
                'calendar_updated', 'cancellation_policy']

other_cols = ['amenities']


# ### I. Datetime Columns
# We could convert datetime columns that are read in as strings using the ```pd.to_datetime``` function, or we could pass them into ```pd.read_csv``` explicitly using the ```parse_dates=[...]``` argument.

# In[ ]:


listing['host_since'] = pd.to_datetime(listing['host_since'])
listing['first_review'] = pd.to_datetime(listing['first_review'])
listing['last_review'] = pd.to_datetime(listing['last_review'])
listing[datetime_cols].dtypes


# In[ ]:


# extract # of days since being host/reviewed as possible features
listing['num_days_as_host'] = (pd.Timestamp.today() - listing['host_since']).dt.days
listing['num_days_first_review'] = (pd.Timestamp.today() - listing['first_review']).dt.days
listing['num_days_last_review'] = (pd.Timestamp.today() - listing['last_review']).dt.days


# ### II. Numeric Columns that are read in as strings
# 
# These are the columns that include strings such as "%" or "$". While such representation is convenient for viewing, it is not appropriate for quantitative analysis. We should convert them into numerical columns.

# In[ ]:


listing[numeric_cols].select_dtypes('object')


# In[ ]:


listing.host_response_rate = listing.host_response_rate.str.strip("%")
listing.host_acceptance_rate = listing.host_acceptance_rate.str.strip("%")
listing.extra_people = listing.extra_people.str.strip("$").str.replace(",", "")
listing.price = listing.price.str.strip("$").str.replace(",", "")
listing[listing[numeric_cols].select_dtypes('object').columns] = listing[listing[numeric_cols].select_dtypes('object').columns].astype(np.float64)


# In[ ]:


boolean_cols = list(set(nominal_cols).difference(['neighbourhood_group_cleansed']))
listing[boolean_cols] = listing[boolean_cols].replace("f", 0).replace("t", 1)


# ### III. Encode Ordered Categorical (Ordinal) Columns into Integers
# 
# Ordinal columns are categorical variables that include categories that are ordered in nature. Instead of using ```OneHotEncoding```, it is more appropriate to convert them into integers to reflect the ordered nature of these variables.

# In[ ]:


listing[ordinal_cols]


# In[ ]:


cats = ['within an hour', 'within a few hours', 
        'within a day', 'a few days or more']
for i in range(len(cats)):
    listing['host_response_time'] = listing['host_response_time'].replace(cats[i], i)


# In[ ]:


print(listing['room_type'].unique())
print(listing['bed_type'].unique())
print(listing['calendar_updated'].unique())
print(listing['cancellation_policy'].unique())


# In[ ]:


cats = ['Shared room', 'Private room', 'Entire home/apt']
for i in range(len(cats)):
    listing['room_type'] = listing['room_type'].replace(cats[i], i)

cats = ['Couch', 'Futon', 'Pull-out Sofa', 'Airbed', 'Real Bed']
for i in range(len(cats)):
    listing['bed_type'] = listing['bed_type'].replace(cats[i], i)

cats = ['strict', 'moderate', 'flexible']
for i in range(len(cats)):
    listing['cancellation_policy'] = listing['cancellation_policy'].replace(cats[i], i)


# In[ ]:


listing['calendar_updated'] = listing['calendar_updated'].str.strip(" ago")
calendar_cats = listing['calendar_updated'].unique()
for i in range(len(calendar_cats)):
    if calendar_cats[i] == 'never':
        to_replace = 365 * 3
    elif calendar_cats[i] == 'today':
        to_replace = 0
    elif calendar_cats[i] == 'yesterday':
        to_replace = 1
    elif calendar_cats[i] == 'week':
        to_replace = 7
    else:
        multiple, period = calendar_cats[i].split(" ")
        if period == 'weeks':
            to_replace = int(multiple) * 7
        elif period == 'months':
            to_replace = int(multiple) * 30
        elif period == 'days':
            to_replace = int(multiple)
    listing['calendar_updated'] =     listing['calendar_updated'].replace(calendar_cats[i], to_replace)


# In[ ]:


listing['calendar_updated']


# ### IV. Extract Features from Amenities Columns. For each amenity provided, create a new binary variable representing if the amenity is present

# In[ ]:


listing['amenities']


# In[ ]:


listing['amenities'] = listing['amenities'].apply(lambda x: set(x.strip("{").strip("}").replace('"', '').split(",")).difference(['']))


# In[ ]:


# from tqdm import tqdm
# all_amenities = set()
# for i in tqdm(range(len(listing))):
#     item = listing.loc[i, 'amenities']
#     all_amenities = all_amenities.union(set(item))

all_amenities = set(['24-Hour Check-in',
 'Air Conditioning',
 'Breakfast',
 'Buzzer/Wireless Intercom',
 'Cable TV',
 'Carbon Monoxide Detector',
 'Cat(s)',
 'Dog(s)',
 'Doorman',
 'Dryer',
 'Elevator in Building',
 'Essentials',
 'Family/Kid Friendly',
 'Fire Extinguisher',
 'First Aid Kit',
 'Free Parking on Premises',
 'Gym',
 'Hair Dryer',
 'Hangers',
 'Heating',
 'Hot Tub',
 'Indoor Fireplace',
 'Internet',
 'Iron',
 'Kitchen',
 'Laptop Friendly Workspace',
 'Lock on Bedroom Door',
 'Other pet(s)',
 'Pets Allowed',
 'Pets live on this property',
 'Pool',
 'Safety Card',
 'Shampoo',
 'Smoke Detector',
 'Smoking Allowed',
 'Suitable for Events',
 'TV',
 'Washer',
 'Washer / Dryer',
 'Wheelchair Accessible',
 'Wireless Internet'])


# In[ ]:


for item in all_amenities:
    colname = "has_" + item.replace(" ", "_")
    listing[colname] = listing['amenities'].apply(lambda x: int(item in x))


# In[ ]:


del listing['amenities']


# #### Use longitude and latitude as geographical features instead of categorical variables neighborhood (which are of high ordinality).

# In[ ]:


del listing['host_neighbourhood']


# In[ ]:


listing.columns = listing.columns.str.replace("(", "").str.replace(")", "").str.replace("/", '')


# #### We already have columns ```has_washer``` and ```has_dryer``` so there is no need to also include ```has_Washer_Dryer``` as it is redundant information. Also upon examination ```host_total_listings_count``` is identical to column ```host_listings_count``` so we should consider removing it.

# In[ ]:


del listing['has_Washer__Dryer']
del listing['host_total_listings_count']


# In[ ]:


listing.info()


# ## Lastly, reorganize the order of columns

# In[ ]:


id_col = ['id', 'host_id']
numeric_cols = ['host_since', 'num_days_as_host', 'host_response_time', 
                'host_response_rate', 'host_acceptance_rate', 
                'host_listings_count', 'latitude', 'longitude', 'room_type',
                'accommodates', 'bathrooms', 'beds', 'bed_type', 'square_feet', 'guests_included', 
                'extra_people', 'minimum_nights', 'maximum_nights', 'calendar_updated', 
                'number_of_reviews', 'first_review', 'num_days_first_review', 
                'last_review', 'num_days_last_review',
                'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 
                'review_scores_checkin', 'review_scores_checkin', 'review_scores_communication', 
                'review_scores_location', 'review_scores_value', 'cancellation_policy', 
                'calculated_host_listings_count', 'reviews_per_month']
cat_cols = ['host_has_profile_pic', 'host_identity_verified',
            'host_is_superhost', 'instant_bookable',
            'require_guest_phone_verification', 'require_guest_profile_picture',
            'has_Wireless_Internet', 'has_Essentials',
            'has_Wheelchair_Accessible', 'has_Indoor_Fireplace',
            'has_Carbon_Monoxide_Detector', 'has_Cats', 'has_Safety_Card',
            'has_Shampoo', 'has_Hot_Tub', 'has_Other_pets', 'has_Pets_Allowed',
            'has_Gym', 'has_First_Aid_Kit', 'has_BuzzerWireless_Intercom',
            'has_Doorman', 'has_Suitable_for_Events', 'has_Fire_Extinguisher',
            'has_Dogs', 'has_Pets_live_on_this_property', 'has_Breakfast',
            'has_Heating', 'has_Laptop_Friendly_Workspace', 'has_Smoking_Allowed',
            'has_Internet', 'has_Air_Conditioning', 'has_24-Hour_Check-in',
            'has_TV', 'has_Elevator_in_Building', 'has_Pool', 'has_Cable_TV',
            'has_Free_Parking_on_Premises', 'has_Washer', 'has_Iron',
            'has_FamilyKid_Friendly', 'has_Kitchen', 'has_Dryer',
            'has_Smoke_Detector', 'has_Hair_Dryer', 'has_Lock_on_Bedroom_Door',
            'has_Hangers', 'property_type', 'neighbourhood_group_cleansed']
target = ['price']


# In[ ]:


listing_col = []
listing_col.extend(id_col)
listing_col.extend(numeric_cols)
listing_col.extend(cat_cols)
listing_col.extend(target)
listing = listing[listing_col]
listing.head()


# In[ ]:


calendar['date'] = pd.to_datetime(calendar['date'])
calendar.loc[:, 'price'] = calendar['price'].str.replace('$', '').str.replace(",", "").astype(np.float64)
calendar.info()


# In[ ]:


# save to output
listing.to_csv('/kaggle/working/listings.csv', index=False)
calendar.to_csv('/kaggle/working/calendar.csv', index=False)


# In[ ]:


listing.head()

