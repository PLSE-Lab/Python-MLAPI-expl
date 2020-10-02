#!/usr/bin/env python
# coding: utf-8

# # Fixing Conflicts in the geoNetwork Attributes

# In[ ]:


import numpy as np
import pandas as pd
import random
import re

from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


# In this notebook, we will investigate the data accuracy and consistency issues in the geoNetwork attributes and attempt to remedy the errors.

# First we load the processed training data. The processing is done with my other kernel script "GStore Revenue Data Preprocessing".

# In[ ]:


dat = pd.read_pickle("../input/gstore-revenue-data-preprocessing/train.pkl")


# Now we focus our attention on the location-related data in the dataset, which is stored in the *geoNetwork* section.

# In[ ]:


geo_colnames = [c for c in dat.columns if re.match(r'geoNetwork', c) is not None]


# In[ ]:


geo_colnames


# For now, we ignore the *networkDomain* attribute, although it might prove useful later.

# In[ ]:


pure_geo_columns = [c for c in geo_colnames if c != 'geoNetwork.networkDomain']


# ---------------------------------------

# In[ ]:


for c in pure_geo_columns:
    dat.loc[:, c] = dat[c].cat.add_categories('N/A').fillna('N/A')


# In[ ]:


selected = dat.loc[:, dat.columns.isin(pure_geo_columns)].copy()


# In[ ]:


selected.groupby(pure_geo_columns).size().reset_index().head(30)


# If we look at the first few columns of the *geoNetwork* columns aggregated and counted by attribute values, it is apparent that there are many mistakes and missing values in the data. Most importantly, we see some city names repeated multiple times, often with obviously wrong countries and continents. While there are some cases where cities across the world share names, it makes up only a small portion of all such cases. It might seem there are no other way than to manually investigate each of the cases.

# ## Method 1. Careful Investigation

# If we look closely, we can find that within the six attributes, there are two groups of attributes, three in each group, in which they tend to agree with each other more often than not. The first group is ('geoNetwork.continent', 'geoNetwork.country', 'geoNetwork.subContinent'), and the second group is ('geoNetwork.city', 'geoNetwork.metro', 'geoNetwork.region').
# 
# Let us first look at the first group:

# In[ ]:


country_part = selected.groupby(
    ['geoNetwork.continent', 'geoNetwork.country',
     'geoNetwork.subContinent']).size().reset_index()

country_part.head(20)


# If we try to find duplicated country names in this subset of data, we will actually find none! Therefore there should not be any factual conflicts within attributes in this group.

# In[ ]:


country_part[country_part['geoNetwork.country'].duplicated(keep=False)]


# Things are more interesting in the other group of attributes:

# In[ ]:


city_part = selected.groupby(
    ['geoNetwork.city', 'geoNetwork.metro',
     'geoNetwork.region']).size().reset_index()

city_part.head(20)


# As we see below, there actually exist some duplicated city names in unique combinations of (city, metro, region). If we look closely, most of the cases are caused by missing values that only exist in part of the rows, but in some cases it is actually two different cities with the same names (Guadalajara Spain vs Guadalajara Mexico, Kansas City	 Kansas vs Kansas City Missouri, etc.). If we can pick out those special cases, we can resolve all conflicts within this subset of attributes, and we will only have to investigate if there are any non-error conflicts between the two attribute groups.

# In[ ]:


city_part[city_part['geoNetwork.city'].duplicated(keep=False) & (city_part['geoNetwork.city'] != 'N/A')]


# In[ ]:


city_part.shape


# In[ ]:


city_part.groupby(['geoNetwork.city', 'geoNetwork.region']).size().sort_values(ascending=False).head(10)


# If we combine the city and region field, we will be able to uniquely determine the identity of a city (as long as the city attribute is present). Therefore, we only have to fix N/A issues in the region field.

# In[ ]:


pairs = [('Colombo', 'Western Province'), ('Doha', 'Doha'),
         ('Guatemala City', 'Guatemala Department'), ('Hanoi', 'Hanoi'),
         ('Minsk', 'Minsk Region'), ('Nairobi', 'Nairobi County'), ('Tbilisi',
                                                                    'Tbilisi')]

for c, r in pairs:
    dat.loc[(dat['geoNetwork.city'] == c) &
            (dat['geoNetwork.region'] == 'N/A'), 'geoNetwork.region'] = r


# There are still many cases where the city is not given but the region is available. In these cases, we can still safely drop "metro", as in none of the cases, the metro attribute is present.

# In[ ]:


city_part[city_part['geoNetwork.city'] == 'N/A']


# We have now determined that the *country* attribute group can be determined from country alone, and the *city* attribute group can be determined by city + region. We have now shrunk the 6 geolocation attributes into 3.

# In[ ]:


selected = dat.loc[:, dat.columns.isin(pure_geo_columns)].copy()
cc = selected.groupby(['geoNetwork.city', 'geoNetwork.region','geoNetwork.country']).size().reset_index()

cc.loc[(cc['geoNetwork.city'].duplicated(keep=False) &
        (cc['geoNetwork.city'] != 'N/A'))
       | (cc['geoNetwork.region'].duplicated(keep=False) & (
           (cc['geoNetwork.city'] == 'N/A') &
           (cc['geoNetwork.region'] != 'N/A')))].head(30)


# We now see that almost all of the conflicts between (city, region) and country are due to errors. There are still some ambiguities, but since 1) we do not worry too match about per-record accuracy of the data 2) we cannot determine which subgroup of attributes we should trust more when a conflict happens, we might as well accept an okay solution and resolve all these conflicts by setting to the most common triples. We have to exclude cases where both *city* and *region* are NA because it does not make sense to infer anything about *country* from NAs.

# In[ ]:


most_common = dat.groupby([
    'geoNetwork.city', 'geoNetwork.region'
])['geoNetwork.country'].apply(lambda x: x.mode()).reset_index()
most_common.head()


# In[ ]:


for idx, row in most_common.iterrows():
    dat.loc[(dat['geoNetwork.city'] == row['geoNetwork.city']) &
            (dat['geoNetwork.region'] == row['geoNetwork.region']) &
            ((dat['geoNetwork.city'] != 'N/A') |
             ((dat['geoNetwork.region'] != 'N/A'))
             ), 'geoNetwork.country'] = row['geoNetwork.country']

selected = dat.loc[:, dat.columns.isin(pure_geo_columns)].copy()
cc = selected.groupby(
    ['geoNetwork.city', 'geoNetwork.region',
     'geoNetwork.country']).size().reset_index()
cc.head(30)


# We have resolved the vast majority of geolocation data conflicts!

# ## Method 2. Correction via Attribute Value Association

# Suppose the dataset size and anomaly number are even larger than what we have here, and a manual investigation of the data error is not feasible. We need some semi-reliable ways to fix errors in the geolocation data. Luckily, since geographical data are typically attributes describing fixed facts (cities do not randomly switch continental allegiance after all), we can depend on the typical relationships between attribute values and generalise it to determine what a reasonable attribute value combination should look like when suspected anomalies happen. We can learn the distribution of data values and exploit the learned distribution to identify and correct error. One way to do this is to use a classifier to predict one column value based on others, and find values in the target column that do not match up with the data value or have low prediction confidence. As most classifiers tend to learn the most likely value of the target given input variables, we can use the prediction of the classifier to identify unlikely values in the target column given the other columns. Here, we will be using a simple random forest. As we are using the random forest to model the data distribution instead of actually making predictions, accuracy is not really our biggest concern. In fact, we may want to artificially limit the learning capacity of the algorithm so unlikely edge cases that tend to be data anomalies will not be memorised.

# We have to convert the original data into one-hot format for the network to ingest. For comparison with method 1, we read a new data frame from the original. We still assign missing data a placeholder value.

# In[ ]:


dat2 = pd.read_pickle("../input/gstore-revenue-data-preprocessing/train.pkl")
for c in pure_geo_columns:
    dat2.loc[:, c] = dat2[c].cat.add_categories('N/A').fillna('N/A')
    
selected = dat2.loc[:, pure_geo_columns].copy()


# To simplify the caluclation, we assume we have performed some basic analysis like in method 1 and determined that *continent* and *subcontinent* are strictly determined by *country*, and *metro* can almost always be inferred from *city* and *region*. Therefore, we have only 3 attributes to work with.

# In[ ]:


key_attributes = ['geoNetwork.city', 'geoNetwork.region', 'geoNetwork.country']
selected[key_attributes].describe()


# Same as before, we have three columns of categorical data. We now convert them into one-hot encoding.

# In[ ]:


enc = OneHotEncoder()
transformed = enc.fit_transform(selected[key_attributes].apply(lambda x: x.cat.codes))
N, D = transformed.shape
N, D


# Here, we will be using the *country* column as the target and (*city*, *region*) as the input for the random forest.

# In[ ]:


clf = RandomForestClassifier()


# In[ ]:


clf.fit(transformed[:, :-222], selected['geoNetwork.country'].cat.codes)


# We set suspect anomaly cases as those whose predicted *country* value does not match up with the value in data:

# In[ ]:


pred = clf.predict(transformed[:, :-222])
is_anomaly = pred != selected['geoNetwork.country'].cat.codes


# We look at the first few columns of the detected anomaly cases. We exclude the cases with double NAs in the input columns because it is unrealistic to expect a reliable prediction in those cases. As we see here, the random forest does a pretty decent job at picking out obvious conflicting cases.

# In[ ]:


anomaly_cases = selected[is_anomaly][(selected['geoNetwork.city'] != 'N/A') | (
    selected['geoNetwork.region'] != 'N/A')][key_attributes]
anomaly_cases.head(30)


# We may also look at prediction confidences to find if there are cases where the random forest is uncertain which value is most probable:

# In[ ]:


certainty = clf.predict_proba(transformed[:, :-222])


# In[ ]:


uncertain_idx = np.max(certainty, axis=1) < 0.95
uncertain_cases = selected[key_attributes][uncertain_idx]
uncertain_cases.head()


# Obviously, when we don't know anything about the city, it is impossible for there to be any confidence in the prediction. So we will have to exclude the cases where both *city* and *region* are missing.

# In[ ]:


uncertain_cases[(uncertain_cases['geoNetwork.city'] != 'N/A')
                & (uncertain_cases['geoNetwork.region'] != 'N/A')].groupby(key_attributes).size().reset_index()


# As we see, there are not really any interesting cases where a 0.95 confidence thresholding tells us the algorithm might be making a mistake identifying anomalies. We may alter the confidence level, but the effect on the resulting detected anomalies is minimal.
# 
# Using a classifier for anomaly detection does not fix missing values in *region* as manual investigation did, but it is able to achieve nearly the same level of error fixing and allow us to reduce the chances of strange outliers throwing off further analysis. We now apply the error fixing results using the classifier method.

# In[ ]:


not_na_idx = (selected['geoNetwork.city'] !=
              'N/A') | (selected['geoNetwork.region'] != 'N/A')

target = pd.Categorical.from_codes(
    pred, categories=selected['geoNetwork.country'].cat.categories)

dat2.loc[not_na_idx, 'geoNetwork.country'] = target[not_na_idx]


# Finally, let us look at where the two methods yield different columns:

# In[ ]:


diff_idx = np.any(dat[key_attributes] != dat2[key_attributes], axis=1)
pd.concat([dat[key_attributes][diff_idx], dat2[key_attributes][diff_idx]], axis=1).head(10)


# We can show that the only type of difference between these two methods is the assigning of values to NAs in the *region* column we did in the manual method. If we exclude all cases where the original *region* value is missing, the results of both methods are identical.

# In[ ]:


diff_idx = np.any(dat[key_attributes] != dat2[key_attributes], axis=1) & (dat2['geoNetwork.region'] != 'N/A')
pd.concat([dat[key_attributes][diff_idx], dat2[key_attributes][diff_idx]], axis=1).head(10)


# We now export the fixing results of both methods.

# In[ ]:


for c in pure_geo_columns:
    dat.loc[:, c] = dat[c].cat.remove_categories('N/A')

dat.to_pickle("manual_geo_fix.pkl")

for c in pure_geo_columns:
    dat2.loc[:, c] = dat2[c].cat.remove_categories('N/A')

dat2.to_pickle("classifier_geo_fix.pkl")


# ## Afterthoughts

# It seems strange that so many obvious errors exist in the dataset. What if they are not really data errors? What if there is a reason for the inconsistencies? There are a few possibilities where such a situation would arise, for instance if there are two data sources, one from on-device GPS or registration location and one from ISP location. Let us see if we can find any clues from the domain names by looking at anomaly cases where the domain name field is not empty:

# In[ ]:


original = pd.read_pickle("../input/gstore-revenue-data-preprocessing/train.pkl")
original[is_anomaly & (~pd.isna(original['geoNetwork.networkDomain'])) &
         (~pd.isna(original['geoNetwork.city'])
          | ~pd.isna(original['geoNetwork.region']))][
              key_attributes + ['geoNetwork.networkDomain']].head(30)


# Clearly there is a close relationship between *country* and *networkDomain*. The domain names here seem to suggest that they are the ISP of the user/session connection. We might make an educated guess and deduce that the ISP domain is determined from the client IP address recorded for the session, and the *country* attribute is entirely dependent on this domain. But what about the *city* attribute? How is it determined then? Is it more precise than *country* or otherwise? We are not quite sure. One more thing we can do, is to see which one is more stable for a given user or session.

# In[ ]:


original[key_attributes + ['fullVisitorId']].dropna().groupby('fullVisitorId')[[
    'geoNetwork.city', 'geoNetwork.country'
]].nunique().mean()


# In[ ]:


original[key_attributes + ['sessionId']].dropna().groupby('sessionId')[[
    'geoNetwork.city', 'geoNetwork.country'
]].nunique().mean()


# As we see here, the *city* attribute seems to be slightly more stable for a given user or session. *Country* might even change half way through a session in some rare cases. We do not have much more evidence, but so far, there is slightly more reason to believe that the *city* attribute is a slightly more accurate attribute than *country*, and the fixes we did earlier seem reasonable.

# In[ ]:




