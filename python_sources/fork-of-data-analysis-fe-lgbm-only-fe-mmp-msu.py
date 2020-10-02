#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ### 0. Intro
# 
# It's illustration of FE logic:
# * delete meaningless and trash features
# * correct mistakes in dataset
# * construct new features
# * visualize them

# ### 1. Convert data to json format

# **Use this kernel https://www.kaggle.com/ravann/1-step-by-step-format-data-to-columnar-format/notebook
# **

# ### 2. Data exploring and filtering. Part 1

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
import pickle
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D
import gc

sns.set()
gc.enable()
get_ipython().run_line_magic('matplotlib', 'inline')


# **Load dataset**

# In[ ]:


df_train = pd.read_csv("../input/prepaired-data-of-customer-revenue-prediction/train_flat.csv", converters={'fullVisitorId': str})
df_test = pd.read_csv("../input/prepaired-data-of-customer-revenue-prediction/test_flat.csv", converters={'fullVisitorId': str})


# **View dataset**

# In[ ]:


df_train.head()


# In[ ]:


df_train.shape, df_test.shape


# **Fill missing values in totals_transactionRevenues by zeros**

# In[ ]:


df_train["totals_transactionRevenue"] = df_train["totals_transactionRevenue"].fillna(0)


# **Check whether columns in train and in test are equal**

# In[ ]:


train_col = np.array(df_train.columns)
test_col = np.array(df_test.columns)
print(set(train_col) - set(test_col))


# **Remove strange column trafficSource_campaignCode from test**

# In[ ]:


df_train = df_train.drop(columns=["trafficSource_campaignCode"])


# **View at "socialEngagementType"**

# In[ ]:


print(np.unique(df_train["socialEngagementType"], return_counts=True))
print(np.unique(df_test["socialEngagementType"], return_counts=True))


# **Remove this "usefull" feature from train and test**

# In[ ]:


df_train = df_train.drop(columns=["socialEngagementType"])
df_test = df_test.drop(columns=["socialEngagementType"])


# **Explore feature values in train**

# In[ ]:


df_train_eq_nan = df_train.fillna(-1543)


# In[ ]:


for col_name in np.array(df_train_eq_nan.columns):
    print(col_name)
    try:
        un, cnt = np.unique(np.array(df_train_eq_nan[col_name]), return_counts=True)
        print(un[:10], cnt[:10])
    except Exception:
        un, cnt = np.unique(np.array(df_train_eq_nan[col_name]).astype(str), return_counts=True)
        print(un[:10], cnt[:10])
    print("-" * 43)


# **After watching this values I aggreagated the following groups of featues**

# In[ ]:


numerical = ["visitNumber", "visitStartTime", "totals_bounces", "totals_hits", "totals_newVisits", "totals_pageviews", 
            "trafficSource_adwordsClickInfo.isVideoAd", "trafficSource_adwordsClickInfo.page", 
            "trafficSource_isTrueDirect", "device_isMobile"]

categorial = ["channelGrouping", "date", "device_browser", "device_deviceCategory", "device_operatingSystem",
               "geoNetwork_city", "geoNetwork_continent","geoNetwork_metro", "geoNetwork_country",
              "geoNetwork_networkDomain", "geoNetwork_region",
               "geoNetwork_subContinent", "trafficSource_adContent", "trafficSource_adwordsClickInfo.adNetworkType",
               "trafficSource_adwordsClickInfo.slot", "trafficSource_campaign", "trafficSource_keyword", "trafficSource_medium",
               "trafficSource_source"]

saved_trash = ["fullVisitorId"]

trash_trash = ["totals_visits", "device_browserSize", "device_browserVersion",
                "device_flashVersion", "device_language", "device_mobileDeviceBranding", "device_mobileDeviceInfo",
                "device_mobileDeviceMarketingName", "device_mobileDeviceModel", "device_mobileInputSelector",
                "device_screenColors", "device_screenResolution", "geoNetwork_cityId", "geoNetwork_latitude",
                "geoNetwork_longitude", "geoNetwork_networkLocation", "trafficSource_adwordsClickInfo.criteriaParameters",
                "device_operatingSystemVersion"]
                
wanted_to_trash = ["sessionId", "visitId", "trafficSource_adwordsClickInfo.gclId", "trafficSource_referralPath"]

in_future_wanted_to_trash = ["visitStartTime", "date", "device_isMobile", "geoNetwork_city", "geoNetwork_metro",
                "geoNetwork_networkDomain", "trafficSource_adContent", "trafficSource_keyword"]

to_heal = ["totals_bounces", "totals_newVisits", "totals_pageviews", "trafficSource_adwordsClickInfo.adNetworkType",
            "device_browser", "trafficSource_adwordsClickInfo.isVideoAd", "trafficSource_adwordsClickInfo.slot", 
            "trafficSource_adwordsClickInfo.page", "trafficSource_campaign", "trafficSource_isTrueDirect",
            "trafficSource_medium", "trafficSource_source", "device_operatingSystem", "geoNetwork_city",
            "geoNetwork_region",
            "geoNetwork_continent", "geoNetwork_country", "geoNetwork_metro", "geoNetwork_networkDomain",
            "geoNetwork_subContinent", "trafficSource_adContent", "trafficSource_keyword"]

answer_feature = ["totals_transactionRevenue"]


# * numerical -- _numerical features_
# * categorial -- _categorial features_
# * saved_trash -- _actually fullVisitorId_
# * trash_trash -- _unusefull features. Has a lot of nans or the same value for all objects_
# * wanted_to_trash -- _explore this featuers now but wants to delete them (e.g. all values are different for all objects)_
# * in_future_wanted_to_trash -- <i>features that can be dublicated by others (e.g. divice_isMobile may be duplicated by One-Hot-encoding of device_deviceCategory) or there is not any dependence between them and answer. But now don't remove them</i>
# * to_heal -- _fill nans by other value_
# * answer_feature -- _feature with answer_

# **Remove features from trash_trash**

# In[ ]:


df_train = df_train.drop(columns=trash_trash)
df_test = df_test.drop(columns=trash_trash)


# **Explore features from wanted_to_trash**

# In[ ]:


df_train_eq_nan = df_train.fillna(-1543)
for col_name in wanted_to_trash:
    print(col_name)
    un = None
    try:
        un = np.unique(np.array(df_train_eq_nan[col_name]), return_counts=True)
    except Exception:
        un = np.unique(np.array(df_train_eq_nan[col_name]).astype(str), return_counts=True)
    print(un)
    print("DIFFERENT COUNT: ", un[0].shape[0])
    print("-" * 43)


# **Seems that SessionId and visitId should be removed. Let see at them:**

# **Explore the objects with same SessionId**

# In[ ]:


un = np.unique(df_train["sessionId"], return_counts=True)
idx = np.where(un[1] != 1)[0]
repeat_session_id = un[0][idx]

repeated_df = df_train[df_train["sessionId"].isin(repeat_session_id)].sort_values(["sessionId"]).iloc[:, :20]
repeated_df


# **Let's see on features date and SessionId. Obviously if SessionIds are the same then fullVisitodIds are also the same. It seems that difference between dates for equal SessinId is 1 day. Perhaps the session was started at one day and ended at the next day, but there are two records about this one session.  So we can not only delete feature SessionId but also delete duplicated objects with the same SessionId (delete more earlier objects because there is much more information about later objects).**

# **Check if is that true that there are only two record for such "bad" equal SessionId:**

# In[ ]:


np.all(
    df_train[df_train["sessionId"].isin(repeat_session_id)][["date", "sessionId"]].groupby("sessionId").count() == 2)


# **So delete objects and SessionId:**

# In[ ]:


def parse_datetime(strdate):
    year, month, day = list(map(lambda x: int(x), [strdate[:4], strdate[4:6], strdate[6:8]]))
    return dt.datetime(year=year, month=month, day=day)


# In[ ]:


bad_idxs = []
for session_id in repeat_session_id:
    part_df = repeated_df[repeated_df["sessionId"] == session_id]
    dt1, dt2 = parse_datetime(str(part_df["date"].iloc[0])), parse_datetime(str(part_df["date"].iloc[1]))
    if dt1 < dt2:
        bad_idxs.append(part_df.iloc[0].name)
    else:
        bad_idxs.append(part_df.iloc[1].name)


# In[ ]:


len(bad_idxs)


# In[ ]:


df_train = df_train.drop(bad_idxs)
df_train.shape


# In[ ]:


df_train.index = np.arange(df_train.shape[0])


# In[ ]:


df_train_eq_nan = df_train.fillna(-1543)
for col_name in wanted_to_trash:
    print(col_name)
    un = None
    try:
        un = np.unique(np.array(df_train_eq_nan[col_name]), return_counts=True)
    except Exception:
        un = np.unique(np.array(df_train_eq_nan[col_name]).astype(str), return_counts=True)
    print(un)
    print("DIFFERENT COUNT: ", un[0].shape[0])
    print("-" * 43)


# In[ ]:


df_train = df_train.drop(columns=["sessionId"])
df_test = df_test.drop(columns=["sessionId"])


# **Let's see on VisitId:**

# In[ ]:


un = np.unique(df_train["visitId"], return_counts=True)
idx = np.where(un[1] > 1)[0]
repeat_session_id = un[0][idx]
repeated_df = df_train[df_train["visitId"].isin(repeat_session_id)].sort_values(["visitId"]).iloc[:, :20]
repeated_df


# In[ ]:


len(repeat_session_id)


# **We can see that records with same visitId has different fullVisitorId but the same visitStartTime. Because of simultaneous sessions data was received concurrently and this sessions has the same visidId. So we can't remove records with same visitId but can remove feature visitId**

# In[ ]:


df_train = df_train.drop(columns=["visitId"])
df_test = df_test.drop(columns=["visitId"])


# **Think about feature trafficSource_adwordsClickInfo.gclId: it has a lot of nans**

# In[ ]:


df_train_eq_nan = df_train.fillna(-1543)
un = np.unique(np.array(df_train_eq_nan["trafficSource_adwordsClickInfo.gclId"]).astype(str), return_counts=True)
idx = np.where(un[0] != "-1543")[0]
not_null_CI_train = un[0][idx]
not_null_CI_train, not_null_CI_train.shape


# In[ ]:


df_test_eq_nan = df_test.fillna(-1543)
un = np.unique(np.array(df_test_eq_nan["trafficSource_adwordsClickInfo.gclId"]).astype(str), return_counts=True)
idx = np.where(un[0] != "-1543")[0]
not_null_CI_test = un[0][idx]
not_null_CI_test, not_null_CI_test.shape


# In[ ]:


CI_intersect = set(not_null_CI_train) & set(not_null_CI_test)
len(CI_intersect)


# **There is a little intersect and it seems to be such king of information coding. I suppose we should remove this feature**

# In[ ]:


df_train = df_train.drop(columns=["trafficSource_adwordsClickInfo.gclId"])
df_test = df_test.drop(columns=["trafficSource_adwordsClickInfo.gclId"])


# **Think about feature trafficSource_referralPath:**

# In[ ]:


df_train_eq_nan = df_train.fillna(-1543)
un = np.unique(np.array(df_train_eq_nan["trafficSource_referralPath"]).astype(str), return_counts=True)
idx = np.where(un[0] != "-1543")[0]
not_null_RP_train = un[0][idx]
not_null_RP_train, not_null_RP_train.shape


# In[ ]:


df_test_eq_nan = df_test.fillna(-1543)
un = np.unique(np.array(df_test_eq_nan["trafficSource_referralPath"]).astype(str), return_counts=True)
idx = np.where(un[0] != "-1543")[0]
not_null_RP_test = un[0][idx]
not_null_RP_test, not_null_RP_test.shape


# In[ ]:


del df_train_eq_nan, df_test_eq_nan
gc.collect()


# **Let explore how many objects has revenue when trafficSource_referralPath isn't a nan**

# In[ ]:


revenues = df_train[df_train["trafficSource_referralPath"].isin(not_null_RP_train)]["totals_transactionRevenue"]
revenues[revenues != 0].shape


# In[ ]:


df_train[df_train["totals_transactionRevenue"] != 0].shape


# **So it seems that this feature is important**

# In[ ]:


categorial.append("trafficSource_referralPath")


# **Fill nans:**

# In[ ]:


zero_filling = ["totals_bounces", "totals_newVisits", "totals_pageviews", "trafficSource_adwordsClickInfo.isVideoAd",
                "trafficSource_adwordsClickInfo.page", "trafficSource_isTrueDirect"]
empty_filling = ["device_browser", "trafficSource_adwordsClickInfo.adNetworkType",
                 "trafficSource_adwordsClickInfo.slot", "geoNetwork_city", "geoNetwork_continent", 
                 "geoNetwork_country", "geoNetwork_metro", "geoNetwork_networkDomain", "geoNetwork_region",
                 "geoNetwork_subContinent", "trafficSource_adContent", "trafficSource_keyword",
                 "trafficSource_campaign", "trafficSource_medium", "trafficSource_source",
                 "device_operatingSystem", "trafficSource_referralPath"]


# In[ ]:


df_train[zero_filling] = df_train[zero_filling].fillna(0)
df_train["trafficSource_adwordsClickInfo.isVideoAd"] = df_train["trafficSource_adwordsClickInfo.isVideoAd"].apply(lambda x: 1 if x == 0 else 0)
df_train["trafficSource_isTrueDirect"] = df_train["trafficSource_isTrueDirect"].apply(lambda x: 1 if x else 0)


# In[ ]:


df_test[zero_filling] = df_test[zero_filling].fillna(0)
df_test["trafficSource_adwordsClickInfo.isVideoAd"] = df_test["trafficSource_adwordsClickInfo.isVideoAd"].apply(lambda x: 1 if x == 0 else 0)
df_test["trafficSource_isTrueDirect"] = df_test["trafficSource_isTrueDirect"].apply(lambda x: 1 if x else 0)


# In[ ]:


df_train[empty_filling] = df_train[empty_filling].fillna("@")
df_test[empty_filling] = df_test[empty_filling].fillna("@")


# In[ ]:


df_train["device_isMobile"] = df_train["device_isMobile"].apply(lambda x: 1 if x else 0)
df_test["device_isMobile"] = df_test["device_isMobile"].apply(lambda x: 1 if x else 0)


# In[ ]:


set(df_train.columns) - (set(numerical) | set(categorial))


# In[ ]:


len(numerical) + len(categorial), df_train.shape


# **Save result**

# In[ ]:


#df_train.to_csv("./kernel/data/train_filtered.csv", sep=",", index=False)
#df_test.to_csv("./kernel/data/test_filtered.csv", sep=",", index=False)


# In[ ]:


df_train = pd.read_csv("../input/prepaired-data-of-customer-revenue-prediction/train_filtered.csv", converters={'fullVisitorId': str}, sep=",")
df_test = pd.read_csv("../input/prepaired-data-of-customer-revenue-prediction/test_filtered.csv", converters={'fullVisitorId': str}, sep=",")


# ### 3. Data exploring and filtering. Part 2

# **Add some new OHE features from categorial features**

# In[ ]:


for col_name in categorial:
    print(col_name)
    un_train = np.unique(np.array(df_train[col_name]).astype(str), return_counts=True)
    un_test = np.unique(np.array(df_test[col_name]).astype(str), return_counts=True)
    print("TRAIN: ", un_train[0][:10])
    print("TEST: ", un_test[0][:10])
    print("DIFFERENT TRAIN COUNT: ", un_train[0].shape[0])
    print("DIFFERENT TEST COUNT: ", un_test[0].shape[0])
    print("-" * 43)


# **We can select the following groups of categorial features:**

# In[ ]:


easy_OHE = ["channelGrouping", "device_deviceCategory", "geoNetwork_continent", "geoNetwork_subContinent", "trafficSource_medium"]
easy_OHE_but_prepare = ["trafficSource_adwordsClickInfo.adNetworkType", "trafficSource_adwordsClickInfo.slot"]

bad_categorial = ["device_browser", "date", "device_operatingSystem", "geoNetwork_city", "geoNetwork_metro", "geoNetwork_country",
                  "geoNetwork_networkDomain", "geoNetwork_region", "trafficSource_adContent",
                  "trafficSource_campaign", "trafficSource_keyword", "trafficSource_source",
                 "trafficSource_referralPath"]


# * easy_OHE -- _easy to do One-Hot-Encoding_
# * easy_OHE_but_prepare -- _easy to do One-Hot-Encoding, but feature values in train and in test are not the same_
# * bad_categorial -- _other categorial features_

# In[ ]:


from sklearn.preprocessing import OneHotEncoder as OHE
from sklearn.preprocessing import LabelEncoder as OE


# **Assumption: since we will encode the device_deviceCategory, the isMobile feature can be thrown out, as it will be a duplicate. So we will check it**

# In[ ]:


y_train = np.array(df_train[answer_feature])
id_numeration_train = np.array(df_train[saved_trash])
X_train_numerical = np.array(df_train[numerical])

id_numeration_test = np.array(df_test[saved_trash])
X_test_numerical = np.array(df_test[numerical])


# In[ ]:


train_easy_OHE = np.array(df_train[easy_OHE])
test_easy_OHE = np.array(df_test[easy_OHE])

easy_OEs = [OE() for i in range(train_easy_OHE.shape[1])]
for f in range(train_easy_OHE.shape[1]):
    easy_OEs[f].fit(train_easy_OHE[:, f])
    train_easy_OHE[:, f] = easy_OEs[f].transform(train_easy_OHE[:, f])
    test_easy_OHE[:, f] = easy_OEs[f].transform(test_easy_OHE[:, f])


# In[ ]:


easy_enc = OHE(sparse=False)
easy_enc.fit(train_easy_OHE)
train_easy_OHE_conv = easy_enc.transform(train_easy_OHE)
test_easy_OHE_conv = easy_enc.transform(test_easy_OHE)


# **Suppose that isMobile means that device_deviceCategory equals "mobile":**

# In[ ]:


errors = np.where(
    np.array(train_easy_OHE_conv[:, 9]).astype(bool)
    != np.array(df_train["device_isMobile"]))[0]
errors.shape


# **No, it's not a truth. Suppose that isMobile means that device_deviceCategory equals "mobile" or "tablet":**

# In[ ]:


errors = np.where(
    ((np.array(train_easy_OHE_conv[:, 10]).astype(bool)) | (np.array(train_easy_OHE_conv[:, 9]).astype(bool)))
    != np.array(df_train["device_isMobile"]))[0]
errors.shape


# **It's truth, but there is some errors that are needed to be fixed**

# In[ ]:


display(df_train.iloc[errors, 10:])


# In[ ]:


deviceCategory_idx = 11
is_mobile_idx = 12


# **Apparently, there are errors in deviceCategory rather than in isMobie. Moreover, in erroneous examples in favor of the error says feature OS. Let's make this correction: where isMobile in erroneous examples = 0 -- set desktop, otherwise -- mobile.**

# In[ ]:


df_train.iloc[errors, deviceCategory_idx] = (
    df_train.iloc[errors, is_mobile_idx].apply(lambda x: "desktop" if x == 0 else "mobile"))


# In[ ]:


errors_te = np.where(
    ((np.array(test_easy_OHE_conv[:, 10]).astype(bool)) | (np.array(test_easy_OHE_conv[:, 9]).astype(bool)))
    != np.array(df_test["device_isMobile"]))[0]
errors_te.shape


# In[ ]:


df_test.iloc[errors_te, deviceCategory_idx - 1] = (
    df_test.iloc[errors_te, is_mobile_idx - 1].apply(lambda x: "desktop" if x == 0 else "mobile"))


# In[ ]:


train_easy_OHE = np.array(df_train[easy_OHE])
test_easy_OHE = np.array(df_test[easy_OHE])
easy_OEs = [OE() for i in range(train_easy_OHE.shape[1])]
for f in range(train_easy_OHE.shape[1]):
    easy_OEs[f].fit(train_easy_OHE[:, f])
    train_easy_OHE[:, f] = easy_OEs[f].transform(train_easy_OHE[:, f])
    test_easy_OHE[:, f] = easy_OEs[f].transform(test_easy_OHE[:, f])
easy_enc = OHE(sparse=False)
easy_enc.fit(train_easy_OHE)
train_easy_OHE_conv = easy_enc.transform(train_easy_OHE)
test_easy_OHE_conv = easy_enc.transform(test_easy_OHE)


# In[ ]:


errors2 = np.where(
    ((np.array(train_easy_OHE_conv[:, 10]).astype(bool)) | (np.array(train_easy_OHE_conv[:, 9]).astype(bool)))
    != np.array(df_train["device_isMobile"]))[0]
print(errors2.shape)
errors_te2 = np.where(
    ((np.array(test_easy_OHE_conv[:, 10]).astype(bool)) | (np.array(test_easy_OHE_conv[:, 9]).astype(bool)))
    != np.array(df_test["device_isMobile"]))[0]
print(errors_te2.shape)


# **There is no errors!**

# In[ ]:


easy_OHE_names = ['channelGrouping_(Other)',
 'channelGrouping_Affiliates',
 'channelGrouping_Direct',
 'channelGrouping_Display',
 'channelGrouping_Organic Search',
 'channelGrouping_Paid Search',
 'channelGrouping_Referral',
 'channelGrouping_Social',
 'device_deviceCategory_desktop',
 'device_deviceCategory_mobile',
 'device_deviceCategory_tablet',
 'geoNetwork_continent_@',
 'geoNetwork_continent_Africa',
 'geoNetwork_continent_Americas',
 'geoNetwork_continent_Asia',
 'geoNetwork_continent_Europe',
 'geoNetwork_continent_Oceania',
 'geoNetwork_subContinent_@',
 'geoNetwork_subContinent_Australasia',
 'geoNetwork_subContinent_Caribbean',
 'geoNetwork_subContinent_Central America',
 'geoNetwork_subContinent_Central Asia',
 'geoNetwork_subContinent_Eastern Africa',
 'geoNetwork_subContinent_Eastern Asia',
 'geoNetwork_subContinent_Eastern Europe',
 'geoNetwork_subContinent_Melanesia',
 'geoNetwork_subContinent_Micronesian Region',
 'geoNetwork_subContinent_Middle Africa',
 'geoNetwork_subContinent_Northern Africa',
 'geoNetwork_subContinent_Northern America',
 'geoNetwork_subContinent_Northern Europe',
 'geoNetwork_subContinent_Polynesia',
 'geoNetwork_subContinent_South America',
 'geoNetwork_subContinent_Southeast Asia',
 'geoNetwork_subContinent_Southern Africa',
 'geoNetwork_subContinent_Southern Asia',
 'geoNetwork_subContinent_Southern Europe',
 'geoNetwork_subContinent_Western Africa',
 'geoNetwork_subContinent_Western Asia',
 'geoNetwork_subContinent_Western Europe',
 'trafficSource_medium_@',
 'trafficSource_medium_affiliate',
 'trafficSource_medium_cpc',
 'trafficSource_medium_cpm',
 'trafficSource_medium_organic',
 'trafficSource_medium_referral']


# **Let's encode easy_OHE_but_prepare:**

# In[ ]:


df_test["trafficSource_adwordsClickInfo.adNetworkType"] = (
    df_test["trafficSource_adwordsClickInfo.adNetworkType"].apply(lambda x: "@" if x == 'Content' else x))
df_test["trafficSource_adwordsClickInfo.slot"] = (
    df_test["trafficSource_adwordsClickInfo.slot"].apply(lambda x: "@" if x == 'Google Display Network' else x))


# In[ ]:


for col_name in easy_OHE_but_prepare:
    print(col_name)
    un_train = np.unique(np.array(df_train[col_name]).astype(str), return_counts=True)
    un_test = np.unique(np.array(df_test[col_name]).astype(str), return_counts=True)
    print("TRAIN: ", un_train[0][:10])
    print("TEST: ", un_test[0][:10])
    print("DIFFERENT TRAIN COUNT: ", un_train[0].shape[0])
    print("DIFFERENT TEST COUNT: ", un_test[0].shape[0])
    print("-" * 43)


# In[ ]:


train_easy_OHE_prep = np.array(df_train[easy_OHE_but_prepare])
test_easy_OHE_prep = np.array(df_test[easy_OHE_but_prepare])
easy_OEs_prep = [OE() for i in range(train_easy_OHE_prep.shape[1])]
for f in range(train_easy_OHE_prep.shape[1]):
    easy_OEs_prep[f].fit(train_easy_OHE_prep[:, f])
    train_easy_OHE_prep[:, f] = easy_OEs_prep[f].transform(train_easy_OHE_prep[:, f])
    test_easy_OHE_prep[:, f] = easy_OEs_prep[f].transform(test_easy_OHE_prep[:, f])


# In[ ]:


easy_enc_prep = OHE(sparse=False)
easy_enc_prep.fit(train_easy_OHE_prep)
train_easy_OHE_prep_conv = easy_enc_prep.transform(train_easy_OHE_prep)
test_easy_OHE_prep_conv = easy_enc_prep.transform(test_easy_OHE_prep)


# In[ ]:


easy_OHE_prep_names = ['trafficSource_adwordsClickInfo.adNetworkType_@',
 'trafficSource_adwordsClickInfo.adNetworkType_Google Search',
 'trafficSource_adwordsClickInfo.adNetworkType_Search partners',
 'trafficSource_adwordsClickInfo.slot_@',
 'trafficSource_adwordsClickInfo.slot_RHS',
 'trafficSource_adwordsClickInfo.slot_Top']


# **Let's factorize all bad_categorial features**

# In[ ]:


train_bad_cat = df_train[bad_categorial].copy()
test_bad_cat = df_test[bad_categorial].copy()
for f in bad_categorial:
    train_bad_cat[f], indexer = pd.factorize(train_bad_cat[f])
    test_bad_cat[f] = indexer.get_indexer(test_bad_cat[f])


# In[ ]:


y_train_clf = (y_train.ravel() > 0).astype(int)


# **But some of values of features let's convert via OHE. Explore how feature values are distributed for record with some revenue and with no revenue:**

# In[ ]:


with_rev = df_train.iloc[y_train_clf == True, :][bad_categorial]
no_rev = df_train.iloc[y_train_clf == False, :][bad_categorial]


# In[ ]:


for feature in bad_categorial:
    vals = with_rev[feature]
    un, cnt = np.unique(vals, return_counts=True)
    data = np.hstack((un.reshape((-1, 1)), cnt.reshape((-1, 1))))

    top = 20
    df = pd.DataFrame(data, columns=["feature", "count features"])
    plt.figure(figsize=(15, 8))
    sns.barplot(x="count features", y='feature', data=df.sort_values("count features", ascending=False).iloc[:top])
    plt.title("FEATURE {} HAVE REVENUE".format(feature), fontsize=18)
    plt.show()
    
    vals = no_rev[feature]
    un, cnt = np.unique(vals, return_counts=True)
    data = np.hstack((un.reshape((-1, 1)), cnt.reshape((-1, 1))))
    top = 20
    df = pd.DataFrame(data, columns=["feature", "count features"])
    plt.figure(figsize=(15, 8))
    sns.barplot(x="count features", y='feature', data=df.sort_values("count features", ascending=False).iloc[:top])
    plt.title("FEATURE {} HAVEN'T REVENUE".format(feature), fontsize=18)
    plt.show()


# **Let's make a top of values for each feature:**

# In[ ]:


top_rev =    [5, 0, 6, 20, 6, 4, 7, 10, 2, 5, 5, 16, 5]
top_no_rev = [11, 0, 6, 10, 5, 10, 7, 4, 3, 4, 5, 10, 7] 


# In[ ]:


bad_cat_to_OHE = defaultdict(set)

for feature, t_r, t_n_r in zip(bad_categorial, top_rev, top_no_rev):
    vals = with_rev[feature]
    un, cnt = np.unique(vals, return_counts=True)
    sortidx = np.argsort(cnt)
    bad_cat_to_OHE[feature] |= set(un[sortidx][::-1][:t_r])
    
    vals = no_rev[feature]
    un, cnt = np.unique(vals, return_counts=True)
    sortidx = np.argsort(cnt)
    bad_cat_to_OHE[feature] |= set(un[sortidx][::-1][:t_n_r])
    if "@" in bad_cat_to_OHE[feature]:
        bad_cat_to_OHE[feature].remove("@")
        
    test_vals = set(df_test[feature])
    bad_cat_to_OHE[feature] &= test_vals


# In[ ]:


ordered_dict = dict()
for (name, value) in bad_cat_to_OHE.items():
    ordered_dict[name] = list(value)
    
bad_cat_features_OHE_names = []
for (name, value) in ordered_dict.items():
    if len(value) > 0:
        for val in value:
            bad_cat_features_OHE_names.append("{}_{}".format(name, val))


# In[ ]:


add_len = len(bad_cat_features_OHE_names)
add_len


# In[ ]:


train_bad_cat_to_OHE = np.zeros((df_train.shape[0], add_len))
test_bad_cat_to_OHE = np.zeros((df_test.shape[0], add_len))
for num, feature_value in enumerate(bad_cat_features_OHE_names):
    idx = feature_value.rfind("_")
    feature, value = feature_value[:idx], feature_value[idx + 1:]
    train_bad_cat_to_OHE[:, num] = np.array(df_train[feature] == value).astype(int)
    test_bad_cat_to_OHE[:, num] = np.array(df_test[feature] == value).astype(int)


# **Add to categorial features feature weekday:**

# In[ ]:


bad_categorial += ["weekday"]


# In[ ]:


tr_wd = np.array(pd.to_datetime(df_train["date"], format="%Y%m%d").apply(lambda x: x.weekday())).reshape((-1, 1))
te_wd = np.array(pd.to_datetime(df_test["date"], format="%Y%m%d").apply(lambda x: x.weekday())).reshape((-1, 1))


# **Collect all together:**

# In[ ]:


X_train = np.hstack((X_train_numerical, train_easy_OHE_conv,
                      train_easy_OHE_prep_conv, train_bad_cat_to_OHE, train_bad_cat, tr_wd))
X_test = np.hstack((X_test_numerical, test_easy_OHE_conv,
                     test_easy_OHE_prep_conv, test_bad_cat_to_OHE, test_bad_cat, te_wd))


# In[ ]:


feature_names = numerical + easy_OHE_names + easy_OHE_prep_names + bad_cat_features_OHE_names + bad_categorial
len(feature_names), X_train.shape


# **Let's visualize train and test:**

# In[ ]:


from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import scipy
import pickle
import re
import time
from sklearn.utils import shuffle as skshuffle


# In[ ]:


def visualize_data(data_train, y_train, data_test, amount_train=None, amount_test=None,
                   scale=True, threeD=False, shuffle=False, alpha=1.0):
    
    def visualize_2d(data_train, y_train, data_test):
        tsvd = PCA(n_components=2)
        tsvd.fit(data_train)
        Z_train = tsvd.fit_transform(data_train)
        Z_test = tsvd.transform(data_test)

        classes_amount = np.unique(y_train).shape[0]
        cm = plt.get_cmap('jet')
        plt.figure(figsize=(18, 15))
        plt.scatter(Z_train[:, 0], Z_train[:, 1],
                    c=y_train.ravel(), cmap='RdYlGn_r', alpha=alpha) # green - 1 blue 0
        plt.title("Train data")
        plt.figure(figsize=(18, 15))
        plt.scatter(Z_test[:, 0], Z_test[:, 1],
                    c=["plum"] * Z_test.shape[0], alpha=alpha)
        plt.title("Test data")
        plt.show()
        
    def visualize_3d(data_train, y_train, data_test):
        tsvd = PCA(n_components=3)
        tsvd.fit(data_train)
        Z_train = tsvd.fit_transform(data_train)
        Z_test = tsvd.transform(data_test)

        classes_amount = np.unique(y_train).shape[0]
        cm = plt.get_cmap('jet')
        
        fig = plt.figure(figsize=(18, 15))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(Z_train[:, 0], Z_train[:, 1], Z_train[:, 2],
                c=y_train.ravel(), cmap='RdYlGn_r', alpha=alpha)
        plt.title("Train data")
        fig = plt.figure(figsize=(18, 15))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(Z_test[:, 0], Z_test[:, 1], Z_test[:, 2],
                c=["plum"] * Z_test.shape[0], alpha=alpha)
        plt.title("Test data")
        plt.show()
        
    if amount_train is None:
        amount_train = data_train.shape[0]
    else:
        amount_train = int(amount_train * data_train.shape[0])
        
    if amount_test is None:
        amount_test = data_test.shape[0]
    else:
        amount_test = int(amount_test * data_test.shape[0])
        
    if shuffle:
        data_train, y_train = skshuffle(data_train, y_train)
        data_test = skshuffle(data_test)
    
    if scale:
        scaler = StandardScaler(copy=True)
        scaler.fit(data_train)
        X_train = scaler.transform(data_train)
        X_test = scaler.transform(data_test)
    else:
        X_train = data_train
        X_test = data_test
        
    start = time.time()
    if threeD:
        visualize_3d(X_train[:amount_train, :], y_train[:amount_train], X_test[:amount_train, :])
    else:
        visualize_2d(X_train[:amount_train, :], y_train[:amount_train], X_test[:amount_train, :])
    print(time.time() - start)


# In[ ]:


y_train_clf = np.copy(y_train)
y_train_clf[y_train_clf > 0] = 1
y_train_clf[y_train_clf == 0] = 0


# In[ ]:


visualize_data(X_train, y_train_clf, X_test, amount_train=0.2,
               amount_test=0.2, threeD=False, alpha=0.5, scale=True, shuffle=True)


# In[ ]:


visualize_data(X_train, y_train_clf, X_test, amount_train=0.2,
               amount_test=0.2, threeD=True, alpha=0.5, scale=True, shuffle=True)


# **Red points -- records with some revenue**
