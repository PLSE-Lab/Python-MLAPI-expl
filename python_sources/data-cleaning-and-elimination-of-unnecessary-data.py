#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import seaborn as sns
import matplotlib.pyplot as plt


# **Introduction**   
# 
# In the dataset we have  two files **train.csv** and **test.csv** which are having 903653 and 804684 rows respectively.  The amount of test data is almost equal to the train data which mean that while testing the model it'll test wider range of scenarios and it'll check the accuracy of the model throughly.
# 
# Before getting started with the code it's good to get familiar with the data that we have here. Dataset contains these 12 columns which give information about the visitors , their mode of access, number of visits and more statistical detail about their location and visits.
# 
# Each field of dataset is described below :
# 
# * fullVisitorId- A unique id assigend to each customer.
# * channelGrouping - Channel through which the user accessed store.
# * date - The date on which the user visited the store.
# * device - Device information which was used to access store.
# * geoNetwork - It has information about the location of the visitor.
# * sessionId - An id assigned to the session created for user visit to the store. It is actually a combination of        fullVisitorId and visitId.
# * socialEngagementType - Tells the type of social enagement by the customer.
# * totals - This section contains aggregate values across the session.
# * trafficSource - Detail abou the source from where the traffic is coming.
# * visitId - An id for a specific visit each time customer is visiting the store.
# * visitNumber - Number of visit customer is having to the store.
# * visitStartTime - Timestamp for the visiting time.
# 
# From the fields mentioned above  'device' , 'geoNetwork' , 'totals'  and trafficSource have data in the json format which we can take and add to the data frame as separate columns. Next we'll see how we can flatten these fields to get meaningful data from them in the form of columns.

# In[ ]:


def load_df(csv_path):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}
                     )
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df


# In the **load_df()** i'll explain the steps one by one so you can understand what is happening inside.
# 
# We've four fields as mentioned earlier which are having data in the form of json. So we've defined those fields which we'll handle one by one inside the loop.
# 
# First of all we'll load the csv file into the data frame. While reading the csv we're using converters field to pass through the column value to **json.loads()** which will generate json object for each value in that column.We're doing it for all the columns having json data. Another parameter is **dtype** which is used to define the data type for the columns if we want to change any but we're specifying it because the overview of the data specifies that **fullVisitorId** should be string.
# 
# After the csv is loaded, we'll fetch the values of each column and create a data frame from those json values with the help fo **json_normalize** which gives us a dataframe for the values in that column. Then we're defining the column names for the new data frame that we got from the json values. Next step is to drop the colum in existing data frame for which we're creating new columns and merge both the data frame .**right_index** and **left_index** is true to merge it based on the indexes.

# In[ ]:


train_df = load_df('../input/train.csv')
test_df = load_df("../input/test.csv")

train_df.head()


# Now you can see after we've flattened the json fields available in our dataset and there are new columns which we don't know about yet so let's take a look at those columns. 
# 
# The newly available information in our dataset are
# 
# * device        
#      * device.browser
#      * device.browserSize
#      * device.broserVersion
#      * device.deviceCategory
#      * device.flashVersion
#      * device.isMobile
#      * device.language
#      * device.mobileDeviceBranding
#      * device.mobileDeviceInfo
#      * device.mobileDeviceMarketingName
#      * device.mobileDeviceModel
#      * device.mobileDeviceInputSelector
#      * device.operatingSystem
#      * device.operatingSystemVersion
#      * device.screenColor
#      * deivce.screenResolution
# * geoNetwork     
#      * geoNetwork.city
#      * geoNetwork.cityId
#      * geoNetwork.continent
#      * geoNetwork.country
#      * geoNetwork.longitude
#      * geoNetwork.latitude
#      * geoNetwork.metro
#      * geoNetwork.networkDomain
#      * geoNetwork.networkLocation
#      * geoNetwork.region
#      * geoNetwork.subContinent
# * totals    
#      * totals.bounces
#      * totals.hits
#      * totals.newVisits
#      * totals.pageviews
#      * totals.transactionRevenue
#      * totals.visits
# * trafficSource    
#      * trafficSource.adContent
#      * trafficSource.adwordsClickInfo.adNetworkType
#      * trafficSource.adwordsClickInfo.criteriaParameters
#      * trafficSource.adwordsClickInfo.gclId
#      * trafficSource.adwordsClickInfo.isVideoAd
#      * trafficSource.adwordsClickInfo.page
#      * trafficSource.adwordsClickInfo.slot
#      * trafficSource.adwordsClickInfo.campaign
#      * trafficSource.adwordsClickInfo.campaignCode
#      * trafficSource.adwordsClickInfo.isTrueDirect
#      * trafficSource.adwordsClickInfo.keyword
#      * trafficSource.adwordsClickInfo.medium
#      * trafficSource.adwordsClickInfo.referralPath
#      * trafficSource.adwordsClickInfo.source
#         
#         
#    If we'll look at the above outut then we'll notice that a lot of the columns which are added newly in the dataset are        having null values. Not specifically 'null' but values which doesn't contain any useful data so it'll be better to have a      look at such values before going further.

# In[ ]:


def check_null_values(data_frame,dataset_name):
    null_count = ((data_frame[data_frame.columns] == 'not available in demo dataset') | (data_frame[data_frame.columns].isna())).sum()
    
    #print("\n--------------------{} Dataset Null Count-----------------------".format(dataset_name))
    #print(null_count)

    keys = null_count.keys()
    df_size = len(data_frame) 
    null_col = [col for col in keys if null_count[str(col)] > 0 if null_count[str(col)] == df_size]
    print("No. of columns with null values in {} dataset : {}".format(dataset_name,len(null_col)))
    return null_col,null_count
    
train_null_col,train_null_count = check_null_values(train_df,"Training")
test_null_col,test_null_count = check_null_values(test_df,"Testing")


# After that when i checked the values of all the columns whether those have data or not then i found that few columns are not giving any information. So i checked the count of such columns above and i found out that few columns are not giving any information and are of no use. Simillarly few columns are not having information for some of the rows but for other rows they have information. We've checked for both values which are not useful as well as values which are na(not available).
# 
# We got count of columns which are not having any useful information, count is 17. Since these columns are not useful for getting the desired results so it'll not be a wise decision to keep  it in the dataset as it'll only increase the processing time. In the next step we'll remove these columns and proceed towards processing the data. 
#     

# In[ ]:


col_labels = ['Column Name','Null Count']
null_count_df = pd.DataFrame(columns=col_labels,data={col_labels[0] : list(train_null_count.keys()),
                                                     col_labels[1] : list(train_null_count.values)})
fig,ax = plt.subplots(figsize=(20,8))
null_graph = sns.barplot(x=col_labels[0],y=col_labels[1],data=null_count_df,palette=sns.color_palette("Paired",len(null_count_df)),ax=ax,capsize=0.5)
null_graph.set_xticklabels(labels=[label.split(".")[-1] for label in null_count_df['Column Name']],rotation=60)

print("\nColumns having unnecessary information from training set \n" ,train_null_col)
print("\nColumns having unnecessary information from testing set \n" ,test_null_col)


# Actually earlier we were getting count of null values as a list but that didn't seem interesting and it was dull so i thought to bring it in a more pleasant manner so that we can have a better understanding of these values. These are the columns which we're going to  drop from the dataset. In the bar chart above you'll see a lot of bars which are shooting up high and all those are the columns which have no information that we can use for our analysis. From the chart it'll appear that more than half of the columns are having null value but it's not so. Actually the number of columns are too much to adjust within the screen so it got little bit messy. Sorry for that! I've put only last pat of the column names which are derived from json so that you can clearly understand them. We already have these column names in separate lists which we'll use to remove from the dataset.

# In[ ]:


train_df.drop(columns=train_null_col,inplace=True)
test_df.drop(columns=test_null_col,inplace=True)
train_df.drop(columns=['trafficSource.campaignCode','visitStartTime'],inplace=True)
test_df.drop(columns=['visitStartTime'],inplace=True)

print("Training Set Shape after dropping columns",train_df.shape)
print("Testing Set Shape after dropping columns",test_df.shape)


# We dropped the columns from the dataset and along with that I've removed  **trafficSource.campaignCode** . The reason for dropping campaignCode you'll see by looking at the count of null values in this column for training dataset then it's 903652 which is almost the size of dataset.
# 
# Now you'll see that number of rows for both training and testing set is almost equal which means that both set doesn't have any extra columns which can affect the process. Even though we're done removing extra unnecessary data from the dataset but we still have some null values which has to be handled because all the machine learning algorithms doesn't work on null values. We'll look on the columns which are still having some values null but not all.
# 
# So before this we had handled the columns which were not at all useful for us. But even now we're not done with cleaning the data. There are still some columns which are having nulll values and we'll start managing them now..
# 

# In[ ]:


def handle_null(df,name) :
    df['totals.newVisits'].fillna(0,inplace=True)
    df['totals.pageviews'].fillna(0,inplace=True)
    
    null_col,null_count = check_null_values(df,name)
    df_len = len(df)
    
    null_percent = [round((null_count_val/df_len)* 100,2) for null_count_val in null_count.values]
    null_percent = pd.DataFrame(columns=['Column Name','Null Percent'],data={ 'Column Name' : df.keys(),'Null Percent' : null_percent})
    null_percent.set_index('Column Name',inplace=True)
    return null_percent

train_df['totals.transactionRevenue'].fillna(0,inplace=True)
train_col_null_percent = handle_null(train_df,"Training")
test_col_null_percent = handle_null(test_df,"Testing")

display(train_col_null_percent)


# First of all it'll be better to handle **totals.transactionRevenue** column because that's the column which we'll use to predict in our analysis. If that column will have any null values then training won't be accurate in predicting results. So for that i've replaced all the null values in the column to 0 because those visits by the customer didn't resulted in any purchase by them which means that total revenue from the transaction is 0.
# 
# Now we'll move towards handling other columns which were having some null values and some information. When we start to look at the null value count which we calculated earlier then we'll see that many columns are still having a lot of null values or values which won't be useful for us so for those columns we'll put a threshold and discard them. I could see only 2 columns which we can use by filling some values in place of NaN. These two columns are **totals.newVisits** and **totals.pageviews**.
# 
# **totals.newVisits** column specify the new visits if any by the customer and it had around **77.8%** values available under the column and only **22.2%** were missing which we thought to replace  with 0 . It has only value '1' so we thought the other value 0 to be replaced wherever we're having NaN. Simillarly for **totals.pageViews** which was missing only 100 values which also we replaced with 0 as it's almost negligible value which won't have much affect on our analysis. These two columns we've handled now. 
# 
# Basically while discarding columns which we should consider a threshold percentage. If any column has percentage of null values more than the threshold then we should discard that column . I've represented the number of null values in all the columns in a separate dataframe so that you can get a better understanding of them. As you can see the column are having null percentage of more than 50%. 

# In[ ]:


train_cols_to_discard = [column for column in train_col_null_percent.index.values if int(train_col_null_percent.at[column,'Null Percent']) > 22]
train_cols_to_discard.extend(['fullVisitorId','totals.visits','socialEngagementType','sessionId','visitId','geoNetwork.networkDomain','trafficSource.campaign'])
#print(train_cols_to_discard)
train_df.drop(train_cols_to_discard,axis=1,inplace=True)

test_cols_to_discard = [column for column in test_col_null_percent.index.values if int(test_col_null_percent.at[column,'Null Percent']) > 22]
test_cols_to_discard.extend(['socialEngagementType','totals.visits','sessionId','visitId','geoNetwork.networkDomain','trafficSource.campaign'])
test_df.drop(test_cols_to_discard,axis=1,inplace=True)

train_df.head()


# Generally this threshold is conisdered to be 20% but we're considering it to be **22%** in our case. Since we've one columns i.e. totals.newVisits which is having nearly 22% null values. Here we'll drop the columns which we're going more than our threshold level. 
# 
# While i was observing the data frame, it just came to my mind that we still have 2 columns which are having unique values and their presence won't have much help in analyzing the customer behavior because they're not involved in defining it. These are just unique numbers for identifying session and enagement so i dropped them as well. The question that might be coming to your mind will be that we still have **fullVisitorId** which is another unique id. The reason why i kept it in the dataset is because it's not unique for each row and number of times a visitor comes again which can somewhat tell about customers behavior.
# 
# To get rid of unnecessary data we created a list of columns which are having percentage of null values more than our threshold value and we've used this list to drop columns from training as well as test dataset. So now after all the cleaning of data we're getting 22 columns which are not having columns with substantial null values in it. Now we're in such a stage that we can start proceeding towards analyzing  data because we've data fields which are having useful data.
# 
# **I'll update my analysis here very soon with some graphs and more insight into the data. Thankyou for going through my analysis and i hope that you would've got something useful out of analysis. Please suggest any changes or give feedback if you have any!! Thanks [Aaron](https://www.kaggle.com/stajh05) for your valuable feedback.**

# In[ ]:


train_quarter = [int((((date%10000)/100)/4)+1) for date in train_df['date']]
test_quarter = [int((((date%10000)/100)/4)+1) for date in test_df['date']]

def getDay(date):
    return date%100

train_df = train_df.assign(quarterOfYear=train_quarter)
test_df = test_df.assign(quarterOfYear=test_quarter)

train_df['date'] = train_df['date'].apply(getDay)
test_df['date'] = test_df['date'].apply(getDay)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
col_to_encode = ['channelGrouping','device.browser','device.deviceCategory','device.isMobile','device.operatingSystem','geoNetwork.continent','geoNetwork.country','geoNetwork.subContinent', 'trafficSource.medium', 'trafficSource.source']
le = LabelEncoder()

def encode_cols(data_frame,columns):
    for column in col_to_encode :
        data_frame[column] = le.fit(data_frame[column]).transform(data_frame[column]) 

encode_cols(train_df,col_to_encode)
encode_cols(test_df,col_to_encode)
        
training_col = list(train_df.keys())
training_col.remove('totals.transactionRevenue')
full_visitor_ids = test_df['fullVisitorId'].values

test_col = list(test_df.keys())
test_col.remove('fullVisitorId')


X_train = train_df[training_col] 
y_train = train_df['totals.transactionRevenue']


#print(type(train_df['device.browser'][0]))
#print([col_val for col_val in train_df.columns.values if isinstance(train_df[col_val][0],str)])


#train_df.apply(le.fit_transform)
    
#print(le.fit(train_df['device.deviceCategory']).transform(train_df[['device.deviceCategory','device.browser']]))
train_df.head()
test_df.head()

#X_train.head()


# In[ ]:


from sklearn.ensemble import RandomForestRegressor

#lr_model = LinearRegression()
rf = RandomForestRegressor(n_estimators = 500, random_state = 42)
#model = lr_model.fit(X_train,y_train)
model = rf.fit(X_train,y_train)
#print(model.coef_)


# In[ ]:


prediction = rf.predict(test_df[test_col])
print(len(prediction))


# In[ ]:


import math

test_df = test_df.assign(predictedRevenue=prediction)
groupby_visitorId = test_df.groupby('fullVisitorId').sum()

def calculate_log(revenue):
    return math.log10(abs(revenue) + 1)

predicted_df = groupby_visitorId['predictedRevenue'].apply(calculate_log)
predicted_df_new = pd.DataFrame({'fullVisitorId' : list(predicted_df.keys()),'PredictedLogRevenue' : predicted_df.values})
predicted_df_new.to_csv('output_pred.csv',index=False)

