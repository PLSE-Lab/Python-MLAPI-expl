#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Facebook advertising is one of the biggest platforms that allows organizations to reach out to their consumers online. Whether the organization's objectives are to increase the traffic to their web property, generate leads, brand awareness or sell more products, Facebook Ad Campaigns are part of every social media strategy. The platform has features to define the target audience based on their location, age, gender, language , behaviour, connections or interest.
# 
# This analysis provides insights based on the current defined KPI's of an organization, which in this case is to improve conversion through their website. The purpose of the analysis is also to extract neccessary knowledge from the collected data set to see the underperforming facebook campaigns, explore possibilities to reduce spent and improve conversion and finally create a predictive model to automate facebook ad placement for the next quarter.

# # Table of Contents
# 
# 1. Introduction
# 
# 2. About the Data Set
# 
# 3. Data Collection 
# 
# 4. Data Wrangling
# 
# 5. Data Exploration
# 
# 6. Model Development
# 
# 7. Model Evaluation
# 
# 8. Conclusion

# # About the Data Set
# 
# The description of each variables within the data set are defined below;
# 
# -  ad_id: unique ID for each ad.
# 
# -  xyz_campaign: an ID associated with each ad campaign of XYZ.
# 
# -  fb_campaign_id: an ID associated with how Facebook tracks each campaign.
# 
# -  age: age of the person to whom the ad is shown.
# 
# -  gender: gender of the person to whom the ad is shown.
# 
# -  interest: a code specifying the categorty to which the person's interest belongs (interests are as mentioned in the person's Facebook public profile) 
# 
# -  Impressions: the number of times the ad was shown.
# 
# -  Clicks: Number of clicks on for that ad.
# 
# -  Spent: Amount paid by company xyz to Facebook, to show that ad.
# 
# -  Total conversion: Total number of people who enquired about the product after seeing the ad.
# 
# -  Approved conversion: Total number of people who bought the product after seeing the ad.

# # Data Collection

# In[ ]:


# importing neccessary libraries
import pandas as pd
import numpy as np


# In[ ]:


# import the data set and create the data frame

fb_ads=pd.read_csv('../input/fb_conversion_data.csv')
fb_ads.head(5)


# In[ ]:


# review to see if there are any missing data
missing_data=fb_ads.isnull()
missing_data.head(3)


# In[ ]:


for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")


# There are no missing data in our data set.

# In[ ]:


# review the data types
fb_ads.dtypes


# All data variables are numeric except age and gender are objects. The age variable values are defined as ages between two numbers rather than average number. Gender variable is also not numerical datatype. For future analysis and model development, it will be required to convert these variables to numeric data type.

# In[ ]:


# review full information on the data set
fb_ads.info()


# In[ ]:


fb_ads.shape


# The data set consists of 1143 rows and 11 columns.

# In[ ]:


fb_ads.columns


# Based on the imported data, very minor data wrangling is required. 
# 
# -  Average the grouped age and convert the data type to numeric.
# 
# -  Convert Gender variable M being 0 and F being 1 numeric variable.

# # Data Wrangling

# In[ ]:


# Review the age and gender distribution briefly

age_count= fb_ads['age'].value_counts()
age_count= age_count.to_frame()


# In[ ]:


age_count.head()


# In[ ]:


# rename column to age count and index age
age_count.rename(columns={'age': 'Age Count'}, inplace=True)
age_count.index.name='Age'
age_count.head(5)


# In[ ]:


# visualize the age count
# import neccessary libraries
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


age_count.plot(kind='bar',
              figsize=(15,10),
              color='orange')

plt.title('Age Count Frequency')
plt.xlabel('Age Count')
plt.ylabel('Frequency')

plt.show()


# Organization's Facebook Ad Campaigns targets consumers ages between 30-34 years old followed by 45-49, 35-39 and 40-44 years of age.

# In[ ]:


# similarly review the gender distribution
gender_count=fb_ads['gender'].value_counts()
gender_count=gender_count.to_frame()
gender_count.rename(columns={'gender': 'Gender Count'}, inplace=True)
gender_count.index.name='Gender Count'
gender_count.head(5)


# In[ ]:


# visualize gender count
gender_count.plot(kind='bar', figsize=(15,10), color='red')

plt.title('Gender Frequency Distribution')
plt.xlabel('Gender Count')
plt.ylabel('Frequency')

plt.show()


# Organization's Facebook Ad Campaigns target more male audience than female audience.

# In[ ]:


# Convert Age to average age and numeric variable
fb_ads['age'][fb_ads['age']=='30-34']=32
fb_ads['age'][fb_ads['age']=='35-39']=37
fb_ads['age'][fb_ads['age']=='40-44']=42
fb_ads['age'][fb_ads['age']=='45-49']=47
fb_ads[['age']]=fb_ads[['age']].astype('int')


# In[ ]:


# convert Gender to 0 and 1 values and numeric variable
fb_ads['gender'][fb_ads['gender']=='M']=0
fb_ads['gender'][fb_ads['gender']=='F']=1
fb_ads[['gender']]=fb_ads[['gender']].astype('int')


# In[ ]:


fb_ads.head(5)


# In[ ]:


fb_ads.dtypes


# All variables are within the dataset are numeric and ready for data exploration.

# # Data Exploration

# In[ ]:


fb_ads['age'].unique()
fb_ads['interest'].unique()


# In[ ]:


### Review the Spent distribution on gender

plt.figure(figsize=(15,10))
sns.boxplot(x='age', y='Spent', data=fb_ads)

plt.show()


# There are many outliers in the 30-34 years old age group, this can explain the fact that, even though the majority of the organization's Facebook Ads are targeting the age group between 30-34 years old, the organization is spending the most for the 45-49 age group. 

# In[ ]:


# Review the Spent distribution on Gender
plt.figure(figsize=(15,10))
sns.boxplot(x='gender', y='Spent', data=fb_ads)

plt.show()


# Similar to the age group, even though the majority of the facebook ads are targeting males, the organization is spending most of their facebook adverstising budget on females.

# In[ ]:


# Review the Spent distribution on specific campaigns
plt.figure(figsize=(15,10))
sns.boxplot(x='xyz_campaign_id', y='Spent', data=fb_ads)

plt.show()


# Organization is spending significant amount of their budget on Campaign 1178. Further exploration of this specific campaign will provide more insights on the overall Facebook Ad Placement of the organization.

# In[ ]:


# Review the Spent distribution on interests.
plt.figure(figsize=(15,10))
sns.boxplot(x='interest', y='Spent', data=fb_ads)

plt.show()


# There are a lot of outliers on interest 16 and 10. Even though majority of the spent is on interest 100, 101 and 107, the overall Facebook Ad Budget allocation is distributed evenly. The review of the outliers on interest 16, 27, 28 and 29 would benefit the next quarter Facebook Ad Placement of the organization.

# The data set collected from Organization's Facebook Advertising Platform, provides important variables such as Impressions, Clicks, Spent, Total Conversion and Approved Conversion, however there are other standard engagement and conversion metrics such as Click Through Rate (CTR) which is the percentage of how many impressions become clicks, Conversion Rate(CR), which is the percentage of clicks that results in conversion, Cost Per Click(CPC), which is cost of each consumer click. 
# 
# In order to improve the insights from the dataset, these features can be added to the dataframe.

# In[ ]:


# Creating CTR and CPC as new feautures and adding them to the dataframe

fb_ads['CTR']=(fb_ads['Clicks']/fb_ads['Impressions'])*100
fb_ads['CPC']=fb_ads['Spent']/fb_ads['Clicks']


# In[ ]:


fb_ads.head(5)


# In[ ]:


# Review of overall correlation between variables

plt.figure(figsize=(15,10))
sns.heatmap(fb_ads.corr())

plt.show()


# The strongest correlation are between Impressions, Clicks, Spent, Total_Conversion and Approved Conversion. 

# In[ ]:


# detail correlation between Impressions and Clicks

fb_ads[['Impressions', 'Clicks']].corr()


# In[ ]:


plt.figure(figsize=(15,10))
sns.regplot(x='Impressions', y='Clicks', data=fb_ads)
plt.ylim(0,)


# In[ ]:


# look at pearson correlation and pvalue
# import neccesary library
from scipy import stats


# In[ ]:


pearson_1, p_value_1 = stats.pearsonr(fb_ads['Impressions'], fb_ads['Clicks'])
print(pearson_1);
print(p_value_1)


# The correlation between Impressions and Clicks are expected as the more Facebook Ads of a campaign displayed the more users will click on the ads however the correlation is also significant and linear relationship is strong.

# In[ ]:


# detail correlation between Clicks and Spent

fb_ads[['Spent', 'Clicks']].corr()


# In[ ]:


plt.figure(figsize=(15,10))
sns.regplot(x='Spent', y='Clicks', data=fb_ads)
plt.ylim(0,)


# Similar to Impressions and Clicks correlation, Clicks and Spent has a strong linear relationship and significant correlation.

# In[ ]:


# detail distribution of Campaign and CTR

plt.figure(figsize=(15,10))
sns.boxplot(x='xyz_campaign_id', y='CTR', data=fb_ads)
plt.show()


# The earlier analysis showed that the organization is spending majority of their facebook advertising budget on Campaign 1178, Further exploring the distribution of campaign for Click Through Rate, it is evident that campaign 916 and 936 have better consumer engagement. Even though the amount of facebook ads being served to the users for campaign 1178 is much higher than campaign 916 and 936, the engagement is much lower.

# In[ ]:


# filter campaign 1178 data and review in detail

is_1178=fb_ads['xyz_campaign_id']==1178
campaign_1178=fb_ads[is_1178]
campaign_1178.head(5)


# In[ ]:


# review the correlation between Spent and CTR on Campaign 1178
campaign_1178[['Spent', 'CTR']].corr()


# In[ ]:


plt.figure(figsize=(15,10))
sns.regplot(x='CTR', y='Spent', data=campaign_1178)
plt.ylim(0,)


# There is a positive linear relationship between Spent and Click Through Rate. This is expected as organization is spending their facebook budget on campaign 1178 and there should be clicks for this campaign ad. However the correlation is weak and should be much more stronger considering the other campaign ads(916 and 936). Another important fact to note is that, there is an organic engagement to this particular campaign. 

# Based on the data analysis, the features that can help create a predictive model are, Impressions, Clicks, Spent, Total Conversion, Approved Conversion, CTR and CPC. The objective of the organization is not only improve conversion in terms of sales but also to improve awareness. Hence, the target feature for the predictive model will be Total Conversion.

# # Model Development
# 
# The model foundation that is chosen for this particular predictive model is Decision Tree. Within the different decision tree libraries, the predictive model leverages Random Forest as to define the most efficient depth of the branches.

# In[ ]:


# select feature set X

X = fb_ads[['Impressions', 'Clicks', 'Spent', 'CTR']]
X.head(5)


# In[ ]:


# select target feature y

y=fb_ads[['Total_Conversion']]
y.head(5)


# In[ ]:


# create training and test data and split the feature set data
# import neccessary libraries
from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.3, random_state=1)


# In[ ]:


X_train.head(5)


# In[ ]:


y_train.head(5)


# In[ ]:


X_train.isnull().sum()


# In[ ]:


y_train.isnull().sum()


# In[ ]:


# create the model with Random Forest
# import neccessary libraries
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


fb_ads_model=RandomForestRegressor(random_state=1)
fb_ads_model.fit(X_train, y_train)


# In[ ]:


# create prediction with the model
fb_ads_pred=fb_ads_model.predict(X_test)


# In[ ]:


print(fb_ads_pred[0:5])


# In[ ]:


print(y_test[0:5])


# # Conclusion

# From a broad overview of the data, it is clear that the organization is targeting consumers that are between 30-34 years old the most. There are more male audience is being targeted in the total amount of Facebook Ads. However, they are spending the most of their facebook campaign ad budget on the consumers that are between 45-49 years old and female audience. Based on this finding, revisit of the current consumer personas for the business is recommended.
# 
# It is also clear that the more business spends on facebook ad campaigns the higher the conversions are. Campaign 1178 gets the most Spent within the Facebook Campaign Ads, however gets the least amount of engagement. Overall spent for the interest categories are distributed evenly with interest 100, 101, 107 getting the maximum spent. Based on these findigs, revisit to campaign creative, strategy of Campaign 1178 and review of Facebook Ad Placements on interest 16,27,28 and 29 (which has the highest outliers) is strongly recommended.
# 
# The more the business spent, the more the engagement is for all campaigns, however the correlation and positive linear relationship is weak. Based on thos analysis, revisit of the spent distribution between campaigns are recommended.
# 
# In regards to the predictive model that is developed; It is certainly required to further evaluate the model and make neccessary adjustments for the quarter.
