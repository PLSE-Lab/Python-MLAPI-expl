#!/usr/bin/env python
# coding: utf-8

# # 1. Importing the datasets as dataframes:

# In[9]:


import pandas as pd

dataset_path = "../input/customers details for bank loan approval/Customers Details for Bank loan Approval/"

df_application_data = pd.read_csv(dataset_path+'application_data.csv')

df_previous_application = pd.read_csv(dataset_path+'previous_application.csv')


# In[ ]:


df_application_data.head() # glancing the dataset application_data


# In[ ]:


df_previous_application.head() # glancing the dataset previous_application


# In[ ]:


# Cheking the shape of the datasets (application_data and previous_application)

print('shape of application_data: ',df_application_data.shape)

print('shape of previous_application: ',df_previous_application.shape)


# # 2. Data Quality-Check, Preparation and Cleaning:
# 
# ## 2.1 Finding the percentage of missing values for all the columns in both the datasets :

# In[ ]:


# Displaying the top5 columns with the highest percentage of missing values in application_data

percent_missing_app = df_application_data.isnull().sum() * 100 / len(df_application_data)

percent_missing_app.sort_values(ascending=False).head() # please remove head() to display all columns


# In[ ]:


# Displaying the top5 columns with the highest percentage of missing values in previous_application

percent_missing_app = df_previous_application.isnull().sum() * 100 / len(df_previous_application)

percent_missing_app.sort_values(ascending=False).head() # please remove head() to display all columns


# ## 2.2 Identifying the number of columns that have more than 50% of null values in both datasets :

# In[ ]:


# Identifying columns that has 50 percent of the Null values in df_application_data

row_count_50perc_app = df_application_data.shape[0]/2 # Finding the count of 50% of rows

Colms_with_null_50perc_more_app = df_application_data.columns[df_application_data.isnull().sum()>row_count_50perc_app]

print('Total count of columns with 50% null value in df_application_data: ',len(Colms_with_null_50perc_more_app))

# Identifying columns that has 50 percent of the Null values in df_previous_application

row_count_50perc_prev = df_previous_application.shape[0]/2 # Finding the count of 50% of rows

Colms_with_null_50perc_more_prev = df_previous_application.columns[df_previous_application.isnull().sum()>row_count_50perc_prev]

print('Total count of columns with 50% null value in df_previous_application: ',len(Colms_with_null_50perc_more_prev))


# ## 2.3 Removing the columns that have more than 50% of null values in both datasets :

# In[ ]:


# Dropping columns with more than 50% of null values in both dataset

df_application_data = df_application_data.drop(Colms_with_null_50perc_more_app, axis=1)

df_previous_application = df_previous_application.drop(Colms_with_null_50perc_more_prev, axis=1)

# Checking the shape after dropiing the columns

print('shape of application_data after removing 50% of null valued columns: ',df_application_data.shape)

print('shape of previous_application after removing 50% of null valued columns: ',df_previous_application.shape)


# **Please Note:**
# 
# It will be benificial to consider only the less missing values columns by default for analysis. So we will try to get the columns with very less missing values (around 13%) for further analysis.

# ## 2.4 Identifying columns that has less than 13% of missing values :

# In[ ]:


# Identifying columns that has 13 percent or less of the Null values in df_application_data

row_count_13perc_app = df_application_data.shape[0] * 0.13 # Finding the count of 13% of rows

Colms_with_null_13perc_or_less_app = df_application_data.columns[df_application_data.isnull().sum()<row_count_13perc_app]

print('Total count of columns with 13% or less null value in df_application_data: ',len(Colms_with_null_13perc_or_less_app))

print('\ncolumns are: ', Colms_with_null_13perc_or_less_app)

# Identifying columns that has 13 percent or less of the Null values in df_previous_application

row_count_13perc_prev = df_previous_application.shape[0]*0.13 # Finding the count of 13% of rows

Colms_with_null_13perc_or_less_prev = df_previous_application.columns[df_previous_application.isnull().sum()<row_count_13perc_prev]

print('\nTotal count of columns with 13% or less null value in df_prev_application: ',len(Colms_with_null_13perc_or_less_prev))

print('\ncolumns are: ', Colms_with_null_13perc_or_less_prev)


# ## 2.5 Classifying the above list of less than 13% null valued columns in Categorical and non-categorical in both datasets.
# 
# **Please note:** Futher analysis will be carried out only in these columns.

# In[ ]:


# categorical columns in application_data dataset

category_clmns_app_data = ['SK_ID_CURR','TARGET','NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR',
                                     'FLAG_OWN_REALTY','CNT_CHILDREN','NAME_TYPE_SUITE','NAME_INCOME_TYPE',
                                     'NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE',
                                     'REGION_RATING_CLIENT','FLAG_MOBIL','FLAG_EMP_PHONE','FLAG_WORK_PHONE',
                                     'FLAG_CONT_MOBILE','FLAG_PHONE','FLAG_EMAIL','ORGANIZATION_TYPE','FLAG_MOBIL',
                                     'FLAG_EMP_PHONE','FLAG_WORK_PHONE','FLAG_CONT_MOBILE','FLAG_DOCUMENT_4',
                                     'FLAG_DOCUMENT_5','FLAG_DOCUMENT_6','FLAG_DOCUMENT_7','FLAG_DOCUMENT_8',
                                     'REG_REGION_NOT_WORK_REGION','LIVE_REGION_NOT_WORK_REGION',
                                     'REG_CITY_NOT_LIVE_CITY','REG_CITY_NOT_WORK_CITY','LIVE_CITY_NOT_WORK_CITY',
                                     'WEEKDAY_APPR_PROCESS_START','CNT_FAM_MEMBERS','REG_REGION_NOT_LIVE_REGION','ORGANIZATION_TYPE',
                                     'NAME_TYPE_SUITE','REGION_RATING_CLIENT_W_CITY', 'HOUR_APPR_PROCESS_START','FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3',
                                     'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6',
                                     'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9',
                                     'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12',
                                     'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15',
                                     'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18',
                                     'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']

# Non-categorical columns in application_data dataset

non_category_clmns_app_data = ['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE',
                                         'REGION_POPULATION_RELATIVE','DAYS_BIRTH','DAYS_EMPLOYED', 
                                         'DAYS_REGISTRATION','DAYS_ID_PUBLISH',
                                         'EXT_SOURCE_2','OBS_30_CNT_SOCIAL_CIRCLE', 
                                         'DEF_30_CNT_SOCIAL_CIRCLE','OBS_60_CNT_SOCIAL_CIRCLE', 
                                         'DEF_60_CNT_SOCIAL_CIRCLE','DAYS_LAST_PHONE_CHANGE',]

# categorical columns in previous_application dataset

category_clmns_prev_data = ['SK_ID_PREV', 'SK_ID_CURR','NAME_CONTRACT_TYPE', 'HOUR_APPR_PROCESS_START','WEEKDAY_APPR_PROCESS_START',
                                     'FLAG_LAST_APPL_PER_CONTRACT','NFLAG_LAST_APPL_IN_DAY','NAME_CASH_LOAN_PURPOSE',
                                      'NAME_CONTRACT_STATUS','NAME_PAYMENT_TYPE', 'CODE_REJECT_REASON', 
                                      'NAME_CLIENT_TYPE','NAME_GOODS_CATEGORY', 'NAME_PORTFOLIO', 'NAME_PRODUCT_TYPE',
                                      'CHANNEL_TYPE', 'NAME_SELLER_INDUSTRY','NAME_YIELD_GROUP', 'PRODUCT_COMBINATION']

# Non-categorical columns in previous_application dataset

non_category_clmns_prev_data = ['AMT_APPLICATION','AMT_CREDIT','DAYS_DECISION',
                                         'SELLERPLACE_AREA']


# ## 2.6 Keeping only the columns in the main dataframes that has less than 13% of missing values in the dataframes for further analysis.

# In[ ]:


df_application_data = df_application_data[Colms_with_null_13perc_or_less_app]
df_previous_application = df_previous_application[Colms_with_null_13perc_or_less_prev]

# Checking the shape of the modified dataframes:

print(df_application_data.shape)
print(df_previous_application.shape)


# ## 2.7 Finding Outliers:
# 
# In general, outliers are very high and low values.
# 
# Please Note: We are not considering the variables whose standard deviation is less than 85% from its mean, since outliers are expected only in the cases of large variations. 

# In[ ]:


numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

num_df1 = df_application_data.select_dtypes(include=numerics)
num_df2 = df_previous_application.select_dtypes(include=numerics)

# We are not considering the numerical columns that are categorical, in detecting outliers.

categorical_num_columns_df1 = ['SK_ID_CURR','TARGET','FLAG_MOBIL','FLAG_EMP_PHONE','FLAG_WORK_PHONE','FLAG_CONT_MOBILE',
                           'FLAG_PHONE','FLAG_EMAIL','FLAG_DOCUMENT_2','FLAG_DOCUMENT_3','FLAG_DOCUMENT_4','FLAG_DOCUMENT_5',
                           'FLAG_DOCUMENT_6','FLAG_DOCUMENT_7','FLAG_DOCUMENT_8','FLAG_DOCUMENT_9','FLAG_DOCUMENT_10',
                           'FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13','FLAG_DOCUMENT_14','FLAG_DOCUMENT_15',
                           'FLAG_DOCUMENT_16','FLAG_DOCUMENT_17','FLAG_DOCUMENT_18','FLAG_DOCUMENT_19','FLAG_DOCUMENT_20',
                           'FLAG_DOCUMENT_21'] 

categorical_num_columns_df2 = ['SK_ID_PREV','SK_ID_CURR','NFLAG_LAST_APPL_IN_DAY']

# Removing the numerical_categorical columns 
num_df1 = num_df1.drop(columns=categorical_num_columns_df1)
num_df2 = num_df2.drop(columns=categorical_num_columns_df2)

clmn_app_non_cate_outlier = num_df1.columns[((num_df1.std()/num_df1.mean())*100).abs()>85].tolist()

print("Columns that has outliers in application_data.csv, \nwhich has a standard deviation of more than 85% from its mean:\n\n ", clmn_app_non_cate_outlier)

clmn_prev_non_cate_outlier = num_df2.columns[((num_df2.std()/num_df2.mean())*100).abs()>85].tolist()

print("\nColumns that has outliers in previous_application.csv, \nwhich has a standard deviation of more than 85% from its mean:\n\n", clmn_prev_non_cate_outlier)


# **Please Note:**
# 
# In our datasets we are not removing any outliers because these outliers can be actual/correct values, not necessary that all outliers are mis-interupted values.
# 
# **For example:**
# 
# In a general banking dataset consider that, under debt column, in one of the rows it is mentioned as 10000 crores, where as the average debt calculated among all rows is only 5 crores. Here it is not necessary that the particular row holding 10000 crore (outlier) is a misinterupted value. There could be a person (eg: Vijay Malliah) who has a debt of 10000 Crores.

# ## 2.8 Dealing with the missing values:

# In[ ]:


# Finding the list of actual null columns from the dataframes (df_application_data and 
#df_previous_application),that contains all columns which has less than 13% Null values

clmn_app_null_cate = [i for i in df_application_data.columns[df_application_data.isna().any()].tolist() if i in category_clmns_app_data]
                      
print('List of all categorical columns that has actual missing values in among less than 13% null value columns in application_data: ', clmn_app_null_cate)
                      
clmn_app_null_non_cate = [i for i in df_application_data.columns[df_application_data.isna().any()].tolist() if i in non_category_clmns_app_data]                    
                      
print('\n\nList of all non-categorical columns that has actual missing values in among less than 13% null value columns in application_data: ', clmn_app_null_non_cate)

clmn_prev_null_cate = [i for i in df_previous_application.columns[df_previous_application.isna().any()].tolist() if i in category_clmns_prev_data]                      

print('\n\nList of all categorical columns that has actual missing values in previous_application_data: ', clmn_prev_null_cate)                       
                       
clmn_prev_null_non_cate = [i for i in df_previous_application.columns[df_previous_application.isna().any()].tolist() if i in non_category_clmns_prev_data]                      

print('\n\nListist of all non-categorical columns that has actual missing values in previous_application_data: ', clmn_prev_null_non_cate)


# ### 2.8.1 Dealing with the missing calues in categorical variable:
# 
# For categorical variables, an optimistic soltuion to replace the null values is to replace with the most most occuring categorical value. In other words, we are replacing the missing values with the mode.

# In[ ]:


# Repalcing null categorical variable with mode in application_data dataframe

for column in clmn_app_null_cate:
    df_application_data[column].fillna(df_application_data[column].mode()[0], inplace=True)


# In[ ]:


# Repalcing null categorical variable with mode in previous_application dataframe

for column in clmn_prev_null_cate:
    df_previous_application[column].fillna(df_previous_application[column].mode()[0], inplace=True)


# ### 2.8.2 Deling with the missing values in non-categorical variables that has outliers:
# 
# For variables that contains outliers (more standard deviation) we cannot fill the null values with mean because the deviation will impact the mean. For example: consider that in a room, among with many typical IT employees there are steve jobs and Bill gates. Here if we calculate the average salary of all people it will not match with the salary of any of the typical IT employees.
# 
# In this case median is the best option to choose in replacing the null values

# In[ ]:


# Preparing a list containing null, non-categoricall variables containing outliers in application_data

clmn_app_null_non_cate_outlier = [i for i in clmn_app_null_non_cate if i in clmn_app_non_cate_outlier]

# Preparing a list containing null, non-categoricall variables containing outliers in previous_application

clmn_prev_null_non_cate_outlier = [i for i in clmn_prev_null_non_cate if i in clmn_prev_non_cate_outlier]


# In[ ]:


# Replacing null,non-categorical variable (having outliers) with median in application_data dataframe

for column in clmn_app_null_non_cate_outlier:
    df_application_data[column].fillna(df_application_data[column].median(), inplace=True)
    
# Replacing null,non-categorical variable (having outliers) with median in previous_application dataframe

for column in clmn_prev_null_non_cate_outlier:
    df_previous_application[column].fillna(df_previous_application[column].median(), inplace=True)


# ### 2.8.3 Deling with the missing values in non-categorical variables that does not have outliers:
# 
# For less outlier variables we can consider replacing it with mean, since the deviation among non-outlier values will be very less.  

# In[ ]:


# Preparing a list containing null, non-categoricall variables containing no outliers in application_data

clmn_app_null_non_cate_no_outlier = [i for i in clmn_app_null_non_cate if i not in clmn_app_non_cate_outlier]

# Preparing a list containing null, non-categoricall variables containing no outliers in previous_application

clmn_prev_null_non_cate_no_outlier = [i for i in clmn_prev_null_non_cate if i not in clmn_prev_non_cate_outlier]


# In[ ]:


# Replacing null,non-categorical variable (having no outliers) with mean in application_data dataframe

for column in clmn_app_null_non_cate_no_outlier:
    df_application_data[column].fillna(df_application_data[column].mean(), inplace=True)
    
# Replacing null,non-categorical variable (having no outliers) with mean in previous_application dataframe

for column in clmn_prev_null_non_cate_no_outlier:
    df_previous_application[column].fillna(df_previous_application[column].mean(), inplace=True)


# ### 2.8.4 Finally, checking if there is any columns that are still having null values in the dataframes  application_data and previous application.

# In[ ]:


print('Is there any null values in df_application_data: ',df_application_data.isnull().values.any())
print('Is there any null values in df_previous_application: ',df_previous_application.isnull().values.any())


# # 3. Ananlysis to be carried:
# 
# **Objective:**
# 
# By Analysing the given datasets we could sense that the columns which gives direct information on the loan approval/rejection are 'Target' and 'NAME_CONTRACT_STATUS' repectively on 'application_data.csv' and 'Previous_application.csv'. Further, I'm gonna perform univariate and bi-variate analysis to get more insight on these columns and the co-relation with the other columns, so that, we can understand how loan approval/rejection is estimated, and which are the cariables influences this.

# ## 3.1 Finding if there is Imbalance in data:
# 
# Imbalance of data is a major concern while dealing with classification problems. 
# 
# **For example:** 
# 
# As per our problem statement we have to find the patter to judge whether a person can re-pay the loan or becoming a defaulter. In order to find the patter we have to train/analyse our model using the already available dataset with the correct classification variable ('Target').
# 
# If the dataset is imbalanced, in other words if the dataset has more data about defaulters and less data about the people who could repay the loan, then our model would becomes more biased to defaulters which may lead to more incorrect prediction of defaulters.   
# 
# Since we are concentrated much on the 'Tareget' variable to detect whether to approve/reject loan, we are going to focus on finding if there is any imbalance on this variable.

# In[ ]:


import matplotlib.pyplot as plt

count_1 = 0; count_0 = 0 # initialisation

for i in df_application_data['TARGET'].values:
    if i == 1:
        count_1 = count_1+1
    else:
        count_0 = count_0+1
        
count_1_perc = (count_1/(count_1 + count_0))*100

count_0_perc = (count_0/(count_1 + count_0))*100

X = ['Defaulter','Non-Defaulter']

Y = [count_1_perc, count_0_perc]

plt.bar(X,Y, width = 0.8)

plt.ylabel('(%) of the defaulter/Non-defaulter data')

plt.show()

print('Ratios of imbalance in percentage with respect to non-defaulter and defaulter datas are: %f and %f'%(count_0_perc,count_1_perc))
print('Ratios of imbalance in real-numbers with respect to non-defaulter and defaulter datas is %f : 1 (approx)'%(count_0/count_1))


# From the above graph we can very clearly see that there is an imbalabce between the defaulters data and the non-defaulters. As mentioned earlier, imbalance of data will tend to create a biased model. One best technique to avoid the curse of imbalanced data is by undersampling the larger classified dataset and by oversampling the less classified dataset. So that, the final dataset will have a balanced/equal number of data among all the labels.

# ## 3.2 Dividing the data into two sets, i.e. Target=1 and Target=0 for application_data:

# In[ ]:


df_application_data_T_1 = df_application_data[df_application_data.TARGET == 1]
df_application_data_T_0 = df_application_data[df_application_data.TARGET == 0]


# ## 3.3 Dividing the data into two sets, i.e. Target=1 and Target=0 for previous_application dataset:

# ### 3.3.1 Creating a target variable for the previous_application dataset:
# 
# In order to create a target varible in prev_app_data we are going to consider the column: NAME_CONTRACT_STATUS
# 
# We are going classify the status 'Refused' as 1 and 'Approved' as 0. The reason for this assumption is that if the previous company rejects the loan for a person then the only reason for doing so is that the company would have sensed that the person might be having payment difficulties. In other hand, if the previous company approved the loan for a person then the only reason would be that the company would have sensed that the person whould not have any payment difficulties.
# 
# We are going to remove all other rows where the status is 'cancelled' or 'unused offer' 

# In[ ]:


Prev_target=[]
for i in df_previous_application['NAME_CONTRACT_STATUS'].tolist():
    if i == 'Approved':
        Prev_target.append(0)
    elif i == 'Refused':
        Prev_target.append(1)
    else:
        Prev_target.append(None)
        
# Creting a 'Target' variable with  Approved = 0, Refused = 1, all other as Null

df_previous_application['Target'] = Prev_target 

# Removing all rows that are having Null in Target varible.

df_previous_application = df_previous_application.loc[(df_previous_application['Target'] == 1) | (df_previous_application['Target'] == 0)] 

df_previous_application.head()


# ### 3.3.2 Dividing the data into two sets, i.e. Target=1 and Target=0 for previous_application_data:

# In[ ]:


df_previous_application_T_1 = df_previous_application[df_previous_application.Target == 1]
df_previous_application_T_0 = df_previous_application[df_previous_application.Target == 0]


# ## 3.4 Univariate Analysis on selected ordered-categorical variables that are common in both datasets:

# ### 3.4.1 Univariate analysis on selected ordered-categorical variable ('WEEKDAY_APPR_PROCESS_START') in both datasets for Target = 1

# In[ ]:


# Plotting the count and rank varaibles for WEEKDAY_APPR_PROCESS_START in both datasets for Target = 1

plt.figure(figsize=(18,6))

plt.subplot(121)

plt.title('Count of Applications filled by defaulters (T=1) on these days in application_data')

plt.ylabel('Count of defaulters');plt.xlabel('Day on which the loan applciation is filled [FIG: 1]')

df_application_data_T_1['WEEKDAY_APPR_PROCESS_START'].value_counts().plot(kind='bar')

plt.subplot(122)

plt.title('Count of Applications filled by defaulters (T=1) on these days in previous_application')

plt.ylabel('Count of defaulters');plt.xlabel('Day on which the loan applciation is filled [FIG: 2]')

df_previous_application_T_1['WEEKDAY_APPR_PROCESS_START'].value_counts().plot(kind='bar')

plt.show()


# ### 3.4.2 Univariate analysis on selected ordered-categorical variable ('WEEKDAY_APPR_PROCESS_START') in both datasets for Target = 0

# In[ ]:


# Plotting the count and rank varaibles for WEEKDAY_APPR_PROCESS_START in both datasets for Target = 0

plt.figure(figsize=(18,6))

plt.subplot(121)

plt.title('Count of Applications filled by non-defaulters (T=0) on these days in application_data')

plt.ylabel('Count of non-defaulters');plt.xlabel('Day on which the loan applciation is filled [FIG: 3]')

df_application_data_T_0['WEEKDAY_APPR_PROCESS_START'].value_counts().plot(kind='bar')

plt.subplot(122)

plt.title('Count of Applications filled by non-defaulters (T=0) on these days in application_data')

plt.ylabel('Count of non-defaulters');plt.xlabel('Day on which the loan applciation is filled [FIG: 4]')

df_previous_application_T_0['WEEKDAY_APPR_PROCESS_START'].value_counts().plot(kind='bar')

plt.show()


# *Analysis from 3.4.1 and 3.4.2:*
# 
# * FIG:1&2, shows that the maximum of defaulters who filled the application is on Tuesdays and the less number of applications filled is on Sunday.
# 
# * FIG:3, shows the same pattern as above among non-defaulters too which is a total contradiction.
# 
# * FIG:4, shows the similar pattern too where very less applications are filled on Sundays among non-defaulters.
# 
# *Conclusion:* We cannot decide much using the days of the week on which the applications are filled in identifying the defaulters, because both in the case of defaulters and non-defaulters the pattern followed is very similar.

# ### 3.4.3 Univariate analysis on selected ordered-categorical variable ('HOUR_APPR_PROCESS_START') in both datasets for Target = 1

# In[ ]:


# Plotting the count and rank varaibles for WEEKDAY_APPR_PROCESS_START in both datasets for Target = 1

plt.figure(figsize=(18,6))

plt.subplot(121)

plt.title('Count of Applications filled by defaulters (T=1) on these hours in application_data')

plt.ylabel('Count of defaulters');plt.xlabel('Hour on which the loan applciation is filled [FIG: 5]')

df_application_data_T_1['HOUR_APPR_PROCESS_START'].value_counts().plot(kind='bar')

plt.subplot(122)

plt.title('Count of Applications filled by defaulters (T=1) on these hours in previous_application')

plt.ylabel('Count of defaulters');plt.xlabel('Hour on which the loan applciation is filled [FIG: 6]')

df_previous_application_T_1['HOUR_APPR_PROCESS_START'].value_counts().plot(kind='bar')

plt.show()


# ### 3.4.4 Univariate analysis on selected ordered-categorical variable ('HOUR_APPR_PROCESS_START') in both datasets for Target = 0

# In[ ]:


# Plotting the count and rank varaibles for WEEKDAY_APPR_PROCESS_START in both datasets for Target = 0

plt.figure(figsize=(18,6))

plt.subplot(121)

plt.title('Count of Applications filled by non-defaulters (T=0) on these hours in application_data')

plt.ylabel('Count of non-defaulters');plt.xlabel('Hour on which the loan applciation is filled [FIG: 7]')

df_application_data_T_0['HOUR_APPR_PROCESS_START'].value_counts().plot(kind='bar')

plt.subplot(122)

plt.title('Count of Applications filled by non-defaulters (T=0) on these hours in previous_application')

plt.ylabel('Count of non-defaulters');plt.xlabel('Hour on which the loan applciation is filled [FIG: 8]')

df_previous_application_T_0['HOUR_APPR_PROCESS_START'].value_counts().plot(kind='bar')

plt.show()


# *Analysis from 3.4.3 and 3.4.4:*
# 
# * FIG:5&6, shows that the maximum of defaulters who filled the application is on 10th and 11th hour.
# 
# * FIG:7&8, shows the same pattern as above among non-defaulters too which is a total contradiction.
# 
# *Conclusion:* We cannot decide much using the hours on which the applications are filled in identifying the defaulters, because both in the case of defaulters and non-defaulters the pattern followed is very similar.

# ## 3.5 Univariate Analysis on selected ordered-categorical variables that are present in application_data datasets:

# ### 3.5.1 Univariate analysis on selected ordered-categorical variable ('CNT_CHILDREN') in application_data datasets for Target = 1&0

# In[ ]:


# Plotting the count and rank varaibles for CNT_CHILDREN in application_data datasets for Target = 1&0

plt.figure(figsize=(18,6))

plt.subplot(121)

plt.title('Count of defaulters (T=1) based on CNT_CHILDREN in application_data')

plt.ylabel('Count of defaulters');plt.xlabel('CNT_CHILDREN [FIG: 9]')

df_application_data_T_1['CNT_CHILDREN'].value_counts().plot(kind='bar')

plt.subplot(122)

plt.title('Count of non-defaulters (T=0) based on CNT_CHILDREN in application_data')

plt.ylabel('Count of non-defaulters');plt.xlabel('CNT_CHILDREN [FIG: 10]')

df_application_data_T_0['CNT_CHILDREN'].value_counts().plot(kind='bar')

plt.show()


# *Analysis from 3.5.1:*
# 
# * FIG:9, shows that the maximum of the defaulters have 0 children while applying for the loan
# 
# * FIG:10, shows the same pattern as above where maximum number of non-defaulters are also having 0 children while applying for the loan
# 
# *Conclusion:* We cannot decide much using CNT_Children in identifying the defaulters, because both in the case of defaulters and non-defaulters the pattern followed is the same.

# ### 3.5.2 Univariate analysis on selected ordered-categorical variable ('NAME_EDUCATION_TYPE') in application_data datasets for Target = 1&0

# In[ ]:


# Plotting the count and rank varaibles ('NAME_EDUCATION_TYPE') for in application_data datasets for Target = 1&0

plt.figure(figsize=(18,6))

plt.subplot(121)

plt.title('Count of defaulters (T=1) based on NAME_EDUCATION_TYPE in application_data')

plt.ylabel('Count of defaulters');plt.xlabel('NAME_EDUCATION_TYPE [FIG: 11]')

df_application_data_T_1['NAME_EDUCATION_TYPE'].value_counts().plot(kind='bar')

plt.subplot(122)

plt.title('Count of non-defaulters (T=0) based on CNT_CHILDREN in NAME_EDUCATION_TYPE')

plt.ylabel('Count of non-defaulters');plt.xlabel('NAME_EDUCATION_TYPE [FIG: 12]')

df_application_data_T_0['NAME_EDUCATION_TYPE'].value_counts().plot(kind='bar')

plt.show()


# *Analysis from 3.5.2:*
# 
# * FIG:11, shows that the maximum of the defaulters have secondary level of education while applying for the loan
# 
# * FIG:12, shows the same pattern as above where maximum number of non-defaulters are also having secondary level of education while applying for the loan
# 
# *Conclusion:* We cannot decide much using NAME_EDUCATION_TYPE  in identifying the defaulters, because both in the case of defaulters and non-defaulters the pattern followed is the same.

# ## 3.6 Univariate Analysis on selected ordered-categorical variables that are present in previous_application datasets:

# ### 3.6.1 Univariate analysis on selected ordered-categorical variable ('NAME_CLIENT_TYPE') in application_data datasets for Target = 1&0

# In[ ]:


# Plotting the count and rank varaibles ('NAME_CLIENT_TYPE') for in application_data datasets for Target = 1&0

plt.figure(figsize=(18,6))

plt.subplot(121)

plt.title('Count of defaulters (T=1) based on NAME_CLIENT_TYPE in previous_application_data')

plt.ylabel('Count of defaulters');plt.xlabel('NAME_CLIENT_TYPE [FIG: 13]')

df_previous_application_T_1['NAME_CLIENT_TYPE'].value_counts().plot(kind='bar')

plt.subplot(122)

plt.title('Count of non-defaulters (T=0) based on NAME_CLIENT_TYPE in previous_application')

plt.ylabel('Count of non-defaulters');plt.xlabel('NAME_CLIENT_TYPE [FIG: 14]')

df_previous_application_T_0['NAME_CLIENT_TYPE'].value_counts().plot(kind='bar')

plt.show()


# *Analysis from 3.6.1:*
# 
# * FIG:13, shows that the maximum of the defaulters were been repeated customers while applying for the loan
# 
# * FIG:14, shows the same pattern as above where maximum number of non-defaulters are also repeated customers while applying for the loan
# 
# *Conclusion:* We cannot decide much using NAME_Client_TYPE  in identifying the defaulters, because both in the case of defaulters and non-defaulters the pattern followed is the same.

# ## 3.7 Univariate Analysis on selected unordered-categorical variables that are present in both datasets

# ### 3.7.1 Univariate analysis on selected unordered-categorical variable ('NAME_CONTRACT_TYPE') in both datasets for Target = 1

# In[ ]:


# Plotting the count and rank varaibles for NAME_CONTRACT_TYPE in both datasets for Target = 1

plt.figure(figsize=(18,6))

plt.subplot(121)

plt.title('Count of defaulters (T=1) based on the type of loans availed in application_data')

plt.ylabel('Count of defaulters');plt.xlabel('Types of Loan [FIG: 9]')

df_application_data_T_1['NAME_CONTRACT_TYPE'].value_counts().plot(kind='bar', width=0.4)

plt.subplot(122)

plt.title('Count of defaulters (T=1) based on the type of loans availed in previous_application')

plt.ylabel('Count of defaulters');plt.xlabel('Types of Loan [FIG: 10]')

df_previous_application_T_1['NAME_CONTRACT_TYPE'].value_counts().plot(kind='bar',width=0.8)

plt.show()


# ### 3.7.2 Univariate analysis on selected unordered-categorical variable ('NAME_CONTRACT_TYPE') in both datasets for Target = 0

# In[ ]:


# Plotting the count and rank varaibles for WEEKDAY_APPR_PROCESS_START in both datasets for Target = 0

plt.figure(figsize=(18,6))

plt.subplot(121)

plt.title('Count of non-defaulters (T=0) based on the type of loans availed in application_data')

plt.ylabel('Count of non-defaulters');plt.xlabel('Types of Loan [FIG: 11]')

df_application_data_T_0['NAME_CONTRACT_TYPE'].value_counts().plot(kind='bar')

plt.subplot(122)

plt.title('Count of non-defaulters (T=0) based on the type of loans availed in previous_application')

plt.ylabel('Count of non-defaulters');plt.xlabel('Types of Loan [FIG: 13]')

df_previous_application_T_0['NAME_CONTRACT_TYPE'].value_counts().plot(kind='bar')

plt.show()


# *Analysis from 3.7.1 and 3.7.2:*
# 
# * FIG:9&10, shows that the maximum of defaulters availed cash loans
# 
# * FIG:11&12, shows the same pattern as above where maximum number of non-defaulters availed cash loan too.
# 
# *Conclusion:* We cannot decide much using type of loan availed in identifying the defaulters, because both in the case of defaulters and non-defaulters the pattern followed is the same.

# ## 3.8 Univariate Analysis on selected unordered-categorical variables that are present only in application_data datasets

# ### 3.8.1 Univariate analysis on selected unordered-categorical variable ('NAME_TYPE_SUITE') in application_data datasets for Target = 1&0

# In[ ]:


# Plotting the count and rank varaibles for NAME_CONTRACT_TYPE in application_data datasets for Target = 1&0

plt.figure(figsize=(18,6))

plt.subplot(121)

plt.title('Count of defaulters (T=1) based on NAME_TYPE_SUITE in application_data')

plt.ylabel('Count of defaulters');plt.xlabel('NAME_TYPE_SUITE [FIG: 13]')

df_application_data_T_1['NAME_TYPE_SUITE'].value_counts().plot(kind='bar')

plt.subplot(122)

plt.title('Count of non-defaulters (T=0) based on NAME_TYPE_SUITE in application_data')

plt.ylabel('Count of non-defaulters');plt.xlabel('NAME_TYPE_SUITE [FIG: 14]')

df_application_data_T_0['NAME_TYPE_SUITE'].value_counts().plot(kind='bar')

plt.show()


# *Analysis from 3.8.1:*
# 
# * FIG:13, shows that the maximum of the defaulters was unaccompanied while applying for the loan
# 
# * FIG:14, shows the same pattern as above where maximum number of non-defaulters were unaccompanied too while applying for the loan
# 
# *Conclusion:* We cannot decide much using type of NAME_TYPE_SUITE in identifying the defaulters, because both in the case of defaulters and non-defaulters the pattern followed is the same.

# ### 3.8.2 Univariate analysis on selected unordered-categorical variable ('FLAG_OWN_CAR') in application_data datasets for Target = 1&0

# In[ ]:


# Plotting the count and rank varaibles for FLAG_OWN_CAR in application_data datasets for Target = 1&0

plt.figure(figsize=(18,6))

plt.subplot(121)

plt.title('Count of defaulters (T=1) based on FLAG_OWN_CAR in application_data')

plt.ylabel('Count of defaulters');plt.xlabel('FLAG_OWN_CAR (Y=owning a car & N = Not owning a car) [FIG: 15]')

df_application_data_T_1['FLAG_OWN_CAR'].value_counts().plot(kind='bar')

plt.subplot(122)

plt.title('Count of non-defaulters (T=0) based on FLAG_OWN_CAR in application_data')

plt.ylabel('Count of non-defaulters');plt.xlabel('FLAG_OWN_CAR (Y=owning a car & N = Not owning a car) [FIG: 16]')

df_application_data_T_0['FLAG_OWN_CAR'].value_counts().plot(kind='bar')

plt.show()


# *Analysis from 3.8.2:*
# 
# * FIG:15, shows that the maximum of the defaulters are not having a car while applying for the loan
# 
# * FIG:16, shows the same pattern as above where maximum number of non-defaulters were also not having car while applying for the loan
# 
# *Conclusion:* We cannot decide much using type of FLAG_OWN_CAR in identifying the defaulters, because both in the case of defaulters and non-defaulters the pattern followed is the same.

# ### 3.8.3 Univariate analysis on selected unordered-categorical variable ('CODE_GENDER') in application_data datasets for Target = 1&0

# In[ ]:


# Plotting the count and rank varaibles for CODE_GENDER in application_data datasets for Target = 1&0

plt.figure(figsize=(18,6))

plt.subplot(121)

plt.title('Count of defaulters (T=1) based on CODE_GENDER in application_data')

plt.ylabel('Count of defaulters');plt.xlabel('CODE_GENDER (F= Female & M = Male) [FIG: 17]')

df_application_data_T_1['CODE_GENDER'].value_counts().plot(kind='bar')

plt.subplot(122)

plt.title('Count of non-defaulters (T=0) based on CODE_GENDER in application_data')

plt.ylabel('Count of non-defaulters');plt.xlabel('CODE_GENDER (F= Female & M = Male) [FIG: 18]')

df_application_data_T_0['CODE_GENDER'].value_counts().plot(kind='bar')

plt.show()


# *Analysis from 3.8.3:*
# 
# * FIG:17, shows that the maximum of the defaulters are Female
# 
# * FIG:18, shows the same pattern as above where maximum number of non-defaulters were also Female
# 
# *Conclusion:* We cannot decide much using Gender in identifying the defaulters, because both in the case of defaulters and non-defaulters the pattern followed is the same.

# ### 3.8.4 Univariate analysis on selected unordered-categorical variable ('FLAG_OWN_REALTY') in application_data datasets for Target = 1&0

# In[ ]:


# Plotting the count and rank varaibles for FLAG_OWN_REALTY in application_data datasets for Target = 1&0

plt.figure(figsize=(18,6))

plt.subplot(121)

plt.title('Count of defaulters (T=1) based on FLAG_OWN_REALTY in application_data')

plt.ylabel('Count of defaulters');plt.xlabel('FLAG_OWN_REALTY (Y= owning house/flat & N = not owning house/flat) [FIG: 19]')

df_application_data_T_1['FLAG_OWN_REALTY'].value_counts().plot(kind='bar')

plt.subplot(122)

plt.title('Count of non-defaulters (T=0) based on FLAG_OWN_REALTY in application_data')

plt.ylabel('Count of non-defaulters');plt.xlabel('FLAG_OWN_REALTY (Y= owning house/flat & N = not owning house/flat) [FIG: 20]')

df_application_data_T_0['FLAG_OWN_REALTY'].value_counts().plot(kind='bar')

plt.show()


# *Analysis from 3.8.4:*
# 
# * FIG:19, shows that the maximum of the defaulters are owning a house/flat
# 
# * FIG:20, shows the same pattern as above where maximum number of non-defaulters were also owning a house/flat
# 
# *Conclusion:* We cannot decide much on owning a house/Flat in identifying the defaulters, because both in the case of defaulters and non-defaulters the pattern followed is the same.

# ### 3.8.5 Univariate analysis on selected unordered-categorical variable ('FLAG_OWN_REALTY') in application_data datasets for Target = 1&0

# In[ ]:


# Plotting the count and rank varaibles for NAME_INCOME_TYPE in application_data datasets for Target = 1&0

plt.figure(figsize=(18,6))

plt.subplot(121)

plt.title('Count of defaulters (T=1) based on NAME_INCOME_TYPE in application_data')

plt.ylabel('Count of defaulters');plt.xlabel('NAME_INCOME_TYPE [FIG: 21]')

df_application_data_T_1['NAME_INCOME_TYPE'].value_counts().plot(kind='bar')

plt.subplot(122)

plt.title('Count of non-defaulters (T=0) based on NAME_INCOME_TYPE in application_data')

plt.ylabel('Count of non-defaulters');plt.xlabel('NAME_INCOME_TYPE [FIG: 22]')

df_application_data_T_0['NAME_INCOME_TYPE'].value_counts().plot(kind='bar')

plt.show()


# *Analysis from 3.8.5:*
# 
# * FIG:21, shows that the maximum of the defaulters are working people
# 
# * FIG:22, shows the same pattern as above where maximum number of non-defaulters are also working people
# 
# *Conclusion:* We cannot decide much on NAME_INCOME_TYPE because both in the case of defaulters and non-defaulters the pattern followed is the same.

# ### 3.8.6 Univariate analysis on selected unordered-categorical variable ('NAME_FAMILY_STATUS') in application_data datasets for Target = 1&0

# In[ ]:


# Plotting the count and rank varaibles for NAME_FAMILY_STATUS in application_data datasets for Target = 1&0

plt.figure(figsize=(18,6))

plt.subplot(121)

plt.title('Count of defaulters (T=1) based on NAME_FAMILY_STATUS in application_data')

plt.ylabel('Count of defaulters');plt.xlabel('NAME_FAMILY_STATUS [FIG: 23]')

df_application_data_T_1['NAME_FAMILY_STATUS'].value_counts().plot(kind='bar')

plt.subplot(122)

plt.title('Count of non-defaulters (T=0) based on NAME_FAMILY_STATUS in application_data')

plt.ylabel('Count of non-defaulters');plt.xlabel('NAME_FAMILY_STATUS [FIG: 24]')

df_application_data_T_0['NAME_FAMILY_STATUS'].value_counts().plot(kind='bar')

plt.show()


# *Analysis from 3.8.6:*
# 
# * FIG:23, shows that the maximum of the defaulters are Married people
# 
# * FIG:24, shows the same pattern as above where maximum number of non-defaulters are also Married people
# 
# *Conclusion:* We cannot decide much on NAME_FAMILY_STATUS because both in the case of defaulters and non-defaulters the pattern followed is the same.

# ### 3.8.7 Univariate analysis on selected unordered-categorical variable ('NAME_HOUSING_TYPE') in application_data datasets for Target = 1&0

# In[ ]:


# Plotting the count and rank varaibles for NAME_HOUSING_TYPE in application_data datasets for Target = 1&0

plt.figure(figsize=(18,6))

plt.subplot(121)

plt.title('Count of defaulters (T=1) based on NAME_HOUSING_TYPE in application_data')

plt.ylabel('Count of defaulters');plt.xlabel('NAME_HOUSING_TYPE [FIG: 25]')

df_application_data_T_1['NAME_HOUSING_TYPE'].value_counts().plot(kind='bar')

plt.subplot(122)

plt.title('Count of non-defaulters (T=0) based on NAME_HOUSING_TYPE in application_data')

plt.ylabel('Count of non-defaulters');plt.xlabel('NAME_HOUSING_TYPE [FIG: 26]')

df_application_data_T_0['NAME_HOUSING_TYPE'].value_counts().plot(kind='bar')

plt.show()


# *Analysis from 3.8.7:*
# 
# * FIG:25, shows that the maximum of the defaulters are living in house/appartment
# 
# * FIG:26, shows the same pattern as above where maximum number of non-defaulters are also living in house/appartment
# 
# *Conclusion:* We cannot decide much on NAME_HOUSING_TYPE because both in the case of defaulters and non-defaulters the pattern followed is the same.

# ## 3.9 Univariate Analysis on selected unordered-categorical variables that are present only in previous_application datasets

# ### 3.9.1 Univariate analysis on selected unordered-categorical variable ('PRODUCT_COMBINATION') in previous_application datasets for Target = 1&0

# In[ ]:


# Plotting the count and rank varaibles for PRODUCT_COMBINATION in previous_application datasets for Target = 1&0

plt.figure(figsize=(18,6))

plt.subplot(121)

plt.title('Count of defaulters (T=1) based on PRODUCT_COMBINATION in previous_application')

plt.ylabel('Count of defaulters');plt.xlabel('PRODUCT_COMBINATION [FIG: 27]')

df_previous_application_T_1['PRODUCT_COMBINATION'].value_counts().plot(kind='bar')

plt.subplot(122)

plt.title('Count of non-defaulters (T=0) based on PRODUCT_COMBINATION in previous_application')

plt.ylabel('Count of non-defaulters');plt.xlabel('PRODUCT_COMBINATION [FIG: 28]')

df_previous_application_T_0['PRODUCT_COMBINATION'].value_counts().plot(kind='bar')

plt.show()


# *Analysis from 3.9.1:*
# 
# * FIG:27, shows that the maximum of the defaulters are in category 'cash X-sell: low'
# 
# * FIG:28, shows that the maximum of non-defaulters are in category 'POS household with interest'
# 
# *Conclusion:* We can frame a pattern by using the variable 'PRODUCT_COMBINATION'. That is, if the product_combination is 'cash X-sell: low' the person who applies for the loan might become a defaulter. In other hand, if the product_combination is 'POS household with interest' the person who applies for the loan might become a non-defaulter. By knowing this we can approve the loan for the non-defaulters and reject for people who tend to be defaulters. 

# ### 3.9.2 Univariate analysis on selected unordered-categorical variable ('NAME_YIELD_GROUP') in previous_application datasets for Target = 1&0

# In[ ]:


# Plotting the count and rank varaibles for NAME_YIELD_GROUP in previous_application datasets for Target = 1&0

plt.figure(figsize=(18,6))

plt.subplot(121)

plt.title('Count of defaulters (T=1) based on NAME_YIELD_GROUP in previous_application')

plt.ylabel('Count of defaulters');plt.xlabel('NAME_YIELD_GROUP [FIG: 29]')

df_previous_application_T_1['NAME_YIELD_GROUP'].value_counts().plot(kind='bar')

plt.subplot(122)

plt.title('Count of non-defaulters (T=0) based on NAME_YIELD_GROUP in previous_application')

plt.ylabel('Count of non-defaulters');plt.xlabel('NAME_YIELD_GROUP [FIG: 30]')

df_previous_application_T_0['NAME_YIELD_GROUP'].value_counts().plot(kind='bar')

plt.show()


# *Analysis from 3.9.2:*
# 
# * FIG:29, shows that the maximum of the defaulters are in category 'XNA'
# 
# * FIG:30, shows that the maximum of non-defaulters are in category 'middle'
# 
# *Conclusion:* We can frame a pattern by using the variable 'NAME_YIELD_GROUP'. That is, if the NAME_YIELD_GROUP is 'XNA' the person who applies for the loan might become a defaulter. In other hand, if the NAME_YIELD_GROUP is 'middle' the person who applies for the loan might become a non-defaulter. By knowing this we can approve the loan for the non-defaulters and reject for people who tend to be defaulters. 

# ### 3.9.3 Univariate analysis on selected unordered-categorical variable ('CHANNEL_TYPE') in previous_application datasets for Target = 1&0

# In[ ]:


# Plotting the count and rank varaibles for CHANNEL_TYPE in previous_application datasets for Target = 1&0

plt.figure(figsize=(18,6))

plt.subplot(121)

plt.title('Count of defaulters (T=1) based on CHANNEL_TYPE in previous_application')

plt.ylabel('Count of defaulters');plt.xlabel('CHANNEL_TYPE [FIG: 31]')

df_previous_application_T_1['CHANNEL_TYPE'].value_counts().plot(kind='bar')

plt.subplot(122)

plt.title('Count of non-defaulters (T=0) based on CHANNEL_TYPE in previous_application')

plt.ylabel('Count of non-defaulters');plt.xlabel('CHANNEL_TYPE [FIG: 32]')

df_previous_application_T_0['CHANNEL_TYPE'].value_counts().plot(kind='bar')

plt.show()


# *Analysis from 3.9.3:*
# 
# * FIG:31, shows that the maximum of the defaulters are in category 'Credit and cash offices'
# 
# * FIG:32, shows that the maximum of non-defaulters are in category 'country-wide'
# 
# *Conclusion:* We can frame a pattern by using the variable 'CHANNEL_TYPE'. That is, if the CHANNEL_TYPE is 'Credit and cash offices' the person who applies for the loan might become a defaulter. In other hand, if the CHANNEL_TYPE is ''country-wide'' the person who applies for the loan might become a non-defaulter. By knowing this we can approve the loan for the non-defaulters and reject for people who tend to be defaulters. 

# ### 3.9.4 Univariate analysis on selected unordered-categorical variable ('NAME_PORTFOLIO') in previous_application datasets for Target = 1&0

# In[ ]:


# Plotting the count and rank varaibles for NAME_PORTFOLIO in previous_application datasets for Target = 1&0

plt.figure(figsize=(18,6))

plt.subplot(121)

plt.title('Count of defaulters (T=1) based on NAME_PORTFOLIO in previous_application')

plt.ylabel('Count of defaulters');plt.xlabel('NAME_PORTFOLIO [FIG: 33]')

df_previous_application_T_1['NAME_PORTFOLIO'].value_counts().plot(kind='bar')

plt.subplot(122)

plt.title('Count of non-defaulters (T=0) based on NAME_PORTFOLIO in previous_application')

plt.ylabel('Count of non-defaulters');plt.xlabel('NAME_PORTFOLIO [FIG: 34]')

df_previous_application_T_0['NAME_PORTFOLIO'].value_counts().plot(kind='bar')

plt.show()


# *Analysis from 3.9.4:*
# 
# * FIG:33, shows that the maximum of the defaulters are in category 'Cash'
# 
# * FIG:34, shows that the maximum of non-defaulters are in category 'POS'
# 
# *Conclusion:* We can frame a pattern by using the variable 'NAME_PORTFOLIO'. That is, if the NAME_PORTFOLIO is 'Cash' the person who applies for the loan might become a defaulter. In other hand, if the NAME_PORTFOLIO is 'POS' the person who applies for the loan might become a non-defaulter. By knowing this we can approve the loan for the non-defaulters and reject for people who tend to be defaulters. 

# ## 3.10 Univariate Analysis on Numerical variables that are common in both datasets for target 1&0:

# ### 3.10.1 Univariate Analysis on Numerical variables (AMT_CREDIT) that are common in both datasets with target = 1&0

# In[ ]:


# Plotting the Mean for varible AMT_CREDIT in both datasets for Target = 1&0

num_colm = 'AMT_CREDIT'

mean_prev_T_0 = df_previous_application_T_0[num_colm].mean()

mean_prev_T_1 = df_previous_application_T_1[num_colm].mean()

mean_appl_T_0 = df_application_data_T_0[num_colm].mean()

mean_appl_T_1 = df_application_data_T_1[num_colm].mean()

x = ['AMT_CREDIT_mean_T_0_in_prev_data','AMT_CREDIT_mean_T_1_in_prev_data','AMT_CREDIT_mean_T_0_in_appl_data','AMT_CREDIT_mean_T_1_in_appl_data']

y = [mean_prev_T_0,mean_prev_T_1,mean_appl_T_0,mean_appl_T_1]

plt.figure(figsize=(16,6))

plt.ylabel('AMT_CREDIT_Mean')

plt.title('Mean of "AMT_CREDIT" in both datasets application_data (appl_data) & previous_application (prev_data) for target = 1&0 [FIG:35]')

plt.bar(x,y,width=0.4)

plt.show()


# *Analysis from 3.10.1:*
# 
# * FIG:35, shows that in general (mean) the value of AMT_CREDIT is high for defaulters (T-1) in both the data sets (prev_data & (appl_data).
# 
# *Conclusion:* We have to extra careful in providing loan with applicats holding more AMT_CREDIT because they may most likely to become a defaulter

# ## 3.11 Segmented Univariate Analysis on Numerical variables in application_data datasets:

# ### 3.11.1 Segmented Univariate Analysis on Numerical variable (AMT_INCOME_TOTAL) in application_data datasets for Target 1&0 on the basis of Gender:

# In[ ]:


# Plotting the Mean for varible AMT_INCOME_TOTAL in both datasets for Target = 1&0 on the basis of Gender in application_data dataset

num_colm = 'AMT_CREDIT'

mean_appl_T_0_M = df_application_data_T_0[df_application_data_T_0.CODE_GENDER=='M'][num_colm].mean()

mean_appl_T_0_F = df_application_data_T_0[df_application_data_T_0.CODE_GENDER=='F'][num_colm].mean()

mean_appl_T_1_M = df_application_data_T_1[df_application_data_T_1.CODE_GENDER=='M'][num_colm].mean()

mean_appl_T_1_F = df_application_data_T_1[df_application_data_T_1.CODE_GENDER=='F'][num_colm].mean()

x_male = ['AMT_INCOME_TOTAL_mean_T_0_Male','AMT_INCOME_TOTAL_mean_T_1_Male']

y_male = [mean_appl_T_0_M,mean_appl_T_1_M]

x_Female = ['AMT_INCOME_TOTAL_mean_T_0_Female','AMT_INCOME_TOTAL_mean_T_1_Female']

y_Female = [mean_appl_T_0_F,mean_appl_T_1_F]

plt.figure(figsize=(18,6))

plt.subplot(121)

plt.ylabel('AMT_INCOME_TOTAL_Mean')

plt.xlabel('[FIG:36]')

plt.title('Mean of "AMT_CREDIT" in datasets appl_data & prev_data for target = 1&0 For Male')

plt.bar(x_male,y_male)

plt.subplot(122)

plt.ylabel('AMT_INCOME_TOTAL_Mean')

plt.xlabel('[FIG:37]')

plt.title('Mean of "AMT_CREDIT" in datasets appl_data & prev_data for target = 1&0 For Female')

plt.bar(x_Female,y_Female)

plt.show()


# *Analysis from 3.11.1:*
# 
# * FIG:36, shows that in general (mean) among male the defaulters (T-1) do have less income compared to non-defaulters (T-0).
# 
# * FIG:37, shows that in general (mean) among Female the defaulters (T-1) do have less income compared to non-defaulters (T-0).
# 
# *Conclusion:* We have to extra careful in providing loan for applications holding less income since they may most likely to become defaulters.

# ### 3.11.2 Segmented Univariate Analysis on Numerical variable (AMT_ANNUITY) in application_data datasets for Target 1&0 on the basis of Gender:

# In[ ]:


# Plotting the Mean for varible AMT_ANNUITY in both datasets for Target = 1&0 on the basis of Gender in application_data dataset

num_colm = 'AMT_ANNUITY'

mean_appl_T_0_M = df_application_data_T_0[df_application_data_T_0.CODE_GENDER=='M'][num_colm].mean()

mean_appl_T_0_F = df_application_data_T_0[df_application_data_T_0.CODE_GENDER=='F'][num_colm].mean()

mean_appl_T_1_M = df_application_data_T_1[df_application_data_T_1.CODE_GENDER=='M'][num_colm].mean()

mean_appl_T_1_F = df_application_data_T_1[df_application_data_T_1.CODE_GENDER=='F'][num_colm].mean()

x_male = ['AMT_ANNUITY_mean_T_0_Male','AMT_ANNUITY_mean_T_1_Male']

y_male = [mean_appl_T_0_M,mean_appl_T_1_M]

x_Female = ['AMT_ANNUITY_mean_T_0_Female','AMT_ANNUITY_mean_T_1_Female']

y_Female = [mean_appl_T_0_F,mean_appl_T_1_F]

plt.figure(figsize=(18,6))

plt.subplot(121)

plt.ylabel('AMT_ANNUITY_Mean')

plt.xlabel('[FIG:38]')

plt.title('Mean of "AMT_ANNUITY" in application_data dataset for target = 1&0 For Male')

plt.bar(x_male,y_male)

plt.subplot(122)

plt.ylabel('AMT_ANNUITY_Mean')

plt.xlabel('[FIG:39]')

plt.title('Mean of "AMT_ANNUITY" in application_data dataset for target = 1&0 For FeMale')

plt.bar(x_Female,y_Female)

plt.show()


# *Analysis from 3.11.2:*
# 
# * FIG:38, shows that in general (mean) among male the defaulters (T-1) do have less AMT_ANNUITY compared to non-defaulters (T-0).
# 
# * FIG:39, shows that in general (mean) among Female the defaulters (T-1) do have less AMT_ANNUITY compared to non-defaulters (T-0).
# 
# *Conclusion:* We have to extra careful in providing loan for applications holding less AMT_ANNUITY since they may most likely to become defaulters.

# ### 3.11.3 Segmented Univariate Analysis on Numerical variable (AMT_GOODS_PRICE) in application_data datasets for Target 1&0 on the basis of Gender:

# In[ ]:


# Plotting the Mean for varible AMT_GOODS_PRICE in both datasets for Target = 1&0 on the basis of Gender in application_data dataset

num_colm = 'AMT_GOODS_PRICE'

mean_appl_T_0_M = df_application_data_T_0[df_application_data_T_0.CODE_GENDER=='M'][num_colm].mean()

mean_appl_T_0_F = df_application_data_T_0[df_application_data_T_0.CODE_GENDER=='F'][num_colm].mean()

mean_appl_T_1_M = df_application_data_T_1[df_application_data_T_1.CODE_GENDER=='M'][num_colm].mean()

mean_appl_T_1_F = df_application_data_T_1[df_application_data_T_1.CODE_GENDER=='F'][num_colm].mean()

x_male = ['AMT_GOODS_PRICE_mean_T_0_Male','AMT_GOODS_PRICE_mean_T_1_Male']

y_male = [mean_appl_T_0_M,mean_appl_T_1_M]

x_Female = ['AMT_GOODS_PRICE_mean_T_0_Female','AMT_GOODS_PRICE_mean_T_1_Female']

y_Female = [mean_appl_T_0_F,mean_appl_T_1_F]

plt.figure(figsize=(18,6))

plt.subplot(121)

plt.ylabel('AMT_GOODS_PRICE_Mean')

plt.xlabel('[FIG:40]')

plt.title('Mean of "AMT_GOODS_PRICE" in dataset: appl_data for target = 1&0 For Male')

plt.bar(x_male,y_male)

plt.subplot(122)

plt.ylabel('AMT_GOODS_PRICE_Mean')

plt.xlabel('[FIG:41]')

plt.title('Mean of "AMT_GOODS_PRICE" in dataset: appl_data for target = 1&0 For FeMale')

plt.bar(x_Female,y_Female)

plt.show()


# *Analysis from 3.11.3:*
# 
# * FIG:38, shows that in general (mean) among male the defaulters (T-1) do have less AMT_GOODS_PRICE compared to non-defaulters (T-0).
# 
# * FIG:39, shows that in general (mean) among Female the defaulters (T-1) do have less AMT_GOODS_PRICE compared to non-defaulters (T-0).
# 
# *Conclusion:* We have to extra careful in providing loan for applications holding less AMT_GOODS_PRICE since they may most likely to become defaulters.

# ## 3.12 Univariate Analysis on Numerical variables in Previous_application datasets for target 1&0:

# ### 3.12.1 Univariate Analysis on Numerical variable (AMT_APPLICATION) in previous_application with target = 1&0

# In[ ]:


# Plotting the Mean for varible AMT_APPLICATION in previous_application with target = 1&0

num_colm = 'AMT_APPLICATION'

mean_prev_T_0 = df_previous_application_T_0[num_colm].mean()

mean_prev_T_1 = df_previous_application_T_1[num_colm].mean()

x = ['AMT_APPLICATION_mean_T_0','AMT_APPLICATION_mean_T_1']

y = [mean_prev_T_0,mean_prev_T_1]

plt.figure(figsize=(14,6))

plt.ylabel('AMT_APPLICATION_Mean')

plt.title('Mean of "AMT_APPLICATION" in  previous_application for target = 1&0 [FIG:42]')

plt.bar(x,y)

plt.show()


# *Analysis from 3.12.1:*
# 
# * FIG:42, shows that in general (mean) the defaulters (T-1) possess more in AMT_APPLICATION compared to non-defaulters (T-0).
# 
# *Conclusion:* We have to extra careful in providing loan for applications holding more AMT_APPLICATION since they may most likely to become defaulters.

# ## 3.13 BI-Variate Analysis for numerical variables in Previous_Application for both T=0&1

# ### 3.13.1 BI-Variate Analysis for numerical variables ('AMT_CREDIT','AMT_APPLICATION') in Previous_Application for both T=0&1

# In[ ]:


bivaritate_variables = ['AMT_CREDIT','AMT_APPLICATION']

x_t_0 = df_previous_application_T_0[bivaritate_variables[0]].values

x_t_1 = df_previous_application_T_1[bivaritate_variables[0]].values

y_t_0 = df_previous_application_T_0[bivaritate_variables[1]].values

y_t_1 = df_previous_application_T_1[bivaritate_variables[1]].values

############# Function to viualise the Linear regression line  ################

from numpy import *
import matplotlib.pyplot as plt

def plot_best_fit(intercept, slope):
    axes = plt.gca()
    x_vals = array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, 'r-')

############# Utilising Linear Regression Algorithm from Sklearn #############

from sklearn.linear_model import LinearRegression
    
def Regression(X,Y):   
    regr = LinearRegression()
    regr.fit(X,Y)
    return regr

######################### Main Code To Plot the Graph ##########################

plt.figure(figsize=(14,16))

plt.subplot(211)

plt.scatter(x_t_0,y_t_0)

regr = Regression(x_t_0.reshape(-1,1) ,y_t_0.reshape(-1,1))

plot_best_fit((regr.intercept_)[0],(regr.coef_)[0][0])

plt.title('Bi-Variate Analysis on AMT_CREDIT & AMT_APPLICATION For T=0 in previous_data(Fig 43)')

plt.xlabel('AMT_CREDIT')

plt.ylabel('AMT_APPLICATION')

plt.subplot(212)

plt.scatter(x_t_1,y_t_1)

regr = Regression(x_t_1.reshape(-1,1) ,y_t_1.reshape(-1,1))

plot_best_fit((regr.intercept_)[0],(regr.coef_)[0][0])

plt.title('Bi-Variate Analysis on AMT_CREDIT & AMT_APPLICATION For T=1 in previous_data(Fig 44)')

plt.xlabel('AMT_CREDIT')

plt.ylabel('AMT_APPLICATION')

plt.show()

print('Actual Co-relation value between ',bivaritate_variables,'for non-default applicants (t-0) \n in previous_application dataset is as shown in below co-relation matrix: \n\n',df_previous_application_T_0[bivaritate_variables].corr())

print('\n\nActual Co-relation value between ',bivaritate_variables,'for default applicants (t-1) \n in previous_application dataset is as shown in below co-relation matrix: \n\n',df_previous_application_T_1[bivaritate_variables].corr())


# *Analysis from 3.13.1:*
# 
# * FIG:43, shows that AMT_CREDIT & AMT_APPLICATION are positively co-related (an upward trend shows that it is possitively co-related) in case of non-defaulters (T-0)
# 
# * FIG:44, shows that AMT_CREDIT & AMT_APPLICATION are positively co-related (an upward trend shows that it is possitively co-related) in case of defaulters (T-1)
# 
# *Conclusion:* In general, AMT_CREDIT and AMT_APPLICATION are both positively co-related both in case of defaulters and non-defaulters. 

# ## 3.14 BI-Variate Analysis for numerical variables in Application_data for both T=0&1

# ### 13.14.1 BI-Variate Analysis for numerical variables ('AMT_CREDIT','AMT_INCOME_TOTAL') in Application_data for both T=0&1

# In[ ]:


bivaritate_variables = ['AMT_CREDIT','AMT_INCOME_TOTAL']

x_t_0 = df_application_data_T_0[bivaritate_variables[0]].values

x_t_1 = df_application_data_T_1[bivaritate_variables[0]].values

y_t_0 = df_application_data_T_0[bivaritate_variables[1]].values

y_t_1 = df_application_data_T_1[bivaritate_variables[1]].values

############# Function to viualise the Linear regression line  ################

from numpy import *
import matplotlib.pyplot as plt

def plot_best_fit(intercept, slope):
    axes = plt.gca()
    x_vals = array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, 'r-')

############# Utilising Linear Regression Algorithm from Sklearn #############

from sklearn.linear_model import LinearRegression
    
def Regression(X,Y):   
    regr = LinearRegression()
    regr.fit(X,Y)
    return regr

######################### Main Code To Plot the Graph ##########################

plt.figure(figsize=(14,14))

plt.subplot(211)

plt.scatter(x_t_0,y_t_0)

regr = Regression(x_t_0.reshape(-1,1) ,y_t_0.reshape(-1,1))

plot_best_fit((regr.intercept_)[0],(regr.coef_)[0][0])

plt.title('Bi-Variate Analysis on AMT_CREDIT & AMT_INCOME_TOTAL For T=0 in previous_data(Fig 43)')

plt.xlabel('AMT_CREDIT')

plt.ylabel('AMT_INCOME_TOTAL')

plt.subplot(212)

plt.scatter(x_t_1,y_t_1)

regr = Regression(x_t_1.reshape(-1,1) ,y_t_1.reshape(-1,1))

plot_best_fit((regr.intercept_)[0],(regr.coef_)[0][0])

plt.title('Bi-Variate Analysis on AMT_CREDIT & AMT_INCOME_TOTAL For T=1 in previous_data(Fig 44)')

plt.xlabel('AMT_CREDIT')

plt.ylabel('AMT_INCOME_TOTAL')

plt.show()

print('Actual Co-relation value between ',bivaritate_variables,'for non-default applicants (t-0) \n in application_data dataset is as shown in below co-relation matrix: \n\n',df_application_data_T_0[bivaritate_variables].corr())

print('\n\nActual Co-relation value between ',bivaritate_variables,'for default applicants (t-1) \n in application_data dataset is as shown in below co-relation matrix: \n\n',df_application_data_T_1[bivaritate_variables].corr())


# *Analysis from 3.14.1:*
# 
# * FIG:43, shows that AMT_CREDIT & AMT_INCOME_TOTAL are positively co-related (a slight upward trend shows that it is possitively co-related) in case of non-defaulters (T-0)
# 
# * FIG:44, shows that AMT_CREDIT & AMT_INCOME_TOTAL are positively co-related (a slight upward trend shows that it is possitively co-related) in case of defaulters (T-1)
# 
# *Conclusion:* In general, AMT_CREDIT and AMT_INCOME_TOTAL are both positively co-related both in case of defaulters and non-defaulters. 

# ### 13.14.2 BI-Variate Analysis for numerical variables ('AMT_CREDIT','AMT_ANNUITY') in Application_data for both T=0&1

# In[ ]:


bivaritate_variables = ['AMT_CREDIT','AMT_ANNUITY']

x_t_0 = df_application_data_T_0[bivaritate_variables[0]].values

x_t_1 = df_application_data_T_1[bivaritate_variables[0]].values

y_t_0 = df_application_data_T_0[bivaritate_variables[1]].values

y_t_1 = df_application_data_T_1[bivaritate_variables[1]].values

############# Function to viualise the Linear regression line  ################

from numpy import *
import matplotlib.pyplot as plt

def plot_best_fit(intercept, slope):
    axes = plt.gca()
    x_vals = array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, 'r-')

############# Utilising Linear Regression Algorithm from Sklearn #############

from sklearn.linear_model import LinearRegression
    
def Regression(X,Y):   
    regr = LinearRegression()
    regr.fit(X,Y)
    return regr

######################### Main Code To Plot the Graph ##########################

plt.figure(figsize=(14,14))

plt.subplot(211)

plt.scatter(x_t_0,y_t_0)

regr = Regression(x_t_0.reshape(-1,1) ,y_t_0.reshape(-1,1))

plot_best_fit((regr.intercept_)[0],(regr.coef_)[0][0])

plt.title('Bi-Variate Analysis on AMT_CREDIT & AMT_ANNUITY For T=0 in previous_data(Fig 45)')

plt.xlabel('AMT_CREDIT')

plt.ylabel('AMT_ANNUITY')

plt.subplot(212)

plt.scatter(x_t_1,y_t_1)

regr = Regression(x_t_1.reshape(-1,1) ,y_t_1.reshape(-1,1))

plot_best_fit((regr.intercept_)[0],(regr.coef_)[0][0])

plt.title('Bi-Variate Analysis on AMT_CREDIT & AMT_ANNUITY For T=1 in previous_data(Fig 46)')

plt.xlabel('AMT_CREDIT')

plt.ylabel('AMT_ANNUITY')

plt.show()

print('Actual Co-relation value between ',bivaritate_variables,'for non-default applicants (t-0) \n in application_data dataset is as shown in below co-relation matrix: \n\n',df_application_data_T_0[bivaritate_variables].corr())

print('\n\nActual Co-relation value between ',bivaritate_variables,'for default applicants (t-1) \n in application_data dataset is as shown in below co-relation matrix: \n\n',df_application_data_T_1[bivaritate_variables].corr())


# *Analysis from 3.14.2:*
# 
# * FIG:45, shows that AMT_CREDIT & AMT_ANNUITY are positively co-related (an upward trend shows that it is possitively co-related) in case of non-defaulters (T-0)
# 
# * FIG:46, shows that AMT_CREDIT & AMT_ANNUITY are positively co-related (an upward trend shows that it is possitively co-related) in case of defaulters (T-1)
# 
# *Conclusion:* In general, AMT_CREDIT and AMT_ANNUITY are both positively co-related both in case of defaulters and non-defaulters. 

# ### 13.14.3 BI-Variate Analysis for numerical variables ('AMT_CREDIT','AMT_GOODS_PRICE') in Application_data for both T=0&1

# In[ ]:


bivaritate_variables = ['AMT_CREDIT','AMT_GOODS_PRICE']

x_t_0 = df_application_data_T_0[bivaritate_variables[0]].values

x_t_1 = df_application_data_T_1[bivaritate_variables[0]].values

y_t_0 = df_application_data_T_0[bivaritate_variables[1]].values

y_t_1 = df_application_data_T_1[bivaritate_variables[1]].values

############# Function to viualise the Linear regression line  ################

from numpy import *
import matplotlib.pyplot as plt

def plot_best_fit(intercept, slope):
    axes = plt.gca()
    x_vals = array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, 'r-')

############# Utilising Linear Regression Algorithm from Sklearn #############

from sklearn.linear_model import LinearRegression
    
def Regression(X,Y):   
    regr = LinearRegression()
    regr.fit(X,Y)
    return regr

######################### Main Code To Plot the Graph ##########################

plt.figure(figsize=(14,14))

plt.subplot(211)

plt.scatter(x_t_0,y_t_0)

regr = Regression(x_t_0.reshape(-1,1) ,y_t_0.reshape(-1,1))

plot_best_fit((regr.intercept_)[0],(regr.coef_)[0][0])

plt.title('Bi-Variate Analysis on AMT_CREDIT & AMT_GOODS_PRICE For T=0 in previous_data(Fig 47)')

plt.xlabel('AMT_CREDIT')

plt.ylabel('AMT_GOODS_PRICE')

plt.subplot(212)

plt.scatter(x_t_1,y_t_1)

regr = Regression(x_t_1.reshape(-1,1) ,y_t_1.reshape(-1,1))

plot_best_fit((regr.intercept_)[0],(regr.coef_)[0][0])

plt.title('Bi-Variate Analysis on AMT_CREDIT & AMT_ANNUITY For T=1 in previous_data(Fig 48)')

plt.xlabel('AMT_CREDIT')

plt.ylabel('AMT_GOODS_PRICE')

plt.show()

print('Actual Co-relation value between ',bivaritate_variables,'for non-default applicants (t-0) \n in application_data dataset is as shown in below co-relation matrix: \n\n',df_application_data_T_0[bivaritate_variables].corr())

print('\n\nActual Co-relation value between ',bivaritate_variables,'for default applicants (t-1) \n in application_data dataset is as shown in below co-relation matrix: \n\n',df_application_data_T_1[bivaritate_variables].corr())


# *Analysis from 3.14.3:*
# 
# * FIG:47, shows that AMT_CREDIT & AMT_GOODS_PRICE are positively co-related (an upward trend shows that it is possitively co-related) in case of non-defaulters (T-0)
# 
# * FIG:48, shows that AMT_CREDIT & AMT_GOODS_PRICE are positively co-related (an upward trend shows that it is possitively co-related) in case of defaulters (T-1)
# 
# *Conclusion:* In general, AMT_CREDIT and AMT_GOODS_PRICE are both positively co-related both in case of defaulters and non-defaulters. 

# # 4. Final Report on the Analysis from both the Datasets:

# ## 4.1 Evidence from points 3.9.1 to 3.9.4 (Categorical variables)

# * It is an evident that if the categorical variables [PRODUCT_COMBINATION, NAME_YIELD_GROUP, CHANNEL_TYPE, NAME_PORTFOLIO] contains the categories [cash X-sell: low, XNA, Credit and cash offices, Cash] then the applicant is more likely to become a defaulter and the loan providing company should be more careful in providing loan for these applicants. 
# 
# * On the other hand, if the above mentioned categorical variables contains the categories [POS household with interest, middle, country-wide, POS] then the is not likely to become a defaulter and the loan providing company should not be cancelling the application in providing the loan for these applicants.

# ## 4.2 Evidence from points 3.10 to 3.12 (Numerical variables)

# * For male applications income (AMT_INCOME_TOTAL) plays a role in becoming a default or not. Higher the income it is likely to become defaulter. So the loan lending company should be careful in providing loans for male applicats who has less income.
# 
# * In considering the numerical variable AMT_ANNUITY, in both male and female, if the value of AMT_ANNUITY is less then the applicants are most likely to become defaulters, so the lending company should be more careful in lending loans to these applicants.
# 
# * In considering the numerical variable AMT_GOODS_PRICE, in both male and female, if the value of AMT_GOODS_PRICE is less then the applicants are most likely to become defaulters, so the lending company should be more careful in lending loans to these applicants.
# 
# * In considering the numerical variable AMT_APPLICATION, in general, if the value of AMT_APPLICATION is high then the applicants are most likely to become defaulters, so the lending company should be more careful in lending loans to these applicants.

# ## 4.3 Evidence from points 3.13 to 3.14 (Numerical variables)

# We have taken a numerical variable (AMT_CREDIT) which is common in both datasets and tried finding the co-relation with each variables mentioned in above point (4.2) and found that all variables are positively co-related with AMT_CREDIT. In other words, each and every numerical variables (AMT_CREDIT, AMT_INCOME_TOTAL, AMT_ANNUITY, AMT_GOODS_PRICE, AMT_APPLICATION) that we have discussed or positively co-related with each other.
