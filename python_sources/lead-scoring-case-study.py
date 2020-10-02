#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Supress Warnings

import warnings
warnings.filterwarnings('ignore')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#encoding
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


#plots

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

#for learning models
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

#metrics
from sklearn import metrics




#for data set object columns evaluation
import collections

pd.set_option('display.max_columns', 1000)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Business Problem
# 

# An education company named X Education sells online courses to industry professionals. The typical lead conversion rate at X education is around 30%. 
# 1. Improve Target Lead Conversion using past data of conversion
# 1. Implement Logistic Regression model and assigned Predicted Lead Score (0..100)
# 1. Find Hot Leads(Higher Score) and the features influencing Hot Leads

# # Solution Steps
# 1. Read Data set
# 1. Examine Data set Properties
# 1. Inspecting the Dataframe
# 1. Data Preparation
#     1. Rename Columns
# 	1. CATEGORICAL Features
# 		1. Fixing Features(Columns) Data Types
# 		1. Replace Select wtih np.nan.
# 		1. Drop less frequently present features(0.1% only)
# 		1. Lets manage all individual changes that we require based on the feature keys
# 		1. Feature : Last Activity analysis
# 		1. Feature: Specialization Analysis
# 		1. Feature: How_did_you_hear_about_X_Education analysis
# 		1. Drop Features with >70% Null
# 	1. Numerical Features - Data Preparation
# 		1. Numerical Feature Analysis
# 		1. Imbalance Score
# 		1. View and Handle Missing Data in Numerical Features
# 1. Encoding
#     1. Dummy Encoding
#     1. Frequency Encoding
# 1. Model based on Dummy Encoded Data set 
#     1. Test-Train Split
#     1. Feature Scaling and Initial RFE
#     1. Feature Selection Using RFE
#         1. Model 1 Outcome
# 		1. Model 2 Outcome
# 		1. Model 3 Outcome
# 		1. Model 4 Outcome
# 		1. Model 5 Outcome
# 		1. Checking VIFs
# 		1. Model 6 Outcome
#     1. Model Evaluation
# 		1. Prediction and Lead Score assignment
# 		1. Metrics based on Confusion Matrix
# 		1. ROC and AUC Metrics and Cut off selection
# 		1. Redo Prediction based on Selected Cut off
# 		1. Redo Metrics based on Confusion Matrix
# 		1. Metrics based on Precision and Recall
# 		1. Precision and recall tradeoff
# 		1. Analysis of Metrics
#     1. Making predictions on the test set and Evaluation
#     1. Analysis of Metrics
#     1. Analysis of Selected Features
#         1. Correlation Analysis
#         1. Numeric Features
#         1. Categorical Dummy Features
#         1. View the Co-efficient of the Selected Model
#         1. Top Positively influencing Features
#         1. Top Negatively influencing Features
# 1. Model based on Frequency Encoded Data set 
#     1. Test-Train Split
#     1. Feature Scaling and Initial RFE
# 		1. Feature Selection Using RFE
#         1. Model 7 Outcome
# 		1. Model 8 Outcome
# 		1. Checking VIFs
# 		1. Model 9 Outcome
# 		1. Model 10 Outcome
#     1. Model Evaluation
# 		1. Prediction and Lead Score assignment
# 		1. Metrics based on Confusion Matrix
# 		1. ROC and AUC Metrics and Cut off selection
# 		1. Redo Prediction based on Selected Cut off
# 		1. Redo Metrics based on Confusion Matrix, Precision Recall
# 		1. Precision and recall tradeoff
# 		1. Analysis of Metrics
#     1. Making predictions on the test set and Evaluation
#     1. Analysis of Metrics
#     1. Analysis of Selected Features
#         1. Correlation Analysis
#         1. View the Co-efficient of the Selected Model
#         1. Top Positively influencing Features
#         1. Top Negatively influencing Features

# ## Read Data Set

# In[ ]:


#Read the data
df_lead_score=pd.read_csv("../input/Leads.csv")
#df_lead_score=pd.read_csv("Leads.csv")


# ## Examine Data Set properties

# In[ ]:


#examine the data set
print('Samples in the data set:', df_lead_score.shape[0])
print('Columns in the data set:', df_lead_score.shape[1])
print('\n\n Columns Data Type:')
print(df_lead_score.info())
print('\n\n Nulls in Columns:',df_lead_score.isnull().sum().sum())


# In[ ]:


#function to get missing values
def getMissingPercentageFeature(col,input_df):
    null_counts = (input_df[col].isnull().sum()/len(input_df[col]))
    print("\tMissing values for Feature:",col,"-",round(null_counts*100,4))


# In[ ]:


#function to get missing values
def getMissingPercentage(input_df):
    null_counts = (input_df.isnull().sum()/len(input_df)).sort_values(ascending=False)
    null_counts=null_counts[null_counts!=0]
    null_counts=round(null_counts*100,4)
    #print(null_counts)
    plt.figure(figsize=(16,8))
    plt.xticks(np.arange(len(null_counts))+0.5,null_counts.index,rotation='vertical')
    plt.ylabel('Fraction of Rows with Missing Columns')
    plt.bar(np.arange(len(null_counts)),null_counts)
    return null_counts


# In[ ]:


#function to drop columns
def dropcolumns(df,percentage):
    df_missing_percentage=getMissingPercentage(df)
    columns_to_be_dropped=df_missing_percentage[df_missing_percentage>=percentage]
    result_df=df.drop(columns=columns_to_be_dropped.index,axis=1)
    return result_df,list(columns_to_be_dropped.index)


# In[ ]:


#view the null % for each column based on number of rows in the data set
getMissingPercentage(df_lead_score)


# # Data Preparation

# In[ ]:


#Create array(index set) for object features and non-object features
def classifyfeatures(df):
    object_features = df.select_dtypes(include = ["object"]).columns
    non_object_features = df.select_dtypes(exclude = ["object"]).columns
    return(list(object_features),list(non_object_features))


# In[ ]:


#function to display features in a numbered way
def printFeatures(featureslist):
    print( "Number of features:",len(featureslist))
    for i,col in enumerate(featureslist):
        print("Feature:",i+1,col)


# In[ ]:


#function to display features in a numbered way
def printCountOfFeatures(featureslist,featurelistname):
    print( "Number of features for",featurelistname,":",len(featureslist))


# ### Rename Columns

# In[ ]:


#function to replace feature keys characters for python usage
def replacecolumnkeys(col,df):
    df[col]= df[col].str.replace(' ', '_')
    df[col]= df[col].str.replace(',', '_')
    df[col]= df[col].str.replace('-', '_')
    df[col]= df[col].str.replace('/', '_')
    df[col]= df[col].str.replace(':', '_')
    return df


# In[ ]:


#assign list of object and non-object features / columns for the data set
obj_features_lead_score, non_obj_features_lead_score=classifyfeatures(df_lead_score)

#before cleaning
printCountOfFeatures(obj_features_lead_score,'object_features')
printCountOfFeatures(non_obj_features_lead_score,'non_object_features ')

#print categorical features
print( "\n\nCategorical features:\n")
printFeatures(obj_features_lead_score)
#print Non Object features
print( "\n\nNon Object features:\n")
printFeatures(non_obj_features_lead_score)


# ### CATEGORICAL Features
# #### Fixing Features(Columns) Data Types

# In[ ]:


#copy data set for cleaning
df_lead_score_cleaned=df_lead_score.copy()

#Fixing column names
# Replace space with underscore
df_lead_score_cleaned.columns = df_lead_score_cleaned.columns.str.replace(" ", "_")

# Reassign feature groups after fixing of Categorical features
obj_features_lead_score, non_obj_features_lead_score=classifyfeatures(df_lead_score_cleaned)

#lets remove Prospect_ID,Lead_Number,Converted from the object features list for handling perspective
obj_features_lead_score.remove('Prospect_ID')
non_obj_features_lead_score.remove('Lead_Number')
non_obj_features_lead_score.remove('Converted')


# In[ ]:


#function to print unique keys and values of a feature
def uniquecolumnkeys(columns,df):
        for i,col in enumerate(columns):
            #print count having less than 0
            print("\nFeature:",i+1,col)
            print(".....................................")
            print("\tCategories:",df[col].unique())
            freq = collections.Counter(df[col])
            print("\tMost Common Key:",freq.most_common()[0][0])
            for key, value in freq.items(): 
                print("\t",key," :", value)


# In[ ]:


#lets view the frequency for each of the object
uniquecolumnkeys(obj_features_lead_score,df_lead_score_cleaned)


# #### Replace Select wtih np.nan 
# 1. Select means that user hasn't given any value for a specific question / information parameter, so we can consider it as not keyed in value(nan)

# In[ ]:


# lets replace value Select with nan, Select means that user hasn't given any value for a specific question / information parameter
df_lead_score_cleaned[obj_features_lead_score]=df_lead_score_cleaned[obj_features_lead_score].replace('Select',np.nan)


# #### Drop less frequently present features(0.1% only)
# 1. Analyse the frequency of categorical columns, and drop the very less frequently used features (0.1% only)
# 1. i.e., 14/9240, which is the maximum frequency that we are dropping from the below list

# In[ ]:


#Features with Yes/No values
#Feature: 3 Do_Not_Email - we can keep
#Feature: 4 Do_Not_Call, only 2 values are no, so we can drop it
#Feature: 11 Search , only 14 values are no, so we can drop it
#Feature: 12 Magazine all are no - we can drop it as well
#Feature: 13 Newspaper_Article, only 2 values are no, so we can drop it
#Feature: 14 X_Education_Forums , only 1 is yes, so we can drop it
#Feature: 15 Newspaper , only 1 is yes, so we can drop it
#Feature: 16 Digital_Advertisement all are no, so we can drop it
#Feature: 17 Through_Recommendations , only 7 yes, so we can drop it
#Feature: 18 Receive_More_Updates_About_Our_Courses, all are no, so we can drop it
#Feature: 21 Update_me_on_Supply_Chain_Content , all are no, so we can drop it
#Feature: 22 Get_updates_on_DM_Content , all are no, so we can drop it
#Feature: 27 I_agree_to_pay_the_amount_through_cheque , all are no, so we can drop it
features_dropped=['Do_Not_Email','Do_Not_Call','Search','Magazine','Newspaper_Article','X_Education_Forums','Newspaper','Digital_Advertisement','Digital_Advertisement',
                 'Through_Recommendations','Receive_More_Updates_About_Our_Courses','Update_me_on_Supply_Chain_Content','Get_updates_on_DM_Content',
                 'I_agree_to_pay_the_amount_through_cheque']
df_lead_score_cleaned=df_lead_score_cleaned.drop(columns=features_dropped,axis=1)


# #### Lets manage all individual changes that we require based on the feature keys
# 

# In[ ]:


#Feature: 1 Lead_Origin , no nans


# In[ ]:


# function for univariate / bivariate and ratio analysis
from numpy import nan
def TargetvsColFrequency(col1,col2,df):
    freq = collections.Counter(df[col1])
    del freq[nan]
    fig, ax =plt.subplots(figsize=(10,5))
    sns.set_context({"figure.figsize": (10, 4)})
    ax = sns.countplot(y=col1,data=df,hue=col2,order = df[col1].value_counts().index)
    df_final1=pd.DataFrame()
    df_final0=pd.DataFrame()
    df_temp1=pd.DataFrame()
    df_temp0=pd.DataFrame()
    rowvalue1=0
    rowvalue0=0
    for key, value in freq.items(): 
        if key=='nan':
            continue
        df_plot=pd.DataFrame(df[df[col1]==key].groupby(col2)[col1].count())
        df_plot['RATIO']=df_plot/df_plot.sum()      
        title=str(key)+' in '+str(col1)
        print("\n",title,"\n............................\n",df_plot)
        df_plot.reset_index(inplace=True)
        fig, ax =plt.subplots(figsize=(5,5))
        ax.set_title(title)
        sns.barplot(y='RATIO',x=col2,data=df_plot)
        plt.tight_layout()
        try:
            rowvalue1=df_plot['RATIO'][1] 
            df_temp1[key] =[rowvalue1]
        except KeyError:
            continue
        try:
            rowvalue0=df_plot['RATIO'][0] 
            df_temp0[key] =[rowvalue0]
        except KeyError:
            continue
    df_final1= pd.concat([df_final1,df_temp1], axis=1, sort=False)
    df_final1=df_final1.reset_index()
    df_final1=pd.DataFrame(df_final1.T.iloc[:,-1].sort_values(ascending=False)).rename(columns={'0':'RATIO'})
    df_final1.drop(df_final1.tail(1).index,inplace=True)
    fig, ax =plt.subplots(1,2,figsize=(15,7))
    title=str(col1)+ ' % Of Ratio for Converted Leads'
    ax[0].set_title(title)
    df_final1.iloc[:,-1].plot(kind='barh',ax=ax[0])
    
    df_final0= pd.concat([df_final0,df_temp0], axis=1, sort=False)
    df_final0=df_final0.reset_index()
    df_final0=pd.DataFrame(df_final0.T.iloc[:,-1].sort_values(ascending=False)).rename(columns={'0':'RATIO'})
    df_final0.drop(df_final0.tail(1).index,inplace=True)
    title1=str(col1)+ ' % Of Ratio for Not Converted Leads'
    ax[1].set_title(title1)
    df_final0.iloc[:,-1].plot(kind='barh',ax=ax[1])
    plt.tight_layout()


# In[ ]:


# Feature: 2 Lead_Source
#view before cleaning
uniquecolumnkeys(['Lead_Source'],df_lead_score_cleaned)
TargetvsColFrequency('Lead_Source','Converted',df_lead_score_cleaned)


# ### Lead_source analysis
# 1. Welingak Website has most number of conversion followed by reference category
# 2. Bing and then followed by Facebook  / referal sites/Olark Chat are not leading to enough conversion

# In[ ]:


# Feature: 2 Lead_Source
## 1. replace google with Goolge in Lead Score
df_lead_score_cleaned['Lead_Source']=df_lead_score_cleaned['Lead_Source'].replace('google','Google')
## 2. Replace nan  : 36 with Most Common Key: Google (mode) , as this wouldnt create major bias towards Google as a lead_source
df_lead_score_cleaned['Lead_Source']=df_lead_score_cleaned['Lead_Source'].replace(np.nan,'Google')

#view after cleaning
uniquecolumnkeys(['Lead_Source'],df_lead_score_cleaned)
TargetvsColFrequency('Lead_Source','Converted',df_lead_score_cleaned)


# In[ ]:


#Feature: 5 Last_Activity
#view before cleaning
uniquecolumnkeys(['Last_Activity'],df_lead_score_cleaned)
TargetvsColFrequency('Last_Activity','Converted',df_lead_score_cleaned)


# #### Feature : Last Activity analysis
# 1. Phone conversation and SMS sent are leading to more conversions
# 2. Email bounced ( wrong emails may be) and Olark Chat conversation and Converted to Lead are not leading to enough conversion

# In[ ]:


#Feature: 5 Last_Activity
## 1. Replace nan  : 103 with Most Common Key: Email Opened (mode)
df_lead_score_cleaned['Last_Activity']=df_lead_score_cleaned['Last_Activity'].replace(np.nan,'Email Opened')

#view after cleaning
uniquecolumnkeys(['Last_Activity'],df_lead_score_cleaned)
TargetvsColFrequency('Last_Activity','Converted',df_lead_score_cleaned)


# In[ ]:


#Feature: 6 Country
#view before cleaning
uniquecolumnkeys(['Country'],df_lead_score_cleaned)
TargetvsColFrequency('Country','Converted',df_lead_score_cleaned)


# In[ ]:


#Feature: 6 Country
 ## 1.  Replace nan  : 2461 with unknown as value. We think we shouldnt update it with Most Common Key: India, as this would create high bias over india
df_lead_score_cleaned['Country']=df_lead_score_cleaned['Country'].replace(np.nan,'unknown')

## 2. lets group the countries as India, AMERICAS,EMEA, APAC,UAE 

df_lead_score_cleaned['Country']=df_lead_score_cleaned['Country'].replace(['Russia', 'Kuwait', 'Oman' ,'United Kingdom' ,'Bahrain', 'Ghana','Qatar','Saudi Arabia',
                                                                           'Belgium', 'France','Netherlands','Sweden', 'Nigeria','Germany',
                                                                           'Uganda', 'Kenya', 'Italy' ,'South Africa', 'Tanzania'
                                                                           ,'Liberia','Switzerland' ,'Denmark'],'EMEA')

df_lead_score_cleaned['Country']=df_lead_score_cleaned['Country'].replace(['Singapore','Sri Lanka','China','Hong Kong','Asia/Pacific Region','Malaysia',
                                                                          'Philippines','Bangladesh','Vietnam','Indonesia','Australia'],'APAC')


df_lead_score_cleaned['Country']=df_lead_score_cleaned['Country'].replace(['United States','Canada'],'AMERICAS')


#view after cleaning
uniquecolumnkeys(['Country'],df_lead_score_cleaned)
TargetvsColFrequency('Country','Converted',df_lead_score_cleaned)


# In[ ]:


#Feature: 7 Specialization
#view before cleaning
uniquecolumnkeys(['Specialization'],df_lead_score_cleaned)
TargetvsColFrequency('Specialization','Converted',df_lead_score_cleaned)


# In[ ]:


#Feature: 7 Specialization
## 1. Most Common Key is nan. So lets replace it with a value of Other as a value
df_lead_score_cleaned['Specialization']=df_lead_score_cleaned['Specialization'].replace(np.nan,'Other')

#view after cleaning
uniquecolumnkeys(['Specialization'],df_lead_score_cleaned)
TargetvsColFrequency('Specialization','Converted',df_lead_score_cleaned)


# #### Feature: Specialization Analysis
# 1. 'Other' is the most commonly used. We recommend to capture the industry if user is entering other to specifically know the type of industry. Since this category of leads are not converting well, it is essential to understand the group with additional data in future
# 2. Health care and Banking/investment/insurance leads are well converted

# In[ ]:


#Feature: 8 How_did_you_hear_about_X_Education
#view before cleaning
uniquecolumnkeys(['How_did_you_hear_about_X_Education'],df_lead_score_cleaned)
TargetvsColFrequency('How_did_you_hear_about_X_Education','Converted',df_lead_score_cleaned)


# In[ ]:


#Feature: 8 How_did_you_hear_about_X_Education
## 1. Most Common Key is nan. So lets replace it with a value of Other as a value
df_lead_score_cleaned['How_did_you_hear_about_X_Education']=df_lead_score_cleaned['How_did_you_hear_about_X_Education'].replace(np.nan,'Other')

#view after cleaning
uniquecolumnkeys(['How_did_you_hear_about_X_Education'],df_lead_score_cleaned)
TargetvsColFrequency('How_did_you_hear_about_X_Education','Converted',df_lead_score_cleaned)


# #### Feature: How_did_you_hear_about_X_Education analysis
# 1. SMS sent are leading to more conversions if it is a follow up action as per feature 'Last Activity', but SMS as promotion is not converting well
# 2. Email as promotion leads to higher conversion of leads
# 

# In[ ]:


#Feature: 9 What_is_your_current_occupation

#view before cleaning
uniquecolumnkeys(['What_is_your_current_occupation'],df_lead_score_cleaned)
TargetvsColFrequency('What_is_your_current_occupation','Converted',df_lead_score_cleaned)


# In[ ]:


#Feature: 9 What_is_your_current_occupation
## 1. nan  : 2709  So lets replace it with a value of Other as a value
df_lead_score_cleaned['What_is_your_current_occupation']=df_lead_score_cleaned['What_is_your_current_occupation'].replace(np.nan,'Other')

#view after cleaning
uniquecolumnkeys(['What_is_your_current_occupation'],df_lead_score_cleaned)
TargetvsColFrequency('What_is_your_current_occupation','Converted',df_lead_score_cleaned)


# In[ ]:


# Feature: 10 What_matters_most_to_you_in_choosing_a_course

#view before cleaning
uniquecolumnkeys(['What_matters_most_to_you_in_choosing_a_course'],df_lead_score_cleaned)
TargetvsColFrequency('What_matters_most_to_you_in_choosing_a_course','Converted',df_lead_score_cleaned)


# In[ ]:


# Feature: 10 What_matters_most_to_you_in_choosing_a_course
## 1. nan  : 2709. So lets replace it with a value of Other as a value
df_lead_score_cleaned['What_matters_most_to_you_in_choosing_a_course']=df_lead_score_cleaned['What_matters_most_to_you_in_choosing_a_course'].replace(np.nan,'Other')

#view after cleaning
uniquecolumnkeys(['What_matters_most_to_you_in_choosing_a_course'],df_lead_score_cleaned)
TargetvsColFrequency('What_matters_most_to_you_in_choosing_a_course','Converted',df_lead_score_cleaned)


# In[ ]:


#Feature: 19 Tags
#view before cleaning
uniquecolumnkeys(['Tags'],df_lead_score_cleaned)
TargetvsColFrequency('Tags','Converted',df_lead_score_cleaned)


# In[ ]:


#Feature: 19 Tags
## 1. Most Common Key is nan. So lets replace it with a value of Other as a value
df_lead_score_cleaned['Tags']=df_lead_score_cleaned['Tags'].replace(np.nan,'Other')

#view after cleaning
uniquecolumnkeys(['Tags'],df_lead_score_cleaned)
TargetvsColFrequency('Tags','Converted',df_lead_score_cleaned)


# In[ ]:


#Feature: 20 Lead_Quality
#view before cleaning
uniquecolumnkeys(['Lead_Quality'],df_lead_score_cleaned)
TargetvsColFrequency('Lead_Quality','Converted',df_lead_score_cleaned)


# In[ ]:


#Feature: 20 Lead_Quality
## 1. Most Common Key: nan so lets move all nan into a bucket with value as 'unknown'
df_lead_score_cleaned['Lead_Quality']=df_lead_score_cleaned['Lead_Quality'].replace(np.nan,'Other')
#view after cleaning
uniquecolumnkeys(['Lead_Quality'],df_lead_score_cleaned)
TargetvsColFrequency('Lead_Quality','Converted',df_lead_score_cleaned)


# In[ ]:


#Feature: 23 Lead_Profile
#view before cleaning
uniquecolumnkeys(['Lead_Profile'],df_lead_score_cleaned)
TargetvsColFrequency('Lead_Profile','Converted',df_lead_score_cleaned)


# In[ ]:


#Feature: 23 Lead_Profile
## 1. Most Common Key: nan 
## 2. Other Leads  : 487, lets move all of them into Other as value
df_lead_score_cleaned['Lead_Profile']=df_lead_score_cleaned['Lead_Profile'].replace(np.nan,'Other')
df_lead_score_cleaned['Lead_Profile']=df_lead_score_cleaned['Lead_Profile'].replace('Other Leads','Other')

#view after cleaning
uniquecolumnkeys(['Lead_Profile'],df_lead_score_cleaned)
TargetvsColFrequency('Lead_Profile','Converted',df_lead_score_cleaned)


# In[ ]:


#Feature: 24 City
#view before cleaning
uniquecolumnkeys(['City'],df_lead_score_cleaned)
TargetvsColFrequency('City','Converted',df_lead_score_cleaned)


# In[ ]:


#Feature: 24 City
## 1. Most Common Key: nan 
## 2. there are many 'other' type city values like - Other Metro Cities  : 380,Other Cities  : 686,Other Cities of Maharashtra  : 457, Tier II Cities  : 74
## so lets move nan to Other Cities  : 686, which is the max value in other cities
df_lead_score_cleaned['City']=df_lead_score_cleaned['City'].replace(np.nan,'Other Cities')
df_lead_score_cleaned['City']=df_lead_score_cleaned['City'].replace('Other Cities','Other')

#view after cleaning
uniquecolumnkeys(['City'],df_lead_score_cleaned)
TargetvsColFrequency('City','Converted',df_lead_score_cleaned)


# In[ ]:


#Feature: 25 Asymmetrique_Activity_Index and Feature: 26 Asymmetrique_Profile_Index
# 1. Most Common Key: nan , lets treat them later based on respective score values. no changes to be done now
#Feature: 29 Last_Notable_Activity - no nans


# In[ ]:


#features to be not considered initially as they have significant number of No values
#featueres 'Asymmetrique_Activity_Index' and 'Asymmetrique_Profile_Index' are redundant as we have the corresponding score features.
features_excluded=['Asymmetrique_Activity_Index','Asymmetrique_Profile_Index']
df_lead_score_cleaned.drop(features_excluded,axis=1,inplace=True)


# In[ ]:



#The below features are categorical than non-object type, so we have moved them to object dtype
df_lead_score_cleaned['Asymmetrique_Profile_Score']=df_lead_score_cleaned['Asymmetrique_Profile_Score'].astype(object)
df_lead_score_cleaned['Asymmetrique_Activity_Score']=df_lead_score_cleaned['Asymmetrique_Activity_Score'].astype(object)


# In[ ]:


# Asymmetrique_Activity_Score data cleaning
df_lead_score_cleaned['Asymmetrique_Activity_Score']=df_lead_score_cleaned['Asymmetrique_Activity_Score'].replace(np.nan,'Other')

#view after cleaning
uniquecolumnkeys(['Asymmetrique_Activity_Score'],df_lead_score_cleaned)
TargetvsColFrequency('Asymmetrique_Activity_Score','Converted',df_lead_score_cleaned)


# In[ ]:


# Asymmetrique_Profile_Score data cleaning
df_lead_score_cleaned['Asymmetrique_Profile_Score']=df_lead_score_cleaned['Asymmetrique_Profile_Score'].replace(np.nan,'Other')

#view after cleaning
uniquecolumnkeys(['Asymmetrique_Profile_Score'],df_lead_score_cleaned)
TargetvsColFrequency('Asymmetrique_Profile_Score','Converted',df_lead_score_cleaned)


# In[ ]:


# Reassign feature groups after fixing of Categorical features
obj_features_lead_score, non_obj_features_lead_score=classifyfeatures(df_lead_score_cleaned)
#lets remove Prospect_ID,Lead_Number,Converted from the object features list for handling perspective
obj_features_lead_score.remove('Prospect_ID')
non_obj_features_lead_score.remove('Lead_Number')
non_obj_features_lead_score.remove('Converted')


# #### Drop Features with >70% Null

# In[ ]:


#lets view the missing dat and then drop columns having >70% of missing values
df_lead_score_cleaned,dropped_columns1=dropcolumns(df_lead_score_cleaned,70)


# In[ ]:


# print dropped Columns
print('Dropped Columns Set 1 having >70% missing values:')
print('\t number of columns:',len(dropped_columns1))
print('\t columns:',dropped_columns1)


# ### Numerical Features - Data Preparation

# In[ ]:


#print per feature value <0, >0 and equal to 0
def lessgreaterequaltoZero(columns,df):
    for i,col in enumerate(columns):
        #print count having less than 0
        print("Feature:",i+1,col)
        print("\tless than 0:",len(df[df[col]<0][col]))
        #print count having greater than 0
        print("\tgreater than 0:",len(df[df[col]>0][col]))
        #print count having 0
        print("\tequal to 0:",len(df[df[col]==0][col]))
        getMissingPercentageFeature(col,df)


# In[ ]:


#print for amount columns
lessgreaterequaltoZero(non_obj_features_lead_score,df_lead_score_cleaned)


# In[ ]:


#lets visualize the numeric features

df_lead_score_cleaned_not_converted=df_lead_score_cleaned[df_lead_score_cleaned.Converted==0]
df_lead_score_cleaned_converted=df_lead_score_cleaned[df_lead_score_cleaned.Converted==1]

fig=plt.figure(figsize=(10,5)) ## setting over-all figure size (optional)

plt.subplot(2,3,1) 

plt.subplot(2,3,1) 
ax1=sns.boxplot(df_lead_score_cleaned_not_converted.TotalVisits)
ax1.set_title('Not Converted Leads',color='red',size=15)

plt.subplot(2,3,2) 
sns.boxplot(df_lead_score_cleaned_not_converted.Total_Time_Spent_on_Website)

plt.subplot(2,3,3) 
sns.boxplot(df_lead_score_cleaned_not_converted.Page_Views_Per_Visit)



plt.subplot(2,3,4) 
ax2=sns.boxplot(df_lead_score_cleaned_converted.TotalVisits)
ax2.set_title('Converted Leads',color='green',size=15)

plt.subplot(2,3,5) 
sns.boxplot(df_lead_score_cleaned_converted.Total_Time_Spent_on_Website)

plt.subplot(2,3,6) 
sns.boxplot(df_lead_score_cleaned_converted.Page_Views_Per_Visit)

plt.tight_layout()


# #### Numerical Feature Analysis
# 1. Converted leads have spent considerable amount on time on the website, as their median is around 750, when compared to not converted leads having a median of 250
# 2. Total Visits, Time spent and Page views per view have significant outliers. Lets treat them later if required.

# #### Imbalance Score

# In[ ]:


#calculate imbalance
imbalance_lead_score=df_lead_score_cleaned['Converted'].sum()/(df_lead_score_cleaned['Converted'].count()+df_lead_score_cleaned['Converted'].sum())
print("Imbalance for Converted Lead Score %",round(imbalance_lead_score*100,2))


# In[ ]:


#print categorical features
print( "Categorical features:\n")
printFeatures(obj_features_lead_score)
#print Non Object features
print( "\n\nNon Object features:\n")
printFeatures(non_obj_features_lead_score)


# #### View and Handle Missing Data in Numerical Features

# In[ ]:


#view the null % for each column based on number of rows in the data set
getMissingPercentage(df_lead_score_cleaned[non_obj_features_lead_score])


# In[ ]:


#lets view the stats for numerical features BEFORE treating for missing values
print('stats for Page_Views_Per_Visit:\n',df_lead_score_cleaned.Page_Views_Per_Visit.describe())
print('\nstats for TotalVisits:\n',df_lead_score_cleaned.TotalVisits.describe())


# In[ ]:


#lets replace missing values with median value 50% value for each

# lets replace nan with median of 2 for Page_Views_Per_Visit
df_lead_score_cleaned.Page_Views_Per_Visit=df_lead_score_cleaned.Page_Views_Per_Visit.replace(np.nan,2)

# lets replace nan with median of 3 for TotalVisits
df_lead_score_cleaned.TotalVisits=df_lead_score_cleaned.TotalVisits.replace(np.nan,3)


# In[ ]:


#lets view the stats for numerical features AFTER treating for missing values
print('stats for Page_Views_Per_Visit:\n',df_lead_score_cleaned.Page_Views_Per_Visit.describe())
print('\nstats for TotalVisits:\n',df_lead_score_cleaned.TotalVisits.describe())


# In[ ]:


# evaluate data set for nulls
df_lead_score_cleaned.isnull().sum().sum()


# In[ ]:


dropped_columns=['Prospect_ID','Lead_Number']
df_lead_score_cleaned.drop(dropped_columns,axis=1,inplace=True)


# ## Encoding

# ### Dummy Encoding

# In[ ]:


#function to create dummies
def CreateDummies(df,feature):
    # create dummy data frame with new name with original column name as prefix  
    feature_dummy = pd.get_dummies(df[feature], drop_first = True,prefix=feature,prefix_sep='_')
    
    #concat feature_dummy with df    
    df=pd.concat([df,feature_dummy],axis=1)
    
    #drop original column
    df=df.drop(feature,axis=1)
    
    # return original df
    return df


# In[ ]:


# create dummy for A_free_copy_of_Mastering_The_Interview
df_lead_score_cleaned_encoded=CreateDummies(df_lead_score_cleaned,obj_features_lead_score)


# In[ ]:


df_lead_score_cleaned_encoded.T.head(162)


# ### Frequency Encoding

# In[ ]:


#function for frequency based encoding for unordered categorical variables
def convertColsToFreqEncoding(df,col):
    for x in col:
        tempDict = df[x].value_counts().to_dict()
        df[x] = df[x].map(tempDict)
    return df


# In[ ]:


#create frequency based encoding data set for model building at a later time
df_lead_score_FequEncoding_cleaned = df_lead_score_cleaned.copy()
df_lead_score_FequEncoding_cleaned = convertColsToFreqEncoding(df_lead_score_FequEncoding_cleaned,obj_features_lead_score)


# ## Model based on Dummy Encoded Data set 
# 

# In[ ]:


#prepare X adn y
X = df_lead_score_cleaned_encoded.drop(['Converted'], axis=1)
y=df_lead_score_cleaned_encoded['Converted']


# ### Test-Train Split

# In[ ]:


# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# ### Feature Scaling and Initial RFE

# In[ ]:


def FeatureScalingdf(df_train,features,scaler):
    #1.instantiate an object
    if scaler=='minmax':
        scaler=MinMaxScaler()
    if scaler=='std':
        scaler=StandardScaler()

    #2. create a list of numeric variables
    #already done numeric_features_car

    #3.fit the object on data set
    df_train[features]=scaler.fit_transform(df_train[features])

    # 4. Asssess Numerical features
    print( "Numerical features of Training Data set after Scaling:")
    return df_train,scaler


# In[ ]:


#1. create a list of numeric variables
features_for_scaling=['Total_Time_Spent_on_Website','Page_Views_Per_Visit','TotalVisits']

#2. Apply scalling
X_train,X_train_scaler=FeatureScalingdf(X_train,features_for_scaling,'std')
print(X_train.head())

#3. lets scale test data set as well 
X_test[features_for_scaling]=X_train_scaler.transform(X_test[features_for_scaling])
print(X_test.head())

#4.Create logistic regression object
logreg = LogisticRegression()

#5. run RFE
rfe = RFE(logreg, 30)             # running RFE with 30 variables as output
rfe = rfe.fit(X_train, y_train)
print(rfe.support_)


# ### Feature Selection Using RFE

# In[ ]:


#create RFE Rank data frame for exploration
feature_list=[]
rank_list=[]
for i in range (0, 158):
    df_temp=pd.DataFrame()
    feature_list.append(list(zip(X_train.columns, rfe.support_, rfe.ranking_))[i][0])
    rank_list.append(list(zip(X_train.columns, rfe.support_, rfe.ranking_))[i][2])
df_rfe=pd.DataFrame({'feature':feature_list,'rank':rank_list})

#view RFE features data set
df_rfe.sort_values(by='rank',ascending=True)[:50]


# In[ ]:


#model 1
import statsmodels.api as sm
col = X_train.columns[rfe.support_]
X_train_sm = sm.add_constant(X_train[col])
logm1 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm1.fit()
res.summary()


# #### Model 1 Outcome
# 1. Feature 'Tags_Lateral student' has high P value, lets drop it

# In[ ]:


#model 2
col=list(col)
col.remove('Tags_Lateral student')
X_train_sm = sm.add_constant(X_train[col])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# #### Model 2 Outcome
# 1. Feature 'Tags_number not provided' has high P value, lets drop it

# In[ ]:


#model 3
col.remove('Tags_number not provided')
X_train_sm = sm.add_constant(X_train[col])
logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm3.fit()
res.summary()


# #### Model 3 Outcome
# 1. Feature 'Tags_wrong number given' has high P value, lets drop it

# In[ ]:


# Tags_wrong number given
#model 4
col.remove('Tags_wrong number given')
X_train_sm = sm.add_constant(X_train[col])
logm4 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm4.fit()
res.summary()


# #### Model 4 Outcome
# 1. Feature 'Tags_Interested  in full time MBA' has high P value, lets drop it

# In[ ]:


#Tags_Interested in full time MBA
#model 5.1
col.remove('Tags_Interested  in full time MBA')
X_train_sm = sm.add_constant(X_train[col])
logm5 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm5.fit()
res.summary()


# #### Model 5.1 Outcome
# 1. P-values of Tags_invalid number

# In[ ]:


#Tags_Interested in full time MBA
#model 5.2
col.remove('Tags_invalid number')
X_train_sm = sm.add_constant(X_train[col])
logm5 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm5.fit()
res.summary()


# #### Checking VIFs

# In[ ]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


#Tags_Other
#model 6
# high VIF
col.remove('Tags_Other')

X_train_sm = sm.add_constant(X_train[col])
logm6 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm6.fit()
res.summary()


# In[ ]:


#model 6.1
# high p-value
col.remove('Lead_Quality_Other')

X_train_sm = sm.add_constant(X_train[col])
logm6 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm6.fit()
res.summary()


# In[ ]:


#model 6.2
# high p-value
col.remove('Tags_in touch with EINS')

X_train_sm = sm.add_constant(X_train[col])
logm6 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm6.fit()
res.summary()


# In[ ]:


#model 6.3
# high p-value
col.remove('Last_Activity_Email Bounced')

X_train_sm = sm.add_constant(X_train[col])
logm6 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm6.fit()
res.summary()


# In[ ]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# #### Model 6.3 Outcome
# 1. Model 6.3 has better p-values and VIFs<5

# ### Model Evaluation

# #### Prediction and Lead Score assignment

# In[ ]:


def predict_probability(X_sm,y,cutoff):
    
    # PREDICT VALUES IN TRAINING DATA SET

    #1.  Getting the predicted values on the train set
    y_pred = res.predict(X_sm)
    print(y[:10])

    #2. Create data set with Converted values from original data set and the predicted probability
    y_pred_final = pd.DataFrame({'Converted':y.values, 'Conversion_Probability':y_pred,'Lead_Score':round(y_pred*100,0)})
    y_pred_final['LeadId'] = y.index
    print(y_pred_final.head())

    #3. Take a random cut off and review the metrics
    y_pred_final['Predicted'] = y_pred_final.Conversion_Probability.map( lambda x: 1 if x > cutoff else 0)

    #4. View Lead_score distribution ( top 20)
    plt.figure(figsize=(10,10))
    sns.countplot(y = 'Lead_Score',data = y_pred_final,order = y_pred_final['Lead_Score'].value_counts().index[:20],hue='Converted')
    plt.title('Count of Leads per Lead Score')
    plt.show()
   
    return y_pred_final


# In[ ]:


#predict y_train values
y_pred_train_final_dummy_encoded=predict_probability(X_train_sm,y_train,0.32)


# #### Metrics based on Confusion Matrix

# In[ ]:


# Metrics and Analysis
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import brier_score_loss

def confusion_metric_analysis(df,target,predicted):
    #1. Create confusion matrix
    confusion = metrics.confusion_matrix(df[target], df[predicted] )
    print('Confusion Matrix:\n',confusion)

    #2.# Let's check the overall accuracy.
    print('Metrics Accuracy Score:', metrics.accuracy_score(df[target], df[predicted]))
    print('Metrics Balanced Accurancy:',metrics.balanced_accuracy_score(df[target],df[predicted]))

    #3. Assess TP,TN,FP,FN
    TP = confusion[1,1] # true positive 
    TN = confusion[0,0] # true negatives
    FP = confusion[0,1] # false positives
    FN = confusion[1,0] # false negatives
    print('Converted predicted as Converted:',TP)
    print('Not converted predicted as Not converted:',TN)
    print('Converted predicted as Not Converted:',FP)
    print('Not Converted predicted as Converted:',FN)
    # Let's see the sensitivity of our logistic regression model
    print ('Sensitivity :',TP / float(TP+FN))
    # Let us calculate specificity
    print('Specificity : ',TN / float(TN+FP))
    # Calculate false postive rate - predicting Conversion when customer does not have converted
    print('False Positve Rate : ',FP/ float(TN+FP))
    # positive predictive value 
    print ('Positive Predictive Value : ',TP / float(TP+FP))
    # Negative predictive value
    print ('Negative Predictve Value : ',TN / float(TN+ FN))
    
    print('\nMathew / Phi co-efficient:',matthews_corrcoef(df[target],df[predicted]))
    
    print('\nBrier Score for Probabilistic Prediction:',brier_score_loss(df[target],df[predicted]))

    


# In[ ]:


# metrics based on confusion matrix
confusion_metric_analysis(y_pred_train_final_dummy_encoded,'Converted','Predicted')


# #### ROC and AUC Metrics and Cut off selection

# In[ ]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[ ]:


# Metrics ROC,
fpr, tpr, thresholds = metrics.roc_curve( y_pred_train_final_dummy_encoded.Converted, y_pred_train_final_dummy_encoded.Conversion_Probability, drop_intermediate = False )

#Draw ROC
draw_roc(y_pred_train_final_dummy_encoded.Converted, y_pred_train_final_dummy_encoded.Conversion_Probability)


# In[ ]:


# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_pred_train_final_dummy_encoded[i]= y_pred_train_final_dummy_encoded.Conversion_Probability.map(lambda x: 1 if x > i else 0)
y_pred_train_final_dummy_encoded.head()


# In[ ]:


# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_pred_train_final_dummy_encoded.Converted, y_pred_train_final_dummy_encoded[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[ ]:


# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# #### Redo Prediction based on Selected Cut off

# In[ ]:


#set cut off and redo labeling
cutoff=0.32
y_pred_train_final_dummy_encoded['final_predicted'] = y_pred_train_final_dummy_encoded.Conversion_Probability.map( lambda x: 1 if x > cutoff else 0)


# #### Redo Metrics based on Confusion Matrix

# In[ ]:


#analyse the metrics
confusion_metric_analysis(y_pred_train_final_dummy_encoded,'Converted','final_predicted')


# #### Metrics based on Precision and Recall

# In[ ]:


from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import classification_report

def precision_recall_metrics(df,target,predicted):
        #confusion = metrics.confusion_matrix(df[target], y_train_pred_final[predicted])
        #Precision TP / TP + FP
        #print('Precision:',confusion[1,1]/(confusion[0,1]+confusion[1,1])
        #Recall TP / TP + FN
        #print('Recall:',confusion[1,1]/(confusion[1,0]+confusion[1,1])
        #classification report
        target_names = ['0', '1']
        print('\nClassification Report:\n',classification_report(df[target],df[predicted],target_names=target_names))
    


# In[ ]:


from sklearn.metrics import hamming_loss
from sklearn.metrics import log_loss
def metrics_loss(df,target,predicted,probability):
    print('Hamming Loss:\n',hamming_loss(df[target], df[predicted]))
    print('\nLogistic Regression Loss:\n',log_loss(df[target], df[probability]))


# In[ ]:


#analyse the metrics
precision_recall_metrics(y_pred_train_final_dummy_encoded,'Converted','final_predicted')
metrics_loss(y_pred_train_final_dummy_encoded,'Converted','final_predicted','Conversion_Probability')


# #### Precision and recall tradeoff
# 

# In[ ]:


#trade off
#1. get the values
from sklearn.metrics import precision_recall_curve
p, r, thresholds = precision_recall_curve(y_pred_train_final_dummy_encoded.Converted, y_pred_train_final_dummy_encoded.Conversion_Probability)

#2.visualize
plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# #### Analysis of Metrics

# ### Making predictions on the test set and Evaluation

# In[ ]:


# 1. Prediction for X_Test
print('Features shortlisted:',col)

# 2. Prepare X_test
X_test = X_test[col]

#3. Add constant
X_test_sm = sm.add_constant(X_test)

#4 Predictions on test data set
y_test_pred = res.predict(X_test_sm)
print('Predictions',y_test_pred[:10])

#5.Converting y_pred to a dataframe which is an array
y_pred_1 = pd.DataFrame(y_test_pred)

##6.Putting LeadId to index
#y_test_pd = pd.DataFrame(y_test)
#y_test['LeadId'] = y_test.index
#y_test_df=pd.DataFrame(y_test)

#7.Removing index for both dataframes to append them side by side 
y_pred_1.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

#8. Appending y_test_df and y_pred_1
y_pred_final = pd.concat([y_test, y_pred_1],axis=1)

#9.Renaming the column 
y_pred_final= y_pred_final.rename(columns={ 0 : 'Converted_Probability'})

#10. cut off probability
y_pred_final['final_predicted'] = y_pred_final.Converted_Probability.map(lambda x: 1 if x > cutoff else 0)
#y_pred_final['Converted'] = y_pred_final['Converted'].astype(int)

#11. Sensitivity, Specificity Metrics
confusion_metric_analysis(y_pred_final,'Converted','final_predicted')

#12. Precision , Recall Metrics
precision_recall_metrics(y_pred_final,'Converted','final_predicted')

#13.Loss metrics,
metrics_loss(y_pred_final,'Converted','final_predicted','Converted_Probability')

#14. Lead_Score Assignment
y_pred_final['Lead_Score']=round(y_pred_final.Converted_Probability*100,0)

#15. View Lead_score distribution ( top 30)
plt.figure(figsize=(10,7))
sns.countplot(y = 'Lead_Score',data = y_pred_final,order = y_pred_final['Lead_Score'].value_counts().index[:30],hue='Converted')
plt.title('Count of Leads per Lead Score')
plt.show()


# ### Analysis of Metrics

# ### Analysis of Selected Features

# #### Correlation Analysis

# In[ ]:


#plot correlation between amount features for target=1
corr_features=['Converted']
corr_features.extend(col)
colormap = plt.cm.RdBu
plt.figure(figsize=(40,15))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(df_lead_score_cleaned_encoded[corr_features].corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)


# In[ ]:


#view the correlation of features with Converted Leads
df_lead_score_cleaned_encoded[corr_features].corr()['Converted'].sort_values(ascending=False)[1:]


# #### Numeric Features

# In[ ]:


#create not converted data frame
df_not_converted=df_lead_score_cleaned_encoded[df_lead_score_cleaned_encoded.Converted==0][col]
df_not_coverted_numeric=df_lead_score_cleaned_encoded['Total_Time_Spent_on_Website']
df_not_converted_other=df_not_converted.drop('Total_Time_Spent_on_Website',axis=1)

#create converted data frame
df_converted=df_lead_score_cleaned_encoded[df_lead_score_cleaned_encoded.Converted==1][col]
df_coverted_numeric=df_converted['Total_Time_Spent_on_Website']
df_converted_other=df_converted.drop('Total_Time_Spent_on_Website',axis=1)


# In[ ]:


#analyse numeric variable
df_not_coverted_numeric.describe()


# In[ ]:


#combine metrics for further analysis
df_temp1=pd.DataFrame(df_not_coverted_numeric.describe()).reset_index()
df_temp2=pd.DataFrame(df_coverted_numeric.describe()).reset_index()
df_numeric=pd.merge(df_temp1,df_temp2,on='index')
df_numeric.columns=['Metric','Total_Time_Spent_on_Website_Not_converted','Total_Time_Spent_on_Website_converted']
df_numeric


# In[ ]:


#visualize numeric variable through box plot
sns.boxplot(df_numeric.Total_Time_Spent_on_Website_Not_converted)


# In[ ]:


#visualize numeric variable through box plot
sns.boxplot(df_numeric.Total_Time_Spent_on_Website_converted)


# ### Categorical Dummy Features

# In[ ]:


# create count of each dummy feature with value=1
df_not_converted_summary=pd.DataFrame(df_not_converted_other.sum()).reset_index()
df_converted_summary=pd.DataFrame(df_converted_other.sum()).reset_index()
df_not_converted_summary.columns=['feature','not_converted_count']
df_converted_summary.columns=['feature','converted_count']

#create combined data set for comparision and analysis
df_compare=pd.merge(df_not_converted_summary,df_converted_summary,on='feature')

#find the ratio of not_converted count out of total count
df_compare['percent_not_converted']=round(df_compare.not_converted_count/(df_compare.not_converted_count+df_compare.converted_count)*100,2)

#find the ratio of converted coutn out of total count
df_compare['percent_converted']=round(df_compare.converted_count/(df_compare.not_converted_count+df_compare.converted_count)*100,2)

#find the diff btw converted and not converted ratios to see the largest differences
df_compare['percent_diff']=df_compare.percent_converted-df_compare.percent_not_converted
df_compare.sort_values(by='percent_diff',inplace=True,ascending=False)
df_compare


# In[ ]:


#visualise the outcome of ratio differences btw converted and not-converted
plt.figure(figsize=(15,8))
df=df_compare[['feature','percent_diff']].sort_values(by='percent_diff')
sns.barplot(x=df.percent_diff,y=df.feature)


# ### View the Co-efficient of the Selected Model

# In[ ]:


#prepare df and view coff of the model
df = pd.DataFrame(res.params).reset_index()
df.columns =['Feature','Coeff']
df.sort_values(by='Coeff',ascending=False,inplace=True)
df


# #### Top Positively influencing Features

# #### Top Negatively influencing Features

# ## Model based on Frequency Encoded Data set 
# 

# In[ ]:


#view the frequency encoded data set
df_lead_score_FequEncoding_cleaned.head()


# In[ ]:


# create X , y data frames
X_freq = df_lead_score_FequEncoding_cleaned.drop(['Converted'], axis=1)
y_freq=df_lead_score_FequEncoding_cleaned['Converted']


# ### Test-Train Split

# In[ ]:


# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X_freq, y_freq, train_size=0.7, test_size=0.3, random_state=100)


# ### Feature Scaling and Initial RFE

# In[ ]:


#1. create a list of numeric variables
#features_for_scaling=['Total_Time_Spent_on_Website','Page_Views_Per_Visit','TotalVisits']
features_for_scaling=list(X_train.columns)

#2. Apply scalling
X_train,X_train_scaler=FeatureScalingdf(X_train,features_for_scaling,'std')
print(X_train.head())

#3. lets scale test data set as well 
X_test[features_for_scaling]=X_train_scaler.transform(X_test[features_for_scaling])
print(X_test.head())

#4.Create logistic regression object
logreg = LogisticRegression()

#5. run RFE
rfe_freq = RFE(logreg, 19)             # running RFE with 30 variables as output
rfe_freq = rfe_freq.fit(X_train, y_train)
print(rfe_freq.support_)


# ### Feature Selection Using RFE
# 

# In[ ]:


X_train_sm = sm.add_constant(X_train[features_for_scaling])
logm7 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm7.fit()
res.summary()


# ### Model 7 Outcome
# 1. Lets remove feature 'A_free_copy_of_Mastering_The_Interview')' which has got a  high p value

# In[ ]:


features_for_scaling.remove('A_free_copy_of_Mastering_The_Interview')
X_train_sm = sm.add_constant(X_train[features_for_scaling])
logm8 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm8.fit()
res.summary()


# #### Model 8 Outcome
# 1. City is having higher p-values, but lets check VIF before removing City

# #### Checking VIFs

# In[ ]:


vif = pd.DataFrame()
vif['Features'] = X_train[features_for_scaling].columns
vif['VIF'] = [variance_inflation_factor(X_train[features_for_scaling].values, i) for i in range(X_train[features_for_scaling].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# 1. Remove 'Asymmetrique_Profile_Score' due to VIF>5

# In[ ]:


features_for_scaling.remove('Asymmetrique_Profile_Score')
X_train_sm = sm.add_constant(X_train[features_for_scaling])
logm9 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm9.fit()
res.summary()


# In[ ]:


vif = pd.DataFrame()
vif['Features'] = X_train[features_for_scaling].columns
vif['VIF'] = [variance_inflation_factor(X_train[features_for_scaling].values, i) for i in range(X_train[features_for_scaling].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# ### Model 9 Outcome
# 1. Lets remove 'City' which has high p-value

# In[ ]:


features_for_scaling.remove('City')
X_train_sm = sm.add_constant(X_train[features_for_scaling])
logm10 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm10.fit()
res.summary()


# In[ ]:


vif = pd.DataFrame()
vif['Features'] = X_train[features_for_scaling].columns
vif['VIF'] = [variance_inflation_factor(X_train[features_for_scaling].values, i) for i in range(X_train[features_for_scaling].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# #### Model 10 Outcome
# 1. Looks good interms of p-value and VIFs

# ## Model Evaluation

# #### Prediction and Lead Score assignment

# In[ ]:


# PREDICT VALUES IN TRAINING DATA SET
y_train_pred_final=predict_probability(X_train_sm,y_train,0.5)


# #### Metrics based on Confusion Matrix

# In[ ]:


#analysis of metrics based on confusion matrix
confusion_metric_analysis(y_train_pred_final,'Converted','Predicted')


# #### ROC and AUC Metrics and Cut off selection

# In[ ]:


# Metrics ROC,
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Conversion_Probability, drop_intermediate = False )

#Draw ROC
draw_roc(y_train_pred_final.Converted, y_train_pred_final.Conversion_Probability)


# In[ ]:



# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Conversion_Probability.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[ ]:


# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# #### Redo Prediction based on Selected Cut off

# In[ ]:



# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# In[ ]:


#reassing cutoff
cutoff=0.35
y_train_pred_final['final_predicted'] = y_train_pred_final.Conversion_Probability.map( lambda x: 1 if x > cutoff else 0)


# #### Redo Metrics based on Confusion Matrix, Precision Recall

# In[ ]:



#analyse the metrics
confusion_metric_analysis(y_train_pred_final,'Converted','final_predicted')


#analyse the metrics
precision_recall_metrics(y_train_pred_final,'Converted','final_predicted')
metrics_loss(y_train_pred_final,'Converted','final_predicted','Conversion_Probability')


# #### Precision and recall tradeoff

# In[ ]:


#trade off
#1. get the values
from sklearn.metrics import precision_recall_curve
p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Conversion_Probability)

#2.visualize
plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# #### Analysis of Metrics

# ## Making predictions on the test set and Evaluation

# In[ ]:


# 1. Prediction for X_Test
print('Features shortlisted:',features_for_scaling)

# 2. Prepare X_test
X_test = X_test[features_for_scaling]

#3. Add constant
X_test_sm = sm.add_constant(X_test)

#4 Predictions on test data set
y_test_pred = res.predict(X_test_sm)
print('Predictions',y_test_pred[:10])

#5.Converting y_pred to a dataframe which is an array
y_pred_1 = pd.DataFrame(y_test_pred)

##6.Putting LeadId to index
#y_test_pd = pd.DataFrame(y_test)
#y_test['LeadId'] = y_test.index
#y_test_df=pd.DataFrame(y_test)

#7.Removing index for both dataframes to append them side by side 
y_pred_1.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

#8. Appending y_test_df and y_pred_1
y_pred_final = pd.concat([y_test, y_pred_1],axis=1)

#9.Renaming the column 
y_pred_final= y_pred_final.rename(columns={ 0 : 'Converted_Probability'})

#10. cut off probability
y_pred_final['final_predicted'] = y_pred_final.Converted_Probability.map(lambda x: 1 if x > cutoff else 0)
#y_pred_final['Converted'] = y_pred_final['Converted'].astype(int)

#11. Sensitivity, Specificity Metrics
confusion_metric_analysis(y_pred_final,'Converted','final_predicted')

#12. Precision , Recall Metrics
precision_recall_metrics(y_pred_final,'Converted','final_predicted')

#13.Loss metrics,
metrics_loss(y_pred_final,'Converted','final_predicted','Converted_Probability')

#14. Lead_Score Assignment
y_pred_final['Lead_Score']=round(y_pred_final.Converted_Probability*100,0)

#15. View Lead_score distribution ( top 50)
plt.figure(figsize=(10,10))
sns.countplot(y = 'Lead_Score',data = y_pred_final,order = y_pred_final['Lead_Score'].value_counts().index[:50],hue='Converted')
plt.title('Count of Leads per Lead Score')
plt.show()


# In[ ]:


#plot correlation between amount features for target=1
corr_features=['Converted']
corr_features.extend(features_for_scaling)
colormap = plt.cm.RdBu
plt.figure(figsize=(40,15))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(df_lead_score_FequEncoding_cleaned[corr_features].corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)


# In[ ]:


#view the correlation of features with Converted Leads
df_lead_score_FequEncoding_cleaned[corr_features].corr()['Converted'].sort_values(ascending=False)[1:]


# In[ ]:


#prepare df and view coff of the model
df = pd.DataFrame(res.params).reset_index()
df.columns =['Feature','Coeff']
df.sort_values(by='Coeff',ascending=False,inplace=True)
df

