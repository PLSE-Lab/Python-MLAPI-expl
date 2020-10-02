#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Hello Guys, In this kernel we will try to perform exploratory data analysis to understand the naukri.com dataset and find some insights from it. 
# 
# if you like this kernel explaination, please vote and put your comments.

# # Data Description
# This is naukri.com job portal dataset and it was created by teams at PromptCloud and DataStock. This dataset holds up to 30K unique data sample records. The dataset gives you information about job opportunities published by recruiters and has some importand features or columns such as Industry, Location, Roles, Exp required in Years etc which can be analysed.

# In[ ]:


#import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


#find the dataset file name
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


#let us create dataframe to read data set
df = pd.read_csv('/kaggle/input/jobs-on-naukricom/home/sdf/marketing_sample_for_naukri_com-jobs__20190701_20190830__30k_data.csv')


# In[ ]:


#let us see the data for first 5 records from dataframe
df.head()


# In[ ]:


#let us check the dataframe structure and its column type
df.info()


# **Take-away**: there are 11 columns and all these are object type

# In[ ]:


#let us find number of rows and column in given dataset
df.shape


# **Take-away**: as per the shape, dataset has 30000 rows and 11 columns

# **Features**
# 1. Uniq Id | object | Uniq Id value
# 2. Crawl Timestamp | object | web crawling time
# 3. Job Title | object | job work title 
# 4. Job Salary | object | job salart in range
# 5. Job Experience Required | object | Experience required to apply the job
# 6. Key Skills | object | key skill required for job
# 7. Role Category | object | Role Category
# 8. Location | object | job location
# 9. Functional Area | object | functional role for job
# 10. Industry | object | type of industry
# 11. Role | object | type of Role

# # Exploratory Data Analysis

# we will perform exploratory data analysis on given dataset to understand below points
# 
# - Find Unwanted Columns or Features which has only one value or unique values and which will not play important role in analysis
# - Handle the missing values if there is any
# - Explore, what are categorical features we have in given dataset
# - Understand the features distribution
# - Find the Top 15 Industries and Functional Areas for which maximum number of oportunies are available
# - Find the Top 15 Locations where job opportunities are very high
# - Find the Top 15 Role and Role category to which maximum number of recruiters are looking
# - Find the Top 15 Job exp required in Years for which maximum number of oportunies are available
# - Find the Top Job experiences levels for which maximum number of oportunies are available
# - Find the Salary Ranges for which maximum number of oportunies are available
# - Find the Top 20 Key Skills for which maximum number of oportunies are available

# **1. Find Unwanted Columns**

# **Take-away**: Uniq Id and Crawl Timestamp columns seem to be not usefull.

# **2. Find Missing Values**

# In[ ]:


# find missing values
features_na = [features for features in df.columns if df[features].isnull().sum() > 0]
for feature in features_na:
    print(feature, np.round(df[feature].isnull().mean(), 4),  ' % missing values and actual count is '+str(df[feature].isnull().sum()))

print('Total entries:{}'.format(len(df)))


# In[ ]:


# check what will be new shape after droping missing values
df.dropna(inplace=False).shape


# **Take-away**: there are some missing values found in job title, job salary and in other features, specially Role Category feautre has max 2305 missing values. the number of missing values in each feature are not much. even if we drop all missing value still we will have good amount data to analyse. let us drop all missing values permanently.

# In[ ]:


#drop missing values
df.dropna(inplace=True)


# In[ ]:


#check shape
df.shape


# **3. Find Features with One Value**

# In[ ]:


for column in df.columns:
    print(column,df[column].nunique())


# **Take-away**: No feature with only one value

# **4. Explore the Categorical Features**
# 
# let us try to explore the all important categorical features or columns

# In[ ]:


categorical_features=[feature for feature in df.columns if ((df[feature].dtypes=='O') & (feature not in ['Uniq Id','Crawl Timestamp']))]
categorical_features


# In[ ]:


for feature in categorical_features:
    print('The feature is {} and number of categories are {}'.format(feature,len(df[feature].unique())))


# **Take-away**:
# - there are 9 categorical features
# - all features seems to be have very high number categorical values. let us understand these features one by one
# - job title feature analysis can be ignored as it has high count around 22k uniq values and also same information can be found by anyzing the role feature

# **5. Find Categorical Feature Distribution**
# - let us understand these categorical features one by one

# **Industry**

# In[ ]:


df.groupby('Industry',sort=True)['Industry'].count()[0:15]


# Industry column has uncleaned data, let us clean this before analysing. let us group similary industries into generic one

# In[ ]:


from re import search

def get_comman_job_industry(x):
    x = x.replace(",", " /")
    if (search('it-software', x.lower())):
        return 'Software Services'
    elif (search('call ', x.lower())):
        return 'Call Centre'
    elif (search('banking', x.lower()) or search('insurance', x.lower()) or search('finance', x.lower())):
        return 'Financial Services'
    elif (search('recruitment', x.lower())): 
        return 'Recruitment'
    elif (search('pharma', x.lower())): 
        return 'Pharma'
    elif (search('isp', x.lower())): 
        return 'Telcom / ISP'
    elif (search('ecommerce', x.lower())): 
        return 'Ecommerce'
    elif (search('fmcg', x.lower())): 
        return 'FMCG'
    elif (search('ngo', x.lower())): 
        return 'NGO'
    elif (search('medical', x.lower())): 
        return 'Medical'
    elif (search('aviation', x.lower())): 
        return 'Aviation'
    elif (search('fresher ', x.lower())): 
        return 'Fresher'
    elif (search('education', x.lower())): 
        return 'Education'
    elif (search('construction', x.lower())): 
        return 'Construction'
    elif (search('consulting', x.lower())): 
        return 'Consulting'
    elif (search('automobile', x.lower())): 
        return 'Automobile'
    elif (search('travel', x.lower())): 
        return 'Travels'
    elif (search('advertising', x.lower()) or search('broadcasting', x.lower())): 
        return 'Advertising'
    elif (search('transportation', x.lower())): 
        return 'Transportation'
    elif (search('agriculture', x.lower())): 
        return 'Agriculture'
    elif (search('agriculture', x.lower())): 
        return 'Agriculture'
    elif (search('industrial', x.lower())): 
        return 'Industrial Products'
    elif (search('media', x.lower())): 
        return 'Entertainment'
    elif (search('teksystems', x.lower()) or search('allegis', x.lower()) or search('aston', x.lower())
         or search('solugenix', x.lower()) or search('laurus', x.lower()) ):
        return 'Other'
    else:
        return x.strip()


# let us apply above custom method on Industry column and create new column that is New_Industry. and finaly get the Industry level count order by its count

# In[ ]:


df['New_Industry']=df['Industry'].apply(get_comman_job_industry)
df.groupby('New_Industry',sort=True)['New_Industry'].count().sort_values(ascending=False)[0:15]


# let us plot these group by numbers into bar plot

# In[ ]:


plt.figure(figsize=(18,6), facecolor='white')
df.groupby('New_Industry',sort=True)['New_Industry'].count().sort_values(ascending=False)[0:15].plot.bar(color='green')
plt.xlabel('Industry')
plt.ylabel('Count')
plt.title('Distribution of Top 15 Industry')
plt.show()


# **Take-away**: we can see from the barplot above, the industry as Software Services are the maximum followed by Recruitment and  then by Financial Services.

# **Functional Area**

# In[ ]:


df.groupby('Functional Area',sort=True)['Functional Area'].count()[0:20]


# functional area column also has uncleaned data, let us clean this before analysing. let us group similary functional area into generic one and update unkown description to others

# In[ ]:


from re import search

def get_comman_func_area(x):
    x = x.replace(",", " /")
    if (search('beauty', x.lower())):
        return 'Beauty / Fitness'
    elif (search('teaching', x.lower())):
        return 'Teaching  / Education'
    elif (search('other', x.lower())):
        return 'Others'
    elif (search('teksystems', x.lower()) or search('allegis', x.lower()) or search('aston', x.lower())
         or search('solugenix', x.lower()) or search('laurus', x.lower()) ):
        return 'Other'
    else:
        return x.strip()


# let us apply above custom method on functional area column and create new column that is New_Functional_Area. and finaly get the New_Functional_Area level count and sort in desc order

# In[ ]:


df['New_Functional_Area']=df['Functional Area'].apply(get_comman_func_area)
df.groupby('New_Functional_Area',sort=True)['New_Functional_Area'].count().sort_values(ascending=False)[0:15]


# In[ ]:


plt.figure(figsize=(18,6), facecolor='white')
df.groupby('New_Functional_Area',sort=True)['New_Functional_Area'].count().sort_values(ascending=False)[0:15].plot.bar()
plt.xlabel('Functional Area')
plt.ylabel('Count')
plt.title('Distribution of Top 15 Functional Area')
plt.show()


# **Take-away**: we can see from the barplot above, the Functional Area such as IT Software - Application Programming are the maximum followed by Sales and then by BPO/KPO.

# **Location**

# In[ ]:


df.groupby('Location',sort=True)['Location'].count()[0:30]


# Location column has multiple locatios separated by comm (,). let us separate all these location into individual rows, so that it will easy to analyse its count based on job opportunities

# In[ ]:


def get_location(df):
    df_new=pd.DataFrame()
    for index, row in df.iterrows():
        for loc in row['Location'].split(','):
            loc_df = pd.DataFrame([loc])
            df_new = pd.concat([df_new,loc_df],ignore_index=True)
    return df_new    


# let us create new dataframe to hold these location splits

# In[ ]:


Location_df = get_location(df)
Location_df.columns = ['Location']


# In[ ]:


Location_df.groupby('Location',sort=True)['Location'].count().sort_values(ascending=False)[0:30]


# there are same location with different name, let us group these location

# In[ ]:


from re import search
def get_comman_location(x):
    x = x.replace(",", " /")
    if (search('bengaluru', x.lower()) or search('bangalore', x.lower())):
        return 'Bengaluru'
    elif (search('ahmedabad', x.lower())):
        return 'Ahmedabad'
    elif (search('chennai', x.lower())):
        return 'Chennai'
    elif (search('coimbatore', x.lower())):
        return 'Coimbatore'
    elif (search('delhi', x.lower()) or search('noida', x.lower()) or search('gurgaon', x.lower())):
        return 'Delhi NCR'
    elif (search('hyderabad', x.lower())):
        return 'Hyderabad'
    elif (search('kolkata', x.lower())):
        return 'Kolkata'
    elif (search('mumbai', x.lower())):
        return 'Mumbai'
    elif (search('Pune', x.lower())):
        return 'pune'
    elif (search('other', x.lower())):
        return 'Others'
    else:
        return x.strip()


# let us apply above custom method on location column and create new column that is New_Location. and finaly get the location level count

# In[ ]:


Location_df['New_Location']=Location_df['Location'].apply(get_comman_location)


# In[ ]:


Location_df.groupby('New_Location',sort=True)['New_Location'].count().sort_values(ascending=False)[0:15]


# In[ ]:


plt.figure(figsize=(18,6), facecolor='white')
Location_df.groupby('New_Location',sort=True)['New_Location'].count().sort_values(ascending=False)[0:15].plot.bar(color="red")
plt.xlabel('Location')
plt.ylabel('Count')
plt.title('Distribution of Top 15 Locations')
plt.show()


# **Take-away**: we can see from the barplot above, the location such as Delhi NCR are the maximum followed by Bengaluru and then by Mumbai.

# **Role Category**

# In[ ]:


df.groupby('Role Category',sort=True)['Role Category'].count().sort_values(ascending=False)[0:15]


# In[ ]:


plt.figure(figsize=(18,6), facecolor='white')
df.groupby('Role Category',sort=True)['Role Category'].count().sort_values(ascending=False)[0:15].plot.bar(color="yellow")
plt.xlabel('Role Category')
plt.ylabel('Count')
plt.title('Distribution of Top 15 Role Category')
plt.show()


# **Take-away**: we can see from the barplot above, the role category such as Programming & Design are the maximum followed by voice and then by retail sales.

# **Role**

# In[ ]:


df.groupby('Role',sort=True)['Role'].count().sort_values(ascending=False)[0:15]


# In[ ]:


plt.figure(figsize=(18,6), facecolor='white')
df.groupby('Role',sort=True)['Role'].count().sort_values(ascending=False)[0:15].plot.bar(color="brown")
plt.xlabel('Role')
plt.ylabel('Count')
plt.title('Distribution of Top 15 Role')
plt.show()


# **Take-away**: we can see from the barplot above, the role such as software devloper are the maximum followed by Non Tech and then by sales.

# **Job Experience Required**

# In[ ]:


df.groupby('Job Experience Required',sort=True)['Job Experience Required'].count()[0:30]


# this column has two years range with unit as either Years and yrs. let us update all these units to yrs

# In[ ]:


df['New_Job_Exp']=df['Job Experience Required'].apply(lambda x: x.replace("Years", "yrs"))


# In[ ]:


df.groupby('New_Job_Exp',sort=True)['New_Job_Exp'].count().sort_values(ascending=False)[0:15]


# let us also plot these group by numbers into bar plot

# In[ ]:


plt.figure(figsize=(18,6), facecolor='white')
df.groupby('New_Job_Exp',sort=True)['New_Job_Exp'].count().sort_values(ascending=False)[0:15].plot.bar(color="orange")
plt.xlabel('Job Exp Required in Years')
plt.ylabel('Count')
plt.title('Distribution of Top 15 Job Experience Year Range Required')
plt.show()


# Let us understand what kind of job experiences level are most commonly published. For better understanding, we will divided the various ranges as follows:
# - 0-2 : Freshers
# - 2-5 : Intermediate
# - 5-8 : Lead
# - 8-12 : Manager
# - 12-16 : Senior Manager
# - 16-20 : Executive
# - 20-above : Senior Executive

# In[ ]:


import re
def get_exp_level(x):
    if re.findall('-',x):
        lst =x.replace('yrs','').strip().split('-')
        #print (x)
        lvl =(int(lst[0].strip())+int(lst[1].strip()))/2
        if (lvl >= 0 and lvl <= 2):
            return ('Freshers')
        elif (lvl >= 2 and lvl <= 5):
            return ('Intermediate')
        elif (lvl >= 5 and lvl <= 8):
            return ('Lead')
        elif (lvl >= 8 and lvl <= 12):
            return ('Manager')
        elif (lvl >= 12 and lvl <= 16):
            return ('Senior Manager')
        elif (lvl >= 16 and lvl <= 20):
            return ('Executive')
        elif (lvl >= 20):
            return ('Senior Executive')
        else:
            return('Others')
    else:
        return('Others')


# let us apply above custom method on New_Job_Exp column and create new column that is New_Exp_Level. and finaly get the exp level count

# In[ ]:


df['New_Exp_Level']=df['New_Job_Exp'].apply(get_exp_level)


# In[ ]:


df.groupby('New_Exp_Level',sort=True)['New_Exp_Level'].count().sort_values(ascending=False)[0:30]


# In[ ]:


plt.figure(figsize=(18,6), facecolor='white')
df.groupby('New_Exp_Level',sort=True)['New_Exp_Level'].count().sort_values(ascending=False).plot.bar(color="purple")
plt.xlabel('Exp Level Required in Years')
plt.ylabel('Count')
plt.title('Distribution of Job Experience Level')
plt.show()


# **Take-away**: As we can see, most job posting required intermediate level experience professionals that is about 2-5 years exp holders followed by Lead Exp (5-8 Years) and then Freshers (0-2 Years).

# **Job Salary**

# In[ ]:


df.groupby('Job Salary',sort=True)['Job Salary'].count()[0:30]


# it is very uncleaned data, let us try to clean this and add generic salary label as per exp level and analyse
# - 0-2 : Freshers : 0-3L
# - 2-5 : Intermediate : 3-8L
# - 5-8 : Lead : 8 -15L
# - 8-12 : Manager : 15 - 22L
# - 12-16 : Senior Manager : 22 - 30L
# - 16-20 : Executive : 30 - 38L
# - 20-above : Senior Executive : 38L - above

# In[ ]:


import re
def get_salary(x):
    if re.findall('-',x):
        lst =x.replace('PA.','').replace(',','').replace('INR','').strip().split('-')        
        try:
            sal1 = int(lst[0].strip())
            sal2 = int(lst[1].strip())
            #print (sal1)
            if (sal1 <= 300000):
                return '0-3L PA'
            elif (sal1 >= 300000 & sal2 <= 800000 ):
                return '3-8L PA'
            elif (sal1 >= 800000 & sal2 <= 1500000 ):
                return '8 -15L PA'
            elif (sal1 >= 1500000 & sal2 <= 2200000 ):
                return '15 - 22L PA'
            elif (sal1 >= 2200000 & sal2 <= 3000000 ):
                return '22 - 30L PA'
            elif (sal1 >= 3000000 & sal2 <= 3800000 ):
                return '30 - 38L PA'
            if (sal1 >= 3800000):
                return '38L - Above PA'
        except:
            return('Others')
    else:
        return('Others')


# let us apply above custom method on Job Salary column and create new column that is New_Job_Salary. and finaly get the Job Salary level count

# In[ ]:


df['New_Job_Salary']=df['Job Salary'].apply(get_salary)


# In[ ]:


df.groupby('New_Job_Salary',sort=True)['New_Job_Salary'].count().sort_values(ascending=False)[0:30]


# **Key Skills**

# **Take-away**: it seems for most of the job openings, no job salary is mentioned.

# In[ ]:


df.groupby('Key Skills',sort=True)['Key Skills'].count()[0:30]


# key skill are sperated by pipeline (|), let us split and arrange the skills in group

# In[ ]:


def get_skills(df):
    df_new=pd.DataFrame()
    for index, row in df.iterrows():
        for skill in row['Key Skills'].split('|'):
            skill_df = pd.DataFrame([skill])
            df_new = pd.concat([df_new,skill_df],ignore_index=True)
    return df_new    


# create new dataframe for key skill to hold all these skill values

# In[ ]:


key_skill_df = get_skills(df)
key_skill_df.columns = ['key_skills']


# In[ ]:


key_skill_df.groupby('key_skills',sort=True)['key_skills'].count().sort_values(ascending=False)[0:20]


# In[ ]:


plt.figure(figsize=(18,6), facecolor='white')
key_skill_df.groupby('key_skills',sort=True)['key_skills'].count().sort_values(ascending=False)[0:20].plot.bar(color="orange")
plt.xlabel('Key Skills')
plt.ylabel('Count')
plt.title('Distribution of Top 20 Key Skills')
plt.show()


# **Take-away**: as per above bar plot, it is clearly visible that some of the top skills are Javascript, HTML, SQL, Sales and Python

# # Conclusion

# - Given dataset has 30000 rows and 11 columns
# - Uniq Id and Crawl Timestamp columns are seems to be not usefull
# - There are some missing values found in job title, job salary and in other features, specially Role Category feautre has max 2305 missing values. the number of missing values in each feature are not much. even if we drop all missing value still we will have good amount data to analyse
# - There are 9 categorical features for which we tried to anyze
# - All features seems to be have very high number of categorical values. 
# - Job title feature analysis can be ignored as it has high count around 22k unique values and also same information can be found by anyzing the role and its categoty features
# - Industry, Functional Area and few other columns have uncleaned data and after cleaning the data we found that
# - Industry as Software Services are the maximum followed by Recruitment and then by Financial Services.
# - Functional Area such as IT Software - Application Programming are the maximum followed by Sales and then by BPO/KPO.
# - Location column has multiple locatios separated by comm (,), after separated these we found that the location such as Delhi NCR are the maximum followed by Bengaluru and then by Mumbai
# - Role category such as Programming & Design are the maximum followed by voice and then by retail sales.
# - Most job posting required intermediate level experience professionals that is about 2-5 years exp holders followed by Lead Exp (5-8 Years) and then Freshers (0-2 Years).
# - it seems for most of the job openings, no job salary is mentioned and hence we could not find any insight from this.
# - Key skill are sperated by pipeline (|), After separated these we found that some of the top skills are Javascript, HTML, SQL, Sales and Python

# if you like this kernel explaination, please vote this and put your comments. thank you.
# 
# https://www.youtube.com/watch?v=qBmd0jVXNb4&list=PLNvKRfckeRUlBZKzWzu0OdjmB4UUEvEWt
