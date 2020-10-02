#!/usr/bin/env python
# coding: utf-8

# # General information

# The following Jupyter notebook was created in order to derive some meaningful insights from the US Permanent Visa Application decisions. Data covers years 2012 - 2017 and includes information on employer, position, wage offered, job posting history, employee education and past visa history, associated lawyers, and final decision. It was collected and distributed by the US Department of Labor. 
# 
# According to Kaggle's dataset context, a permanent labor certification issued by the Department of Labor (DOL) allows an employer to hire a foreign worker to work permanently in the United States. In most instances, before the U.S. employer can submit an immigration petition to the Department of Homeland Security's U.S. Citizenship and Immigration Services (USCIS), the employer must obtain a certified labor certification application from the DOL's Employment and Training Administration (ETA). The DOL must certify to the USCIS that there are not sufficient U.S. workers able, willing, qualified and available to accept the job opportunity in the area of intended employment and that employment of the foreign worker will not adversely affect the wages and working conditions of similarly employed U.S. workers.
# 
# The goal of the below data analysis is checking the general trend in Visa applications, the most popular citizenships, employers, cities and finally, predicting the application decision based on the chosen features. 

# # Importing necessary packages

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# # Exploration of the dataset

# Reading the raw data from the "us_perm_visas.csv" file available within Kaggle's datasets into a dataframe

# In[ ]:


# Decision_date and case_recieved_date are read as dates
df = pd.read_csv('../input/us_perm_visas.csv', low_memory = False, parse_dates=['decision_date', 'case_received_date'])


# Let's take a look at the structure of our dataset by checking the number of observations, columns and displaying 10 first and last rows.

# In[ ]:


# Displaying number of rows and columns
print('Number of Visa Applications:', len(df))
print('Number of Columns:', len(df.columns))


# In[ ]:


# Displaying 5 first rows
df.head()


# In[ ]:


# Displaying 5 last rows
df.tail()


# As we can see, our data set consists of 374362 applications described by 153 attributes. Let's display their names.

# In[ ]:


print(df.columns.values)


# Since we have 2 attributes that may contain similar information - case_number & case_no - let's check their lenghts, number of missing values and example values.

# In[ ]:


print("Length of 'case_number' column is: ", len(df['case_number'])," with",df.case_number.isnull().sum(), "missing values")
print("Length of 'case_no' column is: ", len(df['case_no'])," with", df.case_no.isnull().sum(),"missing values \n")

print("First 2 values of case_number column are : \n", df['case_number'].head(2),"\n")

print("Last 2 values of case_number column are : \n", df['case_number'].tail(2), "\n")
print("First 2 values of case_no column are : \n", df['case_no'].head(2), "\n")
print("Last 2 values of case_no column are : \n", df['case_no'].tail(2))


# As we supposed, these columns contain similar values and their "NaN" values add up to the total number of observations so let's create new column containing only non missing values from both "case_number" and "case_no" columns and then we will remove them.

# In[ ]:


casenoindex = df.columns.get_loc("case_no")
casenumberindex = df.columns.get_loc("case_number")
casenumberlist = []

for value in df.iloc[0:135269,casenoindex]:
    casenumberlist.append(value)
    
for value in df.iloc[135269:374363,casenumberindex]:
    casenumberlist.append(value)
    
df['casenumber'] = casenumberlist
df.drop(df.columns[[casenoindex,casenumberindex]], axis=1, inplace=True)


# Now, let's check the "case_status" column as it may contain information about decision made for respective Visa application and print the length of unique values it contains.

# In[ ]:


#Printing number of unique values for 'case_status' column
for value in df.case_status.unique():
    print(len(df[df['case_status'] == value])," occurrences of status '{}'".format(value))


# 
# 
# 
# Since our observations contain some records with status == "Withdrawn", we will remove them from our dataset and for cases where status is "Certified" or "Certified-Expired" we will use just one value " Certified" so that we will end up having only the desired values namely "Certified" and "Denied".  According to Wikipedia and other internet resources, petitioners have 6 months time to file I-140 form after the receiving the status of "Certified" before it expires and turns to "Certified-Expired" status.
# 
# Form I-140, Immigrant Petition for Alien Worker is a form submitted to the United States Citizenship and Immigration Services (USCIS) by a prospective employer to petition an alien to work in the US on a permanent basis. This is done in the case when the worker is deemed extraordinary in some sense or when qualified workers do not exist in the US.

# In[ ]:


#Removing all withdrawn applications
df = df[df.case_status != 'Withdrawn']

#Combining certified-expired and certified applications and displaying distribution of "case_status" variable
df.loc[df.case_status == 'Certified-Expired', 'case_status'] = 'Certified'
df.case_status.value_counts()


# It's interesting that only 7.2% of Visa applications were denied. Now, let's perform dimensionality reduction by removing rows and columns containing only 'NaN' values and check the dataframe's shape.

# In[ ]:


#Dropping all empty columns
df = df.dropna(axis=1, how='all');

#Dropping all empty rows
df = df.dropna(axis=0, how='all');

df.shape


# It looks like there are neither rows nor columns containing only 'NaN' values so let's check how many columns contains any missing values.

# In[ ]:


# Displaying number of missing values in each column
for column in df.columns:
    print("Attribute '{}' contains ".format(column),  df[column].isnull().sum().sum(), " missing values")


# ## Visualization of the unprocessed data

# Before removing columns which consist mostly of missing values, let's create a new column containing only the year of Visa application submission and perform some visualisation in order to derive initial insights .

# In[ ]:


#Converting the date to contain just the year of application submission
df['year'] = df['decision_date'].dt.year

#Setting plot parameters
fig, ax = plt.subplots()
fig.set_size_inches(12.7, 8.27)
sns.set_context(rc={"font.size":12})
sns.countplot(x="year", hue="case_status", data=df)
ax.set(xlabel='Visa application year', ylabel='Number of Visa applicatons')


# As we can observe, the number of submitted Visa applications increases every year. It's interesting that while the number of possitively considered applications increases, the number of "Denied" ones seems to be similar from year 2013. As a next step, let's see, what where the most popular cities.

# In[ ]:


# Displaying 15 most popular cities
df['employer_city'] = df['employer_city'].str.upper()
df['employer_city'].value_counts().head(15)


# In[ ]:


# Setting plot parameters
fig, ax = plt.subplots()
fig.set_size_inches(13.7, 8.27)
sns.set_context("paper", rc={"font.size":12,"axes.titlesize":12,"axes.labelsize":12})
sns.countplot(x='employer_city', hue='year', data=df, order=df.employer_city.value_counts().iloc[:10].index)
plt.xticks(rotation=90)
ax.set(xlabel='Employer city', ylabel='Number of Visa applications')


# In the last few years, the most popular destination cities were: New York, College Station, Santa Clara, San Jose, Redmond, Mountain View, Houston, SunnyVale, San Francisco and Plano. In most of the cities there was a positive trend in Visa applications. A bizarre situation occured in College Station in 2015 where the number of submitted Visa applications was more or less twice large as in other cities.  
# 
# 
# Now, let's take a look what were the most hiring employers and economic sectors through these years. For "us_economic_sector" variable we have only 120 868 non-missing values, but this should give us an insight. 

# In[ ]:



#Setting plot parameters
fig, ax = plt.subplots()
fig.set_size_inches(12.7, 8.27)
sns.set_context(rc={"font.size":12,"axes.labelsize":13})
sns.countplot(x='employer_name', data=df, palette = sns.cubehelix_palette(8, start=.5, rot=-.75), order=df.employer_name.value_counts().iloc[:10].index)
plt.xticks(rotation=90)

#Iterating over elements in "employer_name" column and displaying counts above bars 
for i, v in df.employer_name.value_counts().head(10).reset_index().iterrows():
    ax.text(i,v.employer_name,v.unique()[1], horizontalalignment='center',verticalalignment='bottom')
    
ax.set(xlabel='Employer name', ylabel='Number of Visa applications')


# As we can see, 9 out of 10 most beneficial companies for Visa applicants are IT industry representatives. This leads to the assumption that IT sector is both most favourable and demanding one in United States. Let's check what is the distribution of industries across all Visa applications.

# In[ ]:


#Creating empty dictionary
us_economic_counts = {}

#Iterating over "us_economic_sector" column and appending values to the "us_economic_counts" dictionary
for value in df['us_economic_sector'].dropna():
    if value in us_economic_counts:
        us_economic_counts[value] += 1
    else:
        us_economic_counts[value] = 1


# In[ ]:


#Creating lists for us economic sectors and their counts
usecolabels = []
usecovalues = []
explode = (0.035, 0, 0, 0,0,0,0,0,0,0)

for key, value in us_economic_counts.items():
    usecolabels.append(key)
    usecovalues.append(value)
    
#Setting plot parameters
plt.figure(figsize=(13,13))    
sns.set_context(rc={"font.size":10,"axes.labelsize":11,"xtick.labelsize" : 11})
plt.pie(usecovalues[:10], labels=usecolabels[:10], explode = explode, autopct='%1.1f%%', pctdistance = 0.9,
          rotatelabels = 90, startangle=140, labeldistance = 1.05) 


# Even our US economic sector sample contained only 120 868 non-missing values, this somehow confirms that IT and Advanced Manufacturing are the most convenient sectors for applying foreigners. As a next step in our EDA, let's take a look at the most desired job titles, citizenships and class of admission of our Visa applicants.  

# In[ ]:


df['job_info_job_title'].value_counts()[:20]


# Since our column contains job titles with different letter casing we need to standarize them so that
# value_counts() method will be able to count them more appropriately. Also, there are lots of same positions 
# like "Computer Systems Analyst" which differ only by the number standing after hyphen so we will 
# split these titles by finding the '-', 'ii' and '/' signs and leaving only the left side of the splitting 
# result. Afterwards, we are going to remove leading and ending spaces, replace "sr." with "senior" values and get rid of  'nan's.

# In[ ]:


#Converting values to lower case
df['job_info_job_title'] = df['job_info_job_title'].str.lower()

#Splitting job titles by '-'
df['job_info_job_title'] = df['job_info_job_title'].astype(str).str.split('-').str[0]
#Splitting job titles by 'ii'
df['job_info_job_title'] = df['job_info_job_title'].astype(str).str.split('ii').str[0]
#Splitting job titles by '/'
df['job_info_job_title'] = df['job_info_job_title'].astype(str).str.split('/').str[0]
#Removing leading and ending spaces
df['job_info_job_title'] = df['job_info_job_title'].astype(str).str.strip()
#Replacing "sr." values with "senior"
df['job_info_job_title'] = df['job_info_job_title'].str.replace('sr.', 'senior')
#Replacing "NaN", "NaT" and "nan" values with np.nan
df['job_info_job_title'].replace(["NaN", 'NaT','nan'], np.nan, inplace = True)


df['job_info_job_title'].value_counts(dropna=True)[:10]


# In[ ]:


#Setting plot parameters
fig, ax = plt.subplots()
fig.set_size_inches(12.7, 8.27)
 #sns.set_context(rc={"font.size":14, "axes.labelsize":12})
sns.countplot(x='job_info_job_title',data=df, 
               palette = sns.diverging_palette(255, 133, l=60, n=10, center="dark"), 
               order=df.job_info_job_title.value_counts().iloc[:10].index)
plt.xticks(rotation=90)

#Iterating over elements in "job_info_job_title" column and displaying counts above bars 
for i, v in df.job_info_job_title.value_counts().head(10).reset_index().iterrows():
    ax.text(i,v.job_info_job_title,v.unique()[1], horizontalalignment='center',verticalalignment='bottom')

#Setting label titles    
ax.set(xlabel='Job Title', ylabel='Number of Visa applications')


# Interestingely, all of the most popular positions except  "assistant professor" are derived from the IT industry. This is another confirmation that there is a huge demand for IT specialists in USA and being one of them increases our chances to obtain a permanent Visa. 

# In[ ]:


#Setting plot parameters
fig, ax = plt.subplots()
fig.set_size_inches(12.7, 8.27)
sns.set_context(rc={"font.size":14, "axes.labelsize":12})
sns.countplot(x='country_of_citizenship',hue='case_status',data=df, 
              palette = sns.diverging_palette(255, 133, l=60, n=7, center="dark"), 
              order=df.country_of_citizenship.value_counts().iloc[:7].index)
plt.xticks(rotation=90)

#Iterating over elements in "country_of_citizenship" column and displaying counts above bars 
for i, v in df.country_of_citizenship.value_counts().head(7).reset_index().iterrows():
    ax.text(i,v.country_of_citizenship,v.unique()[1], horizontalalignment='right',verticalalignment='bottom')

#Setting label titles    
ax.set(xlabel='Country of citizenship', ylabel='Number of Visa applications')


# As we can see, the majority of Visa applications has been submitted by Indian citizens. They constitute to more than half of our observations, we can assume that most of them are computer specialists.

# In[ ]:


#Setting plot parameters
fig, ax = plt.subplots()
fig.set_size_inches(12.7, 8.27)
sns.countplot(x='class_of_admission',data=df, 
              order=df.class_of_admission.value_counts().iloc[:10].index)
plt.xticks(rotation=90)

#Iterating over elements in "class_of_admission" column and displaying counts above bars 
for i, v in df.class_of_admission.value_counts().head(10).reset_index().iterrows():
    ax.text(i,v.class_of_admission,v.unique()[1], horizontalalignment='center',verticalalignment='bottom')
    
ax.set(xlabel='Visa type', ylabel='Number of Visa applications')


# The vast majority of petitioners were applying for the H-1B Visa, which according to the Wikipedia, allows U.S. employers to employ foreign workers in specialty occupations. If a foreign worker in H-1B status quits or is dismissed from the sponsoring employer, the worker must either apply for and be granted a change of status, find another employer (subject to application for adjustment of status and/or change of visa), or leave the United States. 
# 
# 
# Finally, let's try checking on the number and kind of application types. Unfortunately, our data consists only of 126 848 non-missing values for this attribute, but this should give us a general overview.

# In[ ]:


#Setting plot parameters
fig, ax = plt.subplots()
fig.set_size_inches(10.7, 7.27)
sns.countplot(x='application_type', data=df, palette = sns.color_palette("GnBu_d"), order=df.application_type.value_counts().iloc[:10].index)

#Iterating over elements in "application_type" column and displaying counts above bars 
for i, v in df.application_type.value_counts().head(10).reset_index().iterrows():
    ax.text(i,v.application_type,v.unique()[1], horizontalalignment='center',verticalalignment='bottom')
    
ax.set(xlabel='Application type', ylabel='Number of Visa applications')


# Online submission was the most popular form of application type. Here, we can also find "PERM" value which is probably incorrect. My assumtion is that some petitioners thought about this form field as a distinction between "temporary" and "permanent" Visa type. The last plotting activity will be displaying the applicants education level and remuneration.

# In[ ]:


#Setting plot parameters
fig, ax = plt.subplots()
fig.set_size_inches(12.7, 8.27)
 #sns.set_context(rc={"font.size":14, "axes.labelsize":12})
sns.countplot(x='foreign_worker_info_education',data=df, 
               palette = sns.color_palette("Paired"), 
               order=df.foreign_worker_info_education.value_counts().iloc[:10].index)

#Iterating over elements in "job_info_job_title" column and displaying counts above bars 
for i, v in df.foreign_worker_info_education.value_counts().head(10).reset_index().iterrows():
    ax.text(i,v.foreign_worker_info_education,v.unique()[1], horizontalalignment='center',verticalalignment='bottom')

#Setting label titles    
ax.set(xlabel='Education level', ylabel='Number of Visa applications')


# As we can see, over 50% of applicants obtained a university degree. Before plotting the remuneration, we will remove commas from the values so that they are left only with decimal places denoted. Also, since some of the wages are hourly, weekly, bi-weekly and monthly values, we have to calculate the yearly equivalents for them. According to the https://www.timeanddate.com/date/workdays.html website, the average number of working days in USA is 250. We will use this information in our calculations.

# In[ ]:


df[['pw_amount_9089','pw_unit_of_pay_9089']].head(10)


# In[ ]:


#Replacing commas with whitespace character
df['pw_amount_9089'] = df['pw_amount_9089'].str.replace(",","") 

for unit in df.pw_unit_of_pay_9089.unique():
    if unit == "hr" or unit == "Hour":
        df.loc[df['pw_unit_of_pay_9089'] == unit, 'pw_amount_9089'] = df['pw_amount_9089'].apply(lambda x: float(x) * 8 * 250)
        df.loc[df['pw_unit_of_pay_9089'] == unit, 'pw_unit_of_pay_9089'] = df['pw_unit_of_pay_9089'].replace(to_replace = unit, value = "Year") 
    elif unit == "wk" or unit == "Week":
        df.loc[df['pw_unit_of_pay_9089'] == unit, 'pw_amount_9089'] = df['pw_amount_9089'].apply(lambda x: float(x) * 50)
        df.loc[df['pw_unit_of_pay_9089'] == unit, 'pw_unit_of_pay_9089'] = df['pw_unit_of_pay_9089'].replace(to_replace = unit, value = "Year")
    elif unit == "mth" or unit == "Month":
        df.loc[df['pw_unit_of_pay_9089'] == unit, 'pw_amount_9089'] = df['pw_amount_9089'].apply(lambda x: float(x) * 12)
        df.loc[df['pw_unit_of_pay_9089'] == unit, 'pw_unit_of_pay_9089'] = df['pw_unit_of_pay_9089'].replace(to_replace = unit, value = "Year")
    elif unit == "bi" or unit == "Bi-Weekly":  
        df.loc[df['pw_unit_of_pay_9089'] == unit, 'pw_amount_9089'] = df['pw_amount_9089'].apply(lambda x: float(x) * 25)
        df.loc[df['pw_unit_of_pay_9089'] == unit, 'pw_unit_of_pay_9089'] = df['pw_unit_of_pay_9089'].replace(to_replace = unit, value = "Year")
    elif unit =="yr":
         df.loc[df['pw_unit_of_pay_9089'] == unit ,'pw_unit_of_pay_9089'] = df['pw_unit_of_pay_9089'].replace(to_replace = unit, value = "Year")
    else:
        continue
        
#Filling missing values with median 
df['pw_amount_9089']= df['pw_amount_9089'].fillna((df['pw_amount_9089'].median()))

#Changing format from string to float
df['pw_amount_9089'] = df.pw_amount_9089.astype(float)

#Displaying 10 first values
df[['pw_amount_9089','pw_unit_of_pay_9089']].head(10)


# In[ ]:


#Since running "describe" method on "pw_amount_9089" column returned exponential values, I decided to 
#convert them to floats so that they are easier to understand
with pd.option_context('float_format', '{:.2f}'.format): print(df.pw_amount_9089.describe())


# In[ ]:


#Dividing our continuous income values into some categories to facilitate their visualization
df['remuneration'] = pd.cut(df['pw_amount_9089'], [0, 30000, 60000,90000,120000,150000,180000,210000,240000,270000,495748000], right=False, labels=["0-30k", "30-60k","60-90k","90-120k","120-150k","150-180k","180-210k","210-240k","240-270k","270k+"])
salary = df['remuneration'].value_counts()
salary.iloc[np.argsort(salary.index)]


# In[ ]:


# Draw a count plot to show the distribution of remunerations
g = sns.factorplot(x='remuneration', data=df, kind="count",
                   palette="BuPu", size=9, aspect=1.2)

g.set(xlabel='Remuneration', ylabel='Number of applicants')


# As we can see, over 65% of the applicants earn between 60 and 120 thousand dollars yearly.  From this moment, we will start working on the feature selection and data cleansing.

# ## Feature selection and data cleansing

# In[ ]:


#Displaying percentage of non-null values for each feature
i = 0;
for col in df.columns:
    i = i+1;
    print (i-1,"Column: '{}'".format(col),"contains ", np.round(100*df[col].count()/len(df['case_status']),decimals=2),"% non-null values" )


# In[ ]:


#Leaving columns which have more than 330000 non-missing observations
df = df.loc[:,df.count() >= 330000]
df.info()


# Since our dataset consists of 19 attributes which have less than 12% of missing values , we will choose some of them for further analysis and perform imputations. 

# In[ ]:


#Indices of selected features
chosen_attrs = [0,1,2,5,6,8,12,14,17,18]
df = df.iloc[:,chosen_attrs]


# In[ ]:


#Assigning Labels to Case Status
df.loc[df.case_status == 'Certified', 'case_status'] = 1
df.loc[df.case_status == 'Denied', 'case_status'] = 0

#Filling missing values in "employer_state" column with mode
df['employer_state'] = df['employer_state'].fillna(df['employer_state'].mode()[0]);

#Mapping from state name to abbreviation
state_abbrevs = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY',
    'Northern Mariana Islands':'MP', 
    'Palau': 'PW', 
    'Puerto Rico': 'PR', 
    'Virgin Islands': 'VI', 
    'District of Columbia': 'DC'
}

#Capitalizing Keys
us_state_abbrev = {k.upper(): v for k, v in state_abbrevs.items()}
df['employer_state'].replace(us_state_abbrev, inplace=True)
df.employer_state = df.employer_state.astype(str)


# In[ ]:


df['pw_soc_code'] = df['pw_soc_code'].str.replace('.','')
df['pw_soc_code'] = df['pw_soc_code'].str.replace('-','')
df['pw_soc_code'] = df['pw_soc_code'].astype(str).str[0:6]
df['pw_soc_code'].value_counts()

#Finding "nan" values in "pw_soc_code" column and filling them with mode
df.loc[df['pw_soc_code'] == "nan",'pw_soc_code'] = df['pw_soc_code'].mode()[0]

#Finding "None" values in "pw_soc_code" column and filling them with mode
df.loc[df['pw_soc_code'] == "None",'pw_soc_code'] = df['pw_soc_code'].mode()[0]

#Changing type from string to int
df['pw_soc_code'] = df['pw_soc_code'].astype(int)
df['case_status'] = df['case_status'].astype(int)


# In[ ]:


#Replacing missing values with mode
df['class_of_admission']=df['class_of_admission'].fillna((df['class_of_admission'].mode()[0]))
df['country_of_citizenship']=df['country_of_citizenship'].fillna((df['country_of_citizenship'].mode()[0]))
df['employer_city']=df['employer_city'].fillna((df['employer_city'].mode()[0]))
df['employer_name']=df['employer_name'].fillna((df['employer_name'].mode()[0]))
df['employer_name']=df['employer_name'].astype(str).str.upper()
df['pw_source_name_9089']=df['pw_source_name_9089'].fillna((df['pw_source_name_9089'].mode()[0]))
df['remuneration']=df['remuneration'].fillna((df['remuneration'].mode()[0]))


# In[ ]:


df.info()


# ## Data type conversion

# In this step we're going to turn our feature variables into categories.

# In[ ]:


from sklearn.preprocessing import LabelEncoder
categorical_variables = {}

#Creating categories denoted by integers from column values
for col in df.columns:
    cat_var_name = "cat_"+ col
    cat_var_name = LabelEncoder()
    cat_var_name.fit(df[col])
    df[col] = cat_var_name.transform(df[col])
    categorical_variables[col] = cat_var_name

df.info()


# # Applying Machine Learning algorithms

# First thing we're going to do in this part of our analysis will be dividing our final dataset into 2 dataframes. First one will consist of feature variables and the second one only of our target variable - case_status. Afterward we will
# use GridSearch object with cross-validation to find the best parameters for Logistic Regression, k-Nearest Neighbor, Random Forest and Gradient Boosting Classifiers and evaluate how well they will generalize. Cross validation will split the data repeatedly using Stratified K-Folds cross-validator and train multiple models. 

# In[ ]:


#Dividing our final dataset into features(explanatory variables) and labels(target variable)
X = df.loc[:, df.columns != 'case_status']
y = df.case_status

print("The shape of X is: {}".format(X.shape))
print("The shape of y is: {}".format(y.shape))


# ## Logistic Regression 

# In[ ]:


#Importing Logistic Regression Classifier, GridSearchCV, train_test_split and accuracy metrics from sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

#Defining fit_algorithm function
def fit_algorithm(alg, X, y, parameters, cv = 5):
    """
    This function will split our dataset into training and testing subsets, fit cross-validated 
    GridSearch object, test it on the holdout set and return some statistics
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 123)
    grid = GridSearchCV(alg, parameters, cv = cv)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    confmat = confusion_matrix(y_test,y_pred)

    return pd.Series({
        "Train_ACC": np.around(grid.best_score_, decimals=2).astype(str),
        "Test_ACC": np.around(grid.score(X_test, y_test), decimals=2).astype(str),
        "P": np.around(precision_score(y_pred, y_test), decimals=2).astype(str),
        "R": np.around(recall_score(y_pred, y_test),decimals=2).astype(str),
        "F1": np.around(f1_score(y_pred, y_test),decimals=2).astype(str),
        "Best_params": [grid.best_params_],
        "True negatives": confmat[0,0],
        "False negatives": confmat[1,0],
        "True positives": confmat[1,1],
        "False positives": confmat[0,1]
        })


# In[ ]:


#To perform hyper parameter optimisation a list of multiple elements will be entered and the optimal 
#value in that list will be picked using Grid Search object
logreg_params = {'C': [0.001, 0.01, 0.1, 1, 10, 100] }

logreg = fit_algorithm(LogisticRegression(),X,y,logreg_params)
logreg


# As we can see, the Logistic Regression Classifier results in 93% accuracy on both training and testing datasets which is quite a good score. This result was achieved using default "liblinear" solver algorithm and L2 regularization with C parameter = 1.  Now, let's assess the effectiveness of k-Nearest Neighbors algorithm.

# ## k-Nearest Neighbors

# In[ ]:


#Importing k-Nearest Neighbors Classifier
from sklearn.neighbors import KNeighborsClassifier

#Defining range of parameters for kNN Clssifier
knn_params = {'n_neighbors': np.arange(1,11).tolist()}

#Using "fit_algorithm" function with kNN Classifier
knn = fit_algorithm(KNeighborsClassifier(),X,y,knn_params)
knn


# Interestingly, the k-Nearest Neighbors Classifier achieved the same accuracy score as the Logistic Regression Classifier. Even accuracy is the same, there are slight differences in Precision, Recall, True Positive, True Negative, False Positive and False Negative values. Now, let's check, how precise the Random Forest Classifier can be. 

# ## Random Forests

# In[ ]:


#Defining range of parameters for Random Forest Clssifier
forest_params = {'n_estimators': [10,20,30,40,50],
     'max_depth': [15,20,25,30],
     'max_features': [2,3,4],
     'random_state': [123],
     'n_jobs': [-1]
    }
    
# #Importing RandomForestClassifier from sklearn
from sklearn.ensemble import RandomForestClassifier

forest = fit_algorithm(RandomForestClassifier(),X,y, forest_params)
forest


# As we can see, the Random Forest Classifier achieved the accuracy of 94% which is 1% better than Logistic Regression and k-Nearest Neighbors Classifiers. The distribution of True Positives, True Negatives, False Positives and False Negatives is also different. We could try to build more precise model by adjusting the hyperparameters, but we should bear in mind that this is very computationally expensive. In my case, I tried supplying the above Random Forest Classifier with more hyperparameter values, but the Kernel has been killed multiple times so I had to narrow down their number. As the last ML algorithm, we will try Gradient Boosted Machines which are another ensemble method implemented in scikit-learn. The main idea behind this algorithm is to combine multiple decision trees which in contrast to these used in Random Forest classifier are working in a serial manner, where each tree tries to correct the mistakes of the previous one.  

# ## Gradient Boosted Regression Trees (Gradient Boosting Machines)

# In[ ]:


#Importing GradientBoostingClassifier from sklearn
from sklearn.ensemble import GradientBoostingClassifier

#Defining range of parameters for Gradient Boosting Clssifier
gradient_params = {'n_estimators': [100],
     'max_depth': [3],
     'random_state': [123],
     'learning_rate': [0.1]
    }

gradient = fit_algorithm(GradientBoostingClassifier(),X,y,gradient_params)
gradient


# As previously, we achieved similar accuracy of 93% on both training and testing datasets, but this result could be higher, if we tried different values of hyperparameters. The main idea behind gradient boosting is to combine many simple models(trees) where each tree can only provide good predictions on part of the data, and so more and more trees are added to iteratively improve performance. Gradient Boosted Trees are frequently winning entries in machine learning competitions. Except from the number of trees in the ensemble, another important parameter for this algorithm is "learning_rate" which controls how strongly each tree tries to correct the mistakes of the previous trees. Since Random Forests and Gradient Boosted Machines are very computationally expensive and Kaggle kills kernels running longer than 1 hour, I strongly encourage you to test all of  the abovely shown algorithms on your locally set Jupyter Notebooks with broader range of hyperparameter values- it's very likely that they will be able to achieve better accuracy.
# 
# Finally, since the Random Forest Classifier achieved best accuracy on both training and testing datasets I'll built a model using parameter values chosen by Grid Search object and display feature importances so that we will find out which feature is the most important one in obtatining a US Permanent Visa. 

# In[ ]:


# Dataframe made of results 
summary = pd.concat([logreg,knn,forest,gradient],axis=1)
summary.columns = ['Logistic Regression', 'k-Nearest Neighbors','Random Forest','GBoosted Machines']
summary 


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 123)
final_forest = RandomForestClassifier(n_estimators = 50, max_depth = 20, max_features = 4, random_state = 123, n_jobs = -1)
final_forest.fit(X_train, y_train)
importances = final_forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in final_forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Plot the feature importances of the forest
plt.figure(figsize=(10, 8))
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
       color="g", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), X_train.columns, rotation = 90)
plt.xlim([-1, X_train.shape[1]])
plt.show()                                                  


# According to the feature importances returned by Random Forest Classifier all of the chosen variables are significant for predicting Visa application decisions. Most informative features are: class of admission(Visa type), country of citizenship and employer related details like location, name and state. 
