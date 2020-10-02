#!/usr/bin/env python
# coding: utf-8

# <h2> <u> Data Cleaning - India Air Quality Data

# #### Import Libraries

# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# ---

# > #### Import Data

# In[ ]:


df = pd.read_csv('/kaggle/input/india-air-quality-data/data.csv' , encoding='mac_roman')


# ---

# #### Data Description
stn_code
sampling_date
state          : State
location       : City
agency<br>
typeArea       : category
so2            : Sulphur dioxde
no2            : Nitrogen dioxide
rspm           : Respirable Suspended Particulate Matter
spm            : Suspended Particulate Matter
location_monitoring_station : location of monitoring area
pm2_5          : PSI 2.5
date           : Date of recording
# In[ ]:


df.head()


# ---

# In[ ]:


df.info()


# <h5> There are a lot of Missing Values in the data

# In[ ]:


missing_ratio = np.round(df.isna().sum()/len(df)*100,2)
missing_ratio = pd.DataFrame(missing_ratio , columns=['Missing_Ratio'])
missing_ratio.sort_values('Missing_Ratio',ascending=False)


# <h5>The above information shows that the Missing Ratio is very high for some variables<br>
#     Since 'pm_2' has a Missing Ratio of 97.86% and 'spm' has a Missing Ratio of 54.48%, we have to drop the Columns,<br>
#     Other Columns can be treated

# ---

# <h3> <u>  Dropping columns that are not necessary 

# In[ ]:


list(df.columns)


# <h5> These Columns can be dropped from the dataset<br>
#      - 'stn_code' : As it is just a code and is not important for the Analysis <br>
#      - 'sampling_date' : There are two date columns in this dataset, we are dropping this column because the values in this column are in different formats<br>
#      - 'pm_2' : Since more than 97% of data is Missing 
#      - 'spm' : Since more than 50% of data is Missinf

# In[ ]:


df.drop(['stn_code','sampling_date','pm2_5','spm'] , axis=1 , inplace=True)


# ---

# <h3> <u> Getting Categorical and Numerical Columns

# In[ ]:


cat_cols = list(df.select_dtypes(include=['object']).columns)
num_cols = list(df.select_dtypes(exclude=['object']).columns)


# In[ ]:


print('\nNumerical Columns : ' , num_cols)
print('\nCategorical Columns : ' , cat_cols)


# ---

# <h3> <u> Handeling Categorical columns 

# <h4> <u> 'date' - column

# The 'date' column in an object type , we need to convert it to datetime type

# In[ ]:


df['date'] = pd.to_datetime(df['date'])


# Missing Values

# In[ ]:


df['date'].isna().sum()


# Since this column has only 7 Missing Values , we can drop the observations

# In[ ]:


df = df[df['date'].isna()==False]


# <br>

# <h4> <u> 'type' - column

# <h5> Missing Value Treatment

# In[ ]:


print('The number of missing values are : ',df['type'].isna().sum())


# Since there are many missing values , we will replace the missing values as 'NA' (Not Available)

# In[ ]:


df['type'] = df['type'].fillna('NA')


# In[ ]:


df['type'].value_counts()


# <h5>From the above information we can see that data belonging to the same Type of Area is given different names
# <br>We need replace the different names that means the same Type of Area
# <br> The Types of Area can be categorized together as : 'Residential,Rural and Others(RRO)' , 'Industrial' , 'Sensitive' & 'NA'<br>

# In[ ]:


res_str='Residential|RIRUO'
ind_str = 'Industrial'
sen_str = 'Sensitive'

rro_mask = df['type'].str.contains(res_str , regex=True)
ind_mask = df['type'].str.contains(ind_str)
sen_mask = df['type'].str.contains(sen_str)


# In[ ]:


df['type'][rro_mask] = 'RRO'
df['type'][ind_mask] = 'Industrial'
df['type'][sen_mask] = 'Sensitive'


# In[ ]:


df['type'].value_counts()


# <h5>We have now categorised the 'type' column into 4 main categories

# <h4> <u> 'agency' - column

# In[ ]:


print('The number of missing values are : ',df['agency'].isna().sum())


# A huge part of this data is missing so we will treat the missing values as a new category call it 'NA' (Not Available)

# In[ ]:


df['agency'].fillna('NA',inplace=True)


# <h4> <u> 'location monitoring station' - column

# In[ ]:


print('The number of missing values are : ',df['location_monitoring_station'].isna().sum())


# A huge part of this data is missing so we will treat the missing values as a new category call it 'NA' (Not Available)

# In[ ]:


df['location_monitoring_station'].fillna('NA',inplace=True)


# <h4> <u> 'location' - column

# In[ ]:


print('The number of missing values are : ',df['location'].isna().sum())


# <h4> <u> 'state' - column

# In[ ]:


print('The number of missing values are : ',df['state'].isna().sum())


# <h5> At this point we have cleaned all the Categorical Columns in the dataset 

# ---

# <h3> <u> Handeling Numerical columns 

# In[ ]:


num_cols


# <h4> <u> 'so2' - column

# In[ ]:


df['so2'].describe()


# The above information shows that 75% of the values are within the value 13.7 and the maximum value is 909.0<br>
# This means that there are outliers 

# In[ ]:


print('Distribution of SO2')
fig,ax=plt.subplots(1,2,figsize=(13,4))
sns.distplot(df['so2'].dropna() , ax=ax[0])
sns.boxplot(df['so2'].dropna() , ax=ax[1])

so2_skew = df['so2'].skew()
plt.show()
print('Skewness = ',so2_skew)


# The above distribution plot shows that 'so2' values are significantly Right Skewed(Positively Skewed)<br>
# and the box plot shows that there are significant amount of outliers

# ##### Removing Outliers

# Since we saw that 75% of the values lie under 13.7<br>
# we can calculate the upper limit using IQR(Inter Quartile Range) and <br>
# consider values outside upper limit as outliers and remove them

# In[ ]:


Q1=df['so2'].quantile(0.25)
Q3=df['so2'].quantile(0.75)
IQR=Q3-Q1
df=df[~((df['so2']<(Q1-1.5*IQR))|(df['so2']>(Q3+1.5*IQR)))]


# In[ ]:


print('Distribution of SO2')
sns.distplot(df['so2'].dropna())

so2_skew = df['so2'].skew()
plt.show()
print('Skewness = ',so2_skew)


# The above distribution plot shows the significant Positive Skewness has been reduced to acceptable level

# ##### Treating Missing Values

# In[ ]:


print('The number of missing values in SO2 are : ' , df['so2'].isna().sum())


# In[ ]:


print('Distribution of SO2')
sns.kdeplot(df['so2'].dropna())
plt.axvline(df['so2'].mean(), color='r')
plt.axvline(df['so2'].median(), color='g')

plt.legend(['So2','Mean','Median'])
plt.show()


# In[ ]:


df1= df.copy()
df2=df.copy()


# Mean Imputation

# In[ ]:


df1['so2'] = df1['so2'].fillna(df1['so2'].mean())


# Forward Fill

# In[ ]:


df2['so2'] = df2['so2'].fillna(method='ffill')


# In[ ]:


print('Distribution of SO2')

fig,ax=plt.subplots(1,2,figsize=(13,4))

sns.kdeplot(df1['so2'] , ax=ax[0])
ax[0].axvline(df1['so2'].mean(), color='r' )
ax[0].axvline(df1['so2'].median(), color='g')
ax[0].set_title('Mean Imputation')  
ax[0].legend(['So2','Mean','Median'])

sns.kdeplot(df2['so2'] , ax=ax[1])
ax[1].axvline(df2['so2'].mean(), color='r')
ax[1].axvline(df2['so2'].median(), color='g')
ax[1].set_title('Forward Fill')
ax[1].legend(['So2','Mean','Median'])
                    
                    
plt.show()


# The above plots show that filling the missing values with Forward Fill dosen't cause any variation on the data, so we can choose Forward Fill

# In[ ]:


df['so2'] = df['so2'].fillna(method='ffill')


# <br><br>

# <h4> <u> 'no2' - column

# In[ ]:


df['no2'].describe()


# The above information shows that 75% of the values are within the value 30.8 and the maximum value is 876.0<br>
# This means that there are outliers 

# In[ ]:


print('Distribution of NO2')
fig,ax=plt.subplots(1,2,figsize=(13,4))
sns.distplot(df['no2'].dropna() , ax=ax[0])
sns.boxplot(df['no2'].dropna() , ax=ax[1])
no2_skew = df['no2'].skew()
plt.show()
print('Skewness = ',no2_skew)


# The above distribution plot shows that 'no2' values are significantly Right Skewed(Positively Skewed)<br>
# and the box plot shows that there are significant amount of outliers

# ##### Removing Outliers

# Since we saw that 75% of the values lie under 31.0<br>
# we can calculate the upper limit using IQR(Inter Quartile Range) and <br>
# consider values outside upper limit as outliers and remove them

# In[ ]:


Q1=df['no2'].quantile(0.25)
Q3=df['no2'].quantile(0.75)
IQR=Q3-Q1
df=df[~((df['no2']<(Q1-1.5*IQR))|(df['no2']>(Q3+1.5*IQR)))]


# In[ ]:


print('Distribution of NO2')
fig,ax=plt.subplots(1,2,figsize=(13,4))
sns.distplot(df['no2'].dropna() , ax=ax[0])
sns.boxplot(df['no2'].dropna() , ax=ax[1])
no2_skew = df['no2'].skew()
plt.show()
print('Skewness = ',no2_skew)


# The above distribution plot shows the significant Positive Skewness has been reduced to acceptable level

# ##### Treating Missing Values

# In[ ]:


print('The number of missing values in NO2 are : ' , df['no2'].isna().sum())


# In[ ]:


print('Distribution of NO2')
sns.kdeplot(df['no2'])
plt.axvline(df['no2'].mean(), color='r')
plt.axvline(df['no2'].median(), color='g')
plt.legend(['No2','Mean','Median'])
plt.show()


# In[ ]:


df1 = df.copy()
df2 = df.copy()


# In[ ]:


#Mean Imputation
df1['no2'] = df1['no2'].fillna(df1['no2'].mean())
#Forward Fill
df2['no2'] = df2['no2'].fillna(method='ffill')


# In[ ]:


print('Distribution of NO2')

fig,ax=plt.subplots(1,2,figsize=(13,4))

sns.kdeplot(df1['no2'] , ax=ax[0])
ax[0].axvline(df1['no2'].mean(), color='r' )
ax[0].axvline(df1['no2'].median(), color='g')
ax[0].set_title('Mean Imputation')    
ax[0].legend(['No2','Mean','Median'])

sns.kdeplot(df2['no2'] , ax=ax[1])
ax[1].axvline(df2['no2'].mean(), color='r')
ax[1].axvline(df2['no2'].median(), color='g')
ax[1].set_title('Forward Fill')
ax[1].legend(['no2','Mean','Median'])
                    
                    
plt.show()


# The above plots show that filling the missing values with Forward Fill dosen't cause any variation on the data, so we can choose Forward Fill

# In[ ]:


df['no2'] = df['no2'].fillna(method='ffill')


# <br><br>

# <h4> <u> 'rspm' - column

# In[ ]:


df['rspm'].describe()


# The above information shows that 75% of the values are within the value 135.0 and the maximum value is 6307.03<br>
# This means that there are outliers 

# In[ ]:


print('Distribution of RSPM')
fig,ax=plt.subplots(1,2,figsize=(13,4))
sns.distplot(df['rspm'].dropna() , ax=ax[0])
sns.boxplot(df['rspm'].dropna() , ax=ax[1])
plt.show()
print('Skewness = ',df['rspm'].skew())


# The above distribution plot shows that 'rspm' values are significantly Right Skewed(Positively Skewed)<br>
# and the box plot shows that there are significant amount of outliers

# ##### Removing Outliers

# Since we saw that 75% of the values lie under 135.0
# we can calculate the upper limit using IQR(Inter Quartile Range) and
# consider values outside upper limit as outliers and remove them

# In[ ]:


Q1=df['rspm'].quantile(0.25)
Q3=df['rspm'].quantile(0.75)
IQR=Q3-Q1
df=df[~((df['rspm']<(Q1-1.5*IQR))|(df['rspm']>(Q3+1.5*IQR)))]


# In[ ]:


print('Distribution of RSPM')
fig,ax=plt.subplots(1,2,figsize=(13,4))
sns.distplot(df['rspm'].dropna() , ax=ax[0])
sns.boxplot(df['rspm'].dropna() , ax=ax[1])
plt.show()
print('Skewness = ',df['rspm'].skew())


# The above plots show that significant Positiv Skwewness is reduced and the outliers are removed

# ##### Treating Missing Values

# In[ ]:


print('The number of missing values in RSPM are : ' , df['rspm'].isna().sum())


# In[ ]:


df1 = df.copy()
df2 = df.copy()


# In[ ]:


#Mean Imputation
df1['rspm'] = df1['rspm'].fillna(df1['rspm'].mean())
#Forward Fill
df2['rspm'] = df2['rspm'].fillna(method='ffill')


# In[ ]:


print('Distribution of RSPM')

fig,ax=plt.subplots(1,2,figsize=(13,4))

sns.kdeplot(df1['rspm'] , ax=ax[0])
ax[0].axvline(df1['rspm'].mean(), color='r' )
ax[0].axvline(df1['rspm'].median(), color='g')
ax[0].set_title('Mean Imputation')    
ax[0].legend(['rspm','Mean','Median'])

sns.kdeplot(df2['rspm'] , ax=ax[1])
ax[1].axvline(df2['rspm'].mean(), color='r')
ax[1].axvline(df2['rspm'].median(), color='g')
ax[1].set_title('Forward Fill')
ax[1].legend(['rspm','Mean','Median'])
                    
                    
plt.show()


# The above plots show that filling the missing values with Forward Fill dosen't cause any variation on the data, so we can choose Forward Fill

# In[ ]:


df['rspm'] = df['rspm'].fillna(method='ffill')


# ---
