#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Dataset Overview 
# This data set is about the global warming co2 emissions by various countries in the time period  between 1960 to 2014. The data has information about low income and high income countries and countries are  categorized by regions. The data is sourced from this kernel : Who is responsible for the global warming? by Author: Boryana Aleksova. This gives more information about the Kyoto protocol to which few countries have committed to control their CO2 emissions. 
# 

# ### Preliminary data assessment

# In[ ]:


#Read in the data
df = pd.read_csv("../input/API_EN.ATM.CO2E.PC_DS2_en_csv_v2_10576797.csv")


# Need to add the second data file, the 'Metadata.csv' for more information on the income level of the countries
# according to the information provided in the original abstract by Author: Boryana Aleksova 
# As per the documentation, the co2 is measured in metric tons per capita.

# In[ ]:


meta_data= pd.read_csv('../input/Metadata_Country_API_EN.ATM.CO2E.PC_DS2_en_csv_v2_10576797.csv')


# In[ ]:


df.shape


# In[ ]:


meta_data.info()


# In[ ]:


#Look at the first few rows
df.head()


# In[ ]:


#Look at the column names
df.columns


# In[ ]:


#How many unique values does Inidcator code have?
df['Unnamed: 63'].tail()


# In[ ]:


df.info()


# #### Cleaning

# In[ ]:


#Drop unnecessary columns
df_dropped =df.drop(['Indicator Name','Indicator Code', 'Unnamed: 63'], axis = 1)


# In[ ]:


#Rename the required columns
   df_dropped.rename(columns={'Country Name':'country_name','Country Code':'country_code'}, inplace = True)    


# In[ ]:


meta_data.rename(columns={'Country Code': 'country_code','IncomeGroup':'incomegroup','SpecialNotes':'specialnotes',
                           'TableName':'tablename'}, inplace = True)


# In[ ]:


df_dropped.shape


# In[ ]:


#Drop the unwanted column Unnamed5 which has all nan values
meta_data.drop('Unnamed: 5', axis = 1, inplace = True)


# In[ ]:


meta_data.columns


# In[ ]:


meta_data.shape


# In[ ]:


#Check if some country code from the main data  is not in the metadata 
df_dropped[~df_dropped['country_code'].isin(meta_data['country_code'])]


# In[ ]:


#Remove the above row
df_dropped = df_dropped.drop(df_dropped.index[108])


# In[ ]:


df_dropped.shape


# In[ ]:


#Now merge both the datas
merged_byincome= pd.merge(df_dropped, meta_data, on ='country_code', how ='inner')


# In[ ]:


#Check 
merged_byincome.shape


# In[ ]:


merged_byincome.head()


# In[ ]:


merged_byincome.info()


# Need to convert the year from float to integer datatype and then melt the data frame to get the correct structure 
# for tidyness so that the co2 values are as one observation  for each year.

# In[ ]:


merged_byincome.columns


# In[ ]:


# Reshape data
 byincome_melted= pd.melt(merged_byincome, id_vars=['country_name','country_code', 'Region', 'incomegroup','specialnotes',
                                             'tablename'], var_name= 'year', value_name= 'co2_emissions')


# In[ ]:


byincome_melted.shape


# In[ ]:


#Year column are float, change the data type to date to int
byincome_melted['year']= byincome_melted['year'].astype('int')


# In[ ]:


#Check for duplicates
byincome_melted.duplicated().sum()


# In[ ]:


#Check the new reshaped dataframe : SEE some null values
byincome_melted.info()


# In[ ]:


#Check for missing values
byincome_melted.isnull().sum()


# In[ ]:


byincome_melted.head()


# In[ ]:


byincome_melted.describe()


# In[ ]:


# DRop the data with null values 

   byincome_melted.dropna(inplace = True)


# In[ ]:


byincome_melted.shape


# NOW the data is cleaned.

# In[ ]:


byincome_melted.head()


# In[ ]:


byincome_melted.describe()


# In[ ]:


#We see one negative value for the co2 emission which is not correct as per the information provided
#Check
byincome_melted.loc[byincome_melted['co2_emissions'] <0]


# In[ ]:


byincome_melted.loc[byincome_melted['co2_emissions'] < 0 , 'co2_emissions']= 0


# In[ ]:


#Verify any row does not exist with vale less than zero
byincome_melted.loc[byincome_melted['co2_emissions'] <0]


# In[ ]:


byincome_melted.head()


# Questions I have of this data
# * How does the distribution of Co2 emission look like?
# * Have Co2 emissions increased over time
# * How does co2 emission for low income countries compare to high income countries?
# * How do the co2 emission differ between the regions?

#  ## Exploratory analysis
#   Univariate analysis

# In[ ]:


#Let us first look at the CO2 emission distribution 

byincome_melted.describe()


# In[ ]:


#Export the dataframe to a csv file 
byincome_melted.to_csv('global_warming.csv', index= False, header =True)


# In[ ]:


binsize = 5
bins= np.arange(0, byincome_melted['co2_emissions'].max() + 5 , binsize)
plt.figure(figsize=[8,5])
plt.hist(data= byincome_melted, x= 'co2_emissions', bins = bins)
plt.show()


# Above we see that the distribution for co2 emissions is skewed to the right because of the varied range of values; the minimum being zero and the maximum being 100.

# In[ ]:


#Let us look at the number of regions by their counts
byincome_melted['Region'].value_counts()


# Now let us visually look at the counts of Region and income groups.

# In[ ]:


base_color = sns.color_palette()[0]


# In[ ]:


fig,ax = plt.subplots(nrows = 2, figsize= [10,10])
sns.countplot(data = byincome_melted, y= 'Region',  color = base_color, ax= ax[0]);
sns.countplot(data=byincome_melted, y ='incomegroup', color =base_color, ax= ax[1]);


# Above, we see that Sub-saharn Africa region is the highest in count in this dataset. And High income countries are are most frequent here.

# In[ ]:


byincome_melted.reset_index(inplace= True, drop = True)


# In[ ]:


byincome_melted.head()


# In[ ]:


columns = ['Region', 'incomegroup']
fig, ax = plt.subplots(ncols= 1, nrows = 2, figsize = [18,15])

for col in range(len(columns)):
    var= columns[col]
    sns.boxplot(data=byincome_melted, x= var, y= 'co2_emissions', ax= ax[col], color = base_color);


# Above, we see that among the regions, North America has the highest median value of the CO2 emissions than the other regions. Middle East and North Africa have a greater bunch of outliers and show high values nearing 100 metric tons per capita. The lowest number of emissions are shown by South Asia.
# 
# In the second boxplot, it is seen that the high income countries have higher emissions than the others. The overall CO2 emissions show increasing trend as the income level rises for a country.

#  ## Bivariate Relationships
#  Further exploratory and explanatory I have done in Tableau. Below is the link:
#  https://public.tableau.com/profile/anasudame#!/vizhome/GlobalWarming-Dash/Dashboard1

# In[ ]:




