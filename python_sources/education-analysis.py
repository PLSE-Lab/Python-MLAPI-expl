#!/usr/bin/env python
# coding: utf-8

# # Global Education Quality Analysis 
# In this notebook, I analyse the education quality along with other factors such as GDP, Government Expendisure on education, and their qualitative strategy on education. The quality of education can be measure by the  OECD Programme for International Student Assessment (or PISA) as well as the equality of education. Of course, test score is not the most accurate tool to measure education quality but it is what nearest thing we can probably use. Moreover, PISA examination is said to be the most unbiased exam ever.
# 
# The tools I use in this analysis include pandas, numpy, matplotlib, seaborn, as well as Google BigQuery. I decided to use SQL (bigQuery) since the data is huge and using pandas can be slow.

# ### Outline
# 0. Hypothesis Setting
# 1. PISA Score
# 2. GDP 
# 3. Government's Expenditure on Education
# 4. Regression Analysis 1: PISA Score and GDP
# 5. Regression Analysis 2: PISA Score and Expenditure
# 6. Conclusion

# # Hypothesis setting
# 
# I work in an education reform field and would like to learn about what consitutes a good education.
# I believe that the government's interaction is crucial and think that the its expenditure would contribute to a positive impact on education quality.
# 
# I also think that GDP could be correlated as well. Especially the GDP per capita!
# 
# Moreover,

# # Notbook Setting

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))


# ## Data
# We will mainly use 3 sets of data here.
# 1. PISA sccores (2013-2015) provided by PISA. 
# 2. GDP from each country by World Bank.
# 3. Education Statistics by World Bank. We will use SQL, more specifically Google BigQuery,to manipulate this dataset instead of python because the data size is really big.

# # 1) PISA Scores Set
# Firstly, Let's check the data foor PISA set first.
# 
# Before we started, it would be useful to know what kind of questions are in PISA exam.
# 

# In[ ]:


# Read pisa test score
pisa_data = pd.read_csv('/kaggle/input/pisa-scores-2015/Pisa mean perfromance scores 2013 - 2015 Data.csv')
pisa_source = pd.read_csv('/kaggle/input/pisa-scores-2015/Pisa mean performance scores 2013 - 2015 Definition and Source.csv')


# In[ ]:


pisa_data.head()


# In[ ]:


pisa_source.head()


# Ok. so it looks like we will mostly work on the pisa_data part since the source is just some information for the context.

# # Data Cleaning
# As you can see above. There are a lot of missing data including NaN and "..".
# 1. Firstly, we will drop the rows that have all the three year's data being NaN first since they are pretty useless for our objectives here. For this one we can simply use using dropna function. 
# 2. However, for rows with "..", we will use drop function to look for the rows that have '..' and '...'.
# 
# 

# In[ ]:


pisa_data.dropna(subset=['2013 [YR2013]', '2014 [YR2014]','2015 [YR2015]'], thresh = 1, inplace=True)
pisa_data.drop(pisa_data[(pisa_data['2013 [YR2013]'] == '..')&(pisa_data['2014 [YR2014]'] == '..')&(pisa_data['2015 [YR2015]'] == '..')].index, axis=0, inplace=True)
pisa_data.drop(pisa_data[(pisa_data['2013 [YR2013]'] == '...')&(pisa_data['2014 [YR2014]'] == '...')&(pisa_data['2015 [YR2015]'] == '...')].index, axis=0, inplace=True)
pisa_data.head()


# Okay, this is looking better.
# Next, it looks like this data doesn't provide the sum of the scores of every part which are
# 1. Math
# 2. Reading
# 3. Science
# 
# Therefore, we will have to create another row for each country that combine the score from each part.
# Let's do that.
# 
# Also, the number is object (which is like a string), we will have to convert them into float.

# In[ ]:


# It looks like there's a lot of missing values here
_2013_not_null  = pisa_data.loc[(pd.notnull(pisa_data['2013 [YR2013]'] )) & (pisa_data['2013 [YR2013]'] != '..' ) &(pisa_data['2013 [YR2013]'] != '...' ) ].count()
_2014_not_null  = pisa_data.loc[(pd.notnull(pisa_data['2014 [YR2014]'] )) & (pisa_data['2014 [YR2014]'] != '..' ) &(pisa_data['2014 [YR2014]'] != '...' ) ].count()
_2015_not_null  = pisa_data.loc[(pd.notnull(pisa_data['2015 [YR2015]'] )) & (pisa_data['2015 [YR2015]'] != '..' ) &(pisa_data['2015 [YR2015]'] != '...' ) ].count()
# print(_2013_not_null)
# print(_2014_not_null)
# print(_2015_not_null)


# In[ ]:


pisa_data['2015 [YR2015]'] = pisa_data['2015 [YR2015]'].map(lambda x: float(x) if x not in  ['..','...']  else np.nan )
# pisa_data.loc[(pd.notnull(pisa_data['2015 [YR2015]'] )) & (pisa_data['Series Name'] == 'PISA: Mean performance on the mathematics scale') ].info()


# For there, I am 100% sure that the info of others eyars are useless. Let's just focus on the data on 2015.

# In[ ]:


def pisa_sum(country, year):
    math = pisa_data.loc[(pisa_data['Country Name'] == country ) & (pisa_data['Series Code'] == 'LO.PISA.MAT') , [year]][year]
    reading = pisa_data.loc[(pisa_data['Country Name'] == country ) & (pisa_data['Series Code'] == 'LO.PISA.REA') , [year]][year]
    science = pisa_data.loc[(pisa_data['Country Name'] == country ) & (pisa_data['Series Code'] == 'LO.PISA.SCI') , [year]][year]
    sum_score = (float(math)+float(reading)+float(science))/3 if (math.dtype == np.float64) & (reading.dtype == np.float64) & (science.dtype == np.float64) else np.nan                   
    return sum_score
    
countries = pisa_data['Country Name'].unique()
for country in countries:    
    new_df = pd.DataFrame({
            'Country Name': country,
            'Country Code': pisa_data.drop_duplicates(['Country Name']).loc[pisa_data['Country Name'] == country , ['Country Code']]['Country Code'],
            'Series Name': "PISA: Mean performance in total.",
            'Series Code': 'PISA_TOTAL', 
             "2013 [YR2013]": pisa_sum(country, "2013 [YR2013]"), 
             "2014 [YR2014]": pisa_sum(country, "2014 [YR2014]"),
             "2015 [YR2015]": pisa_sum(country, "2015 [YR2015]")
            
        })
    pisa_data = pd.concat([pisa_data, new_df], ignore_index=True, axis = 'index')

    


# ## Looking into PISA Score

# Let's make a new dataframe consiting of only countries and the mean total score of each year.

# In[ ]:


total_df = pisa_data.loc[(pisa_data['Series Name'] == 'PISA: Mean performance in total.') & (pd.notnull(pisa_data['2015 [YR2015]']))].copy()
total_df.sort_values(by='2015 [YR2015]',ascending = False, inplace=True)
total_df.head()


# In[ ]:


_2015_score = total_df[['Country Name','2015 [YR2015]','Country Code']]
# pisa_data.loc[pisa_data['Country Name']]

countries = total_df['Country Name']

fig = plt.figure()
fig.set_size_inches(15,10)
plt.xlabel('Countries')
plt.ylabel("PISA Mean Score")
plt.xticks(rotation='vertical')
bar_graph = plt.bar(countries, _2015_score['2015 [YR2015]'])


# Let's color Thailand to emphasize how shitty we are doing
# Firstly we have to find the index of Thailand
countries = countries.to_list()
thailand_index = countries.index('Thailand')

bar_graph[thailand_index].set_color('red')


# As sewn in the graph, Singapore has the highest score in terms of total MEAN score of PISA. And it exceeds the second place which is Hong Kong a lot.
# I honestly thought that Finland (7th) would be the first because it is always said that Finland has one of the best educational system in the world.

# # Map
# 

# As we can see above that countries with high PISA scores are in North Europe and Asia. However, is this trend consistent? Let us see more in detail by looking in the world map.

# In[ ]:




import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

#merge both data sets using country code/iso_a3 as unique identifiers
geomap_df = world.merge(_2015_score, left_on = 'iso_a3', right_on = 'Country Code')[['geometry','Country Name','2015 [YR2015]']]


fig, ax = plt.subplots()
fig.set_size_inches(20,15)
# fig = plt.figure()
# ax = fig.add_subplot(111)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2%", pad=0.1)
geomap_df.plot(column=geomap_df['2015 [YR2015]'], legend = True, ax=ax, cax=cax, cmap='RdYlGn',linestyle=":",edgecolor='grey' )
# ax = PHL.plot(figsize=(20,20), color='whitesmoke', linestyle=":", edgecolor='black')




# # 2) Country GDP
# 
# Let's look at the GDP within these 10 years!

# In[ ]:


gdp_df = pd.read_csv("../input/gdp-world-bank-data/GDP by Country.csv",skiprows=3)
gdp_df.head()


# In[ ]:


_10_years_span_gdp = gdp_df[['Country Name','Country Code','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']].copy()
len(_10_years_span_gdp['Country Name'])


# In[ ]:


_10_years_span_gdp['mean'] = _10_years_span_gdp.mean(axis =1)
_10_years_span_gdp.sort_values(by=['mean'], inplace=True, ascending=False)
_10_years_span_gdp.head()


# In[ ]:


pisa_and_gdp = pd.merge(_2015_score[['Country Code','2015 [YR2015]']], _10_years_span_gdp, on='Country Code')[['Country Name','mean','2015 [YR2015]']]
pisa_and_gdp.head()


# In[ ]:


sns.scatterplot(x=pisa_and_gdp['2015 [YR2015]'], y=pisa_and_gdp['mean'])

cor = pisa_and_gdp['2015 [YR2015]'].corr(pisa_and_gdp['mean']) 
print('correlation coeefficient')
print(cor)


# In[ ]:


sns.regplot(x=pisa_and_gdp['2015 [YR2015]'], y=pisa_and_gdp['mean'])


# # 3) Government's Expenditure on Education

# ## SQL Big Query Setting
# Here we have to register the account for google could platform (GCP) in order to have access to the BigQuery datasets. Referece on how to do it here.

# In[ ]:


# Set your own project id here
PROJECT_ID = 'kaggle-278402'
from google.cloud import bigquery
# Create a "Client" object
client = bigquery.Client(project=PROJECT_ID)


# In[ ]:


dataset_ref = client.dataset("world_bank_intl_education", project="bigquery-public-data")
dataset = client.get_dataset(dataset_ref)

tables = list(client.list_tables(dataset))
for table in tables:
    print(table.table_id)


# I want to see the expenditure that a Thai government spent to education. Let's search for it.

# In[ ]:


query = """
SELECT DISTINCT indicator_name, indicator_code
FROM `bigquery-public-data.world_bank_intl_education.international_education`
WHERE country_name LIKE '%Thailand%' AND
      indicator_name LIKE '%education%' AND
      indicator_name LIKE '%expenditure%' 
"""

# Set up the query (cancel the query if it would use too much of 
# your quota, with the limit set to 1 Gb)
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**9)
query_job = client.query(query, job_config=safe_config)

# API request - run the query, and convert the results to a pandas DataFrame
query_result = query_job.to_dataframe()

# Print the first five rows
pd.options.display.width = 50
pd.options.display.max_colwidth = 200
pd.set_option('display.max_rows', None)
query_result



# There are several indicators that might be useful to use.
# 1. Government expenditure on education as % of GDP (%)	 - SE.XPD.TOTL.GD.ZS
# 2. Expenditure on education as % of total government expenditure (%) - SE.XPD.TOTL.GB.ZS
# The slight different is that the first one simply tell use how many % of GDP a government spent on education.
# The second one indicates how important the government thinks education is compared to other areas such as transportation, healthcare, or infrastructure.
# I think the first one is more relevant so let us focus on that one.
# 
# Since the score we are looking at is from 2015, it would make sense to look at the expenditure around that time. I would be unfair to look at the information in 2015 since investment takes time. I think it is fair if we calculate the mean of the expenditure spent in the spand of 10 years, which is from 2005-2015.

# In[ ]:


query = """
SELECT country_name,country_code, AVG(value) as mean_spending
FROM `bigquery-public-data.world_bank_intl_education.international_education`
WHERE 
    indicator_code = "SE.XPD.TOTL.GD.ZS" AND
    year > 2004 AND 
    year < 2016
GROUP BY country_name,country_code
ORDER BY mean_spending DESC
"""

# Set up the query (cancel the query if it would use too much of 
# your quota, with the limit set to 1 Gb)
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**9)
query_job = client.query(query, job_config=safe_config)

# API request - run the query, and convert the results to a pandas DataFrame
exp_on_ed = query_job.to_dataframe()

# Print the first five rows
pd.options.display.width = 50
pd.options.display.max_colwidth = 200
pd.set_option('display.max_rows', None)
exp_on_ed.head()


# In[ ]:


countries = exp_on_ed['country_name']

fig = plt.figure()
fig.set_size_inches(15,10)
plt.xlabel('Countries')
plt.ylabel("Expenditure on Education in percentage of GDP")
plt.xticks(rotation='vertical')
bar_graph = plt.bar(countries, exp_on_ed['mean_spending'])


countries = countries.to_list()

thailand_index = countries.index('Thailand')
singapore_index = countries.index('Singapore')
finland_index = countries.index('Finland')

bar_graph[thailand_index].set_color('red')
bar_graph[singapore_index].set_color('green')
bar_graph[finland_index].set_color('yellow')


# Now we're going to do the linear regression. 
# However, the problem is that the dimension of these 2 sets of data are not the same.
# We will have to only select the country that appear in both sets of data. This can be done by inner join.

# # Regression Analysis 1: PISA and GDP

# In[ ]:


pisa_and_gdp = pd.merge(_2015_score[['Country Code','2015 [YR2015]']], _10_years_span_gdp, on='Country Code')[['Country Name','mean','2015 [YR2015]']]
pisa_and_gdp.head()


# ## per capita GDP

# In[ ]:





# In[ ]:


# sns.scatterplot(x=pisa_and_gdp['2015 [YR2015]'], y=pisa_and_gdp['mean'])
sns.regplot(x=pisa_and_gdp['2015 [YR2015]'], y=pisa_and_gdp['mean'])


cor = pisa_and_gdp['2015 [YR2015]'].corr(pisa_and_gdp['mean']) 
print('correlation coeefficient')
print(cor)


# In[ ]:





# # Regression Analysis 2: PISA and Expenditure on Education

# In[ ]:


# score_and_expenditure = pd.merge(_2015_score, exp_on_ed, on='Country Name')
pisa_and_expenditure = exp_on_ed.merge(_2015_score, left_on = 'country_code', right_on = 'Country Code')[['Country Name','Country Code','2015 [YR2015]','mean_spending']]
pisa_and_expenditure.head()


# In[ ]:


# let's define each for easier code
gov_spending = pisa_and_expenditure['mean_spending']
pisa_score = pisa_and_expenditure['2015 [YR2015]']


# In[ ]:


sns.regplot(x=pisa_score, y=gov_spending)


# It looks like there is a positive correlation between two variables, not so strong however.
# Let's calculate the Correlation coefficient to get a better idea between the two variables. In this case, we will use Pearson correlation coefficien since we want to analyze the linear regression.
# 
# Of course, correlation does not indixate causation. 

# In[ ]:


# using pandas
cor = pisa_score.corr(gov_spending) 
print(f"Pearson's Correlation Coeefficient from Pandas: {cor}")
# using numpy
# cor_coef = np.corrcoef(pisa_score, gov_spending)
# print(f"Pearson's Correlation Coeefficient from Numpy: {cor_coef[0,1]}")

# using scipi
# import scipy.stats
# correlation_coef, p_value = scipy.stats.pearsonr(merge_2015_score, merge_exp_on_ed)
# print(f"Pearson's Correlation Coeefficient from Scipy: {correlation_coef}")
# print(f"p_value: {p_value}")

#p value is 3% meaning the there is a correlation. But of does there is not gaurantee that the government's expenditure cause that.


# # Conclusion
# 
# ### Regression Analysis 1: PISA Score and GDP
# **Correlation Coeeficient:** 0.6
# 
# Looks like there is a positive, although not so strong, correlation between PISA Score and GDP. Of course, correlation does not imply causation but I do believe that these two variables effect each other more or less. 
# 
# ### Regression Analysis 2: PISA Score and Expenditure
# **Correlation Coeeficient:** 0.3
# 
# Surprisingly, there is not much correlation between PISA score and government's expenditure on education. I thought that this should correlate somehow. One possible explanation is that the expenditure was not efficiently utylized by the stakeholders. Another possibility is that there might be some kind of corruption behind the scene.
# 
# 

# ### To do
# 1. I made a mistake by focusing too much on GDP in this analysis. However, I should also look at the GDP per capita as well.

# In[ ]:




