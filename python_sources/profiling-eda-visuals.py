#!/usr/bin/env python
# coding: utf-8

# # Teclov Part 1: Data Cleaning
# 
# Let's start with getting the datafiles rounds.csv and companies.txt.
# 

# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_profiling import ProfileReport

# reading data files
# using encoding = "ISO-8859-1" to avoid pandas encoding error
rounds = pd.read_csv("/kaggle/input/spark-fund-investment-analysis/datasets/rounds2.csv", encoding = "ISO-8859-1")
companies = pd.read_csv("/kaggle/input/spark-fund-investment-analysis/datasets/companies.txt", sep="\t", encoding = "ISO-8859-1")


# In[ ]:


profile = ProfileReport(rounds, title='Rounds Profiling Report', html={'style':{'full_width':True}})
profile


# In[ ]:


profile = ProfileReport(companies, title='Companies Profiling Report', html={'style':{'full_width':True}})
profile


# The variables funding_round_code and raised_amount_usd contain some missing values, as shown above. We'll deal with them after we're done with understanding the data - column names, primary keys of tables etc.

# In[ ]:


# look at companies head
companies.head()


# Ideally, the ```permalink``` column in the companies dataframe should be the unique_key of the table, having 66368 unique company names (links, or permalinks). Also, these 66368 companies should be present in the rounds file.
# 
# Let's first confirm that these 66368 permalinks (which are the URL paths of companies' websites) are not repeating in the column, i.e. they are unique.

# In[ ]:


# identify the unique number of permalinks in companies
len(companies.permalink.unique())


# Also, let's convert all the entries to lowercase (or uppercase) for uniformity.

# In[ ]:


# converting all permalinks to lowercase
companies['permalink'] = companies['permalink'].str.lower()
companies.head()


# In[ ]:


# look at unique values again
len(companies.permalink.unique())


# Thus, there are 66368 unique companies in the table and ```permalink``` is the unique primary key. Each row represents a unique company.
# 
# Let's now check whether all of these 66368 companies are present in the rounds file, and if some extra ones are present.

# In[ ]:


# look at unique company names in rounds df
# note that the column name in rounds file is different (company_permalink)
len(rounds.company_permalink.unique())


# There seem to be 90247 unique values of ```company_permalink```, whereas we expected only 66368. May be this is because of uppercase/lowercase issues.
# 
# Let's convert the column to lowercase and look at unique values again.

# In[ ]:


# converting column to lowercase
rounds['company_permalink'] = rounds['company_permalink'].str.lower()
rounds.head()


# In[ ]:


# Look at unique values again
len(rounds.company_permalink.unique())


# There seem to be 2 extra permalinks in the rounds file which are not present in the companies file. Let's hope that this is a data quality issue, since if this were genuine, we have two companies whose investment round details are available but their metadata (company name, sector etc.) is not available in the companies table.

# Let's have a look at the company permalinks which are in the 'rounds' file but not in 'companies'.

# In[ ]:


# companies present in rounds file but not in (~) companies file
rounds.loc[~rounds['company_permalink'].isin(companies['permalink']), :]


# All the permalinks have weird non-English characters. Let's see whether these characters are present in the original df as well. 

# In[ ]:


# looking at the indices with weird characters
rounds_original = pd.read_csv("/kaggle/input/spark-fund-investment-analysis/datasets/rounds2.csv", encoding = "ISO-8859-1")
rounds_original.iloc[[29597, 31863, 45176, 58473], :]


# The company weird characters appear when you import the data file. To confirm whether these characters are actually present in the given data or whether python has introduced them while importing into pandas, let's have a look at the original CSV file in Excel.

# Thus, this is most likely a data quality issue we have introduced while reading the data file into python. Specifically, this is most likely caused because of encoding.
# 
# First, let's try to figure out the encoding type of this file. Then we can try specifying the encoding type at the time of reading the file. The ```chardet``` library shows the encoding type of a file.

# Now let's try telling pandas (at the time of importing) the encoding type. Here's a list of various encoding types python can handle: https://docs.python.org/2/library/codecs.html#standard-encodings.

# Apparently, pandas cannot decode "cp1254" in this case. After searching a lot on stackoverflow and Google, the best conclusion that can be drawn is that this file is encoded using multiple encoding types (may be because the ```company_permalink``` column contains names of companies in various countries, and hence various languages).
# 
# After trying various other encoding types (in vain), this answer suggested an alternate (and a more intelligent) way: https://stackoverflow.com/questions/45871731/removing-special-characters-in-a-pandas-dataframe.
# 
# 

# In[ ]:


rounds['company_permalink'] = rounds.company_permalink.str.encode('utf-8').str.decode('ascii', 'ignore')
rounds.loc[~rounds['company_permalink'].isin(companies['permalink']), :]


# This seems to work fine. 
# 
# Let's now look at the number of unique values in rounds dataframe again.

# In[ ]:


# Look at unique values again
len(rounds.company_permalink.unique())


# Now it makes sense - there are 66368 unique companies in both the ```rounds``` and ```companies``` dataframes. 
# 
# It is possible that a similar encoding problems are present in the companies file as well. Let's look at the companies which are present in the companies file but not in the rounds file - if these have special characters, then it is most likely because the ```companies``` file is encoded (while rounds is not).

# In[ ]:


# companies present in companies df but not in rounds df
companies.loc[~companies['permalink'].isin(rounds['company_permalink']), :]


# Thus, the ```companies``` df also contains special characters. Let's treat those as well.

# In[ ]:


# remove encoding from companies df
companies['permalink'] = companies.permalink.str.encode('utf-8').str.decode('ascii', 'ignore')


# Let's now look at the companies present in the companies df but not in rounds df - ideally there should be none.

# In[ ]:


# companies present in companies df but not in rounds df
companies.loc[~companies['permalink'].isin(rounds['company_permalink']), :]


# Thus, the encoding issue seems resolved now. Let's write these (clean) dataframes into separate files so we don't have to worry about encoding problems again.

# # Part 2: Data Cleaning - II
# 
# Now that we've treated the encoding problems (caused by special characters), let's complete the data cleaning process by treating missing values. 

# In[ ]:


# quickly verify that there are 66368 unique companies in both
# and that only the same 66368 are present in both files

# unqiue values
print(len(companies.permalink.unique()))
print(len(rounds.company_permalink.unique()))

# present in rounds but not in companies
print(len(rounds.loc[~rounds['company_permalink'].isin(companies['permalink']), :]))


# ## Missing Value Treatment
# 
# Let's now move to missing value treatment. 
# 
# Let's have a look at the number of missing values in both the dataframes.

# In[ ]:


# missing values in companies df
companies.isnull().sum()


# In[ ]:


# missing values in rounds df
rounds.isnull().sum()


# Since there are no misisng values in the permalink or company_permalink columns, let's merge the two and then work on the master dataframe.

# In[ ]:


# merging the two dfs
master = pd.merge(companies, rounds, how="inner", left_on="permalink", right_on="company_permalink")
master.head()


# Since the columns ```company_permalink``` and ```permalink``` are the same, let's remove one of them.
# 

# In[ ]:


# print column names
master.columns


# In[ ]:


# removing redundant columns
master =  master.drop(['company_permalink'], axis=1) 


# In[ ]:


# look at columns after dropping
master.columns


# Let's now look at the number of missing values in the master df.

# In[ ]:


# column-wise missing values 
master.isnull().sum()


# Let's look at the fraction of missing values in the columns.

# In[ ]:


# summing up the missing values (column-wise) and displaying fraction of NaNs
round(100*(master.isnull().sum()/len(master.index)), 2)


# Clearly, the column ```funding_round_code``` is useless (with about 73% missing values). Also, for the business objectives given, the columns ```homepage_url```, ```founded_at```, ```state_code```, ```region``` and ```city``` need not be used.
# 
# Thus, let's drop these columns.

# In[ ]:


# dropping columns 
master = master.drop(['funding_round_code', 'homepage_url', 'founded_at', 'state_code', 'region', 'city'], axis=1)
master.head()


# In[ ]:


# summing up the missing values (column-wise) and displaying fraction of NaNs
round(100*(master.isnull().sum()/len(master.index)), 2)


# Note that the column ```raised_amount_usd``` is an important column, since that is the number we want to analyse (compare, means, sum etc.). That needs to be carefully treated. 
# 
# Also, the column ```country_code``` will be used for country-wise analysis, and ```category_list``` will be used to merge the dataframe with the main categories.
# 
# Let's first see how we can deal with missing values in ```raised_amount_usd```.
# 

# In[ ]:


# summary stats of raised_amount_usd
master['raised_amount_usd'].describe()


# The mean is somewhere around USD 10 million, while the median is only about USD 1m. The min and max values are also miles apart. 
# 
# In general, since there is a huge spread in the funding amounts, it will be inappropriate to impute it with a metric such as median or mean. Also, since we have quite a large number of observations, it is wiser to just drop the rows. 
# 
# Let's thus remove the rows having NaNs in ```raised_amount_usd```.

# In[ ]:


# removing NaNs in raised_amount_usd
master = master[~np.isnan(master['raised_amount_usd'])]
round(100*(master.isnull().sum()/len(master.index)), 2)


# Let's now look at the column ```country_code```. To see the distribution of the values for categorical variables, it is best to convert them into type 'category'.

# In[ ]:


country_codes = master['country_code'].astype('category')

# displaying frequencies of each category
country_codes.value_counts()


# By far, the most number of investments have happened in American countries. We can also see the fractions.

# In[ ]:


# viewing fractions of counts of country_codes
100*(master['country_code'].value_counts()/len(master.index))


# Now, we can either delete the rows having ```country_code``` missing (about 6% rows), or we can impute them by ```USA```. Since the number 6 is quite small, and we have a decent amount of data, it may be better to just remove the rows.
# 
# **Note that** ```np.isnan``` does not work with arrays of type 'object', it only works with native numpy type (float). Thus, you can use ```pd.isnull()``` instead.

# In[ ]:


# removing rows with missing country_codes
master = master[~pd.isnull(master['country_code'])]

# look at missing values
round(100*(master.isnull().sum()/len(master.index)), 2)


# Note that the fraction of missing values in the remaining dataframe has also reduced now - only 0.65% in ```category_list```. Let's thus remove those as well.
# 
# **Note**
# Optionally, you could have simply let the missing values in the dataset and continued the analysis. There is nothing wrong with that. But in this case, since we will use that column later for merging with the 'main_categories', removing the missing values will be quite convenient (and again - we have enough data).

# In[ ]:


# removing rows with missing category_list values
master = master[~pd.isnull(master['category_list'])]

# look at missing values
round(100*(master.isnull().sum()/len(master.index)), 2)


# In[ ]:


# look at the master df info for number of rows etc.
master.info()


# In[ ]:


# after missing value treatment, approx 77% observations are retained
100*(len(master.index) / len(rounds.index))


# # Part 3: Analysis
# 
# 
# In this section, we'll conduct the three types of analyses - funding type, country analysis, and sector analysis.
# 
# 
# ## Funding Type Analysis
# 
# Let's compare the funding amounts across the funding types. Also, we need to impose the constraint that the investment amount should be between 5 and 15 million USD. We will choose the funding type such that the average investment amount falls in this range.

# In[ ]:


# first, let's filter the df so it only contains the four specified funding types
df = master[(master.funding_round_type == "venture") | 
        (master.funding_round_type == "angel") | 
        (master.funding_round_type == "seed") | 
        (master.funding_round_type == "private_equity") ]


# Now, we have to compute a **representative value of the funding amount** for each type of invesstment. We can either choose the mean or the median - let's have a look at the distribution of ```raised_amount_usd``` to get a sense of the distribution of data.
# 
# 

# In[ ]:


# distribution of raised_amount_usd
sns.boxplot(y=df['raised_amount_usd'])
plt.yscale('log')
plt.show()


# Let's also look at the summary metrics.

# In[ ]:


# summary metrics
df['raised_amount_usd'].describe()


# Note that there's a significant difference between the mean and the median - USD 9.5m and USD 2m. Let's also compare the summary stats across the four categories.

# In[ ]:


# comparing summary stats across four categories
sns.boxplot(x='funding_round_type', y='raised_amount_usd', data=df)
plt.yscale('log')
plt.show()


# In[ ]:


# compare the mean and median values across categories
df.pivot_table(values='raised_amount_usd', columns='funding_round_type', aggfunc=[np.median, np.mean])


# Note that there's a large difference between the mean and the median values for all four types. For type venture, for e.g. the median is about 20m while the mean is about 70m. 
# 
# Thus, the choice of the summary statistic will drastically affect the decision (of the investment type). Let's choose median, since there are quite a few extreme values pulling the mean up towards them - but they are not the most 'representative' values.
# 

# In[ ]:


# compare the median investment amount across the types
df.groupby('funding_round_type')['raised_amount_usd'].median().sort_values(ascending=False)


# The median investment amount for type 'private_equity' is approx. USD 20m, which is beyond Teclov' range of 5-15m. The median of 'venture' type is about USD 5m, which is suitable for them. The average amounts of angel and seed types are lower than their range.
# 
# Thus, 'venture' type investment will be most suited to them.

# ## Country Analysis
# 
# Let's now compare the total investment amounts across countries. Note that we'll filter the data for only the 'venture' type investments and then compare the 'total investment' across countries.

# In[ ]:


# filter the df for private equity type investments
df = df[df.funding_round_type=="venture"]

# group by country codes and compare the total funding amounts
country_wise_total = df.groupby('country_code')['raised_amount_usd'].sum().sort_values(ascending=False)
print(country_wise_total)


# Let's now extract the top 9 countries from ```country_wise_total```.

# In[ ]:


# top 9 countries
top_9_countries = country_wise_total[:9]
top_9_countries


# Among the top 9 countries, USA, GBR and IND are the top three English speaking countries. Let's filter the dataframe so it contains only the top 3 countries.

# In[ ]:


# filtering for the top three countries
df = df[(df.country_code=='USA') | (df.country_code=='GBR') | (df.country_code=='IND')]
df.head()


# After filtering for 'venture' investments and the three countries USA, Great Britain and India, the filtered df looks like this.

# In[ ]:


# filtered df has about 38800 observations
df.info()


# One can visually analyse the distribution and the total values of funding amount.

# In[ ]:


# boxplot to see distributions of funding amount across countries
plt.figure(figsize=(10, 10))
sns.boxplot(x='country_code', y='raised_amount_usd', data=df)
plt.yscale('log')
plt.show()


# Now, we have shortlisted the investment type (venture) and the three countries. Let's now choose the sectors.

# ## Sector Analysis
# 
# First, we need to extract the main sector using the column ```category_list```. The category_list column contains values such as 'Biotechnology|Health Care' - in this, 'Biotechnology' is the 'main category' of the company, which we need to use.
# 
# Let's extract the main categories in a new column.

# In[ ]:


# extracting the main category
df.loc[:, 'main_category'] = df['category_list'].apply(lambda x: x.split("|")[0])
df.head()


# We can now drop the ```category_list``` column.

# In[ ]:


# drop the category_list column
df = df.drop('category_list', axis=1)
df.head()


# Now, we'll read the ```mapping.csv``` file and merge the main categories with its corresponding column. 

# In[ ]:


# read mapping file
mapping = pd.read_csv("/kaggle/input/spark-fund-investment-analysis/datasets/mapping.csv", sep=",")
mapping.head()


# Firstly, let's get rid of the missing values since we'll not be able to merge those rows anyway. 

# In[ ]:


# missing values in mapping file
mapping.isnull().sum()


# In[ ]:


# remove the row with missing values
mapping = mapping[~pd.isnull(mapping['category_list'])]
mapping.isnull().sum()


# Now, since we need to merge the mapping file with the main dataframe (df), let's convert the common column to lowercase in both.

# In[ ]:


# converting common columns to lowercase
mapping['category_list'] = mapping['category_list'].str.lower()
df['main_category'] = df['main_category'].str.lower()


# In[ ]:


# look at heads
print(mapping.head())


# In[ ]:


print(df.head())


# Let's have a look at the ```category_list``` column of the mapping file. These values will be used to merge with the main df.

# In[ ]:


mapping['category_list']


# To be able to merge all the ```main_category``` values with the mapping file's ```category_list``` column, all the values in the  ```main_category``` column should be present in the ```category_list``` column of the mapping file.
# 
# Let's see if this is true.

# In[ ]:


# values in main_category column in df which are not in the category_list column in mapping file
df[~df['main_category'].isin(mapping['category_list'])]


# Notice that values such as 'analytics', 'business analytics', 'finance', 'nanatechnology' etc. are not present in the mapping file.
# 
# Let's have a look at the values which are present in the mapping file but not in the main dataframe df.

# In[ ]:


# values in the category_list column which are not in main_category column 
mapping[~mapping['category_list'].isin(df['main_category'])]


# If you see carefully, you'll notice something fishy - there are sectors named *alter0tive medicine*, *a0lytics*, *waste ma0gement*, *veteri0ry*, etc. This is not a *random* quality issue, but rather a pattern. In some strings, the 'na' has been replaced by '0'. This is weird - maybe someone was trying to replace the 'NA' values with '0', and ended up doing this. 
# 
# Let's treat this problem by replacing '0' with 'na' in the ```category_list``` column.

# In[ ]:


# replacing '0' with 'na'
mapping['category_list'] = mapping['category_list'].apply(lambda x: x.replace('0', 'na'))
print(mapping['category_list'])


# This looks fine now. Let's now merge the two dataframes.

# In[ ]:


# merge the dfs
df = pd.merge(df, mapping, how='inner', left_on='main_category', right_on='category_list')
df.head()


# In[ ]:


# let's drop the category_list column since it is the same as main_category
df = df.drop('category_list', axis=1)
df.head()


# In[ ]:


# look at the column types and names
df.info()


# ### Converting the 'wide' dataframe to 'long'
# 
# You'll notice that the columns representing the main category in the mapping file are originally in the 'wide' format - Automotive & Sports, Cleantech / Semiconductors etc.
# 
# They contain the value '1' if the company belongs to that category, else 0. This is quite redundant. We can as well have a column named 'sub-category' having these values. 
# 
# Let's convert the df into the long format from the current wide format. First, we'll store the 'value variables' (those which are to be melted) in an array. The rest will then be the 'index variables'.

# In[ ]:


# store the value and id variables in two separate arrays

# store the value variables in one Series
value_vars = df.columns[9:18]

# take the setdiff() to get the rest of the variables
id_vars = np.setdiff1d(df.columns, value_vars)

print(value_vars, "\n")
print(id_vars)


# In[ ]:


# convert into long
long_df = pd.melt(df, 
        id_vars=list(id_vars), 
        value_vars=list(value_vars))

long_df.head()


# We can now get rid of the rows where the column 'value' is 0 and then remove that column altogether.

# In[ ]:


# remove rows having value=0
long_df = long_df[long_df['value']==1]
long_df = long_df.drop('value', axis=1)


# In[ ]:


# look at the new df
long_df.head()
len(long_df)


# In[ ]:


# renaming the 'variable' column
long_df = long_df.rename(columns={'variable': 'sector'})


# In[ ]:


long_df.info()


# The dataframe now contains only venture type investments in countries USA, IND and GBR, and we have mapped each company to one of the eight main sectors (named 'sector' in the dataframe). 
# 
# We can now compute the sector-wise number and the amount of investment in the three countries.

# In[ ]:


# summarising the sector-wise number and sum of venture investments across three countries

# first, let's also filter for investment range between 5 and 15m
df = long_df[(long_df['raised_amount_usd'] >= 5000000) & (long_df['raised_amount_usd'] <= 15000000)]


# In[ ]:


# groupby country, sector and compute the count and sum
df.groupby(['country_code', 'sector']).raised_amount_usd.agg(['count', 'sum'])


# This will be much more easy to understand using a plot.

# In[ ]:


# plotting sector-wise count and sum of investments in the three countries
plt.figure(figsize=(16, 14))

plt.subplot(2, 1, 1)
p = sns.barplot(x='sector', y='raised_amount_usd', hue='country_code', data=df, estimator=np.sum)
p.set_xticklabels(p.get_xticklabels(),rotation=30)
plt.title('Total Invested Amount (USD)')

plt.subplot(2, 1, 2)
q = sns.countplot(x='sector', hue='country_code', data=df)
q.set_xticklabels(q.get_xticklabels(),rotation=30)
plt.title('Number of Investments')


plt.show()


# Thus, the top country in terms of the number of investments (and the total amount invested) is the USA. The sectors 'Others', 'Social, Finance, Analytics and Advertising' and 'Cleantech/Semiconductors' are the most heavily invested ones.
# 
# In case you don't want to consider 'Others' as a sector, 'News, Search and Messaging' is the next best sector.
