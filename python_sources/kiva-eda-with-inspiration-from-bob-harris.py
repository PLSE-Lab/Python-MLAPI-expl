#!/usr/bin/env python
# coding: utf-8

# # Kiva Data in Bars, Time Series and Word Clouds
# 
# It is always a work in progress...

# In[1]:


# Import packages

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, STOPWORDS

# plt.style.use('seaborn')
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.


# In[2]:


# Load data
loans = pd.read_csv('../input/kiva_loans.csv', parse_dates = ['posted_time', 'disbursed_time', 'funded_time', 'date'])
themes_by_region = pd.read_csv('../input/loan_themes_by_region.csv')
region_loc = pd.read_csv('../input/kiva_mpi_region_locations.csv')
themes = pd.read_csv('../input/loan_theme_ids.csv')


# ## Inspecting Data

# In[3]:


loans.shape, themes_by_region.shape, region_loc.shape, themes.shape


# In[ ]:


# Sometimes .sample() is better than .head() to if I want to inspect random lines
loans.drop(['use'], axis=1).sample(5) # use column is too long


# In[4]:


themes.sample(5)


# Let's merge loans and themes datasets together:

# In[5]:


loans = pd.merge(loans, themes.drop(['Partner ID'], axis=1), how='left', on='id')
loans.drop(['use'],axis=1).head()


# We can see that some loans do not have loan theme data. We now have the data table in place for more exploratory data analysis

# ## Basic Exploratory Data Analysis
# Loan amount by sector:

# In[6]:


plt.figure(figsize=(9,6))
sec = loans.groupby('sector').sum()['loan_amount'].reset_index()
sec = sec.sort_values(by='loan_amount', ascending = False)
g = sns.barplot(x='sector', y='loan_amount', ci=None, palette = 'spring', data=sec)
g.set_xticklabels(g.get_xticklabels(), rotation=45)
plt.title('Total loan amount by sector')
plt.xlabel('Sector')
plt.ylabel('Loan amount')
plt.show()


# Loan amount by country. Only top 10 is shown:

# In[7]:


plt.figure(figsize=(9,6))
top10c = loans.groupby('country').sum()['loan_amount'].reset_index()
top10c = top10c.sort_values(by='loan_amount', ascending = False)
g = sns.barplot(x='country', y='loan_amount', ci=None, palette = "cool", data=top10c.head(10))
g.set_xticklabels(g.get_xticklabels(), rotation=45)
plt.title('Top 10 countries in loan amount')
plt.xlabel('Country')
plt.ylabel('Loan amount')
plt.show()


# ## Time Series Plots
# Here we are going to show the evolution of loan amount over time, by country, by world region and by sector

# In[8]:


lcst = loans.loc[:,['loan_amount', 'country','sector', 'posted_time']]
lcst.set_index('posted_time', inplace=True)
lcst.head()


# In[9]:


# Overall loan amount evolution
plt.figure(figsize=(9,6))
plt.plot(lcst.resample('M').sum())
plt.title('Loan amount by Month')
plt.show()


# The drop in July 2017 is likely due to partial month data.
# 
# The plot_monthly_by_cty function creates a plot of total loan amount by month for any country:

# In[10]:


plt.figure(figsize=(10,6))

def plot_monthly_by_cty(cty_name):
    
    lctm_cty = lcst[lcst.country==cty_name].resample('M').sum()
    month_label=[]
    for dt in lctm_cty.index:
        month_label.append(dt.strftime('%Y-%m'))
    sns.barplot(x=month_label, y=lctm_cty['loan_amount'], color='Blue', alpha=0.7)
    plt.title('Loan amount by Month - ' + cty_name)
    plt.xticks(rotation=90)
    plt.show()

plot_monthly_by_cty('China')


# Similarly, here is a function to plot of total loan amount by month for any sector:

# In[11]:


plt.figure(figsize=(10,6))

def plot_monthly_by_sector(sec_name):
    
    lsm_sec = lcst[lcst.sector==sec_name].resample('M').sum()
    month_label=[]
    for dt in lsm_sec.index:
        month_label.append(dt.strftime('%Y-%m'))
    sns.barplot(x=month_label, y=lsm_sec['loan_amount'], color='Magenta', alpha=0.7)
    plt.title('Loan amount by Month - ' + sec_name)
    plt.xticks(rotation=90)
    plt.show()

plot_monthly_by_sector('Agriculture')


# The next step is to plot a stacked bar chart showing evolution of loan amount by country:

# In[12]:


by_month = pd.DataFrame()

for cty in loans.country.unique():
    lctm_cty = lcst[lcst.country==cty].resample('M').sum()
    lctm_cty.columns = [cty]
    by_month = pd.concat([by_month, lctm_cty],axis=1)

by_month = by_month.fillna(0)
by_month.head()


# Let's show how the loan amount of top 10 countries evolve over time:

# In[13]:


top10list = top10c.head(10)['country'].tolist()

month_label=[]
for dt in by_month.index:
    month_label.append(dt.strftime('%Y-%m'))
by_month.loc[:,top10list].plot(kind='bar', stacked=True, x = np.array(month_label), figsize=(15,7), colormap = 'Set3')
plt.title('Loan Amount Evolution of Top 10 Countries')
plt.legend(bbox_to_anchor=(1.01,0.95))
plt.show()


# Let's do similar plots for sectors:

# In[15]:


by_month_sec = pd.DataFrame()

for sector in loans.sector.unique():
    lsm_sec = lcst[lcst.sector==sector].resample('M').sum()
    lsm_sec.columns = [sector]
    by_month_sec = pd.concat([by_month_sec, lsm_sec],axis=1)

by_month_sec = by_month_sec.fillna(0)
by_month_sec.head()


# In[16]:


month_label=[]
for dt in by_month_sec.index:
    month_label.append(dt.strftime('%Y-%m'))
by_month_sec.plot(kind='bar', stacked=True, x = np.array(month_label), figsize=(15,7))
plt.title('Loan Amount Evolution of All Sectors')
plt.legend(bbox_to_anchor=(1.01, 0.95))
plt.show()


# ## Borrower Genders
# Next, we work on the "borrower_genders" column. It is a string of gender of each borrower. We will convert it into columns of how many male and female borrower in each loan. As the column is quite tidy, we can use CountVectorizer to help with the counts.

# In[18]:


loans.borrower_genders.fillna('Unknown', inplace=True)
cv = CountVectorizer()
gender_count = cv.fit_transform(loans.borrower_genders)
df_gender = pd.DataFrame(gender_count.toarray())
df_gender.columns = ['borrower_' + str(i) for i in cv.vocabulary_.keys()]


# In[19]:


df_gender['borrower_total'] = df_gender['borrower_female']+df_gender['borrower_male']
df_gender.describe()


# Let's look at how gender composition of loans change overtime:

# In[20]:


time_gender = pd.concat([df_gender[['borrower_female','borrower_male']], loans['posted_time']], axis=1)
time_gender.set_index('posted_time', inplace=True)
time_gender.head()


# In[21]:


gender_trend = time_gender.resample('M').sum()
gender_trend.head()


# In[22]:


month_label=[]
for dt in gender_trend.index:
    month_label.append(dt.strftime('%Y-%m'))
gender_trend.plot(kind='bar', stacked=True, x = np.array(month_label), figsize=(15,7))
plt.title('Loan Amount Evolution by gender')
plt.legend(bbox_to_anchor=(1.01, 0.95))
plt.show()


# Women makes the majority of borrowers throughout the period analyzed.  

# ## Partners
# Kiva does not lend directly to the borrowers. It makes the loans through microfinance institutions (MFI). Let's look at the characteristics of those MFI partners.

# In[23]:


themes_by_region.head()


# In[24]:


partner_info = themes_by_region.loc[:,['Partner ID','Field Partner Name']].drop_duplicates()
partner_info.head(10)


# Check if one ID refers to only one partner:

# In[25]:


check = partner_info['Partner ID'].drop_duplicates()
len(check)-len(partner_info) # 0 if no Partner ID is duplicated


# Let's plot loan amount of top 10 partners, first specific to a certain country, then total of all countries for each partner.

# In[29]:


# Loan amount by partner and country
top10p = loans.groupby(['partner_id','country']).sum()['loan_amount'].reset_index()
top10p = top10p.sort_values(by='loan_amount', ascending = False)
# Add names to partner statistics
top10p_name = pd.merge(top10p, partner_info, left_on='partner_id', right_on='Partner ID', how='left')
top10p_name['Partner_country'] = top10p_name['Field Partner Name'] + '@' + top10p_name['country']
top10p_name.head(10)


# In[30]:


plt.figure(figsize=(10,6))
g = sns.barplot(x='loan_amount', y='Partner_country', ci=None, palette='ocean', data=top10p_name.head(10), alpha=0.9)
plt.title('Top 10 Country-Specific Partners')
plt.ylabel('Partner_Country')
plt.xlabel('Loan amount')
plt.show()


# In[33]:


top10p2 = top10p_name.groupby(['partner_id','Field Partner Name']).sum()['loan_amount'].reset_index()
top10p2 = top10p2.sort_values(by='loan_amount', ascending = False).head(10)
plt.figure(figsize=(10,6))
g = sns.barplot(x='loan_amount', y='Field Partner Name', ci=None, palette='ocean', data=top10p2, alpha=0.9)
plt.title('Top 10 Partners')
plt.ylabel('Partner Name')
plt.xlabel('Loan amount')
plt.show()


# The top 10 list are the same. Indicates that MFI partners mostly work on a national scale.

# # Word Cloud of Use of Fund and Activity

# In[34]:


names = loans["use"][~pd.isnull(loans["use"])]

stopwords = set(STOPWORDS)
wordcloud = WordCloud(background_color = 'white', max_font_size=50, min_font_size=5, width=600, height=400, max_words=200, stopwords = stopwords).generate(' '.join(names))
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.title("Wordcloud for Use", fontsize=25)
plt.axis("off")
plt.show() 


# In[49]:


from nltk import FreqDist, word_tokenize
names = loans["activity"][~pd.isnull(loans["activity"])]
word_freq = FreqDist(word_tokenize(' '.join(names).lower()))

stopwords = set(STOPWORDS).add('&')
wordcloud = WordCloud(background_color = 'white', width=600, height=400, max_words=150, stopwords = stopwords).generate_from_frequencies(word_freq)
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.title("Wordcloud for Activity", fontsize=25)
plt.axis("off")
plt.show() 


# Stay tuned for more analysis!
