#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from wordcloud import WordCloud, STOPWORDS
import plotly.offline as py
py.init_notebook_mode(connected=True)


# In[2]:


loans = pd.read_csv('../input/kiva_loans.csv')


# In[3]:


loans.shape


# In[4]:


loans.head()


# In[5]:


plt.figure(figsize=(12,9))
sns.heatmap(loans.isnull(),cmap='viridis',yticklabels=False,cbar=False)
plt.title('Missing Data?\n',fontsize=20)
plt.show()


# In[6]:


type(loans['disbursed_time'][0])


# Let's convert all dates to pandas Timestamp format.

# In[7]:


loans['posted_time'] = pd.to_datetime(loans['posted_time'])
loans['disbursed_time'] = pd.to_datetime(loans['disbursed_time'])
loans['funded_time'] = pd.to_datetime(loans['funded_time'])


# In[8]:


loans['month(dis)'] = loans['disbursed_time'].apply(lambda x: x.month)
loans['month(post)'] = loans['posted_time'].apply(lambda x: x.month)


# In[9]:


plt.figure(figsize=(12,9))
loans['country'].value_counts().head(15).plot.bar()
plt.title('Top Ten Countries by Number of Kiva Loans\n',fontsize=16)
plt.show()
print('Top Ten Countries by Number of Kiva Loans\n\n', loans['country'].value_counts().head(15))


# In[10]:


country_count = loans.groupby('country').count()['loan_amount'].sort_values(ascending=False)
data = [dict(
        type = 'choropleth',
        locations = country_count.index,
        locationmode = 'country names',
        z = country_count.values,
        text = country_count.index,
        colorscale = 'Blue',
        marker = dict(
            line=dict(width=.7)),
        colorbar = dict(
            autotick = False, 
            tickprefix = '',
            title = 'Count of Loans per Country'),)]
layout = dict(title = 'Number of Loans by Country',
             geo = dict(
                 showframe = False,
                 #showcoastlines = False,
                 projection = dict(
                 type = 'Mercatorodes')))
fig = dict(data=data, layout=layout)
py.iplot(fig,validate=False)


# In[11]:


plt.figure(figsize=(12,9))
loans['sector'].value_counts().head(10).plot(kind='bar')
plt.title('Top Ten Sectors by Number of Loans\n',fontsize=16)
plt.show()
print("Top Ten Sectors by Number of Loans\n\n",loans['sector'].value_counts().head(10))


# In[12]:


plt.figure(figsize=(12,9))
sns.distplot(loans[loans['loan_amount']<=5000]['loan_amount'],kde=False,bins=60,color='g')
plt.title('Loan Amount to number of Loans under $5000\n',fontsize=16)
plt.show()
print('\n{0:.1f}%'.format(len(loans[loans['loan_amount']<=5000]['loan_amount']) / len(loans)*100) + ' of all loans were for under $5000 dollars.')
print('\nThe average loan amount (excluding the over $5000 outliers): ${0:.0f} dollars.'.format(loans[loans['loan_amount']<=5000]['loan_amount'].mean()))


# In[13]:


plt.figure(figsize=(12,9))
loans['borrower_genders'].value_counts().head(8).plot(kind='bar')
plt.title('Gender of Kiva Borrowers\n',fontsize=16)
plt.show()
print("Gender of Kiva Borrowers\n")
print(loans['borrower_genders'].value_counts().head(8))
print('\nAbout {0:.0f}% of the borrowers are male.'.format(len(loans[loans['borrower_genders']=='male'])/len(loans)*100))


# Without a doubt females dominate the Kiva Crowdfunding platform.

# In[14]:


loans['year'] = loans['date'].apply(lambda x: int(x[:4]))


# In[58]:


loans['year'].plot()
loans['year'].value_counts().plot(kind="bar",
                                  figsize=(10,10),
                                  fontsize=20)
plt.title('Amount of Kiva loans taken out by the Year',fontsize=16)
plt.show()
print('Amount of Kiva loans taken out by the Year\n')
print(loans['year'].value_counts())


# In[16]:


plt.figure(figsize=(12,9))
loans['repayment_interval'].value_counts().plot(kind='barh')
plt.title('Repayment Intervals\n',fontsize=20)
plt.show()
print('Repayment Intervals\n')
print(loans['repayment_interval'].value_counts().sort_values(ascending=True))


# In[17]:


plt.figure(figsize=(12,9))
sns.distplot(loans[loans['term_in_months']<=80]['term_in_months'],bins=60,kde=False)
plt.title('Term Length in Months of a Loan\n',fontsize=20)
plt.xlabel('Months',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.show()
print('Average Term Length (in months) of a Kiva Loan: ' + str(loans['term_in_months'].mean()))


# In[18]:


plt.figure(figsize=(12,9))
lenders = loans[loans['lender_count']<=100]['lender_count']
sns.distplot(lenders,bins=50,kde=False,color='g')
plt.title('Number of Lenders per Kiva Loan\n',fontsize=20)
plt.xlabel('Lender Count',fontsize=15)
plt.show()


# In[19]:


plt.figure(figsize=(12,9))
loans['activity'].value_counts().head(25).plot.bar()
plt.title('Top 20 Activities\n',fontsize=16)
plt.show()
print('Top 20 Activities\n')
print(loans['activity'].value_counts().head(25))


# In[20]:


activities = loans['activity'][-pd.isnull(loans['activity'])]
wordcloud = WordCloud(max_font_size=50,
                      background_color='black',
                      width=600,
                      height=300).generate(' '.join(activities))
plt.figure(figsize=(12,8))
plt.imshow(wordcloud)
plt.title('Activites WordCloud',fontsize=30)
plt.axis('off')
plt.show()


# In[21]:


#WordCloud for Use
stopwords = set(STOPWORDS)

#more stopwords
more_stopwords = ['buy','sell']
stopwords.update(more_stopwords)

use_desc = loans['use'][-pd.isnull(loans['use'])]
wordcloud = WordCloud(max_font_size=50,
                      stopwords=stopwords,
                      background_color='white',
                      width=600,
                      height=300).generate(' '.join(use_desc))

plt.figure(figsize=(12,8))
plt.imshow(wordcloud)
plt.title('WordCloud for Use Column\n',fontsize=30)
plt.axis('off')
plt.show()


# > I added 'buy' and 'sell' to the stopword already included. I think it is fairly obvious that the reason to take out a micro loan would be to buy or sell. Those are redundant words.

# In[23]:


plt.figure(figsize=(12,9))
loans.groupby('month(post)').count()['posted_time'].plot(kind='bar',color='green')
plt.title('Loan Posted Time by Months\n',fontsize=15)
plt.xlabel('Month',fontsize=15)
plt.show()


# In[24]:


plt.figure(figsize=(12,9))
loans.groupby('month(dis)').count()['posted_time'].plot(kind='bar')
plt.title('Loan Disbursed Time by Months\n',fontsize=15)
plt.xlabel('Month',fontsize=15)
plt.show()


# There is more loan activity in the first half of the year rather than the second half of the year.

# In[25]:


corr = loans.corr()
plt.figure(figsize=(12,9))
sns.heatmap(corr,
            cmap='YlGnBu',
            annot=True)
plt.title('Correlation of Loans Dataset\n',fontsize=15)
plt.show()


# In[26]:


sns.jointplot(data=loans[(loans['loan_amount']<=1200)&(loans['term_in_months']<=30)],
              x='lender_count',
              y='loan_amount',
              kind='hex',
              color='g',
              size=8)


# In[27]:


sns.jointplot(data=loans[(loans['loan_amount']<=1200)&(loans['term_in_months']<=30)],
              x='term_in_months',
              y='loan_amount',
              kind='hex',
              color='purple',
              size=8)


# In[28]:


countryUnfiltered = loans.groupby('country').mean()['funded_amount'].sort_values(ascending=False)


# In[29]:


data = [dict(
        type = 'choropleth',
        locations = countryUnfiltered.index,
        locationmode = 'country names',
        z = countryUnfiltered.values,
        text = countryUnfiltered.index,
        colorscale = 'Blue',
        marker = dict(
            line=dict(width=.7)),
            colorbar = dict(
            autotick = False, 
            tickprefix = '$',
            title = 'Loan Amount'),)]
layout = dict(title = 'Average Loan Amount by Country',
             geo = dict(
                 showframe = False,
                 #showcoastlines = False,
                 projection = dict(
                 type = 'Mercatorodes')))
fig = dict(data=data, layout=layout)
py.iplot(fig,validate=False)
print('Average Loan Amount by Country\n',countryUnfiltered.head(15))


# The data above is misleading in that the top 'funded' countries are also the countries with only a few loans. Let's filter out the countries with less than 50 loans to get a better idea of who the top funded countries are.

# In[30]:


countryFiltered = loans.groupby('country').filter(lambda x: len(x) > 50)
fundedFilterer = countryFiltered.groupby('country').mean()['funded_amount'].sort_values(ascending=False)


# In[31]:


data = [dict(
        type = 'choropleth',
        locations = fundedFilterer.index,
        locationmode = 'country names',
        z = fundedFilterer.values,
        text = fundedFilterer.index,
        colorscale = 'Blue',
        marker = dict(
            line=dict(width=.7)),
        colorbar = dict(
            autotick = False, 
            tickprefix = '$',
            title = 'Loan Amount'),)]
layout = dict(title = 'Average Loan Amount by Country',
             geo = dict(
                 showframe = False,
                 #showcoastlines = False,
                 projection = dict(
                 type = 'Mercatorodes')))
fig = dict(data=data, layout=layout)
py.iplot(fig,validate=False)
print('Filtered Average of Loans by Country\n',fundedFilterer.head(15))


# # Loan Activity - USA v MEX

# In[32]:


usa = loans[loans['country']=='United States']
mex = loans[loans['country']=='Mexico']


# In[33]:


usa['activity'].value_counts().head(10).plot(kind='pie',
                                            fontsize=16,
                                            figsize=(10,10))
plt.title('United States Activity Pie Chart',fontsize=20)
plt.show()


# In[34]:


mex['activity'].value_counts().head(10).plot(kind='pie',
                                            fontsize=16,
                                            figsize=(10,10))
plt.title('Mexico Activity Pie Chart',fontsize=20)
plt.show()


# In[35]:


print('Loan Activity in Monterrey, Mexico:\n\n',mex[mex['region']=='Monterrey']['activity'].value_counts())


# These simple pie charts and table show that not all Kiva loans are used for entrepreneurial purposes, and that some countries, and regions within countries, use this capital for different purposes. The table above shows the ENTIRE loan activity distribution of Monterrey, Mexico.

# # Loan Themes by Region

# In[36]:


loan_theme = pd.read_csv('../input/loan_themes_by_region.csv')


# In[37]:


loan_theme.shape


# In[38]:


loan_theme.head()


# In[39]:


plt.figure(figsize=(12,9))
sns.heatmap(loan_theme.isnull(),yticklabels=False,cbar=False,cmap='plasma')
plt.title('Missing Data?\n',fontsize=20)
plt.show()


# In[40]:


plt.figure(figsize=(12,9))
count = loan_theme['Field Partner Name'].value_counts().head(15)
sns.barplot(count.values, count.index)
plt.title('Field Partner Name and Count\n',fontsize=20)
plt.show()
print('Field Partner Name and Count\n')
print(count)


# In[41]:


plt.figure(figsize=(12,9))
loan_theme['sector'].value_counts().plot(kind='bar')
plt.title('Loans by Sector\n',fontsize=20)
plt.show()
print('Loans by Sector\n')
print(loan_theme['sector'].value_counts())


# In[42]:


plt.figure(figsize=(12,9))
loan_theme['Loan Theme Type'].value_counts().sort_values(ascending=True).tail(15).plot(kind='barh',color='g')
plt.title('Loan Themes\n',fontsize=20)
plt.show()
print('Loan Theme Type\n')
print(loan_theme['Loan Theme Type'].value_counts().head(15))


# In[43]:


#WordCloud for Use
stopwords = set(STOPWORDS)

#more stopwords
more_stopwords = ['buy','sell','general']
stopwords.update(more_stopwords)

themes = loan_theme['Loan Theme Type'][-pd.isnull(loan_theme['Loan Theme Type'])]
wordcloud = WordCloud(max_font_size=50,
                      stopwords=stopwords,
                      background_color='black',
                      width=600,
                      height=300).generate(' '.join(themes))

plt.figure(figsize=(12,8))
plt.imshow(wordcloud)
plt.title('WordCloud for Loan Themes\n',fontsize=30)
plt.axis('off')
plt.show()


# In[44]:


plt.figure(figsize=(12,9))
loan_theme['country'].value_counts().head(15).plot(kind='bar',color='brown')
plt.title('Loans by Country\n',fontsize=20)
plt.show()
print('Loans by Country\n')
print(loan_theme['country'].value_counts().head(15))


# In[45]:


location = loan_theme.groupby('country')['country'].count().sort_values(ascending=False)
data = [dict(
        type = 'choropleth',
        locations = location.index,
        locationmode = 'country names',
        z = location.values,
        text = location.index,
        colorscale = 'Blue',
        marker = dict(
            line=dict(width=.7)),
        colorbar = dict(
            autotick = False, 
            tickprefix = '',
            title = 'Number of Loans'),)]
layout = dict(title = 'Number of Loans by Country',
             geo = dict(
                 showframe = False,
                 #showcoastlines = False,
                 projection = dict(
                 type = 'Mercatorodes')))
fig = dict(data=data, layout=layout)
py.iplot(fig,validate=False)


# Similar to the other dataset, though some countries are a little off. The Philippines is still the top by quite a margin. 

# In[46]:


plt.figure(figsize=(12,9))
sns.distplot(loan_theme[loan_theme['amount']<=10000]['amount'],kde=False,bins=50,color='purple')
plt.title('Dollar Amount per Loan\n',fontsize=20)
plt.ylabel('Count',fontsize=15)
plt.xlabel('Dollars',fontsize=15)
plt.show()


# Filtered out all loans over $10,000 dollars. 

# In[54]:


#rural = loan_theme.groupby('country').filter(lambda x: len(x) > 50)
plt.figure(figsize=(12,9))
sns.distplot(loan_theme['rural_pct'].dropna(),kde=False,color='black',bins=80)
plt.title('Rural Percentage Count\n',fontsize=20)
plt.xlabel('Rural Percentage',)
plt.show()
loan_theme.groupby('country')['rural_pct'].mean().sort_values(ascending=False).head(15)


# Filtered out countries with less than 50 instances.

# In[48]:


countryFilter = loan_theme.groupby('country').filter(lambda x: len(x)>50)
avgCountryAmount = countryFilter.groupby('country')['amount'].mean().sort_values(ascending=True)
avgCountryAmount.plot(kind='barh',
                      figsize=(12,9),
                      color='red')
plt.title('Average Loan Amount by Country (only countries with over 50 loans)\n',fontsize=20)
plt.show()
print(countryFilter.groupby('country')['amount'].mean().sort_values(ascending=False))


# In[49]:


data = [dict(
        type = 'choropleth',
        locations = avgCountryAmount.index,
        locationmode = 'country names',
        z = avgCountryAmount.values,
        text = avgCountryAmount.index,
        colorscale = 'Blue',
        marker = dict(
            line=dict(width=.7)),
        colorbar = dict(
            autotick = False, 
            tickprefix = '$',
            title = 'Loan Amount'),)]
layout = dict(title = 'Average Loan Amount by Country',
             geo = dict(
                 showframe = False,
                 #showcoastlines = False,
                 projection = dict(
                 type = 'Mercatorodes')))
fig = dict(data=data, layout=layout)
py.iplot(fig,validate=False)


# Hmm. These loan amounts are much higher than the amounts in the other dataset.

# In[50]:


partner = loan_theme[['Field Partner Name','country','rural_pct']].dropna()
partner = partner.groupby('Field Partner Name').filter(lambda x: len(x) >= 50)


# In[55]:


plt.figure(figsize=(12,9))
sns.distplot(partner.groupby('Field Partner Name')['rural_pct'].mean().sort_values(ascending=False),kde=False,bins=60)
plt.title('Field Partners and their Rural Percentages\n',fontsize=20)
plt.xlabel('Rural Percentage',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.show()
print(partner.groupby('Field Partner Name')['rural_pct'].mean().sort_values(ascending=False))


# This is a list and histogram of the count and rural percentages of the Field Partners with over 50 instances (loans). 

# # That is my quick and dirty EDA. More to come soon. Feedback is appreciated!

# In[ ]:




