#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


kiva_loan = pd.read_csv('../input/kiva_loans.csv')


# In[4]:


pd.isnull(kiva_loan).any()


# In[5]:


kiva_loan.head()


# In[6]:


kiva_loan.describe()


# The median amount of loan on Kiva is 450 dollars . The minimum loan amaount is 25 dollars  and the maximum loan amaount is 100000 dollars. We can also tell that on average each loan has 20 contributors. The average  number of months a loan is expected to be paid back is 13 and the maximum amount of time was 158 months while the minimum amount of time is one month.

# In[7]:


sns.countplot(data=kiva_loan, y='sector')


# The majority of people who have gotten loans on Kiva are in Agriculture and food sectors which are very much related. The third sector is Retail. 

# In[8]:


kiva_loan['country'].nunique()


# This dataset has records from 87 different countiries.

# In[9]:


kiva_loan['country'].unique()


# I am very curious to know which of these nations have the largest loan amount.

# In[10]:


loan_amount_by_nation = kiva_loan.groupby('country')['funded_amount'].sum().reset_index()
loan_amount_by_nation.columns = ['Country','Total Amount']
loan_df = loan_amount_by_nation.sort_values(by='Total Amount',ascending=False)
loan_df.head(10)


# Philippines is leading woth the total number of loans at 54476375 dollars follolwed closely by the Republic of Kenya at 32248405, Peru and Paraguay. The figure below represents the top 10 borrowers. 

# In[11]:


plt.figure(figsize=(12,6))
sns.barplot(data=loan_df.head(10),x='Country',y='Total Amount')


# I would also like to know the last 10 borrowers

# In[12]:


plt.figure(figsize=(12,6))
sns.barplot(data=loan_df.tail(10),x='Country',y='Total Amount')


# Which sector are most of the borrowers in from the top nation?

# In[13]:


phil = kiva_loan[kiva_loan['country'] == 'Philippines']


# In[14]:


sns.countplot(data=phil,y='sector')


# In Philippinesmost of the borrowers are in Retail, food and agriculture follow closely

# In[15]:


ke = kiva_loan[kiva_loan['country'] == 'Kenya']


# In[16]:


sns.countplot(data=ke,y='sector')


# In Kenya most of the borrowers are in Agricture. It would probably be good idea for KIva to come up with products that are specifically targeted towards people who are in food and agriculture since they form majority of the borrowers. 

# In[17]:


region_count = ke.groupby('region')['funded_amount'].sum().reset_index()
region_count.columns= ['Region','Total Amount Funded']
region_count.sort_values(by='Total Amount Funded',ascending=False).head(10)


# It is not surprising that Webuye is leading, owing to the fact that is known for subsistence farming in Kenya.

# In[18]:


piv = ke.pivot_table(columns=['region','sector'],aggfunc='count')
piv


# I am curious to see the most used words in activity and use

# In[19]:


from wordcloud import WordCloud, STOPWORDS
corpus = ' '.join(kiva_loan['activity'])
corpus = corpus.replace('.', '. ')
wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white',width=2400,height=2000).generate(corpus)
plt.figure(figsize=(12,15))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# We ealrier noticed that agriculture was leading in the number of loans, it is not suprising that Agricultural related activities such as farming and production appear many times.

# In[20]:


from wordcloud import WordCloud, STOPWORDS
corpus = ' '.join(kiva_loan['use'].astype(str))
corpus = corpus.replace('.', '. ')
wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white',width=2400,height=2000).generate(corpus)
plt.figure(figsize=(12,15))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# Most of the loans were used for business like activities like buying and selling. For example buying fertilizer, selling gloceries and purchasing additional stock

# More on the way...

# In[ ]:




