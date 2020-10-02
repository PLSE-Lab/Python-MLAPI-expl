#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from zipfile import ZipFile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import Counter
import random


# In[ ]:


def makeDf(file):
    return file[:-4], pd.read_csv(os.path.join('../input',file))
dataFiles = {}
for file in [f for f in os.listdir('../input') if f.endswith('csv')]:
    f_name, df = makeDf(file)
    dataFiles[f_name] = df


# In[ ]:


dataFiles.keys()


# In[ ]:


dataFiles['kiva_loans'].head(3)


# # Performing a lookup by ID

# In[ ]:


dataFiles['loan_theme_ids'].where(dataFiles['loan_theme_ids']['id'] == 653068).dropna()


# In[ ]:


kiva_startup_loans = dataFiles['loan_theme_ids'].where(dataFiles['loan_theme_ids']['Loan Theme Type'] == 'Startup').dropna()


# # Cursory look at the data

# In[ ]:


theme_type_frequency = Counter(dataFiles['loan_theme_ids']['Loan Theme Type'])


# In[ ]:


#sorted(theme_type_frequency)
#plt.figure(figsize=(96,12))
plt.figure(figsize=(12,96))
x = range(len(theme_type_frequency))
y = theme_type_frequency.values()
#bar = plt.bar(theme_type_frequency.keys(), y)
bar = plt.barh(x,list(y))
#plt.xticks(rotation=90)
plt.yticks(x, theme_type_frequency.keys())
plt.savefig('Kiva_Theme_Distributions.png', orientation='landscape', transparent=False, )


# In[ ]:


themes = theme_type_frequency.keys()


# In[ ]:


dataFiles['kiva_loans'].describe()


# In[ ]:


dataFiles['kiva_mpi_region_locations'].describe()


# In[ ]:


dataFiles['loan_theme_ids'].describe()


# In[ ]:


loan_ids = list(kiva_startup_loans['id'])


# In[ ]:


len(loan_ids)


# ## 1. Find Startup loans without a matching id

# In[ ]:





# ## 2. Find Loans where the Activity is NAN 

# In[ ]:





# # Drilling Down

# ## Loan Size Averages by Activity

# In[ ]:


loan_sizes = dataFiles['kiva_loans'].groupby(['activity']).mean()[['loan_amount','term_in_months']]


# In[ ]:


loan_sizes.head(3)


# In[ ]:


loan_sizes.describe()


# In[ ]:


average_monthly_loan_payment = loan_sizes['loan_amount'].mean() / loan_sizes.term_in_months.mean()


# In[ ]:


print(average_monthly_loan_payment)


# ## Activities by Partner

# In[ ]:


partners = list(set(dataFiles['kiva_loans']['partner_id'].dropna().astype(int)))


# In[ ]:


countries = list(set(dataFiles['kiva_loans']['country'].dropna()))


# In[ ]:


countries = sorted(countries)


# In[ ]:


df = dataFiles['kiva_loans'].where(dataFiles['kiva_loans']['partner_id'] == partners[2]).dropna()
df


# In[ ]:


loans_per_partner = {}
for p in partners:
    try:
        df = dataFiles['kiva_loans'].where(dataFiles['kiva_loans']['partner_id'] == p).dropna()
        loans_per_partner[p] = dict(Counter(list(df['country'])))
    except IndexError as e:
        print('Partner Number: {}'.format(p))
        print(e)


# In[ ]:


loans_per_partner


# In[ ]:


df = dataFiles['kiva_loans'].where(dataFiles['kiva_loans']['partner_id'] == 462).dropna()
df.where(df['country'] == 'Israel').dropna()


# In[ ]:


loans_per_partner[462]


# In[ ]:


country_markers = (list(range(len(countries))))


# In[ ]:


def returnXY(d):
    x = []
    y = list(d.values())
    for c in d.keys():
        x.append(countries.index(c))
    return(x,y)


# In[ ]:


plt.figure(figsize=(96,96))
plt.xticks(country_markers, countries, rotation=90)
plt.yticks(partners)
for partner in loans_per_partner:
    try:
        x,y = returnXY(loans_per_partner[partner])
        if len(x) >= 1:
            plt.scatter(x, [partner for x in range(len(x))], y,)
    except:
        print(partner)
plt.grid(True)
plt.show()


# # High Risk Loans

# ## 1. Loans with Short Terms

# In[ ]:




