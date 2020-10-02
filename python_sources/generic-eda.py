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
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


datasets = {}
datadir = '../input'
for f in os.listdir(datadir):
    datasets[f.replace('.csv', '')] = pd.read_csv(os.path.join(datadir, f))
datasets.keys()


# In[ ]:


datasets['kiva_loans'].head()


# Lets start by looking at the loan amount by various categories such as sector, repayment method, gender etc. to gain an understanding of the loans given out by Kiva

# In[ ]:


def barplots(df, x, y, fig_width = 16, fig_height= 6, rot = 45, ax = None):
    plt.figure(figsize = (fig_width, fig_height))
    plt.xticks(rotation=rot)
    if type(df) == type('str'):
        return sns.barplot(x = x, y =y, ax = ax, data = datasets[df].groupby(x)[y].sum().sort_values().reset_index())
    else:
        return sns.barplot(x = x, y =y, ax = ax, data = df.groupby(x)[y].sum().sort_values().reset_index())

x, y = 'sector', 'loan_amount'
barplots('kiva_loans', x, y)


# The top three categories are Agriculture, Food and Retail, which does make sense as Kiva is a non-profit that is helping people lead a better life. Lets look at loan by the repayment type

# In[ ]:


x, y = 'repayment_interval', 'loan_amount'
barplots('kiva_loans', x, y)


# One of the things that I know about such small scale lending is that is usually females who usually take out such loans to help their family or grow their cottage industry. Lets see if that is the case, but before we can do that we need to process the data a bit. The 'borrower_genders' column contains the gender of each of the borrower so we need to divide the loan amount by gender based on the percentage that gender has in the list in 'borrower_genders' column

# In[ ]:


def getLoanAmtGender(loan_row):
    loan = float(loan_row.loan_amount)

    borrowers = loan_row.borrower_genders.strip(',').split(', ')
    male_borrow_amt = loan * len(list(filter(lambda x: x == 'male', borrowers))) / len(borrowers)
    female_borrow_amt = loan * len(list(filter(lambda x: x == 'female', borrowers))) / len(borrowers)
    loan_row['male_borrow_amt'] = male_borrow_amt
    loan_row['female_borrow_amt'] = female_borrow_amt
    return loan_row
loanAmtGender = datasets['kiva_loans'][~datasets['kiva_loans'].borrower_genders.isnull()].apply(getLoanAmtGender, axis = 1)
loanAmtGender


# In[ ]:


x, y = 'male_borrow_amt', 'female_borrow_amt'
sns.barplot(x = 'borrowed by', y ='amt borrowed', data = loanAmtGender[[x, y]].sum().reset_index().rename(columns={'index': 'borrowed by', 0: 'amt borrowed'}))


# This is same as what I had hoped for, females have borrowed more than twice the amount borrowed by males. It would be interesting to see the two charts above, loans by sector and loans by repayment type to see if there is a gender preference when comes to taking out loans

# In[ ]:


fig = plt.figure() # Create matplotlib figure
ax = fig.add_subplot(111)
loanAmtGender.groupby('repayment_interval')['female_borrow_amt'].sum().plot.bar(ax=ax, width=0.2, position=1, color=sns.color_palette("Paired", n_colors=1))
loanAmtGender.groupby('repayment_interval')['male_borrow_amt'].sum().plot.bar(ax=ax, width=0.2, position=0, color=sns.color_palette("Paired", n_colors=4)[-2])
ax.legend(['Females', 'Males'])


# In[ ]:


fig = plt.figure(figsize = (16, 6)) # Create matplotlib figure
ax = fig.add_subplot(111)
loanAmtGender.groupby('sector')['female_borrow_amt'].sum().plot.bar(ax=ax, width=0.2, position=1, color=sns.color_palette("Paired", n_colors=1))
loanAmtGender.groupby('sector')['male_borrow_amt'].sum().plot.bar(ax=ax, width=0.2, position=0, color=sns.color_palette("Paired", n_colors=4)[-2])
ax.legend(['Females', 'Males'])


# There is no major difference in the type of loans taken out by the two genders. The values just suggest that females utilize such opportunity of small loans much more than  males which is a known trend
# 
# ### Lets Look at the loans now by Country

# In[ ]:


x, y = 'country', 'loan_amount'
barplots('kiva_loans', x, y, fig_width = 24, fig_height= 8, rot = 90)


# It is interesting that Philippines get way more money than other companies. Lets see how the loan distribution looks like in Philippines

# In[ ]:


minmumLoanCutOff = 100000
plt.figure(figsize = (16, 6))
plt.xticks(rotation=45)
sns.barplot(x = 'sector', y ='loan_amount', data = datasets['kiva_loans'][datasets['kiva_loans'].country == 'Philippines'].groupby('sector').loan_amount.sum().reset_index())


# This is very interesting, the loan distribution in Phillipines the largest receiver of the loans, is different than the general loan distribution. I think the type of load issued depends on the country, for e.g. in developing or developed country more loans would be given out for services and retail rather than agriculture. To check this lets see the distribution of the next 3 countries by loan amount - Kenya, United States, Peru

# In[ ]:


f, axes = plt.subplots(3, 1, figsize = (16, 20))
countries = ['Kenya', 'United States', 'Peru']
for i, country in enumerate(countries):
        sns.barplot(x = 'sector', y ='loan_amount', data = datasets['kiva_loans'][datasets['kiva_loans'].country == country].groupby('sector').loan_amount.sum().reset_index(), ax=axes[i])
        axes[i].set_title(country)


# Hmmm, it does sort of follows that. Just looking at the loans sector wise does not provide us with a lot of information as to why the loan was given. Maybe we should use the 'use' column to figure out how exactly the loan was used
