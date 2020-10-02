#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/fec-independent-expenditures.csv')


# In[ ]:


df.head()


# # Top Ten Payee (Non-Contributor)

# In[ ]:


table_count = df.groupby(df['payee_name'])['expenditure_amount'].sum()
table_count = table_count.sort_values(ascending=False)[:10]
payee_index = table_count.index
payee_val = table_count.values
sns.barplot(x = payee_val,y=payee_index,orient='h')
plt.ylabel('Payee Name')
plt.xlabel('Expenditure Amount')


# # Number of Reported  Year

# In[ ]:


fig,ax = plt.subplots(figsize=(8,6))
sns.countplot(df.report_year,ax=ax)
#ticks = plt.setp(ax.get_xticklabels(),rotation=45)
plt.title('Number of Reported Year')
plt.xlabel('Year')
plt.ylabel('Count')


# # Committee Office Dollars

# In[ ]:


table_count = df.groupby(df.committee_name)['office_total_ytd'].sum()
table_count = table_count.sort_values(ascending=False)[:10]
committe_office_idx = table_count.index
committe_office_val = table_count.values
sns.barplot(x = committe_office_val,y=committe_office_idx,orient='h')
plt.title('Committe Office Dollars')
plt.ylabel('Committe Name')
plt.xlabel('Dollars')


# # Category vs Expenditure Amount

# In[ ]:


table_count = df.groupby(df.category_code_full)['expenditure_amount'].sum()
table_count = table_count.sort_values(ascending=False)[:10]
category_code_idx = table_count.index
category_code_val = table_count.values
fig,ax = plt.subplots(figsize=(8,6))
sns.barplot(x = category_code_val,y=category_code_idx,orient='h')
plt.ylabel('Category')
plt.xlabel('Expenditure Amount')
plt.title('Category vs Expenditure Amount')


# # Candidate Office Count

# In[ ]:


plt.title('Candidate Office Count')
sns.countplot(df.candidate_office)
plt.ylabel('Count')
plt.xlabel('Candidate Office')


# In[ ]:


df['support_oppose_indicator'].unique()


# # Number of contributor opposing Trump by state

# In[ ]:


# 'TRUMP, DONALD J'
trump_entry = df[df.candidate_name == 'TRUMP, DONALD J']
trump_entry = trump_entry[trump_entry.support_oppose_indicator == 'O']
trump_state = trump_entry.groupby(trump_entry.payee_state)              [['payee_state','support_oppose_indicator']].size()

trump_index = trump_state.index
trump_val = trump_state.values
fig,ax = plt.subplots(figsize=(8,6))
sns.barplot(x = trump_val,y=trump_index,orient='h',ax=ax)
plt.title('Number of contributor opposing Trump by state')
plt.ylabel('State')
plt.xlabel('Count')


# # Support Trump 

# In[ ]:


# 'TRUMP, DONALD J'
support =['S','SUP']
trump_entry = df[df.candidate_name == 'TRUMP, DONALD J']
trump_entry = trump_entry[trump_entry.support_oppose_indicator.isin(support)]
trump_state = trump_entry.groupby(trump_entry.payee_state)              [['payee_state','support_oppose_indicator']].size()

trump_index = trump_state.index
trump_val = trump_state.values
fig,ax = plt.subplots(figsize=(8,6))
sns.barplot(x = trump_val,y=trump_index,orient='h',ax=ax)
plt.title('Number of contributor Supporting Trump by state')
plt.ylabel('State')
plt.xlabel('Count')


# # Oppose Obama by state

# In[ ]:


# 'OBAMA, BARACK'
obama_entry = df[df.candidate_name == 'OBAMA, BARACK']
obama_entry = obama_entry[obama_entry.support_oppose_indicator == 'O']
obama_state = obama_entry.groupby(obama_entry.payee_state)              [['payee_state','support_oppose_indicator']].size()
obama_idx = obama_state.index
obama_val = obama_state.values

fig,ax = plt.subplots(figsize=(8,6))
sns.barplot(x = obama_val , y=obama_idx,ax=ax,orient='h')
plt.title('Number of contributor opposing Obama by State')
plt.ylabel('State')
plt.xlabel('Count')


# # Support Obama by state

# In[ ]:


# 'OBAMA, BARACK'
obama_entry = df[df.candidate_name == 'OBAMA, BARACK']
obama_entry = obama_entry[obama_entry.support_oppose_indicator.isin(support)]
obama_state = obama_entry.groupby(obama_entry.payee_state)              [['payee_state','support_oppose_indicator']].size()
obama_idx = obama_state.index
obama_val = obama_state.values


fig,ax = plt.subplots(figsize=(8,6))
ax = sns.barplot(x = obama_val , y=obama_idx,ax=ax,orient='h')
 
plt.title('Number of contributor supporting Obama by State')
plt.ylabel('State')
plt.xlabel('Count')


# # Number of contributor opposing Hillary by state

# In[ ]:


#Hilary Clinton
# 'HILLARY RODHAM'
#HILLYER, RICHARD Q
#'CLINTON, HILLARY RODHAM'
hillary_entry = df[df.candidate_name =='CLINTON, HILLARY RODHAM']
hillary_state = hillary_entry[hillary_entry.support_oppose_indicator =='O']
hillary_state = hillary_state.groupby(hillary_state.payee_state)                [['payee_state','support_oppose_indicator']].size()
hillary_idx = hillary_state.index
hillary_val = hillary_state.values

fig,ax = plt.subplots(figsize=(8,6))
sns.barplot(x = hillary_val,y=hillary_idx,orient='h')
plt.title('Number of contributor opposing Hillary by state')
plt.ylabel('State')
plt.xlabel('Count')


# # Number of contributor supporting Hillary by state

# In[ ]:


#Hilary Clinton
# 'HILLARY RODHAM'
#HILLYER, RICHARD Q
#'CLINTON, HILLARY RODHAM'
hillary_entry = df[df.candidate_name =='CLINTON, HILLARY RODHAM']
hillary_state = hillary_entry[hillary_entry.support_oppose_indicator.isin(support)]
hillary_state = hillary_state.groupby(hillary_state.payee_state)                [['payee_state','support_oppose_indicator']].size()
hillary_idx = hillary_state.index
hillary_val = hillary_state.values

fig,ax = plt.subplots(figsize=(8,6))
sns.barplot(x = hillary_val,y=hillary_idx,orient='h')
plt.title('Number of contributor Supporting Hillary by state')
plt.ylabel('State')
plt.xlabel('Count')


# # Number of contributor supporting rate Hillary by state

# In[ ]:


#Hilary Clinton
# 'HILLARY RODHAM'
#HILLYER, RICHARD Q
#'CLINTON, HILLARY RODHAM'
hillary_entry = df[df.candidate_name =='CLINTON, HILLARY RODHAM']
total_hillary = len(hillary_entry)
hillary_state = hillary_entry[hillary_entry.support_oppose_indicator.isin(support)]
hillary_state = hillary_state.groupby(hillary_state.payee_state)                [['payee_state','support_oppose_indicator']].size()
hillary_idx = hillary_state.index
hillary_val = hillary_state.values
hillary_prob = hillary_val / total_hillary

fig,ax = plt.subplots(figsize=(8,6))
sns.barplot(x = hillary_prob,y=hillary_idx,orient='h')
plt.title('Number of contributor Supporting Hillary by state')
plt.ylabel('State')
plt.xlabel('Support Probability')


# # Number of contributor opposing rate Hillary by state

# In[ ]:


#Hilary Clinton
# 'HILLARY RODHAM'
#HILLYER, RICHARD Q
#'CLINTON, HILLARY RODHAM'
hillary_entry = df[df.candidate_name =='CLINTON, HILLARY RODHAM']
total_hillary = len(hillary_entry)
hillary_state = hillary_entry[hillary_entry.support_oppose_indicator == 'O']
hillary_state = hillary_state.groupby(hillary_state.payee_state)                [['payee_state','support_oppose_indicator']].size()
hillary_idx = hillary_state.index
hillary_val = hillary_state.values
hillary_prob = hillary_val / total_hillary

fig,ax = plt.subplots(figsize=(8,6))
sns.barplot(x = hillary_prob,y=hillary_idx,orient='h')
plt.title('Number of contributor Opposing Hillary by state')
plt.ylabel('State')
plt.xlabel('Opposing Probability')


# # # Number of contributor supporting rate Trump by state

# In[ ]:


# 'TRUMP, DONALD J'
support =['S','SUP']
trump_entry = df[df.candidate_name == 'TRUMP, DONALD J']
total_trump = len(trump_entry)
trump_entry = trump_entry[trump_entry.support_oppose_indicator.isin(support)]
trump_state = trump_entry.groupby(trump_entry.payee_state)              [['payee_state','support_oppose_indicator']].size()

trump_index = trump_state.index
trump_val = trump_state.values
trump_prob = trump_val / total_trump
fig,ax = plt.subplots(figsize=(8,6))
sns.barplot(x = trump_prob,y=trump_index,orient='h',ax=ax)
plt.title('Number of contributor Supporting Trump by state')
plt.ylabel('State')
plt.xlabel('Supporting Probability')


# # 

# # Number of contributor opposing rate Trump by state

# In[ ]:


# 'TRUMP, DONALD J'
support =['S','SUP']
trump_entry = df[df.candidate_name == 'TRUMP, DONALD J']
total_trump = len(trump_entry)
trump_entry = trump_entry[trump_entry.support_oppose_indicator == 'O']
trump_state = trump_entry.groupby(trump_entry.payee_state)              [['payee_state','support_oppose_indicator']].size()

trump_index = trump_state.index
trump_val = trump_state.values
trump_prob = trump_val / total_trump
fig,ax = plt.subplots(figsize=(8,6))
sns.barplot(x = trump_prob,y=trump_index,orient='h',ax=ax)
plt.title('Number of contributor Opposing Trump by state')
plt.ylabel('State')
plt.xlabel('Opposing Probability')


# #total Trump support vs total Hillary support

# In[ ]:


# 'TRUMP, DONALD J'
support =['S','SUP']
trump_entry = df[df.candidate_name == 'TRUMP, DONALD J']
total_trump = len(trump_entry)
trump_entry = trump_entry[trump_entry.support_oppose_indicator.isin(support)]
trump_state = trump_entry.groupby(trump_entry.payee_state)              [['payee_state','support_oppose_indicator']].size()

trump_index = trump_state.index
trump_val = trump_state.values
trump_prob = trump_val / total_trump
total_trump_support_prob = trump_prob.sum()

#Hilary Clinton
# 'HILLARY RODHAM'
#HILLYER, RICHARD Q
#'CLINTON, HILLARY RODHAM'
hillary_entry = df[df.candidate_name =='CLINTON, HILLARY RODHAM']
total_hillary = len(hillary_entry)
hillary_state = hillary_entry[hillary_entry.support_oppose_indicator.isin(support)]
hillary_state = hillary_state.groupby(hillary_state.payee_state)                [['payee_state','support_oppose_indicator']].size()
hillary_idx = hillary_state.index
hillary_val = hillary_state.values
hillary_prob = hillary_val / total_hillary
total_hillary_support_prob = hillary_prob.sum()

total_support_prob =[]
total_support_prob.append(total_trump_support_prob)
total_support_prob.append(total_hillary_support_prob)
candidate = ['TRUMP, DONALD J','CLINTON, HILLARY RODHAM']
support_prob = pd.DataFrame({'Candidate':candidate,'Support Rate':total_support_prob})
sns.barplot(data=support_prob,x='Candidate',y='Support Rate')
plt.ylabel('Support Rate')
plt.xlabel('Candidate')
plt.title('Trump vs Hillary Supporting Rate')


# In[ ]:


# 'TRUMP, DONALD J'
support =['S','SUP']
df['support_oppose_indicator'].dropna(axis=0,inplace=True)
trump_entry = df[df.candidate_name == 'TRUMP, DONALD J']
total_trump = len(trump_entry)
trump_entry = trump_entry[trump_entry.support_oppose_indicator.isin(support)]
#trump_state = trump_entry.groupby(trump_entry.payee_state)\
#              [['payee_state','support_oppose_indicator']].size()

trump_index = trump_state.index
trump_val = trump_state.values
#trump_prob = trump_val / total_trump
#total_trump_support_prob = trump_prob.sum()

#Hilary Clinton
# 'HILLARY RODHAM'
#HILLYER, RICHARD Q
#'CLINTON, HILLARY RODHAM'
hillary_entry = df[df.candidate_name =='CLINTON, HILLARY RODHAM']
total_hillary = len(hillary_entry)
hillary_state = hillary_entry[hillary_entry.support_oppose_indicator.isin(support)]
#hillary_state = hillary_state.groupby(hillary_state.payee_state)\
#                [['payee_state','support_oppose_indicator']].size()
#hillary_idx = hillary_state.index
#hillary_val = hillary_state.values
#hillary_prob = hillary_val / total_hillary
#total_hillary_support_prob = hillary_prob.sum()

candidate = ['TRUMP, DONALD J','CLINTON, HILLARY RODHAM']
columns=['Candidate','State','Support Rate']
trump_vs_hillary_by_state=pd.DataFrame(columns=columns)
#support_values = []
for candidate in candidate:
    for state in df.payee_state.unique():
        if candidate == 'TRUMP, DONALD J':
            trump_state_support = trump_entry[trump_entry.payee_state == state]                                  ['support_oppose_indicator'].value_counts().values
            if trump_state_support :
                support_values = trump_state_support.tolist()[0] 
            else:
                support_values = 0
            entry = pd.DataFrame([[candidate,state,support_values]],columns=columns)
            
                
        else:
            hillary_state_support=hillary_state[hillary_state.payee_state == state]                              ['support_oppose_indicator'].value_counts().values
            if hillary_state_support:
                support_values = hillary_state_support.tolist()[0]
            else:
                support_values = 0
            entry = pd.DataFrame([[candidate,state,support_values]],columns=columns)
            
        trump_vs_hillary_by_state = trump_vs_hillary_by_state.append(entry)
trump_vs_hillary_by_state.head()


# In[ ]:


sns.set_style("whitegrid")
sns.set_color_codes('pastel')
fig,ax = plt.subplots(figsize=(10,6))
sns.barplot(data=trump_vs_hillary_by_state,x='State',
            y='Support Rate',hue='Candidate',ax=ax)
ticks = plt.setp(ax.get_xticklabels(),rotation=90)
plt.legend(loc='best',frameon=True)
plt.ylabel('Support Rate')
plt.title('Trump vs Hillary')


# # Hillary Indicator by state

# In[ ]:


hillary_entry = df[df.candidate_name =='CLINTON, HILLARY RODHAM']
columns=['State','Indicator','Length']
hillary_indicator_state = pd.DataFrame(columns=columns)
for state in hillary_entry.payee_state.unique():
    curr_state = hillary_entry[hillary_entry.payee_state == state]
    for indicator in hillary_entry.support_oppose_indicator.unique():
        state_indicator=len(curr_state[curr_state['support_oppose_indicator'] == indicator])
        entry = pd.DataFrame([[state,indicator,state_indicator]],columns=columns)
        hillary_indicator_state = hillary_indicator_state.append(entry)
        
hillary_indicator_state.head()


# In[ ]:


fig,ax = plt.subplots(figsize=(10,6))
sns.barplot(data=hillary_indicator_state,x='State',y='Length',hue='Indicator',ax=ax)
plt.title('Hillary Indicator by state')
plt.ylabel('Indicator Count')
ticks = plt.setp(ax.get_xticklabels(),rotation=90)


# # Trump indicator by state

# In[ ]:


trump_entry = df[df.candidate_name == 'TRUMP, DONALD J']
columns=['State','Indicator','Length']
trump_indicator_state = pd.DataFrame(columns=columns)
for state in trump_entry.payee_state.unique():
    curr_state = trump_entry[trump_entry.payee_state == state]
    for indicator in trump_entry.support_oppose_indicator.unique():
        state_indicator=len(curr_state[curr_state['support_oppose_indicator'] == indicator])
        entry = pd.DataFrame([[state,indicator,state_indicator]],columns=columns)
        trump_indicator_state = trump_indicator_state.append(entry)
        
trump_indicator_state.head()


# In[ ]:


fig,ax = plt.subplots(figsize=(10,6))
sns.barplot(data=trump_indicator_state,x='State',y='Length',hue='Indicator',ax=ax)
plt.title('Trump Indicator by state')
plt.ylabel('Indicator Count')
ticks = plt.setp(ax.get_xticklabels(),rotation=90)


# In[ ]:




