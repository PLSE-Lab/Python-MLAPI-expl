#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
#import networkx as nx

#from subprocess import check_output
#print(check_output(['ls', '../input']).decode('utf8'))


# In[12]:


df_DEMO = pd.read_csv('../input/demographic.csv')
#df_DIET = pd.read_csv('../input/diet.csv')
#df_EXAM = pd.read_csv('../input/examination.csv')
#df_LABS = pd.read_csv('../input/labs.csv')
#df_QUES = pd.read_csv('../input/questionnaire.csv')

# there is a weird character in the meds list somewhere
df_MEDS = pd.read_csv('../input/medications.csv', encoding = 'latin1')


# Survey overview

# In[13]:


num_patients = pd.Series(df_DEMO['SEQN'].dropna().unique()).count()

# list of male patient identifiers for later
num_men = pd.Series(df_DEMO.query('RIAGENDR == 1')['SEQN'].dropna().unique()).count()
ids_men = pd.Series(df_DEMO.query('RIAGENDR == 1')['SEQN'].dropna().unique()).tolist()
# list of female patient identifiers for later
num_women = pd.Series(df_DEMO.query('RIAGENDR == 2')['SEQN'].dropna().unique()).count()
ids_women = pd.Series(df_DEMO.query('RIAGENDR == 2')['SEQN'].dropna().unique()).tolist()

print('Found',num_patients,'patients,',num_men,'men and',num_women,'women.')


# In[14]:


df_MEDS.head()


# How many unique medications are there?

# In[15]:


# Remove no answer (NaN), unknown (55555), refused (77777), and don't know (99999)
med_names = pd.Series(df_MEDS.query(
    "RXDDRUG not in ['99999', '55555', '77777']")['RXDDRUG'].dropna().unique())
    # NB: appending .unique() is eqivalent to calling the set of .tolist()

print('Found',len(med_names),'unique generics.')
#print(('There were {0} answers, {1} unknown, {2} refused, {3} don\'t know, and {4} '
#       'blank.').format(
#           len(df_MEDS.query("RXDDRUG not in ['99999', '55555', '77777']")['RXDDRUG'].dropna()),
#           len(df_MEDS.query("RXDDRUG == '55555'")['RXDDRUG']),
#           len(df_MEDS.query("RXDDRUG == '77777'")['RXDDRUG']),
#           len(df_MEDS.query("RXDDRUG == '99999'")['RXDDRUG']),
#           len(df_MEDS[pd.isnull(df_MEDS['RXDDRUG'])])
#       ))

pop_meds = df_MEDS.groupby('RXDDRUG').size().sort_values(ascending=False)

print()
print('The most common medications are:')
for i, n in enumerate(pop_meds.index[:10].tolist()):
    print('    '+str(i+1)+'.',n)
print()


# Popular medications for men vs. women

# Demographics

# In[16]:


# Whole population
df_DEMO.RIDAGEYR.hist()
plt.suptitle('Distribution of age')

print('For the whole population:\n')
n_meds_taken_whole = df_MEDS['RXDCOUNT'].dropna().astype(int)
pop_meds_whole = df_MEDS.groupby('RXDDRUG'
                                ).size().sort_values(ascending=False).index[:10].tolist()
print('    People were most commonly taking',int(n_meds_taken_whole.mode()),
      'medications; on average they were taking',str(int(n_meds_taken_whole.mean()))+'.')
print(('    The most common medications were {0}, {1}, and {2}'
       '.').format(pop_meds_whole[0],pop_meds_whole[1],pop_meds_whole[2]))


# In[17]:


# Population under 18 years old
print('For people under 18 years old:')
pop_under_18 = df_DEMO.query('RIDAGEYR < 18')['SEQN']

ids_under_18 = set(pop_under_18)



print('    There were',pop_under_18.count(),'people under 18.')


# In[18]:


###
# Population 
###
print('For people over 60 years old:')

###
# Population 
# RIDEXPRG: 
# 1 = pregnant
# 2 = not pregnant
# 3 = unknown
# . = missing
###
print('For pregnant women:')

###
# Population 
# RIAGENDR:
# 1 = male
# 2 = female
# . = missing
###
print('For women 18-35 years old:')

###
# Population 
###
print('For men 18-35 years old:')

