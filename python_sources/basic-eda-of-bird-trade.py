#!/usr/bin/env python
# coding: utf-8

# ## Exploratory data analysis of the bird trade using CITES Wildlife Trade Database
# 
# [A guide to using the CITES Trade Database](https://trade.cites.org/cites_trade_guidelines/en-CITES_Trade_Database_Guide.pdf)
# 
# *All the abbrevations used in the dataset are explained in the above resource.*

# ### Import the necessary libraries

# In[47]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[48]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')


# ### Load the dataset into a Pandas dataframe

# In[49]:


df = pd.read_csv('../input/comptab_2018-01-29 16_00_comma_separated.csv', delimiter=',')
df.head()


# In[50]:


df.info()


# With the interest being only birds ('Aves' Class). A subset of the dataframe with only *Aves* Class was selected for further analysis.

# In[51]:


birds = df[df['Class'] == 'Aves']
birds.head()


# In[52]:


birds.info()


# The *Origin, Importer reported quantity, Exported reported quantity and Unit* had been dropped from further analysis as these columns contained quite a lot of missing data.

# In[53]:


birds = birds.drop(['Origin', 'Importer reported quantity', 'Exporter reported quantity', 'Unit'], axis=1)


# ### Data visualization

# #### Year during which the trade occured.

# In[54]:


f, ax = plt.subplots(1, figsize=(6,4))
sns.countplot(birds.Year, ax=ax)
print(birds.Year.value_counts())


# Most of the data in the dataset was from the year 2016.

# #### Trade vs App.

# In[55]:


plt.figure(figsize=(12,7))
sns.countplot(birds['App.'])


# #### Source and Purpose of Trade

# In[56]:


f, ax = plt.subplots(2, figsize=(12,14))
sns.countplot(birds.Source, ax=ax[0])
sns.countplot(birds.Purpose, ax=ax[1])
print(birds.Source.value_counts())
print(birds.Purpose.value_counts())


# * Birds bred in captivity, birds found in the wild and birds born in captivity accounted for the highest amount of trade.
# * Commercial, personal and scientific led the chart when it came to purpose.

# In[57]:


x = birds[birds['Source'] == 'C']
y = birds[birds['Source'] == 'W']
z = birds[birds['Source'] == 'F']
temp = x.merge(y, how='outer')
top_source = temp.merge(z, how='outer')
plt.figure(figsize=(12,7))
sns.countplot(top_source.Purpose, hue=top_source.Source)


# * Most of the birds bred in captivity and found in the wild were traded for commercial use.
# * Most of the birds born in captivity were traded for personal and commercial use.

# #### Trade by country

# In[58]:


chk_lst = ['Importer', 'Exporter']
for name in chk_lst:
    print(name)
    print(birds[name].unique())


# Due to a large count of unique values, only the top 10 importers and exporters were visualized.

# **Imports**

# In[59]:


birds_importer = birds.Importer.value_counts(normalize=True)
plt.figure(figsize=(12,7))
birds_importer.head(10).plot('bar')
plt.xticks(rotation=0)
plt.xlabel('Importer')
plt.ylabel('Normalized Count')
plt.title('Top 10 Importers')
print('Top 10 countries contributed to {:0.3f}% of imports.'.format(birds_importer.head(10).sum()*100))


# ##### A look at the top-3 importers
# * USA
# * UAE
# * Japan

# In[60]:


birds_US_import = birds[birds['Importer'] == 'US']
f, ax = plt.subplots(2,2,figsize=(15,16))
birds_US_import.Term.value_counts(normalize=True).head().plot('bar', ax=ax[0,0], rot=20, title='Term')
birds_US_import.Purpose.value_counts(normalize=True).head().plot('bar', ax=ax[1,1], rot=0, title='Purpose')
birds_US_import.Source.value_counts(normalize=True).head().plot('bar', ax=ax[1,0], rot=30, title='Source')
birds_US_import.Family.value_counts(normalize=True).head().plot('bar', ax=ax[0,1], rot=30, title='Family')


# In[61]:


birds_AE_import = birds[birds['Importer'] == 'AE']
f, ax = plt.subplots(2,2,figsize=(15,16))
birds_AE_import.Term.value_counts(normalize=True).head().plot('bar', ax=ax[0,0], rot=20, title='Term')
birds_AE_import.Purpose.value_counts(normalize=True).head().plot('bar', ax=ax[1,1], rot=0, title='Purpose')
birds_AE_import.Source.value_counts(normalize=True).head().plot('bar', ax=ax[1,0], rot=30, title='Source')
birds_AE_import.Family.value_counts(normalize=True).head().plot('bar', ax=ax[0,1], rot=30, title='Family')


# In[62]:


birds_JP_import = birds[birds['Importer'] == 'JP']
f, ax = plt.subplots(2,2,figsize=(15,16))
birds_JP_import.Term.value_counts(normalize=True).head().plot('bar', ax=ax[0,0], rot=20, title='Term')
birds_JP_import.Purpose.value_counts(normalize=True).head().plot('bar', ax=ax[1,1], rot=0, title='Purpose')
birds_JP_import.Source.value_counts(normalize=True).head().plot('bar', ax=ax[1,0], rot=30, title='Source')
birds_JP_import.Family.value_counts(normalize=True).head().plot('bar', ax=ax[0,1], rot=30, title='Family')


# * Except USA, the other two countries' imports comprised mostly of live birds (over 80%).
# * Commercial trade dominated into two countries other than USA, where personal trade was seen higher.
# * Over 50% of UAE imports were from the falcon family.
# * Japan and USA imports were dominated by the Parrot family.
# 
# *Falconidae is falcon family*

# **Exports**

# In[63]:


birds_exporter = birds.Exporter.value_counts(normalize=True)
plt.figure(figsize=(12,7))
birds_exporter.head(10).plot('bar')
plt.xticks(rotation=0)
plt.xlabel('Exporter')
plt.ylabel('Normalized Count')
plt.title('Top 10 Exporters')
print('Top 10 countries contributed to {:0.3f}% of exports.'.format(birds_exporter.head(10).sum()*100))


# ##### A look at the top-3 exporters
# * Belgium
# * Netherlands
# * UAE

# In[64]:


birds_US_export = birds[birds['Exporter'] == 'BE']
f, ax = plt.subplots(2,2,figsize=(15,16))
birds_US_export.Term.value_counts(normalize=True).head().plot('bar', ax=ax[0,0], rot=20, title='Term')
birds_US_export.Purpose.value_counts(normalize=True).head().plot('bar', ax=ax[1,1], rot=0, title='Purpose')
birds_US_export.Source.value_counts(normalize=True).head().plot('bar', ax=ax[1,0], rot=30, title='Source')
birds_US_export.Family.value_counts(normalize=True).head().plot('bar', ax=ax[0,1], rot=30, title='Family')


# In[65]:


birds_NL_export = birds[birds['Exporter'] == 'NL']
f, ax = plt.subplots(2,2,figsize=(15,16))
birds_NL_export.Term.value_counts(normalize=True).head().plot('bar', ax=ax[0,0], rot=20, title='Term')
birds_NL_export.Purpose.value_counts(normalize=True).head().plot('bar', ax=ax[1,1], rot=0, title='Purpose')
birds_NL_export.Source.value_counts(normalize=True).head().plot('bar', ax=ax[1,0], rot=30, title='Source')
birds_NL_export.Family.value_counts(normalize=True).head().plot('bar', ax=ax[0,1], rot=30, title='Family')


# In[66]:


birds_AE_export = birds[birds['Exporter'] == 'AE']
f, ax = plt.subplots(2,2,figsize=(15,16))
birds_AE_export.Term.value_counts(normalize=True).head().plot('bar', ax=ax[0,0], rot=20, title='Term')
birds_AE_export.Purpose.value_counts(normalize=True).head().plot('bar', ax=ax[1,1], rot=0, title='Purpose')
birds_AE_export.Source.value_counts(normalize=True).head().plot('bar', ax=ax[1,0], rot=30, title='Source')
birds_AE_export.Family.value_counts(normalize=True).head().plot('bar', ax=ax[0,1], rot=30, title='Family')


# * Most of the exports of the top-3 countries were live brids.
# * Commercial trade took dominance in Netherlands and Belgium.
# * Personal trade was highest in UAE.

# #### Countries that were active in both imports and exports

# In[67]:


exp = birds_exporter.head(10)
imp = birds_importer.head(10)
exp_imp = pd.concat([imp, exp], axis=1)
exp_imp = exp_imp.dropna()
exp_imp = exp_imp.reset_index()
plt.figure(figsize=(12,7))
plt.bar(exp_imp['index'], exp_imp.Importer, align='edge', width=0.2, label='imports')
plt.bar(exp_imp['index'], exp_imp.Exporter, align='edge', width=-0.2, label='exports')
plt.xticks(rotation=0)
plt.legend(loc='best')
plt.xlabel('Country')
plt.ylabel('Normalized Count')
plt.title('Countries with high values of imports and exports')


# UAE and USA had high levels of imports and exports.

# #### Trade by Taxon, Order, Family and Genus

# In[68]:


chk_lst = ['Taxon', 'Order', 'Family', 'Genus']
for name in chk_lst:
    print(name)
    print(birds[name].unique())


# All the four columns had a huge number of variables and only the top 10 variables in each column were visualized.

# ##### Trade by Taxon

# In[69]:


birds_taxon = birds.Taxon.value_counts(normalize=True)
plt.figure(figsize=(12,7))
birds_taxon.head(10).plot('bar')
plt.xticks(rotation=90)
plt.xlabel('Taxon')
plt.ylabel('Normalized Count')
plt.title('Trade by Taxon')
print('Top 10 values contributed to {:0.3f}% of trade.'.format(birds_taxon.head(10).sum()*100))


# Since, the top ten Taxons only contributed to 29% of the total trade and the top Taxon only having a share of ~5%, there wasn't any indication stating that one particular Taxon had a high trade value.
# 
# *Psittacus erithacus is a genus of African parrots.*

# ##### Trade by Order

# In[70]:


birds_order = birds.Order.value_counts(normalize=True)
plt.figure(figsize=(12,7))
birds_order.head(10).plot('bar')
plt.xticks(rotation=30)
plt.xlabel('Order')
plt.ylabel('Normalized Count')
plt.title('Trade by Order')
print('Top 10 values contributed to {:0.3f}% of trade.'.format(birds_order.head(10).sum()*100))


# The top ten brids by Order contributed to 97% of the total trade with 'Psittacoformes' contributing to over 60% of the trade.
# 
# *Psittacoformes is Parrot.*

# ##### Trade by Family

# In[71]:


birds_family = birds.Family.value_counts(normalize=True)
plt.figure(figsize=(12,7))
birds_family.head(10).plot('bar')
plt.xticks(rotation=30)
plt.xlabel('Family')
plt.ylabel('Normalized Count')
plt.title('Trade by Family')
print('Top 10 values contributed to {:0.3f}% of trade.'.format(birds_family.head(10).sum()*100))


# * The top ten brids by Family contributed to 92% of the total trade with 'Psittacidae' contributing to over 50% of the trade.
# * Looking at all the data, Psittacidae family was traded the most.
# 
# *Psittacidae is a family of Parrots.*
# *Falconidae is a family of Falcons.*

# **A closer look at the top-2 traded bird families.**

# In[72]:


birds_family = birds[birds['Family'] == 'Psittacidae']
f, ax = plt.subplots(1,2, figsize=(15,7))
birds_family.Taxon.value_counts(normalize=True).head().plot('bar', ax=ax[0,], rot=30, title='Taxon')
birds_family.Genus.value_counts(normalize=True).head().plot('bar', ax=ax[1,], rot=30, title='Genus')


# Amazon parrot was the most traded parrot followed by Ara macaws.

# In[73]:


birds_family = birds[birds['Family'] == 'Psittacidae']
f, ax = plt.subplots(2,2,figsize=(15,12))
birds_family.Importer.value_counts(normalize=True).head().plot('bar', ax=ax[0,0], rot=20, title='Importer')
birds_family.Purpose.value_counts(normalize=True).head().plot('bar', ax=ax[1,0], rot=0, title='Purpose')
birds_family.Exporter.value_counts(normalize=True).head().plot('bar', ax=ax[0,1], rot=30, title='Exporter')
birds_family.Source.value_counts(normalize=True).head().plot('bar', ax=ax[1,1], rot=30, title='Source')


# USA was the largest importer of parrots while Netherlands and Belgium were the biggest exporters.

# In[74]:


birds_family = birds[birds['Family'] == 'Falconidae']
f, ax = plt.subplots(1,2, figsize=(15,7))
birds_family.Taxon.value_counts(normalize=True).head().plot('bar', ax=ax[0,], rot=30, title='Taxon')
birds_family.Genus.value_counts(normalize=True).head().plot('bar', ax=ax[1,], rot=20, title='Genus')


# Falco genus dominated the Falcon bird trade with over 95% share.

# In[75]:


birds_family = birds[birds['Family'] == 'Falconidae']
f, ax = plt.subplots(2,2,figsize=(15,12))
birds_family.Importer.value_counts(normalize=True).head().plot('bar', ax=ax[0,0], rot=20, title='Importer')
birds_family.Purpose.value_counts(normalize=True).head().plot('bar', ax=ax[1,0], rot=0, title='Purpose')
birds_family.Exporter.value_counts(normalize=True).head().plot('bar', ax=ax[0,1], rot=30, title='Exporter')
birds_family.Source.value_counts(normalize=True).head().plot('bar', ax=ax[1,1], rot=30, title='Source')


# * UAE was the largest importer and exporter of falcons.
# 
# * Most of the falcons sold were for personal use with the source being captivity.

# ##### Trade by Genus

# In[76]:


birds_genus = birds.Genus.value_counts(normalize=True)
plt.figure(figsize=(12,7))
birds_genus.head(10).plot('bar')
plt.xticks(rotation=30)
plt.xlabel('Genus')
plt.ylabel('Normalized Count')
plt.title('Trade by Genus')
print('Top 10 values contributed to {:0.3f}% of trade.'.format(birds_genus.head(10).sum()*100))


# * The top ten brids by Genus contributed to 53% of the total trade with 'Falco' contributing to over 12% of the trade.

# Overall, the parrot family dominated the trade followed by the falcon family.

# ### Trade by Term

# In[77]:


plt.figure(figsize=(12,7))
birds.Term.value_counts(normalize=True).head(10).plot('bar')
plt.xticks(rotation=30)
plt.xlabel('Term,')
plt.ylabel('Normalized Count')
plt.title('Trade by Term')


# Most of the birds traded were live, contributing to over 80% of the trade.

# #### A look the three most traded terms in birds
# * live
# * specimens
# * feathers

# ##### How were live birds traded?

# In[79]:


birds_term = birds[birds['Term'] == 'live']
f, ax = plt.subplots(3,2,figsize=(15,25))
birds_term.Importer.value_counts(normalize=True).head().plot('bar', ax=ax[0,0], rot=20, title='Importer')
birds_term.Purpose.value_counts(normalize=True).head().plot('bar', ax=ax[2,0], rot=30, title='Purpose')
birds_term.Order.value_counts(normalize=True).head().plot('bar', ax=ax[1,0], rot=30, title='Order')
birds_term.Family.value_counts(normalize=True).head().plot('bar', ax=ax[1,1], rot=30, title='Family')
birds_term.Exporter.value_counts(normalize=True).head().plot('bar', ax=ax[0,1], rot=30, title='Exporter')
birds_term.Source.value_counts(normalize=True).head().plot('bar', ax=ax[2,1], rot=30, title='Source')


# * Most of the live brids exported were from the parrot family.
# * Commercial trade and personal trade occupied the lion share in purpose of trade.
# * As already observed, *'live'* trade term was mostly seen for birds that were in captivity.
# * Interestingly, USA exported and imported nearly the same anumber of live birds.

# #### Details of the feather trade

# In[82]:


birds_term = birds[birds['Term'] == 'feathers']
f, ax = plt.subplots(3,2,figsize=(15,25))
birds_term.Importer.value_counts(normalize=True).head().plot('bar', ax=ax[0,0], rot=20, title='Importer')
birds_term.Purpose.value_counts(normalize=True).head().plot('bar', ax=ax[2,0], rot=30, title='Purpose')
birds_term.Order.value_counts(normalize=True).head().plot('bar', ax=ax[1,0], rot=30, title='Order')
birds_term.Family.value_counts(normalize=True).head().plot('bar', ax=ax[1,1], rot=30, title='Family')
birds_term.Exporter.value_counts(normalize=True).head().plot('bar', ax=ax[0,1], rot=30, title='Exporter')
birds_term.Source.value_counts(normalize=True).head().plot('bar', ax=ax[2,1], rot=30, title='Source')


# * USA dominated the imports and exports of feathers.
# * The pheasant family followed by the parrot family led the feather trade.
# * Feathers were mostly traded for commercial purpose.
# * Intersetingly, confescated or seized specimens accounted almost 30% of feathers trade. It was closely followed by birds in the wild and in captivity.

# **Details of the specimens trade**

# In[83]:


birds_term = birds[birds['Term'] == 'specimens']
f, ax = plt.subplots(3,2,figsize=(15,25))
birds_term.Importer.value_counts(normalize=True).head().plot('bar', ax=ax[0,0], rot=20, title='Importer')
birds_term.Purpose.value_counts(normalize=True).head().plot('bar', ax=ax[2,0], rot=30, title='Purpose')
birds_term.Order.value_counts(normalize=True).head().plot('bar', ax=ax[1,0], rot=30, title='Order')
birds_term.Family.value_counts(normalize=True).head().plot('bar', ax=ax[1,1], rot=30, title='Family')
birds_term.Exporter.value_counts(normalize=True).head().plot('bar', ax=ax[0,1], rot=30, title='Exporter')
birds_term.Source.value_counts(normalize=True).head().plot('bar', ax=ax[2,1], rot=30, title='Source')


# * USA led the imports and ranked second in the exports of bird specimens.
# * Parrot family and falcon family led the charts in specimens trade as well.
# * As expected, around 80% of the specimens trade was for scientific purpose.
# * Over 50% of the birds traded as specimens were caught in the wild.
