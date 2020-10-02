#!/usr/bin/env python
# coding: utf-8

# ### Simple EDA using Wikipedia tables that was stored as CSV

# In[ ]:


# First thing first we have to import some libraries for data manipulation and visualization
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from PIL import Image
from pylab import rcParams
rcParams['axes.xmargin'] = 0
rcParams['axes.ymargin'] = 0
plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')


# [Demographics of Israel From Wikipedia link](https://en.wikipedia.org/wiki/Demographics_of_Israel)

# In[ ]:


# I have Scraped 5 tables data from Wikipedia article
df_religious = pd.read_csv("../input/demographics-of-israe/df_religious.csv")
df_births = pd.read_csv("../input/demographics-of-israe/df_births.csv")
df_bd = pd.read_csv("../input/demographics-of-israe/df_bd.csv")
df_structure = pd.read_csv("../input/demographics-of-israe/df_structure.csv")
df_fertility = pd.read_csv("../input/demographics-of-israe/df_fertilty.csv")

df_religious


# In[ ]:


# A couple of thing to notice is that we have to rearrange the columns headers and change the data types,
## set the Group column as an index, sort the values and add 'Total' row. Lets do that.
df_religious.info()


# In[ ]:


# Population and Religious (df_religious)
df_religious = pd.DataFrame(df_religious.values[1:], columns=df_religious.iloc[0])
df_religious.set_index('Group', inplace=True)
df_religious['Population'] = df_religious['Population'].astype(int)
df_religious['%'] = df_religious['%'].astype(str)
df_religious['%'] = df_religious['%'].str.replace("%","").astype('float')
df_religious[:-1].sort_values(by='Population', ascending=False)
df_religious.loc['Total',:] = df_religious.sum(axis=0).round()
del df_religious.columns.name
df_religious


# In[ ]:


# Births, in absolute numbers, by mother's religion (df_births)
df_births = pd.DataFrame(df_births.values[1:], columns=df_births.iloc[0])
df_births[['Year','Jewish','Muslim','Christian','Druze','Others','Total']] = df_births[['Year','Jewish','Muslim','Christian','Druze','Others','Total']].apply(pd.to_numeric)
df_births['% Muslim'] = df_births['% Muslim'].astype(str)
df_births['% Muslim'] = df_births['% Muslim'].str.replace("%","").astype('float')
df_births['% Jewish'] = df_births['% Jewish'].astype(str)
df_births['% Jewish'] = df_births['% Jewish'].str.replace("%","").astype('float')
df_births['Year'] = pd.to_datetime(df_births['Year'],format='%Y')
df_births.set_index('Year', inplace=True)
df_births.index = df_births.index.year
del df_births.columns.name
df_births.head()


# In[ ]:


# Births and deaths (df_bd)
df_bd = pd.DataFrame(df_bd.values[1:], columns=df_bd.iloc[0])
for i in df_bd.columns:
    df_bd[i] = df_bd[i].str.replace(" ","").apply(pd.to_numeric) # replacing spaces and changing data type to numeric
df_bd['Year'] = pd.to_datetime(df_bd['Year'],format='%Y')
df_bd.set_index('Year', inplace=True)
df_bd.index = df_bd.index.year
del df_bd.columns.name
df_bd.rename(columns={'Population (x1000)':'Population'}, inplace=True)
df_bd['Population'] = df_bd['Population'] * 1000 
df_bd['TFR'] = df_bd['TFR'].ffill().bfill() # filling missing values forward and backward fill
df_bd.head()


# In[ ]:


df_structure


# In[ ]:


# Structure of the population (01.07.2012)  (df_structure)
# In this table, as one can see from the code above I had to perform manipulation on all the columns to delete some spaces between the numbers
# and then to transform the data-type to numeric, a well to set the Age Group as an index with a categorical type.

df_structure = pd.DataFrame(df_structure.values[1:], columns=df_structure.iloc[0])
df_structure['Male'] = df_structure.Male.str.replace(" ","").apply(pd.to_numeric)
df_structure['Female'] = df_structure.Female.str.replace(" ","").apply(pd.to_numeric)
df_structure['Total'] = df_structure.Total.str.replace(" ","").apply(pd.to_numeric)
df_structure['%'] = (df_structure['%'].str.replace(" ","").apply(pd.to_numeric))/100
df_structure = df_structure.iloc[1:]
df_structure['Age Group'] = df_structure['Age Group'].astype('category')
df_structure.set_index('Age Group', inplace=True)
del df_structure.columns.name
df_structure.head()


# In[ ]:


# Total fertility rate since 2000 (df_fertilty)
df_fertility = pd.DataFrame(df_fertility.values[1:], columns=df_fertility.iloc[0])
df_fertility = df_fertility.apply(pd.to_numeric)
df_fertility.set_index('Year', inplace=True)
del df_fertility.columns.name
df_fertility.head()


# In[ ]:


# Generate descriptive statistics that summarize the central tendency, dispersion and shape of a dataset.
df_bd.describe().T.round()


# In[ ]:


#Transform Yearly Population data to yearly precent_change
pct = df_bd.Population.pct_change()*100
pct.tail()


# In[ ]:


pct.plot(kind='bar',figsize=(18,8), title=f'Population Percent Change. Yearly Average {np.round(pct.mean(),2)}%',
 color=['g' if x > pct.mean() else 'r' for x in pct]);
# Green Bar = above percent change average, Red Bar = below percent change average 


# In[ ]:


# Next, let's produce analysis of Israeli population by age & gender:

fig, ax = plt.subplots(1,2)
df_structure.iloc[:,:2].plot.barh(stacked=True, figsize=(14,6), width = 0.9,
title='Population by Age(range) & Sex (2012)', ax=ax[0]);
ax[0].get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:.0f}".format(int(x)/1000)+'k'))                        
df_structure[['Male','Female']].sum().plot(kind='bar', ylim=(0,4500000), rot=True, figsize=(14,6), ax=ax[1]);
for i, v in enumerate(df_structure[['Male','Female']].sum()):
    plt.text(i-0.15,v+100000, '{:,}'.format(v));
    plt.text(i-0.15, v-1000000,str((v/df_structure[["Total"]].sum()[0]).round(3)*100)+"%", fontsize=20)
plt.title("Israel Population by gender (2012)")
plt.xlabel("Gender");


# In[ ]:


# And by religion:
plt.style.use('fivethirtyeight')
fig, ax = plt.subplots(1,2)
df_religious.iloc[:-1,:-1].T.plot.bar(figsize=(12,6),stacked=True, ax=ax[0], legend=False);
ax[0].get_yaxis().set_major_formatter(plt.FuncFormatter(lambda y, p: format(int(y), ',')))
ax[0].set_xticks([])
ax[0].set_title("Total Population",fontweight="bold");
plt.pie(df_religious['%'][:-1],labels=df_religious.index[:-1],autopct='%1.1f%%', shadow=True,
        explode = (0.05, 0.05, 0.05, 0.05,0.05), startangle=50);
plt.title("%Population by Religious (2017)",fontweight="bold");


# In[ ]:


# visualize the absolute numbers of live births vs deaths for the overall population:
plt.style.use('seaborn-white')
fig, ax = plt.subplots(1, 1, figsize=(14,8))
ax.fill_between(df_bd.index, y1=df_bd['Live births'].values, y2=0, alpha=0.5, label=df_bd.columns[1], linewidth=2);
ax.fill_between(df_bd.index, y1=df_bd['Deaths'].values, y2=0, alpha=0.5,label = df_bd.columns[2], linewidth=2);
ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda y, p: format(int(y), ',')))
ax.set_title('Live births Vs Deaths (absolute numbers)', fontsize=18)
ax.set_xlabel('Year')
ax.legend(loc=2, fontsize=20);
for y in range(0, 170000, 10000):    
    plt.hlines(y, xmin=df_bd.index.min(), xmax=df_bd.index.max(), colors='black', alpha=0.7, linestyles="--", lw=0.5)


# In[ ]:


df_births[['Jewish','Muslim']].plot(kind='bar', figsize=(14,7), ylim=(0,150000),
                                    title="Births, in absolute numbers by religion");


# In[ ]:


#  Normalized growth rate science 1996 (*not including Christian,Druze,Others as relatively insignificant)
ax = plt.figure()
ax = (df_births.iloc[:,[0,1,5]]/df_births.iloc[0,[0,1,5]]-1).plot(figsize=(12,8),title="Births, growth rate (since 1996)")
vals = ax.get_yticks()
ax.set_yticklabels(['{:,.1%}'.format(x) for x in vals]);


# In[ ]:


df_fertility[['Jews', 'Muslims', 'Total']].plot(figsize=(12,8), title=("Fertility Rate by religion"));


# In[ ]:


# Now, for the final chart, let's perform a simple population projection, by taking the last 10 years average growth rate of the overall population and compound it forward:
df_bdp = pd.DataFrame({'Population_est':range(70)}, index=(range(df_bd.index[-1],df_bd.index[-1]+70)))
df_bdp.loc[2018,'Population_est'] = df_bd.iloc[-1,0]
for i in range(1,len(df_bdp)):
    df_bdp.iloc[i,0] = (df_bdp.iloc[i-1,0] * (1+pct[-10:].mean()/100)).round(0)


# In[ ]:


plt.style.use('seaborn-white')
fig, ax = plt.subplots(figsize=(14,8))
im = np.array(Image.open('../input/picture/pf3.jpg'))
x1, x2 = df_bd.index.min(), df_bdp.index.max()-3
y1, y2 = df_bd.Population.min(), df_bdp.Population_est.max() *1.1
ax.imshow(im, extent=[x1, x2, y1, y2], aspect='auto')
ax.plot(df_bd.index, df_bd.Population, lw=8)
ax.plot(df_bdp.index,df_bdp.Population_est,lw=8, color='r',alpha=0.8)
y_formatter = ticker.FuncFormatter(lambda x, loc: "{:,.1f}M".format(x*1e-6))
ax.yaxis.set_major_formatter(y_formatter);
ax.set_title("Israel Population (past & future estimation)",fontweight="bold");
ax.set_xlabel("Year",fontweight="bold",);

