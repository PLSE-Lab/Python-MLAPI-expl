#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from numpy.random import randn

from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


sns.set(color_codes=True)


# In[ ]:


data_lebron = pd.read_csv('../input/lebron_career.csv')


# In[ ]:


data_lebron.columns = map(str.upper, data_lebron.columns)
# change all columns names into upper case


# In[ ]:


data_lebron = data_lebron.drop(['GAME'], axis=1)
data_lebron['DATE'] = pd.to_datetime(data_lebron['DATE'])


# In[ ]:


data_lebron['MP'] = data_lebron['MP'].astype(str)
data_lebron['MP'] = data_lebron['MP'].str.replace(':','.')
data_lebron['MP'] = data_lebron['MP'].astype(float)
data_lebron['AGE'] = data_lebron['AGE'].astype(str)
data_lebron['AGE'] = data_lebron['AGE'].str.replace('-','.')
data_lebron['AGE'] = data_lebron['AGE'].astype(float)


# In[ ]:


data_lebron['RESULT'] = data_lebron['RESULT'].str[:1]


# In[ ]:


data_lebron.head()


# In[ ]:


data_lebron.describe()


# In[ ]:


fig, ax = plt.subplots(figsize=(16,7))

data_lebron_opp = data_lebron.pivot_table(columns='RESULT',index='OPP', values='PTS')
data_lebron_opp.plot(ax=ax, kind='bar')

ax.set_ylim(0, 40)
ax.set_title("Lebron Career Points Against Each Team", fontsize=14, )
ax.legend(loc='upper right', title='Game Result')

fig.autofmt_xdate()


# In[ ]:


fig, ax = plt.subplots(figsize=(6, 14))

sns.heatmap(data_lebron_opp, center=25, cmap="rainbow", vmin=15, vmax=35, annot=True, robust=True, linewidth=.1)


# In[ ]:


fig, ax = plt.subplots(figsize=(14, 6))
from scipy.stats import norm
#distribution normalization.
sns.distplot(data_lebron_opp.iloc[0], fit=norm, ax=ax)


# In[ ]:


fig, ax = plt.subplots(figsize=(14, 6))

sns.kdeplot(data_lebron.FGA, data_lebron.PTS, shade=True, cmap='Blues', cbar=True)


# In[ ]:


import statsmodels.api as sm
#simple linear regression
X = data_lebron['AGE']
y = data_lebron['PTS']

model = sm.OLS(y, X).fit()
predictions = model.predict(X)

model.summary()


# In[ ]:


sns.lmplot('AGE', 'PTS', data=data_lebron, order=2,
          line_kws={'color':'indianred', 'linewidth':1.3})


# In[ ]:


fig, ax = plt.subplots(figsize=(14, 6))

sns.kdeplot(data_lebron.THREE, shade=True, color='indianred', cut=0)


# In[ ]:


sns.lmplot('AGE', 'FTA', data=data_lebron, order=4,
          line_kws={'color':'indianred', 'linewidth':1.3})


# In[ ]:


data_jordan = pd.read_csv('../input/jordan_career.csv')


# In[ ]:


data_jordan = data_jordan.fillna(0)


# In[ ]:


data_jordan.columns = map(str.upper, data_jordan.columns)
data_jordan = data_jordan.drop(['GAME'], axis=1)
data_jordan['DATE'] = pd.to_datetime(data_jordan['DATE'])
data_jordan['MP'] = data_jordan['MP'].astype(str)
data_jordan['MP'] = data_jordan['MP'].str.replace(':','.')
data_jordan['MP'] = data_jordan['MP'].astype(float)
data_jordan['AGE'] = data_jordan['AGE'].astype(str)
data_jordan['AGE'] = data_jordan['AGE'].str.replace('-','.')
data_jordan['AGE'] = data_jordan['AGE'].astype(float)


# In[ ]:


data_jordan['RESULT'] = data_jordan['RESULT'].str[:1]


# In[ ]:


data_jordan.describe()


# In[ ]:


ser1 = data_lebron.mean()
ser2 = data_jordan.mean()

lebron_mean = pd.DataFrame(ser1).transpose()
jordan_mean = pd.DataFrame(ser2).transpose()

dataset = pd.DataFrame(pd.concat([lebron_mean, jordan_mean], ignore_index=True))


# In[ ]:


dataset = dataset.rename(index={0:'Lebron', 1:'Jordan'})


# In[ ]:


fig, ax = plt.subplots(figsize=(16,6))

dataset.transpose().plot(ax=ax, kind='bar', colormap='rainbow')

ax.set_title("Jordan&Lebron Career Average Comparasion", fontsize=14, color='indianred')
ax.set_xlabel("Technical Categories", fontsize=12)
ax.set_ylabel("Points", fontsize=12)

ax.set_ylim(0, 45)
ax.legend(fontsize=12, title='Player Name')
ax.tick_params('y', direction='out', left=True, right=True, labelright=True, labelleft=True)

fig.autofmt_xdate()

#for i in ax.patches:
    #ax.text(i.get_x()-.025, i.get_height()+1, str(round(i.get_height(),2)))


# In[ ]:


fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, polar=True)

labels = np.array(['MP', 'FG', 'FGP', 'FTA', 'TRB', 'AST', 'STL', 'PTS'])
stats1 = dataset.loc['Lebron', labels].values
stats2 = dataset.loc['Jordan', labels].values

angles=np.linspace(0, 2*np.pi, len(labels)+1, endpoint=False)
stats1=np.concatenate((stats1,[stats1[0]]))
stats2=np.concatenate((stats2,[stats2[0]]))

ax.plot(angles, stats1, 'o-', linewidth=1.3)
ax.fill(angles, stats1, alpha=0.25)
ax.set_thetagrids(angles * 180/np.pi, labels)

ax.plot(angles, stats2, 'o-', linewidth=1.3)
ax.fill(angles, stats2, alpha=0.25)
ax.set_thetagrids(angles * 180/np.pi, labels)

ax.set_title("Jordan&Lebron Career Average Overview")
ax.grid(True)
# set the legend and the location.
label_texts=('Jordan','Lebron')
ax.legend(label_texts, loc='upper right', bbox_to_anchor=(0.03 , 0.01))


# In[ ]:


sns.lmplot('AGE', 'PTS', data=data_jordan, order=4,
          line_kws={'color':'indianred'})


# In[ ]:


fig, ax = plt.subplots(figsize=(14, 6))

sns.kdeplot(data_jordan.PTS, shade=True, color='indianred', label='Jordan')
sns.kdeplot(data_lebron.PTS, shade=True, label='Lebron')


# In[ ]:


data_jordan_opp = pd.pivot_table(data_jordan, index='OPP', columns='RESULT', values='PTS')


# In[ ]:


fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(5, 14))
fig.subplots_adjust(wspace=0.5)

ax1.set_title("Jordan Career Average.")
sns.heatmap(data_jordan_opp, ax=ax1, annot=True, cmap="rainbow",
            center=25, linewidths=.1, cbar=False, robust=True)
#fig.colorbar(ax1.collections[0], ax=ax1,location="left", use_gridspec=False, pad=0.01)


sns.heatmap(data_lebron_opp, ax=ax2, center=25, cmap="rainbow", vmin=15, vmax=35, annot=True, 
            cbar=True, linewidth=.1)
ax2.set_title("Lebron Career Average.")
#fig.colorbar(ax2.collections[0], ax=ax2,location="right", use_gridspec=False, pad=0.01)


# In[ ]:


df_lebron = data_lebron.set_index('DATE')
df_jordan = data_jordan.set_index('DATE')


# In[ ]:


lebron_month = df_lebron.groupby(pd.Grouper(freq='M')).mean()
jordan_month = df_jordan.groupby(pd.Grouper(freq='M')).mean()


# In[ ]:


lebron_month['PTS'].min()


# In[ ]:


fig, ax = plt.subplots(figsize=(14,5))

lebron_month['PTS'].dropna().plot(ax=ax)
jordan_month['PTS'].dropna().plot(ax=ax)


# In[ ]:


fig, ax = plt.subplots(figsize=(14,5))

df_lebron['PTS'].dropna().groupby(pd.Grouper(freq='Y')).sum().plot(ax=ax)
df_jordan['PTS'].dropna().groupby(pd.Grouper(freq='Y')).sum().plot(ax=ax)


# In[ ]:


fig, ax = plt.subplots(figsize=(14,5))

df_lebron['AST'].dropna().groupby(pd.Grouper(freq='Y')).sum().plot(ax=ax)
df_jordan['AST'].dropna().groupby(pd.Grouper(freq='Y')).sum().plot(ax=ax)


# In[ ]:


fig, ax = plt.subplots(figsize=(14,5))

df_lebron['STL'].dropna().groupby(pd.Grouper(freq='Y')).sum().plot(ax=ax)
df_jordan['STL'].dropna().groupby(pd.Grouper(freq='Y')).sum().plot(ax=ax)


# In[ ]:


fig, ax = plt.subplots(figsize=(14,5))

df_lebron['TRB'].dropna().groupby(pd.Grouper(freq='Y')).sum().plot(ax=ax)
df_jordan['TRB'].dropna().groupby(pd.Grouper(freq='Y')).sum().plot(ax=ax)


# In[ ]:


fig, ax = plt.subplots(figsize=(14,5))

df_lebron['THREE'].dropna().groupby(pd.Grouper(freq='Y')).sum().plot(ax=ax)
df_jordan['THREE'].dropna().groupby(pd.Grouper(freq='Y')).sum().plot(ax=ax)

plt.title('Jordan vs Lebron Three Points Made per Year')
plt.xlabel('Year')
plt.ylabel('Three Points Made')

#ax.xaxis.set_major_locator(dates.YearLocator())

