#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import warnings
warnings.filterwarnings("ignore")

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns 
import squarify
import plotly.graph_objs as go

from sklearn import preprocessing
from plotly.offline import init_notebook_mode, iplot
get_ipython().run_line_magic('pylab', 'inline')
import plotly.offline as py


from collections import Counter

import os
print(os.listdir("../input"))

py.init_notebook_mode(connected=True)
get_ipython().run_line_magic('matplotlib', 'inline')

# import multiple csvs
from glob import glob

# set decimal format to not extend beyond 8
pd.options.display.float_format = '{:.2f}'.format

gdf = pd.read_csv("../input/google-job-skills/job_skills.csv")
ddf = pd.read_csv("../input/us-technology-jobs-on-dicecom/dice_com-job_us_sample.csv")
kdf = pd.read_csv("../input/../input/kaggle-survey-2017/multipleChoiceResponses.csv", encoding="ISO-8859-1", low_memory=False)
# a way to import all into one df right away with option to ignore index
'''df = pd.concat([pd.read_csv(f) for f in glob.glob('*.csv')], ignore_index = False)'''


# In[ ]:


gdfs = gdf.shape
ddfs = ddf.shape
kdfs = kdf.shape

print("\tShape of Google Data: " + str(gdfs) + "\n \n\t\t   *****" + "\n" + "\n\tShape of Dice Data: " + str(ddfs) + "\n \n\t\t   *****" + "\n" + "\n\tShape of Kaggle: " + str(kdfs))


# In[ ]:


kdf.head()


# In[ ]:



title = kdf["CurrentJobTitleSelect"]

time = kdf.LearningDataScienceTime
lps = kdf['LearningPlatformSelect'].str.split(",", expand=True)
lps = lps.values.flatten()
lps = pd.Series(lps)
lps = lps[lps.notnull()].to_frame()
lps.columns=["LearningMethod"]
timejob = pd.concat([title, lps], axis=1)
timejob.dropna()


# In[ ]:


from plotly.tools import FigureFactory as FF
from plotly.offline import plot, download_plotlyjs, init_notebook_mode, iplot
from plotly.graph_objs import *
init_notebook_mode()


# In[ ]:


kdf['LearningDataScienceTime'].value_counts().plot.bar()


# ## Let's Look at What Groups of Employed People Use What to Learn

# ## Random Color Heatmap Generator
# First, we have to create a list of lists which represent all available color schemes for heatmaps.

# In[ ]:


cmap = plt.cm.get_cmap()
colors = cmap(np.arange(cmap.N))
colors = colors.tolist()
l = [x for x in range (0, len(colors))]
ch = int(random.choice(l))
colors[ch]


# ## How to Use It
# Inset your dataframe and columns to compare in the format (df, val1, val2). A heatmap will appear showing the percent of each of value 1 chose value 2.
# 
# For example, see the code and result right below. 

# In[ ]:


def hm(df1, val1, val2):
    cmap = plt.cm.get_cmap()
    colors = cmap(np.arange(cmap.N))
    colors = colors.tolist()
    l = [x for x in range (0, len(colors))]
    ch = int(random.choice(l))
    
    relative_counts = df1.groupby([df1[val1], kdf[val2]]).size().groupby(level=0).apply(lambda x:
                                                                        100*x / float(x.sum())).reset_index(name="percentage")
    str1, str2 = str(val1), str(val2)
    relcounts = relative_counts.pivot(str1, str2, "percentage")
    fig,ax = plt.subplots(figsize=(25,15))
    ax = sns.heatmap(relcounts, linewidth=.5, cmap=colors[ch], annot=True, fmt="f")


# ### Respondents Job Titles and Job Resources
# Resource respondents which various current job titles used to get their current role.

# In[ ]:


hm(kdf, "CurrentJobTitleSelect", "JobSearchResource")


# ## Done Manually
# 

# In[ ]:



relative_counts = timejob.groupby([timejob.CurrentJobTitleSelect, timejob.LearningMethod]).size().groupby(level=0).apply(lambda x:                            100 * x / float(x.sum())).reset_index(name="percentage")
relcounts = relative_counts.pivot("CurrentJobTitleSelect", "LearningMethod", "percentage")
fig, ax = plt.subplots(figsize=(25,15))
ax = sns.heatmap(relcounts, linewidth=.5, cmap="Greens", annot=True, fmt="f")


# In[ ]:


hm(timejob, "CurrentJobTitleSelect", "LearningMethod")


# In[ ]:


ax = sns.countplot(x = lps['LearningMethod'], data = lps, order=lps['LearningMethod'].value_counts(ascending=True).index)
ax.set(ylabel='\nCount')
ax.set_axisbelow(True)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=12,rotation = 40, ha = "right")
ax.grid(True)
sns.set_style("dark")
sns.set_context("notebook", rc={"font.size":12,"axes.titlesize":20,"axes.labelsize":12})   
plt.title("Learning Method Recommendation\n")
plt.tight_layout
plt.figure(figsize=(20,10))

plt.show()


# In[ ]:


relative_counts = kdf.groupby([kdf.EmploymentStatus, kdf.JobSkillImportanceDegree]).size().groupby(level=0).apply(lambda x:
                                                 100 * x / float(x.sum())).reset_index(name="percentage")
relcounts = relative_counts.pivot("EmploymentStatus", "JobSkillImportanceDegree", "percentage")
fig, ax = plt.subplots(figsize=(8,8))
ax = sns.heatmap(relcounts, linewidth=.5, cmap="Blues")


# In[ ]:


kdf['EmploymentStatus'].value_counts()


# In[ ]:





# Many more observations came from Dice data, which makes since because it covers jobs which are from a range of businesses compared to Google; Dice data also includes more features.
# 
# Kaggle includes an extraordinary amount of observations (that is, 16,716 people surveyed) as well as features (questions asked). This is logical but overwhelming. First, I'd like to pay attention to Google and Dice.
# 
# Let's identify the columns may be different names for the same thing in the Dice/Google dfs as well as what kind features were reporting in Kaggle's 2017 survey.

# In[ ]:


print('The total number of respondents:',kdf.shape[0])
print('Total number of Countries with respondents:',kdf['Country'].nunique())
print('Country with highest respondents:',kdf['Country'].value_counts().index[0],'with',kdf['Country'].value_counts().values[0],'respondents')
print('Youngest respondent:',kdf['Age'].min(),' and Oldest respondent:', kdf['Age'].max())
print('{} instances seem too old (>70 years old)'.format(len(kdf[kdf['Age']>65])))
print('{} instances seem too old (<18 years old)'.format(len(kdf[kdf['Age']<18])))


# Clean up salaries - remove all nulls, replace commas, use tilde to get the data which doesn't contain hyphens, etc. Convert all to float

# In[ ]:


kdf['GenderSelect'].value_counts().plot.bar()


# ## What Language Should we Learn?

# In[ ]:


plt.figure(figsize(12,7))
ax = sns.countplot(x = kdf['LanguageRecommendationSelect'], data = kdf)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()


# In[ ]:


kcomp = kdf[kdf['CompensationAmount'].notnull()]
kcomp['CompAmountClean'] = kcomp['CompensationAmount'].str.replace(',','')

kcomp = kcomp.loc[~kcomp['CompAmountClean'].str.contains('-')]
kcomp['CompAmountClean'] = kcomp['CompAmountClean'].astype(float)

rates = pd.read_csv('../input/kaggle-survey-2017/conversionRates.csv')
kcomp_merged = pd.merge(left=kcomp, right=rates, left_on='CompensationCurrency', right_on='originCountry')
kcomp_merged['CompensationAmountUSD'] = kcomp_merged['CompAmountClean'] * kcomp_merged['exchangeRate']

kdfcomp = kcomp_merged[['CompensationAmountUSD', 'GenderSelect','MajorSelect','CurrentJobTitleSelect','FormalEducation','Country','Age','LanguageRecommendationSelect','LearningPlatformSelect', 'EmploymentStatus']]
kdfcomp.head(10)


# In[ ]:


printfriendlycomp = kdfcomp['CompensationAmountUSD'].astype(int)
print('Max Salary in USD $', printfriendlycomp.max())
print('Min Salary in USD $', printfriendlycomp.min())
print('Median Salary in USD $', printfriendlycomp.median())


# That huge salary discrepency has to be delt with. The upper end is much too large to be realistic. 

# In[ ]:


plt.subplots(figsize=(15,8))
# get only the values less than a million dollars
salary=kdfcomp[kdfcomp['CompensationAmountUSD'] < 1000000]
sns.distplot(salary['CompensationAmountUSD'])
plt.title('Salary Distribution\n', size = 20)
plt.show()


# Find unique FormalEducation values.

# In[ ]:


edset = set(kdfcomp['FormalEducation'])
edset


# ## Employment

# In[ ]:


kdf['EmploymentStatus'].value_counts().plot.bar()


# ## Let's see it as a percent of all respondents

# In[ ]:


(kdf['EmploymentStatus'].value_counts()/len(kdf['EmploymentStatus'])).plot.bar()


# In[ ]:


status = kdfcomp['EmploymentStatus'].value_counts()
sns.barplot(y = status.index, x = status.values, alpha = 0.6)
plt.show()


# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(21, 18)
ax = sns.boxplot(x=salary["GenderSelect"],y=salary["CompensationAmountUSD"], hue=kdfcomp['FormalEducation'], data=salary, palette="Set3")
ax.legend(loc='middle', bbox_to_anchor=(.4, .5))


# In[ ]:


plt.subplots(figsize=(10,8))
sns.boxplot(y=salary['GenderSelect'], x = salary['CompensationAmountUSD'], data = salary)
plt.ylabel(' ')
plt.show()


#   ## Age
# 

# In[ ]:


plt.subplots(figsize=(15,8))
salary['Age'].hist(bins=50, edgecolor='black')
plt.xticks(list(range(0,80,5)))
plt.title('Age Range')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:


kdfmajorlist = [x for x in kdfcomp['MajorSelect']]
majorlist = {}
for x in kdfmajorlist:
    if x in majorlist:
        majorlist[x] +=1
    else:
        majorlist[x] = 1
majorlist


# In[ ]:


groups = df_clean.groupby(['MajorSelect'])

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111)

sorted_medians = groups['CompensationAmountUSD'].median().sort_values()
ax = sorted_medians.plot(kind='barh', color=(117/255., 148/255., 205/255.))
# xaxis and yaxis conf
ax.xaxis.tick_top()
ax.set_xlabel('USD', fontsize=14)
ax.xaxis.set_label_position('top')
ax.yaxis.label.set_visible(False)
# configure ticks
plt.tick_params(
        axis='both',  # changes apply to the x-axis and y-axis
        which='both',  


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




groups = kdf.groupby(['MajorSelect'])

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111)

sorted_medians = groups['CompensationAmount'].median().sort_values()
ax = sorted_medians.plot(kind='barh', color=(117/255., 148/255., 205/255.))
# xaxis and yaxis conf
ax.xaxis.tick_top()
ax.set_xlabel('USD', fontsize=14)
ax.xaxis.set_label_position('top')
ax.yaxis.label.set_visible(False)
# configure ticks
plt.tick_params(
        axis='both',  # changes apply to the x-axis and y-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='on',  # ticks along the top edge are off
        right='off',
        left='off'
    )
# configure xticks & yticks
plt.yticks(fontsize=14, alpha=0.8)
plt.xticks(fontsize=14, alpha=0.8)
# remove border figure
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
plt.xticks(alpha=0.8)
plt.yticks(alpha=0.8)
plt.title('Median Compensation in USD by Undergraduate Major', fontsize=16, alpha=0.8, y=1.13, x=0.2)
# source
plt.text(60000, 0.01,
             'Source: Kaggle ML and Data Science Survey, 2017',
             fontsize=11,
             style='italic',
             alpha=0.7)
plt.tight_layout()
plt.show()


# Youngest and oldest respondent seems fake. Should investigate this.

# In[ ]:


kdfcols = [x for x in kdf.columns]
print("GDF Cols\n" + str(list(gdf.columns)) + "\n\nDDF Cols\n" + str(list(ddf.columns)) + "\n\nKDF Cols\n" + str(kdfcols))


# It looks like they are some obvious ones which can be adjusted: jobtitle, company, joblocation_address, etc. 
# 
# There are also some that stand out as useless for our sake: shift, jobid, uniq_id for sure and others which deserve better names for our own recognition. Let's see what some of them are.

# In[ ]:


resp_coun=kdf['Country'].value_counts()[:15].to_frame()
sns.barplot(resp_coun['Country'],resp_coun.index,palette='inferno')
plt.title('Top 15 Countries by number of respondents')
plt.xlabel('')
fig=plt.gcf()
fig.set_size_inches(10,10)
plt.show()
tree=kdf['Country'].value_counts().to_frame()
squarify.plot(sizes=tree['Country'].values,label=tree.index,color=sns.color_palette('RdYlGn_r',52))
plt.rcParams.update({'font.size':20})
fig=plt.gcf()
fig.set_size_inches(40,15)
plt.show()


# In[ ]:


age = kdf[(kdf['Age']>=16) & (kdf['Age']<= 70)]
plt.figure(figsize=(10,8))
sns.boxplot(y = age['Age'], data = age)
plt.title("Age Box Plot\n", fontsize=16)
plt.ylabel("Age\n", fontsize = 16)
plt.show()


# In[ ]:


plt.subplots(figsize=(22,12))
sns.countplot(y=kdf['GenderSelect'],order=kdf['GenderSelect'].value_counts().index)
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
countries = kdf['Country'].value_counts().head(30)
sns.barplot(y = countries.index, x =countries.values, alpha = 0.6)
plt.title("Country Distribution of the survey participants\n", fontsize = 16)
plt.xlabel("Number of Participants\n", fontsize = 16)
plt.ylabel("Country", fontsize = 16)
plt.show()


# In[ ]:


edu = kdf['FormalEducation'].value_counts()
labels = (np.array(edu.index))

values = (np.array((edu / edu.sum())*100))

trace = go.Pie(labels=labels, values=values,
              hoverinfo='label+percent',
               textfont=dict(size=20),
                showlegend=False)

layout = go.Layout(
    title='Formal Education of the survey participants'
)

data_trace = [trace]
fig = go.Figure(data=data_trace, layout=layout)
py.iplot(fig, filename="Formal_Education")


# In[ ]:


plt.style.use('fivethirtyeight')
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

plot = kdf[kdf.GenderSelect.isnull() == False].groupby(kdf.GenderSelect).GenderSelect.count().plot.bar()
plot = plt.title("Number of Respondents by Gender")


# In[ ]:


df0 = {}
use_features = [x for x in data_response.columns if x.find('LearningPlatformUsefulness') != -1]


# In[ ]:


filtered_kdf = kdf[(kdf.GenderSelect.isnull() == False) & (kdf.Country.isnull() == False)]

def getFemaleMaleRatio(df):
    counts_by_gender = kdf.groupby('GenderSelect').GenderSelect.count()
    return counts_by_gender[0]/counts_by_gender[1]

group_by_country = filtered_kdf.groupby(kdf.Country)
ratios = group_by_country.apply(getFemaleMaleRatio)
print("Maximum Female/Male Ratio: ", ratios.idxmax(), ratios.max())
print("Minimum Female/Male Ratio: ", ratios.idxmin(), ratios.min())

fig, ax = plt.subplots()
kdf[kdf.GenderSelect == 'Male'].Age.plot.hist(bins=100, ax=ax, alpha=0.5)
kdf[kdf.GenderSelect == 'Female'].Age.plot.hist(bins=100, ax=ax, alpha=0.8)
legend = ax.legend(['Male', 'Female'])
plot = plt.title("Age distribution for Male and Female Data Scientists")


# In[ ]:


import random
gdf['Responsibilities'][random.randint(0, 1250)]


# Get an idea by getting a random observation. I think this is helpful to get a feel for what the cell contains, especially in considering where we might get something comparable to relevant skills from.

# In[ ]:



# function to allow you to enter df and col name to retrieve a random full length sample of a observation
# useful to see what the df looks like 
def dobs(df, cols):
    import random
    if cols is str:
        countcol = df[cols].count()
        randobs = random.randint(0,countcol)
        obs = df[cols][randobs]
    # allow someone to enter 
    else:
        countcol = df.iloc[:, cols].count()
        randobs = random.randint(0,countcol)
        obs = df.iloc[randobs, cols]
    print(df.columns[cols] + "\n" + obs)


# In[ ]:


gdf.iloc[:, 2].count()


# In[ ]:


dobs(gdf, 4)


# In[ ]:


gdf['Minimum Qualifications'].count()


# In[ ]:


# make df and gdf columns match
ddf = ddf.rename(columns={'jobtitle':'Title', 'company':'Company', 'joblocation_address':'Location'})


# find out features we can't combine right away
missing_features = []
for col in gdf.columns:
    if col not in ddf.columns:
        missing_features.append(col)
missing_features

# find out features which exist in both
gsc = gdf[gdf.columns.intersection(ddf.columns)]
dsc = ddf[ddf.columns.intersection(gdf.columns)]


# In[ ]:


dobs(dsc, )


# In[ ]:


# jobs in general
ddf['Title'].count()
gdf['Title'].count()


# In[ ]:



gdj = gdf[gdf['Title'].str.contains('Data') | gdf['Title'].str.contains('data')]
gdj.head()


# In[ ]:


gdj['Minimum Qualifications'].iloc[0]


# In[ ]:


import re
# get all which start with experience
skllcon= gdf['Minimum Qualifications'].str.contains('(?:^|\W)experience\s(.*)', case=False, regex=True)
gdf['skills1']
skllcon


# In[ ]:


gdf['skills']


# In[ ]:


import re


# In[ ]:


match = re.search(r'experience\s(.*)', gdj['Minimum Qualifications'].iloc[21])
print(match)


# In[ ]:


ge = gdf['skills'].str.split(',')
ge


# In[ ]:





# In[ ]:


# skills expanded, parsed by comma
se = df['skills'].str.split(',', expand=True)
ge = gdf['skills'].str.split(',', expand=True)

# generate col names
num = 1
new_names = []

for col in se.columns:
    col = "Skill " + str(num)
    new_names.append(col)
    num += 1
# set col names
se.columns = new_names

# turn into a new df with expanded columns
newdf = pd.concat([df,se], axis=1)
newdf


# ### Just the data analysis jobs

# In[ ]:


# grab any from newdf with 'jobtitle' str
d_j = newdf[newdf['jobtitle'].str.contains('Data Analyst')]
D_j = newdf[newdf['jobtitle'].str.contains('data analyst')]
d__j = newdf[newdf['jobtitle'].str.contains('data  analyst')]
DJ = newdf[newdf['jobtitle'].str.contains('dataanalyst')]
dj = newdf[newdf['jobtitle'].str.contains('DataAnalyst')]

#combine them all into one df by row
daj = pd.concat([d_j,D_j,d__j,DJ,dj], axis=0)
daj.head()


# In[ ]:


# capture just the skills
dajskills = daj.loc[:, 'Skill 1':]
dajskills


# In[ ]:





# In[ ]:


# remove the header
dajskills.columns = range(stcksklls.shape[1])
dajskills


# In[ ]:


# get them all in one Series
dajskills.stack()
print(stcksklls)


# stcksk = stcksklls.stack()
# print(stcksk.head())

# # turn into Series
# dastcked = dastck.stack()
# 
# cb = pd.crosstab(index=dastcked, columns="count")
# cb

# skillser = df.concat([newdf], axis=0)

# In[ ]:


dajlist = list(dajskills.values.flatten())
dajlist = [x for x in dajlist if x is not None]


# In[ ]:



skills = {}
for x in dajlist:
    if x in skills:
        skills[x] +=1
    else:
        skills[x] = 1
skilldf = pd.Series(skills)
skilldf
skilldf.keys


# In[ ]:


daj = pd.Series(dajlist)
daj = daj.str.lower()
daj = daj.str.strip()
daj


# In[ ]:


dajlen = [len(x) for x in daj]
avgdaj = sum(dajlen)/(len(dajlen))


# In[ ]:


mypalette = sns.color_palette('GnBu_d', 40)
plt.figure(figsize=(20,10))
sns.countplot(y=df['company'], order=df['company'].value_counts().index, palette=mypalette)
plt.ylabel('Company Name', fontsize=14)
plt.xlabel('Number of Job postings', fontsize=14)
plt.title("Companies with most job postings", fontsize=18)
plt.ylim(20.5,-0.5)
plt.show()


# In[ ]:


mypalette = sns.color_palette('GnBu_d', 40)
plt.figure(figsize=(20,10))
sns.countplot(y=daj, order=daj.value_counts().index, palette=mypalette)
plt.ylabel('Skill', fontsize=14)
plt.xlabel('Count of Mentions', fontsize=14)
plt.title("Demand for Skills for Data Analysts", fontsize=18)
plt.ylim(20.5,-0.5)
plt.show()


# In[ ]:




