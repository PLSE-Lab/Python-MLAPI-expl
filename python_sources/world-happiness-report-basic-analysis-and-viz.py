#!/usr/bin/env python
# coding: utf-8

# #### Beginner learning project
# 
# This is a beginner learning project to practice basic data science libraries.

# ## Data Input

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# #### Data input

# In[ ]:


data2015 = pd.read_csv('../input/world-happiness/2015.csv')
data2016 = pd.read_csv('../input/world-happiness/2016.csv')
data2017 = pd.read_csv('../input/world-happiness/2017.csv')
data2018 = pd.read_csv('../input/world-happiness/2018.csv')
data2019 = pd.read_csv('../input/world-happiness/2019.csv')


# #### Explore data columns and shape

# In[ ]:


print(data2015.shape)
data2015.head()


# In[ ]:


print(data2016.shape)
data2016.head()


# In[ ]:


print(data2017.shape)
data2017.head()


# In[ ]:


print(data2018.shape)
data2018.head()


# In[ ]:


print(data2019.shape)
data2018.head()


# ## Data cleaning

# Fix all columns to be the same. Identify the common columns. 2018 and 2019 has the same data format. while
# 
# - Rank
# - Region
# - Country
# - Score
# - GDP per capita
# - Family/ Social support
# - Healthy  life expectancy
# - Freedom
# - Generosity
# - Perceptions of corruption
# 
# Happiness score are based on the family, life expectancy, freedom, generosity, etc. THe values are how these parameters contribute to to evaluating happiness. it might be unreliable to use this as predictor of score. instead can use at as dependent variable

# In[ ]:


todrop15, todrop16 = ['Standard Error', 'Dystopia Residual'], ['Upper Confidence Interval', 'Lower Confidence Interval', 'Dystopia Residual']
data2015.drop(todrop15, axis = 1, inplace = True)
data2016.drop(todrop16, axis = 1, inplace = True)
data2015.columns, data2016.columns


# In[ ]:


columns1516mapping = {'Happiness Rank':'rank', 'Happiness Score': 'score',
                     'Economy (GDP per Capita)': 'GDP_per_capita', 'Health (Life Expectancy)': 'life_expectancy',
                     'Trust (Government Corruption)': 'gov_trust'}

data2015.rename(columns = columns1516mapping, inplace = True)
data2016.rename(columns = columns1516mapping, inplace = True)
data2015.columns == data2016.columns

#fixed 2015 and 2016 columns


# In[ ]:


#lower case the columns
data2015.columns = data2015.columns.str.lower()
data2016.columns = data2016.columns.str.lower()
data2015.columns == data2016.columns


# Fix 2017 data. Drop some columns and replace name

# In[ ]:


data2017.drop(['Whisker.high','Whisker.low','Dystopia.Residual'], axis = 1, inplace = True)
data2017.shape, data2017.columns


# In[ ]:


columns17mapping = {'Happiness.Rank':'rank', 'Happiness.Score': 'score',
                     'Economy..GDP.per.Capita.': 'GDP_per_capita', 'Health..Life.Expectancy.': 'life_expectancy',
                     'Trust..Government.Corruption.': 'gov_trust'}
data2017.rename(columns = columns17mapping, inplace = True)
data2017.shape, data2017.columns


# In[ ]:


#make lower case
data2017.columns = data2017.columns.str.lower()
data2017.shape, data2017.columns


# In[ ]:


for i in data2017.columns:
    print(i in data2016.columns, i)
#all columns are now in 2016 columns. 1 missing is region


# Fix 2018 and 2019 data. Drop some columns and replace name

# In[ ]:


data2018.columns


# In[ ]:


columns1819mapping = {'Overall rank':'rank','Country or region':'country',
                     'GDP per capita': 'GDP_per_capita', 'Freedom to make life choices':'freedom',
                      'Healthy life expectancy': 'life_expectancy',
                     'Perceptions of corruption': 'gov_trust', 'Social support':'family'}

data2018.rename(columns = columns1819mapping, inplace=  True)
data2019.rename(columns = columns1819mapping, inplace = True)
data2018.columns = data2018.columns.str.lower()
data2019.columns = data2019.columns.str.lower()
data2018.columns == data2019.columns


# In[ ]:


data2019.columns


# In[ ]:


#check if all 2018,2019 columns in 2015,2016

for i in data2018.columns:
    print(i in data2016.columns, i)


# All data now have the same columns. Now lets map the region to country based from the 2015 and 2016 data

# In[ ]:


empty_dict = {}
for country, region in zip(data2015['country'],data2015['region']):
    empty_dict[country] = region
for country, region in zip(data2016['country'],data2016['region']):
    empty_dict[country] = region


# In[ ]:


data2017['region'] = data2017['country'].map(empty_dict)
data2018['region'] = data2018['country'].map(empty_dict)
data2019['region'] = data2019['country'].map(empty_dict)


# Insert years column to know what year the data is from

# In[ ]:


data2015['year'] = 2015
data2016['year'] = 2016
data2017['year'] = 2017
data2018['year'] = 2018
data2019['year'] = 2019


# In[ ]:


columnlis = []
columnlis.append([data2015.columns.values,data2016.columns.values,data2017.columns.values,data2018.columns.values,data2019.columns.values])
data2015.shape, data2016.shape, data2017.shape, data2018.shape, data2019.shape, np.unique(columnlis)


# Can see that there are only 10 unique columns among all the columns. The next thing to do is to stack the dataframes into its appropriate columns. 

# In[ ]:


df = pd.concat([data2015,data2016,data2017,data2018,data2019], axis = 0, ignore_index = True)
df.info()
df.head()


# There are missing 1 missing data in government trust and 19 missing data in region. Check the data
# 
# missing gov_trust

# In[ ]:


df[df['gov_trust'].isnull()]


# The missing data is from UAE in year 2018.  Try to impute data with some previous data of UAE

# In[ ]:


df.loc[df['country'] == 'United Arab Emirates', 'gov_trust'] = df.loc[df['country'] == 'United Arab Emirates', 'gov_trust'].fillna(df.loc[df['country'] == 'United Arab Emirates', 'gov_trust'].mean())


# In[ ]:


df.loc[df['country'] == 'United Arab Emirates', 'gov_trust']
#filled with the mean


# Missing regions

# In[ ]:


df[df['region'].isnull()]


# Both taiwan and Hongkong is just error in naming

# In[ ]:


df[df['country'] == 'Taiwan']


# In[ ]:


df[df['country']=='Hong Kong']


# Manually change taiwan and hong

# In[ ]:


#Taiwan
df.loc[347,'country'] = 'Taiwan'
df.loc[347,'region'] = 'Eastern Asia'

#Hongkong
df.loc[385,'country'] = 'Hong Kong'
df.loc[385,'region'] = 'Eastern Asia'


# Check other countries with missing region and manually change them

# In[ ]:


df.loc[df['region'].isnull(), 'country']


# There are only 6 data. we can manually edit these data

# In[ ]:


#check unique region
df['region'].unique()


# In[ ]:


df[df['country'].str.contains('Trinidad')]


# In[ ]:


#manullay change
df.loc[[507, 664],'country'] = 'Trinidad and Tobago'
df.loc[[507, 664],'region'] = 'Latin America and Caribbean'


# In[ ]:


df[df['country'] == 'Macedonia']


# In[ ]:


df[df['country'] == 'North Macedonia']


# It seems that the 2019 data is just mis entry

# In[ ]:


# we can set north macedonia to macedonia
df.loc[709, 'country'] = 'Macedonia'
df.loc[709, 'region'] = 'Central and Eastern Europe'


# In[ ]:


#from google search, gambia is west african country

df.loc[745,'region'] = 'Middle East and Northern Africa'


# In[ ]:


df[df['country'] == 'North Cyprus']


# In[ ]:


df[df['country'] == 'Northern Cyprus']


# Missing data is from northen cyprus. lets manually input as western europe

# In[ ]:


# Manually input western europe
df.loc[[527,689], 'country'] = 'North Cyprus'
df.loc[[527,689], 'region'] = 'Western Europe'


# Check if there are still missing data

# In[ ]:


#there are no missing data
df.isnull().any(1).sum()


# ## Explore each variable!

# #### Country

# In[ ]:


year_grouped = df.groupby('country')['year'].count()
year_grouped.value_counts()


# There are 142 countries which appeared 142 times, 10 countries that appeared four times, 7 country which appeared once, 4 countries which appeared thrice, and 1 country which appeared 7 times. Check the countries who appeared less than 4 times

# In[ ]:


plt.style.use('ggplot')
plt.figure(figsize = (10,8))
year_grouped[year_grouped < 4].plot.barh()
plt.title('Countries with less than 4 entries in the data')
plt.xlabel('Count')
plt.locator_params(axis='x', nbins=4)


# There is also a mistype on Somaliland region. 

# In[ ]:


df[df['country'].str.contains('Somaliland')]


# In[ ]:


df.loc[90,'country'] = 'Somaliland Region'


# #### Check the region

# In[ ]:


df['region'].unique()


# There are 10 regions all in all

# In[ ]:


plt.style.use('ggplot')
plt.figure(figsize = (10,6))
df['region'].value_counts().sort_values().plot.barh()
plt.title('Number of countries by region')
plt.xlabel('Counts')


# North america has the least amount entires. Check the countries in north america

# In[ ]:


df.loc[df['region'] == 'North America', 'country'].unique()


# There are only 2 countries in North America

# #### Rank

# In[ ]:


rank_group = df.groupby(['year','rank'])['country'].count()
rank_group[rank_group > 1]


# In 2015 and 2016 there are two countries with the same rank. Explore these countries

# In[ ]:


c1 = (df['rank'] == 82)
c2 = (df['year'] == 2015)

df[c1&c2]
#Jordan and Montenegro both has rank 82 in 2015


# In[ ]:


c1 = (df['year'] == 2016)
c2 = (df['rank'].isin([34,57,145]))

df[c1&c2]


# These countries have the same rank and scores

# #### Score

# In[ ]:


df.describe().loc[:,'score']


# In[ ]:


plt.figure(figsize = (10,6))
sns.kdeplot(df['score'])
plt.hist(df['score'], density = True, color = 'orange')
plt.ylabel('Prob')


# #### GDP

# In[ ]:


df.describe().loc[:,'gdp_per_capita']


# In[ ]:


plt.figure(figsize = (10,6))
sns.kdeplot(df['gdp_per_capita'])
plt.hist(df['gdp_per_capita'], density = True, color = 'orange')
plt.ylabel('Prob')


# The data of GDP per capita is skewed

# #### Other indices

# In[ ]:


df.describe().loc[:,'family':'generosity']


# #### Correlation of continous variables

# In[ ]:


plt.figure(figsize = (10,8))
sns.heatmap(df.corr(), annot = True, cmap = 'RdBu')


# - score is highly correlated to the gdp per capita, family, life expectancy, freedom, and government trust.
# - life expectancy is highly correlated to gdp and family
# - government trust and freedom are correlated

# #### Categorical data of country, region, year, rank, on cont variables that might be grouped by category

# #### top10 countries per year

# In[ ]:


def top(df, col = 'rank', n = 10):
    return df.sort_values(by = col)[:n]


# In[ ]:


top10 = df.groupby('year', as_index = False).apply(top, n = 10)
top10.head()


# In[ ]:


fig = plt.figure(figsize = (15,12))
ax = fig.add_subplot()
ax = sns.lineplot(data = top10, y = 'rank', x = 'year', hue = 'country')
ax.set_yticklabels(['','1st','2nd','3rd','4th','5th','6th','7th','8th','9th','10th',''])
plt.locator_params(axis='y', nbins=11)
plt.locator_params(axis='x', nbins=6)
plt.legend(ncol = 3, frameon = False, fontsize = 8)
ax.set_ylim(11, 0)
plt.title('Top 10 countries per year')


# #### Check Denmark

# In[ ]:


denmark = df[df['country'] == 'Denmark']
denmark


# Compare Denmark to average

# In[ ]:


denmark_details = denmark.describe().loc['mean','score':'generosity']
denmark_details.name = 'Denmark'


# In[ ]:


idx = pd.IndexSlice
average_by_region = df.groupby('region').describe().loc[:,idx['score':'generosity','mean']]
average_by_region.columns = average_by_region.columns.swaplevel().droplevel()
denmark_compared = average_by_region.T.merge(denmark_details, right_index = True, left_index = True).T
denmark_compared


# In[ ]:


fig, ax = plt.subplots(figsize = (15,15))
denmark_compared.drop(['family','gov_trust','generosity'],
                     axis = 1).sort_values(by = ['score'], 
                                           ascending = False).T.plot.bar(ax = ax, width = .8)
plt.title('Denmark compared to other regions')
ax.set_xticklabels(['Score','GDP','Life Expectancy','Freedom'], rotation = 60);


# The happiness indices of Denmark is high. Region with the lowest indices are africa and and asia

# #### Top10 most changed countries

# In[ ]:


rank = df.loc[:,['country','rank','year','score']]
rank.head()


# In[ ]:


rankgroup = rank.groupby(['country','year'])['rank'].mean().unstack()
rankgroup['rank diff'] = rankgroup.max(1) - rankgroup.min(1)


# In[ ]:


top10changes = rankgroup.sort_values(by = 'rank diff', ascending = False).head(10)


# In[ ]:


country_top10_change = top10changes.drop('rank diff', 1).stack().reset_index().rename(columns = {0:'rank'})


# In[ ]:


plt.figure(figsize = (15,10))
sns.lineplot(data = country_top10_change, x = 'year', y = 'rank', hue = 'country', palette = sns.color_palette("hls", 10))
plt.legend(ncol = 2)
plt.locator_params(axis='x', nbins=6)
plt.gca().invert_yaxis()
plt.title('Top 10 countries with the highest change in rank');


# Venezuela, Algeria, Zambia dropped their rank from 2015 to 2019 whereas Ivory Coast, Benin, and Honduras increased ranking

# #### Distribution of happiness score by region

# In[ ]:


fig = plt.figure(figsize = (15,12))
ax = fig.add_subplot()
sns.boxplot(data = df, x = 'region', y = 'score', ax = ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right');


# In[ ]:


df.loc[df['region'].str.contains('Australia'),'country'].unique()


# Western europe and Australia and new zealand has low spread because there are only few countries in their region. Bottom region by happiness index are sub-suharan africa and southern asia.

# #### Check the top country by region

# In[ ]:


def top_score(df, col = 'score', n = 5, ascending = True):
    return df.sort_values(by = col)[-n:]


# In[ ]:


top_byregion = df.groupby(['country','region'], as_index = False)['score'].mean().groupby('region').apply(top_score, n = 1).reset_index(drop = True)
fig = plt.figure(figsize = (12,10))
ax = fig.add_subplot()
top_byregion.set_index(['region','country']).plot.bar(ax = ax)

#make 2 rows for xticklabels 
indexes = list(top_byregion.set_index(['region','country']).index.values)
multilist = []
for i in indexes:
    multilist.append(list(i))
region_country = []
for i in multilist:
    region_country.append('\n'.join(i))
ax.set_xticklabels(region_country);
plt.legend('')
plt.title('Happiest Country by Region')


# The difference between the happiest country by region is minimal. Pakistan in Southern Asia is the least happiest among the top happiest countries

# #### Check top5 countries in western europe and middle east ( because there are only 2 countries in NA and australia and Nz)

# In[ ]:


EuMe = df[(df['region'] == 'Western Europe') | df['region'].str.contains('Middle East')]
Top5EuMe = EuMe.groupby(['country','region'], as_index = False)['score'].mean().groupby('region').apply(top_score).reset_index(drop = True)

#plotting
fig = plt.figure(figsize = (12,10))
ax = fig.add_subplot()
sns.barplot(x = 'country', y ='score', data = Top5EuMe, hue = 'region', ax = ax)
plt.title('Top 5 highest scores in Regions WE and ME-NA')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right');


# On average, Countries on Western Europe have relatively close happiness scores. In middle east and north africa, Israel has the happiest population

# #### Bottom countries in southern asia and and subsuharan africa

# In[ ]:


def bottom(df, col = 'score', n = 5, ascending = True):
    return df.sort_values(by = col)[:n]


# In[ ]:


SubAfr_EastAsia = df[(df['region'] == 'Southern Asia') | df['region'].str.contains('Sub-')]
Bottom5 =  SubAfr_EastAsia.groupby(['country','region'], as_index = False)['score'].mean().groupby('region').apply(bottom, ascending = False, n = 5).reset_index(drop = True)

#plotting
fig = plt.figure(figsize = (12,10))
ax = fig.add_subplot()
sns.barplot(x = 'country', y ='score', data = Bottom5, hue = 'region', ax = ax)
plt.title('Bottom 5 countries in Southern Asia and Sub-Suharan Africa')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right');


# Here are the bottom countries in their happiness score
# 

# #### Mean and std of happiness score by Region

# In[ ]:


region_score = df.groupby(['region','year'], as_index = False).agg({'score':['mean',np.std]}).rename(columns = {'mean':'mean_score', 'std':'std_score'})
region_score.columns = region_score.columns.droplevel(1)
region_score.columns = ['region','year','mean_score','std_score']


# In[ ]:


c = ['red','blue','green','cyan','magenta','yellow','black','brown','gray','pink']


# In[ ]:


def year_plot(df, title, col = ['region'], ylim = [3.5,7]):
    fig = plt.figure(figsize = (12,10))
    ax = plt.subplot()
    for i in df.groupby(col):
        plt.plot(i[1].year, i[1].mean_score, label = i[0], marker = 'o')
    plt.legend(ncol = 2, fontsize = 8, columnspacing = .5)
    plt.ylim(ylim)
    plt.title(title)
    plt.locator_params(axis='x', nbins=6)
    
year_plot(region_score, title = 'Happiness Score by region')


# The trend of happiness score by region shows that western Europe Scores have been increasing since 2015 whereas the region Southeastern Asia and Middle East have been decreasing since 2017

# #### Top10 by happiness index

# In[ ]:


def topN(df, col,  title = 'Please insert title', n=10):
    plt.figure(figsize = (10,8))
    df.groupby('country')[col].mean().sort_values(ascending = False)[:n].plot(kind = 'bar', title = title) 


# In[ ]:


topN(df, col = 'generosity', title = 'Top10 most generous country')


# Asian countries are more generous

# In[ ]:


topN(df, col = 'freedom', title = 'Top10 most free country')


# Turns out US is not the most free country

# In[ ]:


topN(df, col = 'gov_trust', n = 10, title = 'Top 10 countries with highest trust on its government')


# I somehow expected singapore to be here

# In[ ]:


topN(df, col = 'family', n = 10, title = 'Top 10 countries with highest family/social support')


# Denmark is included in top 3 in governmetn trust and social support. Iceland is the top1. The difference is minimal

# In[ ]:


topN(df, col = 'gdp_per_capita', n = 10, title = 'Top 10 countries with highest trust on its government')


# Qatar and UAE has the highest GDP per capita. Let's try to locate the rank of the countries with top10 highest GDP

# In[ ]:


Top10GDP_rank = df.groupby('country')['gdp_per_capita','rank'].mean().nlargest(10, 'gdp_per_capita').loc[:,'rank']
fig = plt.figure(figsize = (10,8))
ax = fig.add_subplot()
ax.plot(Top10GDP_rank.index,Top10GDP_rank, linestyle = '', marker = 'o')
ax.set_xticklabels(Top10GDP_rank.index, rotation = 60)
ax.set_ylim(0, 80)
for country, rank in zip(Top10GDP_rank.index,Top10GDP_rank):
    ax.annotate(rank,(country, rank + 3))


# #### Philippines

# In[ ]:


ph = df[df['country'] == 'Philippines']
ph


# #### Compare Philippines to average

# In[ ]:


average = df.describe().loc['mean','score':'generosity']
average.name = 'Global Average'


# In[ ]:


ph_mean = ph.describe().loc['mean','score':'generosity']
ph_mean.name = 'Philippine'


# In[ ]:


fig = plt.figure(figsize = (10,8))
ax = fig.add_subplot()
pd.DataFrame([average, ph_mean]).T.plot.barh(ax = ax)
ax.set_ylabel('Indices')
ax.set_xlabel('Value')
ax.set_title('Philippine happiness indices compared to global average')


# We can fairly say the philippine is near average. It has higher scores on family and freedom but has low life expectanc

# In[ ]:


fig = plt.figure(figsize = (10,8))
ax = fig.add_subplot()
ax.plot(ph['year'], ph['rank'])
ax.locator_params(axis='x', nbins=6)
ax.invert_yaxis()
ax.set_ylabel('Rank')
ax.set_xlabel('year')
ax.set_title('Philippine ranking on happiness (2015 - 2019)')

for year, rank in zip(ph['year'],ph['rank']):
    ax.annotate(rank,(year, rank + 1))


# Philippine ranking has improved over the years

# In[ ]:


fig = plt.figure(figsize = (10,8))
ax = fig.add_subplot()
ph.set_index('year').loc[:,'gdp_per_capita':'generosity'].plot(kind = 'line', ax = ax, marker = 'o')
plt.legend(ncol = 2)
ax.locator_params(axis='x', nbins=6)
ax.set_ylim(0, 1.4)
plt.title('Happiness index in the Philippines')


# Some parameters were stable from 2015 to 2019. The family or social support has increased from 2016 to 2019. Generosity decreased

# #### Ph rank in Southeastern asian country

# In[ ]:


fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot()
df[df['region'] == 'Southeastern Asia'].groupby('country').agg('mean').drop(['rank','year'],1).sort_values(by = 'score', ascending = False).T.plot(kind = 'bar', ax = ax, width = .8)
ax.set_title('Philippine compared to other regions')
ax.set_ylabel('Value')
ax.set_xlabel('Variables')
plt.legend(ncol = 3)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right');


# Philippines is the color gray. We ranked 4 in SEA in terms of happiness score. Our life expectancy is among the lowest. Singapore has very high value in government trust compared to other SEA countries. We can see here that singapore is the best SEA country

# ## Continous variables

# In[ ]:


df.head()


# In[ ]:


indices = df.groupby('country').agg('mean').loc[:,'gdp_per_capita':'generosity']
plt.figure(figsize = (10,8))
for i in indices.columns:
    sns.distplot(df[i], label = i)
plt.legend()
plt.title('Distribution of happiness indices')


# In[ ]:


regions = df['region'].unique()
regions


# #### Scatterplot of gdp_per_capita and score

# In[ ]:


fig = plt.figure(figsize = (12,10))
ax = fig.add_subplot()
sns.scatterplot(data = df, x = 'score', y ='gdp_per_capita', hue = 'region', ax = ax)
ax.set_title('Scatterplot of GDP and score');


# #### Check GDP per capita and generosity

# In[ ]:


sns.lmplot(data = df, x = 'gdp_per_capita', y='generosity', height = 10)
plt.title('GDP per capita vs generosity')


# Contrary to my hypothesis that GDP is correlated to generosity, it seems that there is no linear relationship between the two variables

# In[ ]:


plt.figure(figsize = (12,10))
sns.scatterplot(data= df, x = 'family', y = 'life_expectancy', size = 'gdp_per_capita')
plt.legend(fontsize = 'large')
plt.title('Social Support and Life expectancy')


# Social support and family expectancy is likely to have a positive linear relationship. GDP is also a contributing factor to life expectancty

# In[ ]:


sns.lmplot(data = df, x = 'gov_trust', y = 'freedom', hue = 'region', height = 10, legend_out = False)
plt.title('Government Trust vs Freedom')


# In most regions, government trust is linearly related to freedom

# #### Confidence interval for score

# In[ ]:


plt.figure(figsize = (10,8))
sns.kdeplot(df['score'], shade = True)


# #### Try to estimate using normal distribution

# In[ ]:


import scipy.stats as st
mean = df['score'].mean()
std = df['score'].std()

xs = np.linspace(0,9, 10000)
ps = st.norm.pdf(xs, mean, std)

plt.figure(figsize = (10,8))
sns.kdeplot(df['score'], shade = True)
plt.plot(xs, ps)


# It seems like a bad estimate

# In[ ]:


mean = df['score'].mean()
std = df['score'].std()

xs = np.linspace(0,9, 10000)
ps = st.norm.pdf(xs, mean, std)

p_lognorm = st.lognorm.fit(df['score'])
pdf_lognorm = st.lognorm.pdf(xs, *p_lognorm)

p_skewnorm = st.skewnorm.fit(df['score'])
pdf_skewnorm = st.skewnorm.pdf(xs, *p_skewnorm)

plt.figure(figsize = (10,8))
sns.kdeplot((df['score']), shade = True)
plt.plot(xs, ps, label = 'normal')
plt.plot(xs, pdf_lognorm, label = 'log norm')
plt.plot(xs, pdf_skewnorm, label = 'skew norm')
plt.legend()


# I don't have enough knowledge yet to compute for the probability of an empirical distribution

# ## From stack overflow

# In[ ]:


import scipy.stats as st
def get_best_distribution(data):
    dist_names = ["norm", "exponweib", "weibull_max", "weibull_min", "pareto", "genextreme"]
    dist_results = []
    params = {}
    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(data)

        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = st.kstest(data, dist_name, args=param)
        print("p value for "+dist_name+" = "+str(p))
        dist_results.append((dist_name, p))

    # select the best fitted distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    # store the name of the best fit and its p value

    print("Best fitting distribution: "+str(best_dist))
    print("Best p value: "+ str(best_p))
    print("Parameters for the best fit: "+ str(params[best_dist]))

    return best_dist, best_p, params[best_dist]


# In[ ]:


best_dist, best_p, params = get_best_distribution(df['score'])


# #### For next time

# In[ ]:




