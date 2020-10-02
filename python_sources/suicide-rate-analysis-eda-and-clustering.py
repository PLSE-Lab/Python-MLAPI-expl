#!/usr/bin/env python
# coding: utf-8

# <font size="5">EDA and Visualization</font>

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")
plt.style.use('ggplot')


# In[ ]:


df = pd.read_csv("../input/master.csv")
df.head()


# In[ ]:


df.shape


# In[ ]:


## Summarizing the data
df.describe()


# In[ ]:


## Renaming some columns for better interpretation
df.rename(columns={" gdp_for_year ($) ":
                  "gdp_for_year", "gdp_per_capita ($)":
                  "gdp_per_capita"}, inplace=True)
df.head()


# In[ ]:


## Suicides number curve (1985-2016)
ns = df['suicides_no'].groupby(df.year).sum()
ns.plot(figsize=(8,6), linewidth=2, fontsize=15,color='black')
plt.xlabel('year', fontsize=15)
plt.ylabel('suicides_no',fontsize=15)


# <font size="2">According to this plot numbers of suicides had been decreasing overall</font>

# In[ ]:


## Mean suicides number by gender and 100k population
df["year"] = pd.to_datetime(df["year"], format = "%Y")
data = df.groupby(["year", "sex"]).agg("mean").reset_index()
sns.lineplot(x = "year", y = "suicides/100k pop", hue = "sex", data = df)
plt.xlim("1985", "2015")
plt.title("Evolution of the mean suicides number per 100k population (1985 - 2015)");


# In[ ]:


df = df.groupby(["year", "sex", "age"]).agg("mean").reset_index()

sns.relplot(x = "year", y = "suicides/100k pop", 
            hue = "sex", col = "age", col_wrap = 3, data = df, 
            facet_kws=dict(sharey=False), kind = "line")

plt.xlim("1985", "2015")
plt.subplots_adjust(top = 0.9)
plt.suptitle("Evolution of suicide by sex and age category (1985 - 2015)", size=15);


# In[ ]:


## Number of suicides in 1985
year_1985 = df[(df['year'] == 1985)]
year_1985 = year_1985.groupby('country')[['suicides_no']].sum().reset_index()

## Sorting values in ascending order
year_1985 = year_1985.sort_values(by='suicides_no', ascending=False)

## Styling output dataframe
year_1985.style.background_gradient(cmap='Purples', subset=['suicides_no'])


# In[ ]:


#Number of suicides in 2016
year_2016 = df[(df['year'] == 2016)]
year_2016 = year_2016.groupby('country')[['suicides_no']].sum().reset_index()

# Sort values in ascending order
year_2016 = year_2016.sort_values(by='suicides_no', ascending=False)

# Styling output dataframe
year_2016.style.background_gradient(cmap='Oranges', subset=['suicides_no'])


# In[ ]:


## Suicides number by generation and sex
f,ax = plt.subplots(1,1,figsize=(13,6))
ax = sns.barplot(x = df['generation'], y = 'suicides_no',
                  hue='sex',data=df, palette='autumn')


# In[ ]:


## Suicides number by age and sex
f,ax = plt.subplots(1,1,figsize=(13,6))
ax = sns.barplot(x = df['age'], y = 'suicides_no',
                  hue='sex',data=df, palette='Accent')


# <font size="2">These barplots show that generation of boomers have the highest suicide rate, males in general are more likely to commit suicides than females as well as people from age groups 35-54 yrs and 55-74 yrs</font>

# In[ ]:


## Suicides number by year
f,ax = plt.subplots(1,1,figsize=(16,6))
ax = sns.barplot(x = df['year'], y = 'suicides_no',
                data=df, palette='Spectral')


# In[ ]:


## Correlation of features
f,ax = plt.subplots(1,1,figsize=(10,10))
ax = sns.heatmap(df.corr(),annot=True, cmap='coolwarm')


# <font size="2">The correlation between the factors except population with GDP for year is low</font>

# In[ ]:


data = df['suicides_no'].groupby(df.country).sum().sort_values(ascending=False)
f,ax = plt.subplots(1,1,figsize=(10,20))
ax = sns.barplot(data.head(20), data.head(20).index, palette='Reds_r')


# <font size="2">The highest number of suicides is in Russian Federation</font>

# In[ ]:


data = df['suicides_no'].groupby(df.country).sum().sort_values(ascending=False)
f,ax = plt.subplots(1,1,figsize=(10,20))
ax = sns.barplot(data.tail(20),data.tail(20).index,palette='Blues_r')


# <font size="2">The lowest number of suicides is in San Marino</font>

# In[ ]:


## Suicides number by year (high to low)
year_suicides = df.groupby('year')[['suicides_no']].sum().reset_index()
year_suicides.sort_values(by='suicides_no', ascending=False).style.background_gradient(cmap='Greens', subset=['suicides_no'])


# <font size=2>The highest number of suicides was in 1999 and the lowest in 2016</font>

# In[ ]:


## Suicides number by age group
age_grp = df.groupby('age')[['suicides_no']].sum().reset_index()
age_grp.sort_values(by='suicides_no', ascending=False).style.background_gradient(cmap='Greys', subset=['suicides_no'])


# <font size=2> People with age 35-54 and 55-74 committed suicides the most</font>

# In[ ]:


## Suicides number per 100k population
per100k = df.groupby(['country', 'year'])[['suicides/100k pop']].sum().reset_index()
per100k.sort_values(by='suicides/100k pop', ascending=False).head(20).style.background_gradient(cmap='Reds', subset=['suicides/100k pop'])


# <font size=2> Lithuania has the highest number of suicides per 100k population </font>

# <font size="2">There are missing values in our dataset (HDI for year). Let's check how many</font>

# In[ ]:


df.count()


# <font size="2">Since HDI for year is continious, we can fill those missing values with mean values</font>

# In[ ]:


df.fillna(df.mean(), inplace=True)

## We don't need the column "country-year", so we'll just drop it
df.drop("country-year", axis=1, inplace=True)
df.head()


# In[ ]:


df.count()


# <font size="2">As we can see, there are no missing values anymore. Now we need to check the type of our data</font>

# In[ ]:


df.dtypes


# In[ ]:


(df.dtypes=="object").index[df.dtypes=="object"]


# In[ ]:


## Turning object types into category and integer types
df[["country","age","sex","generation"]] = df[["country","age","sex","generation"]].astype("category")
## Converting number strings with commas into integer
df['gdp_for_year'] = df['gdp_for_year'].str.replace(",", "").astype("int")
df.info()


# In[ ]:


## Checking the relationship between gdp for year and number of suicides
f, ax = plt.subplots(1,1, figsize=(10,8))
ax = sns.scatterplot(x="gdp_for_year", y="suicides_no", data=df, color='purple')


# <font size="2">The relationship between "gdp_for_year" and "suicides_no" is not linear. Hence, GDP is not something that has a real impact on suicide rate </font>

# In[ ]:


## Checking the relationship between gdp per capita and number of suicides
f, ax = plt.subplots(1,1, figsize=(10,8))
ax = sns.scatterplot(x="gdp_per_capita", y="suicides_no", data=df, color='yellow')


# <font size=2> Again, GDP has no real impact on suicide rate </font>

# In[ ]:


## Checking the relationship between Hdi and number of suicides
f, ax = plt.subplots(1,1, figsize=(10,8))
ax = sns.scatterplot(x="HDI for year", y="suicides_no", data=df, color='cyan')


# In[ ]:


##Suicides by age and gender in Russian Federation
f, ax = plt.subplots(1,1, figsize=(10,10))
ax = sns.boxplot(x='age', y='suicides_no', hue='sex',
                 data=df[df['country']=='Russian Federation'],
                 palette='Set1')


# <font size='2'> Males in Russia aged from 35 to 54 yrs and females aged 55-74 yrs commit suicide more often </font>

# In[ ]:


##Suicides by age and gender in Brazil
f, ax = plt.subplots(1,1, figsize=(10,10))
ax = sns.boxplot(x='age', y='suicides_no', hue='sex',
                 data=df[df['country']=='Brazil'],
                 palette='Set2')


# <font size='2'> In Brazil both males and females commit suicides mostly at age 35-54 </font>

# <font size="5">Machine Learning</font>

# In[ ]:


## Using cat.codes method to convert category into numerical labels
columns = df.select_dtypes(['category']).columns
df[columns] = df[columns].apply(lambda fx: fx.cat.codes)
df.dtypes


# <font size="3">**K-means Clustering**<font>

# <font size='2'>The task is to cluster the countries into two groups - the ones with high number of suicides and the ones with low number of suicides. For that we have to drop the 'suicides_no' column from the dataset and make it unlabeled</font>

# In[ ]:


from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
x = df.drop('suicides_no', axis=True)
y = df['suicides_no']
kmeans = KMeans(n_clusters=2)
kmeans.fit(x)
KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=600,
    n_clusters=2, n_init=10, n_jobs=1, precompute_distances='auto',
    random_state=0, tol=0.0001, verbose=0)
y_kmeans = kmeans.predict(x)
x, y_kmeans = make_blobs(n_samples=600, centers=2, cluster_std=0.60, random_state=0)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x[:,0], x[:,1], c=y_kmeans, cmap='cool')


# In[ ]:


from sklearn.metrics import silhouette_score
print(silhouette_score(x, y_kmeans))


# <font size='2'>Great! The model was able to cluster correctly with a 71% without even tweaking any parameters of the model itself and scaling the values of the features</font>

# <font size='5'>Conclusions</font>
# 

# <font size='2'>Data cleaning is very important, as real world data is usually messy. Visualizing the data is also a very important step because it makes it easier for a lot of people to understand the data and detect patterns, trends and outliers. K-means clustering algorithm (which found a strong structure in our dataset) was easy to implement in this case, since we had some domain knowledge that told us the number of suicides committed by people in different countries, so we didn't have to pre-specify the number of clusters(k). However, this doesn't always happen that way. As for suicides and factors that influence them one can say while age and gender can be some of those factors, Gdp and Hdi not really, because even in countries with high Gdp and Hdi a lot of people commit suicide. Other than that, there's not enough data available for better analysis, as there are other biological, psychological and social factors that may cause suicides (race, ethnicity, social isolation, contagion, religion, etc.), as well as geographical (climate)</font>
