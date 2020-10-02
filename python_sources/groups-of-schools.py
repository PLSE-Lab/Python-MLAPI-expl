#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_cell_magic('javascript', '', "$.getScript('https://kmahelona.github.io/ipython_notebook_goodies/ipython_notebook_toc.js')")


# <h1 id="tocheading">Table of Contents</h1>
# <div id="toc"></div>

# # Groups of Schools
# 
# This is my first public kernel!  I will do a very simple thing: group schools by different sets of characteristics and visualize the effects on another sets of characteristics. For example, I will use the ethnicity, framework measures, student achievement, etc...
# 
# My hope is that by providing different groupings, PASSNYC may get a better idea of what works for one group or another.
# 
# Feel free to borrow any of the code present here into your own work!

# # Preparing the data
# 
# Here we do a bunch of things to get the data nice and beautiful. You can skip over this part if you want.

# In[ ]:


import re

import numpy as np
import pandas as pd

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', None)


df = pd.read_csv('../input/data-science-for-good/2016 School Explorer.csv')

# set index
df = df.set_index('Location Code')
df.index.name = 'DBN'

# remove unused columns
columns = [
    'Adjusted Grade',   
    'New?',
    'Other Location Code in LCGMS',
    'School Name',
    'SED Code',
    'Address (Full)',
    'Rigorous Instruction Rating',
    'Collaborative Teachers Rating',
    'Supportive Environment Rating',
    'Effective School Leadership Rating',
    'Strong Family-Community Ties Rating',
    'Trust Rating',
    'Student Achievement Rating',
    'District',
    'City',
    'Zip'
]
df = df.drop(columns, axis=1)

# split columns with amount of 4s
result_columns = [c for c in df.columns if re.match(r'Grade \d.+', c)]
df4s = df[result_columns]
df = df.drop(result_columns, axis=1)

# fix bad column name
df4s = df4s.rename(columns={
    'Grade 3 Math - All Students tested': 'Grade 3 Math - All Students Tested'
})

# parse values
def parse_pct(x):
    if isinstance(x, str) and x[-1] == '%':
        return float(x[:-1]) / 100
    else:
        return x
    
def parse_income(x):
    if isinstance(x, str):
        return float(x[1:].replace(',', ''))
    else:
        return np.nan
    
df = df.applymap(parse_pct)
df['Community School?'] = df['Community School?'].apply(lambda x: int(x == 'Yes'))
df['School Income Estimate'] = df['School Income Estimate'].apply(parse_income)

# fix bad attendance rate values
df.loc[df['Student Attendance Rate'] == 0.0, 'Student Attendance Rate'] = np.nan
df.loc[df['Percent of Students Chronically Absent'] == 1.0, 'Percent of Students Chronically Absent'] = np.nan

# fix bad 'Rigorous Instruction %' values
df.loc[df['Rigorous Instruction %'] == 0, 'Rigorous Instruction %'] = np.nan

# fix school with bad framework entries
df.loc['84M202', 'Rigorous Instruction %':'Trust %'] = np.nan

# summarize available grades
def parse_grade(x):
    if x == 'PK':
        return -1
    elif x == '0K':
        return 0
    else:
        return int(x)
    
def summarize_grades(x):
    low = parse_grade(x['Grade Low'])
    high = parse_grade(x['Grade High'])
    
    low_category = None
    if low in (-1, 0):
        low_category = 'kinder'
    elif low in (5, 6, 7):
        low_category = 'elementary(5-7)'
        
    high_category = None
    if high in (4, 5, 6):
        high_category = 'elementary(4-6)'
    if high == 8:
        high_category = 'middle(8)'
    if high == 12:
        high_category = 'high(12)'
        
    if low_category and high_category:
        return "{} - {}".format(low_category, high_category)
    else:
        return 'other'

df['Available Grades'] = df.apply(summarize_grades, axis=1)
df['SE Grade'] = df['Grades'].str.contains('SE')
df = df.drop(['Grades', 'Grade Low', 'Grade High'], axis=1)

# add borough information
boroughs = pd.read_csv('../input/retrieve-school-boroughs/NYC Schools Boroughs.csv', index_col=0)
df = df.join(boroughs)

df.head()


# In[ ]:


### percentage of 4s for each school
### percentage4s is used by the plot_percentage_of_4s function


# melt df so that column names become a variable
melted = df4s.reset_index().melt(id_vars='DBN', value_name='Count')

# split column names (now a variable) into 'Grade', 'Test' and 'Group'
pattern = r'^Grade (?P<Grade>\d) (?P<Test>\w+) (?:- |)(?P<Group>.*)$'
split_cols = melted.variable.str.extract(pattern)

# merge new variables with DBN and Count
split = pd.concat([
    melted[['DBN', 'Count']],
    split_cols
], axis=1)[['Test', 'DBN', 'Grade', 'Group', 'Count']]

# calculate total percentage of 4s per grade in each school
def calculate_grade_pct(df):
    all_students_tested = df[df.Group == 'All Students Tested'].Count.sum()  # average for both tests
    all_students_4s = df[df.Group == '4s - All Students'].Count.sum()
    
    if all_students_tested > 0:
        return all_students_4s / all_students_tested
    else:
        return np.nan
aggregate = split.groupby(['DBN', 'Grade']).apply(calculate_grade_pct).rename('Mean 4s').reset_index()

# unstack results so we can use them with parallel_coordinates
planified = aggregate.set_index(['DBN', 'Grade'])['Mean 4s'].unstack()
planified.columns.name = None
planified.columns = ["Grade {}".format(c) for c in planified.columns]

# nicer name
percentage4s = planified.copy()
percentage4s.head()


# In[ ]:


### available grades sorted by incindence
### grades is used by the plot_available_grades function

available_grades = df['Available Grades'].value_counts().index
available_grades.tolist()


# # Plotting utilities
# 
# Let's prepare functions for:
# 
# - Plotting clusters statistics
# - Plotting clusters in a map
# - Plotting continuous features in a map

# In[ ]:


### functions here are made to be called indirectly (but nothing impedes you of calling them directly)


import warnings

import matplotlib.cbook
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import display
from mpl_toolkits.basemap import Basemap
from pandas.plotting import parallel_coordinates

# ignore deprecation warnings
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)


# map util

NYC_MAP = plt.imread('../input/nycmap/basemap.png')

def plot_basemap(ax=None):
    m = Basemap(projection='merc',
                llcrnrlat=40.486,
                llcrnrlon=-74.261,
                urcrnrlat=40.925,
                urcrnrlon=-73.691,
                resolution='c',
                epsg='3623')
    m.imshow(NYC_MAP, origin='upper', ax=ax)
    m.drawmapboundary(ax=ax)
    return m
    
# minimap

def plot_minimap(data, ax):
    map = plot_basemap(ax)
    lat = data['Latitude'].values
    lon = data['Longitude'].values
    x, y = map(lon, lat)    
    ax.scatter(x, y, alpha=0.5, s=10)
    ax.set_title('Location of {} Schools'.format(data.shape[0]))
    
# parallel coordinates

def plot_ethnic_distribution(data, ax):
    data = data[['Percent Hispanic', 'Percent Black', 'Percent White', 'Percent Asian']].copy()
    data['dummy'] = '-'
    parallel_coordinates(data, 'dummy', linewidth=1.0, ax=ax)
    ax.set_title("Ethnic distribution")
    ax.set_ylim(0.0, 1.0)
    
def plot_framework_measures(data, ax):
    data = data[['Rigorous Instruction %', 'Collaborative Teachers %', 
                 'Supportive Environment %', 'Effective School Leadership %', 
                 'Strong Family-Community Ties %', 'Trust %']].copy()
    data.columns = [c[:-2] for c in data.columns]
    data['dummy'] = '-'
    parallel_coordinates(data, 'dummy', linewidth=1.0, ax=ax)
    ax.set_title("Framework Measures")
    ax.set_ylim(0.0, 1.0)
    
def plot_percentage_of_4s(DBN, ax):
    data = percentage4s.loc[DBN]
    data['dummy'] = '-'
    parallel_coordinates(data, 'dummy', linewidth=1.0, ax=ax)
    ax.set_title("Percentage of 4s")
    ax.set_ylim(0.0, 1.0)
    
# univariate KDE

def plot_eni(data, ax):
    data = data[data['Economic Need Index'].notnull()]
    sns.distplot(data['Economic Need Index'], hist=False, kde_kws={"shade": True}, ax=ax)
    ax.set_xlim(0, 1)
    ax.set_title('Economic Need Index')
    ax.set_xlabel('')
    
def plot_ell(data, ax):
    sns.distplot(data['Percent ELL'], hist=False, kde_kws={"shade": True}, ax=ax)
    ax.set_xlabel('Percent ELL (English Language Learner)')
    ax.set_xlim(0, 1)
    ax.set_title('Percent ELL (English Language Learner)')
    ax.set_xlabel('')
    
def plot_estimated_income(data, ax):
    data = data[data['School Income Estimate'].notnull()].copy()
    data['School Income Estimate (thousands)'] = data['School Income Estimate'] / 1000
    sns.distplot(data['School Income Estimate (thousands)'], hist=False, kde_kws={"shade": True}, ax=ax)
    ax.set_xlim(0.0, 200.0)
    ax.set_title('Estimated School Income (thousands)')
    ax.set_xlabel('')
    
# bivariate KDE

def plot_proficiency(data, ax):
    # remove null entries
    notnull = data[['Average ELA Proficiency',
                    'Average Math Proficiency']].notnull().all(axis=1)
    data = data.loc[notnull]
    
    # plot
    x = data['Average ELA Proficiency']
    y = data['Average Math Proficiency']
    sns.kdeplot(x, y, shade=True, cmap='Blues', ax=ax)
    ax.set_xlim(1.0, 4.5)
    ax.set_ylim(1.0, 4.5)
    
def plot_attendance(data, ax):
    # remove null entries
    notnull = data[['Student Attendance Rate', 
                    'Percent of Students Chronically Absent']].notnull().all(axis=1)
    data = data.loc[notnull]
    
    # plot KDE
    x = data['Student Attendance Rate']
    y = data['Percent of Students Chronically Absent']
    sns.kdeplot(x, y, shade=True, cmap='Blues', ax=ax)
    ax.set_xlim(0.5, 1.00)
    ax.set_ylim(0.00, 0.5)
   
# bar plot

def plot_community_school(data, ax):
    cnt = data.groupby('Community School?').size()
    
    xs = ['Community School', 'Regular School']
    ys = [cnt.get(1, 0), cnt.get(0, 0)]
    ax.bar(xs, ys)
    ax.set_title('Community School?')
    
def plot_available_grades(data, ax):
    grades_cnt = data['Available Grades'].value_counts().to_dict()
    
    xs = available_grades
    ys = [grades_cnt.get(g, 0) for g in available_grades]
    
    ax.bar(xs, ys)
    ax.set_title('Grades Available')
    ax.tick_params(rotation=90)
    
def plot_unavailable(data, ax):
    eni = data['Economic Need Index'].isnull().mean()
    sie = data['School Income Estimate'].isnull().mean()
    attendance = data[['Student Attendance Rate', 
                       'Percent of Students Chronically Absent']].isnull().any(axis=1).mean()
    proficiency = data[['Average ELA Proficiency',
                        'Average Math Proficiency']].isnull().any(axis=1).mean()
    framework = data[['Rigorous Instruction %', 'Collaborative Teachers %', 
                      'Supportive Environment %', 'Effective School Leadership %', 
                      'Strong Family-Community Ties %', 'Trust %']].isnull().any(axis=1).mean()

    xs = ['Attendance', 'Economic Need Index', 'NYS Test Proficiency', 
          'Framework Measures', 'School Income Estimate']
    ys = [attendance, eni, proficiency,
          framework, sie]

    ax.bar(xs, ys)
    ax.set_title('Percentage of null values')
    ax.tick_params(rotation=20)
    ax.set_ylim(0.0, 1.0)


# In[ ]:


### functions here are made to be called directly


# mapping

def plot_cluster_map(cluster, data, **kwargs):
    plt.figure(figsize=(20,12))
    m = plot_basemap()
    for c, d in data.groupby(cluster):        
        lat = d['Latitude'].values
        lon = d['Longitude'].values
        x, y = m(lon, lat)
        plt.scatter(x, y, label=c, **kwargs)
    plt.title(cluster.name)
    plt.legend()
    
def plot_continuous_map(features, data, **kwargs):
    fig, ax = plt.subplots(figsize=(20,12))
    m = plot_basemap()    
    
    lat = data['Latitude'].values
    lon = data['Longitude'].values
    x, y = m(lon, lat)    
    plt.scatter(x, y, c=features, **kwargs)
    
    plt.title(features.name)
    plt.colorbar(fraction=0.046, pad=0.04)  # magic padding https://stackoverflow.com/a/26720422/


# aggregate grids

def plot_schools_info(data, title=''):
    fig = plt.figure(figsize=(20, 8))
    fig.suptitle(title)
    
    gs = gridspec.GridSpec(2, 4)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1:])
    ax3 = plt.subplot(gs[1, 0])
    ax4 = plt.subplot(gs[1, 1])
    ax5 = plt.subplot(gs[1, 2])
    ax6 = plt.subplot(gs[1, 3])

    plot_minimap(data, ax1)
    plot_framework_measures(data, ax2)
    plot_available_grades(data, ax3)
    plot_community_school(data, ax4)
    plot_estimated_income(data, ax5)
    plot_unavailable(data, ax6)
    
def plot_students_info(data, title=''):
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(title)
    
    gs = gridspec.GridSpec(3, 4)
    ax1 = plt.subplot(gs[0, :3])
    ax2 = plt.subplot(gs[0, 3])
    ax3 = plt.subplot(gs[1, :3])
    ax4 = plt.subplot(gs[1, 3])
    ax5 = plt.subplot(gs[2, 1])
    ax6 = plt.subplot(gs[2, 2])

    plot_ethnic_distribution(data, ax1)
    plot_ell(data, ax2)
    plot_percentage_of_4s(data.index, ax3)
    plot_eni(data, ax4)
    plot_proficiency(data, ax5)
    plot_attendance(data, ax6)
    
    
### clusters utils

def plot_clusters(clusters, data):
    # plot big map with clusters
    display.display_markdown('#### Map', raw=True)
    plot_cluster_map(clusters, data)
    plt.show()
    
    # unique clusters by number of occurences
    clusters = pd.Series(clusters)
    unique = clusters.value_counts().index.tolist()
    
    # plot schools characteristics
    display.display_markdown('#### Schools Characteristics', raw=True)
    for cluster in unique:
        d = data[clusters == cluster]
        plot_schools_info(d, title="Cluster {!r}".format(cluster))
        plt.show()
    
    # plot students characteristics    
    display.display_markdown('#### Students Characteristics', raw=True)
    for cluster in unique:
        d = data[clusters == cluster]
        plot_students_info(d, title="Cluster {!r}".format(cluster))
        plt.show()


# # Visualizing continuous features
# 
# This is not the objective of this kernel, so, the plots will be here but the interpretation is up to the reader! :)

# ## School Income Estimate
# 
# *note: this value is present in only about 2/3 of the schools*

# In[ ]:


plot_continuous_map(df['School Income Estimate'], df)


# ## Economic Need Index
# 
# This measures the poverty among *students*.

# In[ ]:


plot_continuous_map(df['Economic Need Index'], df, vmin=0, vmax=1, cmap='Reds')


# ## Percentual of ELL (English Language Learners)

# In[ ]:


plot_continuous_map(df['Percent ELL'], df, vmin=0, vmax=1)


# ## Student Attendance Rate

# In[ ]:


plot_continuous_map(df['Student Attendance Rate'], df, cmap="Blues")


# ## Percent of Students Chronically Absent
# 
# A student is considered chronically absent if he misses more than *10%*  of his classes.

# In[ ]:


plot_continuous_map(df['Percent of Students Chronically Absent'], df, cmap="Reds")


# ## Average ELA (English Language Arts) Proficiency

# In[ ]:


plot_continuous_map(df['Average ELA Proficiency'], df, vmin=1.8, vmax=4.2)  # normalize both proficiency cmaps


# ## Average Math Proficiency

# In[ ]:


plot_continuous_map(df['Average Math Proficiency'], df, vmin=1.8, vmax=4.2)


# I said I wouldn't comment, but, let me make a note. Look at the northeastern part of Brooklyn and at Bronx across all maps. These areas seem to be suffering in all the metrics we used (except 'Percent ELL', but this has no good or bad sides). Meaning... All school factors are probably interrelated with one another in some deeper sense.
# 
# ![New York City - Boroughs](https://i.imgur.com/xQ0XrkL.jpg)

# # Binary Categories
# 
# Here we will select some categories, and visualize the difference between the two groups of schools are are present or not in them.

# ## Community Schools
# 
# "A type of publicly funded school that serves as both an educational institution and a center of community life."

# In[ ]:


plot_clusters(df['Community School?'], df)


# ### Remarks
# 
# *from now on I'll be doing remarks on things that may not be so obvious by looking at the plots (and some obvious things too)*
# 
# Community schools characteristics:
# 
# - They seem to be concentrated in the worst performing regions of the city
# - The majority of them are middle schools (instead of elementary schools)
# - Their income is cut at about \$60,000 (usual schools go up to \$200,000)
# - There are more missing values (for the School Income Estimate)
# 
# Community school students characteristics:
# 
# - There are very few white and asian students
# - There are a bit more of ELL
# - Students are poorer
# - Student performance and attendance are overall worse
# 
# Unintuitive findings:
# 
# - Despite having seemingly worse conditions, the performance of community schools on the NYC Framework for Great School is quite similar to that of regular schools

# ## Schools with Special Education (SE)
# 
# Special education is about teaching children who have special needs or behavioral problems.

# In[ ]:


plot_clusters(df['SE Grade'], df)


# In[ ]:


# checking if Percentage of 4s has fewer students than usual (yes!)

fig, ax = plt.subplots()
plot_percentage_of_4s(df.sample(25, random_state=1).index, ax)


# In[ ]:


# checking if Average Proficiency has two clusters (no!)

d = df[df['SE Grade']]
plt.scatter(d['Average ELA Proficiency'], d['Average Math Proficiency']);


# ### Remarks
# 
# There are only 25 schools with Special Education, and they look quite similar to the usual schools.
# 
# - They are fairly spread across the city
# - *Perhaps* schools with a high overall student achievement don't have SE classes

# ## Schools with missing values

# In[ ]:


df.isnull().sum().sort_values(ascending=False).head(15)


# Most missing values are in the School Income Estimate. Missing values are also present in:
# 
# - Average NYS Test Proficiency
# - Student Attendance
# - NYC Framework measures
# - Economic Need Index
# 
# Since there are a lot of sources of NA, let's just look at the main one:

# ## Missing 'School Income Estimate'

# In[ ]:


income_na = df['School Income Estimate'].isnull().rename('Income Unavailable?')
plot_clusters(income_na, df)


# ### Remarks
# 
# - All characteristics look the same, except
# - Most schools with NA do not have a kindergarten (and most schools without NA *do* have a kindergarten)

# # Multi-valued categories
# 
# Here we will visualize categories which have more than 1 value.

# ## Borough
# 
# New York is composed of five boroughs:
# 
# - Manhattan
# - Brooklyn
# - Queens
# - Bronx
# - Staten Island

# In[ ]:


plot_clusters(df['Borough'], df)


# In[ ]:


df.groupby('Borough')['Effective School Leadership %'].describe()


# In[ ]:


df.groupby('Borough')['Strong Family-Community Ties %'].describe()


# ### Remarks
# 
# - The estimated school income varies a lot between boroughs
# 
#   *It has been shown that this value is very correlated to the Economic Need Index.*
# 
# - As expected, the ENI also varies a lot
# 
# - The ethnic composition of schools seem to reflect that of their regions
# 
# - The performance across boroughs is very varied. Manhattan performs best
# 
# - Tiny variances in the NYC Framework may indicate areas where PASSNYC might have more ease or difficulty when intervening

# ## Available grades
# 
# These started catching my attention... Let's take a look.

# In[ ]:


plot_clusters(df['Available Grades'], df)


# In[ ]:


_aa = df[df['Available Grades'] == 'elementary(5-7) - middle(8)']
plt.scatter(_aa['Student Attendance Rate'], _aa['Percent of Students Chronically Absent'])


# ### Remarks
# 
# *the KDE for Estimated School Income for middle schools means nothing (NAs don't appear there)*
# 
# - Almost all schools that start at the kindergarten have their 'School Income Estimate'. Other schools almost never do
# - The percentage of 4s from elementary grades look more distributed than the same indicator in middle grades
# - There seems to be a clustering pattern in the attendance of typical middle schools (5,6,7 to 8th grade)

# ## Other kinds of NA
# 
# While we understood the main factor for a school to have a missing 'School Income Estimate Value', there are another NAs. Let's look into them.

# In[ ]:


def get_null_columns(x):
    return tuple([c for c in x[x.isnull()].index if c != 'School Income Estimate'])

def summarize_null_columns(columns):
    return {
        (): 'none',
        ('Average ELA Proficiency', 'Average Math Proficiency'): 'ela/math proficiency',
        ('Student Attendance Rate', 'Percent of Students Chronically Absent'): 'attendance',
        ('Economic Need Index',
          'Student Attendance Rate',
          'Percent of Students Chronically Absent',
          'Rigorous Instruction %',
          'Collaborative Teachers %',
          'Supportive Environment %',
          'Effective School Leadership %',
          'Strong Family-Community Ties %',
          'Trust %',
          'Average ELA Proficiency',
          'Average Math Proficiency'): 'lots of stuff'
    }.get(columns, 'other')

null_columns = df.apply(get_null_columns, axis=1).apply(summarize_null_columns).rename('Null columns')
plot_clusters(null_columns, df)


# ### Remarks
# 
# - Overall, it's difficult to say much because there are few schools.
# - Schools without ELA/Math proficiency teach up until the 4th grade, or, their upper-level students do not register for the state tests
# - Schools without attendance information come mainly from Manhattan and have high-performing students

# # Clustering
# 
# Here I will cluster schools based on different characteristics. The focus, instead of creating a powerful grouping, will be on creating a simple thing that is easy to comprehend.

# ## Group middle schools by Percent of Students Chronically Absent
# 
# *this pattern emerged when splitting schools by the grades available*

# In[ ]:


# check only middle schools

d = df[df['Available Grades'] == 'elementary(5-7) - middle(8)'].copy()
d = d[df['Percent of Students Chronically Absent'].notnull()]  # ignore nulls
d.head()


# In[ ]:


sns.distplot(d['Percent of Students Chronically Absent']);


# And there you are, a seemingly bimodal distribution. Let's use a clustering algorithm to estimate a cutoff point.

# In[ ]:


from sklearn.cluster import KMeans

model = KMeans(n_clusters=2)
model.fit(d[['Percent of Students Chronically Absent']])
model.cluster_centers_


# The cutoff is the middle of the cluster centers:

# In[ ]:


cutoff_point = model.cluster_centers_.mean()
cutoff_point


# ### Plotting

# In[ ]:


clusters = (d['Percent of Students Chronically Absent'] > cutoff_point).rename('Chronic Absent > 21.7%')
plot_clusters(clusters, d)


# ### Remarks
# 
# - The schools lots of with chronic skippers (class-skippers) are mostly present in regions with lots of educationals problems (Bronx and Northeastern Brooklyn)
#   
# - The framework measures look surprisingly close for both categories
#   
#   *this leads me to believe that the framework percentages are relative*
#   
# - The percentage of chronic skippers is a great indicator of the overall health of schools.

# ## Group schools by ethnicity
# 
# Here I will group schools by their ethnic distribution (e.g., their percentage of students in each ethnicity).
# 
# The measure used will be the cosine distance, that means: two points are considered close to one another if the angle between them is small.

# In[ ]:


# separate the dataframe

columns = ['Percent Hispanic',
           'Percent Black',
           'Percent White',
           'Percent Asian']
d = df[columns].copy()
d['Percent Other'] = (1 - d.sum(axis=1)).apply(lambda x: max(0.0, x))
d.head()


# In[ ]:


from sklearn.metrics.pairwise import pairwise_distances

# calculate the distance matrix

distances = pairwise_distances(d, metric='cosine', n_jobs=-1)
distances


# In[ ]:


from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# plot the silhouette score to determine optimal number of clusters

rounds = []
n_clusters_range = range(2, 25)
for n_clusters in n_clusters_range:
    model = AgglomerativeClustering(n_clusters, linkage='average', affinity='precomputed')
    labels = model.fit_predict(distances)    
    score = silhouette_score(distances, labels, metric='precomputed')
    rounds.append([n_clusters, score])
    
rounds_df = pd.DataFrame(rounds, columns=['n_clusters', 'score'])
plt.bar(rounds_df['n_clusters'], rounds_df['score'])
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score");


# In[ ]:


# create the clusters

model = AgglomerativeClustering(n_clusters=4, linkage='average', affinity='precomputed')
labels = model.fit_predict(distances)
clusters = pd.Series(labels, index=d.index, name='Cluster')
print(clusters.value_counts())


# ### Plotting

# In[ ]:


plot_clusters(clusters, df.loc[d.index])


# ### Remarks
# 
# - The map produced very crispy areas. This indicates a nice clustering
# - However, the meaning of clusters is not well defined. For example, Cluster 0 (mostly consisted of asians), has a school which is composed mainly of blacks (though still having asians)
# - Schools basic characteristics are kept the same, except for the school estimated income (which is higher where there are more white and asian students)
# - Students characteristics vary a lot:
#   - Hispanic and black students are overall poorer
#   - They also have worse performance and attendance ratings
#   - Hispanic and asian students have a tendency to be english language learners

# ## Group schools by framework measures
# 
# Here I will group the schools based on the [NYC Framework for Greate Schools][1].
# 
# The distance used is the pearson correlation, useful in grouping entries with a similar "shape" across the compared metrics.
# 
# For example, a school with 'Collaborative Teachers %' slightly higher than 'Rigorous Instruction %' will tend to be grouped with a school that has this same characteristic.
# 
# [1]: https://www.schools.nyc.gov/about-us/vision-and-mission/framework-for-great-schools

# In[ ]:


# separate the DataFrame

columns = [c for c in df.columns if c.endswith('%')]
d = df[columns]
d = d[d.notnull().all(axis=1)]
d.head()


# In[ ]:


from sklearn.metrics.pairwise import pairwise_distances

# create the distance matrix

distances = pairwise_distances(d, metric='correlation', n_jobs=-1)
distances


# In[ ]:


# choose number of clusters based on the silhouette score

rounds = []
n_clusters_range = range(2, 14)
for n_clusters in n_clusters_range:
    model = AgglomerativeClustering(n_clusters, linkage='complete', affinity='precomputed')
    labels = model.fit_predict(distances)    
    score = silhouette_score(distances, labels, metric='precomputed')
    rounds.append([n_clusters, score])
    
rounds_df = pd.DataFrame(rounds, columns=['n_clusters', 'score'])
plt.bar(rounds_df['n_clusters'], rounds_df['score'])
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score");


# In[ ]:


# create clusters

model = AgglomerativeClustering(n_clusters=5, linkage='complete', affinity='precomputed')
labels = model.fit_predict(distances)
clusters = pd.Series(labels, index=d.index, name='Cluster')
print(clusters.value_counts())


# ### Plotting

# In[ ]:


plot_clusters(clusters, df.loc[d.index])


# ### Remarks
# 
# - School characteristics keep the same through clusters, except for Cluster 1, where there are more middle schools.
# 
#   *This may indicate an area that needs improvement (and is relevant because it involves students about to take SHSAT*
#   
# - Cluster 3 presents a visible grouping in the Economic Need Index and Student Attendance Rate (cluster 1 also does, but it is less visible)
# 
# - The student performance across groups is remarkably similar, however, the attendance indicators vary. This goes against our findings that lower attendance is correlated with lower performance. Why does it happen?

# # Work in Progress
# 
# This is a work in progress. Here are some things that will be done later:
# 
# - Create subclusters for NYC Framework metrics
# - Summarize all findings in a concise way
# 
# Cya!
