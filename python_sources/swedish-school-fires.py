# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 17:58:12 2016

@author: Richard
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from pylab import *
import seaborn as sns
# Any results you write to the current directory are saved as output.
#muni_data = pd.read_csv('../input/municipality_indicators.csv')
fire_data = pd.read_csv('../input/school_fire_cases_1998_2014.csv')

#print(muni_data.head())
#print(fire_data.head())

#print(muni_data['municipality_name'].value_counts())

#print(fire_data['Municipality'].value_counts())

#muni_data['Municipality'] = muni_data['municipality_name']
#data = pd.merge(fire_data, muni_data, how = 'left', on = 'Municipality')
#data = data.drop(['municipality_id', 'municipality_name'], axis = 1)
data = fire_data
print(data.head())

muni_group = data.groupby('Municipality')
names = []
cases = []
avg_cases = []
avg_pop = []
yearly_cases = []
yr = []
muni_yr = []
for muni in muni_group:
    muni = muni[1]
    muni_name = str(muni['Municipality'].unique()).replace('[','').replace(']','').replace("'","")
    num_cases = muni['Cases'].sum()
    mean_cases = muni['Cases'].mean()
    pop = muni['Population'].mean()
    
    names.append(muni_name)
    cases.append(num_cases)
    avg_cases.append(mean_cases)
    avg_pop.append(pop)
    
    muni_year = muni.groupby('Year')
    for year in muni_year:
        year = year[1]
        yr_cases = year['Cases'].sum()
        yearly_cases.append(yr_cases)
        yr.append(str(year['Year'].unique()).replace('[','').replace(']','').replace("'",""))
        muni_yr.append(muni_name)

muni_df = pd.DataFrame([names, cases, avg_cases, avg_pop]).transpose()
muni_df.columns = ['Municipality', 'Cases', 'Avg_cases', 'Avg_pop']
muni_df = muni_df.sort('Cases')
names = muni_df['Municipality'].apply(lambda x: u'%s' %x)
cases = muni_df['Cases']
avg_cases = muni_df['Avg_cases']
avg_pop = muni_df['Avg_pop']

tempx = range(0,len(names))
cla()
clf()
plt.bar(tempx[-10:], cases[-10:])
plt.xticks(tempx[-10:], names[-10:], rotation = 'vertical')
plt.title('Top 10 Total Arson Cases by Municipality')
plt.tight_layout()
fileName = 'top_10_fires_sweden'
plt.savefig(fileName, type = 'png')

cla()
clf()
plt.bar(tempx[-10:], avg_cases[-10:])
plt.xticks(tempx[-10:], names[-10:], rotation = 'vertical')
plt.title('Top 10 Average Arson Cases per Year by Municipality')
plt.tight_layout()
fileName = 'top_10_avg_fires_sweden'
plt.savefig(fileName, type = 'png')

cla()
clf()
plt.scatter(avg_pop, cases)
plt.title('School Fires vs Average Population')
plt.tight_layout()
fileName = 'population_vs_fires_sweden'
plt.savefig(fileName, type = 'png')


top10_muni = names[-10:]
tot_avg_cases = data['Cases'].mean()

#Color scheme
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
colors = tableau20[0:10]
for i in range(10):    
    r, g, b = tableau20[i]    
    colors[i] = (r / 255., g / 255., b / 255.)  

muni_yr_cases = pd.DataFrame([muni_yr, yr, yearly_cases]).transpose()
muni_yr_cases.columns = ['Municipality', 'Year', 'Cases']
muni_yr_cases = muni_yr_cases[muni_yr_cases['Municipality'].isin(top10_muni)]
group_muni_yr = muni_yr_cases.groupby('Municipality')
cla()
clf()
count = 0
for df in group_muni_yr:
    df = df[1]
    name = str(df['Municipality'].unique()).replace('[','').replace("'","").replace(']','')
    mean_cases = df['Cases'].mean()
    case_std = df['Cases'].std()
    t_val = mean_cases / case_std/ float(len(df['Cases']))
    if mean_cases > tot_avg_cases:
        plt.plot(df['Year'], df['Cases'], color = colors[count], label = name)
        #sns.distplot(df['Cases'],hist = False, color = colors[count],label = name)
        count = count + 1
plt.legend()
plt.title("School Fire Cases of 10 Most Affected Municipalities")
plt.tight_layout()
fileName = 'fires_years'
plt.savefig(fileName, type = 'png')
    

year_groups = data.groupby('Year')
year_cases = []
years = []
for year in year_groups:
    year = year[1]
    cases = year['Cases'].sum()
    yr = str(year['Year'].unique()).replace('[','').replace(']','').replace("'","")
    year_cases.append(cases)
    years.append(yr)
cla()
clf()
plt.plot(years, year_cases)
#plt.xticks(range(len(years)), years)
plt.title("Yearly School Fires in Sweden")
plt.tight_layout()
fileName = 'yearly_fires_sweden'
plt.savefig(fileName, type = 'png')