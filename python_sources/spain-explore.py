import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as dt
import re
import math
import csv
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
import scipy.stats as st
import seaborn as sns
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import PCA as sklearnPCA
from numpy.random import seed
from numpy.random import randn
from scipy.stats import mannwhitneyu

data = pd.read_csv('../input/spain-p1/dfadcb30-fab3-45d0-8cf9-a0f95cd560c8_Data.csv', na_values='..', encoding='utf-8')

data=data.dropna(how='all', axis=0)
data=data.drop(columns=['Series Code', 'Country Name', 'Country Code'])

data.tail()
data.columns
data['Series Name'].unique()


#rename year columns
data.columns =data.columns.str.replace('YR','')
#creating renaming dictionary
d=dict(zip(data.columns[1:], data.columns[1:].str[:4]))
print (d)
#renaming
data = data.rename(columns=d)

data.columns
data['1990'].unique

#convert multiple columns to float
cols = data.columns[2:]
data[cols] = data[cols].apply(pd.to_numeric, errors='coerce')
data['1990'].dtype

#transpose
data.set_index('Series Name',inplace=True)
data1=data.T
data1.columns
data1.reset_index(level=0, inplace=True)
data1 = data1.rename(columns={'index':'year'})
data1['year']=data1['year'].astype('int')
data1=data1[data1.year > 2000]


data1=data1.drop(columns=['Data from database: World Development Indicators', 'Last Updated: 05/21/2018'])


#search for missing values
data1.dtypes
missing_values_count = data1.isnull().sum().nlargest(5)
dr=missing_values_count.index

#drop columns with missing > 10
data1=data1.drop(columns=dr)

measure=data1.columns
measure2=list(reversed(measure))
measure

#creating annula growth variables
data1[['Growth rate GDP','Growth rate International Tourism receipts']]=data1.sort_values(['year'])[['GDP (current US$)','International tourism, receipts (current US$)']].pct_change()
data1['Growth rate Education']=data1.sort_values(['year'])['Educational attainment, at least completed post-secondary, population 25+, total (%) (cumulative)'].pct_change()


#describe growth rates
data1['Growth rate GDP'].describe()
data1['Growth rate GDP'].hist()

#vilualization
sns.set()
sns.set_palette('Set2',3)
sns.set_style("white")

#vizualizing a bunch of measures, just for a quick look
for u,u1 in zip(measure, measure2):
    sns.stripplot(x=u, y=u1, data=data1)
    plt.show()

# extracting the most interesting ones

#tourism
#compare to other countries
all_cont=pd.read_csv('../input/allcountries/ebf83a7a-718d-4793-8f3e-9dcaf2c6bca1_Data.csv', na_values='..', encoding='utf-8')
d1=dict(zip(all_cont.columns[4:], all_cont.columns[4:].str[:4]))
print (d1)
#renaming
all_cont = all_cont.rename(columns=d1)
all_cont=all_cont.drop(columns=['1990', '2000'])


all_cont['Country Code'].unique()
trael=all_cont[(all_cont['Series Name']=='Travel services (% of commercial service exports)')&(all_cont['Country Code'].isin(['ESP', 'FRA', 'CEB', 'ECS']))]
trael.columns

#resaping data for a smooth representation
trael1= trael.melt(id_vars=['Country Name', 'Country Code', 'Series Name', 'Series Code'])
trael1.columns

trael1.variable=trael1.variable.astype('int')
trael1.variable.describe()
trael1 = trael1.rename(columns={'variable':'year', 'value': 'Travel services (% of commercial service exports)'})
trael1.groupby('Country Name')['Travel services (% of commercial service exports)'].describe()
trael1.to_csv('by country comparison.csv',sep=';', encoding='utf-8-sig', index = True)


#% of commercial  service exports in years 2008-2017
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.boxplot(x='Country Name', y='Travel services (% of commercial service exports)', data=trael1, ax=ax)


#Unemployment

sns.set_palette('Set3',6)
sns.jointplot(x='Unemployment, total (% of total labor force) (national estimate)', y='International tourism, receipts (% of total exports)', data=data1, kind="kde")
sns.jointplot(x='Unemployment, total (% of total labor force) (national estimate)', y='International tourism, receipts (% of total exports)', data=data1, kind="reg")
sns.jointplot(x='Unemployment, total (% of total labor force) (national estimate)', y='International tourism, receipts (% of total exports)', data=data1, kind="scatter")

f,ax=plt.subplots(figsize=(5,6))
sns.factorplot(data=data1, y="Unemployment, total (% of total labor force) (national estimate)", x="year", size=8)
sns.factorplot(data=data1, y="International tourism, receipts (% of total exports)", x="year", size=8)

data1['Unemployment, total (% of total labor force) (national estimate)'].describe()
data1['Unemployment, total (% of total labor force) (national estimate)'].kurtosis()

data1['International tourism, receipts (% of total exports)'].describe()
data1['International tourism, receipts (% of total exports)'].kurtosis()


sns.jointplot(x='Unemployment, total (% of total labor force) (national estimate)', y='Travel services (% of commercial service exports)', data=data1, kind="reg")


plt.show()

#GDP
sns.jointplot(x='GDP (current US$)', y='International tourism, receipts (current US$)', data=data1)
sns.jointplot(x='GDP (current US$)', y='International tourism, receipts for travel items (current US$)', data=data1)
sns.jointplot(x='Growth rate GDP', y='Growth rate International Tourism receipts', data=data1, kind="reg")

data1['Growth rate International Tourism receipts'].describe()
data1['Growth rate International Tourism receipts'].hist()
data1['Growth rate International Tourism receipts'].kurtosis()
data1['Growth rate International Tourism receipts'].skew()
data1['Growth rate GDP'].describe()
data1['Growth rate GDP'].kurtosis()
data1['Growth rate GDP'].skew()
data1['Growth rate GDP'].hist()

sns.boxplot(x='Growth rate International Tourism receipts', data=data1)
sns.boxplot(x='Growth rate GDP', data=data1)

plt.show()

#Education
sns.jointplot(x='Educational attainment, at least completed post-secondary, population 25+, total (%) (cumulative)', y='International tourism, receipts (% of total exports)', data=data1,  kind="reg")
sns.jointplot(x='Educational attainment, at least completed post-secondary, population 25+, total (%) (cumulative)', y='Unemployment, total (% of total labor force) (national estimate)', data=data1,  kind="reg")
#pearson coefficient suggest strong negative and statistically significant  collelation between variables. However, in the conditions of limited # of records, we can't make any stable conclusions
sns.jointplot(x='Educational attainment, at least completed post-secondary, population 25+, total (%) (cumulative)', y='Travel services (% of commercial service exports)', data=data1)

sns.jointplot(x='Growth rate Education', y='Growth rate International Tourism receipts', data=data1, kind="reg")
sns.factorplot(data=data1, y="Growth rate Education", x="year", size=8)



nationality=pd.read_csv('../input/nationalyt/guests by nationality.csv')
nationality.columns
nationality=nationality.melt(id_vars=['nationality','measure '])
nationality = nationality.rename(columns={'variable':'MoY', 'measure ':'measure'})
nationality['value']=nationality['value'].str.replace(' ','')

#delete missing values
nationality['value'].replace('', np.nan, inplace=True)
nationality=nationality.dropna(axis=0, how='any')
nationality['value']=nationality['value'].astype('int')
nationality.dtypes

fig, ax = plt.subplots(figsize=(15,15))
sns.boxplot(y='nationality', x='value', hue='measure', data=nationality)

#---------------
#AUTONOMOUS COMMUNITIES

exped=pd.read_csv('../input/spain-redone/10839 (2).csv', decimal=',')
exped.head()
exped.dtypes
exped.columns

exped=exped.rename(columns={' ':'month_year'})

exped['Total expenditure']=exped['Total expenditure'].str.replace(' ','')
exped['Total expenditure']=exped['Total expenditure'].str.replace(',','.')
exped['Total expenditure']=exped['Total expenditure'].astype('float')

exped['Average expenditure per person']=exped['Average expenditure per person'].str.replace(' ','')
exped['Average expenditure per person']=exped['Average expenditure per person'].astype('int')

#renaming categories
exped['region'].unique()
exped['region']=exped['region'].str.replace('Balears, Illes','BALEARS')
exped['region']=exped['region'].str.replace('Canarias','CANARIAS')
exped['region']=exped['region'].str.replace('Madrid, Comunidad de','MADRID')
exped['region']=exped['region'].str.replace('Comunitat Valenciana','COMUNITAT VALENCIANA')
exped['region']=exped['region'].str.replace('Total', 'National average ')

#creating year variable
exped['month_year'].unique()
exped['month_year']=exped['month_year'].str.replace('    ','')
exped['year']=exped.apply(lambda exped: exped['month_year'][:4], axis=1)
exped['year'].unique()
exped['year']=exped['year'].astype('int')
f = {'Average expenditure per person' : 'mean','Average daily expenditure per person.': 'mean','Average duration of the trips' : 'mean', 'Total expenditure':'sum'}
exped_y=exped.groupby(['year', 'region']).agg(f)


#boxplot to explore change between years and regions
sns.boxplot(x='year', y='Average expenditure per person', data=exped)

#add average to the macro table
#avr=exped.groupby(['year'])['Average expenditure per person'].mean()
#data1=data1.merge(avr.to_frame(), left_on='year', right_index=True)

#explore GDP per region
GDP=pd.read_csv('../input/nominal-gdp-per-capita-of-spain-by-regions/GDP_capita_spanish_regions.csv')
GDP= GDP.melt(id_vars=['REGION'])
GDP = GDP.rename(columns={'variable':'year', 'value': 'GDP', 'REGION':'region'})
GDP.dtypes
GDP['year']=GDP['year'].astype('int')

#the expent to which spend effects region GDP
data_AC_y = pd.merge(GDP,exped_y, how='inner', on=['region','year'])
data_AC_y
data_AC_y.columns
data_AC_y.dtypes

sns.violinplot(x='Average daily expenditure per person.',y='GDP', hue='region', data=data_AC_y)
data_AC_y['Growth rate GDP']=data_AC_y.sort_values(['year', 'region'])['GDP'].pct_change()
data_AC_y['Growth rate Total expenditure']=data_AC_y.sort_values(['year', 'region'])['Total expenditure'].pct_change()
sns.jointplot(x='Growth rate Total expenditure',y='Growth rate GDP', data=data_AC_y, kind='reg')

#
AC_room=pd.read_csv('../input/autonomous-community-ds/2057_AC.csv', na_values='.', decimal=',')
AC_room= AC_room.melt(id_vars=['region', 'measure'])
AC_room = AC_room.rename(columns={'variable':'MoY'})
AC_rate=pd.read_csv('../input/autonomous-community-ds/2059_AC.csv', na_values='.', decimal=',')
AC_rate= AC_rate.melt(id_vars=['region', 'measure'])
AC_rate = AC_rate.rename(columns={'variable':'MoY'})
AC_hotel=pd.read_csv('../input/autonomous-community-ds/9667_AC.csv', na_values='.', decimal=',')
AC_hotel= AC_hotel.melt(id_vars=['region', 'measure'])
AC_hotel.columns
AC_hotel['variable'].unique()
AC_hotel['variable']=AC_hotel['variable'].str[:7]
AC_hotel = AC_hotel.rename(columns={'variable':'MoY'})

AC_exped=pd.read_csv('../input/exped-ac/AC_10839.csv', decimal=',')
AC_exped = AC_exped.rename(columns={' ':'MoY'})
AC_exped=AC_exped.melt(id_vars=['region', 'MoY'])
AC_exped.columns
AC_exped.region.unique()
AC_exped = AC_exped.rename(columns={'variable':'measure'})

AC_all=pd.concat([AC_hotel,AC_rate], join="inner", axis=0)
AC_all=pd.concat([AC_all,AC_room], join="inner", axis=0)
AC_all=pd.concat([AC_all,AC_exped], join="inner", axis=0)
AC_all.measure.unique()
AC_all.dtypes
AC_all['value']=AC_all['value'].str.replace(' ','')
AC_all['value']=AC_all['value'].str.replace(',','.')
AC_all['value']=AC_all['value'].astype('float')
AC_all['year']=AC_all['MoY'].str[:4]
AC_all['month']=AC_all['MoY'].str[5:]
AC_all['year']=AC_all['year'].astype('int')
AC_all['month']=AC_all['month'].astype('int')

AC_all.month.unique()

#creating MoM cariables
sns.boxplot(y='region', x='value', data=AC_all[(AC_all.measure == 'Employed personnel, persons        ') & (AC_all.region != 'National Total')])
AC_all['Growth rate MoM TE']=AC_all[(AC_all.measure == 'Total expenditure')].sort_values(['measure', 'region', 'month', 'year'])['value'].pct_change()
AC_all['Growth rate N of Hotels']=AC_all[(AC_all.measure == 'Estimated number of open establishments')].sort_values(['measure', 'region', 'month', 'year'])['value'].pct_change()
AC_all['Growth Rate ADEPP']=AC_all[(AC_all.measure == 'Average daily expenditure per person')].sort_values(['measure', 'region', 'month', 'year'])['value'].pct_change()
AC_all['Growth Rate Empoyment']=AC_all[(AC_all.measure == 'Employed personnel, persons        ')].sort_values(['measure', 'region', 'month', 'year'])['value'].pct_change()

AC_all['Growth Rate ADEPP'].isna().sum()
AC_all['Growth rate MoM TE'].isna().sum()

exped=AC_all[(AC_all.measure == 'Total expenditure')].groupby(['region']).sum()
exped.to_csv('exped.csv',sep=';', encoding='utf-8-sig', index = True)

#calculation month on month growth for each month and region
AC_all[(AC_all.MoY == '2018M03') & (AC_all.measure =='Total expenditure')]
sns.boxplot(y='region', x='Growth rate MoM TE', data=AC_all[(AC_all.measure == 'Total expenditure')])
mean_reg=AC_all.groupby('region')['Growth rate MoM TE'].describe()

#removing outliers
AC_all1=AC_all[np.abs(AC_all['Growth rate MoM TE']-AC_all['Growth rate MoM TE'].mean())<=(3*AC_all['Growth rate MoM TE'].std())] #keep only the ones that are within +3 to -3 standard deviations in the column 'Data'.
AC_all1=AC_all1[~(np.abs(AC_all1['Growth rate MoM TE']-AC_all1['Growth rate MoM TE'].mean())>(3*AC_all1['Growth rate MoM TE'].std()))]
sns.boxplot(y='region', x='Growth rate MoM TE', data=AC_all1[(AC_all1.measure == 'Total expenditure')])
mean_reg=AC_all1[(AC_all1.measure == 'Total expenditure')].groupby('region')['Growth rate MoM TE'].mean()
lest_reg=mean_reg.nlargest(3)
list_reg=lest_reg.index
mean_reg.to_csv('mean.csv',sep=';', encoding='utf-8-sig', index = True)

sns.swarmplot(x="year", y='Growth rate MoM TE', hue="region", data=AC_all1[AC_all1.region.isin(list_reg)])
sns.swarmplot(x="month", y='Growth rate MoM TE', hue="region", data=AC_all1[AC_all1.region.isin(list_reg)])
f, ax = plt.subplots(figsize=(10, 8))
sns.barplot(x="month", y='Growth rate MoM TE', hue="region", data=AC_all1[AC_all1.region.isin(list_reg)])
sns.pointplot(x="month", y='Growth rate MoM TE', hue="region", data=AC_all1[AC_all1.region.isin(list_reg)])

f, ax = plt.subplots(figsize=(15, 8))
sns.pointplot(x="year", y='GDP', hue="region", data=GDP[GDP.region.isin(['CANARIAS','CASTILLA-LA MANCHA','CASTILLA Y LEON','CEUTA','EXTREMADURA', 'National average'])])

#-------------------------------------
#TRIPS
trip_form=pd.read_csv('../input/trip-form/H_10841.csv', decimal=',')
trip_form.columns
trip_form.dtypes
trip_form=trip_form.melt(id_vars=['form of organization of the trip', 'measure'])
trip_form.variable=trip_form.variable.str[:8]
trip_form.variable=trip_form.variable.str[:8]
trip_form = trip_form.rename(columns={'variable':'MoY'})
trip_form['year'] = trip_form.MoY.str[:4]
trip_form['month'] = trip_form.MoY.str[6:]
trip_form.year=trip_form.year.astype('int')
trip_form.month=trip_form.month.str.replace('.','')
trip_form.month=trip_form.month.astype('int')
trip_form.value=trip_form.value.str.replace(' ','')
trip_form.value=trip_form.value.str.replace(',','.')
trip_form.value=trip_form.value.astype('float64')



trip_purp=pd.read_csv('../input/trip-pur/H_10840.csv', decimal=',')
trip_purp=trip_purp.melt(id_vars=['measure', 'Trip purpose'])
trip_purp.columns
trip_purp = trip_purp.rename(columns={'variable':'MoY'})
trip_purp.value=trip_purp.value.str.replace(' ','')
trip_purp.value=trip_purp.value.str.replace(',','.')
trip_purp.value=trip_purp.value.astype('float64')
trip_purp['year'] = trip_purp.MoY.str[:4]
trip_purp.year=trip_purp.year.astype('int')
trip_purp['month'] = trip_purp.MoY.str[6:]
trip_purp.month=trip_purp.month.astype('int')

number=pd.read_csv('../input/business-number-redone/13864 (1).csv')
number['Number of tourists']=number['Number of tourists'].str.replace(' ','')
number['Number of tourists']=number['Number of tourists'].astype('int')
number['year'] = number.MoY.str[:4]
number.year=number.year.astype('int')
number['month'] = number.MoY.str[6:]
number.month=number.month.astype('int')

gr=trip_purp[(trip_purp.measure == 'Average duration of the trips') & (trip_purp['Trip purpose']=='Business')].groupby('month').mean()
gr=gr.reset_index(level='month')
gr['month'].corr(gr['value'])


f, ax = plt.subplots(figsize=(7, 6))
sns.boxplot(x="month", y='value', data=trip_purp[(trip_purp.measure == 'Average duration of the trips') & (trip_purp['Trip purpose']=='Business')], whis=np.inf, palette="PuBuGn_d")
# showing number of tourists
sns.swarmplot(x="month", y="Number of tourists", data=number, size=7, color=".3", linewidth=0)
ax.xaxis.grid(True)
ax.set(ylabel="")
sns.despine(trim=True, left=True)


sns.boxplot(x="month", y='value', data=trip_purp[(trip_purp.measure == 'Average duration of the trips') & (trip_purp['Trip purpose']=='Business')])


sns.boxplot(x="year", y='value', hue="Trip purpose", data=trip_purp[trip_purp.measure == 'Average duration of the trips'])
sns.boxplot(x="year", y='value', hue="Trip purpose", data=trip_purp[trip_purp.measure == 'Average duration of the trips'])
sns.barplot(x="year", y='value', hue="Trip purpose", data=trip_purp[trip_purp.measure == 'Total expenditure'])
trip_purp[trip_purp.measure == 'Average duration of the trips'].groupby('Trip purpose').mean()

trip_purp[trip_purp.measure == 'Average daily expenditure per person.'].groupby('Trip purpose').mean()
trip_purp[trip_purp.measure == 'Average expenditure per person'].groupby('Trip purpose').mean()

data1=trip_purp[(trip_purp.measure == 'Average expenditure per person') & (trip_purp['Trip purpose'] == 'Business')].groupby(['month', 'year']).mean()
data2=trip_purp[(trip_purp.measure == 'Average expenditure per person') & (trip_purp['Trip purpose'] != 'Business')].groupby(['month', 'year']).mean()
data2.columns
stat, p = mannwhitneyu(data1, data2)
print('Statistics=%.3f, p=%.3f' % (stat, p))



