#!/usr/bin/env python
# coding: utf-8

# # Summary Report - Visual Exploration of Data on Covid-19 Outbreak, Worldwide. 
# 
# The data was scraped from 'worldometer' web site at https://www.worldometers.info/coronavirus/ and the analysis was carried out using 'Python' programming language and various related libraries.
# 
# The worldometer web site provides the data more on cumulative basis and there is no accumulated data on daily basis. This report and effort includes the process of gathering daily data.
# 
# The report's focus is more on data extraction, pre-processing and visualisation than statistical analysis or predictive modelling (these are expected to be carried out later along with the analysis of daily data).
# 
# Hope you will find the report interesting..
# 
# Thank you!!!
# 
# **Srinivas**

# First, we start with the loading the required packages.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import seaborn as sns
from pylab import rcParams
import re
get_ipython().run_line_magic('matplotlib', 'inline')
pd.options.display.float_format = '{:.2f}'.format


# Then we access the website link, create open link and read the web page content. 

# In[ ]:


req = Request('https://www.worldometers.info/coronavirus/', headers={'User-Agent': 'Firefox/75.0'})
webpage = re.sub(r'<.*?>', lambda g: g.group(0).upper(), urlopen(req).read().decode('utf-8') )


# In[ ]:


#display(webpage)#printing shows the contents of webpage


# Now, let us access the tables in the webpage.

# In[ ]:


tables = pd.read_html(webpage)
#display(tables) # we can print tables


# We can notice in the website, the default is a table represented by 'Now' button, which is updated throughout the day. So, it is better we capture, yesterday's table ('yesterday' button on web page) for our analysis purpose. This table corresponds index '1' in our list of tables captured in the previous step. So let us access and capture it.

# In[ ]:


df = tables[1]


# Now, we are at the right place and got the required table as Pandas dataframe. It is time to have a look at the dataframe and do some preprocessing befor we start visually anlysing the data.

# In[ ]:


df.info()
display(df.columns.tolist())


# In[ ]:


df.head(10)


# In[ ]:


df.tail(10)


# We can see some of our data is not in its right format. 'NewCases' and 'NewDeaths' are in Object format, while they shall be in integer/ float format. The column 'Tot cases/ 1M pop' has some hex characters. We can also see some of the column names contain 'comma' and also columns, NewCases and Newdeaths contain 'comma' and '+' signs.
# 
# We can observe some of the columns do not have data, represented by NANs. We can safely assume this data as 'Zero' as no such cases would have been reported.
# 
# Let us correct the colum names and values first.

# In[ ]:


# ignore this block.. for learning purpose
# df.columns.tolist() 
# display([(i, hex(ord(i))) for i in df.columns[13]])


# In[ ]:


df = df.rename(columns={'Country,Other': 'Country_or_Other','Serious,Critical': 'Serious_or_Critical','Tot\xa0Cases/1M pop':'Cases_per_1M_pop', 'Tests/  1M pop': 'Tests_per_1M_pop','Deaths/1M pop':'Deaths_per_1M_pop','Tests/ 1M pop':'Tests_per_1M_pop'})
df['NewCases'] = df['NewCases'].str.replace(',','')
df['NewDeaths'] = df['NewDeaths'].str.replace(',','')
# df['NewRecovered'] = df['NewRecovered'].str.replace('+','')
# df['NewRecovered'] = df['NewRecovered'].str.replace(',','')


# In[ ]:


df['NewCases'] = pd.to_numeric(df['NewCases']).fillna(0)
df['NewCases'] = df['NewCases'].astype(np.int64)
df['NewDeaths'] = pd.to_numeric(df['NewDeaths']).fillna(0)
df['NewDeaths'] = df['NewDeaths'].astype(np.int64)
# df['NewRecovered'] = pd.to_numeric(df['NewRecovered']).fillna(0)
# df['NewRecovered'] = df['NewRecovered'].astype(np.int64)
df['ActiveCases'] = df['ActiveCases'].fillna(0).astype(np.int64)
df['Serious_or_Critical'] = df['Serious_or_Critical'].fillna(0).astype(np.int64)
df['TotalDeaths'] = df['TotalDeaths'].fillna(0).astype(np.int64)
df['TotalRecovered'] = df['TotalRecovered'].fillna(0).astype(np.int64)
df['TotalTests'] = df['TotalTests'].fillna(0).astype(np.int64)
df.head(10)


# Let us have relook at the dataframe infrmation. We can see all the columns names and type is converted into what we need.

# In[ ]:


df.info()
df.index


# We don't want the first and last rows of our dataframe as they correspond to the totals consolidated across. So, we will drop them.
# 
# We will also create a new column 'Dead_to_Recovered', which is a percentage number of 'TotalDeaths' to 'TotalRecovered'. We will also drop 'NewCases' and 'NewDeaths' from the main dataframe as they are daily data and all other columns are cumulative data.
# 
# Then, we create a seperate dataframe to gather and keep a track of daily data by extracting data from 'NewCases' and 'NewDeaths' columns.

# In[ ]:


df1 = df.drop(df.index[0:8]).drop(df.index[-8:])
df2 = df1[['Country_or_Other','NewCases', 'NewDeaths']]
display(df2.head())
df2.tail()


# In[ ]:


cum_data = df1.drop(columns=['NewCases','NewDeaths'])
cum_data['Dead_to_Recovered'] = 100*cum_data['TotalDeaths']/cum_data['TotalRecovered']
cum_data = cum_data.sort_values('TotalCases', ascending=False)
cum_data['TotalCases_Percent'] = 100*cum_data['TotalCases']/cum_data['TotalCases'].sum()
cum_data['TotalDeaths_Percent'] = 100*cum_data['TotalDeaths']/cum_data['TotalDeaths'].sum()
cum_data['TotalRecovered_Percent'] = 100*cum_data['TotalRecovered']/cum_data['TotalRecovered'].sum()
cum_data['TotalActive_Percent'] = 100*cum_data['ActiveCases']/cum_data['ActiveCases'].sum()
cum_data['TotalTests_Percent'] = 100*cum_data['TotalTests']/cum_data['TotalTests'].sum()
display(cum_data.columns)
cum_data.head()


# Let us check the totals of various categry of cases.

# In[ ]:


cum_data[['TotalCases','TotalDeaths','TotalRecovered', 'ActiveCases','Serious_or_Critical','TotalTests']].sum()


# Next, we create a date column in our daily data dataframe and convert it into right date format.

# In[ ]:


df4 = df2.copy()
df4['Date_Time'] = pd.to_datetime('today')+ pd.DateOffset(-1)#for yesterday
df4.head()


# After this, we will write this daily data to our local disk in .csv format and keep that file updated on daily basis by appending everyday updates. This file creation is done once only, hence commented out from getting executed everytime.
# 
# After the creation, we access it for updation purpose and we update the file based on a condition that the newly accessed updates are new by a day at least.

# In[ ]:


#df4.to_csv('xxxxxxxxxxxxxxxxxxxxxxxxxxx/worldometers_covid19_uptoMay2nd2020.csv', sep =',', index=False)


# In[ ]:


df5 = pd.read_csv('https://raw.githubusercontent.com/valmetisrinivas/Covid19_Worldometers/master/worldometers_covid19_uptoMay2nd2020.csv')
df5['Date_Time'] = pd.to_datetime(df5['Date_Time'])
df5.head()


# The below code updates the file based on the condition that we created (happens only if updates are new by a day).

# In[ ]:


if df5['Date_Time'].dt.date.max() < df4['Date_Time'].dt.date.min():
    daily_data = df4.append(df5)
    daily_data.to_csv('xxxxxxxxxxxxxxxxxx/worldometers_covid19_uptoMay2nd2020.csv', sep =',', index=False)
else:
    daily_data = df5.copy()

daily_data['Date'] = pd.to_datetime(daily_data['Date_Time'].dt.date)
display(daily_data.shape)
daily_data['NewCases'] = daily_data['NewCases'].fillna(0).astype(np.int64)
daily_data['NewDeaths'] = pd.to_numeric(daily_data['NewDeaths']).fillna(0).astype(np.int64)
daily_data['NewRecovered'] = pd.to_numeric(daily_data['NewRecovered']).fillna(0).astype(np.int64)


# Below are top 30 countries with daily new cases arranged by latest date and the new cases in desecending order.

# In[ ]:


daily_data.sort_values(['Date', 'NewCases'], ascending=False).drop('Date_Time', axis=1).head(30)


# Below are top 30 countries with daily new deaths arranged by latest date and the new deaths in desecending order.

# In[ ]:


daily_data.sort_values(['Date', 'NewDeaths'], ascending=False).drop('Date_Time', axis=1).head(30)


# After, that we create a list of top 30 worst hit countries (based on confirmed cases) for our analysis purposes. Let us see which are those countries.

# In[ ]:


select_countries = cum_data['Country_or_Other'][0:30]
select_countries


# Then we create seperate dataframes for cumulative and daily data for only these select countries from our previously created cumulative and daily dataframes, for our analysis. The, let us examine these newly created dataframes.

# In[ ]:


select_cum = cum_data[cum_data['Country_or_Other'].isin(select_countries)]
display(select_cum.shape)
select_daily = daily_data[daily_data['Country_or_Other'].isin(select_countries)].drop(columns = 'Date_Time')
display(select_daily.shape)
display(select_cum.head(2))
display(select_daily.head(2))


# Let us have a look at the total number of confirmed cases and deaths.

# In[ ]:


select_cum[['Country_or_Other','TotalCases','TotalDeaths']]


# Let us also look at the percentages as part of global totals.

# In[ ]:


select_cum_percents=select_cum[['Country_or_Other','TotalCases_Percent','TotalDeaths_Percent','TotalRecovered_Percent','TotalActive_Percent','TotalTests_Percent']]
select_cum_percents


# We arrange this data in atidy format for plotting purposes.

# In[ ]:


sps=pd.melt(select_cum_percents, id_vars='Country_or_Other',value_name='Percentage', var_name='Type')
sps['Type'] = sps['Type'].str.replace("_Percent","")
display(sps.head())
display(sps[sps['Type']=='TotalDeaths'].sort_values('Percentage', ascending=False).head())
display(sps[sps['Type']=='TotalRecovered'].sort_values('Percentage', ascending=False).head())


# Now, we start visualisation of our data. First, let us the top 30 affected countries in terms of the perecntages of the total global figures for total affected cases, deaths, receovered, active and tests in each country.

# In[ ]:


sns.set()
c=sns.catplot(data=sps,x='Type',y='Percentage',col='Country_or_Other', kind='bar', col_wrap=6)
for axes in c.axes.flat:
    axes.set_xticklabels(axes.get_xticklabels(), rotation=45, horizontalalignment='right', size=20)
c.set_titles(size=20)
c.fig.suptitle('Percentage contribution for various case type global numbers - top 30 affected countries', y=1.02, size=25)
plt.show()


# We can see that nearly 28 to 29% of all global confirmed cases and global deaths are in USA, followed by 8-9% of confirmed cases and deaths are in Brazil.

# In the following first 2 plots, we will see the total numbers, averages (with standard deviations), all visually, across various categories for the top 30 infected countries.

# In[ ]:


rcParams['figure.figsize'] = 15, 5
fig, ax = plt.subplots()

ax.get_yaxis().get_major_formatter().set_scientific(False)

# Add a bar for the total confimred cases column 
ax.bar("Confimred", cum_data['TotalCases'].sum())
plt.text(-.1, cum_data['TotalCases'].sum() + 50000, str(cum_data['TotalCases'].sum()),fontweight='bold')

# Add a bar for the total active cases column 
ax.bar("ActiveCases", cum_data['ActiveCases'].sum())
plt.text(-.1+1, cum_data['ActiveCases'].sum() + 50000, str(cum_data['ActiveCases'].sum()),fontweight='bold')

# Add a bar for the total recovered cases column 
ax.bar("Recovered", cum_data['TotalRecovered'].sum())
plt.text(-.1+2, cum_data['TotalRecovered'].sum() + 50000, str(cum_data['TotalRecovered'].sum()),fontweight='bold')

# Add a bar for the total deaths column 
ax.bar("Deaths", cum_data['TotalDeaths'].sum())
plt.text(-.1+3, cum_data['TotalDeaths'].sum() + 50000, str(cum_data['TotalDeaths'].sum()),fontweight='bold')

# Add a bar for the total critical cases column
ax.bar("Serious_or_Critical", cum_data['Serious_or_Critical'].sum())
plt.text(-.1+4, cum_data['Serious_or_Critical'].sum() + 50000, str(cum_data['Serious_or_Critical'].sum()),fontweight='bold')

# Label the y-axis
ax.set_ylabel("Total Numbers")

# Plot title
plt.title('Total numbers across the world')

plt.show()


# In[ ]:


fig, ax = plt.subplots()

# Add a bar for the total confimred cases column with mean/std
ax.bar("Confimred", cum_data['TotalCases'].mean(), yerr=cum_data['TotalCases'].std())

# Add a bar for the total active cases column with mean/std
ax.bar("ActiveCases", cum_data['ActiveCases'].mean(), yerr=cum_data['ActiveCases'].std())

# Add a bar for the total recovered cases column with mean/std
ax.bar("Recovered", cum_data['TotalRecovered'].mean(), yerr=cum_data['TotalRecovered'].std())

# Add a bar for the total deaths column with mean/std
ax.bar("Deaths", cum_data['TotalDeaths'].mean(), yerr=cum_data['TotalDeaths'].std())

# Add a bar for the total critical cases column with mean/std
ax.bar("Serious_or_Critical", cum_data['Serious_or_Critical'].mean(), yerr=cum_data['Serious_or_Critical'].std())

# Label the y-axis
ax.set_ylabel("Numbers")

# Plot title
plt.title('Average numbers with corresponding standard deviation')

plt.show()


# Similarly, we see the how the numbers for test conducted by nations vary across the world. We will first see the numbers for top 50 countries and then the total and average(with standard deviation) for all countries. 

# In[ ]:


fig, ax = plt.subplots()
rcParams['figure.figsize'] = 15, 5

test_data = cum_data.sort_values('Tests_per_1M_pop', ascending=False)
test_data = test_data.head(50).set_index('Country_or_Other').sort_values('Tests_per_1M_pop', ascending=False).fillna(0)

# Plot a bar-chart of tests conducted per million people as a function of country
ax.bar(test_data.index,test_data['Tests_per_1M_pop'])

# Set the x-axis tick labels to the country names
ax.set_xticklabels(test_data.index, rotation = 90)

# Set the y-axis label
ax.set_ylabel("Tests Conducted/ Million")

# Plot title
plt.title('Total tests conducted - top 50 countries')

plt.show()


# Here, interestingly we see apart from small countries like Faroe islands and Iceland, rich gulf nations like UAE and Bahrain in the top leading the number of tests conducted per million population.

# In[ ]:


fig, ax = plt.subplots()
ax.get_yaxis().get_major_formatter().set_scientific(False)

# Add a bar for the total tests conducted
ax.bar("Total", cum_data['TotalTests'].sum())

# Add a bar for the tests conducted column with mean/std
ax.bar("Average & Std deviation", cum_data['TotalTests'].mean(), yerr=cum_data['TotalTests'].std())

# Label the y-axis
ax.set_ylabel("Total Numbers")

# Plot title
plt.title('Total and average (with SD) number of tests conducted across the world')

plt.show()


# Then, let us see which are the 30 worst hit countries in terms of confirmed Covid cases.

# In[ ]:


select_cum1 = select_cum.sort_values('TotalCases', ascending=False).set_index('Country_or_Other').fillna(0)

rcParams['figure.figsize'] = 15, 5
fig, ax = plt.subplots()

# Plot a bar-chart of total confirmed cases as a function of country
ax.bar(select_cum1.index,select_cum1['TotalCases'])

# Set the x-axis tick labels to the country names
ax.set_xticklabels(select_cum1.index, rotation = 90)

# Set the y-axis label
ax.set_ylabel("Total Confirmed Cases")

# Plot title
plt.title('Total confirmed cases - top 30 hit countries')

plt.show()


# We see USA outweighs all other countries by a significant number and dominates the plot, thus is the worst hit country in terms of absolute numbers. If we look at the below boxplot which compares the distribution of actives case, deaths and recovered cases, we can clearly see the single, far out outlier in active cases, which is USA, confirming how badly it is hit.

# In[ ]:


fig, ax = plt.subplots()
rcParams['figure.figsize'] = 15, 5

# Plot a histogram of "Weight" for mens_rowing
ax.boxplot([select_cum1['ActiveCases'],select_cum1['TotalDeaths'],select_cum1['TotalRecovered']])

ax.set_ylabel("Number of cases")
# Add x-axis tick labels:
ax.set_xticklabels(['Active Cases', 'Total Deaths','Total Recovered'])

# Plot title
plt.title('Distribution of various category of cases - top 30 hit countries')

plt.show()


# Then we look at the total deaths oocured so far.

# In[ ]:


fig, ax = plt.subplots()
rcParams['figure.figsize'] = 15, 5
select_cum1 = select_cum1.sort_values('TotalDeaths', ascending=False).fillna(0)

# Plot a bar-chart of total deaths as a function of country
ax.bar(select_cum1.index,select_cum1['TotalDeaths'])

# Set the x-axis tick labels to the country names
ax.set_xticklabels(select_cum1.index, rotation = 90)

# Set the y-axis label
ax.set_ylabel("Total Deaths")

# Plot title
plt.title('Total deaths - top 30 hit countries')

plt.show()


# Again, as expected USA is worst in terms of absolute numbers. Then we see at the number of cases recovered so far from Covid-19.

# In[ ]:


fig, ax = plt.subplots()
rcParams['figure.figsize'] = 15, 5
select_cum1 = select_cum1.sort_values('TotalRecovered', ascending=False).fillna(0)

# Plot a bar-chart of total recovered cases as a function of country
ax.bar(select_cum1.index,select_cum1['TotalRecovered'])

# Set the x-axis tick labels to the country names
ax.set_xticklabels(select_cum1.index, rotation = 90)

# Set the y-axis label
ax.set_ylabel("Total Recovered Cases")

# Plot title
plt.title('Total recovered cases - top 30 hit countries')

plt.show()


# Though, as one can guess USA tops the list, other countries are not very far behind. This shows that compared to USA, other countries are doing better in terms of recovery. However, we can notice that UK and Netherlands did not report recovered numbers yet, hence do not give an accurate picture for these two countries.
# 
# Apart from knowing confirmed cases, deaths and recoveries, it is important to know, how many tests countries have been conducted testing people for Covid-19 disease.
# 
# The below boxplot shows that the number of tests conducted was more spread and varied across nations than the number of confirmed cases. Besides, there are a few large outliers, indicating that some countries have conducted very extensive testing.

# In[ ]:


fig, ax = plt.subplots()
rcParams['figure.figsize'] = 15, 5
ax.get_yaxis().get_major_formatter().set_scientific(False)

# Plot a histogram of "Weight" for mens_rowing
ax.boxplot([select_cum1['TotalTests'],select_cum1['TotalCases']])

ax.set_ylabel("Number")
# Add x-axis tick labels:
ax.set_xticklabels(['Total Tests', 'Total Confirmed'])

# Plot title
plt.title('Distribution of tests conducted vs confirmed cases - top 30 hit countries')

plt.show()


# Let us look at which of our select countries have conducted how many tests per million of their population.

# In[ ]:


fig, ax = plt.subplots()
rcParams['figure.figsize'] = 15, 5

select_cum1 = select_cum1.sort_values('Tests_per_1M_pop', ascending=False).fillna(0)

# Plot a bar-chart of test conducted per a million of population as a function of country
ax.bar(select_cum1.index,select_cum1['Tests_per_1M_pop'])

# Set the x-axis tick labels to the country names
ax.set_xticklabels(select_cum1.index, rotation = 90)

# Set the y-axis label
ax.set_ylabel("Tests Conducted/ Million")

# Plot title
plt.title('Tests conducted per million - top 30 hit countries')

plt.show()


# The above picture shows Israel, Portugal and Spain are leading in terms of tests conducted per Million of people. We can notice that a small gulf country, UAE had spent a lot conducting about 120000 test per million of people and leading the pack compared to a country like India which has a population of more than 1.3 billion but conducted hardly any tests.
# 
# Then we see how the select countries have fared in terms of confirmed cases, test conducted and deaths.

# In[ ]:


fig, ax = plt.subplots()
rcParams['figure.figsize'] = 15, 5
select_cum1 = select_cum1.sort_values('Tests_per_1M_pop', ascending=False).fillna(0)

# Plot a bar-chart for different parameters as a function of country
ax.bar(select_cum1.index,select_cum1['Tests_per_1M_pop'],label='Tests Conducted')
ax.bar(select_cum1.index,select_cum1['Cases_per_1M_pop'],bottom=select_cum1['Tests_per_1M_pop'],label='Confirmed')
ax.bar(select_cum1.index,select_cum1['Deaths_per_1M_pop'],bottom= select_cum1['Tests_per_1M_pop']+select_cum1['Cases_per_1M_pop'],label='Dead')

# Set the x-axis tick labels to the country names
ax.set_xticklabels(select_cum1.index, rotation = 90)

# Set the y-axis label
ax.set_ylabel("Number of cases")

# Plot title
plt.title('Total per Million Population')

plt.legend()

plt.show()


# From the above, we see UAE is doing a great job testing (high), containing (confirmed not so high ) and very low deaths. Doing lots of tests, containing confirmed cases and maintaining low deahs is a best metric to evaluate a country's performance in the fight against Covid-19. UAE is doing the best job here.
# 
# So far we have looked at visualisations based on absolute numbers and per Million numbers. Now, let us see how our select countries are doing in terms of confirmed cases/ Million vis-a-vis deaths/Million. This would give us the actual severity of hit.

# In[ ]:


rcParams['figure.figsize'] = 15, 10
fig, ax = plt.subplots()

x= select_cum1['Deaths_per_1M_pop']
y= select_cum1['Cases_per_1M_pop']
jittered_y = y + (y*.1) * np.random.rand(len(y)) -0.05
jittered_x = x + (x*.1) * np.random.rand(len(x)) -0.05

# Add data: deaths per million, cases per million to data with tests per million as color
scatter=ax.scatter(jittered_x, jittered_y, c=select_cum1['Tests_per_1M_pop'], s=select_cum1['TotalRecovered']/select_cum1['TotalDeaths'],  cmap='Paired')

# Set the x-axis label to cnfirmed cases per Million
ax.set_ylabel('Confirmed cases / Million (log scale)')

# Set the y-axis label to Deaths per Million
ax.set_xlabel('Deaths/ Million')
for i, txt in enumerate(select_cum1.index):
    ax.annotate(txt, (jittered_x[i],jittered_y[i]))
plt.title('Severity of Covid-19 Impact - top 30 hit countries')
plt.yscale('log')
legend1 = ax.legend(*scatter.legend_elements(), loc="lower right", title="Tests_per_1Million")
ax.add_artist(legend1)
handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)
legend2 = ax.legend(handles, labels, loc="lower center", title="Recovered_to_Dead")
plt.show()


# From the above plot, we can make out that, in reality, Belgium is the worst hit country; the severity of hit for these countries is way above USA. This is beacuse of the low population of hese countries but high number of confirmed cases and deaths. We can also see Qatar is doing great to keep the death rate low given the exteremely high number of confirmed cases per million.

# There is one other parameter that we can look at, which measures the immunity levels of people or the effective treatment that infected people are receiving. This can be measured by looking at how many people died when compared to how many people recovered. 

# In[ ]:


select_cum1 = select_cum.set_index('Country_or_Other').sort_values('Dead_to_Recovered',ascending=False)

rcParams['figure.figsize'] = 15, 5

fig, ax = plt.subplots()

# Plot a bar-chart of dead to recovered as a function of country
ax.bar(select_cum1.index,select_cum1['Dead_to_Recovered'])

# Set the x-axis tick labels to the country names
ax.set_xticklabels(select_cum1.index, rotation = 90)

# Set the y-axis label
ax.set_ylabel("% dead against recovered")

# Plot title

plt.title('Number people dead for 100 people recovered - top 30 hit countries')
plt.show()


# As we can see from the plot, most of the European and American countries are doing bad on recovery. As mentioned earlier, we can also notice that UK and Netherlands did not report recovered numbers yet.

# Now, let us do the analysis of daily data for all the countries and our select top 30 major hit countries. Let us first calculate the daily sums for all countries. Then we will also calculate the cumulative sums for both categories. 

# In[ ]:


daily_data1 = daily_data.set_index('Date').fillna(0).groupby('Date').sum()
display(daily_data1.tail())
daily_data2 = daily_data1.cumsum()
display(daily_data2.tail())


# Since, we do not have daily data until 3rd May, 2020, let us calculate the cumulative new cases and new deaths until the same period, to use in our cumulative plots.

# Then, let us develop a function that writes plots a timeseries plot. Since the new cases and new deaths vary a lot in terms of value scales, we will develop the function to have two axes for each of them.

# In[ ]:


# Define a function called plot_timeseries
def plot_timeseries(axes, x, y, color, xlabel, ylabel):

  # Plot the inputs x,y in the provided color
  axes.plot(x, y, color=color)

  # Set the x-axis label
  axes.set_xlabel(xlabel)

  # Set the y-axis label
  axes.set_ylabel(ylabel, color=color)

  # Set the colors tick params for y-axis
  axes.tick_params('y', colors=color)


# Now, let us plot the daily total numbers for both newcases and new deaths.

# In[ ]:


fig, ax = plt.subplots()
# Plot the new daily cases time-series in blue
plot_timeseries(ax, daily_data1.index, daily_data1['NewCases'], "blue", "Date" , "Number of confirmed cases")
plt.scatter(daily_data1.index, daily_data1['NewCases'], color='b')

# Create a twin Axes object that shares the x-axis
ax2 = ax.twinx()

# Plot the new daily deaths data in red
plot_timeseries(ax2, daily_data1.index, daily_data1['NewDeaths'], "red", "Date" , "Number of deaths")
plt.scatter(daily_data1.index, daily_data1['NewDeaths'], color='r')

plt.title('Daily confirmed cases & deaths')
plt.show()


# In[ ]:


fig, ax = plt.subplots()

# Plot the new cumulative cases time-series in blue
plot_timeseries(ax, daily_data2.index, daily_data2['NewCases']+3559352, "blue", "Date" , "Cumulative no. confirmed of cases")

# Create a twin Axes object that shares the x-axis
ax2 = ax.twinx()

# Plot the new cumulative deaths data in red
plot_timeseries(ax2, daily_data2.index, daily_data2['NewDeaths']+248525, "red", "Date" , "Cumulative no. of deaths")
plt.title('Cumulative confirmed cases & deaths')
plt.show()


# In[ ]:


fig, ax = plt.subplots()
# Create a twin Axes object that shares the x-axis
ax2 = ax.twinx()

# Plot the new cumulative cases time-series in green
plot_timeseries(ax, daily_data2.index, daily_data2['NewCases']+3559352, "green", "Date" , "Cumulative no. confirmed of cases")

# Plot the new cumulative deaths data in green
plot_timeseries(ax2, daily_data2.index, daily_data2['NewDeaths']+248525, "orange", "Date" , "Cumulative no. of deaths")

# Plot the new daily cases time-series in blue
plot_timeseries(ax, daily_data1.index, daily_data1['NewCases'], "blue", "Date" , "Confirmed cases")

# Plot the new daily deaths data in red
plot_timeseries(ax2, daily_data1.index, daily_data1['NewDeaths'], "red", "Date" , "Deaths")

plt.suptitle('Daily confirmed cases (Blue) & daily deaths (Red)')
plt.title('Cumulative confirmed cases (in Green) and deaths (in Orange)')

plt.show()


# Now, let us look at the daily confirmed cases and deaths reported in our select top 30 hit countries.

# In[ ]:


fig, ax = plt.subplots()
confirmed = select_daily.fillna(0).set_index(['Date', 'Country_or_Other']).NewCases
confirmed = confirmed.unstack()
ax.plot(confirmed, marker='*')
ax.set_ylabel('Daily new confirmed cases')
plt.legend(confirmed.columns,prop={'size': 10}, loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=8)
plt.title('Daily confirmed cases - Top 30 hit nations')
plt.show()


# The plot indicates USA still maintains the record of maximum number of confirmed cases reported everyday. Then we will see if USA is maintaining that record on the number of deaths reported everyday.

# In[ ]:


fig, ax = plt.subplots()
deaths = select_daily.fillna(0).set_index(['Date', 'Country_or_Other']).NewDeaths
deaths = deaths.unstack()
ax.plot(deaths, marker='*')
ax.set_ylabel('Daily new deaths')
plt.legend(deaths.columns,prop={'size': 10}, loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=8)
plt.title('Daily deaths - Top 30 hit nations')
plt.show()


# Unfortunately, Brazil seems to have taken over from USA the dubious distinction of recording maximum deaths too, over the past few days. Now, let us look at the average number of confirmed cases and also deaths per country across the globe on each day. 
# 
# One other worrying fact is India is fast catching up with USA and Brazil. 
# 
# Finally, let us see how the average daily number of confirmed cases and deaths are moving across the world.

# In[ ]:


sns.set()
sns.relplot(x='Date', y='NewCases', data=select_daily, kind='line', ci='sd', aspect=15/5, marker='o')
plt.xticks(rotation=90)
plt.title('Average number of confirmed cases across states on a given day with associated standard deviation')
plt.show()


# In[ ]:


sns.set()
sns.relplot(x='Date', y='NewDeaths', data=select_daily, kind='line', ci='sd', aspect=15/5, marker='o')
plt.xticks(rotation=90)
plt.title('Average number of confirmed cases across states on a given day with associated standard deviation')
plt.show()


# Though the average number of confirmed cases per country for a given day seem to have increased significantly from about 2500 to 3900 over the past one month, the number of deaths recorded seem to be steady hovering around 140 mark.

# ### Key Findings:
# 
# 1. Doing lots of tests, containing confirmed cases and maintaining low number of deaths is a best metric to evaluate a country's performance in the fight against Covid-19. Gulf nation, UAE is doing the best job here.
# 
# 2. While USA is worst hit in terms of absolute numbers, it is Belgium, Spain & UK which are actually severly hit, given their high number of infected per million along with high number of deaths per million.
# 
# 3. UK, Spain and Netherlands did not report recovered cases. That leaves them with a very high number of active cases for over 4+ months. This is puzzling given the fact that both the Prince and Prime Minister of UK have recovred within 2 weeks of they got classified as confirmed cases. Reasons for the same to be ascertained as 4+ months is too long for Covid-19 cases to remain active (which is not the case in other countries). 
# 
# 4. Qatar inspite of its extremely high rate of infection, did a great job in getting the recovered numbers high and keeping deaths to a very low.
# 
# 5. Lack of enough testing in countries with large population like India should be a major concern as that does not give the correct data on the actually infected.
# 
# 6. Most of the European and American countries seem to be not doing great on recovery of infected as against deaths.
# 
# 7. One seriously worrying factis that India though started very late, is catching up very fast with Brazil and USA on the number of confirmed cases and deaths getting recorded.
