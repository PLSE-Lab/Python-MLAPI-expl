#!/usr/bin/env python
# coding: utf-8

# # Which Education Indicators are Related to GDP?
# Data Source: https://www.kaggle.com/worldbank/world-development-indicators

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ### Visual inspection of the data set

# In[ ]:


data = pd.read_csv('../input/Indicators.csv')
print("Columns:", data.shape[0])
print("Rows:", data.shape[1])
data.head()


# In[ ]:


# list countries for visual inspection
data['CountryName'].unique().tolist()


# #### Remove country labels that are not countries
# (thanks https://www.kaggle.com/smondal93/exploring-global-inequality-and-growth)

# In[ ]:


exclude_list = ['Arab World', 'Caribbean small states', 'Central Europe and the Baltics',
 'East Asia & Pacific \(all income levels',
 'East Asia & Pacific \(developing only', 'Euro area',
 'Europe & Central Asia \(all income levels',
 'Europe & Central Asia \(developing only', 'European Union',
 'Fragile and conflict affected situations',
 'Heavily indebted poor countries \(HIPC', 'High income',
 'High income: nonOECD', 'High income: OECD',
 'Latin America & Caribbean \(all income levels',
 'Latin America & Caribbean \(developing only',
 'Least developed countries: UN classification', 'Low & middle income',
 'Low income', 'Lower middle income',
 'Middle East & North Africa \(all income levels',
 'Middle East & North Africa \(developing only', 'Middle income',
 'North America' 'OECD members' ,'Other small states',
 'Pacific island small states', 'Small states', 'South Asia',
 'Sub-Saharan Africa \(all income levels',
 'Sub-Saharan Africa \(developing only' ,'Upper middle income' ,'World', 'North America', 'OECD members']


# In[ ]:


# list all education indicators
data.loc[data['IndicatorName'].str.contains('education|student|enrolment'),         'IndicatorName'].unique().tolist()


# In[ ]:


# select education indicators for further analysis:
ed_indicators_list = ['Pupil-teacher ratio in primary education (headcount basis)',                     'Pupil-teacher ratio in secondary education (headcount basis)',                     'Theoretical duration of primary education (years)',                     'Theoretical duration of secondary education (years)',                      'Official entrance age to lower secondary education (years)',                     'Official entrance age to primary education (years)',                     'Primary to secondary general education transition rate, both sexes (%)',                     'Expenditure on primary as % of government expenditure on education (%)',                     'Expenditure on secondary as % of government expenditure on education (%)',                     'Expenditure on tertiary as % of government expenditure on education (%)',                     'Gross enrolment ratio, pre-primary, female (%)',                     'Gross enrolment ratio, pre-primary, male (%)',                     'Gross enrolment ratio, primary, female (%)',                     'Gross enrolment ratio, primary, male (%)',                     'Gross enrolment ratio, secondary, female (%)',                     'Gross enrolment ratio, secondary, male (%)',                     'Gross enrolment ratio, tertiary, female (%)',                     'Gross enrolment ratio, tertiary, male (%)',                     'All education staff compensation, primary (% of total expenditure in primary public institutions)',                     'All education staff compensation, secondary (% of total expenditure in secondary public institutions)',                     'All education staff compensation, total (% of total expenditure in public institutions)']


# In[ ]:


# list all GDP indicators to select the appropriate GDP metric
# data.loc[data['IndicatorName'].str.contains('GDP'),'IndicatorName'].unique().tolist()


# ### Set up a new DataFrame that contains all education indicators and GDP per capita for each country

# In[ ]:


# remove country names that aren't countries:
country_pattern = '|'.join(exclude_list)
countries_filtered_2000 = data[(~data['CountryName'].str.match(country_pattern)) & (data['Year']==2000)]

# set up DataFrame to receive indicators by country
country_ed_indicators_2000 = pd.DataFrame(countries_filtered_2000['CountryName'].unique())
country_ed_indicators_2000.rename(columns={0:'CountryName'},inplace=True)

for n in ed_indicators_list:
    temp = countries_filtered_2000.loc[(countries_filtered_2000['IndicatorName'] == n),('CountryName','Value')]
    temp.rename(columns={'Value':str(n)},inplace=True)
    country_ed_indicators_2000=country_ed_indicators_2000.merge(temp, on='CountryName', how='left')

# Get GDP data from 2010 (the objective is to compare indicators from 2000 with GDP in 2010)
countries_filtered_2010 = data[(~data['CountryName'].str.match(country_pattern)) & (data['Year']==2010)]
gdp_2010 = countries_filtered_2010.loc[(countries_filtered_2010['IndicatorName'] == 'GDP per capita (current US$)'),                                       ('CountryName','Value')]
gdp_2010.rename(columns={'Value':'GDP per capita (current US$)'},inplace=True)
country_ed_indicators_2000 = country_ed_indicators_2000.merge(gdp_2010, on='CountryName', how='left')


# In[ ]:


# rename indicators with shorter names to make graphs a little more readable
long_keys = country_ed_indicators_2000.keys().tolist()
short_keys = ['CountryName',              'Pupil:Teacher primary ed (headcount)',                     'Pupil:Teacher secondary ed (headcount)',                     'Primary ed duration (years)',                     'Secondary ed duration (years)',                     'Secondary entrance age (years)',                     'Primary entrace age (years)',                     'Primary to secondary transition rate (%)',                     'Spend on primary as % of govt ed spend (%)',                     'Spend on secondary as % of govt ed spend (%)',                     'Spend on tertiary as % of govt ed spend (%)',                     'Gross enrol ratio, pre-primary, female (%)',                     'Gross enrol ratio, pre-primary, male (%)',                     'Gross enrol ratio, primary, female (%)',                     'Gross enrol ratio, primary, male (%)',                     'Gross enrol ratio, secondary, female (%)',                     'Gross enrol ratio, secondary, male (%)',                     'Gross enrol ratio, tertiary, female (%)',                     'Gross enrol ratio, tertiary, male (%)',                     'Ed staff comp, primary (% of spend in primary)',                     'Ed staff comp, secondary (% of spend in secondary)',                     'Ed staff comp, total (% of total spend)']

for i in range(0,len(short_keys)):
    country_ed_indicators_2000.rename(columns={str(long_keys[i]): str(short_keys[i])}, inplace=True)


# ### Calculate pairwise correlations and plot
# Source: https://seaborn.pydata.org/examples/many_pairwise_correlations.html

# In[ ]:


# calculate pairwise correlations among all indicators:
corr_2000 = country_ed_indicators_2000.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr_2000, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
plt.subplots(figsize=(12, 12))
plt.title('Correlation Matrix', fontsize=14)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)

# Generate a custom diverging colormap
cmap = sns.diverging_palette(20, 200, n=13)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_2000, mask=mask, cmap=cmap, vmax=.75, vmin=-.6, center=0,
            square=True, linewidths=8, cbar_kws={"shrink": .5}, robust=True)

plt.show()


# ### Now plot only the correlation with GDP

# In[ ]:


# extract correlation between each education indicator and GDP:
gdp_corr_2000 = corr_2000.iloc[:20,21].sort_values(ascending=True)

# set up the color map to diverge betwen positive and negative:
col_pos = np.asarray(sns.diverging_palette(200,220,n=32))
col_pos = col_pos[17:]
# sns.palplot(col_pos)
col_neg = np.asarray(sns.diverging_palette(20,30,n=12))
col_neg = col_neg[:5]
# sns.palplot(col_neg)
col_diverge = np.concatenate((col_neg,col_pos), axis=0)
# sns.palplot(col_diverge)

# plot
plt.figure(figsize=(4,10))
plt.barh(gdp_corr_2000.keys(),gdp_corr_2000, align='center', height = 0.75, color=col_diverge)
plt.yticks(fontsize=12)
plt.xlabel('Correlation', fontsize=12)
plt.title('Correlation between Education Indicators and GDP', fontsize =12)
plt.plot([0., 0.], [-0.25, 19.25], "k-", linewidth=0.3)

plt.show()


# ### Create scatter plots for the indicators with the largest correlations

# In[ ]:


# create new DataFrame with Gross enrol ratio in tertiary for males
male_ratios = pd.DataFrame(columns=['Gross enrol ratio, tertiary (%)',                                   'GDP per capita (current US$)','sex'])
male_ratios['Gross enrol ratio, tertiary (%)'] = country_ed_indicators_2000['Gross enrol ratio, tertiary, male (%)']
male_ratios['GDP per capita (current US$)'] = country_ed_indicators_2000['GDP per capita (current US$)']
male_ratios['sex'] = 'Male'

# create new DataFrame with Gross enrol ratio in tertiary for females
female_ratios = pd.DataFrame(columns=['Gross enrol ratio, tertiary (%)',                                   'GDP per capita (current US$)','sex'])
female_ratios['Gross enrol ratio, tertiary (%)'] = country_ed_indicators_2000['Gross enrol ratio, tertiary, female (%)']
female_ratios['GDP per capita (current US$)'] = country_ed_indicators_2000['GDP per capita (current US$)']
female_ratios['sex'] = 'Female'

# combine the male and female DataFrames
all_ratios = pd.concat([male_ratios,female_ratios])

# plot
ax = sns.scatterplot(all_ratios['Gross enrol ratio, tertiary (%)'],all_ratios['GDP per capita (current US$)'],                hue=all_ratios['sex'], palette="Blues")


# In[ ]:


ax = sns.scatterplot(country_ed_indicators_2000['Spend on primary as % of govt ed spend (%)'],           country_ed_indicators_2000['GDP per capita (current US$)'], palette="gist_gray_r")


# In[ ]:


# create new DataFrame with Pupil:Teacher primary
primary_ratios = pd.DataFrame(columns=['Pupil:Teacher (headcount)',                                   'GDP per capita (current US$)','Grade level'])
primary_ratios['Pupil:Teacher (headcount)'] = country_ed_indicators_2000['Pupil:Teacher primary ed (headcount)']
primary_ratios['GDP per capita (current US$)'] = country_ed_indicators_2000['GDP per capita (current US$)']
primary_ratios['Grade level'] = 'Primary'

# create new DataFrame with Pupil:Teacher secondary
secondary_ratios = pd.DataFrame(columns=['Pupil:Teacher (headcount)',                                   'GDP per capita (current US$)','Grade level'])
secondary_ratios['Pupil:Teacher (headcount)'] = country_ed_indicators_2000['Pupil:Teacher secondary ed (headcount)']
secondary_ratios['GDP per capita (current US$)'] = country_ed_indicators_2000['GDP per capita (current US$)']
secondary_ratios['Grade level'] = 'Secondary'

# combine the primary and secondary DataFrames
pupil_teacher = pd.concat([primary_ratios,secondary_ratios])

ax = sns.scatterplot(pupil_teacher['Pupil:Teacher (headcount)'], pupil_teacher['GDP per capita (current US$)'],                                   hue=pupil_teacher['Grade level'], palette="Blues")

