#!/usr/bin/env python
# coding: utf-8

# # Table of Contents
# 1. [Introduction](#ch1)
# 1. [The Dataset](#ch2)
# 1. [Missing Values](#ch3)
# 1. [Numerical Variables](#ch4)
#     1. [id, unique_id, unique_formula_id](#ch5)
#     1. [zip_code_of_death](#ch6)
# 1. [Temporal Variables](#ch7)
#     1. [date_of_injury_resulting_in_death](#ch8)
#         1. [date_month](#ch9)
#         1. [date_day](#ch10)
#         1. [date_year](#ch11)
#     1. [Year](#ch12)
# 1. [Categorical Variables](#ch13)
#     1. [age](#ch14)
#     1. [gender](#ch15)
#     1. [race](#ch16)
#     1. [city_of_death](#ch17)
#     1. [state_of_death](#ch18)
#     1. [county_of_death](#ch19)
#     1. [agency_responsible_for_death](#ch20)
#     1. [cause_of_death](#ch21)
#     1. [intentional_use_of_force](#ch22)
#     1. [symptoms_of_mental_illness](#ch23)
#     1. [dispositions](#ch24)
# 1. [Relationship among variables](#ch25)
#     1. [age and year](#ch26)
#     1. [age and race](#ch27)
#     1. [age, race, and year](#ch28)
#     1. [age and gender](#ch29)
#     1. [age, gender, and year](#ch30)
#     1. [age and cause_of_death](#ch31)
#     1. [age, cause_of_death, year](#ch32)
#     1. [age and intentional_use_of_force](#ch33)
# 	1. [age, intentional_use_of_force, year](#ch34)
#     1. [age and symtoms_of_mental_illness](#ch35)
# 	1. [age and symptoms_of_mental_illness, year](#ch36)
#     1. [Top 5 agencies with the most death by year](#ch37)
#     1. [Have some agencies improved?](#ch38)
#     1. [Top 5 agencies with the most death by race](#ch39)
#     1. [Top 5 agencies with the most death by gender](#ch40)
#     1. [Top 5 agencies with the most death by cause_of_death](#ch41)
#     1. [Top 5 agencies with the most death by intentional_use_of_force](#ch42)
#     1. [Top 5 agencies with the most death by symptoms_of_mental_illness](#ch43)
#     1. [year and gender](#ch44)
#     1. [year and race](#ch45)
#     1. [year and cause_of_death](#ch46)
#     1. [year and symptoms_of_mental_illness](#ch47)
#     1. [year and intentional_use_of_force](#ch48)
# 1. [Basic Text Analysis](#ch49)
#     1. [Word Cloud: Police Vocabulary](#ch50)
#     1. [Word Cloud: Don't give these names to your child :)](#ch51)
# 1. [Some officers are getting killed during the encounter](#52)
# 1. [Basic geospatial mapping](#ch53)

# <a id="ch1"></a>
# ## Introduction
# In the days since a Minneapolis police officer killed an unarmed black man named George Floyd by kneeling on his neck, there has been a massive national response. Protesters took to the streets across the country, calling for justice and an end to the disproportionate killings of Black Americans by police. 
# The goal of this notebook is to analyze 28 000+ police fatalities across the USA from 2000 to 2020.

# <a id="ch2"></a>
# ## The dataset

# In[ ]:


import pandas as pd
import numpy as np
import pprint
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')


# In[ ]:


path= "../input/police-fatalities-in-the-us-from-2000-to-2020/police_fatalities.csv"
data = pd.read_csv(path)
data.head()


# In[ ]:


data.shape


# In[ ]:


# print the columns
print("Column names: ")
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(data.columns.tolist())


# The dataset has 28335 rows and 29 columns.

# <a id="ch3"></a>
# ## Missing Values

# Before starting the analysis, let's look at the columns who are not populated.

# In[ ]:


def get_columns_stats(df: pd.DataFrame)-> None:
    col_with_na = [col for col in df.columns if data[col].isnull().sum() > 0]
    col_without_na = [col for col in df.columns if data[col].isnull().sum() == 0]
    print(f"We have {len(col_with_na)} columns with missing values and {len(col_without_na)} without missing values")
    print()
    print("Variable with missing values")
    print()
    print(f'{"Variable":<65} {"Number of missings":<20} {"Percent of missings":<20}')
    print()
    for col in col_with_na:
        print(f'{col:<65} {df[col].isnull().sum():<20} {np.round(data[col].isnull().mean()*100, 3)}%')
    print()
    print("variable without missing values")
    print(col_without_na)


# In[ ]:


get_columns_stats(data)


# Most columns are populated except 'Unique ID formula' and 'Video' which are mostly empty. I won't analyse those two columns. Let's rename the variables

# In[ ]:


columns = ['id','name', 'age', 'gender', 'race', 'race_with_imputations', 'imputation_probability','url_image_of_deceased', 'date_of_injury_resulting_in_death', 'address_of_injury', 'city_of_death', 'state_of_death',
          'zip_code_of_death', 'county_of_death', 'full_address', 'latitude', 'longitude', 'agency_responsible_for_death', 'cause_of_death',
          'description_circumstances_surrounding_death', 'dispositions', 'intentional_use_of_force', 'link_news_article_or_photo',
          'symptoms_of_mental_illness', 'video', 'date_and_description', 'unique_id_formula', 'unique_id', 'year']
data.columns = columns


# In[ ]:


pp.pprint(data.columns.tolist())


# In[ ]:


get_columns_stats(data)


# Now, we can start the analysis. Let's look at the numerical variables

# <a id="ch4"></a>
# ## Numerical Variables

# In[ ]:


# make list of numerical variables
num_vars = [var for var in data.columns if data[var].dtypes != 'O']

print('Number of numerical variables: ', len(num_vars))
print()
pp.pprint(num_vars)
print()
# visualise the numerical variables
get_columns_stats(data[num_vars])
print()
data[num_vars].head()


# 'latitude' and 'longitude' are geolocation data, I will analyze them later. 'unique_id_formula' is almost empty, I will remove it. 'unique_id' and 'id' have the same number of missing values. Let's check if they are the same.

# <a id="ch5"></a>
# ### id, unique_id, unique_id_formula

# In[ ]:


data[data.id.isnull()]


# In[ ]:


data[data.unique_id.isnull()]


# Both the missing value in 'id', and 'unique_id' refer to the same record.

# In[ ]:


# convert id to float
#print(data['id'].astype(float).values == data['unique_id'].values)


# In[ ]:


data[data.id == 'Victor Sanchez Ancira']


# In[ ]:


# check if values in id and unique_id are the same
print((data[~data.index.isin([24866,28334])].id.astype(float).values == 
      data[~data.index.isin([24866,28334])].unique_id.astype(float).values).all())


# In[ ]:


print(f"Number of Unique id: {len(data['id'].unique())}")
print(f"Number of Unique unique_id: {len(data['unique_id'].unique())}")


# 'id' and 'unique_id' refer to the same thing. Let's drop both id and unique_id_formula

# In[ ]:


data.drop(['id', 'unique_id_formula'], axis=1, inplace=True)


# In[ ]:


pp.pprint(data.columns.tolist())


# <a id="ch6"></a>
# ### zip_code_of_death

# In[ ]:


# some useful function
import itertools
import seaborn as sns
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


def top_n_bar(df: pd.DataFrame, col: str, n: int, figsize:Tuple[int, int]=(13, 10), s_elt:float = 0.6) -> None:
    out_series = df[col].value_counts()
    total_size = sum(out_series.tolist())
    out = dict(itertools.islice(out_series.items(), n)) 
    pd_df = pd.DataFrame(list(out.items()))
    pd_df.columns =[col, "Count"] 
    plt.figure(figsize=figsize)
    ax = sns.barplot(y=pd_df.index, x=pd_df.Count, orient='h')
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    ax.set(xlabel="Count", ylabel=col)
    ax.set_yticklabels(pd_df[col])
    for i, v in enumerate(pd_df["Count"].iteritems()):
        ax.text(v[1] ,i, "{:,}".format(v[1]), color='m', va ='top', rotation=0)
        ax.text(v[1]+ s_elt ,i, "({:0.2f}%)".format((v[1]/total_size)*100), color='m', va ='top', rotation=0)
    plt.tight_layout()
    plt.show()


def check_variable(df: pd.DataFrame, col: str): 
    number_unique = df[col].nunique()
    print(f'variable {col} has {number_unique} unique values')
    if number_unique< 200:
        print(f'These are the {col} values')
        print(f'{df[col].unique()}')


# In[ ]:


check_variable(data, 'zip_code_of_death')


# In[ ]:


top_n_bar(data, 'zip_code_of_death', 20, (13, 10))


# 60620, 60628, 60616, which belong to Chicago, are zip codes with the most deaths.

# <a id="ch7"></a>
# ## Temporal Variables

# The following are the temporal variables:  'date_of_injury_resulting_in_death', and 'year'

# <a id="ch8"></a>
# ### date_of_injury_resulting_in_death

# The 'date_of_injury_resulting_in_death' represent the day in which the incident occured.

# In[ ]:


check_variable(data, 'date_of_injury_resulting_in_death')


# In[ ]:


data['date_of_injury_resulting_in_death'][:10]


# The 'date_of_injury_resulting_in_death' is an object type with the following format: mm/dd/year. Let's add the following additional columns to the data: date_day, date_month, date_year

# In[ ]:


data['date_month'], data['date_day'], data['date_year'] = zip(*data['date_of_injury_resulting_in_death'].apply(lambda x: x.split('/')))


# In[ ]:


pp.pprint(data.columns.tolist())


# <a id="ch9"></a>
# ###  date_month

# In[ ]:


check_variable(data, 'date_month')


# The 'date_month' does not have any anomalies. It has 12 months from 01 to 12 that represent the month.

# In[ ]:


top_n_bar(data, 'date_month', 12, (13, 10), s_elt=100.0)


# Most fatalities occurs in March, May, and July

# <a id="ch10"></a>
# ###  date_day

# In[ ]:


check_variable(data, 'date_day')


# The date_day column does not contain anomalies and has values between 01 and 31, representing a day in a month.

# In[ ]:


top_n_bar(data, 'date_day', 31, (13, 10), s_elt=34.0)


# Most fatalities occur on the first day of each month

# <a id="ch11"></a>
# ###  date_year

# In[ ]:


check_variable(data, 'date_year')


# date_year has '2100', which is an anomaly. Let's check it

# In[ ]:


data[data['date_year'] == '2100']


# The 'date_year' contains only one record, and it's mostly NaN. I will remove it

# In[ ]:


data.drop(labels=28334, inplace=True)
check_variable(data, 'date_year')


# In[ ]:


data.date_year.isnull().sum()


# In[ ]:


top_n_bar(data, 'date_year', 20, s_elt=65.0)


# The fatalities have seen a steady increase since 2017, with a 0.8% increase on average each year.

# <a id="ch12"></a>
# ###  year

# The 'year' variable refers to the year of the subject's death. Let's check if it's the same as date_year.

# In[ ]:


check_variable(data, 'year')


# In[ ]:


data.year.isnull().sum()


# The 'year' variable has 24 missing values. Let's check if the non-missing values are the same in 'date_year.'

# In[ ]:


(data[~data.year.isnull()]['year'].values == data[~data.year.isnull()]['date_year'].astype('float').values).all()


# They are both the same.

# <a id="ch13"></a>
# ## Categorical Variables

# In[ ]:


temporal_vars = ['date_day', 'date_month', 'date_year', 'date_of_injury_resulting_in_death']
cat_vars = [var for var in data.columns if data[var].dtypes == 'O' and var not in temporal_vars]


# In[ ]:


print('Number of categorical variables: ', len(cat_vars))


# In[ ]:


get_columns_stats(data[cat_vars])


# <a id="ch14"></a>
# ### age
# The 'age' variable refers to the age of the subject's death.

# In[ ]:


check_variable(data, 'age')


# The 'age' values are either the exact subject's age - e.g., 17 - or the range within which the subject's age falls - e.g., 40s, 18-25. Let's count range ages and ages less that 1 year.

# In[ ]:


unspecific_age = ['20s-30s', '18-25', '25-30', '40-50', '46/53', '45 or 49', '40s', '30s', '50s', '70s', '60s']
age_less_than_a_year = ['3 months', '6 months', '9 months', '10 months', '2 months', '7 months', '8 months', '4 months', '3 days', '11 mon','7 mon' ]


# In[ ]:


data[data['age'].isin(unspecific_age + age_less_than_a_year)].shape[0]


# The ages with values within a range or less than a year are too small. I can impute them. It won't have much of an effect on the overall statistics.

# In[ ]:


age_imputation_dict = {
    '20s-30s':25,
    '40-50': 45,
    '18-25':22,
    '46/53':46,
    '45 or 49': 45, 
    '40s': 45,
    '30s':35,
    '50s': 55,
    '60s': 65,
    '18 months': 2,
    '70s': 75,
    '25-30': 27,
    '20s':25,
    '25`': 25, 
    '55.':55
}


# In[ ]:


data["age"].replace(age_imputation_dict, inplace=True)
data['age']= data['age'].apply(lambda x: 0.0 if x in age_less_than_a_year else float(x))


# In[ ]:


check_variable(data, 'age')


# In[ ]:


top_n_bar(data, 'age', 50, (10, 12), s_elt=34.0)


# In[ ]:


figure(num=None, figsize=(15, 9), dpi=80, facecolor='w', edgecolor='k')

sns.distplot(data['age']);


# In[ ]:


figure(num=None, figsize=(15, 9), dpi=80, facecolor='w', edgecolor='k')
sns.boxplot(y="age", data=data)


# The age group most killed is between 20 and 40 years old.

# <a id="ch15"></a>
# ### gender

# In[ ]:


check_variable(data, 'gender')


# In[ ]:


top_n_bar(data, 'gender', 5, s_elt=1400.0)


# About 90% of people killed are men.

# <a id="ch16"></a>
# ### race

# The 'race' variable refers to the subject's race.

# In[ ]:


check_variable(data, 'race')


# There is a typo: 'Hispanic/Latino' and 'HIspanic/Latino.' Let's fix it.

# In[ ]:


data[data['race'] == 'HIspanic/Latino']


# In[ ]:


data.loc[27546, 'race'] = 'Hispanic/Latino'


# In[ ]:


check_variable(data, 'race')


# In[ ]:


top_n_bar(data, 'race', 7, s_elt=400.0)


# Roughly 31% of the data have 'Race unspecified' values. Let's take a closer look at it.

# In[ ]:


plt.rcParams["figure.figsize"] = [20, 10]
cm = data.groupby(["date_year", "race"])["race"].count()
cm = cm.unstack(fill_value=0)
cm.plot.bar()


# In[ ]:


print('race before 2010')
top_n_bar(data[data['year'] <= 2009], 'race', 7, s_elt=400.0)


# In[ ]:


print('race after 2009')
top_n_bar(data[data['year'] > 2009], 'race', 7, s_elt=400.0)


# European-American/White are the most reportedly killed since 2000.They account for 9,140 (32.36%) followed by African-American/Black - 6,029 (21.28%) -, and Hispanic/Latino- 3,648 (12.87%).

# <a id="ch17"></a>
# ### city_of_death

# The 'city_of_death' represents the city location where the injury cause death happened.

# In[ ]:


check_variable(data, 'city_of_death')


# In[ ]:


top_n_bar(data, 'city_of_death', 20, s_elt=12.0)


# Since 2000, Chicago, Houston, and Los Angeles are cities where most deaths occur with 447, 440, 407 deaths reported respectively.

# <a id="ch18"></a>
# ### state_of_death

# The 'state_of_death' represents the state location where the injury cause death happened.

# In[ ]:


check_variable(data, 'state_of_death')


# In[ ]:


top_n_bar(data, 'state_of_death', 51, s_elt=160.0)


# Since 2000, California, Texas, and Florida are states where most deaths occurred with 4548 (16.05%), 2,500(8.82%), 1,714(6.05%) deaths reported respectively.

# <a id="ch19"></a>
# ### county_of_death

# The 'county_of_death' represents the county location where the injury cause death happened.

# In[ ]:


check_variable(data, 'county_of_death')


# In[ ]:


top_n_bar(data, 'county_of_death', 30, s_elt=45.0)


# Since 2000, Los Angeles, Cook, and Harris are counties where most deaths occurred with 1,217 (4.30%), 599(2.12%), 550(1.95%) deaths reported respectively.

# <a id="ch20"></a>
# ### agency_responsible_for_death

# The 'agency_responsible_for_death' represents the agency responsible of the subject's death.

# In[ ]:


check_variable(data, 'agency_responsible_for_death')


# In[ ]:


top_n_bar(data, 'agency_responsible_for_death', 20, s_elt=20.0)


# Since 2000, the **Los Angeles Police Department**,the **Chicago Police Department**, the **Los Angeles County Sheriff's Department**, the **City of New York Police Department**, and the **Houston Police Department** are agencies where most deaths occurred. They account for **469 (1.66%), 423 (1.50%), 345 (1.22%), 326 (1.15%), 310 (1.10%)** deaths respectively.

# <a id="ch21"></a>
# ### cause_of_death

# The 'cause_of_death' variable represents the cause of the subject's death.

# In[ ]:


check_variable(data, 'cause_of_death')


# In[ ]:


top_n_bar(data, 'cause_of_death', 16, s_elt=1050.0)


# Since 2000, 71% (20,094) of deaths are caused by gunshots, while 20% (5,803) are due to vehicle accidents.

# <a id="ch22"></a>
# ### intentional_use_of_force

# The 'intentional_use_of_force' variable indicates whether the officer intentionally used force.

# In[ ]:


check_variable(data, 'intentional_use_of_force')


# In[ ]:


top_n_bar(data, 'intentional_use_of_force', 11, s_elt=1050.0)


# The author did not give a reasonable explanation for this variable. Nevertheless, I will group them in the following categories:
# 
# 	1 - Yes  ('Yes', and  'Intentional Use of Force'): indicates an intentional use of force.
#     2 - No : indicates the officer did not intentionally use force.
#     3 - Intenional Use of Force, Deadly ( 'Intenional Use of Force, Deadly' and  'Intenional Use of Force, Deadly' ):  indicates an intentional use of force which resulted to the subject's death.
#     4 - Vehicle/Pursuit (Vehicle/Pursuit + Pursuit + Vehicle): indicates an intentional use of force with the vehicle.
#     5 - Undetermined
#     6 - Unknown
#     7 - Suicide

# In[ ]:


data.loc[data[data['intentional_use_of_force'].isin(['Yes','Intentional Use of Force'])].index, 'intentional_use_of_force'] = 'Yes'
data.loc[data[data['intentional_use_of_force'].isin(['Intenional Use of Force, Deadly', 'Intentional Use of Force, Deadly'])].index, 'intentional_use_of_force'] = 'Intentional Use of Force, Deadly'
data.loc[data[data['intentional_use_of_force'].isin(['Vehicle/Pursuit', 'Vehicle','Pursuit'])].index, 'intentional_use_of_force'] = 'Vehicle/Pursuit'


# In[ ]:


top_n_bar(data, 'intentional_use_of_force', 11, s_elt=1050.0)


# Since 2000, most deaths (61%) occur by an intentional use of force.
# 

# <a id="ch23"></a>
# ### symptoms_of_mental_illness

# The 'symptoms_of_mental_illness' variable indicates whether the officer was aware of the subject's mental illness before the interaction.

# In[ ]:


check_variable(data, 'symptoms_of_mental_illness')


# In[ ]:


top_n_bar(data, 'symptoms_of_mental_illness', 5, s_elt=1050.0)


# In most interactions (66%), the officer is not aware of the subject's mental illness status.

# <a id="ch24"></a>
# ### dispositions

# I don't understand this variable. Below are its values.

# In[ ]:


check_variable(data, 'dispositions')


# In[ ]:


top_n_bar(data, 'dispositions', 30,  s_elt=600)


# <a id="ch25"></a>
# ## Relationship among variables

# In this section, I will check whether there is some patterns among variables.

# In[ ]:





def plotbox(df, x, y, figsize=(15, 9), orientation='v'):
    figure(num=None, figsize=figsize, dpi=80, facecolor='w', edgecolor='k')
    IQRs =  df.groupby([x])[y].quantile(0.75) -  df.groupby([x])[y].quantile(0.25)
    medians = df.groupby([x])[y].median()
    percentile_25th = df.groupby([x])[y].quantile(0.25)
    percentile_75th = df.groupby([x])[y].quantile(0.75)
    lower_boundary = df.groupby([x])[y].min()
    upper_boundary = percentile_75th + 1.5*IQRs
    max_value = df.groupby([x])[y].max()
    order = df.groupby([x])[y].sum().index

    
    
    if orientation == 'v':
        box_plot = sns.boxplot(x=x,y=y,data=df)
        vertical_offset_median = df[y].median() * 0.05 #
        vertical_offset_percentile_25th = df[y].quantile(0.25) * 0.05 
        vertical_offset_percentile_75th = df[y].quantile(0.75) * -0.05 
        vertical_lower_boundary_offset = df[y].min()*0.05
        vertical_max_boundary_offset = df[y].max()*0.05
        vertical_upper_boundary_offset = (df[y].quantile(0.75) - (df[y].quantile(0.75)  - (df[y].quantile(0.75) -  df[y].quantile(0.25))*1.5))*0.05
    
        for xtick in box_plot.get_xticks():
            box_plot.text(xtick,medians[xtick] + vertical_offset_median,medians[xtick], 
                horizontalalignment='center',size='x-small',color='w',weight='semibold')
            box_plot.text(xtick,percentile_25th[xtick] + vertical_offset_percentile_25th,percentile_25th[xtick], 
                horizontalalignment='center',size='x-small',color='w',weight='semibold')
            box_plot.text(xtick,percentile_75th[xtick] + vertical_offset_percentile_75th,percentile_75th[xtick], 
                horizontalalignment='center',size='x-small',color='w',weight='semibold')
            box_plot.text(xtick + 0.25,lower_boundary[xtick] + vertical_lower_boundary_offset,lower_boundary[xtick], 
                horizontalalignment='center',size='small',color='b',weight='semibold')
            box_plot.text(xtick -0.25,upper_boundary[xtick] + vertical_upper_boundary_offset,upper_boundary[xtick], 
                horizontalalignment='center',size='x-small',color='b',weight='semibold')
            box_plot.text(xtick,max_value[xtick] + vertical_max_boundary_offset,max_value[xtick], 
                horizontalalignment='center',size='x-small',color='b',weight='semibold')
    else:
        box_plot = sns.boxplot(x=y,y=x,data=df, order=order)
        vertical_offset_median = df[y].median()*0.05 #
        vertical_offset_percentile_25th = df[y].quantile(0.25)*0 
        vertical_offset_percentile_75th = df[y].quantile(0.75)*0
        vertical_lower_boundary_offset = df[y].min()*0
        vertical_max_boundary_offset = df[y].max()*0
        vertical_upper_boundary_offset = (df[y].quantile(0.75) - (df[y].quantile(0.75)  - (df[y].quantile(0.75) -  df[y].quantile(0.25))*1.5))*0
    
        
        for ytick in box_plot.get_yticks():
            box_plot.text(medians[ytick], ytick + 0.05, medians[ytick], 
                verticalalignment='center',size='x-small',color='b',weight='semibold')
            box_plot.text(percentile_25th[ytick], ytick + 0.05,percentile_25th[ytick], 
                verticalalignment='center',size='x-small',color='b',weight='semibold')
            box_plot.text(percentile_75th[ytick], ytick + 0.05, percentile_75th[ytick], 
                verticalalignment='center',size='x-small',color='b',weight='semibold')
            box_plot.text(lower_boundary[ytick],ytick + 0.05, lower_boundary[ytick], 
                verticalalignment='center',size='small',color='b',weight='semibold')
            box_plot.text(upper_boundary[ytick],ytick + 0.05, upper_boundary[ytick], 
                verticalalignment='center',size='x-small',color='b',weight='semibold')
            box_plot.text(max_value[ytick], ytick + 0.05, max_value[ytick], 
                verticalalignment='center',size='x-small',color='b',weight='semibold')
            
    plt.show()
    


# In[ ]:


plotbox(data, x='date_year',y='age')


# Before 2009, the median age of death was between 30 and 32. Since 2010, the median age of death is between 33 and 35. 75% of deaths are less than 46 years.

# In[ ]:


plotbox(data, x='race',y='age', orientation='h')


# Minorities are more likely to be killed at a younger age

# <a id="ch28"></a>
# ### age, race and year

# In[ ]:


for year in data.date_year.unique():
    print(year)
    plotbox(data[data.date_year == year], x='race',y='age', orientation='h')


# In[ ]:


## median death age by race each year
median_df = data.groupby(['date_year', 'race']).age.median().reset_index()
ax = sns.lineplot(x="date_year", y="age", hue='race',estimator=None, lw=3,
                  data=median_df [median_df ['race'] != 'Race unspecified'])
ax.set_title('Median Age Death By Race')


# In[ ]:


print('African American/Black median age range')
print(median_df[median_df.race == 'African-American/Black']['age'].max() - median_df[median_df.race == 'African-American/Black']['age'].min())
print('Hispanic/Latino median age range')
print(median_df[median_df.race == 'Hispanic/Latino']['age'].max() - median_df[median_df.race == 'Hispanic/Latino']['age'].min())
print('European-American/White median age range')
print(median_df[median_df.race == 'European-American/White']['age'].max() - median_df[median_df.race == 'European-American/White']['age'].min())
print('Native American/Alaskan median age range')
print(median_df[median_df.race == 'Native American/Alaskan']['age'].max() - median_df[median_df.race == 'Native American/Alaskan']['age'].min())
print('Middle Eastern median age range')
print(median_df[median_df.race == 'Middle Eastern']['age'].max() - median_df[median_df.race == 'Middle Eastern']['age'].min())
print('Asian/Pacific Islander median age range')
print(median_df[median_df.race == 'Asian/Pacific Islander']['age'].max() - median_df[median_df.race == 'Asian/Pacific Islander']['age'].min())


# ##### todo: write something later

# Half of the black deaths each year are people younger than 30 years old.

# <a id="ch29"></a>
# ### age and gender

# In[ ]:


plotbox(data, x='gender',y='age', orientation='h')


# In[ ]:


for year in data.date_year.unique():
    print(year)
    plotbox(data[data.date_year == year], x='gender',y='age', orientation='h')


# In[ ]:


## median death age by race each year
median_df_gender = data.groupby(['date_year', 'gender']).age.median().reset_index()
ax = sns.lineplot(x="date_year", y="age", hue='gender',estimator=None, lw=3,
                  data=median_df_gender)
ax.set_title('Median Age Death By gender')


# In[ ]:


len(data[data.gender == 'Transgender'])


# In[ ]:


data[data.gender == 'Transgender']


# In[ ]:


print('Transgender median age range')
print(median_df_gender[median_df_gender.gender == 'Transgender']['age'].max() - median_df_gender[median_df_gender.gender == 'Transgender']['age'].min())


# In[ ]:


print('Male median age range')
print(median_df_gender[median_df_gender.gender == 'Male']['age'].max() - median_df_gender[median_df_gender.gender == 'Male']['age'].min())


# In[ ]:


print('Female median age range')
print(median_df_gender[median_df_gender.gender == 'Female']['age'].max() - median_df_gender[median_df_gender.gender == 'Female']['age'].min())


# ##### todo: write something later

# <a id="ch31"></a>
# ### age and cause_of_death

# In[ ]:


plotbox(data, x='cause_of_death', y='age', orientation='h')


# <a id="ch32"></a>
# ### age, cause_of_death and year

# In[ ]:


for year in data.date_year.unique():
    print(year)
    plotbox(data[data.date_year == year], x='cause_of_death',y='age', orientation='h')


# In[ ]:


my_colors = ['#6D7815', 
                    '#49392F',
                    '#4924A1', 
                    '#A1871F', 
                    '#9B6470',  
                    '#7D1F1A',  
                    '#9C531F', 
                    '#6D5E9C',  
                    '#493963', 
                    '#638D8D',  
                    '#6D6D4E', 
                    '#682A68', 
                    '#A13959', 
                    '#D1C17D',
                    '#445E9C',
                    '#44685E'
             ]

median_df = data.groupby(['date_year', 'cause_of_death']).age.median().reset_index()
ax = sns.lineplot(x="date_year", y="age", hue='cause_of_death',estimator=None, lw=3,
                  data=median_df, palette=my_colors )
ax.set_title('Median Age Death By Cause of death')


# #### todo: write something later

# <a id="ch33"></a>
# ### age and intentional_use_of_force

# In[ ]:


plotbox(data, x='intentional_use_of_force', y='age', orientation='h')


# <a id="ch34"></a>
# ### age, intentional_use_of_force, and year

# In[ ]:


for year in data.date_year.unique():
    print(year)
    plotbox(data[data.date_year == year], x='intentional_use_of_force',y='age', orientation='h')


# In[ ]:


median_df = data.groupby(['date_year', 'intentional_use_of_force']).age.median().reset_index()
ax = sns.lineplot(x="date_year", y="age", hue='intentional_use_of_force',estimator=None, lw=3,
                  data=median_df)
ax.set_title('Median Age Death By intentional_use_of_force')


# #### todo: write something later

# <a id="ch35"></a>
# ### age and symtoms_of_mental_illness

# In[ ]:


plotbox(data, x='symptoms_of_mental_illness', y='age', orientation='h')


# <a id="ch36"></a>
# ### age, symtoms_of_mental_illness and year

# In[ ]:


for year in data.date_year.unique():
    print(year)
    plotbox(data[data.date_year == year], x='symptoms_of_mental_illness',y='age', orientation='h')


# In[ ]:


median_df = data.groupby(['date_year', 'symptoms_of_mental_illness']).age.median().reset_index()
ax = sns.lineplot(x="date_year", y="age", hue='symptoms_of_mental_illness',estimator=None, lw=3,
                  data=median_df)
ax.set_title('Median Age Death By symptoms_of_mental_illness')


# #### todo: write something later

# <a id="ch37"></a>
# ### Top 5 agencies with the most deaths per year

# In[ ]:


for year in data['date_year'].unique():
  print(year)
  top_n_bar(data[data['date_year'] == year], 'agency_responsible_for_death', 5, s_elt=2, figsize=(10, 5))


# #### todo: write something

# <a id="ch38"></a>
# ### Have some agencies improved?

# In[ ]:


top_5_agency_year = []
for year in data.date_year.unique():
    out_series = data[data['date_year'] == year]['agency_responsible_for_death'].value_counts()
    total_size = sum(out_series.tolist())
    out = dict(itertools.islice(out_series.items(), 5))
    top_5_agency_year.append(list(out.keys()))
top_agency_list = [agency for agency_year in top_5_agency_year for agency in agency_year]
top_agency_list = list(set(top_agency_list))
for agency in top_agency_list:
  print(agency)
  top_n_bar(data[data['agency_responsible_for_death'] == agency], 'date_year', 20, s_elt=2, figsize=(10, 5))


# #### todo: write something later

# <a id="ch39"></a>
# ### Top 5 agencies with the most death by race

# In[ ]:


for race in data['race'].unique():
    print(race)
    if race == 'Middle Eastern':
        top_n_bar(data[data['race'] == race], 'agency_responsible_for_death', 5, figsize=(12, 5))
    elif race in ['European-American/White', 'Native American/Alaskan']:
        top_n_bar(data[data['race'] == race], 'agency_responsible_for_death', 5, s_elt=7, figsize=(12, 5))
    else:
        top_n_bar(data[data['race'] == race], 'agency_responsible_for_death', 5, s_elt=10, figsize=(12, 5))
        
  


# ##### todo: write something later

# <a id="ch40"></a>
# ### Top 5 agencies with the most death by gender

# In[ ]:


for gender in ['Male', 'Female','Transgender']:
    print(gender)
    if gender in ['Transgender']:
      top_n_bar(data[data['gender'] == gender], 'agency_responsible_for_death', 5, figsize=(10, 5))
    elif gender == 'Female': 
        top_n_bar(data[data['gender'] == gender], 'agency_responsible_for_death', 5, s_elt=10, figsize=(10, 5))
    else:
        top_n_bar(data[data['gender'] == gender], 'agency_responsible_for_death', 5, s_elt=30, figsize=(10, 5))


# #### todo: write something later

# <a id="ch41"></a>
# ### Top 5 agencies with the most death by cause_of_death

# In[ ]:


for cause_of_death in data['cause_of_death'].unique():
    print(cause_of_death)
    if cause_of_death in ['Vehicle']:
        top_n_bar(data[data['cause_of_death'] == cause_of_death], 'agency_responsible_for_death', 5, s_elt=15, figsize=(10, 5))
    elif cause_of_death in ['Gunshot']:
        top_n_bar(data[data['cause_of_death'] == cause_of_death], 'agency_responsible_for_death', 5, s_elt=25, figsize=(10, 5))
    else:
        top_n_bar(data[data['cause_of_death'] == cause_of_death], 'agency_responsible_for_death', 5, figsize=(10, 5))
        


# #### todo: write something later

# <a id="ch42"></a>
# ### Top 5 agencies with the most death by intentional_use_of_force

# In[ ]:


for use_of_force in ['Vehicle/Pursuit', 'Intentional Use of Force, Deadly', 'Suicide',
       'Yes', 'No', 'Undetermined', 'Unknown']:
    print(use_of_force)
    if use_of_force in ['Undetermined', 'Unknown']:
        top_n_bar(data[data['intentional_use_of_force'] == use_of_force], 'agency_responsible_for_death', 5, figsize=(10, 5))
    elif use_of_force == 'Intentional Use of Force, Deadly':
        top_n_bar(data[data['intentional_use_of_force'] == use_of_force], 'agency_responsible_for_death', 5, s_elt=25, figsize=(10, 5))
    
    else:
        top_n_bar(data[data['intentional_use_of_force'] == use_of_force], 'agency_responsible_for_death', 5, s_elt=15, figsize=(10, 5))
  


# #### todo: write something

# <a id="ch43"></a>
# ### Top 5 agencies with the most death by symptoms_of_mental_illness

# In[ ]:


for mental_illness in ['No', 'Drug or alcohol use', 'Unknown', 'Yes']:
    print(mental_illness)
    top_n_bar(data[data['symptoms_of_mental_illness'] == mental_illness], 'agency_responsible_for_death', 5, s_elt=15, figsize=(10, 5))
        


# #### todo: write something

# <a id="ch44"></a>
# ### year and gender

# In[ ]:


for year in data['date_year'].unique():
  print(year)
  top_n_bar(data[data['date_year'] == year], 'gender', 5, s_elt=60, figsize=(10, 5))


# In[ ]:


plt.rcParams["figure.figsize"] = [20, 10]
cm = data.groupby(["date_year", "gender"])["gender"].count()
cm = cm.unstack(fill_value=0)
cm.plot.bar()


# #### todo: write something

# <a id="ch45"></a>
# ### year and race

# In[ ]:


for year in data['date_year'].unique():
  print(year)
  top_n_bar(data[data['date_year'] == year], 'race', 5, s_elt=25, figsize=(10, 5))


# In[ ]:


plt.rcParams["figure.figsize"] = [20, 10]
cm = data[data.race != 'Race unspecified'].groupby(["date_year", "race"])["race"].count()
cm = cm.unstack(fill_value=0)
cm.plot.bar()


# #### todo: write something later

# <a id="ch46"></a>
# ### year and cause_of_death

# In[ ]:


for year in data['date_year'].unique():
  print(year)
  top_n_bar(data[data['date_year'] == year], 'cause_of_death', 5, s_elt=25, figsize=(10, 5))


# In[ ]:


plt.rcParams["figure.figsize"] = [20, 10]
cm = data.groupby(["date_year", 'cause_of_death'])['cause_of_death'].count()
cm = cm.unstack(fill_value=0)
cm.plot.bar()


# #### todo write something later

# <a id="ch47"></a>
# ### year and symptoms_of_mental_illness

# In[ ]:


for year in data['date_year'].unique():
  print(year)
  top_n_bar(data[data['date_year'] == year], 'symptoms_of_mental_illness', 5, s_elt=50, figsize=(10, 5))


# In[ ]:


plt.rcParams["figure.figsize"] = [20, 10]
cm = data.groupby(["date_year", 'symptoms_of_mental_illness'])['symptoms_of_mental_illness'].count()
cm = cm.unstack(fill_value=0)
cm.plot.bar()


# #### todo: write something

# <a id="ch48"></a>
# ### year and intentional_use_of_force

# In[ ]:


for year in data['date_year'].unique():
  print(year)
  top_n_bar(data[data['date_year'] == year], 'intentional_use_of_force', 5, s_elt=55, figsize=(10, 5))


# In[ ]:


plt.rcParams["figure.figsize"] = [20, 10]
cm = data.groupby(["date_year", 'intentional_use_of_force'])['intentional_use_of_force'].count()
cm = cm.unstack(fill_value=0)
cm.plot.bar()


# #### todo : write something

# <a id="ch49"></a>
# ## Basic Text Analysis

# In[ ]:


import re
import string
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from collections import Counter
from wordcloud import WordCloud


# In[ ]:


import nltk
nltk.download('punkt')
nltk.download('stopwords')


# In[ ]:


def process_text(text):
    """
    process the text data
    """
    stopwords_english = stopwords.words('english')
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
    text = text.lower()
    text_tokens = word_tokenize(text)

    texts_clean = []
    for word in text_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation
              and len(word) > 2):  
            texts_clean.append(word)

    return texts_clean


# <a id="ch50"></a>
# ### Word Cloud: Police Vocabulary
# 

# In[ ]:


corpus_df = data[['unique_id', 'description_circumstances_surrounding_death', 'date_and_description']]


# In[ ]:


corpus_df.date_and_description[0]


# In[ ]:


corpus_df.description_circumstances_surrounding_death[0]


# In[ ]:


corpus_df.date_and_description[1]


# In[ ]:


corpus_df.description_circumstances_surrounding_death[1]


# In[ ]:


corpus_text =corpus_df.description_circumstances_surrounding_death.to_list()


# In[ ]:


text_tokens_list = [process_text(token) for token in corpus_text]


# In[ ]:


text_tokens = [token for token_list in text_tokens_list for token in token_list]


# In[ ]:


frequency_text = Counter(text_tokens)


# In[ ]:


wc = WordCloud()
wc.generate_from_frequencies(frequencies=dict(frequency_text))
plt.imshow(wc) 


# In[ ]:


text_df = pd.Series(dict(frequency_text))
text_df = pd.DataFrame(list(dict(frequency_text).items()),columns=['words', 'count'], index=np.arange(len(frequency_text)))


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 15))

# Plot horizontal bar graph
text_df.sort_values(by='count')[-50:].plot.barh(x='words',
                      y='count',
                      ax=ax,
                      color="green")

ax.set_title("Common words")


plt.show()


# <a id="ch51"></a>
# ### Word Cloud: Don't give these names to your child :)

# In[ ]:


names_text = data.name.to_list()
names_tokens_list =  [process_text(token) for token in names_text]
names_tokens = [token for token_list in names_tokens_list for token in token_list]


# In[ ]:


frequency_names = Counter(names_tokens)


# In[ ]:


frequency_names.most_common(10)


# In[ ]:


len(frequency_names)


# In[ ]:


wc = WordCloud()
wc.generate_from_frequencies(frequencies=dict(frequency_names))
plt.imshow(wc) 


# In[ ]:


name_df = pd.Series(dict(frequency_names))
name_df = pd.DataFrame(list(dict(frequency_names).items()),columns=['name', 'count'], index=np.arange(len(frequency_names)))


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 15))

# Plot horizontal bar graph
name_df.sort_values(by='count')[-50:].plot.barh(x='name',
                      y='count',
                      ax=ax,
                      color="green")

ax.set_title("Common names")


plt.show()


# <a id="ch52"></a>
# ### Some officers are getting killed during the encounter

# In[ ]:


police_df = data[data.link_news_article_or_photo.str.contains('officer-killed', na=False)]
police_df.head()


# In[ ]:


police_df.link_news_article_or_photo.tolist()


# In[ ]:


plt.rcParams["figure.figsize"] = [20, 10]
cm = police_df.groupby(["date_year", "gender"])["gender"].count()
cm = cm.unstack(fill_value=0)
cm.plot.bar()


# In[ ]:


top_n_bar(police_df, 'gender', 4)


# In[ ]:


top_n_bar(police_df, 'race', 4)


# <a id="ch53"></a>
# ### Basic geospatial mapping
# 

# In[ ]:


import folium
from folium import Choropleth, Circle, Marker
from folium.plugins import HeatMap, MarkerCluster
import math


# In[ ]:


#Create the map
map_plot = folium.Map(location=[48, -102], tiles='cartodbpositron', zoom_start=3)

# Add points to the map
mc = MarkerCluster()
for idx, row in data.iterrows():
    if not math.isnan(row['longitude']) and not math.isnan(row['latitude']):
        mc.add_child(Marker([row['latitude'], row['longitude']]))
map_plot.add_child(mc)

# # Display the map
map_plot


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




