#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # dala load and cleaning

# In[ ]:


# load data
df = pd.read_excel("/kaggle/input/covid-in-europe-time-peaksgenderage-group-study/11bb715b-9833-4eda-97e1-8ffa5a7810a5.xlsx")
df.head()


# In[ ]:


#check size of the datatset
df.shape


# In[ ]:


#clean data: drop rows, containing unknown or empty age group and gender
df_clean= df.drop(df[(df['Age group'] == "-") | (df['Gender'] == "-") | (df['Gender'] == "Unknown")].index)
df_clean.shape


# In[ ]:


#in order to make dataframe better plotable, I set up datetime format
df_clean["week"] = df_clean["Reporting week"].str.extract('.*(\d{2})', expand = False) 
df_clean.head()


# In[ ]:


df_clean["year"] = [2020]*47068
df_clean.head()


# In[ ]:


#create date column in datetime format 
df_clean['date'] = pd.to_datetime(df_clean.week.astype(str)+
                           df_clean.year.astype(str).add('-2') ,format='%W%Y-%w')
df_clean.date.head()


# In[ ]:


# drop unnessessary columns ['Reporting week',"Onset week", "Source", "week", "year"]
# drop columns "Cases" and "Deaths", as they contain only one "0" value
df_clean.drop(['Reporting week',"Onset week", "Source", "week", "year", "Cases" , "Deaths"], axis=1, inplace=True) 

df_clean.head()


# # first descriptive study

# In[ ]:


# genaral descriptive statistics
df_clean.describe(include='all')


# Data includes info on 47068 confirmed covid cases, from 26 EU countries, from  22 weeks period, 
# describe 7 age groups and gender info. For part of the data it is known about hospitalisation and if intensive care were applyed. Also sourse of infection (from which country patient was infected).

# In[ ]:


df_clean["Reporting country"].value_counts().head(10)


# 
# The most of the confirmed cases are reported from Germany... It is actually different from media news, which we used to know

# In[ ]:


df_clean["Age group"].value_counts()


# Most of the confirmed cases belong to "25-49" and "50-64" age groups.

# In[ ]:


df_clean["Gender"].value_counts()


# in general there are more confirmed cases for males

# In[ ]:


# available data about hospitalisation
df_clean["Hospitalisation"].value_counts()


# In[ ]:


df_clean["Intensive care"].value_counts()


# There are 27085 unknown cases, but also exact indication, that  3348 patienten were treated by ICU 
# (Intensive care unit) and 16635 patienten were treated without. Later, i will look to % of deaths in those both groups

# In[ ]:


df_clean["Outcome"].value_counts()


# I find extremely valuable, that Outcome cases contains not only "Fatal", but also "Died from other causes".

# In[ ]:


df_clean["Country of infection"].value_counts().head(10)
# would be interesting to map the migrations for covid infection


# # Visualisation

# # Question 1. Define peak of the pandemy in Europe over time

# In[ ]:


# I would like to check development of all possible outcomes for patientes with confirmed Covid-19 in time
# for this I first make a grouping 
time_outcome_count= df_clean.groupby(['date',"Outcome"]).size().reset_index(name='count')
time_outcome_count.head()


# In[ ]:


#plot possible Outcomes for patientes with confirmed covid-19 in Europe: development over time

import matplotlib.pyplot as plt

time_outcome_count.pivot_table(index="date", columns="Outcome", values="count", fill_value=0).plot                                                                    (figsize=(13, 6)).                                                                                        legend(loc=2, 
                                                                                     fontsize=  'large',
                                                                                     edgecolor="None", 
                                                                                     borderpad=1.5,
                                                                                    title="Outcomes",
                                                                                     title_fontsize=14)

plt.rc('axes', edgecolor='gray')  # axis color
plt.ylabel('Number of cases', fontsize=12).set_color('gray')
plt.xlabel("Reporting date", fontsize=12).set_color('gray')
plt.tick_params(axis='x', labelsize=9, colors='gray', which='both')
plt.tick_params(axis='y', labelsize=9, colors='gray', which='both')

plt.annotate('The data provided by \n European Centre for Disease Prevention and Control', 
             xy=(0.98, 0.9), xycoords='axes fraction', size=10,
             ha='right', #horisontal alignment
             va='baseline').set_color('gray')# add text about origin of data
plt.title('Weekly development of outcomes for patientes with confirmed Covid-19 in Europe', fontsize=16)


# The peak of the covid-19 pandemie in European countries reffer to beginning of April.

# # Questtion 2. Total number of of confirmed covid-19 patients and their outcomes in Europe

# In[ ]:


# in order to look at number of confirmed covid-19 patients and their outcomes in European countries
# first step  is grouping:
df_clean.groupby(["Reporting country","Outcome"]).size().reset_index(name='count').head()


# In[ ]:


# pivot this grouping table for plotting
country_outcomes_pivot = df_clean.groupby(["Reporting country","Outcome"]).size().reset_index(name='count').        pivot_table(index="Reporting country", columns='Outcome', values="count", fill_value=0)
country_outcomes_pivot.head()


# In[ ]:


# sort pivot table adscending by sum of all columns (total nb of registred cases) and 
# plot with horisontal stacked bars
country_outcomes_pivot.assign(tmp=country_outcomes_pivot.sum(axis=1)).sort_values('tmp', ascending=True).drop('tmp', 1).plot.barh(stacked=True, figsize=(12, 7), width=0.8).legend(fontsize='small',#loc=1,
       edgecolor="None", borderpad=1.2)
plt.rcParams["legend.labelspacing"] = 0.02 # vertical space between legend entries
plt.xlabel('Number of cases', fontsize=10)
plt.annotate('The data provided by European Centre for Disease Prevention and Control', 
             xy=(0.03, 0.02), xycoords='axes fraction', size=10).set_color('gray')# add text about origin of data
plt.title('Outcomes for patientes with confirmed Covid-19', fontsize=16)


# It seems like the best documented data are coming from Germany. Therefore, later I would like to study this part of the dataset with more details. Especially strange looks data from Italy, Netherlands, Denmark, UK, France, Belgium... In athe next steps I would exclude those countries, and look partition "all confirmed covid cases" to "death cases" in the rest of the countries.

# # Question 3. Is pandemy peak is varying over the Europe? Plot "all confirmed covid cases" and "death cases" in the selected European countries.

# In[ ]:


# reshape df: count cumularive cases number per reporting week
country_date_outcome = df_clean.groupby(["Reporting country","date","Outcome"]).size().reset_index(name='Count').pivot_table(index=["Reporting country","date"], columns="Outcome", values="Count", fill_value=0).reset_index()

country_date_outcome = country_date_outcome.assign(Total_cases=country_date_outcome.sum(axis=1))
country_date_outcome.head()


# In[ ]:


#subset selected countries and columns
country_date_outcome[(country_date_outcome["Reporting country"]==
                      "Austria")|(country_date_outcome["Reporting country"]== 
                    "Germany")#|(country_date_outcome["Reporting country"]== "Italy")
                    |(country_date_outcome["Reporting country"]== 
                    "Czechia")|(country_date_outcome["Reporting country"]== 
                    "Poland")|(country_date_outcome["Reporting country"]== 
                    "Portugal")|(country_date_outcome["Reporting country"]== 
                    "Finland")#|(country_date_outcome["Reporting country"]==  "Netherlands")
                   |(country_date_outcome["Reporting country"]== 
                    "Norway")|(country_date_outcome["Reporting country"]== "Sweden")][['Reporting country', 
                                'date','Fatal',
                                'Total_cases']].head()


# In[ ]:


# melt 2 columns (Fatal and Total_cases) in one (Outcome) with value equal to Fatal or Total_cases value
table_to_plot=country_date_outcome[(country_date_outcome["Reporting country"]==
                      "Austria")|(country_date_outcome["Reporting country"]== 
                    "Germany")#|(country_date_outcome["Reporting country"]== "Italy")
                    |(country_date_outcome["Reporting country"]== 
                    "Czechia")|(country_date_outcome["Reporting country"]== 
                    "Poland")|(country_date_outcome["Reporting country"]== 
                    "Portugal")|(country_date_outcome["Reporting country"]== 
                    "Finland")#|(country_date_outcome["Reporting country"]==  "Netherlands")
                   |(country_date_outcome["Reporting country"]== 
                    "Norway")|(country_date_outcome["Reporting country"]== "Sweden")][['Reporting country', 
                                'date','Fatal',
                                'Total_cases']].melt(id_vars=['Reporting country', 'date']).rename(columns={"value": "Confirmed cases"})
table_to_plot.head()


# In[ ]:


#plot facets with seaborn
import seaborn as sns
import matplotlib.pyplot as plt
#sns.set(style="ticks") #to get rig gray background
#sns.set_context("paper", rc={"axes.labelsize":16}) 
sns.set(rc={"font.size":20,"axes.titlesize":18,"axes.labelsize":14},style="ticks")

g = sns.relplot(x="date", y="Confirmed cases",
                 col="Reporting country", 
                hue="Outcome",#style="Outcome",
                 kind="line", 
                legend=False, 
                data=table_to_plot, col_wrap=4,
                height=4, aspect=1)#.set(title = "Total amount of confirmed covid-19 cases to death cases in the selected European countries")

g.set_ylabels("Number of cases")
g.set_xlabels("Date")

#g.legend(fontsize='x-large', title_fontsize='20')
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('Total amount of confirmed Covid-19 cases and deaths in the selected European countries')

plt.legend(("Deaths", "Total cases"),loc='upper right',fontsize='medium',
           edgecolor="None", borderpad=1 )


# While overall peak of the pandemie in European countries correspond to beginning of April, there is a shift to end of April - beginning of May in Italy and Poland, and "flatted" peak extended from April to June in Sweden. Also for Sweden and Finland the smallest amount of cases are registered.

# # Question 4. Age groups distribution for patients with total confirmed covid-19 cases and death cases in selected European countries

# In[ ]:


# in order to plot age distribution in two subplots: total cases and fatal (death) cases, I create 2 pivot tables: 

#First, for selected countries:
country_age_total_pivot=df_clean[(df_clean["Reporting country"]==
                      "Austria")|(df_clean["Reporting country"]== 
                    "Germany")|(df_clean["Reporting country"]== 
                    "Czechia")|(df_clean["Reporting country"]== 
                    "Poland")|(df_clean["Reporting country"]== 
                    "Portugal")|(df_clean["Reporting country"]== 
                    "Finland")|(df_clean["Reporting country"]== 
                    "Norway")|(df_clean["Reporting country"]== "Sweden")].\
    groupby(["Reporting country","Age group"]).size().reset_index(name='count').\
    pivot_table(index="Reporting country", columns="Age group", values="count",
               fill_value=0)
country_age_total_pivot.head()


# In[ ]:


#Second, for selected countries and dead patiensts:
country_age_death_pivot = df_clean[(df_clean.Outcome == "Fatal")& ((df_clean["Reporting country"]==
                      "Austria")|(df_clean["Reporting country"]== 
                    "Germany")|(df_clean["Reporting country"]== 
                    "Czechia")|(df_clean["Reporting country"]== 
                    "Poland")|(df_clean["Reporting country"]== 
                    "Portugal")|(df_clean["Reporting country"]== 
                    "Finland")|(df_clean["Reporting country"]== 
                    "Norway")|(df_clean["Reporting country"]== "Sweden"))].\
groupby(["Reporting country","Age group"]).size().reset_index(name='count').\
pivot_table(index="Reporting country", columns="Age group", values="count",
               fill_value=0)
country_age_death_pivot.head()


# In[ ]:


#its occur, that one age group is missing in death dataframe. I set it to "0"
country_age_death_pivot['05-14']=0
country_age_death_pivot = country_age_death_pivot[['00-04','05-14' ,'15-24', '25-49', '50-64', '65-79', '80+']]
country_age_death_pivot.head()


# In[ ]:


#plot two pivots as subplots
import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=1, ncols=2)#, sharey=True)
fig.set_size_inches(15, 4)
plt.subplots_adjust(top=0.8)
fig.suptitle('Age groups for patients with confirmed Covid-19 in Europe', fontsize=20)
plt.annotate('The data provided by European Centre for Disease Prevention and Control', 
             xy=(0.1, 0.02), xycoords='axes fraction',  fontsize=10).set_color('gray')
 
country_age_total_pivot.assign(tmp=country_age_total_pivot.sum(axis=1)).sort_values('tmp', ascending=True).drop('tmp', 1).plot(kind="barh", stacked=True, ax=axes[0], legend=False).legend(fontsize=12,
      title="Age group", edgecolor="None")
axes[0].set_title("Total cases", fontsize=14)
axes[0].set_xlabel('Number of cases', fontsize=12)#.set_color('gray')
axes[0].set_ylabel( "Reporting country", fontsize=12)#.set_color('gray')

country_age_death_pivot.assign(tmp=country_age_death_pivot.sum(axis=1)).sort_values('tmp', ascending=True).drop('tmp', 1).plot(kind="barh", stacked=True, ax=axes[1]).legend(fontsize=12,#loc=3, 
    title="Age group",edgecolor="None", borderpad=1)
plt.rcParams["legend.facecolor"] = "None"
plt.rcParams["legend.framealpha"] = 0.1
axes[1].set_title("Fatal cases", fontsize=14)
axes[1].set_xlabel('Number of cases', fontsize=12)
axes[1].set_ylabel('')
axes[1].tick_params(axis='y', labelleft=False, left=True)

plt.subplots_adjust(wspace=0.05)


# There are small differences between size of groups for 15+ years old patients with confirmed covid-19 in presented European countries. There are more cases in age diapason 25-79 y.o. for most of the countries. Fatal cases show bigger risc for patients of over 65 years old. There are no registered deaths of patients of young patients younger 24 years.

# # Question 5. Is there differentiation between age group and gender: all cases vs deaths

# In[ ]:


# in order to plot age distribution in two subplots: total cases and fatal (death) cases, I create 2 pivot tables: 

#First, for all confirmed covid-19 cases:
ds_total = df_clean[["Age group","Gender"]].groupby(["Age group","Gender"]).size().reset_index(name='count').pivot_table(index="Age group", columns="Gender", values="count",
               fill_value=0)
ds_total.loc['Total',:] = ds_total.sum(axis=0)
ds_total


# In[ ]:


#Second, for all fatal covid-19 cases:
ds_death=df_clean[df_clean.Outcome == "Fatal"][["Age group","Gender"]].groupby(["Age group","Gender"]).size().reset_index(name='count').pivot_table(index="Age group", columns="Gender", values="count",
               fill_value=0)#.plot(kind="barh", stacked=False)
ds_death.loc['Total',:] = ds_death.sum(axis=0)
ds_death


# In[ ]:


#plot two pivots as subplots
fig, axes = plt.subplots(nrows=1, ncols=2)#, sharey=True)
fig.set_size_inches(12, 4)
plt.subplots_adjust(top=0.8)
fig.suptitle('Age vs gender groups for patients with confirmed Covid-19 in Europe', fontsize=18)
plt.annotate('The data provided by European Centre for Disease Prevention and Control', 
             xy=(-0.9, 0.02), xycoords='axes fraction',  fontsize=8).set_color('gray')

ds_total.plot(kind="barh", stacked=False, color=['tab:red', "tab:blue"], ax=axes[0]).legend(loc='lower right')#, 
                                                                                            #title="Gender")
axes[0].set_title("Total cases", fontsize=14)
axes[0].set_xlabel('Number of cases', fontsize=12)#.set_color('gray')
axes[0].set_ylabel('Age group', fontsize=12)

ds_death.plot(kind="barh", stacked=False, color=['tab:red', "tab:blue"], ax=axes[1]).legend(loc='lower right')#, title="Gender")
#plt.rcParams["legend.framealpha"] = 0.1
axes[1].set_title("Fatal cases", fontsize=14)
axes[1].set_xlabel('Number of cases', fontsize=12)#.set_color('gray')
axes[1].set_ylabel('')#.set_color('gray')
axes[1].tick_params(axis='y', labelleft=False, left=True)
plt.subplots_adjust(wspace=0.05)


# Left figure show, that frequency of confirmed covid-19 cases are more frequent between patients, who older than 25 y.o. Gender seems to play not important role for chance to get infections: its only minor differences between gender groups within all possible age groups. However male patients are dominated (ca 1000 cases!) in total number of registered covid-19 cases. This trend is more prominent for fatal cases, shown in right figure. Male patients with fatal outcome are dominated in all age groups, except of 80+ (possibly because there are in general less males, than females in that age group). In the presented dataset was no registered fatal cases for patients younger 25 years.

# # Question 6. Does intensive care unit influence to survival rate?

# In[ ]:


# create table to plot
icu= df_clean[(df_clean["Intensive care"] != "Unknown")][["Intensive care", 
                                                     "Outcome" ]].\
groupby(["Intensive care","Outcome"]).size().reset_index(name='count').\
pivot_table(index="Intensive care", columns="Outcome", values="count",fill_value=0)
icu


# In[ ]:


ax= icu.plot(figsize=(11,4), width=0.5,
            kind="barh", stacked=True)
ax.legend(fontsize='x-small',
          edgecolor="None", 
          borderpad=1.8)
# find the values in plot-patches, calculate respecive value in % and insert as text 
for i,j in enumerate (ax.patches):
    if j.get_width()>500:
        if i % 2: #even
            ax.text( j.get_x()+200,j.get_y()+0.2, str(round(j.get_width()*100/icu.sum(axis=1)[1], 0))+'%', fontsize=9) 
        else: #odd
            ax.text( j.get_x()+200,j.get_y()+0.2, str(round(j.get_width()*100/icu.sum(axis=1)[0], 0))+'%', fontsize=9)
                
plt.xlabel('Number of cases', fontsize=12)
plt.ylabel('', fontsize=14)
plt.annotate('The data provided by European Centre for Disease Prevention and Control', 
             xy=(0.36, 0.05), xycoords='axes fraction',  fontsize=10).set_color('gray')# add text about origin of data
plt.title("Outcomes by the ICU (Intensive care unit) treatment vs hospitalisation without ICU", fontsize=16)


# For known cases, where ICU was applyed, part of alive and death patients are comparable (ca 41% vs ca 31%). Obviously, ICU applyed for hard patients, however it is effective only in ca 31% of cases. Patients are died in 9% of cases without ICU. Pehaps they were fond not hard enought for ICU, or were hospitalzsed too late, or ICU was not possible for some reasons.

# # Conclusions:
# 
# **1. Timing peak of the pandemy**
# In general, the peak of the covid-19 pandemie in European countries reffer to beginning of April. There are regional differences: e.g. shifted peak to end of April - beginning of May in Italy and Poland, and "flatted" peak extended from April to June in Sweden. Moreover for Sweden and Finland the smallest amount of cases are registered.
#  
# **2. Infection risc for different age groups**
# There are small differences between size of groups for  patients older than 15 y.o. The most of confirmed covid-19 cases are in age diapason 25-79 y.o. for presented european countries. Fatal cases show bigger risc for patients of over 65 years old. There are no registered deaths of young patients younger 25 years.
# Total frequency of confirmed covid-19 cases are more often between patients, who older than 25 y.o.
# 
# **3. Infection rate between age group and gender**
# In general, the available dataset doesn not show big differences in frequency of infection or death for males and females of varying age groups.  However male patients are dominated (ca 1000 cases!) in total number of registered covid-19 cases. This trend is more prominent for fatal cases: male patients with fatal outcome are dominated in all age groups, except of 80+ (possibly because there are in general less males, than females in that age group). 
# 
# **4. How intensive care unit influence to survival rate**
# There are death cases in both, ICU and no ICU groups. For known cases, where ICU was applyed, it was effective in over 41% of cases, while 31% of patients were dead. Without ICU patients were died in 9% of cases. Pehaps, those part of patients, who died without ICU, were diagnosted or/and hospitalzsed too late, or ICU was not possible for some reasons.
