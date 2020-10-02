#!/usr/bin/env python
# coding: utf-8

# ### Index
# 1. [Getting the Top 5 countries participated in the survey](#Getting-the-Top-5-countries-participated-in-the-survey)
# 1. [Analyzing only Top 2 participating countries - India and USA](#Analyzing-only-Top-2-participating-countries---India-and-USA)
# 1. [Analyzing users on the different age groups of these countries](#Analyzing-users-on-the-different-age-groups-of-these-countries)
# 1. [Degrees of USA and Indian Kagglers ](#Degrees-of-USA-and-Indian-Kagglers)
# 1. [Professional role of Kagglers](#Professional-role-of-Kagglers)
# 1. [Comparing the compensation provided in both countries](#Comparing-the-compensation-provided-in-both-countries)
# 1. [Popular IDEs in both countries](#Popular-IDEs-in-both-countries)
# 1. [Which programming language is used most often?](#Which-programming-language-is-used-most-often?)
# 1. [Type of data encountered at work/school](#Type-of-data-encountered-at-work/school)
# 1. [Time spent by kagglers at various phases of Data Science Project](#Time-spent-by-kagglers-at-various-phases-of-Data-Science-Project)
#      * [Gathering Data](#1.-Gathering-Data)
#      * [Cleaning Data](#2.-Cleaning-Data)
#      * [Data Visualization](#3.-Data-Visualization)
#      * [Model building](#4.-Model-building-/-Model-selection)
#      * [Putting the model into production](#5.-Putting-the-model-into-production)
#      * [Finding insights](#6.-Finding-insights-in-the-data-and-communicating-with-stakeholders)
# 1. [MOOCs Popularity](#MOOCs-Popularity)
# 1. [Frameworks used in last 5 years](#In-the-below-sections-the-data-is-compared-on-the-basis-of-their-use-in-last-5-years)
#     * [Data Visualization Libraries](#Data-Visualization-Libraries)
#     * [Hosted Notebooks](#Hosted-Notebooks)
#     * [Big Data and Analytics](#Big-Data-and-Analytics-Products)

# In[ ]:


# Importing the libraries
get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Customizing the plot style
plt.style.use('seaborn')


# In[ ]:


mcq_responses_df = pd.read_csv('../input/multipleChoiceResponses.csv', header=1, low_memory=False)


# ## Getting the Top 5 countries participated in the survey

# In[ ]:


# Groupping the data by countries
country = mcq_responses_df.groupby(mcq_responses_df.columns[4])[mcq_responses_df.columns[0]].count()
country = country.drop('Other')

# Getting the Top 5 countries on the basis of count
country = country.nlargest(5)

# Customizing the plot size
plt.rcParams['figure.figsize'] = [10, 8]

# Plotting the dataframe
country.plot(kind='bar', rot=0)
plt.xticks(fontsize=11)
plt.yticks(country[0:], fontsize=11)
plt.xlabel('Country', fontsize=14)
plt.ylabel('Total Developers (in survey)', fontsize=14)
plt.show()


# ## Analyzing only Top 2 participating countries - India and USA

# In[ ]:


# Counting the Genders in USA
gender_usa = mcq_responses_df.loc[mcq_responses_df[mcq_responses_df.columns[4]] == 'United States of America',
                                  mcq_responses_df.columns[1]]

male_usa, female_usa = gender_usa.value_counts()[0:2]

# Counting the Genders in India
gender_india = mcq_responses_df.loc[mcq_responses_df[mcq_responses_df.columns[4]] == 'India',
                                  mcq_responses_df.columns[1]]

male_india, female_india = gender_india.value_counts()[0:2]

gender_india_usa = pd.DataFrame({'Country':['United States of America', 'United States of America',
                                           'India', 'India'], 
                                'Gender':['Male', 'Female', 'Male', 'Female'],
                                'Count': [male_usa, female_usa, male_india, female_india]})

# Customizing the plot size
plt.rcParams['figure.figsize'] = [8, 6] 
sns.barplot(x='Country', y='Count', hue='Gender', data=gender_india_usa, 
            palette='deep')
plt.yticks(gender_india_usa.loc[:, 'Count'])
plt.show()


# ### Observations:
# * **`United States has more number of female Kagglers than India`**
# * **`India outperforms in the term of male Kagglers`**

# ## Analyzing users on the different age groups of these countries

# In[ ]:


# Getting age groups of USA
age_group_usa = mcq_responses_df.loc[mcq_responses_df[mcq_responses_df.columns[4]] == 'United States of America',
                                     mcq_responses_df.columns[3]]
age_group_usa = age_group_usa.value_counts()

# Getting age groups of India
age_group_india = mcq_responses_df.loc[mcq_responses_df[mcq_responses_df.columns[4]] == 'India',
                                  mcq_responses_df.columns[3]]
age_group_india = age_group_india.value_counts()

# Customizing the plot size
plt.rcParams['figure.figsize'] = [15, 7.5]
# Custom colors
colors = ['#94DCF1', '#8AD5EB', '#75C4DB', '#6AB9D0',
          '#5AB4CF', '#48A9C5', '#35A9CB', '#179CC3',
          '#2696B8', '#1F8FB0', '#168BAD', '#1481A1']


def annotate_bar_percentage(series):
    series_length = len(series.index)
    total = sum(series[0:])
    
    for i in range(series_length):
        plt.annotate(str(round(series[series_length - (i+1)] * 100 / total, 2)) + '%', 
                     xy=(0,0), xytext=(0, series_length-i-1),
                    fontsize=11)

# Plotting USA Age Group
plt.subplot(1, 2, 1)
plt.barh(age_group_usa.index, age_group_usa[0:], color=colors)
plt.xlabel('Frequency', fontsize=12)
plt.ylabel('Age Group')
plt.title('USA', fontsize=14)
annotate_bar_percentage(age_group_usa)

# Plotting India Age Group
plt.subplot(1, 2, 2)
plt.barh(age_group_india.index, age_group_india[0:], color=colors)
plt.xlabel('Frequency', fontsize=12)
plt.title('India', fontsize=14)
annotate_bar_percentage(age_group_india)
plt.show()


# ### Observations:
# *  **`India has more share of young Kagglers in 18-21 and 22-24 segment`**
# * **`USA has even more number of Kaggler shares after 40 years of age than India`** 
# * **`What surprises that US has even Kagglers of 80+, keep going`**
# 
# ### From the above plots we can conclude that the young generation is preparing to drive India through Data Science.

# ## Degrees of USA and Indian Kagglers 

# In[ ]:


# USA Kagglers Degree
degree_usa = mcq_responses_df.loc[mcq_responses_df[mcq_responses_df.columns[4]] == 'United States of America',
                                  mcq_responses_df.columns[5]]

degree_usa = degree_usa.value_counts()
degree_usa = degree_usa.drop(labels='I prefer not to answer')
degree_usa.index = ['Masters', 'Bachelors', 'Doctoral', 'Dropouts', 'Professional', 'High School']

# Indian Kagglers Degree
degree_india = mcq_responses_df.loc[mcq_responses_df[mcq_responses_df.columns[4]] == 'India',
                                  mcq_responses_df.columns[5]]

degree_india = degree_india.value_counts()
degree_india = degree_india.drop(labels='I prefer not to answer')
degree_india.index = degree_usa.index

# Customizing the plot size
plt.rcParams['figure.figsize'] = [11.5, 4.5]
# Custom color
colors = ['#14C4A9', '#16B9A0', '#15A48E',
          '#2A9383', '#1D8374', '#148C7B']

# Plotting USA Degrees
plt.subplot(1, 2, 1)
plt.barh(degree_usa.index, degree_usa[0:],
         color=colors, alpha=0.75)
annotate_bar_percentage(degree_usa)
plt.title('USA', fontsize=14)
plt.yticks(fontsize=11.25)

# Plotting India Degrees
plt.subplot(1, 2, 2)
plt.barh(degree_india.index, degree_india[0:],
         color=colors, alpha=0.75)
annotate_bar_percentage(degree_india)
plt.title('India', fontsize=14)
plt.yticks(fontsize=11.25)
plt.tight_layout()
plt.show()


# ### Observations:
# * **`There are more number of Doctoral Kagglers in USA`**
# * **`USA lags behind in terms of Bachelors Degree Kagglers in India`**
# 
# ## *Truly Youth is the new emerging power of India*

# ## Professional role of Kagglers

# In[ ]:


def profession_df(country):
    profession = mcq_responses_df.loc[mcq_responses_df[mcq_responses_df.columns[4]] == country,
                                   mcq_responses_df.columns[7]]
    profession = profession.value_counts(dropna=False)
    profession = pd.DataFrame({'Role':profession.index, 'Count':profession[0:]})
    profession = profession.dropna()
    profession = profession[profession['Role'] != 'Other']
    profession = profession.nlargest(10, columns='Count')
    profession = profession.sort_values(by='Role')
    return profession

# Generating profession dataframe
profession_usa = profession_df('United States of America')
profession_india = profession_df('India')

# Customizing the plot size
plt.rcParams['figure.figsize'] = [10, 12]
# Custom color
colors1 = ['#2980B9', '#3498DB', '#1ABC9C', '#16A085', '#27AE60', 
          '#2ECC71', '#F1C40F', '#F39C12', '#E67E22', '#D35400']

# Plotting USA roles
plt.subplot(2, 1, 1)
plt.pie(profession_usa['Count'], labels=profession_usa['Role'],
        autopct='%1.1f%%', colors=colors1)
plt.axis('equal')
plt.title('USA', fontsize=16)

# Plotting India roles
plt.subplot(2, 1, 2)
plt.pie(profession_india['Count'], labels=profession_india['Role'],
        autopct='%1.1f%%', colors=colors1)
plt.axis('equal')
plt.title('India', fontsize=16)
plt.show()


# ### Observations:
# * **`There are more number of student Kagglers in India than USA`** 
# *I think [D J Patil](https://www.youtube.com/watch?v=UuAJMzpoq5E) is the source of inspiration*
# * **`USA has large share of Data Scientists while India has more number of Software Engineer`**

# ## Comparing the compensation provided in both countries
# ### The trending roles as per the current industry demand:
# * Data Scientist
# * Data Engineer
# * Data Analyst
# * Software Engineer

# In[ ]:


def compensation_df(country, profile):
    compensation = mcq_responses_df.loc[(mcq_responses_df[mcq_responses_df.columns[4]] == country) &
                                            (mcq_responses_df[mcq_responses_df.columns[7]] == profile),
                                             mcq_responses_df.columns[12]]
    
    compensation = compensation.value_counts(dropna=False)
    compensation = compensation.drop(index=['I do not wish to disclose my approximate yearly compensation',None])
    
    # Sorting rows by values before '-'
    indices = compensation.index
    num_bef_delim = []
    for ind in indices:
        if ind.find('-') > 0:
            num_bef_delim.append(int(ind[:ind.find('-')]))
        else:
            num_bef_delim.append(int(ind[:3]))
    compensation = pd.DataFrame({'Salary_Range':num_bef_delim, 'Count': compensation[0:]})
    compensation = compensation.sort_values(by='Salary_Range')
    return compensation

# Data Scientists USA and India
profiles = ['Data Scientist', 'Data Engineer', 'Data Analyst', 'Software Engineer']

# Customizing the plot size
plt.rcParams['figure.figsize'] = [20, 20]
# Custom color
colors2 = ['#5499C7', '#EC7063', '#F1C40F', '#D35400',
          '#CD6155', '#3DB948', '#C0392B', '#239B56']

for profile in range(len(profiles)):
    compensation_usa = compensation_df('United States of America', profiles[profile])
    compensation_india = compensation_df('India', profiles[profile])
    plt.subplot(2, 2, profile+1)
    plt.plot(compensation_usa['Salary_Range'], compensation_usa['Count'], 
             label='USA', marker='^', color=colors2[profile])
    plt.plot(compensation_india['Salary_Range'], compensation_india['Count'], 
             label='India', marker='*', color=colors2[profile+1])
    plt.title(profiles[profile]+' Salary USA and India', fontsize=20)
    plt.xlabel('Salary Range x 1000 (in $)', fontsize=14)
    plt.ylabel('No. of Kagglers', fontsize=14)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=16)
    plt.legend()    


# ### Observations:
# 
# #### 1. Data Scientist Graph
# * **`Data Scientists are more in 0-100k$  in India than USA`**
# * **`USA has a high frequency of Data Scientists in the range 100-200k$ than India`**
# 
# #### 2. Data Engineer Graph
# * **`Data Engineers are earning more in USA than India`**
# * **`India has more Engineers in the range 0-50k$`**
# 
# #### 3. Data Analyst Graph
# * **`USA has more number of Data Analysts but the salary scale is not good as compared to India`**
# * **`India has professionals that are earning more than 300k$ `**
# 
# #### 4. Software Engineer Graph
# * **`India has more number of Software Engineers but lags behind USA in terms of salary`**
# * **`Indian Engineers are in the range 0-100k$, and the number is quite high in mid of this range`**

# ## Popular IDEs in both countries

# In[ ]:


def generate_df(country, index1, index2, col_name):
    df = mcq_responses_df.loc[(mcq_responses_df[mcq_responses_df.columns[4]] == country),
                               mcq_responses_df.columns[index1]:mcq_responses_df.columns[index2]]
    old_cols = df.columns
    new_cols = []
    
    for i in old_cols:
        t = i.split()[::-1]
        new_name = []
        for j in t:
            if j == '-':
                break
            else:
                new_name.append(j)
        new_cols.append(''.join(new_name[::-1]))
    
    # Updating the column names
    df.columns = new_cols
    # Generating the dataframe
    df = pd.concat([pd.DataFrame([[i, df[i].value_counts()[0]]], columns=[col_name, 'Count']) 
                        for i in df.columns], ignore_index=True)
    
    df = df.sort_values(by=col_name)
    df = df.reset_index(drop=True)
    return df

ide_usa = generate_df('United States of America', 29, 41, 'IDE')
ide_india = generate_df('India', 29, 41, 'IDE')

# Customizing the plot size
plt.rcParams['figure.figsize'] = [12, 14]

# Plotting USA IDEs
plt.subplot(2, 1, 1)
plt.pie(ide_usa['Count'], labels=ide_usa['IDE'], 
        autopct='%1.1f%%', colors=colors1 + ['#e74c3c', '#d0d3d4'])
plt.title('USA', fontsize=18)
plt.axis('equal')

# Plotting India IDEs
plt.subplot(2, 1, 2)
plt.pie(ide_india['Count'], labels=ide_india['IDE'], 
        autopct='%1.1f%%', colors=colors1 + ['#e74c3c', '#d0d3d4'])
plt.title('India', fontsize=18)
plt.axis('equal')
plt.show()


# ### Observations:
# * **`Jupyter is used in majority, in both coutries and the users are nearly equal`**
# * **`RStudio users are more in USA`**
# * **`Notepad++ is the second most used IDE in India`**
# * **`Users of MATLAB, Visual Studio and Visual Studio Code are nearly equal`**

# ## Programming language used in both countries on a regular basis

# In[ ]:


# Generating programming language dataframe of both countries
language_usa = generate_df('United States of America', 65, 80, 'ProgrammingLanguage') # generate_df defined in above section
language_india = generate_df('India', 65, 80, 'ProgrammingLanguage')

# Customizing the plot size
plt.rcParams['figure.figsize'] = [15, 12]

# Figure Title
plt.suptitle('Programming Language users in USA and India', fontsize=20)

# USA Programming Language Plot
plt.subplot(1, 2, 1)
plt.barh(language_usa['ProgrammingLanguage'], language_usa['Count'], color=colors2)
plt.title('USA', fontsize=16)
annotate_bar_percentage(language_usa['Count'])

# India Programming Language Plot
plt.subplot(1, 2, 2)
plt.barh(language_india['ProgrammingLanguage'], language_india['Count'], color=colors2)
plt.title('India', fontsize=16)
annotate_bar_percentage(language_india['Count'])
plt.show()


# ### Observations:
# * **`Python is the most used programming language in both countries`**
# * **`SQL has more users than R in both the graphs and it is used more in USA`**
# * **`Users of C/C++ is quite high in India`**

# ## Which programming language is used most often?

# In[ ]:


# Frequent Language USA
frequent_language_usa = mcq_responses_df.loc[(mcq_responses_df[mcq_responses_df.columns[4]] == 'United States of America'),
                               mcq_responses_df.columns[84]]
frequent_language_usa = frequent_language_usa.value_counts().nlargest(10)

# Frequent Language India
frequent_language_india = mcq_responses_df.loc[(mcq_responses_df[mcq_responses_df.columns[4]] == 'India'),
                               mcq_responses_df.columns[84]]
frequent_language_india = frequent_language_india.value_counts().nlargest(10)

# Customizing the plot size
plt.rcParams['figure.figsize'] = [14, 8]

# Figure Title
plt.suptitle('Top 10 Programming Languages used by Kagglers in USA and India frequently', fontsize=18)

# Plotting frequent languages USA
plt.subplot(1, 2, 1)
plt.pie(frequent_language_usa[0:], labels=frequent_language_usa.index, 
        autopct='%1.1f%%', colors=colors1)
plt.title('USA', fontsize=16)
plt.axis('equal')

# Plotting frequent languages India
plt.subplot(1, 2, 2)
plt.pie(frequent_language_india[0:], labels=frequent_language_india.index,
        autopct='%1.1f%%', colors=colors1)
plt.title('India', fontsize=16)
plt.axis('equal')
plt.show()


# ### Observations:
# * **`Python is used more frequently in India than USA`**
# * **`R is the runner-up in frequent use`**
# * **`SQL is most frequently used than Java in USA on the other hand it is opposite in India`**

# ## Type of data encountered at work/school

# In[ ]:


# Data Type USA
data_usa = generate_df('United States of America', 250, 260, 'Data_Type')
data_usa['Data_Type'] = data_usa['Data_Type'].str[:-4]

# Data Type India
data_india = generate_df('India', 250, 260, 'Data_Type')
data_india['Data_Type'] = data_india['Data_Type'].str[:-4]

# Customizing the plot size
plt.rcParams['figure.figsize'] = [15, 8]

# Plotting USA data type
plt.subplot(1, 2, 1)
plt.pie(data_usa['Count'], labels=data_usa['Data_Type'], 
        autopct='%1.1f%%', colors=colors1)
plt.title('USA', fontsize=16)
plt.axis('equal')

# Plotting India ML Frameworks
plt.subplot(1, 2, 2)
plt.pie(data_india['Count'], labels=data_india['Data_Type'], 
        autopct='%1.1f%%', colors=colors1)
plt.title('India', fontsize=16)
plt.axis('equal')
plt.show()


# ### Observations:
# * **`Numerical data has major share in both countries, it is slightly more in USA`**
# * **`India has less share of Genetic and Geospatial data than USA`**
# * **`USA and India both have approximately equal share of Sensor data`**. India will be the upcoming leader in IoT.

# ## Time spent by kagglers at various phases of Data Science Project

# ### 1. Gathering Data

# In[ ]:


def plot_time_spent(index):
    # Time spent in USA
    time_usa = mcq_responses_df.loc[mcq_responses_df[mcq_responses_df.columns[4]] == 'United States of America',
                                    mcq_responses_df.columns[index]]
    time_usa = time_usa.dropna()
    time_usa_mean, time_usa_mode = time_usa.mean(), time_usa.mode()[0]

    # Time spent in India
    time_india = mcq_responses_df.loc[mcq_responses_df[mcq_responses_df.columns[4]] == 'India',
                                      mcq_responses_df.columns[index]]
    time_india = time_india.dropna()
    time_india_mean, time_india_mode = time_india.mean(), time_india.mode()[0]

    # Plotting mean
    plt.bar(np.array([1, 2]), [time_usa_mean, time_india_mean], 
            label='mean', width=0.15, color='#7EACA3')
    # Plotting mode
    plt.bar(np.array([1, 2]) + 0.15, [time_usa_mode, time_india_mode],
            label='mode', width=0.15, color='#AB7330')

    # Annotation
    plt.annotate(round(time_usa_mean, 1), xy=(1,0), xytext=(0.94, time_usa_mean-0.7),
                 color='#ffffff', fontsize=12.5)
    plt.annotate(str(round(np.sum(time_usa == time_usa_mode)/ len(time_usa) * 100, 1))+'%', 
                 xy=(1.15, time_usa_mode), xytext=(1.25, time_usa_mode+3), 
                 arrowprops={'color':'#221155' }, fontsize=12.5)
    plt.annotate(round(time_india_mean, 1), xy=(2, 0), xytext=(1.94, time_india_mean-0.7),
                 color='#ffffff', fontsize=12.5)
    plt.annotate(str(round(np.sum(time_india == time_india_mode)/ len(time_india) * 100, 1))+'%', 
             xy=(2.15, time_india_mode), xytext=(2.20, time_india_mode+3), 
             arrowprops={'color':'#661155' }, fontsize=12.5)
    plt.xticks(np.array([1, 2]) + 0.075, ['USA', 'India'], fontsize=12.5)
    plt.ylabel('Time spent(in %)', fontsize=12)
    plt.legend(loc='upper center', fontsize=12, frameon=True)
    plt.show()

# Customizing plot size
plt.rcParams['figure.figsize'] = [6, 6]

# Gathering Data
plot_time_spent(277)


# ### Observations:
# * **`On an average 16.9% of time spent by a kaggler in USA`**
# * **`Indian kagglers spent 17.2% of their time in gathering data`**
# * **`27.8% of the kagglers spent 10% of their time in USA`**
# * **`In India 31.8% of the kagglers also spent 10% of their time`**

# ### 2. Cleaning Data

# In[ ]:


# Cleaning Data
plot_time_spent(278)


# ### Observations:
# * **`On an average 24.9% of time spent by a kaggler in USA`**
# * **`Indian kagglers spent 21.9% of their time in data cleaning`**
# * **`19.1% of the kagglers spent 20% of their time in USA`**
# * **`In India 22.9% of the kagglers also spent 20% of their time`**

# ### 3. Data Visualization

# In[ ]:


# Data Visualization
plot_time_spent(279)


# ### Observations:
# * **`On an average 13.8% of time spent by a kaggler in USA`**
# * **`Indian kagglers spent 14.4% of their time in data visualization`**
# * **`33.7% of the kagglers spent 10% of their time in USA`**
# * **`In India 35.9% of the kagglers also spent 10% of their time`**

# ### 4. Model building / Model selection

# In[ ]:


# Model building
plot_time_spent(280)


# ### Observations:
# * **`On an average 19.4% of time spent by a kaggler in USA`**
# * **`Indian kagglers spent 20.4% of their time in model building`**
# * **`21.1% of the kagglers spent 20% of their time in USA`**
# * **`In India 24.3% of the kagglers also spent 20% of their time`**

# ### 5. Putting the model into production

# In[ ]:


# Model production
plot_time_spent(281)


# ### Observations:
# * **`On an average 7.5% of time spent by a kaggler in USA`**
# * **`Indian kagglers spent 10.1% of their time in model production`**
# * **`40.1% of the kagglers spent no time in USA in this phase`**
# * **`In India 34.8% of the kagglers spent 10% of their time`**

# ### 6. Finding insights in the data and communicating with stakeholders

# In[ ]:


# Finding insights in the data
plot_time_spent(282)


# ### Observations:
# * **`On an average 12.6% of time spent by a kaggler in USA`**
# * **`Indian kagglers spent 11.0% of their time in finding insights`**
# * **`27.1% of the kagglers spent 10% of their time in USA`**
# * **`In India 32.2% of the kagglers also spent 10% of their time`**

# ## MOOCs Popularity

# In[ ]:


# MOOC USA
mooc_usa = generate_df('United States of America', 291, 301, 'MOOCs')

# MOOC India
mooc_india = generate_df('India', 291, 301, 'MOOCs')

# Customizing the plot size
plt.rcParams['figure.figsize'] = [16, 10]

# Plotting USA ML Frameworks
plt.subplot(1, 2, 1)
plt.barh(mooc_usa['MOOCs'], mooc_usa['Count'], color=colors1)
plt.title('USA', fontsize=16)
annotate_bar_percentage(mooc_usa['Count'])

# Plotting India ML Frameworks
plt.subplot(1, 2, 2)
plt.barh(mooc_india['MOOCs'], mooc_india['Count'], color=colors1)
plt.title('India', fontsize=16)
annotate_bar_percentage(mooc_india['Count'])
plt.show()


# ### Observations:
# * **`Coursera is most popular in both countries, because it offers specialization from prestigious universities`**
# * **Siraj Raval's [The School.AI](https://www.theschool.ai/) is getting popularity in India than USA**
# * **`Kaggle Learn is more popular in India among kagglers than USA`**

# # In the below sections the data is compared on the basis of their use in last 5 years

# ## Machine Learning Frameworks

# In[ ]:


# ML Frameworks USA
ml_framework_usa = generate_df('United States of America', 88, 104, 'MLFramework')

# ML Frameworks India
ml_framework_india = generate_df('India', 88, 104, 'MLFramework')

# Customizing the plot size
plt.rcParams['figure.figsize'] = [14, 8]

# Plotting USA ML Frameworks
plt.subplot(1, 2, 1)
plt.barh(ml_framework_usa['MLFramework'], ml_framework_usa['Count'], color=colors1)
plt.title('USA', fontsize=16)
annotate_bar_percentage(ml_framework_usa['Count'])

# Plotting India ML Frameworks
plt.subplot(1, 2, 2)
plt.barh(ml_framework_india['MLFramework'], ml_framework_india['Count'], color=colors1)
plt.title('India', fontsize=16)
annotate_bar_percentage(ml_framework_india['Count'])
plt.show()


# ### Observations:
# * **`Scikit-Learn is the most used popular ML Framework in 5 years in both countries`**. Increase in Python users and well-maintained documentation of scikit-learn are responsible for its popularity.
# * **`User shares are nearly equal for all the libraries`**

# ## Data Visualization Libraries

# In[ ]:


# Visualization Library USA
library_usa = generate_df('United States of America', 110, 120, 'data_visual_lib')

# Visualization Library USA
library_india = generate_df('India', 110, 120, 'data_visual_lib')

# Customizing the plot size
plt.rcParams['figure.figsize'] = [14, 7.5]

# Plotting USA library
plt.subplot(1, 2, 1)
plt.pie(library_usa['Count'], labels=library_usa['data_visual_lib'], 
        autopct='%1.1f%%', colors=colors1)
plt.title('USA', fontsize=16)
plt.axis('equal')

# Plotting USA library
plt.subplot(1, 2, 2)
plt.pie(library_india['Count'], labels=library_india['data_visual_lib'],
       autopct='%1.1f%%', colors=colors1)
plt.title('India', fontsize=16)
plt.axis('equal')
plt.show()


# ### Observations:
# * **`Matplotlib is the most used library in both countries and is used more in India`**
# * **`Seaborn is used more in India than USA`**
# * **`Plotly is used equally in both countries`**

# ## Hosted Notebooks

# In[ ]:


# Notebooks USA
notebooks_usa = generate_df('United States of America', 45, 53, 'Notebook')

# Notebooks USA
notebooks_india = generate_df('India', 45, 53, 'Notebook')

# Customizing the plot size
plt.rcParams['figure.figsize'] = [12, 16]

# Plotting USA notebooks
plt.subplot(2, 1, 1)
plt.pie(notebooks_usa['Count'], labels=notebooks_usa['Notebook'], 
        autopct='%1.1f%%', colors=colors1)
plt.title('USA', fontsize=16)
plt.axis('equal')

# Plotting India notebooks
plt.subplot(2, 1, 2)
plt.pie(notebooks_india['Count'], labels=notebooks_india['Notebook'],
       autopct='%1.1f%%', colors=colors1)
plt.title('India', fontsize=16)
plt.axis('equal')
plt.show()


# ### Observations:
# * **`Kaggle Kernels are widely used in both countries`**
# * **`Google Colab notebooks gained popularity in less time`**
# * **`Indian kagglers used Google Colab and Floydhub more than American kagglers`**

# ## Big Data and Analytics Products

# In[ ]:


# Big Data USA
big_data_usa = generate_df('United States of America', 224, 246, 'Big_Data_Analytics')

# Notebooks USA
big_data_india = generate_df('India', 224, 246, 'Big_Data_Analytics')

# Customizing the plot size
plt.rcParams['figure.figsize'] = [20, 16]

# Plotting USA notebooks
plt.subplot(1, 2, 1)
plt.barh(big_data_usa['Big_Data_Analytics'], big_data_usa['Count'], color=colors1)
plt.title('USA', fontsize=16)
annotate_bar_percentage(big_data_usa['Count'])

# Plotting India notebooks
plt.subplot(1, 2, 2)
plt.barh(big_data_india['Big_Data_Analytics'], big_data_india['Count'], color=colors1)
plt.title('India', fontsize=16)
annotate_bar_percentage(big_data_india['Count'])
plt.show()


# ### Observations:
# * **`AWS Redshift is used more in USA than India`**
# * **`Google Big Query is used widely in India than USA`**
# * **`Databricks is second most used in India and it comes third in USA although it is most used in  USA than India in last 5 years`**
