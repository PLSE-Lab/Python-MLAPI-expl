#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # **EXECUTIVE SUMMARY**
# 
# 

# The world is undergoing one of the most pathetic times in history due to the recent COVID virus Pandemic. 
# In this analysis I have tried to do analysis to predict the risks and vulnerabiity of Heathcare workers who are involved in treating C19 confirmed patients.
# 
# A new dataset has been created from https://www.medscape.com/viewarticle/927976#vp_11 where the name,age and country along with department of work of the Healthcare workers are documented and this list is sadly but slowly updating.
# 
# From there I have analysed the following.
# 
# 1.Relationship between Death and Age of Healthcare workers worldwide.
# 
# 2.Relationship between death of Healthcare workers and total confirmed populations of different countries.
# 
# 3.Relationship between death of Healthcare workers and their department of work worldwide.
# 
# It is to be mentioned here that while creating the excel sheet I have grouped different subcategories into major groups.For example Nursing assistant,Nurse etc are all included into one general subcategory 'Nurse' and likewise General practitioner, Physicians etc are categorized as 'General Physician'. A excel sheet which contains these relationships is also uploaded here. The Healthcare Worker category for example encompasses hospital staffs, hospital porters, receptionists etc.
# 
# 
# 4.Death rate in countries like Italy ,UK,USA and Iran is more and hence for these countries individual graphs depicting the number of death per category for each of these countries are plotted and analysed.
# 
# 5.Finally death with respect to countries is plotted and analysed.
# 
# 

# 

# In[ ]:


# Importing necessary library
import pandas as pd
import numpy as np
import glob  
import matplotlib.pyplot as plt
import plotly as py
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
pyo.init_notebook_mode()
import seaborn as sns
import sklearn


#Reading the dataset containing information on death worldwide of Healthcare personnels
#made from Memorandum

dataset = pd.read_csv('../input/world-death-in-healthcare/Worldwide_Deaths_In_Healthcare.csv')


#PREPROCCESSING TO HANDLE MISSING AGES WITH MEAN.
#Since in many places age was missing, so I decided to use most_frequent ages for those
#missing data, as that would make lowest bias possible
#however median is also another available option

Pre_Proc = dataset.iloc[:,0:5].values#[:,:-1]means taking all rows and all coloumns except the last one


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer = imputer.fit(Pre_Proc[:,1:2])
Pre_Proc[:,1:2] = imputer.transform(Pre_Proc[:,1:2])


#Transforming pre_proc to dataframe for further proccessing
#and operation simplicity

Pre_Proc = pd.DataFrame(Pre_Proc)
Pre_Proc.columns = ["Name","Age","Department","Country","Death_Count_of_total_healthcare_individuals_in_that_country"]

#now reading data which speaks of total number of confirmed cases per country
#this includes total population not only healthcare workers

covid = pd.read_csv('../input/worldwide-confirmed-data/confirmeddata.csv').rename(columns={'country_region':'Country'})
Pre_Proc = Pre_Proc.merge(covid[['Country', 'total_confirmed_pop_country_wise']], how='left', on='Country')

#printing number of unique countries where where healthcare workers death due to C19
#have occurred along with number of unique departments throughout the world
#from where healthcare workers death is recorded
unique_countries = Pre_Proc.Country.nunique()
print('total number of unique countries',unique_countries)
unique_depts = Pre_Proc.Department.nunique()
print('total number of unique departments',unique_depts)


# The total number of countries where death of physicians due to COVID 19 and total number of departments affected due to COVID 19 is given above which clearly shows that Healthcare Workers are affected worldwide and irrespective of departments in which they work

# In[ ]:


#Plotting the age variations of death of HC workers
sns.kdeplot(Pre_Proc.Age)


# It can be seen that people of age above 40 is at higher risk although there are cases where health personnels of age below 40 and above 20 had also fallen to this deathly desease. Due to lack of available data, this couldn't be introspected further but a possible factor could had been pre-medical conditions. 

# In[ ]:


#  Relation between number of deceased workers by total number of confirmed population worldwide
fig,axes = plt.subplots(1,1,figsize=(20,10))
plot1 = Pre_Proc['total_confirmed_pop_country_wise'].value_counts().plot(kind='bar')
plt.title('Number Of Health Workers who died of COVID-19 by total_confirmed_population_country_wise', fontsize=24)
plt.ylabel('number of Health Care workers', fontsize=20)
plt.xticks(fontsize=20)
plt.tight_layout()


# This clearly indicates that as the number of confirmed cases for a country increases, the death of Healthcare personnels increase. This is very obvious owing to the fact, that  per doctor/nurse and healthcare workers number of patients are more in these pandemic regions which has lead to higher deaths. Possible reasons can also be limitation of protective equipments, limitation of hospital beds etc.

# In[ ]:


#  Relation between number of deceased workers by Department worldwide
fig,axes = plt.subplots(1,1,figsize=(20,10))
plot1 = Pre_Proc['Department'].value_counts().plot(kind='bar')
plt.title('Number Of Health Workers who died of COVID-19 by Department worldwide', fontsize=24)
plt.ylabel('number of Health Care workers', fontsize=20)
plt.xticks(fontsize=20)
plt.tight_layout()


# So as the data suggests, the most number of deaths are recorded in the group of General Physicians and Nurses. Although one can assume that most vulerable groups are Emergency medicine or Critical Care and Respiraory problems but it seems that as the family physicians or general practioners are consulted primarily when people faced distress physically so they have contracted it most knowingly or unknowing followed by nurses responsible for primary well being of patients as well. The availability of the protective equipments can also be a factor alongside most contact.

# In[ ]:


#  Relation between number of deceased workers by Department in Italy 

temp1 = Pre_Proc[Pre_Proc.Country == 'Italy']
fig,axes = plt.subplots(1,1,figsize=(20,10))
plot1 = temp1['Department'].value_counts().plot(kind='bar')
plt.title('Number Of Health Workers who died of COVID-19 by Department Italy', fontsize=24)
plt.ylabel('number of Health Care workers', fontsize=20)
plt.xticks(fontsize=20)
plt.tight_layout()

#  Relation between number of deceased workers by Department in US 

temp2 = Pre_Proc[Pre_Proc.Country == 'US']
fig,axes = plt.subplots(1,1,figsize=(20,10))
plot1 = temp2['Department'].value_counts().plot(kind='bar')
plt.title('Number Of Health Workers who died of COVID-19 by Department US', fontsize=24)
plt.ylabel('number of Health Care workers', fontsize=20)
plt.xticks(fontsize=20)
plt.tight_layout()

#  Relation between number of deceased workers by Department in United Kingdom 

temp3 = Pre_Proc[Pre_Proc.Country == 'United Kingdom']
fig,axes = plt.subplots(1,1,figsize=(20,10))
plot1 = temp3['Department'].value_counts().plot(kind='bar')
plt.title('Number Of Health Workers who died of COVID-19 by Department United Kingdom', fontsize=24)
plt.ylabel('number of Health Care  workers', fontsize=20)
plt.xticks(fontsize=20)
plt.tight_layout()

#  Relation between number of deceased workers by Department in Iran 

temp4 = Pre_Proc[Pre_Proc.Country == 'Iran']
fig,axes = plt.subplots(1,1,figsize=(20,10))
plot1 = temp4['Department'].value_counts().plot(kind='bar')
plt.title('Number Of Health Workers who died of COVID-19 by Department Iran', fontsize=24)
plt.ylabel('number of Health Care workers', fontsize=20)
plt.xticks(fontsize=20)
plt.tight_layout()


# I have further analysed this to understand the country wise trend/variation. It seems that the most vulnerable group in Italy is General Physicians while in US the Nurses have fallen most. In UK ,the Nurses and Healthcare workers like hospital staffs , administrative staffs have highest death rate while in Iran it is of General Physicans and Nurses.
# 
# A number of factors can be analysed based on this report.
# 1.The age group of these groups
# 2.The availability of protective equipments to these groups while treating/coming in contact to patients.
# 3.Patient to Healthcare workers ratio

# In[ ]:


#  Relation between number of deceased workers by Country worldwide

fig,axes = plt.subplots(1,1,figsize=(20,10))
plot1 = Pre_Proc['Country'].value_counts().plot(kind='bar')
plt.title('Number Of Health Care Workers who died of COVID-19 by Country', fontsize=24)
plt.ylabel('number of Health Care workers', fontsize=20)
plt.xticks(fontsize=20)
plt.tight_layout()


# The data analyses in which country maximum  death is recorded. The Pandemic countries dominates the chart however in lesser Pandemic countries and non Pandemic countries too, death has been recorded.

# In[ ]:


#  Relation between number of deceased workers by Age in Italy 

temp1 = Pre_Proc[Pre_Proc.Country == 'Italy']
fig,axes = plt.subplots(1,1,figsize=(20,10))
plot1 = temp1['Age'].value_counts().plot(kind='bar')
plt.title('Number Of Health Workers who died of COVID-19 by Age Italy', fontsize=24)
plt.ylabel('number of Health Care workers', fontsize=20)
plt.xticks(fontsize=20)
plt.tight_layout()

#  Relation between number of deceased workers by Age in US 

temp2 = Pre_Proc[Pre_Proc.Country == 'US']
fig,axes = plt.subplots(1,1,figsize=(20,10))
plot1 = temp2['Age'].value_counts().plot(kind='bar')
plt.title('Number Of Health Workers who died of COVID-19 by Age US', fontsize=24)
plt.ylabel('number of Health Care workers', fontsize=20)
plt.xticks(fontsize=20)
plt.tight_layout()

#  Relation between number of deceased workers by Age in United Kingdom 

temp3 = Pre_Proc[Pre_Proc.Country == 'United Kingdom']
fig,axes = plt.subplots(1,1,figsize=(20,10))
plot1 = temp3['Age'].value_counts().plot(kind='bar')
plt.title('Number Of Health Workers who died of COVID-19 by Age United Kingdom', fontsize=24)
plt.ylabel('number of Health Care  workers', fontsize=20)
plt.xticks(fontsize=20)
plt.tight_layout()

#  Relation between number of deceased workers by Age in Iran 

temp4 = Pre_Proc[Pre_Proc.Country == 'Iran']
fig,axes = plt.subplots(1,1,figsize=(20,10))
plot1 = temp4['Age'].value_counts().plot(kind='bar')
plt.title('Number Of Health Workers who died of COVID-19 by Age Iran', fontsize=24)
plt.ylabel('number of Health Care workers', fontsize=20)
plt.xticks(fontsize=20)
plt.tight_layout()


# From the age graph it ca be understood that in these countries where death rate is very high, the age of the personnels are in the most risky zone which might have been the reason for more deaths. Although since missing vaue has been taken care of by using most frequent value hence it is not possible to rely on this data fully but it gives us a general view of the most vulnerable age groups.

# This is a small overall analysis which I have tried to pen down .I pay all my respect for the fallen heroes and this work is dedicated to their sacrifice. 
# Pease visit covid_death_healthcare.csv dataset uploaded by me to understand the departments(groups) which has been formed based on subcategories.
# 
# **So in short it can be concluded that,
# 1. Health workers of age above 40 and between age 40- 60 are at higher risks
# 2.General Physicians and Nurses are at higher risk of contraction but this depends on the group/individuals coming in more contact to patients in different regions.
# 3.Healthcare workers at countries with more confirmed cases are in high risk and death rates are more in US, Italy, United Kingdom ,Iran etc. 
# This may owe to the fact that patient and support worker ratio is very high as due to more patients and less resources. Also other factors might be  availability of treatments, protective equipments etc in places of less confirmed cases but comparatively more death rate of health workers.
# 
# Pre health conditions of these individuals might be another factor too.******

# Printing the list showing dept vs country death worldwide

# In[ ]:



list_view = Pre_Proc[['Department', 'Death_Count_of_total_healthcare_individuals_in_that_country','Country']].copy()
list_view = list_view.groupby(['Department','Country']).count()


# 

# In[ ]:


pd.set_option("display.max_rows", None, "display.max_columns", None)
print(list_view)

