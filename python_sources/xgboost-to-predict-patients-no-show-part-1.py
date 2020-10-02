#!/usr/bin/env python
# coding: utf-8

# **XGBoost to predict patients no-show**
# 
# The objective of this notebook is to produce an XGBoost model to predict an appointment's no-show, given a patient. 
# The data set shows information of appointments in public hospitals in Vitoria, Espirito Santo, Brazil.
# 
# The following steps will be followed: 
# 
# 1. Univariate Analysis (with Feature Engineering)
# 
# 2. Consistency Check
# 
# 3. Bivariate Analysis (in relation with no-show) 
# 
# 4. Hyperparameter Tunning 
# 
# 5. Model Analysis 

# In[ ]:


import numpy as np 
import pandas as pd 
from scipy import stats as ss
import statsmodels.api as sm
import sklearn.metrics as ssm
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.proportion import proportions_ztest

import os
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv("../input/KaggleV2-May-2016.csv")
print('variables: ' + str(df.columns))


# In[ ]:


df.info()


# There are no missing values for any variable, but some variable types are wrong. (PatientId should be integer, ScheduledDay and AppointmentDay should be datetime objects). 
# 
# For PatientId, we will check if there are comma separated values (which could explain the float type) and see if we can convert the variable type to integer. 
# 

# In[ ]:


df[~ df.PatientId.apply(lambda x: x.is_integer())]


# Studying further, it can be shown that this are typos: there are no patient ids matching the integer part of the above. So, the variable type will be changed. 

# In[ ]:


df['PatientId'] = df['PatientId'].astype('int64')


# 
# Our index should be AppointmentID, has we are trying to determine no-show probability per appointment, given a certain patient. This will allow us to do feature engineering, using patient history. 

# In[ ]:


df.set_index('AppointmentID', inplace = True)


# In[ ]:


df.shape


# **1. Univariate Analysis **
# 
# a) PatientId
# 
# As mentioned before, to build new features related to the apointment, is necessary to check how many patients there are, and how much appointments per patient there is. But first, let's check the variable type. 

# In[ ]:


df['PatientId'].dtype


# As we changed if before, we have that PatientId is integer. 

# In[ ]:


print('Total appointments: ' + format(df.shape[0], ",d"))
print('Distinct patients: ' + format(df['PatientId'].unique().shape[0], ",d"))


# The mean number of appointments per patient is 1.7 (which doesn't say much...) Let's check how many of this patients have more than one appointment associated. As XGBoost learns the best path for missing values (instead of just erasing the observation, as another models do), we could build the number of previous appointments (or the previous no-show rate) leaving empty or zero for first-time patients. 

# In[ ]:


print('Patients with more than one appointment: ' + format((df['PatientId'].value_counts() > 1).sum(), ",d"))


# Nearly 40% of patients have more than one appointment. It's enough to justify the creation of the new variable: number of previous appointments booked and no-show rate based on previous appointments. 

# In[ ]:


df['PreviousApp'] = df.sort_values(by = ['PatientId','ScheduledDay']).groupby(['PatientId']).cumcount()


# In[ ]:


a = df.groupby(pd.cut(df.PreviousApp, bins = [-1, 0,1,2,3,4,5, 85], include_lowest = True))[['PreviousApp']].count()
b = pd.DataFrame(a)
b.set_index(pd.Series(['0', '1', '2', '3', '4', '5', '> 5']))


#  We need to build the rate of previous no-show per patients, for those with more than 1 PreviousApp.

# In[ ]:


df['NoShow'] = (df['No-show'] == 'Yes')*1


# In[ ]:


df['PreviousNoShow'] = (df[df['PreviousApp'] > 0].sort_values(['PatientId', 'ScheduledDay']).groupby(['PatientId'])['NoShow'].cumsum() / df[df['PreviousApp'] > 0]['PreviousApp'])


# In[ ]:


df['PreviousNoShow'].describe()


# More than half of people with previous appointment have gone to all the appointments scheduled. Later on we'll study if this variable is important to predict No-Show, even though it has lots of missings. 
# 
# b) Gender 
# 
# 

# In[ ]:


df['Gender'].value_counts()


# In[ ]:


colors = ['lightcoral', 'lightskyblue']

plt.pie([71840, 38687], explode = (0.1, 0), labels = ['Female', 'Male'], colors = colors, autopct='%1.1f%%') 

plt.title('Patient Gender', fontsize=15)
plt.gcf().set_size_inches(8, 8)
plt.show()


# We can see that almost two thirds of the appointments are done by women, a number much higher than men. There are no missing nor atypical values.
# 
# c) ScheduledDay 
# 
# First we must change the variable type to DateTime. 

# In[ ]:


df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['ScheduledDay2'] = df.apply(lambda x: x.ScheduledDay.strftime("%x"), axis = 1)
scheduled_days = df.groupby(['ScheduledDay2'])[['ScheduledDay']].count()


# In[ ]:


scheduled_days.reset_index(inplace = True)
scheduled_days.columns = ['Date', 'Count']


# In[ ]:


scheduled_days['Date'] = pd.to_datetime(scheduled_days['Date'])


# In[ ]:


print('first scheduled: ' + str(scheduled_days.Date.min()))
print('most recent scheduled: ' + str(scheduled_days.Date.max()))


# In[ ]:


sns.scatterplot(x = 'Date', y = 'Count', data = scheduled_days)
plt.title('Number of Appointments per Scheduled Day')
plt.xlabel('Scheduled Day')
plt.xlim('2015-12', '2016-07')
plt.gcf().set_size_inches(10, 6)
plt.show()


# There appears to be some days with significantly less appointments (maybe during weekends). Also, we can see that most of the appointments where scheduled between April and June 2016. As the appointments are not equally distributed in time (more appointments in some months), we decide to take as variable the appointment's scheduled day of week (Monday, Tuesday, and so on). 

# In[ ]:


df['WeekdayScheduled'] = df.apply(lambda x: x.ScheduledDay.isoweekday(), axis = 1)
df['WeekdayScheduled'].value_counts()


# There appears to be 24 atypical observations: appointments scheduled on Saturdays. We will delete this values. 
# 
# On the other hand, most appointments are scheduled on Tuesdays and Wednesdays. No appointments were scheduled on Sundays. 

# In[ ]:


df = df[df['WeekdayScheduled'] < 6]


# In[ ]:


colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightpink']

plt.pie([df['WeekdayScheduled'].value_counts()[1], df['WeekdayScheduled'].value_counts()[5], 
         df['WeekdayScheduled'].value_counts()[4], df['WeekdayScheduled'].value_counts()[3], df['WeekdayScheduled'].value_counts()[2]], 
        labels = ['Monday','Friday','Thursday','Wednesday' ,'Tuesday'], 
        colors = colors, autopct='%1.1f%%') 

plt.title('Day of Week - Scheduled', fontsize=15)
plt.gcf().set_size_inches(8, 8)
plt.show()


# d) AppointmentDay 
# 
# Same as before, we change to datetime

# In[ ]:


df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
appoint_days = df.groupby(['AppointmentDay'])[['No-show']].count()


# The hour associated for each appointment is 00:00:00, which means that the hour of appointment was not recorded. 

# In[ ]:


appoint_days.reset_index(inplace = True)
appoint_days.columns = ['Date', 'Count']
appoint_days['Date'] = pd.to_datetime(appoint_days['Date'])


# In[ ]:


print('first appointment: ' + str(appoint_days.Date.min()))
print('most recent appointment: ' + str(appoint_days.Date.max()))


# We can see that, even though appointments were scheduled over seven months, the appointments themselves occured between April 29th and June 8th 2016 (41 days).

# In[ ]:


sns.scatterplot(x = 'Date', y = 'Count', data = appoint_days)
plt.title('Number of Appointments per Day')
plt.xlabel('Appointment Day')
plt.xlim('2016-04-28', '2016-06-09')
plt.gcf().set_size_inches(10, 6)
plt.show()


# There's one day ('2016-05-14'') with a much lower number of appointments. Checking the date, May 14th was Saturday which can explain the low number. The rest of days with appointments are during workweek (M-F). 
# 
# Looking at the plot, there seems to be a relation between the number of appointments and the day of the week, so we build the variable WeekdayAppointment (Monday is 1 and Sunday 7): 

# In[ ]:


df['WeekdayAppointment'] = df.apply(lambda x: x.AppointmentDay.isoweekday(), axis = 1)
df['WeekdayAppointment'].value_counts()


# Most of appointments happened between Monday and Wednesday. Again, we see the atypical values which we will delete from the base to avoid overfitting to particular cases. 

# In[ ]:


colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightpink']
df2 = df[df['WeekdayAppointment'] < 6]

plt.pie([df2['WeekdayAppointment'].value_counts()[1], df2['WeekdayAppointment'].value_counts()[5], 
         df2['WeekdayAppointment'].value_counts()[4], df2['WeekdayAppointment'].value_counts()[3], df2['WeekdayAppointment'].value_counts()[2]], 
        labels = ['Monday','Friday','Thursday','Wednesday' ,'Tuesday'], 
        colors = colors, autopct='%1.1f%%') 

plt.title('Day of Week - Appointment', fontsize=15)
plt.gcf().set_size_inches(8, 8)
plt.show()


# e) Age 

# In[ ]:


df2['Age'].describe()


# We have to filter the data, in order to delete negative ages. As for the maximum, it is rare but possible for a patient to be 115 years old so we won't delete these observations. 

# In[ ]:


print('Number of obs with negative age: ' + format(df2[df2['Age'] < 0].shape[0]))


# In[ ]:


df2 = df2[df2['Age'] >=0]
ages = df2.groupby(['Age'])[['PatientId']].count()
ages.reset_index(inplace = True)
ages.columns = ['Age', 'Count']


# In[ ]:


ax = sns.boxplot(x=df2['Age'], orient = 'v')

#plt.xlabel(' ')
plt.ylabel(' ')
plt.title('Boxplot - Age', fontsize = 15)
plt.gcf().set_size_inches(8, 8)
plt.show()


# According to the boxplot, there are a few outliers with ages above 110. Checking the data, the observations aged 115 years old are from two persons which is possible, so the data will be taken in account.  

# In[ ]:


df2[df2['Age'] > 110]


# f) Neighbourhood
# 
# From the metadata included in the dataset, we can see that the neighbourhood refers to where the appointment takes place. 

# In[ ]:


print('Number of different Neighbourhoods: ' + format(df2['Neighbourhood'].value_counts().size))


# Given that the objective is to train an XGBoost model  and due to the lack of information regarding metadata, the neighbourhood won't be used in the model. 
# 
# g) Scholarship
# 
# According to the information given, this variable indicates if the patient is part of a Social Welfare program given by the Brazilian government which gives financial aid to poor families. If families have children, the money is conditionate to children attending school and being vaccinated. 
# 
# This variable will be used as an indicator if the patient is part of a less fortunate family. 

# In[ ]:


df2['Scholarship'].value_counts() 


# In[ ]:


colors = ['lightskyblue','lightcoral']

plt.pie([df2['Scholarship'].value_counts()[1] ,df2['Scholarship'].value_counts()[0] ], explode = (0.1, 0), labels = ['Yes', 'No'], colors = colors, autopct='%1.1f%%') 

plt.title('Scholarship (receives government aid)', fontsize=15)
plt.gcf().set_size_inches(8, 8)
plt.show()


# More than 90% appointments are associated with patients who don't receive government aid. 
# 
# h) Hipertension
# 
# The variable is one if the patient has hipertension diagnosed. 

# In[ ]:


df2['Hipertension'].value_counts()


# In[ ]:


plt.pie([df2['Hipertension'].value_counts()[1] ,df2['Hipertension'].value_counts()[0] ], 
        explode = (0.1, 0), labels = ['Yes', 'No'], colors = colors, autopct='%1.1f%%') 

plt.title('Hypertension Diagnosed', fontsize=15)
plt.gcf().set_size_inches(8, 8)
plt.show()


# More than 80% of patients do not have hipertension. 
# 
# i) Diabetes
# 
# Indicates whether the patient has diabetes. 

# In[ ]:


df2['Diabetes'].value_counts()


# In[ ]:


plt.pie([df2['Diabetes'].value_counts()[1] ,df2['Diabetes'].value_counts()[0] ], 
        explode = (0.1, 0), labels = ['Yes', 'No'], colors = colors, autopct='%1.1f%%') 

plt.title('Diabetes Diagnosed', fontsize=15)
plt.gcf().set_size_inches(8, 8)
plt.show()


# More than 90% of patients do not have diabetes. 
# 
# j) Alcoholism
# 
# The variable is equal to one if the patient has alcoholism. 

# In[ ]:


df2['Alcoholism'].value_counts()


# In[ ]:


plt.pie([df2['Alcoholism'].value_counts()[1] ,df2['Alcoholism'].value_counts()[0] ], 
        explode = (0.1, 0), labels = ['Yes', 'No'], colors = colors, autopct='%1.1f%%') 

plt.title('Alcoholism', fontsize=15)
plt.gcf().set_size_inches(8, 8)
plt.show()


# Only 3% of the appointments are associated with alcoholic patients. 
# 
# k) Handcap
# 
# This variable indicates the number of handicaps a patient is suffering from. 

# In[ ]:


df2['Handcap'].value_counts()


# As most of patients do not have a handicap associated, we re-group the variable into a boolean: HasHandicap, which is cero if the patient has no handicaps and one in any other case. 

# In[ ]:


df2['HasHandicap'] = (df['Handcap'] > 0)*1


# In[ ]:


plt.pie([df2['HasHandicap'].value_counts()[1] ,df2['HasHandicap'].value_counts()[0] ], 
        explode = (0.1, 0), labels = ['Yes', 'No'], colors = colors, autopct='%1.1f%%') 

plt.title('Does the patient have any handicap?', fontsize=15)
plt.gcf().set_size_inches(8, 8)
plt.show()


# Only 2% of patients suffer from some kind of handicap. 
# 
# l) SMS_received
# 
# Indicates whether a SMS was sent to the patient to remind him/her of the appointment. 

# In[ ]:


df2['SMS_received'].value_counts()


# In[ ]:


plt.pie([df2['SMS_received'].value_counts()[1] ,df2['SMS_received'].value_counts()[0] ], 
        explode = (0.1, 0), labels = ['Yes', 'No'], colors = colors, autopct='%1.1f%%') 

plt.title('Was a message sent to the patient to remind of the appointment?', fontsize=15)
plt.gcf().set_size_inches(8, 8)
plt.show()


# Reminder SMS were sent for 32.1% of the appoitments. 
# 
# **Additional Features: **
# 
# m) PreviousDisease
# 
# This variable will summarize all patients with some disease diagnosed: hipertension, diabetes or alcoholism.

# In[ ]:


df2['PreviousDisease'] = df2.apply(lambda x: ((x.Hipertension == 1 )| x.Diabetes == 1 | x.Alcoholism == 1)*1, axis = 1)


# In[ ]:


df2['PreviousDisease'].value_counts()


# In[ ]:


plt.pie([df2['PreviousDisease'].value_counts()[1] ,df2['PreviousDisease'].value_counts()[0] ], 
        explode = (0.1, 0), labels = ['Yes', 'No'], colors = colors, autopct='%1.1f%%') 

plt.title('Does the patient have any previous disease (hipertension, diabetes or alcoholism)?', fontsize=15)
plt.gcf().set_size_inches(8, 8)
plt.show()


# The reason to create this variable is to check if patients with diseases (no matter which) have similar behavior regarding medical appointment no-show. 
# 
# n) DaysBeforeApp
# 
# Indicates the number of days between scheduled day and appointment day. 

# In[ ]:


def get_day(x):
    return x.date()

df2['DaysBeforeApp'] = ((df2.AppointmentDay.apply(get_day) - df2.ScheduledDay.apply(get_day)).astype('timedelta64[D]')).astype(int)


# In[ ]:


df2['DaysBeforeApp'].value_counts()


# Most of appointments where scheduled less than one day in advance. There are negative values which must be studied further:

# In[ ]:


df2[df2['DaysBeforeApp'] < 0]


# From the above, we see that all five scheduled after the appointment are No-shows. This might mean that the hospital made a mistake and scheduled appointments that where not real. For this reason, this observations will be deleted (as there are only 5 of them, it won't have a major impact in the model). 

# In[ ]:


df3 = df2[df2['DaysBeforeApp'] >= 0]


# In[ ]:


days_before = df3.groupby(['DaysBeforeApp'])[['No-show']].count()
days_before.reset_index(inplace = True)
days_before.columns = ['Days Ahead', 'Count']


# In[ ]:


sns.scatterplot(x = 'Days Ahead', y = 'Count', data = days_before)
plt.title('Number of Appointments by Lead Days ')
plt.xlabel('Lead Days')
#plt.xlim('2016-04-28', '2016-06-09')
plt.gcf().set_size_inches(10, 6)
plt.show()


# Most appointments are scheduled less than a day in advance. We will categorize this variable to group similar situations (this will implicate that, for the model, one hot encoding will be necessary in order to use the categorized variable). 

# In[ ]:


def DaysBeforeCat(days):
    if days == 0:
        return '0 days'
    elif days in range(1,3):
        return '1-2 days'
    elif days in range(3,8):
        return '3-7 days'
    elif days in range(8, 32):
        return '8-31 days'
    else:
        return '> 31 days'
    
df3['DaysBeforeCat'] = df3.DaysBeforeApp.apply(DaysBeforeCat)


# In[ ]:


df3['DaysBeforeCat'].value_counts()


# In[ ]:


colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightpink']

plt.pie([df3['DaysBeforeCat'].value_counts()[0], df3['DaysBeforeCat'].value_counts()[3], 
         df3['DaysBeforeCat'].value_counts()[2], df3['DaysBeforeCat'].value_counts()[1], df3['DaysBeforeCat'].value_counts()[4]], 
        labels = ['0 days','1-2 days' ,'3-7 days','8-31 days','> 31 days'], 
        explode = (0.1, 0, 0, 0, 0),
        colors = colors, autopct='%1.1f%%') 

plt.title('Lead Days', fontsize=15)
plt.gcf().set_size_inches(8, 8)
plt.show()


# o) No-show 
# 
# For XGBoost, it's very important to know the proportion of yes/no in the sample, as the parameter scale_pos_weight allows to work with unbalanced data without sampling or discarding observations. 

# In[ ]:


df3['No-show'].value_counts()[1]


# In[ ]:


ns = df3['No-show'].value_counts()[1]
show = df3['No-show'].value_counts()[0]
rate = (show + 0.0) / ns
print('For every no-show, there are {:1.2f} shows'.format(rate))


# We have approximately 4 shows for every no-show. 
# 
# **2. Consistency Check**
# 
# We will check a couple of "common sense" rules regarding the data we have:
# * Medical conditions (such as hypertension, diabetes and alcoholism) should have a unique value per patient
# * Schedule day should take place before appointment day
# * For every patient, the age should not differ in more than one and, if this is the case, the appointment with higher age should take place after the appointment with lower age
# * Gender, handicap and scholarship should have a unique value per patient

# In[ ]:


def unique_condition(df, var, cols):
    if df.groupby(cols).ngroups == df[var].unique().size:
        return 'Sizes match: unique value per ' + var
    else: 
        return 'Mismatch: more than one value per ' + var

unique_condition(df3, 'PatientId', ['PatientId','Hipertension', 'Diabetes', 
                                    'Alcoholism', 'Gender', 'Handcap', 'Scholarship'])


# From the above, we have checked that there's one value of Hipertension, Diabetes, Alcoholism, Gender, Handcap and Scholarship per patient, which were our first and last conditions.
# 
# For scheduled day before appointment day, we check that the amount of days before is at least zero (less than zero would mean that the scheduling was done after the appointment - as we do not have the the hour of appointment, we assume all appointments were scheduled at 00:00:00 hrs). 

# In[ ]:


print('Reservations scheduled after appointment time: ' + str(df3[df3['DaysBeforeApp'] < 0].size))


# Last, but not least, we'll check that ages do not differ in more than a year per patient: 

# In[ ]:


inconsist = []
for num in df3['PatientId'].unique():
    ages = df3[df3['PatientId'] == num]['Age'].unique()
    if ages.size == 1:
        break
    if ages.size > 2:
        inconsist.append(num)
        print('Patient ' + str(num)+ 'has age inconsistency')
    else:
        if abs(ages[0]-ages[1]) > 1:
            inconsist.append(num)
            print('Patient ' + str(num)+ 'has age inconsistency')
            
if len(inconsist) == 0:
    print('There is no inconsistency in ages')


# We found no inconsistencies in our dataset. 
# 
# **3. Bivariate Analysis**
# 
# So far we have managed to collect and analyze the following features:  Gender, ScheduledDay, AppointmentDay, Age, Neighbourhood, Scholarship, Hipertension, Diabetes, Alcoholism, Handcap, SMS_received, PreviousApp, PreviousNoShow, WeekdayScheduled, WeekdayAppointment, HasHandicap, PreviousDisease, DaysBeforeApp, DaysBeforeCat. 
# 
# Based on previous analysis, we've discarded ScheduledDay, AppointmentDay (we are going to use the weekday of both), Neighbourhood (too many categories), Handcap and DaysBeforeApp (as we categorized them).  Now we will study each feature and its relation with No-Show: 
# 
# a) Gender

# In[ ]:


sns.set()

def cat_var(df3, var):
    
    print(df3.groupby([var])['NoShow'].mean())
    
    ns_rate = [df3.groupby([var])['NoShow'].mean()[i] for i in df3[var].unique()]
    s_rate = [1-df3.groupby([var])['NoShow'].mean()[i] for i in df3[var].unique()]
    barWidth = 0.5

    plt.bar(df3[var].unique(), ns_rate, color='lightcoral', edgecolor='white', width=barWidth, label = 'No-Show')
    plt.bar(df3[var].unique(), s_rate, bottom=ns_rate, color='mediumseagreen', edgecolor='white', width=barWidth, label = 'Show')
    plt.axhline(y=df3['NoShow'].mean(), color='black', linewidth= 0.8, linestyle='--', label = 'Overall mean')
    plt.xticks(df3[var].unique())
    plt.xlabel(var)
    plt.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    plt.title('No-Show Rate by '+ var, fontsize=15)
    plt.gcf().set_size_inches(6, 6)
    plt.show() 
    
    counts = np.array(df3.groupby([var])['NoShow'].sum())
    nobs = np.array(df3.groupby([var])['NoShow'].count())

    table = df3.groupby(['NoShow', var]).size().unstack(var)
    pvalue = ss.chi2_contingency(table.fillna(0))[1]
    
    print('Means test p-value: {:1.3f}'.format(pvalue))
    if pvalue < 0.05:
        print('Reject null hypothesis: no-show rate is different for at least one group')
    else:
        print('Cannot reject no-show rates are same for all groups')


# In[ ]:


cat_var(df3, 'Gender')


# Based on the analysis, we cannot say that no-show rates are different for men and women, which could indicate that the variable Gender is not a relevant one when predicting no-show. We shall include the variable as well because, interacting with others, Gender could gain relevance and became a good discriminator. 
# 
# b) Age

# In[ ]:


ax = sns.boxplot(x="NoShow", y="Age",data= df3, palette = 'RdBu')
ax.set_xticklabels(['Show', 'No Show'])
plt.xlabel(' ')
plt.ylabel('Age (in years)')
plt.title('Age Boxplot by No-Show', fontsize = 15)
plt.gcf().set_size_inches(12, 8)
plt.show()


# From the boxplot, we can see that people who don't show up to appointments tend to be younger than those who attend (based on quartiles). 

# In[ ]:


print('Correlation with No-Show: %.3f' % ss.pointbiserialr(df3['NoShow'], df3['Age'])[0])


# Correlation with no-show is very close to zero,which could indicate that the variable is not of much interest when predicting no-show. 
# 
# c) Scholarship

# In[ ]:


cat_var(df3, 'Scholarship')


# From the above, we can assure that patients who are part of the Social Welfare program have significantly higher no-show rates than people without this government support. This could mean that the variable is very important for predicting no-show.
# 
# d) Hypertension

# In[ ]:


cat_var(df3, 'Hipertension')


# As before, we can reject that people with hypertension present same no-show rates as patients without hypertension diagnosed. This difference is significant, which indicates that the variable is of interest for predicting no-show. People with hypertension tend to have lower no-show rates (maybe because they are in some kind of treatment...)
# 
# e) Diabetes

# In[ ]:


cat_var(df3, 'Diabetes')


# From the test, we can reject the hypothesis that no-show rates are the same for patients with diabetes and those without. This implies that the variable Diabetes is of interest for our model. 
# 
# f) Alcoholism

# In[ ]:


cat_var(df3, 'Alcoholism')


# As mentioned above, we cannot reject that the mean rates are equal, despite the group (Alcoholic vs Non-alcoholic). Therefore, the variable Alcoholism may be not important to discriminate between no shows/shows. 
# 
# g) SMS_received

# In[ ]:


cat_var(df3, 'SMS_received')


# If we see the no-show rates per group, there appears to be something strange in the data, as one could suppose that if a patient receives a SMS to remind him/her the appointment, this patient is less likely to miss the appointment. 
# 
# We will analyze how SMS are sent in relation with anticipation of scheduling. 

# In[ ]:


aux


# In[ ]:


aux = df3.groupby(['DaysBeforeApp'])[['SMS_received']].agg(['count','sum'])
aux.columns = ['count', 'SMS_received']
aux[:20]['SMS_received'].plot()
plt.gcf().set_size_inches(10, 6)
plt.xlabel('Anticipation Days')
plt.ylabel('Count (n)')
plt.title('Frequency of Days Before Appointment (anticipation)')
plt.xticks(range(0, 20))
plt.show()


# From the plot above, we can see that there are no SMS sent for appointments scheduled less than two days before. Even so, the number of messages sent grows at 4 days before. So we will filter the data by appointments scheduled with more than 4 days of anticipation and re-analyze the variable. 

# In[ ]:


fourdaysormore = df3[df3['DaysBeforeApp'] > 3]
cat_var(fourdaysormore, 'SMS_received')


# Now the analysis makes more sense: patients who didn't receive a reminder SMS have higher no-show rates than those who did receive a SMS. This differences are statistically significant, so the variable together with days before appointment is of interest for our model. 
# 
# h) PreviousApp
# 
# We will study no-show rates grouped by number of previous appointments, for categories with at least 30 appointments: 

# In[ ]:


prevapp = df3.groupby(['PreviousApp'])[['NoShow']].agg(['count', 'mean'])
prevapp.columns = ['count', 'NoShow_rate']


# In[ ]:


prevapp.reset_index(inplace = True)


# In[ ]:


import warnings
warnings.filterwarnings("ignore")
prevapp = prevapp[(prevapp['count'] > 30) & (prevapp['PreviousApp'] > 0)]


# In[ ]:


fig = plt.figure()

count = fig.add_subplot(111)
rate = count.twinx()

count.set_ylabel('N')
rate.set_ylabel('No-Show rate')

line1 = count.bar(prevapp['PreviousApp'], prevapp['count'])
line2 = rate.plot(prevapp['PreviousApp'], prevapp['NoShow_rate'], color = 'red', label = 'No-show Rate')
count.legend([line1, line2], ['Count', 'No-show Rate'])
plt.gcf().set_size_inches(12, 8)
count.set_xlabel('Number of Previous Appointments')
plt.title('Number of Previous Appointments: total and no-show rates (n > 30)')
plt.show()


# We decided to eliminate from the plot appointments without previous appointments, as they are a much higher number than other categories and makes the plot more confusing. From the above, and considering we are only observing groups with more than 30 observations, we can say that no show rate are higher for patients with 10-12 previous appointments and then the no show rates descend drastically (patients with lots of previous appointments are likely to be in a treatment as the data is condensed in 41 days). 

# In[ ]:


print('Correlation with No-Show (all appointments): %.3f' % ss.pointbiserialr(df3['NoShow'], df3['PreviousApp'])[0])
print('Correlation with No-Show (1 or more previous app): %.3f' % ss.pointbiserialr(df3[df3['PreviousApp'] > 0]['NoShow'], df3[df3['PreviousApp'] > 0]['PreviousApp'])[0])


# The point biserial correlation is -0.035 and if we calculate the correlation without first-time-appointments, the value is closer to -1, indicating a stronger negative correlation. 
# 
# i) PreviousNoShow
# 
# We will study no-show rates, grouping appointments by PreviousNoShow deciles. 

# In[ ]:


prop_ns = df3.groupby(pd.cut(df3['PreviousNoShow'], np.arange(0, 1.05, 0.05), include_lowest = True))[['NoShow']].mean()
prop_ns = prop_ns.reset_index()
prop_ns['middle'] = np.arange(0.025, 1.025, 0.05)
prop_ns.iloc[0,2] = 0
prop_ns.iloc[19,2] = 1


# In[ ]:


no_na = df3.dropna(subset = ['PreviousNoShow'])


# In[ ]:


prop_ns = prop_ns.drop([18], axis = 0)
plt.plot(prop_ns['middle'], prop_ns['NoShow'], color = '#16a4e3')

plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel('Previous Appointments No-Show Rate', labelpad=10)
plt.ylabel('No-Show (%)')
plt.grid(True)
plt.gcf().set_size_inches(12, 8)
plt.title('No Show Rate by Proportion of Previous Appointments No-Show', fontsize = 15)

plt.show()

print('Correlation with No-Show: %.3f' % ss.pointbiserialr(no_na['NoShow'], no_na['PreviousNoShow'])[0])


# (The interval [0.9, 0.95] was deleted as there were no appointments with previous no show rate between this values). As expected, previous patient behavior and no-show rate have an almost perfect linear relation. This supports the hypothesis that people tend to mantain certain behaviors in time. Moreover, the correlation between the variables is very close to 1, indicating a strong and positive correlation between previous no show and appointment noshow. For this, pevious behavior is a variable of interest for our model. 
# 
# j) WeekdayScheduled
# 

# In[ ]:


cat_var(df3, 'WeekdayScheduled')


# From the above, the variable apparently isn't of interest as no-show rates are not statistically different between groups (per Scheduled Day). 
# 
# k) WeekdayAppointment

# In[ ]:


cat_var(df3, 'WeekdayAppointment')


# Unlike weekday scheduled, for weekday appointment the differences between group are significant. Appointments on Friday have higher no-show rates than rest of the week, while the lowest no-show rate per group is for appointments on Thursday. 
# 
# l) HasHandicap

# In[ ]:


cat_var(df3, 'HasHandicap')


# Patients who have at leat one handicap have lower no-show rates than those who doesn't have, and this difference is statistically significant. The variable is of interest for our model. 
# 
# m) PreviousDisease

# In[ ]:


cat_var(df3, 'PreviousDisease')


# As expected, no-show rates per group are different. This is consistent with previous analysis, as we saw before that Diabetes and Hypertension are both importante features to predict no-show (variables can separate groups with different no-show mean rates). 
# 
# n) DaysBeforeCat 

# In[ ]:


cat_var(df3, 'DaysBeforeCat')


# From the plot, it is clear that no-show rates are very different in each group: moreover, rates seem to be increasing as the number of days of anticipation is higher. For appointments scheduled in the same day (0 days of anticipation), no-show rates are dramatically lower and very close to zero (only 4.6%). This is clearly evidence enough to consider the variable as highly interesting and important to predict no-show. 
# 
# 
# **SHALL CONTINUE IN PART 2 WITH HYPERPARAMETER TUNNING AND MODEL ANALYSIS**
