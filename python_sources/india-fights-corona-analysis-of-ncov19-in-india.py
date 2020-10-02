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


# In[ ]:


age_data=pd.read_csv('/kaggle/input/covid19-in-india/AgeGroupDetails.csv')


# In[ ]:


print(age_data)


# In[ ]:


age_X=age_data.iloc[0:9,1]
age_y=age_data.iloc[0:9,2]


# In[ ]:


age_group_list=[]
age_cases_list=[]


# In[ ]:


for i in range(0,9):
    age_group_list.append(age_X[i])
    age_cases_list.append(age_y[i])


# In[ ]:


print(age_group_list)
print(age_cases_list)


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


fig=plt.figure(figsize=(8,8))
plt.pie(age_cases_list,labels=age_group_list,autopct = '%1.1f%%')
plt.legend()
plt.title('India-Covid19 Cases vs Age')


# This analysis goes against the saying that people above the age of 60 are more vulnerable to the virus. The analysis here shows that prople in the age group of 20-29 were most infected by virus and their number is approximately twice of the number of people infected in the age range 60-69. The bulk of the infected population comes from the age group of 20-39 which is like around 47% as compared to people in the age group of 60-79 which is around 17%.

# In[ ]:


data_gender=pd.read_csv('/kaggle/input/covid19-in-india/IndividualDetails.csv')


# In[ ]:


data_gender


# In[ ]:


gender=data_gender.iloc[:,4]


# In[ ]:


count_male=0
count_female=0


# In[ ]:


for i in range(0,27889):
    if(gender[i]=='M'):
        count_male=count_male+1
    elif(gender[i]=='F'):
        count_female=count_female+1


# In[ ]:


print(count_male)
print(count_female)


# In[ ]:


gender_list=[count_male,count_female]


# In[ ]:


print(gender_list)


# In[ ]:


gender_label=['Male','Female']


# In[ ]:


plt.figure(figsize=(5,5))
plt.pie(gender_list,labels=gender_label,autopct = '%1.1f%%')
plt.legend()
plt.title('India-Covid19 cases vs Gender')


# The analysis shows that the number of males infected by novel-Coronavirus is approximately twice the number of females infected by Covid-19. However, there were a lot of missing data and therefore only those with non NaN values has been taken into consideration. 

# In[ ]:


data_cases=pd.read_csv('https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv')


# In[ ]:


data_cases


# In[ ]:


country_name=data_cases.iloc[:,1]
country_confirmed=data_cases.iloc[:,2]
country_recovered=data_cases.iloc[:,3]
country_deaths=data_cases.iloc[:,4]


# In[ ]:


india_confirmed=[]
india_recovered=[]
india_deaths=[]


# In[ ]:


for i in range(0,len(data_cases)):
    if(country_name[i]=='India'):
        india_confirmed.append(country_confirmed[i])
        india_recovered.append(country_recovered[i])
        india_deaths.append(country_deaths[i])


# In[ ]:


print(india_confirmed)
print(india_recovered)
print(india_deaths)


# In[ ]:


days=[]


# In[ ]:


day=1


# In[ ]:


for i in range(0,len(data_cases)):
    if(country_name[i]=='India'):
        days.append(day)
        day=day+1
        


# In[ ]:


len(days)


# In[ ]:


plt.figure(figsize=(15,10))
plt.xticks(rotation=90,fontsize=12)

g_1=plt.plot(days,india_confirmed)
g_2=plt.plot(days,india_recovered,color='green')
g_3=plt.plot(days,india_deaths,color='red')
plt.title('Covid-19 India -- Confirmed vs Recovered vs Death')
plt.xlabel('Days',fontsize=18)
plt.ylabel('Total Cases',fontsize=18)
plt.legend()


# The graph of India has been rising continously and has not seen any decline as now. However, the good thing is the recovered graph is also rising exponentially and this is our source of hope. But, the death curve also has seen a slight rise in past few days.

# Let us see how does the curve of India compare with other countries viz China, Italy, Germany, UK and US

# In[ ]:


confirmed_china=[]
confirmed_italy=[]
confirmed_germany=[]
confirmed_uk=[]
confirmed_us=[]


# In[ ]:


for i in range(0,len(data_cases)):
    if(country_name[i]=='China'):
        confirmed_china.append(country_confirmed[i])
    elif(country_name[i]=='Italy'):
        confirmed_italy.append(country_confirmed[i])
    elif(country_name[i]=='Germany'):
        confirmed_germany.append(country_confirmed[i])
    elif(country_name[i]=='United Kingdom'):
        confirmed_uk.append(country_confirmed[i])
    elif(country_name[i]=='US'):
        confirmed_us.append(country_confirmed[i])


# In[ ]:


len(confirmed_us)


# In[ ]:


fig=plt.figure(figsize=(15,10))
plt.plot(days,india_confirmed,label='India')
plt.plot(days,confirmed_china,label='China')
plt.plot(days,confirmed_uk,label='United Kingdom')
plt.plot(days,confirmed_italy,label='Italy')
plt.plot(days,confirmed_germany,label='Germany')
plt.plot(days,confirmed_us,label='USA')
plt.yticks(rotation=90,fontsize=12)
plt.xlabel('Days',fontsize=12)
plt.ylabel('Cases')
plt.legend()
plt.show()


# As we can see, India and US are pretty badly placed with their curve showing an upward trend while China has managed to flatten the curve. Germany and Italy has seen a dip in cases which is a good sign and UK is probably on the way.

# Now, let us move on to the states of India. We will first analyze the testing details in each state.

# In[ ]:


data_test_state=pd.read_csv('/kaggle/input/covid19-in-india/StatewiseTestingDetails.csv')


# In[ ]:


data_test_state


# In[ ]:


state_testing_date=data_test_state.iloc[:,0]
state_testing_name=data_test_state.iloc[:,1]
state_testing_samples=data_test_state.iloc[:,2]


# In[ ]:


testing_name_state=[]
testing_samples=[]


# In[ ]:


for i in range(0,len(data_test_state)):
    if(state_testing_date[i]=='2020-05-20'):
        testing_samples.append(state_testing_samples[i])
        testing_name_state.append(state_testing_name[i])
       


# In[ ]:


figure=plt.figure(figsize=(15,10))
plt.bar(testing_name_state,testing_samples,color='orange')
plt.xticks(rotation=90,fontsize=12)
plt.show()


# As we can see, the highest number of tests are being conducted in Tamil Nadu followed by Maharashtra, Andhra Pradesh, Rajasthan and Uttar Pradesh. This also explains the fact why these states top the chart in number of confirmed cases.

# In[ ]:


data_states_cases=pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv')


# In[ ]:


data_states_cases


# In[ ]:


data_states_date=data_states_cases.iloc[:,1]
data_states_name=data_states_cases.iloc[:,3]
data_states_cc=data_states_cases.iloc[:,8]


# In[ ]:


delhi_confirmed=[]
maharashtra_confirmed=[]
tamilnadu_confirmed=[]
uttarpradesh_confirmed=[]
westbengal_confirmed=[]
madhyapradesh_confirmed=[]
chhattisgarh_confirmed=[]
andrapradesh_confirmed=[]


# In[ ]:


delhi_date=[]
maharashtra_date=[]
tamilnadu_date=[]
uttarpradesh_date=[]
westbengal_date=[]
madhyapradesh_date=[]
chhattisgarh_date=[]
andrapradesh_date=[]


# In[ ]:


for i in range(0,len(data_states_cases)):
    if(data_states_name[i]=='Delhi'):
        delhi_confirmed.append(data_states_cc[i])
        delhi_date.append(data_states_date[i])
    


# In[ ]:


figure=plt.figure(figsize=(30,10))
plt.plot(delhi_date,delhi_confirmed)
plt.xticks(rotation=90,fontsize=12)
plt.xlabel('Date',fontsize=20)
plt.ylabel('Cases',fontsize=20)
plt.title('Delhi',fontsize=20)
plt.show()


# In[ ]:


for i in range(0,len(data_states_cases)):
    if(data_states_name[i]=='Maharashtra'):
        maharashtra_confirmed.append(data_states_cc[i])
        maharashtra_date.append(data_states_date[i])


# In[ ]:


figure=plt.figure(figsize=(30,10))
plt.plot(maharashtra_date,maharashtra_confirmed)
plt.xticks(rotation=90,fontsize=12)
plt.xlabel('Date',fontsize=20)
plt.ylabel('Cases',fontsize=20)
plt.title('Maharashtra',fontsize=20)
plt.show()


# In[ ]:


for i in range(0,len(data_states_cases)):
    if(data_states_name[i]=='Uttar Pradesh'):
        uttarpradesh_confirmed.append(data_states_cc[i])
        uttarpradesh_date.append(data_states_date[i])


# In[ ]:


figure=plt.figure(figsize=(30,10))
plt.plot(uttarpradesh_date,uttarpradesh_confirmed)
plt.xticks(rotation=90,fontsize=12)
plt.xlabel('Date',fontsize=20)
plt.ylabel('Cases',fontsize=20)
plt.title('Uttar Pradesh',fontsize=20)
plt.show()


# In[ ]:


for i in range(0,len(data_states_cases)):
    if(data_states_name[i]=='Madhya Pradesh'):
        madhyapradesh_confirmed.append(data_states_cc[i])
        madhyapradesh_date.append(data_states_date[i])


# In[ ]:


figure=plt.figure(figsize=(30,10))
plt.plot(madhyapradesh_date,madhyapradesh_confirmed)
plt.xticks(rotation=90,fontsize=12)
plt.xlabel('Date',fontsize=20)
plt.ylabel('Cases',fontsize=20)
plt.title('Madhya Pradesh',fontsize=20)
plt.show()


# In[ ]:


for i in range(0,len(data_states_cases)):
    if(data_states_name[i]=='Chhattisgarh'):
        chhattisgarh_confirmed.append(data_states_cc[i])
        chhattisgarh_date.append(data_states_date[i])


# In[ ]:


figure=plt.figure(figsize=(30,10))
plt.plot(chhattisgarh_date,chhattisgarh_confirmed)
plt.xticks(rotation=90,fontsize=12)
plt.xlabel('Date',fontsize=20)
plt.ylabel('Cases',fontsize=20)
plt.title('Chhattisgarh',fontsize=20)
plt.show()


# In[ ]:


for i in range(0,len(data_states_cases)):
    if(data_states_name[i]=='Tamil Nadu'):
        tamilnadu_confirmed.append(data_states_cc[i])
        tamilnadu_date.append(data_states_date[i])


# In[ ]:


figure=plt.figure(figsize=(30,10))
plt.plot(tamilnadu_date,tamilnadu_confirmed)
plt.xticks(rotation=90,fontsize=12)
plt.xlabel('Date',fontsize=20)
plt.ylabel('Cases',fontsize=20)
plt.title('Tamil Nadu',fontsize=20)
plt.show()


# In[ ]:


for i in range(0,len(data_states_cases)):
    if(data_states_name[i]=='West Bengal'):
        westbengal_confirmed.append(data_states_cc[i])
        westbengal_date.append(data_states_date[i])


# In[ ]:


figure=plt.figure(figsize=(30,10))
plt.plot(westbengal_date,westbengal_confirmed)
plt.xticks(rotation=90,fontsize=12)
plt.xlabel('Date',fontsize=20)
plt.ylabel('Cases',fontsize=20)
plt.title('West Bengal',fontsize=20)
plt.show()


# In[ ]:


state_hos_details=pd.read_csv('/kaggle/input/covid19-in-india/HospitalBedsIndia.csv')


# In[ ]:


state_hos_details


# In[ ]:


hospital_state_name=state_hos_details.iloc[0:35,1]
hospital_state_primary_hos=state_hos_details.iloc[0:35,2]
hospital_state_community_hos=state_hos_details.iloc[0:35,3]
hospital_state_subd_hos=state_hos_details.iloc[0:35,4]
hospital_state_d_hos=state_hos_details.iloc[0:35:,5]
hospital_state_total_hos=state_hos_details.iloc[0:35:,6]


# In[ ]:


hospital_state_name=np.asarray(hospital_state_name)
hospital_state_primary_hos=np.asarray(hospital_state_primary_hos)
hospital_state_community_hos=np.asarray(hospital_state_community_hos)
hospital_state_subd_hos=np.asarray(hospital_state_subd_hos)
hospital_state_d_hos=np.asarray(hospital_state_d_hos)
hospital_state_total_hos=np.asarray(hospital_state_total_hos)


# In[ ]:


plt.figure(figsize=(15,10))
plt.xticks(rotation=90,fontsize=12)
plt.xlabel('State Name',fontsize=13)
plt.ylabel('Number of Primary HealthCare centres',fontsize=13)
plt.bar(hospital_state_name,hospital_state_primary_hos)
plt.show()


# In[ ]:


plt.figure(figsize=(10,5))
plt.xticks(rotation=90,fontsize=12)
plt.xlabel('State Name',fontsize=13)
plt.ylabel('Number of Community HealthCare centres',fontsize=13)
plt.bar(hospital_state_name,hospital_state_community_hos)
plt.show()


# In[ ]:


plt.figure(figsize=(10,5))
plt.xticks(rotation=90,fontsize=12)
plt.xlabel('State Name',fontsize=13)
plt.ylabel('Number of Sub District Hospitals',fontsize=13)
plt.bar(hospital_state_name,hospital_state_subd_hos)
plt.show()


# In[ ]:


plt.figure(figsize=(10,5))
plt.xticks(rotation=90,fontsize=12)
plt.xlabel('State Name',fontsize=13)
plt.ylabel('Number of District Hospitals',fontsize=13)
plt.bar(hospital_state_name,hospital_state_d_hos)
plt.show()


# In[ ]:


plt.figure(figsize=(10,5))
plt.xticks(rotation=90,fontsize=12)
plt.xlabel('State Name',fontsize=13)
plt.ylabel('Number of Total Hospitals',fontsize=13)
plt.bar(hospital_state_name,hospital_state_total_hos)
plt.show()


# My analysis here shows that Uttar Pradesh has the highest number of Public Hospitals in the country followed by Maharashtra, Rajasthan , Karnataka, Tamil Nadu and Gujarat. So, in case of further explosion in the country, they will be better equiped to fight with it. However, it doesn't mean that states with less Public Care Hospitals aren't equiped well. The large number of hospitals in these states is because of the fact that these are also some of the most largest and populous state in the country. However, the fact is greater number of hospitals ensure more beds and therefore treatment to large number of people. 

# # **Predicting the future number of cases in India**

# In[ ]:


data_cases


# In[ ]:


growth_list_india=[]


# In[ ]:


for i in range(0,len(data_cases)):
    if(country_name[i]=='India'):
        growth_list_india.append(country_confirmed[i])


# In[ ]:


growth_list_india


# In[ ]:


growth_india=[]


# In[ ]:


for i in range(0,len(growth_list_india)-1):
    growth_india.append(growth_list_india[i+1]-growth_list_india[i])
    


# In[ ]:


growth_india


# In[ ]:


sum_growth=0


# In[ ]:


for i in range(0,len(growth_india)):
    sum_growth=sum_growth+growth_india[i]
averagr_growth_rate=sum_growth/len(growth_india)
    


# In[ ]:


averagr_growth_rate


# In[ ]:


average_rate=[]


# In[ ]:


for i in range(len(growth_list_india)-20,len(growth_list_india)-1):
    average_rate.append(growth_list_india[i+1]/growth_list_india[i])
    


# In[ ]:


sum_growth_mul=0


# In[ ]:


for i in range(0,len(average_rate)):
    
    sum_growth_mul=sum_growth_mul+average_rate[i]


    


# In[ ]:


sum_growth_mul=sum_growth_mul/len(average_rate)


# In[ ]:


sum_growth_mul


# As we can see in the last 20 days, the average growth rate has been around 1.22 cases of the previous day while on an average for the whole time, cases in India are rising at a speed of 1126 cases/day, which we know is wrong, since we have recently seen a spike in number of cases. So, we will take the average rate and predict the rise in cases.

# In[ ]:


prediction_for_next_15_days=[]


# In[ ]:


prediction_for_next_15_days.append(growth_list_india[len(growth_list_india)-1])


# In[ ]:


for i in range(1,15):
    prediction_for_next_15_days.append(prediction_for_next_15_days[i-1]*sum_growth_mul)


# In[ ]:


prediction_for_next_15_days


# **This is a huge number if cases rise at this rate. This means by the first week of July we can see about 23 Lakh cases. God forbids this happens**.

# In[ ]:


days_predicted=[]
for i in range(0,15):
    days_predicted.append(i)


# In[ ]:


plt.plot(days_predicted,prediction_for_next_15_days,color='Orange')
plt.xlabel('Days',fontsize=12)
plt.ylabel('Cases',fontsize=12)
plt.title('Prediction for next 15 days',fontsize=15)


# In[ ]:




