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


# <div align='center'><font size="7" color="#F39C12">Medical Appointment No Show</font></div>
# <hr>
# 
# 
# <div align='left'><font size="5" color="#F39C12">A general Introduction</font></div>
# If you have any symptom on your body people want to visit hospital and talk with dochter.But they need a appointment to meet doctor. However according to the US Study is found that up tp 30% of patients miss their appointments, and $150 billion is lost every year 
# 
# **Reference: [US Study](https://www.scisolutions.com/uploads/news/Missed-Appts-Cost-HMT-Article-042617.pdf)**

# <div align='left'><font size="6" color="#F39C12">check list</font></div>
# 
# **Notice** : check the information about dataset, especially column data.You can check out the data from kaggle : [Original dataset](https://www.kaggle.com/joniarroba/noshowappointments)
# 
# Strongly recommended to read out what is this dataset about and the meaning of each columns
# 
# - Summary of columns data
#   1. PatientId : Identification of a patient 
#   2. AppointmentID : Identification of each appointment
#   3. Gender : Male or Female
#   4. ScheduledDay : The day of the actuall appointment, when they have to visit the doctor.
#   5. AppointmentDay : The day someone called or registered the appointment, this is before appointment of course.
#   6. Age : How old is patient  
#   7. Neighbourhood : Where the appointment takes place.
#   8. Scholarship : True of False . Observation, this is a broad topic, consider reading this article https://en.wikipedia.org/wiki/Bolsa_Fam%C3%ADlia
#   9. Hipertension : True or False
#   10. Diabetes: True or False 
#   11. Alcoholism : True or False
#   12. Handcap : True or False
#   13. SMS_received : 1 or more messages sent to the patient.
#   14. No-show : True or False 

# 
# 
# <div align='left'><font size="6" color="#F39C12">Structure</font></div>
# <hr>
# - **[1. Importing the prerequisite libraries](#imports)**
# - **[2. load the dataset](#loading)**
# - **[3. Data cleaning](#AC)**
# - **[4. Exploratory Data Analysis](#eda)**

# # <a name="imports"></a> 1. Importing the prerequisite libraries

# In[ ]:


import pandas as pd
# pandas = excel in python

import numpy as np
# numpy = mathe in python

import matplotlib.pyplot as plt
import seaborn as sns 
# visualization libraries

import datetime
# date library

import warnings
warnings.filterwarnings('ignore')
# ignore warning signal 


# # <a name="loading"></a> 2. load the dataset

# ## 2.1 read dataset

# In[ ]:


df=pd.read_csv('/kaggle/input/noshowappointments/KaggleV2-May-2016.csv')


# ## 2.2 Head & Tail of dataset

# In[ ]:


df.head(3)


# # <a name="AC"></a> 3. Data cleaning & Feature engineering 

# ### 3.1 Misssing value

# In[ ]:


df.isnull().any().any()
# No missing value


# ### 3.2 Duplication 

# In[ ]:


df.duplicated().sum()
# No duplication 


# ### 3.3 Incorrect datatype

# In[ ]:


df.info()
# wrong type of data: ScheduledDay,AppointmentDay


# In[ ]:


## Converting the date information in string to datetime type:
df['ScheduledDay'] = pd.to_datetime(df.ScheduledDay)
df['AppointmentDay'] = pd.to_datetime(df.AppointmentDay)


# ### 3.4 change the column name
# 
#  - Hipertension is spanisch -> in english called **Hypertension**
#  - Handcap -> **Handicap**
#  

# In[ ]:


df.rename(columns = {'Hipertension': 'Hypertension', 'Handcap': 'Handicap'}, inplace = True)


# ### 3.5 Statistical Error

# In[ ]:


df.describe()


#  > Negative age value -> wrong

# In[ ]:


df[df['Age']==-1]


# In[ ]:


df.drop([99832],inplace=True)


# ### 3.6 Age of Infant

# In[ ]:


df[df['Age']==0].shape


# In[ ]:


df[(df.Age <= 0) & ((df.Hypertension.astype(int) == 1) | (df.Diabetes.astype(int) == 1) | (df.Alcoholism.astype(int) == 1))]


# Observation 
# 
# > There 3539 patients who are 0 years old.These are small infant with few months of Age. In the normal case they don't have `Diabetes`,`Hipertension` and `Alcoholism`. And then we check that these are infant.(No case of Infant with `Diabetes`,`Hipertension` or `Alcoholism`)

# ### 3.7 Unique value of Neighborhood

# In[ ]:


print("Unique Values of `Neighbourhood`:{}"
      .format(np.sort(df.Neighbourhood.unique())))


# ### 3.8 Waiting Time

# In[ ]:


#df['AppointmentDay'] = np.where((df['AppointmentDay'] - df['ScheduledDay']).dt.days < 0, df['ScheduledDay'], df['AppointmentDay'])

# Get the Waiting Time in Days of the Patients.
df['Waiting_Time_days'] = df['AppointmentDay'] - df['ScheduledDay']
df['Waiting_Time_days'] = df['Waiting_Time_days'].dt.days


# **Notice**
# 
# There are exceptions that waiting times are negative.That mean scheduledDay was earlier than Appointment day.We need to remove it

# In[ ]:


Negative = df[df['Waiting_Time_days'] < 0].index
df.drop(Negative, inplace=True)


# ### 3.9 Split show vs Noshow

# In[ ]:


df2 = df[df['No-show']=='Yes']
df2.to_csv('only_Noshow.csv',index=False)


# # <a name="eda"></a> 4. Exploratory Data Analysis

# ### 4.1 No Show vs Show with pie-chart 

# In[ ]:


plt.figure(figsize=(17,7)) 
sorted_counts = df['No-show'].value_counts()
ax=plt.pie(sorted_counts, labels = sorted_counts.index, startangle = 90,
        counterclock = False,pctdistance=0.8 ,wedgeprops = {'width' : 0.4}, autopct='%1.0f%%');

plt.title('No Show vs Show',fontsize=20);


# Observation 
# 
# > No Show: 20%  vs  Show: 80%

# ### 4.2 Gender Comparision only with no show

# In[ ]:


plt.figure(figsize=(10,7))
sns.countplot(x='Gender',data=df2,hue='Gender')
plt.title("Gender comparision (No Show)".title(),fontsize=30)
plt.ylabel('Numer of No Show cases');


# Observation 
# 
# > Female > Male 

# ### 4.3 No show rate of differrent gender

# In[ ]:


df.groupby(['Gender','No-show'])['No-show'].count()


# In[ ]:


a = (df.groupby(['Gender','No-show'])['No-show'].count()[1] / sum(df.groupby(['Gender','No-show'])['No-show'].count()[0:2]))
print("The rate of female's no show is "+"{:.2f}".format(a*100)+'%')
b = (df.groupby(['Gender','No-show'])['No-show'].count()[3] / sum(df.groupby(['Gender','No-show'])['No-show'].count()[2:4]))
print("The rate of male's no show is "+"{:.2f}".format(b*100)+'%')


# Observation
# 
# > Gender is not a important fator of No show appointment.Because the rate of different gender shows very similar values.

# ### 4.4 Age histogram

# In[ ]:


plt.subplots(figsize=(20,8))
plt.hist(x='Age',bins=20,data=df2,edgecolor='black',color='red')
plt.title('Age histogram',fontsize=30)
plt.xlabel('Age')
plt.ylabel('Number of Noshow')
x1 = list(range(0,120,5))
plt.xticks(x1);


# Observation 
# 
# > Age interval from 0 to 5 shows the highst number of no show cases.This is because of parent.Age from 0 to 5 can't visit hospital without next of kin(NOK)

# ### 4.5 Age histogram with different Gender

# In[ ]:


f,ax=plt.subplots(2,1,figsize=(15,15))
df2[df2['Gender']=='M'].Age.plot.hist(ax=ax[0],bins=20,edgecolor='black',color='red')
ax[0].set_title('Age histogram(Men)',fontsize=30)
ax[0].set_xlabel('Age')
x1=list(range(0,110,5))
ax[0].set_xticks(x1)
df2[df2['Gender']=='F'].Age.plot.hist(ax=ax[1],bins=20,edgecolor='black',color='blue')
ax[1].set_title('Age histogram(Women)',fontsize=30)
ax[1].set_xlabel('Age')
x2=list(range(0,110,5))
ax[1].set_xticks(x2);


# Observation 
# 
# > both histograms show right skewed distribution

# ### 4.6 No Show with Time 

# In[ ]:


df2['Hour'] = pd.to_datetime(df2['ScheduledDay']).dt.hour
df2['Minute'] = pd.to_datetime(df2['ScheduledDay']).dt.minute
df2['Second'] = pd.to_datetime(df2['ScheduledDay']).dt.second
df2['Count'] = 1
df2.head()


# In[ ]:


keys = [pair for pair, df2 in df2.groupby(['Hour'])]

plt.figure(figsize=(20,5))
plt.plot(keys, df2.groupby(['Hour']).count()['Count'])
plt.xlabel('hour')
plt.ylabel('Number of Noshow')
plt.xticks(keys);


# Observation 
# 
# > High no show cases in the morning (7am to 10am). Probably because of the morning they want to sleep more rather show up appointment

# ### 4.7 Noshow with Receiving SMS

# In[ ]:


No_show_Adult = df2[df2['Age']>=18]


# Considering the Age over 18. Assumed not all minor have a cell phone.So it's not considered in this visualization.

# In[ ]:


plt.figure(figsize=(7,5))
sns.countplot(x='SMS_received',hue='No-show',data=No_show_Adult)
plt.title("SMS_received & Noshow",fontsize=30)
plt.ylabel('Number of Noshow');


# Observation
# 
# > Receiving SMS is not a crucial factor of determininig No-show. As you can see above there are a lot of cases that they didn't show up their medical appointment even they received a message from hospital or doctor.

# ### 4.8 No show with Neighborhood 

# #### 4.8.1 No show counting with Neighbourhood

# In[ ]:


plt.figure(figsize=(20,4))
plt.xticks(rotation='vertical')
ax = sns.countplot(x=np.sort(df2.Neighbourhood))
ax.set_title("No show by Neighbourhood",fontsize=30);


# #### 4.8.2 No show rate with Neighbourhood

# In[ ]:


RateNeighbourhood = df[df['No-show']=='No'].groupby(['Neighbourhood']).size() / df.groupby(['Neighbourhood']).size() 


# In[ ]:


plt.figure(figsize=(20,4))
plt.xticks(rotation='vertical')
sns.barplot(x=RateNeighbourhood.index,y=RateNeighbourhood)
plt.title("No show rate by Neighbourhood",fontsize=30);


# Observation 
# 
# > Rate of No show by neighbourhood are almost same.

# ### 4.9 No show with Hypertension

# #### 4.9.1 No show counting by Hypertension

# In[ ]:


plt.figure(figsize=(10,7))
sns.countplot(data=df,x='No-show',hue='Hypertension')
plt.title("No show with Hypertension",fontsize=30);


# #### 4.9.2 No show rate with Hypertension

# In[ ]:


RateHypertension = df[df['No-show']=='No'].groupby(['Hypertension']).size() / df.groupby(['Hypertension']).size() 


# In[ ]:


plt.figure(figsize=(7,4))
plt.xticks(rotation='vertical')
sns.barplot(x=RateHypertension.index,y=RateHypertension)

plt.title("No show rate by Neighbourhood",fontsize=20)
plt.ylabel("Rate of No show");


# ### 4.10 No show with Handicap 

# In[ ]:


plt.figure(figsize=(10,7))
sns.countplot(data=df,x='No-show',hue='Handicap')
plt.title("No show with Handicap",fontsize=30);


# In[ ]:


RateHandicap = df[df['No-show']=='No'].groupby(['Handicap']).size() / df.groupby(['Handicap']).size() 


# In[ ]:


plt.figure(figsize=(7,4))
plt.xticks(rotation='vertical')
sns.barplot(x=RateHandicap.index,y=RateHandicap)
plt.title("No show rate by Handicap",fontsize=20)
plt.ylabel("Rate of No show");


# ### 4.12 No show with Waiting Time days

# In[ ]:


plt.figure(figsize=(30,7))
sns.countplot(data=df,x='Waiting_Time_days',hue='No-show')
plt.title("No show with Waiting Time",fontsize=30);


# In[ ]:


RateWaitingTimeDay = df[df['No-show']=='No'].groupby(['Waiting_Time_days']).size() / df.groupby(['Waiting_Time_days']).size() 


# In[ ]:


plt.figure(figsize=(30,7))
plt.xticks(rotation='vertical')
sns.barplot(x=RateWaitingTimeDay.index,y=RateWaitingTimeDay)
plt.title("No show rate by WaitingTimeDay",fontsize=20)
plt.ylabel("Rate of No show");


# ### 4.12 Correlation matrix of factors 

# In[ ]:


df.replace({'Yes':1, 'No': 0},inplace=True)
# In order to get a correation in number we need to conver into numeric value


# In[ ]:


c= df.corr()
mask = np.triu(np.ones_like(c, dtype=np.bool))


# In[ ]:


sns.heatmap(df.corr(),annot=True,cmap='RdYlGn',linewidths=0.2,mask=mask) #data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(10,8)


# ## Reference
# 
# Useful Kernal : [Predict Show/NoShow - EDA+Visualization+Model](https://www.kaggle.com/samratp/predict-show-noshow-eda-visualization-model)
# 
# If you like my notebook please give a upvote 
# 
# Thank you all very much
