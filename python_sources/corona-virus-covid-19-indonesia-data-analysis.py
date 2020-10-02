#!/usr/bin/env python
# coding: utf-8

# # So I'am going to do EDA Analysis On Corona Virus (Covid-19) cases In Indonesia
# Using Python and several Libraries 
# 
# Below is the avalable data in csv 
# Lets setup some libraries and some method that we will using

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from datetime import date
import math 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

plt.rcParams["font.family"] = "serif"



def barchart(xVar , yVar , xName , yName , title): 
    plt.figure(figsize=(12,5))
    plt.grid(zorder=0)
    plt.bar(xVar , yVar,edgecolor='black' ,color='lightgreen', zorder= 3)
    plt.plot(yVar, color = 'red' , zorder = 4 )
    plt.xlabel(xName)
    plt.xticks(rotation=90)
    plt.ylabel(yName)
    plt.title(title , fontweight='bold')
    #set axis if needed
    #plt.ylim(0,1)    
    #draw line 
    #plt.axhline(y=0.0, color='r' , linestyle='-')
    plt.show()
    plt.clf()
    
def multiBarchart(xData, multiYData, multiYLabel , xLabel , yLabel , title):    
    plt.figure(figsize=(12,5)) 
    x = np.arange(len(xData))
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    ax = plt.subplot(111)
    ax.set_xticks(x)
    ax.set_xticklabels(xData)
    interval = 0.2
    mid = math.ceil(len(multiYData)/2)
    
    
    nextCounter = 1
    for i in range(0 , len(multiYData)):    
        if(i<=mid-1):
            position = (i+1)*interval*-1
            ax.bar(x-0.2, multiYData[i], width=0.2, color='b', align='center',label=multiYLabel[i])
        elif(i>=mid-1):
            position = (nextCounter) * interval
            nextCounterer = nextCounter + 1
            ax.bar(x+position, multiYData[i] , width = 0.2 , color = 'g', align ='center', label=multiYLabel[i])
        else:
            ax.bar(x,multiYData[i] , width = 0.2 , color = 'blue', align = 'center',label=multiYLabel[i])
    ax.legend()
    plt.show()


def pieChart(title,labels, values):
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
    patches, texts = plt.pie(values,colors=colors, startangle=90)
    plt.legend(patches, labels, loc="best")
    plt.title(title)
    # Set aspect ratio to be equal so that pie is drawn as a circle.
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    plt.clf()
    
    
def histogram(xData , num_bins , facecolor , title, xLabel , yLabel):
    plt.grid(zorder = 0)
    n, bins, patches = plt.hist(xData, num_bins, facecolor=facecolor, alpha=1 , zorder = 2)
    plt.title(title , fontweight= 'bold')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.show() 
    plt.clf()
    
def multiLinePlot(xData , multiYData , multiYLabel , xAxisLabel , yAxisLabel , plotTitle):
    plt.figure(figsize=(10,5))
    plt.grid(zorder = 0)
    for i in range(0,len(multiYData)):
        plt.plot( xData, multiYData[i],label=multiYLabel[i] , zorder=3 , linewidth = 3)
    # Add legend
   
    plt.legend(loc='upper right')
    # Add title and x, y labels
    plt.title(plotTitle, fontsize=16, fontweight='bold')
    plt.xlabel(xAxisLabel)
    plt.xticks(rotation=90)
    plt.ylabel(yAxisLabel)
    plt.show()
    
# Any results you write to the current directory are saved as output.
cases = pd.read_csv('/kaggle/input/indonesia-coronavirus-cases/cases.csv')

cases.keys()


# lets see how The Data Described**

# In[ ]:


cases.tail(5)


# # Let's see how many people tested in March 2020

# In[ ]:


## 1. JUMLAH YANG DIPERIKSA 10 HARI TERAKHIR 
##    HOW MANY PATIENTS BEING CHECKED LAST 10 TEN DAYS 
last_Data = cases.tail(1)
last_Date = last_Data['date'].values[0] 

last_fiveDay = cases.tail(10)
barchart(cases.date , cases.acc_tested , 'Last Checked Date', 'How Many People  Being Checked' , 
         'Accumulative how many people being checked \n for Covid-19 until ' + last_Date + ' in Indonesia') 


# > 

# **positive and negative Tested Covid-19 in Indonesa**

# In[ ]:


acc_negative = cases.tail(1).acc_negative[cases.tail(1).first_valid_index()]
acc_confirmed = cases.tail(1).acc_confirmed[cases.tail(1).first_valid_index()]
values = [acc_negative , acc_confirmed]
labels = ['Negative Covid-19', 'Positive Covid-19']
pieChart('Confirmed Positive and Negative Cases  \n in Indonesia Until ' + last_Date,
        labels,
        values)


# ## Lets see trend the positive , negatif and deceased rate by this month Ten Days 
# 
# it seems the positive rate keeps increasing and the negative is the opposite, lets keep social distancing and physical distancing :(

# In[ ]:



positive_rate = cases['acc_confirmed'] / cases['acc_tested'] * 100
negative_rate = cases['acc_negative'] / cases['acc_tested'] * 100 
deceased_rate = cases['acc_deceased'] / cases['acc_confirmed'] * 100
multiYValues = [positive_rate, negative_rate , deceased_rate]
multiYLabels = ['Positive Rate', 'Negative Rate', 'deceased Rate']
multiLinePlot( cases['date'] , multiYValues , multiYLabels , 
              'Date ' ,'Percentage Rate', 'Trend Covid-19 Indonesia \n  Case in  March 2020')


# ### Daily Covid-19 Indonesia Confirmed Patients March 2020
# 
# 

# In[ ]:


barchart(cases.date , cases.new_confirmed
         , 'Date', 'Daily Confirmed ' , 
        'Daily Confirmed Patient Day by Day \n until ' + last_Date + ' di Indonesia') 


# **Lets see How the Data described **

# In[ ]:


patient_Dataframe = pd.read_csv('/kaggle/input/indonesia-coronavirus-cases/patient.csv')
print(patient_Dataframe.keys())

# so there are row that column sex is nan, replace with undefined
patient_Dataframe['gender'] = patient_Dataframe['gender'].replace(np.nan , 'undefined' , regex = True)
patient_Dataframe['province'] = patient_Dataframe['province'].replace(np.nan, 'undefined' , regex = True)
patient_Dataframe['hospital'] =patient_Dataframe['hospital'].replace(np.nan , 'undefined' , regex = True)
patient_Dataframe.head(5)


# # **Lets see how many male (Lelaki) and  female (Perempuan) who is positive covid-19**
# 
# there are undefined gender from the data, I hope this will give feedback to the data provider 

# In[ ]:


patientgroupbygender= patient_Dataframe[['gender', 'patient_id']].groupby('gender').count()
pieChart('Gender Positive Patient Comparison Covid \n in  Indonesia until ' + last_Date,
        patientgroupbygender.index,
        patientgroupbygender.patient_id)


# In[ ]:


patientgroupbycity = patient_Dataframe[['province', 'patient_id']].groupby('province').count()
pieChart('Confirmed Covid 19 Patient \n Distribution by Province in Indonesia until ' + last_Date,
        patientgroupbycity.index,
        patientgroupbycity.patient_id)


# # Lets see the distribution of the positive covid-19 patient by where they're being hospitalized  
# 
# there also unupdated data about the whereabout of the patients based by the hospital

# In[ ]:


patientgroupbyhospital = patient_Dataframe[['hospital', 'patient_id']].groupby('hospital').count()
pieChart('Grafik jumlah pasien Covid-19 berdasarkan \n Rumah sakit (City) di Indonesia sejak ' + last_Date,
        patientgroupbyhospital.index,
        patientgroupbyhospital.patient_id)


# # Lets see the Age Distribution of positive covid patient using histogram
# 
# from the histogram we see that there ranged 0 - 80 can also potentially 
# become covid 19 patient, from the distribution we see that age range 45 to 80 are the most who 
# vulnerable to Covid-19 , so please take care the elders !

# In[ ]:


histogram(patient_Dataframe.age , 6 , 'red',
         'Confirmed Covid Patient Distribusion \n case in Indonesia until  ' + last_Date,
         'Age',
         'Frecuency')


# 
