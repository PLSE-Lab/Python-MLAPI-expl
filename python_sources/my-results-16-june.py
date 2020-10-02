#!/usr/bin/env python
# coding: utf-8

# # Curve fitting of COVID-19 cases of India
# 
# Created by (c) Shardav Bhatt on 17 June 2020
# 

# # 1. Introduction
# 
# Jupyter Notebook Created by Shardav Bhatt
# 
# Data reference: https://www.mohfw.gov.in/ (upto 17 June 2020)
# 
# In this notebook, I have considered data of COVID-19 cases in India to perform curve fitting on it. The graphs given data and fitted data are shown. Separate graphs of number of cases, number of deaths and number of recovered are shown for cummulative data as well as daily data. The curve fitting is in terms of polynomial fitting based on method of least square.

# # 2. Importing necessary modules

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


# # 3. Extracting data from file

# In[ ]:


data = pd.read_csv('/kaggle/input/covid19-india-cases-marchmay-2020/data_16June.csv')
d = data.values
dates = d[:,0]
days = np.array(d[:,1],dtype='int16')
cummulative_cases = np.array(d[:,2], dtype='float64')
cummulative_deaths = np.array(d[:,3], dtype='float64')
cummulative_recovered = np.array(d[:,4], dtype='float64')


# # 4. Generating daily data from the cummulative data

# In[ ]:


def daily_data (y):
  daily = [None]*len(y)
  daily[0] = y[0]
  for i in range(1,len(y)):
    daily[i] = y[i]-y[i-1]
  return np.array(daily)

daily_new_cases = daily_data(cummulative_cases)
daily_new_deaths = daily_data(cummulative_deaths)
daily_new_recovered = daily_data(cummulative_recovered)


# # 5. Determining proportion of deaths and recovered cases

# In[ ]:


def proportion (x,y):
  prop = [None]*len(y)
  for i in range(len(y)):
    prop[i] = (y[i]/x[i])*100
  return np.array(prop)

prop_death = proportion (cummulative_cases, cummulative_deaths)
prop_recovered = proportion (cummulative_cases, cummulative_recovered)


# # 6. Funtion to check best degreee of polynomial
# Here I am trying different degrees of polynomial and checking accuracy of it using Mean Squared Error and $R^2$ score. I am trying polynomials upto degree 100 to check which degree is best.

# In[ ]:


def fit (x,y):
  for i in range (0,101,5):
    f = np.polyfit(x,y,deg=i)
    fval = np.polyval(f,x)
    print('Degree = %d \tMSE = %10.2f \t R^2 Score = %10.6f' %(i,mean_squared_error(y,fval),r2_score(y,fval)))


# # 7. Function to plot data and fitted data
# 
# This function plots the given data and the fitted data. Given data is considered from the uploaded data file. Fitted data is generated using $n$ degree polynomial.

# In[ ]:


def my_plot(x,y,dates,n):
  f = np.polyfit(x,y,deg=n)
  fval = np.polyval(f,x)

  date_list = []
  pos = []
  for i in range(len(dates)):
    if i%5 == 0:
        date_list.append(str(dates[i]).split()[0])
        pos.append(i)

  plt.plot(y,'ro',markersize=2)
  plt.plot(fval,'g',linewidth=1)
  plt.xticks(ticks=pos, rotation='vertical',labels=date_list)
  plt.grid(which = 'both',axis='both')
  plt.text(days[-1],y[-1],str(int(y[-1])))
  plt.ylabel('Number of Cases')
  plt.legend(['Actual Data (https://www.mohfw.gov.in/)','Fitted curve'])
  if n == 1:
    print('\nFitted curve for degree %d is Y = %fx + %f\n' %(n,f[0],f[1]))
  elif n == 2:
    print('\nFitted curve for degree %d is Y = %fx^2 + %fx + %f\n' %(n,f[0],f[1],f[2]))
  elif n == 3:
    print('\nFitted curve for degree %d is Y = %fx^3 + %fx^2 + %fx + %f\n' %(n,f[0],f[1],f[2],f[3]))
  elif n == 4:
    print('\nFitted curve for degree %d is Y = %fx^4 + %fx^3 + %fx^2 + %fx + %f\n' %(n,f[0],f[1],f[2],f[3],f[4]))
  elif n == 5:
    print('\nFitted curve for degree %d is Y = %fx^5 + %fx^4 + %fx^3 + %fx^2 + %fx + %f\n' %(n,f[0],f[1],f[2],f[3],f[4],f[5]))
  else:
    pass


# # 8. Analysis of Cummulative Cases
# It can be observed that after first week of april, the rate of increament of the cases shoot up.

# In[ ]:


fit (days, cummulative_cases)
my_plot(days, cummulative_cases, dates, 5)
plt.title('Cummulative cases of COVID-19 cases in India Mar-May 2020')
plt.show()


# # 9. Analysis of Cummulative deaths

# In[ ]:


fit (days, cummulative_deaths)
my_plot(days, cummulative_deaths, dates, 5)
plt.title('Cummulative Deaths of COVID-19 in India Mar-May 2020')
plt.show()


# # 10. Analysis of Cummulative Recovered cases

# In[ ]:


fit (days, cummulative_recovered)
my_plot(days, cummulative_recovered, dates, 5)
plt.title('Cummulative recovered cases of COVID-19 in India Mar-May 2020')
plt.show()


# # 11. Analysis of daily new cases

# In[ ]:


fit (days, daily_new_cases)
my_plot (days, daily_new_cases, dates, 5)
plt.title('Daily new cases of COVID-19 in India Mar-May 2020')
plt.show()


# # 12. Analysis of daily new deaths

# In[ ]:


fit (days, daily_new_deaths)
my_plot (days, daily_new_deaths, dates, 5)
plt.title('Daily new deaths of COVID-19 in India Mar-May 2020')
plt.show()


# # 13. Analysis of daily new recovered

# In[ ]:


fit (days, daily_new_recovered)
my_plot (days, daily_new_recovered, dates, 5)
plt.title('Daily new recovered cases of COVID-19 in India Mar-May 2020')
plt.show()


# # 14. Comparison of proportions of deaths and recovered cases
# 
# We can observe that Recovery rate has begun to increase since mid april. Death rate is still constant around 3 and slightly decreasing. These are good signs.

# In[ ]:


date_list = []
pos = []
for i in range(len(dates)):
  if i%5==0:
      date_list.append(str(dates[i]).split()[0])
      pos.append(i)

plt.plot(prop_death, 'r')
plt.text(len(prop_death)-6,prop_death[-1],str(round(prop_death[-1],2)))
plt.plot(prop_recovered, 'g')
plt.text(len(prop_recovered)-6,prop_recovered[-1],str(round(prop_recovered[-1],2)))
plt.legend(['Proportion of Death','Proportion of Recovered'])
plt.ylabel('Proportion (%)')
plt.grid(which='both',axis='both')
plt.ylim([0,100])
plt.title('Proportion of deaths and recovered cases COVID-19 in India Mar-May 2020')
plt.xticks(ticks=pos, rotation='vertical',labels=date_list)
plt.show()

