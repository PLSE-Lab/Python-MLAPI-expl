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


get_ipython().system('ls /kaggle/input')


# In[ ]:


df_hospitalbeds = pd.read_csv('/kaggle/input/covid19-in-india/HospitalBedsIndia.csv')
df_covid_india = pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv')
df_pop_2011 = pd.read_csv('/kaggle/input/covid19-in-india/population_india_census2011.csv')
df_icmr_labs=pd.read_csv('/kaggle/input/covid19-in-india/ICMRTestingLabs.csv')
df_indv_details = pd.read_csv('/kaggle/input/covid19-in-india/IndividualDetails.csv')
df_age_group = pd.read_csv('/kaggle/input/covid19-in-india/AgeGroupDetails.csv')
df_satewise_test = pd.read_csv('/kaggle/input/covid-india-updated/statewise_tested_numbers_data.csv')
df_statewise_daily = pd.read_csv('/kaggle/input/covid-india-updated/state_wise_daily.csv')
df_statewise = pd.read_csv('/kaggle/input/covid-india-updated/state_wise.csv')
df_rawdata = pd.read_csv('/kaggle/input/covid-india-updated/raw_data5.csv')
df_tested_numbers= pd.read_csv('/kaggle/input/covid-india-updated/tested_numbers_icmr_data.csv')
# df_death = pd.read_csv('/kaggle/input/covid-india-updated/death_and_recovered2.csv')
# df_districtwise= pd.read_csv('/kaggle/input/covid-india-updated/district_wise.csv')


# In[ ]:


df_satewise_test.head()


# In[ ]:


df_satewise_test.isnull().sum()


# In[ ]:


df_hospitalbeds.head()


# In[ ]:


df_hospitalbeds.isnull().sum()


# In[ ]:


df_hospitalbeds.shape


# In[ ]:


df_statewise_daily.head()


# In[ ]:


df_statewise_daily.isnull().sum()


# In[ ]:


df_statewise_daily.shape


# In[ ]:


print(df_statewise_daily)


# In[ ]:


df_statewise.head()


# In[ ]:


df_statewise.shape


# In[ ]:


df_statewise.isnull().sum()


# In[ ]:


print(df_statewise)


# In[ ]:


df_statewise.dtypes


# In[ ]:


df_age_group.head()


# In[ ]:


print(df_age_group)


# In[ ]:


print(df_rawdata)


# In[ ]:


df_rawdata.dtypes


# In[ ]:


df_rawdata.isnull().sum()


# In[ ]:


df_rawdata.shape


# In[ ]:


df = pd.DataFrame(df_rawdata)
selected_columns= df[["Entry_ID","Date Announced", "Age Bracket","Gender",  "Detected District","Detected State", "State code",
"Num Cases", "Current Status", "Patient Number"]]

df_patients = selected_columns.copy()


# In[ ]:


df_patients.head()


# In[ ]:


df_patients.isnull().sum()


# In[ ]:


df_patients.shape


# In[ ]:


# do not use plot, use plt instead, plot is an inherent function
import matplotlib.pyplot as plt
df = df_statewise



df.plot.bar("State", "Confirmed", rot=90, title= "State Wise Cases Comparison", figsize=(40, 10), fontsize=(20), color = 'b'  )
df.plot.bar("State", "Recovered", rot= 90, title = "State Wise Cases Comparison", figsize = (40, 10), fontsize= (20), color = 'r'  )
df.plot.bar("State", "Deaths", rot= 90, title = "State Wise Cases Comparison", figsize = (40, 10), fontsize= (20), color = 'g'  )


plt.show()


# In[ ]:


from matplotlib.pyplot import figure
figure(num=None, figsize=(15, 3), dpi=80, facecolor='w', edgecolor='k')
ax = plt.subplot(1,1,1)
w=0.3
x=np.arange(len(df.loc[~df['State'].isin(['Total'])]['State'].unique()))
plt.xticks(ticks=x, labels=list(df.loc[~df['State'].isin(['Total'])]['State'].values), rotation='vertical')
# ax.bar(x-w, df.loc[~df['State'].isin(['Total'])]['Confirmed'], width=w, color='b', align='center')
ax.bar(x, df.loc[~df['State'].isin(['Total'])]['Recovered'], width=w, color='g', align='center')
plt.ylabel('# of Cases')
ax1 = ax.twinx()
ax1.bar(x+w, df.loc[~df['State'].isin(['Total'])]['Deaths'], width=w, color='r', align='center')
plt.ylabel('# of Cases')
plt.show()


# In[ ]:


print(df_statewise_daily)


# In[ ]:


df_statewise_daily.dtypes


# In[ ]:


df_statewise_daily.head()


# In[ ]:


df_statewise_daily['Status'].value_counts()


# In[ ]:


df = df_statewise_daily.groupby('Status')


# In[ ]:


df_confirmed= df.get_group('Confirmed')
df_recovered= df.get_group('Recovered')
df_deceased= df.get_group('Deceased')


# In[ ]:


df_confirmed.head()


# In[ ]:


df_confirmed.shape


# In[ ]:


df_confirmed.dtypes


# In[ ]:


df1 = df_confirmed


# In[ ]:


df1.head()


# In[ ]:


column_list = list(df1)
print(column_list)


# In[ ]:


column_list.remove("Date")
column_list.remove("TT")
print(column_list)


# In[ ]:


df1.loc[:,"Total"]= df_confirmed[column_list].sum(axis=1)


# In[ ]:


df1.head()


# In[ ]:


print(df1)


# In[ ]:


df1.plot.bar("Date", "Total", rot=90, title= "Daily Cases", figsize=(40, 10), fontsize=(20), color = 'b'  )


# In[ ]:


df2 = df_recovered


# In[ ]:


col_list = list(df2)
print(col_list)


# In[ ]:


col_list.remove("Date")
col_list.remove("TT")
print(col_list)


# In[ ]:


df2["Total"] = df2[col_list].sum(axis=1)


# In[ ]:


df2.head()


# In[ ]:


df2.plot.bar("Date", "Total", rot=90, title="Daily Recovered", fontsize= (20), figsize=(40, 10), color = 'g')


# In[ ]:


df3=df_deceased


# In[ ]:


col = list(df_deceased)


# In[ ]:


col.remove("Date")
col.remove("TT")
print(col)


# In[ ]:


df3["Total"]=df3[col].sum(axis=1)


# In[ ]:


df3.head()


# In[ ]:


df3.plot.bar("Date", "Total", figsize=(40,10), fontsize= (20), title = "Daily Deceased", color= 'r')


# In[ ]:


print(df_hospitalbeds)


# In[ ]:


df1.head()


# In[ ]:


data = df1['Total']
data = data.reset_index(drop=False)
data.columns = ['Timestep', 'Total']
print(data)


# In[ ]:


print(data)


# In[ ]:


df_t= pd.read_csv('/kaggle/input/covidindialatest/COVID19India.csv')
df_t= df_t.reset_index(drop=True)
df_new = df_t.T


# In[ ]:


df_new
#df_t.head()


# In[ ]:


new_header= df_new.iloc[0]
df_new= df_new[1:]
df_new.columns= new_header
df_new.head()


# In[ ]:


columns = list(df_new)

print(columns)


# In[ ]:



df_new.loc[:,"Active"]= df_new["Confirmed"]-(df_new["Recovered"]+df_new["Deceased"])


# In[ ]:


df_new["New Cases"]= ""


# In[ ]:


print(df_new)


# In[ ]:


#Initialize iterator i
#i = 0
#for i in range(len(df_new["Confirmed"])):
#    if i == 0:
#        df_new["Confirmed"].iloc[i]=0
#    df_new["New Cases"].iloc[i] = df_new["Confirmed"].iloc[i+1] - df_new["Confirmed"].iloc[i]
        
#print(df_new)
        


# In[ ]:


df_new['New Cases'] = df_new['Confirmed'].shift(-1) - df_new['Confirmed']
print(df_new)


# In[ ]:


df_new['Daily Recovered'] = df_new['Recovered'].shift(-1)- df_new['Recovered']
df_new['Daily Deaths']= df_new['Deceased'].shift(-1) - df_new['Deceased']
print(df_new)


# In[ ]:


df_inf = df_new['Confirmed']
df_inf = df_inf.reset_index(drop=False)
df_inf.columns = ['Timestep', 'Confirmed']
df_inf.head()


# In[ ]:


df_i = df_inf['Confirmed']
df_i = df_i.reset_index(drop=False)
df_i.columns = ['Timestep', 'Confirmed']
df_i.head()


# Goal is to find value of a, b, c using logistic function

# In[ ]:


#Define logistic function
import numpy as np
def my_logistic(t, a, b, c): #a, b, c are constants 
    return c / (1 + a*np.exp(-b*t))


# In[ ]:


#Randmize a, b , c
p0 = np.random.exponential(size=3)
p0


# In[ ]:


#setting upper and lower bounds for a, b, c 
bounds = (0, [100000., 3., 1000000000.])


# In[ ]:


import scipy.optimize as optim
x = np.array(df_i['Timestep'])+1
y = np.array(df_i['Confirmed'])
(a, b, c), cov = optim.curve_fit(my_logistic, x, y, bounds=bounds, p0=p0)


# In[ ]:


def my_logistic(t):
    return c/(1 + a*np.exp(-b*t))


# In[ ]:


plt.scatter(x,y)
plt.plot(x, my_logistic(x))
plt.title('Logistic Model vs Real data')
plt.legend(['Logistic Model', 'Real data'])
plt.xlabel('Time')
plt.ylabel('Infections')


# Values of a, b, c 
# where c is the predicted peak of infected persons

# In[ ]:


print(a, b, c)


# Predicted peak is given by the value of C 
# Note: It is too early to verify if this prediction is correct
# We require more data

# In[ ]:


temp = df_new['New Cases']
temp = temp.reset_index(drop=False)
temp.columns = ['Timestep', 'New Cases']

df_spread= temp['New Cases']
df_spread = df_spread.reset_index(drop=False)
df_spread.columns = ['Timestep', 'New Cases']

print(df_spread)


# In[ ]:


df_spread.drop(df_spread.tail(1).index, inplace = True )


# In[ ]:


df_spread.head()


# In[ ]:


print(df_spread)


# In[ ]:


import scipy.optimize as optim
x = np.array(df_spread['Timestep'])+1
y = np.array(df_spread['New Cases'])


# In[ ]:


plt.scatter(x,y)
plt.title('Daily Increase in Infections')
plt.legend(['New Cases'])
plt.xlabel('Time')
plt.ylabel('New Infections')


# In[ ]:


#Define logistic function
import numpy as np
def my_logistic(t, a, b, c): #a, b, c are constants 
    return c / (1 + a*np.exp(-b*t))


# In[ ]:


#Randmize a, b , c
p0 = np.random.exponential(size=3)
p0

#setting upper and lower bounds for a, b, c 
bounds = (0, [100000., 3., 1000000000.])


# In[ ]:


import scipy.optimize as optim
x = np.array(df_spread['Timestep'])+1
y = np.array(df_spread['New Cases'])
(a, b, c), cov = optim.curve_fit(my_logistic, x, y, bounds=bounds, p0=p0)


# In[ ]:


def my_logistic(t):
    return c/(1 + a*np.exp(-b*t))


# In[ ]:


plt.scatter(x,y)
plt.plot(x, my_logistic(x))
plt.title('Logistic Model vs Spread of Infection')
plt.legend(['Logistic Model', 'Spread of Infection'])
plt.xlabel('Time')
plt.ylabel('New Cases')


# In[ ]:


print(a, b, c)


# Predicted peak is given by the value of C 
# Note: It is too early to verify if this prediction is correct We require more data

# In[ ]:


temp = df_new['Recovered']
temp = temp.reset_index(drop=False)
temp.columns = ['Time Steps','Recovered']

df_rec = temp['Recovered']
df_rec = df_rec.reset_index(drop=False)
df_rec.columns = ['Time Steps', 'Recovered']
print (df_rec)


# In[ ]:


#Define logistic function
import numpy as np
def my_logistic(t, a, b, c): #a, b, c are constants 
    return c / (1 + a*np.exp(-b*t))


# In[ ]:


#Randmize a, b , c
p0 = np.random.exponential(size=3)
p0

#setting upper and lower bounds for a, b, c 
bounds = (0, [100000., 3., 1000000000.])


# In[ ]:


import scipy.optimize as optim
x = np.array(df_rec['Time Steps'])+1
y = np.array(df_rec['Recovered'])
(a, b, c), cov = optim.curve_fit(my_logistic, x, y, bounds=bounds, p0=p0)


# In[ ]:


def my_logistic(t):
    return c/(1 + a*np.exp(-b*t))


# In[ ]:


plt.scatter(x,y)
plt.plot(x, my_logistic(x))
plt.title('Logistic Model vs Recovered')
plt.legend(['Logistic Model', 'Recovered'])
plt.xlabel('Time')
plt.ylabel('Recovery')


# In[ ]:


print (a, b , c)


# Predicted peak is given by the value of C
# Note: It is too early to verify if this prediction is correct
# We require more data

# In[ ]:


df_new.head()


# In[ ]:


df_hospitalbeds.head()


# In[ ]:


col_list = list(df_hospitalbeds)
col_list.remove("State/UT")
df_hospitalbeds['Total Hospitals'] = df_hospitalbeds[col_list].sum(axis=1)


# In[ ]:


df_hospitalbeds.head()


# In[ ]:


df_statewise.head()


# In[ ]:


df_state = df_statewise.drop(df_statewise.index[0])

df_state.head()


# In[ ]:


df_state.plot.bar("State", "Active", figsize=(40,10), fontsize= (20), title = "Statewise Active", color= 'r')


# In[ ]:


print(df_covid_india)


# In[ ]:


df_covid_india.head()


# In[ ]:


df_covid_india["State/UnionTerritory"].dtypes


# In[ ]:


df_covid_india.shape


# In[ ]:


df = df_state.sort_values(by=['Confirmed'], ascending = False)
plt.figure(figsize=(12,8), dpi=70)

plt.bar(df['State'][:5], df['Confirmed'][:5],
       color='r')
plt.ylabel('Number of Confirmed Cases', size =12 )
plt.title('States with Maximum number of Cases', size=16)
plt.show()


# In[ ]:


data_1=df_covid_india[df_covid_india['State/UnionTerritory']== "Maharashtra"]
data_1.head()


# In[ ]:


print(data_1)


# In[ ]:


data_1.head(10)


# In[ ]:


data_1["New Cases"]= ""


# In[ ]:


data_1["New Cases"]= data_1["Confirmed"].shift(-1) - data_1["Confirmed"]


# In[ ]:


data_1.head()


# **Maharashrtra New Cases seem to keep rising**

# In[ ]:


data_1.plot.bar("Date", "New Cases", rot=90, title= "Maharashtra Daily Cases", figsize=(40, 10), fontsize=(20), color = 'g'  )
plt.ylabel('Daily Cases',size=12 )

plt.show()


# In[ ]:


data_1.plot.bar("Date", 'Confirmed', rot=90, title= "Total Cases in Maharashtra", figsize= (40,10), fontsize=(20), color='r')
plt.ylabel("Total Cases", size=12)
plt.show()


# **Possible Plateau effect in Maharashtra according to data on total cases**

# In[ ]:


data_2=df_covid_india[df_covid_india['State/UnionTerritory']== "Tamil Nadu"]
data_2.head()


# In[ ]:


df = df_covid_india[df_covid_india['Date']== "03/07/20"]
df.head()


# In[ ]:


df0 = df.sort_values(by=['Confirmed'], ascending = False)
plt.figure(figsize=(12,8), dpi=70)

plt.bar(df0['State/UnionTerritory'][:7], df0['Confirmed'][:7],
       color='r')
plt.ylabel('Number of Confirmed Cases', size =12 )
plt.title('States with Maximum number of Cases', size=16)
plt.show()


# **Top 7 States With Corona Virus**

# In[ ]:


df0.head()


# In[ ]:


data_2["New Cases"]= ""


# In[ ]:


data_2["New Cases"]= data_2["Confirmed"].shift(-1)- data_2["Confirmed"]
print(data_2)


# In[ ]:


data_2.plot.bar("Date", "New Cases", rot=90, title= "Daily New Cases in Tamil Nadu", color='g', figsize= (40, 10), fontsize= (20))
plt.ylabel("Daily New Cases", size=12)
plt.show()


# **In Tamil Nadu too the Number of New Cases seem  to keep rising **

# In[ ]:


data_2.plot.bar("Date", "Confirmed", rot=90, title= "Total Cases in Tamil Nadu", color='r', figsize= (40, 10), fontsize= (20))
plt.ylabel("Total Cases", size=12)
plt.show()


# **Showing no sign of plateau**

# In[ ]:


data_3=df_covid_india[df_covid_india['State/UnionTerritory']== "Delhi"]
data_3.head()


# In[ ]:


data_3["New Cases"]= ""
data_3["New Cases"]= data_3["Confirmed"].shift(-1)- data_3["Confirmed"]


# In[ ]:


print(data_3)


# In[ ]:


data_3.plot.bar("Date", "New Cases", rot=90, figsize=(40,10), title= "Delhi Daily New Cases", fontsize=(20), color='g')
plt.ylabel("Daily New Cases", size=12)
plt.show()


# **Cases in Delhi seem to plateau as of 3rd July 2020**

# In[ ]:


data_3.plot.bar("Date", "Confirmed", rot=90, figsize=(40,10), title= "Delhi Daily Confirmed Cases", fontsize=(20), color='r')
plt.ylabel("Total Cases", size=12)
plt.show()


# **Showing probable signs of plateau**

# In[ ]:


data_4 = df_covid_india[df_covid_india['State/UnionTerritory']== "Uttar Pradesh"]
data_4.head()


# In[ ]:


data_4["New Cases"]= ""
data_4["New Cases"]= data_4["Confirmed"].shift(-1)- data_4["Confirmed"]
print(data_4)


# In[ ]:


data_4.plot.bar("Date", "New Cases", rot=90, title= "Daily New Cases U.P.", figsize=(40,10), color='g', fontsize=(20))
plt.ylabel("Daily New Cases", size=14)
plt.show()


# **There seem to be a plateau in Uttar Pradesh for number of new cases**

# In[ ]:


data_4.plot.bar("Date", "Confirmed", rot=90, title= "Total Infections in U.P.", figsize=(40,10), color='r', fontsize=(20))
plt.ylabel("Total Infections", size=14)
plt.show()


# **Total Cases do seem to plateau/non-exponential in U.P. as of 3rd July 2020**

# In[ ]:


data_5= df_covid_india[df_covid_india["State/UnionTerritory"]== "West Bengal" ]
data_5.head()


# In[ ]:


data_5["New Cases"]= ""
data_5["New Cases"]=data_5["Confirmed"].shift(-1)- data_5["Confirmed"]
print(data_5)


# In[ ]:


data_5.plot.bar("Date", 'New Cases', rot=90, title= "Daily Cases in West Bengal", figsize= (40,10), fontsize=(20), color='g')
plt.ylabel("Daily New Cases", size=12)
plt.show()


# **Cases in West Bengal seem to rise going by the visual graph**

# In[ ]:


data_5.plot.bar("Date", 'Confirmed', rot=90, title= "Total Cases in West Bengal", figsize= (40,10), fontsize=(20), color='r')
plt.ylabel("Total Cases", size=12)
plt.show()


# **Non-exponential graph, but West Bengal data in questionable**

# In[ ]:


data_6= df_covid_india[df_covid_india["State/UnionTerritory"]== "Puducherry" ]
data_6.head()


# In[ ]:


data_6["New Cases"]= ""
data_6["New Cases"]=data_6["Confirmed"].shift(-1)- data_6["Confirmed"]
print(data_6)


# In[ ]:


data_6.plot.bar("Date", 'New Cases', rot=90, title= "Daily Cases in Pondicherry", figsize= (40,10), fontsize=(20), color='g')
plt.ylabel("Daily New Cases", size=12)
plt.show()


# In[ ]:


data_6.plot.bar("Date", 'Confirmed', rot=90, title= "Total Cases in Pondicheery", figsize= (40,10), fontsize=(20), color='r')
plt.ylabel("Total Cases", size=12)
plt.show()


# **Graph is exponential due to sudden outbreak of COVID in Puducherry**

# In[ ]:




