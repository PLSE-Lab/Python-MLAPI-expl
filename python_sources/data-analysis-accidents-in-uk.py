#!/usr/bin/env python
# coding: utf-8

# # Data analysis of the accidents in UK(2004-2014)
# 
# ## Table of Contents:
# * [1-Casualties](#casualties)
#     * [1.1-EDA](#eda_cas)
#     * [1.2-Gender](#gender)
#     * [1.3-Age](#age)
#     * [1.4-Severity](#severity)
# * [2-Accidents](#accidents)
#     * [2.1-EDA](#eda_acc) 
#     * [2.2-Time of the accident](#time)
#     * [2.3-Conditions](#conditions)
# * [3-Vehicles](#vehicles)
#     * [3.1-EDA](#eda_veh) 
#     * [3.2-Junction location](#location)
#     * [3.3-First point of impact](#impact)
# * [4-Key points](#key)
#    

# In[ ]:


#importing the libraries 
import numpy as np 
import pandas as pd 
import os 
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# # Casualties <a class="anchor" id="casualties"></a>

# # EDA <a class="anchor" id="eda_cas"></a>

# In[ ]:


#Importing the dataset
casualties=pd.read_csv("../input/Casualties0514.csv")


# In[ ]:


#Info of the columns
casualties.info()


# In[ ]:


#Checking the null values
casualties.isnull().sum()


# In[ ]:


casualties.head()


# In[ ]:


#Heatmap to see correlations
plt.figure(figsize=(15,10))
corr=casualties.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, linewidths=.5,annot=True,mask=mask)


# As we can see that the most of the variables are incorrelated with each other.

# ## Gender <a class="anchor" id="gender"></a>

# In[ ]:


#Values of the variable gender
casualties['Sex_of_Casualty'].value_counts()


# In[ ]:


#I'm going to drop the values -1 which are the null values
casualties = casualties[casualties.Sex_of_Casualty != -1]


# In[ ]:


#Transforming the variable into a categorical one
def map_sex(sex):
    if sex == 1:
        return 'Male'
    elif sex == 2:
        return 'Female'

casualties['Sex_of_Casualty'] = casualties['Sex_of_Casualty'].apply(map_sex)


# In[ ]:


sns.set(style="darkgrid")
plt.figure(figsize=(12,8))
genderplot = sns.countplot(x='Sex_of_Casualty',data=casualties)
genderplot.set(xlabel='Sex', ylabel='Count')
for p in genderplot.patches: 
    height = p.get_height() 
    genderplot.text(p.get_x()+p.get_width()/2., 
      height + 3, 
      '{:1.2f}'.format((height/len(casualties))*100)+'%',  
      ha="center") 


# Almost the 60% of the casualties are males.

# ## Age band <a class="anchor" id="age"></a>

# In[ ]:


#Turning the variable into a categorical one
def map_age(age):
    if age == 1:
        return '0-5'
    elif age == 2:
        return '6-10'
    elif age == 3:
        return '11-15'
    elif age == 4:
        return '16-20'
    elif age == 5:
        return '21-25'
    elif age == 6:
        return '26-35'
    elif age == 7:
        return '36-45'
    elif age == 8:
        return '46-55'
    elif age == 9:
        return '56-65'
    elif age == 10:
        return '66-75'
    elif age == 11:
        return 'over 75'
    elif age == -1:
        return "Don't know"
    

casualties['Age_Band_of_Casualty'] = casualties['Age_Band_of_Casualty'].apply(map_age)


# In[ ]:


sns.set(style="darkgrid")
plt.figure(figsize=(15,10))
ageplot=sns.countplot(x='Age_Band_of_Casualty',data=casualties,order=['0-5','6-10','11-15','16-20','21-25','26-35','36-45','46-55','56-65','66-75','over 75'])
for p in ageplot.patches: 
    height = p.get_height() 
    ageplot.text(p.get_x()+p.get_width()/2., 
      height + 3, 
      '{:1.2f}'.format((height/len(casualties))*100)+'%',  
      ha="center") 


#    more than one third of the casualties are in the age band of 26-45  years

# ## Severity of the casualties <a class="anchor" id="severity"></a>

# In[ ]:


#obtaining the values of the severity variable
casualties['Casualty_Severity'].value_counts()


# In[ ]:


# turning the variable into a categorical one
def map_severity(severity):
    if severity == 1:
        return 'Fatal'
    elif severity == 2:
        return 'Serious'
    elif severity == 3:
        return 'Slight'
    
casualties['Casualty_Severity'] = casualties['Casualty_Severity'].apply(map_severity)


# In[ ]:


sns.set(style="darkgrid")
plt.figure(figsize=(15,5))
severityplot = sns.countplot(x='Casualty_Severity',hue='Sex_of_Casualty',data=casualties,order=['Slight','Serious','Fatal'])
severityplot.set(xlabel='Severity', ylabel='Count')
for p in severityplot.patches: 
    height = p.get_height() 
    severityplot.text(p.get_x()+p.get_width()/2., 
      height + 3, 
      '{:1.2f}'.format((height/len(casualties))*100)+'%',  
      ha="center") 


# Luckily, the 90% of the accidents finish with slights casualties.

# # Accidents <a class="anchor" id="accidents"></a>

# # EDA <a class="anchor" id="eda_acc"></a>

# In[ ]:


accidents= pd.read_csv("../input/Accidents0514.csv")
accidents = accidents[accidents.Weather_Conditions != -1]
accidents = accidents[accidents.Road_Surface_Conditions != -1]


# In[ ]:


accidents.info()


# In[ ]:


accidents.head()


# In[ ]:


#Heatmap to see correlations
plt.figure(figsize=(15,10))
corr=casualties.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, linewidths=.5,annot=True,mask=mask)


# We can see that the most of the variables are incorrelated with each other.

# ## Time of the accidents<a class="anchor" id="time"></a>

# In[ ]:


acc_time = accidents[['Date','Day_of_Week','Time']]


# In[ ]:


acc_time.head()


# In[ ]:


acc_time.info()


# In[ ]:


acc_time.dropna(axis=0,inplace=True)


# In[ ]:


#creating the column, hour,day,month and year
#creating year column
def year(string):
    return int(string[6:10])
acc_time['Year']=acc_time['Date'].apply(lambda x: year(x))
#creating month column
def month(string):
    return int(string[3:5])
acc_time['Month']=acc_time['Date'].apply(lambda x: month(x))
#creating day column
def day(string):
    return int(string[0:2])
acc_time['Day']=acc_time['Date'].apply(lambda x: day(x))
#creating hour column
def hour(string):
    s=string[0:2]
    return int(s)
acc_time['Hour']=acc_time['Time'].apply(lambda x: hour(x))






# In[ ]:


sns.set(style="darkgrid")
plt.figure(figsize=(15,5))
yearplot = sns.countplot(x='Year',data=acc_time)
yearplot.set(xlabel='Year', ylabel='Count')
for p in yearplot.patches: 
    height = p.get_height() 
    yearplot.text(p.get_x()+p.get_width()/2., 
      height + 3, 
      '{:1.2f}'.format((height/len(acc_time))*100)+'%',  
      ha="center") 
plt.show()
sns.set(style="darkgrid")
plt.figure(figsize=(15,5))
monthplot = sns.countplot(x='Month',data=acc_time)
monthplot.set(xlabel='Month', ylabel='Count')
for p in monthplot.patches: 
    height = p.get_height() 
    monthplot.text(p.get_x()+p.get_width()/2., 
      height + 3, 
      '{:1.2f}'.format((height/len(acc_time))*100)+'%',  
      ha="center") 
plt.show()
sns.set(style="darkgrid")
plt.figure(figsize=(15,5))
weekplot = sns.countplot(x='Day_of_Week',data=acc_time)
weekplot.set(xlabel='Day of week', ylabel='Count')
for p in weekplot.patches: 
    height = p.get_height() 
    weekplot.text(p.get_x()+p.get_width()/2., 
      height + 3, 
      '{:1.2f}'.format((height/len(acc_time))*100)+'%',  
      ha="center") 
weekplot.set(xticklabels=['Monday','Tuesday','Wesnesday','Thursday','Friday','Saturday','Sunday'])
plt.show()
sns.set(style="darkgrid")
plt.figure(figsize=(15,8))
Hourplot = sns.countplot(x='Hour',data=acc_time)
Hourplot.set(xlabel='Hour', ylabel='Count')
for p in Hourplot.patches: 
    height = p.get_height() 
    Hourplot.text(p.get_x()+p.get_width()/2., 
      height + 3, 
      '{:1.2f}'.format((height/len(acc_time))*100)+'%',  
      ha="center") 
plt.show()


# In[ ]:


#Creating a pivot table to get a heatmap with the concentration of accidents by month over the years
#I decide to create a column of ones to get a count of the accidents
acc_time['Ones']=1
table = pd.pivot_table(acc_time, values='Ones', index=['Month'],columns=['Year'], aggfunc=np.sum)
plt.figure(figsize=(20,10))
yticks = np.array(['January','February','March','April','May','June','July','August','September','October','November','December'])
sns.set(rc={"axes.labelsize":36},font_scale=2)
sns.heatmap(table, yticklabels=yticks,linewidths=.1,annot=False,cmap='magma')


# # Conditions <a class='anchor' id='conditions'></a>

# In[ ]:


df_conditions = accidents[['Light_Conditions','Weather_Conditions','Road_Surface_Conditions',]]


# In[ ]:


df_conditions.info()


# In[ ]:


df_conditions['Severity']=casualties['Casualty_Severity']

sns.set(style="darkgrid")
plt.figure(figsize=(15,5))
lightplot = sns.countplot(x='Light_Conditions',data=df_conditions,hue='Severity',hue_order=['Slight','Serious','Fatal'])
lightplot.set(xlabel='Light conditions', ylabel='Count',xticklabels=['Daylight','Darkness Light-Lit','Darkness Light-Unlit','Darkness-No light','Darkness unknown light'])
for p in lightplot.patches: 
    height = p.get_height() 
    lightplot.text(p.get_x()+p.get_width()/2., 
      height + 3, 
      '{:1.2f}'.format((height/len(acc_time))*100)+'%',  
      ha="center")     
plt.show()

sns.set(style="darkgrid")
plt.figure(figsize=(15,8))
weatherplot = sns.countplot(x='Weather_Conditions',data=df_conditions)
weatherplot.set(xlabel='Weather conditions', ylabel='Count',xticklabels=['fine','Raining','Snowing','Fine/winds',
                                                                         'Raining/winds','Snowing/winds','Fog','Other','Unknown'])
for p in weatherplot.patches: 
    height = p.get_height() 
    weatherplot.text(p.get_x()+p.get_width()/2., 
      height + 3, 
      '{:1.2f}'.format((height/len(acc_time))*100)+'%',  
      ha="center") 
plt.show()

sns.set(style="darkgrid")
plt.figure(figsize=(15,8))
roadplot = sns.countplot(x='Road_Surface_Conditions',data=df_conditions)
roadplot.set(xlabel='Road Surface conditions', ylabel='Count',xticklabels=['Dry','Wet','Snow','Frost','flood','oil','mud'])
for p in roadplot.patches: 
    height = p.get_height() 
    roadplot.text(p.get_x()+p.get_width()/2., 
      height + 3, 
      '{:1.2f}'.format((height/len(acc_time))*100)+'%',  
      ha="center") 
plt.show()


# # Vehicles

# # EDA <a class='anchor' id='eda_veh'></a>

# In[ ]:


#Loading the dataset
vehicles=pd.read_csv('../input/Vehicles0514.csv')


# In[ ]:


print(vehicles.shape)
vehicles.head()


# In[ ]:


#list of columns
list(vehicles)


# In[ ]:


#dropping the columns that we are not going to use
vehicles.drop(['Age_of_Driver', 'Age_Band_of_Driver','Sex_of_Driver' ], axis=1,inplace=True)


# In[ ]:


#Heatmap to see correlations
plt.figure(figsize=(15,10))
corr=vehicles.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, linewidths=.5,annot=False,mask=mask)


# # Vehicle Manoeuvre <a class='anchor' id='manoeuvre'></a>

# In[ ]:


vehicles['Vehicle_Manoeuvre'].value_counts()


# In[ ]:


manoeuvre = vehicles
manoeuvre = manoeuvre[manoeuvre.Vehicle_Manoeuvre != -1]
manoeuvre = manoeuvre[manoeuvre.Vehicle_Manoeuvre != 8]
manoeuvre = manoeuvre[manoeuvre.Vehicle_Manoeuvre != 15]
manoeuvre = manoeuvre[manoeuvre.Vehicle_Manoeuvre != 6]
manoeuvre = manoeuvre[manoeuvre.Vehicle_Manoeuvre != 11]
manoeuvre = manoeuvre[manoeuvre.Vehicle_Manoeuvre != 12]
manoeuvre = manoeuvre[manoeuvre.Vehicle_Manoeuvre != 14]
manoeuvre = manoeuvre[manoeuvre.Vehicle_Manoeuvre != 1]
manoeuvre = manoeuvre[manoeuvre.Vehicle_Manoeuvre != 10]
manoeuvre = manoeuvre[manoeuvre.Vehicle_Manoeuvre != 13]



# In[ ]:


plt.figure(figsize=(15,8))

manoplot = sns.countplot(x='Vehicle_Manoeuvre',data=manoeuvre)
manoplot.set(xlabel='Vehicle_Manoeuvre',ylabel='Count',xticklabels=['Parked','Waiting to go','Slowing/stoping','moving off','turning left','turning right','Going ahead \n left hand bend','Going ahead \n right hand bend','Going ahead \n other'])
for p in manoplot.patches: 
    height = p.get_height() 
    manoplot.text(p.get_x()+p.get_width()/2., 
      height + 3, 
      '{:1.2f}'.format((height/len(manoeuvre))*100)+'%',  
      ha="center") 
plt.show()


# # Junction location <a class='anchor' id='location'></a>

# In[ ]:


vehicles['Junction_Location'].value_counts()


# In[ ]:


location = vehicles
location = location[location.Junction_Location != -1]


# In[ ]:


plt.figure(figsize=(15,8))

junctionplot = sns.countplot(x='Junction_Location',data=location)
junctionplot.set(xlabel='Junction_Location',ylabel='Count',xticklabels=['Not in\n junction','aproaching/parked \n junction','cleared \n junction','leaving \n roundabout','entering \n roundabout','leaving \n main road','entering \n main road','entering from \n slip road','mid junction'])
for p in junctionplot.patches: 
    height = p.get_height() 
    junctionplot.text(p.get_x()+p.get_width()/2., 
      height + 3, 
      '{:1.2f}'.format((height/len(location))*100)+'%',  
      ha="center") 
plt.show()


# # 1st point of impact  <a class='anchor' id='impact'></a>

# In[ ]:


vehicles['1st_Point_of_Impact'].value_counts()


# In[ ]:


vehicles['first_point_of_impact']=vehicles['1st_Point_of_Impact']
vehicles = vehicles[vehicles.first_point_of_impact != -1]


# In[ ]:


plt.figure(figsize=(15,8))

impactplot = sns.countplot(x='first_point_of_impact',data=vehicles)
impactplot.set(xlabel='first_point_of_impact',ylabel='Count',xticklabels=['did not \n impact','front','back','offside','nearside'])
for p in impactplot.patches: 
    height = p.get_height() 
    impactplot.text(p.get_x()+p.get_width()/2., 
      height + 3, 
      '{:1.2f}'.format((height/len(vehicles))*100)+'%',  
      ha="center") 
plt.show()


# # Key points

# ### Casualties
# *  Almost the 60% of the casualties are men.
# * one third of the casualties are in the age band 25-45.
# * Just the 1% of the casualties end with fatal injuries.
# 
# ### Accidents
# *  Decrease in accidents every year in the decade 2004-2014 excepting for the last year.
# * Saturday is the day of the week in where more accidents happen.
# * The most comon hours of the day in where accidents occur are at 8-9 and 16-17. These are mainly the peak hours to commute to work.
# * Almost one  third of the accidents occur with the road surface wet.
# 
# ### Vehicles
# * Almost the 50% of the accidents take place in junctions.
# * The first point of impact is the front in the 50% of the vehicles implicated on accidents.
# 

# In[ ]:




