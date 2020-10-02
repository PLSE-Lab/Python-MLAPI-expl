#!/usr/bin/env python
# coding: utf-8

# The whole idea of this Analysis is to perform the Exploratory analysis on the Covid-19 Dataset and observe how corona virus has affected people and how virus progression has taken place over time

# Read the data into a Dataframe. I have analysed the data before hand and have decided to use the below data for this analysis

# In[ ]:


import pandas as pd
case = pd.read_csv("../input/coronavirusdataset/Case.csv")
patient = pd.read_csv("../input/coronavirusdataset/PatientInfo.csv")
time = pd.read_csv("../input/coronavirusdataset/Time.csv")
timeage = pd.read_csv("../input/coronavirusdataset/TimeAge.csv")
timegender = pd.read_csv("../input/coronavirusdataset/TimeGender.csv")


# In[ ]:


#Using Seaborn to plot the graph
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


case.head(5)


# In[ ]:


patient.head(5)


# Get the age of the patient from their year of birth. I am trying to plot the prob distribution plot based on the paitents age and see which age group is mosted affected by this virus.

# In[ ]:


patient['age'] = 2020 - patient['birth_year']
patient['age'].describe()


# From the below graph we can see that the mosted affected age group to catch this virus are either people in early 20s and people in there late 40s to early 50s. I was expecting a solid trend to showing most affected people in mid 30s to late 40s as a working class population and have frequent contacts with people is society. Though the normalised curve has peaked at around mid 40s.

# In[ ]:


plt.figure(figsize=(10,6))
from scipy.stats import norm
sns.distplot(patient['age'], fit=norm);
#fig = plt.figure()
plt.show()


# Below graph is just a detailed version of the above graph to accomodate to see if there is any gender variability relating to coronoa virus. The graph is some what peaked to left for male (males in early 20s) and it has peaked to right for females (females around 50s) but from graph we can not see any significant tend based on gender

# In[ ]:


plt.figure(figsize=(10,6))
age_gender= pd.concat([patient['age'], patient['sex']], axis=1).dropna(); age_gender
sns.kdeplot(age_gender.loc[(age_gender['sex']=='female'), 
            'age'], color='g', shade=True, Label='female')

sns.kdeplot(age_gender.loc[(age_gender['sex']=='male'), 
            'age'], color='r', shade=True, Label='male')
plt.show()


# Calculating percentage of patient recovered, deceased and isolated. We can see that from our dataset less than 1.5% of COVID-19 affected patient have deceased and approx 14% patients already have recovered.

# In[ ]:


df_patient=patient
infected_patient = patient.shape[0]
rp = patient.loc[patient["state"] == "released"].shape[0]
dp = patient.loc[patient["state"] == "deceased"].shape[0]
ip = patient.loc[patient["state"]== "isolated"].shape[0]
rp=rp/patient.shape[0]
dp=dp/patient.shape[0]
ip=ip/patient.shape[0]
print("The percentage of recovery is "+ str(rp*100) )
print("The percentage of deceased is "+ str(dp*100) )
print("The percentage of isolated is "+ str(ip*100) )


# In[ ]:


# Filtering patients that have released
released = df_patient[df_patient.state == 'released']
released.head()


# Below graph shows the probability distribution of released patients based on their age, thought the graph follows the same pattern as total number of patients in our datasets but it has not peaked as high for the patients with age groups greater that 40s compare to patients in their late 20s. It closely follows the pattern for probality distribution of male from above graph

# In[ ]:


plt.figure(figsize=(10,6))
sns.set_style("darkgrid")
plt.title("Age distribution of the released")
sns.kdeplot(data=released['age'], shade=True,legend=True,cumulative=False,cbar=True,kernel='gau')
plt.show()


# We have a higer proportion of female in out data complare to male

# In[ ]:


p = sns.countplot(data=patient,y = 'sex',saturation=1)


# Filtering the people who are deceased from COVID-19

# In[ ]:


dead = df_patient[df_patient.state == 'deceased']
dead.head()


# From below graph we can see that around age group 65, hence elderly patients are at higer risk with COVID-19 

# In[ ]:


plt.figure(figsize=(10,6))
sns.set_style("darkgrid")
plt.title("Age distribution of the deceased")
sns.kdeplot(data=dead['age'], shade=True,legend=True,cumulative=False,cbar=True)


# In our dataset the we have higher number of deceased male patients than female

# In[ ]:


p = sns.countplot(data=dead,y = 'sex',saturation=1)


# Where as if we look at patients that are released, we have more female patients being released

# In[ ]:


p = sns.countplot(data=released,y = 'sex', saturation=1)


# Lets combine the released and deceased patients and see if it follows the same trend

# In[ ]:


frames = [released, dead]
data1 = pd.concat(frames)
data1.head()


# 

# In[ ]:


g = sns.catplot(x="sex", col="state",
                data=data1, kind="count",
                height=4, aspect=1);


# In[ ]:


dead.head()


# In[ ]:


released.head()


# Plotting together released and deceased patients

# In[ ]:


plt.figure(figsize=(6,4))
sns.set_style("darkgrid")
plt.title("Age distribution of the released vs dead")
ax = sns.kdeplot(data=released['age'],shade=True)
sns.kdeplot(dead['age'], ax=ax, shade=True,legend=True,cbar=True)
plt.show()


# From below graph we can clearly see that patient with age more than 60 are at higher risk for COVID-19

# In[ ]:


df_patient1 = df_patient[df_patient.state != 'isolated']
plt.figure(figsize=(6,4))
sns.set(style="whitegrid")
#tips = sns.load_dataset("dead")
#print(tips)
ax = sns.barplot(x="sex", y="age",hue='state', data=df_patient1)#,order=["age", "sex"])


# In[ ]:


df_patient2 = df_patient[df_patient.state == 'isolated']
plt.figure(figsize=(6,4))
sns.set(style="whitegrid")
#tips = sns.load_dataset("dead")
#print(tips)
ax = sns.barplot(x="sex", y="age", data=df_patient2)#,order=["age", "sex"])


# In[ ]:


patient.describe()


# Now lets consider time and see how this virus has progressed over time

# In[ ]:


time.tail(5)


# From below graph we can see that the testing for COVID-19 has increased dramatically since Feb 25th 2020. In the period of less than 1 month the test have risen from almost few hundreds to more than 300k.   

# In[ ]:


plt.figure(figsize=(15,6))
sns.set(style="darkgrid")
# Plot the responses for different events and regions
plt.xticks(rotation=90)
plt.title('seaborn-matplotlib time')
sns.lineplot(x="date", y="test",data=time)


# In[ ]:


timeage['age'] = timeage['age'].str.replace(r'\D', '').astype(int)
timeage.tail()


# Below two plots shows the rise in the confirmed cases and deceased. In confirmed case we can see that in last five days we can see curve to be flattening.

# In[ ]:


plt.figure(figsize=(15,6))
sns.set(style="darkgrid")
# Plot the responses for different events and regions
plt.xticks(rotation=15)
plt.title('seaborn-matplotlib timeage')
sns.lineplot(x="date", y="confirmed",data=timeage)


# In[ ]:


plt.figure(figsize=(15,6))
sns.set(style="darkgrid")
# Plot the responses for different events and regions
plt.xticks(rotation=15)
plt.title('seaborn-matplotlib timeage')
sns.lineplot(x="date", y="deceased",data=timeage)


# This data shows the number of confirmed and deceased cases based on gender. Similar trends can be seen in below two graphs too. Trends keeps to be flattening for confirmed cases and deceased cases kept decreasig. I beleive we are seeing behaviour of flatting for confirmed cases as more people start to take precautions and limit the spread of virus.

# In[ ]:


timegender.tail()


# In[ ]:


plt.figure(figsize=(15,6))
sns.set(style="darkgrid")
# Plot the responses for different events and regions
plt.xticks(rotation=15)
plt.title('seaborn-matplotlib timeage')
sns.lineplot(x="date", y="confirmed",hue='sex',data=timegender)


# In[ ]:


plt.figure(figsize=(15,6))
sns.set(style="darkgrid")
# Plot the responses for different events and regions
plt.xticks(rotation=15)
plt.title('seaborn-matplotlib timeage')
sns.lineplot(x="date", y="deceased",hue='sex',data=timegender)

