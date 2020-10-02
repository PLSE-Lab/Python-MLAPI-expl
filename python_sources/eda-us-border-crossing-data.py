#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# The US border crossing dataset contains information of the inbound crossings at the U.S.-Canada and the U.S.-Mexico borders, thus reflecting the number of vehicles, containers, passengers or pedestrians entering the United States.

# ## Exploratory Data Analysis
# 

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


#importing pandas library as pd for better data manipulation
import pandas as pd
port_data=pd.read_excel('../input/US border crossing Data.xlsx')  #reading the excel file


# In[ ]:


port_data.head() #cheking the top 5 rows of the data


# In[ ]:


port_data.info()  #checking the information about the data (shape,size,type,etc..)


# In[ ]:


port_data.duplicated().any() # chekcing if the data contains any duplicate values.


# In[ ]:


port_data.shape #checking the shape of the data


# In[ ]:


port_data.isnull().sum() # checking for any null values in the data


# In[ ]:


port_data.columns #cheking the names for a better clarity of the columns.


# In[ ]:


port_data.describe(include='all') #by checking in the include all the describe provide all the possible information about the data column vise.


# In[ ]:


#Here we are checking that there should be equal number of port name to port code.
print("There are total {} port name",format(len(port_data["Port Name"].unique())))
print("There are total {} port code",format(len(port_data["Port Code"].unique())))


# The numbers are not matching there is something wrong maybe worng entry or port have multiple codes. lets check

# In[ ]:


port=port_data[["Port Name","Port Code","State"]].drop_duplicates() 
port[port["Port Name"].duplicated(keep=False)]


# As here we can see the Port Eastport have two codes this means it have two sub ports within port or two ports in different state with same name.

# In[ ]:


# printing all the unique Border ,State ,Measure 
l=['Border','State','Measure']

for c in l:
    print(port_data[c].unique())


# In[ ]:


#for better data visualization importing the Matplotlib and seaborn.
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
plt.figure(figsize=(12,10)) #adjusting the figure size

sns.countplot(x='State',data=port_data) #taking the horizontal axis(x) as State values,and giving the Data frame as data


# In[ ]:


state_mean = port_data.groupby(['State','Border'])['Value'].mean().reset_index()

plt.figure(figsize=(20,10))
sns.barplot(x='State',y='Value',hue='Border',data=state_mean)
plt.xlabel("State")
plt.ylabel("Average people entering")
plt.show()


# In[ ]:


plt.figure(figsize=(12,10))

sns.boxplot(x='State',y='Value',data=port_data[port_data['Value']<8000])
plt.show()


# In[ ]:


plt.figure(figsize=(10,8))
sns.kdeplot(port_data['Value'])
plt.title("Distribution of Value Variable")
plt.show()


# In[ ]:


#from datetime import date as dt
port_data['Day_Of_Week'] = port_data['Date'].apply(lambda x: x.dayofweek)

# Creating Year, Month, Week_day columns from date column
port_data['Year'] = port_data['Date'].apply(lambda x: x.year)
port_data['Month'] = port_data['Date'].apply(lambda x: x.month)
port_data['Week_day_name'] = port_data['Date'].apply(lambda x: x.day_name())

port_data.head()


# In[ ]:


plt.figure(figsize=(12,10))

sns.countplot(x='Week_day_name',data=port_data)
plt.xlabel("Week Day")
plt.ylabel("Frequency")
plt.title("Week on Week frequency of people")
# We observe that the highest influx of people in on Thursday folowed by Tuesday whereas others keep constant


# In[ ]:


year_value = port_data.groupby(['Year','Border'])['Value'].sum().reset_index()

year_value['Value'] = round(year_value['Value']/1000000,0)
year_value


# In[ ]:


plt.figure(figsize=(12,10))

#year_value.plot('Year','Value',kind='bar',label='Border', stacked=True)
sns.barplot(x='Year',y='Value',hue='Border',data=year_value)
plt.xlabel("Year")
plt.ylabel("People Count in Million")
plt.title("People from border movement year on year")
plt.show()

# Total people entering the US bifurcated by both borders in Million


# In[ ]:


week_people = port_data.groupby('Week_day_name')['Value'].mean().reset_index()
week_people


# In[ ]:


plt.figure(figsize=(12,8))

sns.barplot(x='Week_day_name',y='Value',data=week_people)
plt.xlabel("Day of Week")
plt.ylabel("Average count of people")
plt.show()
#Average count of people entering the US everyday of the week which looks to be quite constant.


# In[ ]:


plt.figure(figsize=(10,6))

sns.lineplot(x='Month',y='Value',hue='Measure',legend='full',data=port_data)
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.title('Value by MOnth')
plt.show()


# In[ ]:


Measure = port_data.groupby(['Measure','Border'])['Value'].mean().sort_values(ascending=False).reset_index()
Measure


# In[ ]:


plt.figure(figsize=(12,10))

sns.barplot(x='Measure',y='Value',hue='Border',data=Measure)
plt.xticks(rotation=90)
plt.show()

# Average count of people coming in via different modes of transportation bifurcated by both borders.


# In[ ]:


import plotly.graph_objects as go

plt.figure(figsize=(10,6))
sns.lineplot(data=port_data, x='Year', y='Value', hue='Measure',legend='full')
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.title('Measure Values Through Years')

# Year on Year number of people entering via different modes of transport


# In[ ]:


port_us_mexico=port_data[port_data['Border']=='US-Mexico Border']
port_us_Canada=port_data[port_data['Border']=='US-Canada Border']

Mexico = port_us_mexico.groupby('Measure').agg({'Value':'mean'}).reset_index()

plt.figure(figsize=(12,8))
sns.barplot(x='Measure',y='Value',data=Mexico)
plt.xlabel("Measure")
plt.ylabel("Average count of People")
plt.xticks(rotation=90)
plt.title("Average people entering per Measure Mexico Border")
plt.show()


# In[ ]:


Canada = port_us_Canada.groupby('Measure').agg({'Value':'mean'}).reset_index()

plt.figure(figsize=(12,8))
sns.barplot(x='Measure',y='Value',data=Canada)
plt.xlabel("Measure")
plt.ylabel("Average count of People")
plt.xticks(rotation=90)
plt.title("Average people entering per Measure for US-Canada Border")
plt.show()


# In[ ]:


df1 = port_us_mexico.groupby(['State']).agg({'Port Name':'count'}).reset_index()

plt.figure(figsize=(10,5))
sns.barplot(x='State',y='Port Name',data=df1)
plt.xlabel("States")
plt.ylabel("Count of Ports available")
plt.title("State wise port count in US-Mexico Border")
#plt.grid()
plt.show()

# For US-Mexico border, the total number of ports state wise


# In[ ]:


df2 = port_us_Canada.groupby(['State']).agg({'Port Name':'count'}).reset_index()

plt.figure(figsize=(10,5))
sns.barplot(x='State',y='Port Name',data=df2)
plt.xlabel("States")
plt.ylabel("Count of Ports available")
plt.title("State wise port count in US-Canada Border")
plt.grid()
plt.show()
# For US-Canada borders, the total number of ports state wise


# In[ ]:


port_us_mexico.columns


# In[ ]:


month_state=pd.DataFrame(port_us_Canada.groupby(['Month']).agg({'Value':'sum'}).reset_index())
#month_state['month']=month_state.index
plt.figure(figsize=(12,8))
sns.barplot(x='Month',y='Value',data=month_state)
plt.xlabel("month")
plt.ylabel("Value Counts")
plt.title("Number of people entering month wise for US-Canada Border")
plt.show()
# for US-Canada border, total number of people entering month wise


# In[ ]:


month_state=pd.DataFrame(port_us_mexico.groupby(['Month']).agg({'Value':'sum'}).reset_index())
#month_state['month']=month_state.index
plt.figure(figsize=(12,8))
sns.barplot(x='Month',y='Value',data=month_state)
plt.xlabel("month")
plt.ylabel("Value Counts")
plt.title("Number of people entering month wise for US-Mexico Border")
plt.show()
# for US-Mexico border, total number of people entering month wise


# In[ ]:


port_data.head(15)


# In[ ]:


top_ports = port_data.groupby(['Port Name'])['Value'].sum().sort_values(ascending=False).reset_index().head(10)

plt.figure(figsize=(12,8))
g=sns.barplot(y='Port Name',x='Value',data=top_ports)
plt.title("Top 10 Ports of Entry to US")
plt.xlabel("Count of People Entering US")
plt.show()


# In[ ]:


port_us_mexico.groupby(['Port Name'])['Value'].mean().sort_values(ascending=False).reset_index().head(15)


# In[ ]:


port_us_Canada.groupby(['Port Name'])['Value'].mean().sort_values(ascending=False).reset_index().head(15)


# In[ ]:


port_us_mexico['quarter']=port_us_mexico['Date'].dt.quarter
port_us_Canada['quarter']=port_us_Canada['Date'].dt.quarter
port_data['quarter']=port_data['Date'].dt.quarter


# In[ ]:


mexcio_quarter = port_us_mexico.groupby('quarter')['Value'].mean().sort_values(ascending=False).reset_index()


plt.figure(figsize=(12,8))

sns.barplot(x='quarter',y='Value',data=mexcio_quarter)
plt.xlabel("Quarter")
plt.ylabel("Average count of People")
#plt.xticks(rotation=90)
plt.title("Average people entering per Quarter for US-Mexico Border")
plt.show()

#NO observeable seasonal differnce in people coming in


# In[ ]:


Canada_quarter = port_us_Canada.groupby('quarter')['Value'].mean().sort_values(ascending=False).reset_index()


plt.figure(figsize=(12,8))

sns.barplot(x='quarter',y='Value',data=Canada_quarter)
plt.xlabel("Quarter")
plt.ylabel("Average count of People")
#plt.xticks(rotation=90)
plt.title("Average people entering per Quarter for US-Canada Border")
plt.show()


# ## Conclusion
# This concludes your starter analysis! To go forward from here, click the blue "Edit Notebook" button at the top of the kernel. This will create a copy of the code and environment for you to edit. Delete, modify, and add code as you please. Happy Kaggling!
