#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# # Import Data

# In[ ]:


data=pd.read_csv('../input/COVID-19-Bangladesh.csv')
data.sample(10)


# In[ ]:


data.info()


# # Maximum new patient in a day

# In[ ]:


x=data['new_confirmed'].max()
print('Highest affected a single day = ',x)
data[(data['new_confirmed']==x)]['date']


# # Total sample collection

# In[ ]:


print(data['daily_collected_sam'].sum())

#maximum sam collection in a day
data[(data['daily_collected_sam']==data['daily_collected_sam'].max())]['date']


# # Sequence of affected

# In[ ]:


data.plot(x='date',y='total_confirmed',figsize=(30,15),kind='bar',grid='bold',fontsize=15,title='Sequence of affected')
plt.xlabel('Date',fontsize=30)
plt.ylabel('Total Affected',fontsize=30)


# In[ ]:


#sns.lineplot(x="total_recovered", y="total_deaths",hue='date' ,data= data, palette = 'BuGn')


# In[ ]:


# sns.set_style(style='whitegrid')
# sns.countplot(x='total_recovered',hue='date',data=data,palette='rainbow')


# # New patient vs recover patient

# In[ ]:


plt.figure(figsize=(20,5))
plt.plot('date','total_recovered',data=data,c='red')
plt.plot('date','total_deaths',data=data,c='green')
plt.tick_params(axis='x',labelrotation=90.0,labelsize=12.0)
plt.tick_params(axis='y', labelsize = 12.0,pad=20.0)
plt.legend(('total_recovered','total_deaths'))
plt.show()


# # New deaths

# In[ ]:


plt.figure(figsize=(20,5))
plt.scatter('date','new_deaths',data=data,c='red')
plt.tick_params(axis='x',labelrotation=90.0)


# In[ ]:





# In[ ]:


plt.figure(figsize=(15,8))
plt.xticks(rotation=90)
sns.set()
sns.set(style="darkgrid")
#ax = sns.countplot(x=data['daily_collected_sam'], data=data)
sns.barplot(x=data['daily_collected_sam'],y=data['new_confirmed'], data=data)


# # Quarantine Update

# In[ ]:


total_quarantine=data['total_quarantine'].sum()
now_in_quarantine=data['now_in_quarantine'].sum()
released_from_quarantine=data['released_from_quarantine'].sum()
quaraintine=now_in_quarantine,released_from_quarantine
colors = ["green", "red"]
label='Present quarantine','Relesed quarantine'
plt.pie(quaraintine,labels=label,colors=colors,shadow=True,autopct='%1.1f%%',startangle=140)


# In[ ]:


print('Total quarantine =',total_quarantine)


# In[ ]:


#data['total_quarantine'].plot(x=data['date'],kind='bar',figsize=(30,20),fontsize='12',color='red')


# In[ ]:


data.plot(x='date',y='total_quarantine',figsize=(30,15),kind='bar',grid='bold',fontsize=20,color='red')
plt.title('Total Quarantine',fontdict={'fontsize':20})


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




