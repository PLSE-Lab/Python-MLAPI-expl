#!/usr/bin/env python
# coding: utf-8

# Welcome. Exactly one year ago (23 Aug, 2018) being a true data scientist I decided to start tracking my alcohol consumption, so today we are going to explore the data collected over the last year and see if we can find any interesting insights.

# ![Image](https://i.ibb.co/VHsQtxk/quitzilla1-1.jpg)

# ## Loading libraries and data

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


file_folder = '../input/beer-consumption-in-20182019/'
data = pd.read_csv(f'{file_folder}/drink_dataset_money.csv')
print(data.shape)


# In[ ]:


data.head()


# So, we have 365 entries in our dataframe with date, hour, minute and money spent. "drank == 1" obviously means that I drank that day, otherwise the columns are filled with zeros.

# ## EDA

# Let's start exploring the dataset!

# In[ ]:


counts = data['drank'].sum()
print("I drank {} times over the last year, or every {:.2f} days.".format(counts, data.shape[0] / counts))


# Wow, I wonder if it is above or below the average.
# 
# Now lets calculate the **longest drinking and abstinence streaks**. What we want is to find the longest sequence of 0's or 1's in the 'drank' column, so it's possible to achieve in one line using cumsums.

# In[ ]:


s = data['drank'].astype('bool')
print("Longest abstinence time: {} days".format(s.cumsum().value_counts().max() - 1))
print("Longest drinking streak: {} days".format((~s).cumsum().value_counts().max() - 1))


# Honestly, I expected it to be worse than that. And how long can you last without a cup of beer?
# 
# Let's now look at some datetime patterns.

# In[ ]:


data['date'] = pd.to_datetime(data['date'])
data['day'] = data['date'].apply(lambda x: x.day)
data['month'] = data['date'].apply(lambda x: x.month)
data['year'] = data['date'].apply(lambda x: x.year)
data['weekday'] = data['date'].apply(lambda x: x.dayofweek)


# First I am going to explore weekday dependencies.

# In[ ]:


day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

values = data.groupby('weekday')['drank'].agg('sum')
values.index = [day_names[x] for x in values.index]
values.plot(kind='bar', figsize=(15,7), grid=True)
plt.title("What day of the week do I prefer?")
plt.xlabel("Day of the week")
plt.ylabel("Frequency")
plt.show()


# Friday and Saturday are kinda obvious, but I never thought that Monday would be the third popular day! I discover more and more interesting things about myself...
# 
# Let's see what we can find about the hour. I decided not to use minute, cause it's probably too random.

# In[ ]:


values = data.groupby('hour')['drank'].agg('sum')

for i in range(24):
    if i not in values:
        values[i] = 0
        
values.sort_index().plot(kind='bar', figsize=(15,10), grid=True)
plt.title("When do I usually start drinking?")
plt.xlabel("Hour")
plt.ylabel("Frequency")
plt.show()


# It seems like over the last year I never drank in the morning.
# The earliest time to start would be 2PM.
# 
# But what about month?

# In[ ]:


month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

values = data.groupby('month')['drank'].agg('sum')
values.index = [month_names[x - 1] for x in values.index]
values.plot(kind='bar', figsize=(10,5), grid=True)
plt.title("Which month did I drink the most beer?")
plt.xlabel("Month")
plt.ylabel("Frequency")
plt.show()


# Summer is, without a shadow of a doubt, the best time of year to live active and healthy live.

# In[ ]:


data[data['drank'] == 1]['day'].value_counts().sort_index().plot(kind='bar', figsize=(15,7), grid=True)
plt.title("What day of the month is the most popular?")
plt.xlabel("Day of the month")
plt.ylabel("Frequency")
plt.show()


# No days missing, huh. There is also something magical with the middle of the month.
# 
# Let's now look at the money spending!

# In[ ]:


money_spent = data['money_spent'].sum()
print("I spent {:.0f}$ on alcohol over the last year, or {:.2f}$ a day.".format(money_spent, data.shape[0] / money_spent))


# Omg... **254$ is a lot**. Think about all the GPU hours you could acquire on GCP or AWS to fuel your neural nets. I chose the wrong fuel for myself, but you better avoid making my mistakes.

# In[ ]:


values = data.groupby('month')['money_spent'].agg('sum')
values.index = [month_names[x - 1] for x in values.index]
values.plot(kind='bar', figsize=(15,7), grid=True)
plt.title("How much money did I spend each month?")
plt.xlabel("Month")
plt.ylabel("$")
plt.show()


# September was either tough or very fun.

# In[ ]:


values = data.groupby('weekday')['money_spent'].agg('sum')
values.index = [day_names[x] for x in values.index]
values.plot(kind='bar', figsize=(15,7), grid=True)
plt.title("How much money did I spend each weekday?")
plt.xlabel("Day of the week")
plt.ylabel("$")
plt.show()


# Yeah, there is certainly something wrong with Monday. Or with me.

# In[ ]:


values = data[data['drank'] == 1]['money_spent']
values.value_counts().sort_index().plot(kind='bar', figsize=(15,7), grid=True)
plt.title("How much money do I usually spend every time?")
plt.xlabel("Amount spent $")
plt.ylabel("Frequency")
plt.show()


# In[ ]:


print("On average I spend {:.2f}$ every time I drink.".format(values.mean()))


# ## Conclusions

# * I drank alcohol 77 times over the last year. Longest abstinence time - 2 weeks, longest drinking streak - 4 days.
# * Friday, Saturday, and, surprisingly, Monday are the most popular days.
# * I mostly drink in the evening or at night, and never earlier than 2 PM.
# * I spent 254\$, on alcohol over the last year, and 3.30\$ on average every time I drink.
# * Maximum amount spent - 25\$, and 14 times I spent no money at all.
# * Maybe I should quit.
