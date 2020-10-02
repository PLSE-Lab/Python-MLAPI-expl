#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv('../input/FIFA18v2.csv')


# In[3]:


# Data types and list of columns
data.dtypes


# In[4]:


# No of players and no of attributes per player
data.shape


# In[5]:


data.head()


# ## Let's Do Some Plotting

# In[6]:


# Top 10 countries with highest amount of players in FIFA
plt.title('Nationality Distribution Top 10')
plt.xlabel('Nationality')
plt.ylabel('Count of players')
data['Nationality'].value_counts().head(10).plot.bar(figsize= (15,10))


# In[7]:


# Player Age distribution
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
data['Age'].value_counts().plot.bar(figsize= (15,10))


# In[8]:


# Top 10 most valueable players


sorted_data = data.sort_values(by= 'Value', ascending= False).head(10)[['Name','Value']]
sorted_data.plot.bar(x= 'Name', y= 'Value', figsize= (15,10))
plt.title("Most Valueable Players : Top 10")
plt.xlabel("Player Name")
plt.ylabel("Value")
plt.plot();


# In[9]:


# Top 10 highest paid players

sorted_data = data.sort_values(by= 'Wage', ascending= False).head(10)[['Name','Wage']]
sorted_data.plot.bar(x= 'Name', y= 'Wage', figsize= (15,10))
plt.title("Most Paid Players : Top 10")
plt.xlabel("Player Name")
plt.ylabel("Wage")
plt.plot();


# Ok so Lionel Messi is the highest paid player per week and as we can see Madrid and Machester City has got all the money :p

# In[10]:


# Let's see players with maximum difference between current and potential rating i.e those who can have the highest rating boost
temp_data = data.copy()
temp_data['Potential_Increase'] = temp_data['Potential'] - temp_data['Overall']
temp_data.sort_values(by= 'Potential_Increase', ascending= False).head(10)[['Name','Potential_Increase']].plot.bar(x= 'Name', y= 'Potential_Increase', figsize=(15,10))
plt.xlabel('Player Name')
plt.ylabel('Potential Increase')
plt.show();


# ## Let's see what is the average pay and transfer value per age

# In[11]:


data.groupby('Age')['Value'].mean().plot.bar(figsize=(15,10))
plt.xlabel('Age')
plt.ylabel('Value')
plt.title('Mean Value for players wrto. Age')
plt.show();


# In[12]:


data.groupby('Age')['Wage'].mean().plot.bar(figsize=(15,10))
plt.xlabel('Age')
plt.ylabel('Wage')
plt.title('Mean Wage for players wrto. Age')
plt.show();


# In[13]:


temp_data.groupby('Age')['Potential_Increase'].mean().plot.bar(figsize=(15,10))
plt.xlabel('Age')
plt.ylabel('Potential')
plt.title('Mean Potential for players wrto. Age')
plt.show();


# ## As we can see, younger players have a higher potential increase than older and after 30, there is no chance of increasing potential for players (mainly due to Age factor)

# ## Let's see the best attributes players

# In[14]:


# Declaring some default formations
def draw_graph(attribute_name):
    data.sort_values(by= attribute_name, ascending= False).head(10)[['Name',attribute_name]].plot.bar(x= 'Name', y= '{}'.format(attribute_name), figsize=(15,10))
    plt.show();


# In[16]:


# Fastest Players in FIFA 18
draw_graph('Acceleration')


# In[17]:


data.columns


# In[18]:


# Best Dribblers
draw_graph('Dribbling')


# In[19]:


# Best finishers
draw_graph('Finishing')


# In[20]:


# Strongest players
draw_graph('Strength')


# In[22]:


# Best passers of the ball
draw_graph('Short Passing')


# In[24]:


# Now is the moment, best penalty takeers in FIFA 18
draw_graph('Penalties')


# ### Shocker is , NO CR7!!!!

# In[ ]:




