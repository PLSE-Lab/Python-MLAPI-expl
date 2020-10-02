#!/usr/bin/env python
# coding: utf-8

# Below is the team formation analysis of FIFA 19, this may help players playing in Manager Carreer to build the best possilbe team.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv('../input/data.csv', index_col = 'Unnamed: 0')
df.head()


# In[ ]:


df.columns


# In[ ]:


# Filtering the required Columns
data = df[['Name','Age','Overall','Potential','Club','Position','Value','Wage']]
data.head()


# In[ ]:


# Different Postions
x = df['Position'].value_counts().index
y = df['Position'].value_counts().values
plt.bar(x,y)
plt.title('Player Position vs Number of Players')
plt.xlabel('Unique Position')
plt.ylabel('Values')
plt.show()


# # Formation : 4-2-1-3

# In[ ]:


pos = ['GK','RB','LB','RCB','LCB','RDM','LDM','CAM','LW','RW','ST']

team = pd.DataFrame(columns=['Name','Age','Overall','Potential','Club','Position','Value','Wage'])
for val in pos:
    mask = data['Position'] == val
    temp = data[mask].nlargest(n = 1, columns = 'Overall')
    team = pd.concat([team,temp])
    

team.set_index('Position',inplace = True)
team


# In[ ]:


overall = team['Overall'].mean()
potential = team['Potential'].mean()
print("Overall : ",overall)
print("Potential : ",potential)


# # Formation : 4-2-4
# 

# In[ ]:


pos = ['GK','RB','LB','RCB','LCB','RDM','LDM','LW','LS','RS','RW']

team = pd.DataFrame(columns=['Name','Age','Overall','Potential','Club','Position','Value','Wage'])
for val in pos:
    mask = data['Position'] == val
    temp = data[mask].nlargest(n = 1, columns = 'Overall')
    team = pd.concat([team,temp])
    

team.set_index('Position',inplace = True)
team


# In[ ]:


overall = team['Overall'].mean()
potential = team['Potential'].mean()
print("Overall : ",overall)
print("Potential : ",potential)


# # Formation : 4-3-3
# 

# In[ ]:


pos = ['GK','RB','LB','RCB','LCB','RM','LM','CM','LW','RW','ST']

team = pd.DataFrame(columns=['Name','Age','Overall','Potential','Club','Position','Value','Wage'])
for val in pos:
    mask = data['Position'] == val
    temp = data[mask].nlargest(n = 1, columns = 'Overall')
    team = pd.concat([team,temp])
    

team.set_index('Position',inplace = True)
team


# In[ ]:


overall = team['Overall'].mean()
potential = team['Potential'].mean()
print("Overall : ",overall)
print("Potential : ",potential)


# # Formation : 3-5-2

# In[ ]:


pos = ['GK','CB','LCB','RCB','LWB','RWB','RCM','LCM','CM','RF','LF']

team = pd.DataFrame(columns=['Name','Age','Overall','Potential','Club','Position','Value','Wage'])
for val in pos:
    mask = data['Position'] == val
    temp = data[mask].nlargest(n = 1, columns = 'Overall')
    team = pd.concat([team,temp])
    

team.set_index('Position',inplace = True)
team


# In[ ]:


overall = team['Overall'].mean()
potential = team['Potential'].mean()
print("Overall : ",overall)
print("Potential : ",potential)


# # Formation : 4-4-2 Diamond

# In[ ]:


pos = ['GK','RB','LB','RCB','LCB','CDM','RCM','LCM','CAM','RF','LF']

team = pd.DataFrame(columns=['Name','Age','Overall','Potential','Club','Position','Value','Wage'])
for val in pos:
    mask = data['Position'] == val
    temp = data[mask].nlargest(n = 1, columns = 'Overall')
    team = pd.concat([team,temp])
    

team.set_index('Position',inplace = True)
team


# In[ ]:


overall = team['Overall'].mean()
potential = team['Potential'].mean()
print("Overall : ",overall)
print("Potential : ",potential)


# # Formation : 4-3-2-1

# In[ ]:


pos = ['GK','RB','LB','RCB','LCB','CM','RCM','LCM','RAM','LAM','ST']

team = pd.DataFrame(columns=['Name','Age','Overall','Potential','Club','Position','Value','Wage'])
for val in pos:
    mask = data['Position'] == val
    temp = data[mask].nlargest(n = 1, columns = 'Overall')
    team = pd.concat([team,temp])
    

team.set_index('Position',inplace = True)
team


# In[ ]:


overall = team['Overall'].mean()
potential = team['Potential'].mean()
print("Overall : ",overall)
print("Potential : ",potential)


# # Formation : 3-4-3

# In[ ]:


pos = ['GK','LCB','CB','RCB','LCM','RCM','RM','LM','RW','LW','ST']

team = pd.DataFrame(columns=['Name','Age','Overall','Potential','Club','Position','Value','Wage'])
for val in pos:
    mask = data['Position'] == val
    temp = data[mask].nlargest(n = 1, columns = 'Overall')
    team = pd.concat([team,temp])
    

team.set_index('Position',inplace = True)
team


# In[ ]:


overall = team['Overall'].mean()
potential = team['Potential'].mean()
print("Overall : ",overall)
print("Potential : ",potential)


# In[ ]:





# # Formation : 3-4-3 [ Age <= 28]

# In[ ]:


pos = ['GK','LCB','CB','RCB','LCM','RCM','RM','LM','RW','LW','ST']

team = pd.DataFrame(columns=['Name','Age','Overall','Potential','Club','Position','Value','Wage'])
for val in pos:
    mask1 = data['Position'] == val
    mask2 = data['Age'] <= 28
    mask = mask1 & mask2
    temp = data[mask].nlargest(n = 1, columns = 'Overall')
    team = pd.concat([team,temp])
    

team.set_index('Position',inplace = True)
team


# In[ ]:


overall = team['Overall'].mean()
potential = team['Potential'].mean()
print("Overall : ",overall)
print("Potential : ",potential)


# 

# # Formation : 3-4-3 [Considering Maximum Potential and Age Below 28]

# In[ ]:


pos = ['GK','LCB','CB','RCB','LCM','RCM','RM','LM','RW','LW','ST']

team = pd.DataFrame(columns=['Name','Age','Overall','Potential','Club','Position','Value','Wage'])
for val in pos:
    mask1 = data['Position'] == val
    mask2 = data['Age'] <= 28
    mask = mask1 & mask2
    temp = data[mask].nlargest(n = 1, columns = 'Potential')
    team = pd.concat([team,temp])
    

team.set_index('Position',inplace = True)
team


# In[ ]:


overall = team['Overall'].mean()
potential = team['Potential'].mean()
print("Overall : ",overall)
print("Potential : ",potential)


# In[ ]:





# In[ ]:




