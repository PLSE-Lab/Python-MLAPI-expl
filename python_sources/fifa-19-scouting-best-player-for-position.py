#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re
from math import pi
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


get_ipython().run_cell_magic('javascript', '', 'IPython.OutputArea.prototype._should_scroll = function(lines) {\n    return false;\n}')


# In[ ]:


df = pd.read_csv('../input/data.csv')


# In[ ]:


df.describe()


# In[ ]:


df.columns


# In[ ]:


positions = ['CAM', 'CB', 'CDM', 'CF', 'CM', 'LAM',
       'LB', 'LCB', 'LCM', 'LDM', 'LF', 'LM', 'LS', 'LW', 'LWB', 'RAM', 'RB', 'RCB', 'RCM', 'RDM', 'RF',
       'RM', 'RS', 'RW', 'RWB']


# In[ ]:


plt.figure(figsize=(50,40))
sns.heatmap(df.corr(), annot=True)


# In[ ]:


#Categorizing positions into 3 main categories 1.Defenders  2.Midfielders  3.Attackers
pos_field = {'Def': re.findall(r"\wB|\w\wB",str([x for x in positions])),
            'Mid': re.findall(r"\wM|\w\wM",str([x for x in positions])),
            'Att': list(set(re.findall(r"\wS|\wF|\wW",str([x for x in positions]))))}


# In[ ]:


#User input for Top 10 players by position
inputs_good = 0
while inputs_good==0:
    #user_input = input('Enter the position you want top players for: ')    #was trying to get user_input from user but faced difficulty when Kaggle runs the whole kernel
    user_input = 'CAM' 
    input_list = user_input.split(',')

    search = []
    for i in input_list:
        search.append(i.strip().upper())
    inputs_good = all(elem in positions for elem in search)
    if inputs_good:
        print('User wants to search for Top 10: ', ", ".join(search))
    else:
        print('Invalid position. Please re-enter the position (e.g. RAM, CF, CDM)')


# In[ ]:


for i in search:
    print('\n\n','Top 10', i, 'in FIFA 19', '\n')
    print(df.sort_values(i, ascending=False).head(10)[['Name', 'Nationality', 'Club', 'Overall']])


# In[ ]:


# ------- PART 1: Define a function that do a plot for one line of the dataset!
 
def make_spider(row, df, title, color):
    categories=list(df)[1:]
    N = len(categories)
    
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    # Initialise the spider plot
    ax = plt.subplot(1,5,row+1, polar=True )
    
    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='grey', size=8)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([10, 20, 30, 40, 50, 60, 70, 80,90,100], ['10', '20', '30', '40', '50', '60', '70', "80","90","100"], color="grey", size=7)
    plt.ylim(0,100)
    
    # Ind1
    values=df.loc[row].drop('Name').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
    ax.fill(angles, values, color=color, alpha=0.4)
    
    # Add a title
    plt.title(title, size=11, color=color, y=1.1)
    
# ------- PART 2: Apply to all individuals
# initialize the figure
my_dpi=96
plt.figure(figsize=(2000/my_dpi, 2000/my_dpi), dpi=my_dpi)

# Create a color palette:
my_palette = plt.cm.get_cmap("Set2", len(df.index))
plt.tight_layout()    

df0 = df.sort_values(by=search[0], ascending=False).head(5).reset_index()
df_spider = df0[['Name', 'Overall', 'Acceleration', 'Aggression', 'Agility', 'Balance', 'BallControl']].sort_values(by='Overall', ascending=False).head(5).reset_index()
df_spider.drop(columns=['index', 'Overall'], axis=1, inplace=True)

for row in range(len(df_spider.index)):
    make_spider( row=row, df=df_spider, title=df0['Name'][row], color=my_palette(row))


# In[ ]:


#printing all the top 10 players by position for reference
for i in positions:
    print('\n\n','Top 10', i, 'in FIFA 19', '\n')
    print(df.sort_values(i, ascending=False).head(10).reset_index()[['Name', 'Nationality', 'Club', 'Overall']])


# In[ ]:




