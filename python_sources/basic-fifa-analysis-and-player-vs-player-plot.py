#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))


# In[ ]:


data = pd.read_csv("../input/data.csv")


# In[ ]:


#print(data.info())
data.head()


# Lets see what is a max and min overall and who has it. 

# In[ ]:


print(data['Overall'].max())
data[data['Overall']==data['Overall'].max()]


# We don't want to see all columns for players. Let's show only some of them.

# In[ ]:


data[data['Overall']==data['Overall'].max()][['Overall','Name','Nationality','Age']]


# That's better. Let's do the same for min overall without dataframe index

# In[ ]:


print(data[data['Overall']==data['Overall'].           min()][['Overall','Name','Nationality','Age']].          to_string(index = False)
     )


# Let's see correlation between age and overall. 
# 
# Correlation coefficient c is in range between **-1** and **1**.
# 
# If c is close to 1 that means we have a positive correlation.
# A positive correlation indicates that one variable ('age') increases while the other variable ('overall') increases too, or 'age' decreases while 'overall' decreases. Perfect positive correlation is c=1.
# 
# If c is close to -1 we have a negative correlation. That means one variable increases while the other decreases and vice versa. Perfect negative correlation is c=-1.
# If c is close to 0 we don't have any correlation.

# In[ ]:


data['Age'].corr(data['Overall'])


# Let's see correlation between overall and value. 
# 
# If we call data['Overall'].corr(data['Value']) it will be an error because 'Value' is not numeric type. So, we can create new column 'float_value' that will contain only numeric values from column 'Value'

# In[ ]:


import re

# some values has M on the end and some K for example (100M,900K)
# if value has M we will multiply it by 1000
def value2float(value):
    temp = list(value)
    c=1
    if 'M' in temp:
        c=1000
    # subtract everything except numbers 0-9 and point 
    value_float = float(re.sub("[^0-9_.]","",value))*c
    return value_float

data['value_float']=data['Value'].map(lambda x :value2float(x))

# print correlation
print(data['Overall'].corr(data['value_float']))

# print some columns to see how it looks
data[['Name','Overall','Value','value_float']].sample(5)


# It is positive as I expected.
# 
# Let's see correlation between overall and all player ratings from crossing to gkreflexes

# In[ ]:


# index of column "Crossing"
n = data.columns.get_loc("Crossing")

# index of column "GKReflexes"
m = data.columns.get_loc("GKReflexes")

print("Correlation between Overall and")
for i in range(n, m+1):
    print('{:<15} {}'.format(data.columns[i],data['Overall'].corr(data.iloc[:,i])))


# Let's see who has the best overall for every age

# In[ ]:


min_age = data['Age'].min()
max_age = data['Age'].max()

print('{:<3} {:<30} {:<7}'.format('Age','Name','Overall'))
for i in range(min_age,max_age+1):
    
    # all players who has i age
    d1 = data[data['Age']==i]
    
    # player who has the best overall
    d2 = d1[d1['Overall']==d1['Overall'].max()]
    
    # if we have more players or none
    name = d2['Name']
    if not name.empty:
        overall = d2['Overall'].head(1).to_string(index=False)
        name_list = name.to_string(index=False).split('\n')
        name = ','.join(n.strip() for n in name_list)
        print('{:<3} {:<30} {:<7}'.format(i,name,overall))
    else:
        print("We don't have player who has {} years".format(i))
    


# Let's see who has the best overall for every jersey number. it's similar like previous example

# In[ ]:


min_jersey_num = data['Jersey Number'].min()
max_jersey_num = data['Jersey Number'].max()

print('{:<10} {:<30} {:<7}'.format('Jersey Num','Name','Overall'))
for i in range(int(min_jersey_num),int(max_jersey_num)+1):
    
    # all players who has i jersey number
    d1 = data[data['Jersey Number']==i]
    
    # player who has the best overall
    d2 = d1[d1['Overall']==d1['Overall'].max()]
    
    # if we have more players or none
    name = d2['Name']
    if not name.empty:
        overall = d2['Overall'].head(1).to_string(index=False)
        name_list = name.to_string(index=False).split('\n')
        name = ','.join(n.strip() for n in name_list)
        print('{:<10} {:<30} {:<7}'.format(i,name,overall))
    else:
        print("We don't have player who has {} jersey number".format(i))
    


# For every club in some_clubs tuple print top 5 players

# In[ ]:



some_clubs = ('Juventus', 'Real Madrid','FC Barcelona', 'Manchester United')

d2 = data[['Name','Overall','Club']]
grouped_dict = d2.groupby('Club').groups

for key in some_clubs:
    indices = grouped_dict[key]
    print(d2.loc[[*indices]].sort_values(by='Overall',ascending=False).head(5))


# Next, I will make some plots. We can try our new column 'value float' and 'Overall'

# In[ ]:


import matplotlib.pyplot as plt

data.plot(y='value_float',x='Overall',kind='scatter',alpha = 0.3)


# For next plot I got inspiration from here: https://www.kaggle.com/dczerniawko/fifa19-analysis
# but code is different

# In[ ]:


from math import pi

def compare_players_by_id(id1,id2,ratings):
    df1 = data[['ID','Name',*ratings]]
    df2 = df1[(df1['ID']==id1) | (df1['ID']==id2)]

    # ------- PART 1: Create background
    # number of variable
    categories=list(df2)[2:]
    N = len(categories)
 
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
 
    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)
 
    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
 
    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories)
 
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([40,60,80], ["60","80","100"], color='#6d6d6d', size=10)
    plt.ylim(0,100)
 
    # ------- PART 2: Add plots
    # Plot each individual = each line of the data
    # I don't do a loop, because plotting more than 3 groups makes the chart unreadable
 
    # Ind1
    values=df2.loc[df2['ID']==id1].drop(['ID','Name'],axis=1).values.flatten().tolist()
    values += values[:1]
    name_1 = df2.loc[df2['ID']==id1,'Name'].to_string(index=False)
    ax.plot(angles, values, linewidth=1, linestyle='solid', label=name_1)
    ax.fill(angles, values, 'b', alpha=0.1)

    # Ind2
    values=df2.loc[df2['ID']==id2].drop(['ID','Name'],axis=1).values.flatten().tolist()
    values += values[:1]
    name_2 = df2.loc[df2['ID']==id2,'Name'].to_string(index=False)
    ax.plot(angles, values, linewidth=1, linestyle='solid', label=name_2)
    ax.fill(angles, values, 'r', alpha=0.1)
    #ax.set_facecolor(('#b2ff00'))
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))


# In[ ]:


list_of_ratings = ['Curve','Dribbling','Stamina','Vision']
compare_players_by_id(41,169432,list_of_ratings)


# In[ ]:


list_of_ratings = ['BallControl','Acceleration','SprintSpeed','Agility','Reactions']
max_overall_id=data[data['Overall']==data['Overall'].max()][['ID']]                                                    .head(1).values
min_overall_id=data[data['Overall']==data['Overall'].min()][['ID']]                                                    .head(1).values

compare_players_by_id(*max_overall_id[0],*min_overall_id[0],list_of_ratings)


# In[ ]:


data[data['Nationality'].map(lambda x: x.startswith('Bosnia'))]                        .sort_values(by='Overall',ascending=False).head(5)


# In[ ]:


list_of_ratings=list_of_ratings+['Jumping','ShotPower']
compare_players_by_id(180206,180930,list_of_ratings)

