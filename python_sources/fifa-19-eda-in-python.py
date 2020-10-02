#!/usr/bin/env python
# coding: utf-8

# # FIFA19 Explanatory Data Analysis
# 

# ![](http://i.imgur.com/EzhxngF.jpg)

# What will you will find in this Kernel
# - Q 1. Average, maximum and minimum players count.
# - Q 2. Age vs Potential
# - Q 3. Average potential by age
# - Q 4. Players joinee as per year
# - Q 5. Players joinee as per month
# - Q 6. Height and dribblling
# - Q 7. FK Accuracy and Heading Accuracy
# - Q 8. Lefty and Righty player
# - Q 9. Valid contracts
# - Q 10. Overall aggrassion

# In[ ]:


# importing libraries
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# reading dataset
df_data = pd.read_csv('../input/data.csv')
df_data.head() #printing values in dataset


# In[ ]:


df_data.columns


# In[ ]:


def display_graph(ax, title, xlabel, ylabel, legend):
    '''
    Graph theme will be same throught the kernel
    '''
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    plt.title(title, fontsize=18, fontweight='bold')
    plt.legend(legend)
    plt.show()


# ### Q 1. Average, maximum and minimum players count.

# In[ ]:


print('In the age column there is total {} number of players and in dataset there is {} number of null in Age. Also in that data the mean (average age) is {}, maximum age is {}, and minimum age is {}, containing total {} numbers of countries. Now, lets display the other data related information based on Age.'.format(df_data['Age'].sum(), df_data['Age'].isna().sum(), format(df_data['Age'].mean(), '.2f'), df_data['Age'].max(), df_data['Age'].min(), len(df_data['Nationality'].unique())))


# In[ ]:


print('This dataset have {} players having age 16 and {} players who has age more than 42'.format(sum((df_data['Age'] == 16)),sum(df_data['Age'] >= 40)))


# In[ ]:


ax = sns.distplot(df_data[['Age']])
display_graph(ax, 'Age', 'Age count', '', ['Age'])


# ###### Insights:
# From above graph, we can see that average players count are between 21 to 27. There are sudden rise between players 15 to 21, it means that there are less number of players between this ages. But at the tail side, we can see that graph is slowly descreasing, so the players age are slightly decreasing reaching to the age 45

# In[ ]:


ax = sns.distplot(df_data[['Potential']])
display_graph(ax, 'Potential count', 'Potential', '', ['Potential'])


# ### Q 2. Age vs Potential

# In[ ]:


ax = sns.scatterplot(x = 'Age', y='Potential', data=pd.DataFrame(df_data, columns=['Age', 'Potential']))
display_graph(ax, 'Age vs Potential graph', 'Age', 'Potential', ['Age'])


# ###### Insights
# As we can see in this graph, it is clear that there data scattered in the plot above age 40 are very less, it also do not have higher potential. Calculating the average, those players who has age between 20 to 30, can have perform more better than any other ages of players.

# In[ ]:


df_age = pd.DataFrame(df_data, columns=['Name', 'Age', 'Potential', 'Nationality'])
df_age.sort_values(by='Age').head()


# In[ ]:


df_age.sort_values(by='Age').tail()


# ###### Insights
# This is age distribution, of younger and older player as per country wise. To check which country have the youngest and oldest players. So Maxico have the oldest player, named as O. Perez.

# ### Q 3. Average potential by age

# In[ ]:


df_age.groupby('Age', as_index=False).count().head(5)


# In[ ]:


ax = df_age.groupby('Age').mean().plot.bar()
display_graph(ax, 'Average potential count', 'Age', 'Potential', ['Potential'])


# ##### Insights
# Again there is plot of average age and potential graph. To plot it in bar chart, to check the overall potential score. We can see that those player have age 44 have less potential and those who have age 45 have higher potential. But again, We are considered mean value. This plot contains, overall performance based on age.

# ### 4. Players Joinee as per year

# In[ ]:


df_joined = df_data['Joined']


# In[ ]:


df_joined.isna().sum()


# In[ ]:


df_joined.dropna(inplace = True)


# In[ ]:


df_joined = df_joined.apply(lambda x: datetime.strptime(x, '%b %d, %Y'))


# In[ ]:


# get the list of years
df_year = df_joined.apply(lambda x: x.year)


# In[ ]:


ax = df_year.value_counts().plot()
display_graph(ax, 'Players growth over the year', 'Years', 'Number of players', ['Players Growth'])


# ##### Insights
# As we can see in the graph, based on data there are sudden rise of fifa player after the year 2014. Other things are self explanatory. Isn't it? :smile:

# ### Q 5. Players joinee as per month

# In[ ]:


df_month = df_joined.apply(lambda x: x.month)


# In[ ]:


df_month.sort_values(ascending = True, inplace=True)


# In[ ]:


ax = df_month.value_counts(sort = False).plot.bar()
display_graph(ax, 'Enrolling players as per month', 'Month', 'Number of players', ['Player/Month'])


# ##### Insights
# So, coming to the year wise enrollment. In july month there are sudden rise enrollment of players, and in April,  November have the lowest value of forms to enroll the players.

# ### Q 6. Height and dribblling

# In[ ]:


df_height = pd.DataFrame(df_data, columns=['Height', 'Weight', 'Strength', 'Aggression', 'Stamina', 'Dribbling'])


# In[ ]:


df_height.corr()


# In[ ]:


df_height.describe()


# ### Q 7. FK Accuracy and Heading Accuracy

# In[ ]:


accuracy = pd.DataFrame(df_data, columns=['HeadingAccuracy', 'FKAccuracy'])


# In[ ]:


accuracy.head()


# ### Q 8. Lefty and Righty player

# In[ ]:


prefered_type = df_data['Preferred Foot'].value_counts()
prefered_type


# In[ ]:


sum(df_data['Preferred Foot'].isnull())


# In[ ]:


ax = prefered_type.plot.bar()
display_graph(ax, 'Righty/Lefty Players count', 'Preffered leg', 'Number of players', ['Right', 'Left'])


# ##### Insights
# We have total 48 empty values in preferred foot other than that, you can see in graph that we have almost 14000 (precisely 13948) players who is righty and above 4000 (accuratly 4211) lefty players. Now let's plot this players as which country has more lefty/righty players

# ### Q 9. Valid contracts

# In[ ]:


df_data['Contract Valid Until'].value_counts().head(10)


# In[ ]:


df_contract = pd.DataFrame(df_data, columns=['Contract Valid Until'])


# In[ ]:


df_contract.dropna(inplace = True)


# In[ ]:


def get_only_year(dates):
    '''
    some of the date in this df contains 21 Jul, 2018 and some have only names
    so, getting only years value
    '''
    newDates = []
    for i, date in enumerate(dates):
        if(len(date)>4):
            date = date[-4:]
        newDates.append(date)
    return newDates


# In[ ]:


df_contract_valid = get_only_year(df_contract['Contract Valid Until'])


# In[ ]:


df_contract_valid = pd.Series(df_contract_valid)


# In[ ]:


len(df_contract_valid.unique())


# In[ ]:


ax = df_contract_valid.value_counts().plot()
display_graph(ax, 'Contract valid until', 'Years', 
             'Players count', ['Contract'])


# ##### Insights
# As per the data, above 5800 players' contract ending in the 2019. In 2021, it decreses to 4100/4200, then there is slightly drop to 4000 in year 2023 and massive drop of contract occurs at 2022 upto 1500.

# ### Q 10. Overall aggrassion

# In[ ]:


f = (df_data
         .loc[df_data['Position'].isin(['ST', 'GK'])]
         .loc[:, ['Value', 'Overall', 'Aggression', 'Position']]
    )
f = f[f["Overall"] >= 80]
f = f[f["Overall"] < 85]
f['Aggression'] = f['Aggression'].astype(float)


# In[ ]:


ax = sns.boxplot(x="Overall", y="Aggression", hue='Position', data=f)
display_graph(ax, 'Overall Aggression', 'Overall', 'Aggression', ['ST, GK'])


# ### Conclusion
# So, in this dataset, I have explained the rows related to age, potential, accuracy, contract and the preffered type of the player and many more. Major explaination is described in the insights of the charts. This data can be further explained with showing the information of the data as per the country and club wise.

# --------------------------

# ~If you like this kernel please give star to it. Also, follow me on [Twitter](https://twitter.com/krunal3kapadiya) or [Medium](https://medium.com/@krunal3kapadiya) for more updates. You can also check my website https://krunal3kapadiya.app ~
