#!/usr/bin/env python
# coding: utf-8

# # **Olympic Games: pivot and groupby beginners guide**
# 
# **Introduction**
# 
# Another contribution to Kaggle Kernels with a study case coming from [Data Camp](http://www.datacamp.com), a fantastic platform leading Data Science Education and where I found myself devoted on personal studies to aquire my python skills as well as the "data logical thinking".  This time we explore Olympic games data using many different pandas features, mastering data handling as purposed by Manipulating DataFrames with pandas chapter on Data Camp!
# 
# As a further reference, I'd like to mention one of the ["must read" articles](https://www.datacamp.com/community/tutorials/pandas-split-apply-combine-groupby) from Data Camp, where [Hugo Bowne-Anderson](https://www.linkedin.com/in/hugo-bowne-anderson-045939a5/) guide us through the so called split-apply-combine framework. As he wisely states "Groupby objects are not intuitive [...] The split-apply-combine principle is not only elegant and practical, it's something that Data Scientists use daily".
# 
# The split-apply-combine procedure was formalized by Hadley Wickham in 2011 in his paper [The Split-Apply-Combine Strategy for Data Analysis](https://www.jstatsoft.org/article/view/v040i01).
# 
# **Questions to be answered**
# 
#     1. What are the top 15 countries ranked by total number of medals?
#     2. Which countries won medals in the most distinct sports?
#     3. Which countries won medals in the most distinct sports during the cold war?
# 
# This notebook is structured as follows:
# 
#     1. Loading and exploring data
#     2. USA vs URS
#     3. Data Visualization
#     4. Conclusions
#     EXTRA. What about my lovely Hungary?

# # 1. Loading and exploring data
#         - Reading data by using pd.read_csv() and getting further df.info()
#         - Question 1
#         - Getting more detail with .pivot_table()
#         - Finding wrong entries with .groupby() and boolean filters

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Preliminary step reading Olympic Games as DataFrame and inspecting df.head()
medals = pd.read_csv('../input/.csv')
print(medals.info())
medals.head()


# #### Getting further .info() on data quality
# As shown above we have 10 columns with no missing values, which is great!

# ### Question 1 - What are the top 15 countries ranked by total number of medals?
# The columns NOC identifies the countries in our data set, so we select this columns and .value_counts()
# 
# As computed below  **Among the top 15 countries, United States is leading by far with 4335 medals**.

# In[ ]:


# Select the 'NOC' column of medals and .values_counts() it
medal_counts = medals['NOC'].value_counts()

# Print the total amount of medals using .sum()
print('The total medals: %d' % medal_counts.sum())

# Print top 15 countries ranked by medals
# Remember, here the .type is a pandas.Series to conver to DataFrame use medal_counts.to_frame()
print('\nTop 15 countries:\n', medal_counts.head(15))


# ### Refining results and focusing with .pivot_table()
# Note that using a simple .count_values() we loose track of the medal types. Therefore, what if rather than a list we want a bit more details of those top 15 countries? Let's explore our possibilities focusing on the data with .pivot_table().
# 
# For further exploration and learning take note that we could've achieved the same using the .pivot_table(margins=True)  parameter. Also, since margins apply the .sum() in both axis, notice that the total medals and its category breakdown (Gold, Silver, Bronze) is also shown! 
# 
# In addition to that the .type() with .pivot_table() is still a pandas.DataFrame, by using .sort_values() we end up with a pandas.Series only, this might be a problem in some projects. To convert the format back to Dataframe use medals_counts.to_frame(). 

# In[ ]:


# Construct the pivot table: counted
counted = medals.pivot_table(index='NOC', values='Athlete', columns='Medal', aggfunc='count')

# Create the new column: counted['totals']
counted['totals'] = counted.sum(axis='columns')

# Sort counted by the 'totals' column
counted = counted.sort_values('totals', ascending=False)

# Print the top 15 rows of counted
counted.head(15)


# In[ ]:


# Construct the pivot table using the parameter margins=True: counted
counted = medals.pivot_table(index='NOC', values='Athlete', columns='Medal', aggfunc='count', margins=True, margins_name='Totals_all')

# Sort counted by the 'totals' column
counted = counted.sort_values('Totals_all', ascending=False)

counted.head()


# ### Multi indexing with .groupby() and checking wrong entries
# Back to medals.head() we analyse and try to get a better grasp from columns. We start questioning why 'Event_gender' and 'Gender' are in separate columns, can we get any insights from this?
# 
# First we check the unique values from this data pair, selecting both columns and .drop_duplicates(), then we create a multi index with .groupby().count() to better visualize the distribution of the data pair throughout our data set.

# In[ ]:


# Select columns: ev_gen
ev_gen_uniques = medals[['Event_gender', 'Gender']].drop_duplicates()

# Print ev_gen_uniques, the index here is the index from the rows those values appear
ev_gen_uniques


# In[ ]:


# Group medals by the two columns: medals_by_gender
medals_by_gender = medals.groupby(['Event_gender', 'Gender']).count()

# Print medal_count_by_gender
medals_by_gender


# ### Narrowing down using boolean filters 
# After .groupby() we have that 'Event_gender' == 'W' **and** 'Gender'== 'Men' appeared only once throughout the data set, implying that it is probably an error. Let's make sure by checking this single wrong entry using boolean filters.
# 
# ##### Note: Boolean filters are powerfull if used properly, I really like using this technique on data exploration. It looks complicate at the beggining, but you will get used to it!

# In[ ]:


# Create a boolean series mask to be used as a filter: boolean_filter
boolean_filter = (medals.Event_gender == 'W') & (medals.Gender == 'Men')

# Use the mask to select the wrong entry that matches the filter
medals[boolean_filter]


# ### .iloc[] instead of boolean filters
# Note that from the previous .groupby() table view it was said that the index given corresponded to the row where the data pair occured in our data set. This means that the wrong entry 'Event_gender' == 'W' **and** 'Gender'== 'Men' is located on row 23675, let's use an .iloc[] to select it and we will achieve the same result as the boolean filter with less code!
# 
# After checking we correct the wrong entry to enhance overall data quality. Same analysis could be applied to the pairs (X, Men) or (X, Women), here we are leaving those unchanged for simplicity.

# In[ ]:


# Selecting the wrong entry with .iloc[]
medals.iloc[[23675]]


# In[ ]:


# Correcting the 'Gender', column index 6 on data set
medals.iloc[[23675], [6]] = 'Women'
medals.iloc[[23675]]


# # 2. USA vs URS
#         - Question 2
#         - Question 3
#         - Countries consistency on winning the most medals by Olympic edition
# 

# ### Question 2 - Which countries won medals in the most distinct sports?
# We start our comparison checking which of the countries is more diverse when it comes to 'Sport' by using .nunique().
# 
# As shown below  **United States is top one while URS is not even on the top 15 list**.
# 
# ##### Note: For an extra practice I have done two different .groupby() operation, the first hidden by # does not keep the DataFrame structure, returning a pandas.Series, the second, a bit more sophisticated and with a flexible approach, keeps the DataFrame structure and does not index NOC.

# In[ ]:


# Group medals by 'NOC'
# Compute the number of distinct sports in which each country won medals
# Sort the values of Nsports in descending order

# Conventional .groupby()
# Nsports = medals.groupby('NOC')['Sport'].nunique().sort_values(ascending=False)

# Sophisticated .groupby()
Nsports = medals[['NOC', 'Sport']].groupby('NOC', as_index=False).agg({'Sport':'nunique'}).sort_values('Sport', ascending=False)

# Print the top 15 rows of Nsports, notice that it is a Data Frame with no index
Nsports.head(15)


# ### Question 3 - Which country was the most efficient during the cold war?
# Similiar analysis as done in Question 2 with some additional boolean filters.
# In the first part we evaluate which country won medals in the most distinct sports during the cold war, in the second we analyse the countries consistency on winning more medals by Olympic edition.
# 
# As shown below  **During the cold war USA won the space run but  URS won the Olympic sports!**.

# In[ ]:


# Extract all rows for which the 'Edition' is between 1952 & 1988: during_cold_war
during_cold_war = (medals.Edition >= 1952) & (medals.Edition <= 1988)

# Extract rows for which 'NOC' is either 'USA' or 'URS': is_usa_urs
is_usa_urs = medals.NOC.isin(['USA', 'URS'])

# Use during_cold_war and is_usa_urs to create the DataFrame: cold_war_medals
cold_war_medals = medals.loc[during_cold_war & is_usa_urs]

# Group cold_war_medals by 'NOC'
country_grouped = cold_war_medals.groupby('NOC')

# Create Nsports
Nsports = country_grouped['Sport'].nunique().sort_values(ascending=False)
print(Nsports)


# In[ ]:


# Create the pivot table: medals_won_by_country
medals_won_by_country = medals.pivot_table(index='Edition', columns='NOC', values='Athlete', aggfunc='count')

# Slice medals_won_by_country: cold_war_usa_usr_medals
cold_war_usa_usr_medals = medals_won_by_country.loc[1952:1988, ['USA','URS']]
print('Consistency during cold war\n', cold_war_usa_usr_medals.idxmax(axis='columns'))
print('\nTotal counts\n', cold_war_usa_usr_medals.idxmax(axis='columns').value_counts())


# # 3. Data visualization
#         - Changing 'Medal'to an ordered Categorical data .type()
#         - .plot.area() 'Medal' over time for both USA and URS

# ### Changing medals  to pd.Categorical() and .plot.area() medals over time
# You may have noticed that the medals are ordered according to a lexicographic (dictionary) ordering: Bronze < Gold < Silver. However, you would prefer an ordering consistent with the Olympic rules: Bronze < Silver < Gold. We achieve this using Categorical types with pd.Categorical() and its parameters.
# 
# As follows we .plot.area() the US and URS (code hidden) over Olympic editions.

# In[ ]:


# Redefine 'Medal' as an ordered categorical
medals.Medal = pd.Categorical(values=medals.Medal, categories=['Bronze', 'Silver', 'Gold'], ordered=True)

# Create the DataFrame: usa
usa = medals[medals.NOC == 'USA']

# Group usa by ['Edition', 'Medal'] and aggregate over 'Athlete'
usa_medals_by_year = usa.groupby(['Edition', 'Medal'])['Athlete'].count()
# Note that usa.pivot_table(index=['Edition', 'Medal'], values='Athlete', aggfunc='count')
# Produces the same output!!

# Reshape usa_medals_by_year by unstacking
usa_medals_by_year = usa_medals_by_year.unstack(level='Medal')

# Plot the DataFrame usa_medals_by_year
usa_medals_by_year.plot.area(figsize=(12,8), title='USA medals over time in Olympic games')
plt.show()


# In[ ]:


# Create the DataFrame: urs
urs = medals[medals.NOC == 'URS']

# Group usa by ['Edition', 'Medal'] and aggregate over 'Athlete'
usa_medals_by_year = urs.groupby(['Edition', 'Medal'])['Athlete'].count()
# Note that usa.pivot_table(index=['Edition', 'Medal'], values='Athlete', aggfunc='count')
# Produces the same output!!

# Reshape usa_medals_by_year by unstacking
usa_medals_by_year = usa_medals_by_year.unstack(level='Medal')

# Plot the DataFrame usa_medals_by_year
usa_medals_by_year.plot.area(figsize=(12,8), title='URS medals over time in Olympic games')
plt.show()


# # 4. Conclusions
#         - United States is by far the most developed country in terms of sport investment, this is shown by the total number of medals won over time. USA has 15% of the total Olympic medals.
#         - With information from Questions 1, 2 and 3 we conclude that USSR has high repeatability on winning the same sports over time.
#         - From the URS medals over time visualization we conclude that the sports investment were concentrated only during the cold war, contantly declining in efficiency after this period.
#         - Hungary is definetly leading water polo and canoe aquatics disciplines, top 1 in Olympic medals in the world! Check the Extra chapter below!

# # EXTRA. What about my lovely Hungary?
#         - Where is Hungary in the Olympic medals ranking?
#         - What are the Hungarian top 3 sports?
#         - Among those sports, which discipline is the most efficient?
#         - On those top 3 disciplines is hungary top 5 in the world?
#         - Hungarian medals over time.

# In[ ]:


# Finding Hungary on the ranking!
for place, country in enumerate(counted.index):
    if country == 'HUN':
        print('Hungary is the ' + str(place+1) + ' country in the total Olympic medals ranking')
        break


# In[ ]:


# What are the Hungarian top sports?
# You can check on hun_medals the margins=True is used to make sure
# we have the same total amount of medals as stated on counted.head(), Question 1
# 1053 is the right total so we are on the right track!
hun_medals = medals[medals.NOC == 'HUN'].pivot_table(index=['Sport'], columns='Medal', values='Athlete', aggfunc='count', dropna=True, fill_value=0, margins=True)
hun_medals_sort = hun_medals.sort_values('All', ascending=False)
print('Top 3 Hungarian sports according to Olympic medals:')
hun_medals_sort.head(4)


# In[ ]:


hun_medals = medals[medals.NOC == 'HUN'].groupby(['Sport', 'Discipline'])[['Medal']].agg('count').sort_values('Medal', ascending=False).head(3)
hun_medals_nosort = medals[medals.NOC == 'HUN'].groupby(['Sport', 'Discipline'])[['Medal']].agg('count')

#hun_medals = medals[medals.NOC == 'HUN'].groupby(['Sport', 'Discipline']).agg('count')['Medal'].nlargest(6).to_frame()
fen_tot = (hun_medals_sort.All.loc['Fencing'])
aqu_tot = (hun_medals_sort.All.loc['Aquatics'])
can_tot = (hun_medals_sort.All.loc['Canoe / Kayak'])

print('Fencing has ' + str((hun_medals.Medal['Fencing'].values/fen_tot)*100)[2:6] + '% of efficiency')
print('Water polo has ' + str((hun_medals.Medal.loc['Aquatics'].values/aqu_tot)*100)[2:6] + '% of efficiency')
print('Canoe / Kayak has ' + str((hun_medals.Medal.loc['Canoe / Kayak'].values/can_tot)*100)[2:6] + '% of efficiency')

hun_medals


# In[ ]:


# Not very optimized
def hun_sorted_discipline(sport1, sport2, sport3):
    disc_mask = (medals.Discipline == sport1) | (medals.Discipline == sport2) | (medals.Discipline == sport3)
    hun_comp = medals[['NOC', 'Discipline', 'Medal']][disc_mask].groupby(['NOC', 'Discipline']).count().sort_values('Medal', ascending=False).reset_index()
    hun_comp_piv = hun_comp.pivot_table(index='NOC', columns='Discipline', values='Medal', fill_value=0)
    hun_s1 = hun_comp_piv.sort_values(by=[sport1], ascending=False)[[sport1]].head()
    hun_s2 = hun_comp_piv.sort_values(by=[sport2], ascending=False)[[sport2]].head()
    hun_s3 = hun_comp_piv.sort_values(by=[sport3], ascending=False)[[sport3]].head()
    
    print('World top 5 countries per Olympic medals \n')
    print('\n',hun_s1)
    print('\n', hun_s2)
    print('\n', hun_s3)
    
hun_sorted_discipline('Canoe / Kayak F', 'Water polo', 'Fencing')


# In[ ]:


# Create the DataFrame: urs
urs = medals[medals.NOC == 'HUN']

# Group usa by ['Edition', 'Medal'] and aggregate over 'Athlete'
usa_medals_by_year = urs.groupby(['Edition', 'Medal'])['Athlete'].count()
# Note that usa.pivot_table(index=['Edition', 'Medal'], values='Athlete', aggfunc='count')
# Produces the same output!!

# Reshape usa_medals_by_year by unstacking
usa_medals_by_year = usa_medals_by_year.unstack(level='Medal')

# Plot the DataFrame usa_medals_by_year
usa_medals_by_year.plot.area(figsize=(12,8), title='Hungary medals over time in Olympic games')
plt.show()


# ## Conclusions EXTRA
#         - WOW Hungary! Top 3 in Fencing? I didn't know that! 
#         Go check some Hungarian Fancing videos on YouTube it's awesome!

# 
# 
