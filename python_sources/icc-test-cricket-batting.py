#!/usr/bin/env python
# coding: utf-8

# # Test Cricket Batting Analysis

# In[ ]:


# This notbook is analysis of test cricket batting.
# I am doing this first time so please comment what you like or not in this notbook. 
# Suggestion are very important for me.
# if you find anything worng please specify or upvote if you learn from this.


# In[ ]:


# Importing libraries, we need further in this notebook.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# read icc test cricket data in variable test_cricket
# creating test_cricket dataframe by reading icc test cricket csv file.
test_cricket = pd.read_csv('/kaggle/input/icc-test-cricket-runs/ICC Test Batting Figures.csv', encoding='latin1')


# In[ ]:


# check the top of the dataframe
test_cricket.head()


# In[ ]:


# viewing the bottom eight line of the dataframe
test_cricket.tail(8)


# In[ ]:


# Player profile columns is not usable for us.
# Let drop it first
test_cricket.drop('Player Profile',axis=1,inplace=True)


# In[ ]:


# now describe the test cricket dataframe
test_cricket.describe(include='all', exclude=None)


# ## From above two cells we get some information
# #### one is HS column contain ' * ' which means this player was NOT OUT when he made his highest score
# #### second is most of the column contain word ' - ' which means these data are not available for those players

# In[ ]:


# create new column and fill it by True if player is not out when he creates his High Score
test_cricket['HS_Not_Out'] = test_cricket['HS'].str.contains('*',regex=False)

# remove * from HS
test_cricket['HS'] = test_cricket['HS'].str.replace('*','')

# Check our new dataframe cojumn name HS_not_out
# test_cricket.head()


# In[ ]:


# convert HS_Not_Out column into integer means 1 for True and 0 for False
test_cricket['HS_Not_Out'] = test_cricket.HS_Not_Out.astype(int)
# test_cricket.head()


# In[ ]:


# now first check info of dataframe
test_cricket.info()


# In[ ]:


# DataFrame actual memory usage before converting dtype in bytes
test_cricket.memory_usage(deep=True).sum()


# #### above cell tells that some column are in object datatype instead of float32 or int64

# In[ ]:


# this function changes dtype of column after removing - key word from the column

def change_dtype_replace_dash(col,datatype='int64'):
    test_cricket[col] = test_cricket[col].str.replace('-','0')
    test_cricket[col] = test_cricket[col].astype(datatype)


# In[ ]:


# only Avg has float value
change_dtype_replace_dash(col='Avg',datatype='float32')

# creating a list to change dtype as int64 and remove dash
col_list = ['Inn','HS','NO','Runs','100','50','0']
for col in col_list:
    change_dtype_replace_dash(col)
    
# view test_cricket data frame after changes in dtype
# test_cricket.head()

# dataframe info after changes and compare it from old
test_cricket.info()


# In[ ]:


# DataFrame actual memory usage after converting dtype
# result always in bytes
test_cricket.memory_usage(deep=True).sum()


# In[ ]:


# now describe the dataframe
test_cricket.describe()


# <h2> How many players are not out when they create their highest score? </h2>

# In[ ]:


test_cricket['HS_Not_Out'].value_counts()


# Answer <==> out of 3001 players 657 players are not out

# In[ ]:


# As we can see above player column has two things name and country in parentheses
# break this in two part
# create new data_frame by spliting player name and country in different columns
player_country = test_cricket['Player'].str.split("(",expand=True)
player_country.head()


# In[ ]:


# from above table we can see three column instead of two, WHY?
# Let's check it to find difference and improve it
# we first get unique value of column 2
player_country[2].unique()


# In[ ]:


# now check index where this 'PAK)' occurs
player_country[player_country[2] == 'PAK)']


# In[ ]:


# we can get same result by this line also (by checking datatype)
player_country[player_country[2].apply(lambda x: type(x) != type(None))]


# In[ ]:


# updating player name
player_country.iloc[2113,0] = 'Mohammad Nawaz (3)'

# updating country
player_country.iloc[2113,1] = 'PAK'


# In[ ]:


player_country.drop(2,axis=1,inplace=True)
player_country.head()


# In[ ]:


# column 0 contains one extra space after Last name Let's check it
player_country[0].loc[:4].str.len()


# In[ ]:


# remove extra space and replace/add it with our test_cricket player column
test_cricket['Player'] = player_country[0].str.strip()


# In[ ]:


# check extra space is removed or not
test_cricket['Player'].loc[:4].str.len()


# In[ ]:


# let's check all Country name
player_country[1].unique()


# In[ ]:


player_country[1].nunique()


# In[ ]:


# word ICC is not belong to any country
# so remove ), ICC/ and /ICC from player_country dataframe and add it in original dataframe
test_cricket['Country'] = player_country[1].str.replace('/ICC','').str.replace('ICC/','').str.replace(')','')
test_cricket.head()


# In[ ]:


test_cricket.Country.nunique()


# In[ ]:


countries = test_cricket.Country.unique().tolist()
countries


# In[ ]:


# breaking Span column in two column
career_span = test_cricket.Span.str.split('-',expand=True)
career_span.head()


# In[ ]:


# changeing column name
career_span.rename(columns={0:'PStart',1:'PStop'},inplace=True)
career_span.info()


# In[ ]:


career_span.memory_usage(deep=True).sum()


# In[ ]:


# changing data type of both columns
career_span = career_span.astype('int64')


# In[ ]:


career_span.info()


# In[ ]:


career_span.memory_usage(deep=True).sum()


# In[ ]:


# creating new column with data How many years a player played?
career_span['Span_Years'] = career_span.PStop - career_span.PStart
career_span.head()


# In[ ]:


# joining both the dataframe in one data frame
test_cricket = test_cricket.join(career_span)


# In[ ]:


test_cricket.head()


# <h2> Country wise report on how many plyers are not out when they create their High Score  </h2>

# In[ ]:


# this is the country wise counting of players for not out when they create their highest score
test_cricket.groupby('HS_Not_Out').Country.value_counts()


# In[ ]:


plt.figure(figsize=(12,8))
f = sns.barplot(x='HS_Not_Out',y='Country',data=test_cricket,estimator=sum,ci=None)
plt.show()


# <h2> How many players(country wise) join test cricket latest? </h2>

# In[ ]:


test_cricket[test_cricket.PStart == test_cricket.PStart.max()].Country.value_counts()


# Answer <==> 8 from Afganistan and South Africa, 1 from India and Bangladesh each and so on. 

# <h2> Country wise report on Average Score. </h2>

# In[ ]:


plt.figure(figsize=(20,8))
sns.boxplot(y='Avg',x='Country',data=test_cricket)
plt.show()


# # conclusion
# 1. minimum average score for INDIA is 0
# 2. INDIA's average score is in between 0 to 62 (if we exclude two player of avg score of 66 and 119)
# 3. maximum avg score is around 120 for INDIA
# 4. INDIA's avg score of top 25% player is in between 62-50, next 50% players in between 30-10 and last 25% players has 0-10
# 5. INDIA's Avg score madian is 19, we can calculate this data for rest of the countries
# 
# 
# 

# <h2> Relation between Runs and innings</h2>

# In[ ]:


plt.figure(figsize=(10,8))
sns.jointplot(x='Inn',y='Runs',data=test_cricket,color='red',kind='reg')
plt.show()


# <h2> What is the contribution of most of the countries in 50s made at yet? </h2>

# In[ ]:


# creating a dataframe - sum of 50s after grouping test_cricket by country
country_50 = test_cricket.groupby('Country')['50'].sum()

# mearging all country in other variable whoes 50s are less than 30
other = country_50[country_50 < 30].sum()

# removing countries, those 50s are less than 30
country_50 = country_50[country_50 >= 30]

# inserting other variable in dataframe
country_50['OTHER'] = other
country_50


# In[ ]:


plt.figure(figsize=(4,4))
plt.pie(x=country_50,labels=country_50.index,radius=3,autopct='%1.1f%%',colors=['tomato','slateblue','coral','yellowgreen','pink','skyblue','gray','brown','lightskyblue','violet','gold'])
plt.show()


# <h2> Test cricket's top most players who have 'Not Out' record? and they belong to which country? </h2>

# In[ ]:


plt.figure(figsize=(8,4))
sns.barplot(x='Player',y='NO',data=test_cricket.sort_values('NO',ascending=False).head(10),hue='Country',
            dodge=False,palette='Dark2')
plt.xticks(rotation=70)
plt.show()


# We can say that Jm Anderson from ENG is Not Out more than 80 times in his all Innings. In this list no player from India. In this list ENG has 2, WI has 3, SL has 1, NZ has 1, AUS has 2 and SA has 1 players respectively.

# <h2>Each country has how many players </h2>

# In[ ]:


player_count = test_cricket.Country.value_counts()
player_count


# 
# We found that one player played for both India and Eng. and Three players played for India and Pak. <h2>Who are they? </h2>

# In[ ]:


test_cricket[(test_cricket.Country == 'ENG/INDIA') | (test_cricket.Country == 'INDIA/PAK')]


# <h2>Comparision of Top 10 scorer of each country </h2>

# In[ ]:


# first we sort data by runs than grouped by country than use head for 10 players.
Top_10 = test_cricket.sort_values('Runs',ascending=False).groupby('Country').head(10)

# removing countries have less than 10 players
Top_10 = Top_10[Top_10.Country.map(Top_10.Country.value_counts() == 10)]

# giving the rank(in their country) to each player by Runs
Top_10['Rank'] = Top_10.groupby('Country').Runs.rank(ascending=False)

Top_10


# In[ ]:


# creating pivot table
Top_10_pivot = Top_10.pivot_table(index='Country',columns='Rank',values='Runs')
Top_10_pivot


# In[ ]:


plt.figure(figsize=(15,15))
sns.heatmap(Top_10_pivot,linewidths=0.1,annot=True,fmt='.1f',cmap='Spectral')
plt.show()


# white column show rank of 6.5 and blank at 6 and 7 position which means IRE has two player with same 64 Runs that's why they both share 6th and 7th rank and get rank 6.5

# In[ ]:


Top_10[Top_10.Country == 'IRE']


# <h2> India's Runs Distribution </h2>

# In[ ]:


plt.figure(figsize=(12,8))
sns.violinplot(y='Runs',data=test_cricket[test_cricket.Country == 'INDIA'],inner="quartile",color='lightblue')
plt.ylabel('Runs')
plt.xlabel('INDIA')
plt.title('Runs distribution of INDIA')
plt.show()


# In[ ]:




