#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Let's read in the files and display a HEAD

# In[ ]:


happy_df_2015 = pd.read_csv("../input/2015.csv")
happy_df_2016 = pd.read_csv("../input/2016.csv")
happy_df_2017 = pd.read_csv("../input/2017.csv")
happy_df_2015.head()


# ### Let's check wether the Data Frames ahve any null values or not!

# In[ ]:


print(happy_df_2015.isnull().sum())


# In[ ]:


print(happy_df_2016.isnull().sum())


# In[ ]:


print(happy_df_2017.isnull().sum())


# ### We see that it is one of the cleanest data set with which we are dealing!
# #### So let's start with the some basic plotting for each year and then we will go onto merge the data frames!

# In[ ]:


happy_df_2015.describe()


# In[ ]:


happy_df_2015.plot(kind = 'scatter',x = 'Happiness Rank' , y='Trust (Government Corruption)');
plt.title('Happiness rank based on the Corrupted Government!');


# In[ ]:


happy_df_2016.plot(kind = 'scatter',x = 'Happiness Rank' , y='Trust (Government Corruption)');
plt.title('Happiness rank based on the Corrupted Government!');


# In[ ]:


happy_df_2017.columns = happy_df_2017.columns.str.replace('.',' ')
happy_df_2017.head()


# In[ ]:


happy_df_2017.plot(kind = 'scatter',x = 'Happiness Rank' , y='Generosity');
plt.title('Happiness rank based on the Generosity!');


# ## We have looked at the scatter plots and say that: 
# 1. The more corrupted the government is...the lower the happiness values.
# 2. there is no such pattern for the generosity with the Happiness Rank.

# In[ ]:


plt.figure(figsize=(18,9))
sns.heatmap(happy_df_2015.corr(),annot=True);
plt.title("Correlation for 2015");
plt.show();


# ### The Highest Correlated Attributes are *Economy(GDP Per Capita)* and *Health(Life Expectancy)*** with the value of **0.82** in the Year of *2015*
# 
# ## Similarly let's see for 2016 and 2017.

# In[ ]:


plt.figure(figsize=(18,9))
sns.heatmap(happy_df_2016.corr(),annot=True);
plt.title("Correlation for 2016");
plt.show();


# In[ ]:


plt.figure(figsize=(18,9))
sns.heatmap(happy_df_2017.corr(),annot=True);
plt.title("Correlation for 2017");
plt.show();


# ## We looked at the Correlation for individual Data Frames, but in the year of 2016 and 2017, we have columns such as *Whisker High/Low, Upper/Lower Confirdence Level* and many more. 
# ### hence merging them we create a chaos. So we will take up either each data set one by one or just one.
# 
# # Let's start with 2015!
# 
# ### We tried to look at the patterns, but we cannot label the Countries as there are 158 uniques values for Countries!
# ### We have to get a way in order to display for all, or we can even define categories for such cases.
# 

# In[ ]:


happy_df_2015.head()


# In[ ]:


score_more_than_7_2015 = happy_df_2015[happy_df_2015['Happiness Score'] > 7]
score_more_than_7_2016 = happy_df_2016[happy_df_2016['Happiness Score'] > 7]
score_more_than_7_2017 = happy_df_2017[happy_df_2017['Happiness Score'] > 7]
plt.figure(figsize=(15,25))
plt.subplot(221);
score_more_than_7_2015.plot(kind='barh',x = 'Country',y = 'Happiness Score',ax=plt.gca(),legend=None);
#plt.xticks(rotation=90);
plt.xlabel('Happiness Score');
plt.ylabel('Countries Respectively');
plt.title('Countries by Happiness Score! for year 2015');
plt.subplot(222);
score_more_than_7_2016.plot(kind='barh',x = 'Country',y = 'Happiness Score',ax=plt.gca(),legend=None);
#plt.xticks(rotation=90);
plt.ylabel('Happiness Score');
plt.xlabel('Countries Respectively');
plt.title('Countries by Happiness Score! for year 2016');
plt.subplot(223);
score_more_than_7_2017.plot(kind='barh',x = 'Country',y = 'Happiness Score',ax=plt.gca(),legend=None);
#plt.xticks(rotation=90);
plt.ylabel('Happiness Score');
plt.xlabel('Countries Respectively');
plt.title('Countries by Happiness Score! for year 2017');
plt.show();


# # Switerland/Denmark/Norway have been most happiest Countries with Score more than 7.5
# ## Countries such as United States/Austria/Finland have improved their Happiness Score.
# ## We looked at the countires also with Higher *Happiness Score*. Next what we can do is we can look at different parameters and then look for Countires with Happiness Score based on those parameters. This will help us to visualize as which parameters have majorly performed in contributing towards the overall Happiness Score!

# In[ ]:


# Since we have a lot of features to display, we'll slice respectively
we_need_attrs = ['Country','Happiness Score','Economy (GDP per Capita)','Family']
we_need_attrs_2017 = ['Country','Happiness Score','Economy  GDP per Capita ','Family']
compare_main_features_2015 = score_more_than_7_2015.loc[:,we_need_attrs]
compare_main_features_2016 = score_more_than_7_2016.loc[:,we_need_attrs]
compare_main_features_2017 = score_more_than_7_2017.loc[:,we_need_attrs_2017]
compare_main_features_2017
## Now let's plot this variation to see how much each contributes!!.


# In[ ]:


plt.figure(figsize=(15,25));
plt.subplot(3,1,1);
compare_main_features_2015.plot.barh(stacked=True,x = 'Country',ax=plt.gca());
plt.subplot(3,1,2);
compare_main_features_2016.plot.barh(stacked=True,x = 'Country',ax=plt.gca());
plt.subplot(3,1,3);
compare_main_features_2017.plot.barh(stacked=True,x = 'Country',ax=plt.gca());
plt.show();


# ## Now we see that for some countries *Economy* has played an important role, specially for *Norway* and *Switzerland!*
# 
# # Next we can do is plot a Histogram/bar Chart on a Sorted value of GDP to see as acutally which country has performed best to contribute to happiness Index! It is coparatively easy but still!

# In[ ]:


compare_main_features_2015.sort_values(by='Economy (GDP per Capita)',inplace=True)
compare_main_features_2016.sort_values(by='Economy (GDP per Capita)',inplace=True)
compare_main_features_2017.sort_values(by='Economy  GDP per Capita ',inplace=True)


# In[ ]:


plt.figure(figsize=(15,29))
plt.subplot(3,1,1)
compare_main_features_2015.plot(kind='bar',x = 'Country',ax=plt.gca());
plt.title('Visualizing the GDP PER CAPITA for each Country to see the highest contributor and their Happiness Score!')
plt.xticks(rotation=60)
plt.subplot(3,1,2)
compare_main_features_2016.plot(kind='bar',x = 'Country',ax=plt.gca());
plt.xticks(rotation=60)
plt.subplot(3,1,3)
compare_main_features_2017.plot(kind='bar',x = 'Country',ax=plt.gca());
plt.xticks(rotation=60)
plt.show()


# # So we can backup our hypothesis that *GDP* played a part in contributing towards the Happiness Score, as looking on Norway!: *Norway had the Highest GDP PER CAPITA for duration 2015-17* and stayed in top 3 and bagged the *1st Position* in 2017.
# 
# ## Similarly we can do for *Family* also!

# In[ ]:


compare_main_features_2015['mean score'] = compare_main_features_2015['Happiness Score'].mean()
compare_main_features_2016['mean score'] = compare_main_features_2016['Happiness Score'].mean()
compare_main_features_2017['mean score'] = compare_main_features_2017['Happiness Score'].mean()
plt.figure(figsize=(20,15))
plt.subplot(2,2,1)
compare_main_features_2015[['Happiness Score','mean score']].plot(kind = 'bar',ax=plt.gca());
#plt.ylabel('Happiness Score vs. mean Happiness Score')
# Since we are not getting Countis as our X- Axis labels, let's define a condition for that
condition = ['Country','Happiness Score','mean score']
conditon_df_2015 = compare_main_features_2015[condition]
conditon_df_2015.sort_values(by='Happiness Score',inplace=True);
plt.ylabel('Happiness Score vs. mean Happiness Score')
plt.xticks(rotation=60)
plt.subplot(2,2,2)
conditon_df_2015.plot(kind='bar',x='Country',ax=plt.gca());
conditon_df_2016 = compare_main_features_2016[condition]
conditon_df_2016.sort_values(by='Happiness Score',inplace=True);
plt.ylabel('Happiness Score vs. mean Happiness Score')
plt.xticks(rotation=60)
plt.subplot(2,2,3)
conditon_df_2016.plot(kind='bar',x='Country',ax=plt.gca());
conditon_df_2017 = compare_main_features_2017[condition]
conditon_df_2017.sort_values(by='Happiness Score',inplace=True);
plt.ylabel('Happiness Score vs. mean Happiness Score')
plt.xticks(rotation=60)
plt.subplot(2,2,4)
conditon_df_2017.plot(kind='bar',x='Country',ax=plt.gca());
plt.ylabel('Happiness Score vs. mean Happiness Score')
plt.xticks(rotation=60)
plt.show()


# # We did a lot and a lot can be done!
# ## Contribute if you can think of something extraordinary, Till then Upvote if you like my Work!
# 
# ## Thanks for reading and Take care!!

# ## Hey hey....No need to go anywhere...I've got 3 more things to do!
# 1. Find the Regions impact on the Happiness Score.
# 2. Find the rank of the Country based on Sorted Economy...and deduce an inference.
# 3. Little bit exploratory purpose.

# In[ ]:


# we will merge 2015 and 2016 Data Frames, as we are focussing on the Regions
merged_df = pd.concat([happy_df_2015,happy_df_2016])


# In[ ]:


merged_df.head()


# In[ ]:


merged_df.info()


# In[ ]:


# We will drop the columns Upper Confidence Interval, Standard Error, Lower Confidence Interval as they are not present in the 2015 data frame.
merged_df = merged_df.drop(['Upper Confidence Interval','Standard Error','Lower Confidence Interval'],1)
merged_df.info()


# In[ ]:


# Awesome, now let's see how many Regions are present, i.e. unique values
merged_df.Region.nunique()
print(merged_df.Region.value_counts())


# In[ ]:


# Ok, so we are setting the cutoff of 40, as the entries more than 40, tends to be in somewhat Happy Countries
# i.e. Western Europe,Latin America and Carribbean, Central and Eastern Europe and finally Sub-Saharna Africa.
# We are trying to find out the countries from Regions and there Happiness Rank!
western_europe = merged_df.groupby('Region').get_group('Western Europe')
latin_america_and_caribbean = merged_df.groupby('Region').get_group('Latin America and Caribbean')
central_eastern_europe = merged_df.groupby('Region').get_group('Central and Eastern Europe')
sub_saharan_africa = merged_df.groupby('Region').get_group('Sub-Saharan Africa')


# In[ ]:


dfs = [western_europe,latin_america_and_caribbean,central_eastern_europe,sub_saharan_africa]
for i in dfs:
    print(i.info())


# In[ ]:


western_europe = western_europe.sort_values(by='Happiness Score',ascending = False)
latin_america_and_caribbean = latin_america_and_caribbean.sort_values(by='Happiness Score',ascending = False)
central_eastern_europe = central_eastern_europe.sort_values(by='Happiness Score',ascending = False)
sub_saharan_africa = sub_saharan_africa.sort_values(by='Happiness Score',ascending = False)
western_europe.head()


# In[ ]:


plt.figure(figsize=(20,15))
plt.subplots_adjust(wspace=0.4)
plt.tick_params(labelsize=20)
plt.subplot(2,2,1)
plt.title('Happiness Score for Western Europe Region')
sns.barplot(x = 'Happiness Score',y='Country',data = western_europe);
plt.subplot(2,2,2)
plt.title('Happiness Score for Latin America and Caribbean Region')
sns.barplot(x = 'Happiness Score',y='Country',data = latin_america_and_caribbean);
plt.subplot(2,2,3)
plt.title('Happiness Score for Central Europe Region')
sns.barplot(x = 'Happiness Score',y='Country',data = central_eastern_europe);
plt.subplot(2,2,4)
plt.title('Happiness Score for Sub Saharan Region')
sns.barplot(x = 'Happiness Score',y='Country',data = sub_saharan_africa);


# ## So the top 3 Countries from all the Regions are:

# In[ ]:


for i in dfs:
    print(i[:3][['Country','Region']])
    print('--------------------------')


# ## Top 3 Countries from each regions are:
# 1. Western Europe:
#     a. Swtizerland
#     b. Iceland
#     c. Denmark
# 2. Latin America and Caribbean:
#     a. Costa Rica
#     b. Mexico
#     c. Brazil
# 3. Central and Eatern Europe
#     a.Czech Republic
#     b. Uzbekistan
#     c. Slovakia
# 4. Sub Saharan Africa:
#     a. Mauritius
#     b. Nigeria
#     c. Zambia
# 
# ## But wait a second.....Happiest Country is **Norway**, we need to find the region for this Country!
# ### Let's do that!

# In[ ]:


norway = merged_df[merged_df.Country == 'Norway']
norway


# In[ ]:


# No need to panic...we got 2 values as we merged 2 years data and data differed...that's why Norway became happiest country in 2017.


# ## Next we want to sort the data frame based on Economy and then see which country with highest Economy lies where!

# In[ ]:


sorted_df = merged_df.sort_values(by = 'Economy (GDP per Capita)',ascending = False)
sorted_df


# In[ ]:


#sorted_df = sorted_df.set_index('Country').head()


# In[ ]:


# Now let's find their ranks!
print("Rank of Countries with Highest GDP's")
print(sorted_df[:5][['Happiness Rank','Economy (GDP per Capita)','Country']])


# In[ ]:


print("Rank of Countries with Lowest GDP's")
print(sorted_df.iloc[-5:][['Happiness Rank','Economy (GDP per Capita)','Country']])


# ## Awesome, we got the top 5 and lowest 5 Countries based on their GDP, so we can finally say that **You can't buy happiness with just Money!**.
# 
# ## So, we finded the impact of GDP and Region on the Happiness Rank!
# ## Merged Dataset respective to Region.
# 
# 

# In[ ]:


# Now let's see what else can we do, let's have alook at our data frame!
sorted_df.head()


# In[ ]:


duplicateRowsDF = sorted_df[sorted_df.duplicated(['Country'])]
duplicateRowsDF.head()


# In[ ]:


sum_df_df = duplicateRowsDF.groupby(['Dystopia Residual','Economy (GDP per Capita)','Family','Freedom','Generosity','Happiness Score',
                           'Health (Life Expectancy)','Trust (Government Corruption)']).sum()
sum_df_df.head()


# In[ ]:


country_and_rank_df = merged_df[['Happiness Rank','Country']]
mergeing_duplicated_sum = pd.merge(sum_df_df,country_and_rank_df,on = 'Happiness Rank',how='left')
mergeing_duplicated_sum.head(25)


# ## So we get to see that since we have separated data for 2015 and 2016, the same rank is attained by 2 Countries, hence we cannot predict as which country will achieve what rank as new Countries do get added every year.

# In[ ]:


# Let's plot the happiness scores for these countires and end our Kernel EDA!
plt.figure(figsize=(16,9))
top_15 = sorted_df.iloc[:15]
bottom_15 = sorted_df.iloc[-15:]
plt.subplot(1,2,1)
plt.title('Top 15 Countries based on Happiness Score!');
g1 = sns.barplot(x = 'Country' , y = 'Happiness Score',data = top_15);
# this snip tries to provide the count over the bar!
for p in g1.patches:
    g1.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
plt.xticks(rotation=80);
plt.subplot(1,2,2)
plt.title('Bottom 15 Countries based on Happiness Score!');
g2 = sns.barplot(x = 'Country' , y = 'Happiness Score',data = bottom_15);
# this snip tries to provide the count over the bar!
for p in g2.patches:
    g2.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
plt.xticks(rotation=80);


# ## Awesome...We'll end here...Till then Contribute if something comes to your shiny mind..Take Care!
# # Keep Kaggling!
