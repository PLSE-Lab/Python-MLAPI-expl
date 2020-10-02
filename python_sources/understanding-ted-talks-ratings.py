#!/usr/bin/env python
# coding: utf-8

# ## Introduction

# This notebook intends to explore the TED Talks ratings. An exploratory analysis will be performed focusing on the ratings data.  They will be related to the other variables, such as views, comments, duration, languages, tags and speaker occupation.  
# The objective is to understand where the rating comes from, if the other variables could have an influence on it .  
# The main ratings given to each talk will also be explored.

# ## Dataset

# The file ted_main.csv from TED Talks dataset will be used in this analysis.
# 
# In this section, the data will be loaded, cleaned and prepared to the analysis.

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from collections import defaultdict
import datetime
from pandas.plotting import parallel_coordinates
from scipy.stats import pearsonr
import re

get_ipython().run_line_magic('matplotlib', 'inline')


# Reading and exploring the data.

# In[ ]:


ted_data = pd.read_csv('../input/ted_main.csv')
ted_data.keys()


# In[ ]:


ted_data.head()


# The variable duration will be converted to minutes and the dates will be transformed in strings.

# In[ ]:


def transform_date(date):
    date_info = datetime.date.fromtimestamp(date)
    return date_info
    
ted_data['film_date'] = ted_data['film_date'].apply(transform_date)
ted_data['published_date'] = ted_data['published_date'].apply(transform_date)

ted_data['duration'] = ted_data['duration']/60

ted_data.head()


# Checking column data types.

# In[ ]:


ted_data.dtypes


# Checking for missing values:

# In[ ]:


pd.isnull(ted_data).sum()


# Only the variable speaker_occupation contains missing values. Checking these values:

# In[ ]:


print ("First row with missing value")
for index, row in ted_data.iterrows():
    if pd.isnull(row['speaker_occupation']):
        print (row)
        break


# The rows with speaker_occupation NaN will be replaced by Unknown.

# In[ ]:


ted_data.fillna('Unknown', inplace = True)

pd.isnull(ted_data).sum()


# There are no more missing values for speaker occupation.
# 
# During the analyses it was noticed that the variable languages also have some missing values. The talks that had no information about languages showed the number 0.

# In[ ]:


print (ted_data['languages'][ted_data['languages'] == 0].count())
ted_data[ted_data['languages'] == 0].head()


# Some of these cases are dance or music performance but others are talks in one language.  They all will be considered with one idiom.

# In[ ]:


ted_data['languages'] = ted_data['languages'].replace(0, 1)


# Different tables will be created that associate the TED talk with rating, tags, speakers occupation and with the basic information. 
# 
# A new column with the talk ID will be included in each table.

# In[ ]:


ted_data['Talk_ID'] = range(1, len(ted_data)+1)


# ### Rating Data
# 
# First, a table with the rating information will be created.  
# A list with all rating names is shown below.

# In[ ]:


rating_names = set()
for index, row in ted_data.iterrows():
    rating = ast.literal_eval(row['ratings'])
    for item in rating:
        rating_names.add(item['name'])
    
print (rating_names)


# In[ ]:


rating_data = defaultdict(list)
for index, row in ted_data.iterrows():
    rating = ast.literal_eval(row['ratings'])
    rating_data['Talk_ID'].append(row['Talk_ID'])
    names = set()
    for item in rating:
        rating_data[item['name']].append(item['count'])
        names.add(item['name'])

rating_data = pd.DataFrame(rating_data)

rating_data.head()


# The dataframe rating_data contains only the Talk ID and the ratings.  
# The values in each column represent the number of votes the talk received for each rating.
# 
# One column with the total of votes for each talk will be added to the dataframe rating_data. And the data will be ordered according to this variable.

# In[ ]:


rating_data['total'] = rating_data.sum(axis = 1)
rating_data = rating_data.sort_values('total', ascending=False)  


# All the other columns will be modified to represent a percentage of the total of votes the talk received.  
# The total of votes is an interesting metric, but this modification will allow a comparison of the ratings between talks.

# In[ ]:


def column_percentage(column):
    return (column/rating_data['total'])*100

rating_data.loc[:, (rating_data.columns != 'total') &  (rating_data.columns !='Talk_ID')] =     rating_data.loc[:, (rating_data.columns != 'total') &  (rating_data.columns !='Talk_ID')].apply(column_percentage)

print (rating_data.head())


# ### Tags Data
# 
# Next, the tags will be separated in each line and associated with the talk ID in a new dataframe.

# In[ ]:


tags_data = defaultdict(list)
for index, row in ted_data.iterrows():
    themes = ast.literal_eval(row['tags'])
    for item in themes:
        tags_data['Talk_ID'].append(row['Talk_ID'])
        tags_data['tags'].append(item)

tags_data = pd.DataFrame(tags_data)

print (len(tags_data))
print (len(tags_data['tags'].unique()))
tags_data.head()


# The total number of tags used is 19154. However there are only 416 unique tags.

# ### Speakers Occupation Data
# 
# Before building the table for speakers occupation, a search for problematic characters in this field will be performed.

# In[ ]:


problemchars = re.compile(r'[=\+/&<>;\"\-\?%#$@\,\t\r\n]| and ')

problems_occupation = defaultdict(list)
for index, row in ted_data.iterrows():
    occupation = row['speaker_occupation']
    char = problemchars.search(occupation)
    if char:
        chars = char.group()
        problems_occupation[chars].append(occupation)
        
problems_occupation.keys()


# Checking one case:

# In[ ]:


problems_occupation['/']


# During this search it was noticed that the characters / , ; + - and the word "and" were used to separate more than one occupation.  
# In one case the occupation included "..." in the end and that will be removed.  
# 
# The occupation of the speakers will be split.  
# The cases of "HIV/AIDS fighter" and "9/11 mothers" were found using the regular expression, but the strings must not be split. That would introduce an error.  
# In some cases "," indicated a role in a company.  When these strings are split, the role and the name of the company will appear as two different occupations.  

# In[ ]:


mult_occupation = re.compile(r'\/|\,|\;|\+| and ')
end_issue = re.compile(r' \.\.\.')
occupation_data = defaultdict(list)
ignore_cases_list = ['HIV/AIDS fighter','9/11 mothers']

for index, row in ted_data.iterrows():
    occupation = row['speaker_occupation']
    problem_found = False
    if mult_occupation.search(occupation):
        problem_found = True
    if problem_found & (occupation not in ignore_cases_list):
        occupation = re.split('\/|\,|\;|\+| and ', occupation)
        for item in occupation:
            occupation_data['Talk_ID'].append(row['Talk_ID'])
            if end_issue.search(item):
                item = item.strip(' ...')
            occupation_data['speaker_occupation'].append(item.strip().lower())
    #All strings were converted to lowercase in order to avoid the same word in different formats.
    else:
        occupation_data['Talk_ID'].append(row['Talk_ID'])
        occupation_data['speaker_occupation'].append(occupation.lower())

occupation_data = pd.DataFrame(occupation_data)


# In[ ]:


print (occupation_data['speaker_occupation'].value_counts().head())
print ((len(occupation_data)))
print (len(occupation_data['speaker_occupation'].unique()))
occupation_data.head()


# There are 3203 speakers occupations while 1293 are unique. This high number shows the occupation is very diversified.  
# The most common occupations are writer, author, activist, artist and entrepreneur.

# ### Basic TED Talks Data
# 
# Next, the table with the talks basic information will be created.

# In[ ]:


ted_basic_info = ted_data[['Talk_ID', 'title', 'duration','comments','views','languages']]


# Four dataframes were created with different information to be used in the analysis: ted_basic_info, tags_data, occupation_data and rating_data. 

# ## Data Analysis

# In this section the talks ratings will be explored and associated with the other variables.

# ### Talks Ratings
# 
# The ratings most used in general. The next plot shows the number of talks that received at least one vote for each rating.

# In[ ]:


count_talks = defaultdict(list)
for rating in rating_data.columns:
    if (rating != 'Talk_ID') & (rating != 'total'):
        count_talks['rating'].append(rating) 
        count_talks['count'].append(rating_data[rating_data[rating] >0][rating].count())
    


# In[ ]:


sns.barplot(x="rating", y="count", data=count_talks)
plt.ylim(2400, 2600)
plt.xticks(rotation='vertical')


# The total number of talks is 2550. Some ratings seem to be used in all (or almost all) the talks, such as Inspiring, Ingenious and Fascinating. The positive ratings are very frequently used, while the negative ones (Obnoxious, Longwinded, Confusing and Unconvincing) are less present in the Talks.  
# In general all the ratings, positives and negatives, were voted in most of the talks. That shows how peoples opinion may diverge a lot.

# ### The Main Rating
# 
# One additional feature will be added to the dataframe: the main rating. This variable will indicate the rating with most votes for each talk. This is going to be the main characteristic used to compare the rating with the other features.

# In[ ]:


rating_values = rating_data.loc[:, (rating_data.columns != 'total') &  (rating_data.columns !='Talk_ID')] 
rating_data['main_rating'] = rating_values.apply(np.argmax, axis = 1)

rating_data['main_rating'].head()


# The following plot will show how many talks have each main rating.

# In[ ]:


label_order = rating_data['main_rating'].value_counts().sort_values(ascending = False).index

f, ax = plt.subplots(figsize=(12, 8))
ax = sns.countplot(x = 'main_rating', data = rating_data, order = label_order)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:d}'.format(height),
            ha="center") 
plt.xticks(rotation=30)


# Most of the talks are classified by the viewers as Inspiring, followed by Informative.

# The following parallel coordinates chart shows all the votes (in percentage) each TED Talk received for each rating.  
# The lines are colored according to the main rating.  

# In[ ]:


rating_values_updated = rating_data.loc[:, (rating_data.columns != 'total') &  (rating_data.columns !='Talk_ID')]
plt.figure(figsize=(20,15))
parallel_coordinates(rating_values_updated, 'main_rating', colormap=plt.get_cmap("tab20"))
plt.xlabel("Rating")
plt.ylabel("Percentage of Votes")


# In this plot it is possible to notice a higher density of lines for the ratings most used.  
# Higher vote values indicate that the viewers agree most with that main rating for the Talk.  
# 
# Main ratings such as Beautiful, Funny and Jaw-dropping don't have as many talks as Inspiring or Informative, but their talks have a very high percentage of votes for the main rating which means people agree most with the classification.
# 
# Some talks present high values for a rating that is not the main one. For example, several inspiring talks present a high percentage of votes also for Courageous and Beautiful. And several Informative talks present high values for Fascinating and Persuasive. These talks classifications seem not to be a consensus among the viewers or they could be classified with two main ratings. This analysis will consider only the most voted rating as the main rating.
# 
# The negative ratings were voted as the main rating for only a few talks. That shows most talks have great acceptance by the public.  
# It is possible to observe in the plot that one talk was classified as very Obnoxious and some as very Unconvincing
# 
# The following analyses will associate the ratings with the other variables in the dataset.

# ### The total of votes
# Checking the range of the total number of votes.

# In[ ]:


print (rating_data['total'].head(1))
print (rating_data['total'].tail(1))


# The talk with fewer votes has 411.  
# The talk with most votes has 93851.
# 
# Next, a comparison will be made between the total of votes and the number of views and comments.  
# First, this dataframe will be joined with the basic data.

# In[ ]:


rating_and_basic = ted_basic_info.merge(rating_data, how = 'left', on = ['Talk_ID'])


# In[ ]:


sns.regplot(x="views", y="total", data=rating_and_basic)
plt.xlabel("views")
plt.ylabel("Total Votes")


# In[ ]:


sns.regplot(x="comments", y="total", data=rating_and_basic)
plt.xlabel("comments")
plt.ylabel("Total Votes")


# The total of votes and the number of views seems to be highly correlated. That makes sense, since people should watch the TED Talk before rating it. Comments seems to be also correlated with votes, but not as much as views. Probably this correlation results from the fact that both comments and the total of votes are correlated with the number of views.  
# It is possible to notice some outliers in the data with a much higher number of views and votes than most of the other talks. They will be checked next.

# In[ ]:


ted_basic_info.sort_values('views', ascending = False).head()


# There is nothing incorrect about these data. They seem to be interesting talks to be analyzed since they draw more attention from the viewers than most of the talks.  
# The talks with most views are also the ones with most votes, as the plot showed.

# In[ ]:


ted_basic_info.sort_values('comments', ascending = False).head()


# The talks with most comments are not the same ones with the highest number of views (except for one). However they are not necessarily the ones with most votes, as it is possible to notice in the plot of comments vs. total votes.

# ### Correlation between main rating and other variables
# 
# In this section some plots will be built with the main rating and the other talks features in order to explore if there is some relation between them.

# In[ ]:


list_ordered_by_median = rating_and_basic.groupby('main_rating')['views'].median().sort_values(ascending = False).index

f, ax = plt.subplots(figsize=(12, 8))
ax = sns.boxplot(x="main_rating", y="views", data=rating_and_basic, order = list_ordered_by_median)
plt.xticks(rotation=30)
plt.ylim((0,10000000))


# This plot shows an interesting result. Despite being fewer talks, the jaw-dropping ones have in general more views than the others. In second place comes the Funny talks, which are also a low number.  
# The ones with a negative rating are the least viewed, except for the Unconving ones.

# In[ ]:


list_ordered_by_median_com = rating_and_basic.groupby('main_rating')['comments'].median().sort_values(ascending = False).index

f, ax = plt.subplots(figsize=(12, 8))
ax = sns.boxplot(x="main_rating", y="comments", data=rating_and_basic, order = list_ordered_by_median_com)
plt.xticks(rotation=30)
plt.ylim((0,1000))


# The previous plot shows that the most commented talks are the most Persuasive.  
# The Jaw-dropping talks also have a higher median than most of the other ratings.

# In[ ]:


list_ordered_by_median_lan = rating_and_basic.groupby('main_rating')['languages'].median().sort_values(ascending = False).index

f, ax = plt.subplots(figsize=(12, 8))
ax = sns.boxplot(x="main_rating", y="languages", data=rating_and_basic, order = list_ordered_by_median_lan)
plt.xticks(rotation=30)


# The talks classified as Longwinded and Confusing were the least translated, least viewed and least commented.  
# The most translated talks are the OK ones. The others have a very similar median.

# In[ ]:


list_ordered_by_median_dur = rating_and_basic.groupby('main_rating')['duration'].median().sort_values(ascending = False).index

f, ax = plt.subplots(figsize=(12, 8))
ax = sns.boxplot(x="main_rating", y="duration", data=rating_and_basic, order = list_ordered_by_median_dur)
plt.xticks(rotation=30)
plt.ylim((0,40))


# The talks classified as Longwinded are in general longer than the others. And the ones OK have a shorter duration. That may explain why the OK talks are more translated than the others.
# 
# As the data have many outliers, the median was used to compare the groups in the analysis of the last plots.

# Next a heatmap of the correlation between features will be built and it will show which ratings and variables are correlated.

# In[ ]:


#Features to remove from the plot:
features_to_remove = ['Talk_ID', 'title']
corr = rating_and_basic.drop(features_to_remove, axis = 1).corr() 
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('Heatmap of Correlation between Features')


# According to the heatmap, there are two strong negative correlations:  
# 1. duration and languages. That was already observed in the bloxplots analyses.  
# 2. Informative and Beautiful. That shows that the talks are voted for one or another of these ratings. They are not usually used together.
# 
# And there are many positive correlations. Some interesting examples:
# 1. Confusing, Unconvincing, OK, Longwinded and Obnoxious. All the negative ratings are usually voted together.
# 2. comments, views, languages and the total of votes. These features are positively correlated. This behavior already appeared in previous analyses, except for the variable languages.
# 3. Longwinded and duration. As expected, according to previous analyses.
# 4. Jaw-dropping, Fascinating and Ingenious. These ratings are different but it seems they are frequently used together by the viewers.
# 5. Persuasive and comments. It was expected from the boxplots analysis. This relation will be explored below.

# In[ ]:


plt.scatter(rating_and_basic['Persuasive'], rating_and_basic['comments'], alpha = 0.5)
plt.ylabel("comments")
plt.xlabel("Persuasive percentage")


# In[ ]:


pearsonr(rating_and_basic['Persuasive'], rating_and_basic['comments'])


# There is some correlation between the percentage of the votes for the rating Persuasive and the number of comments. However the correlation is not strong.

# ### Ratings, Tags and Speaker Occupation
# 
# Now it will be verified if there is some relation between the talk tags and the talk rating and between the speakers occupation and the talk rating.
# 
# New dataframes with these data will be created.

# In[ ]:


tags_rating = rating_data.merge(tags_data, how = 'left', on = ['Talk_ID'])
occupation_rating = rating_data.merge(occupation_data, how = 'left', on = ['Talk_ID'])


# In[ ]:


tags_rating.head()


# In[ ]:


occupation_rating.head()


# Checking which are the most common tags of the talks with the same main rating.

# In[ ]:


def find_common_var(rating, var):
    #finds the main tags or speakers occupation
    if var == 'tags':
        count_tags = tags_rating[tags_rating['main_rating'] == rating]['tags'].value_counts()
    else:
        count_tags = occupation_rating[occupation_rating['main_rating'] ==                                       rating]['speaker_occupation'].value_counts()
    return count_tags


# In[ ]:


def main_tags_or_occupation(search):
    #var should be tags or speakers_occupation
    #Find the five main tags/occupation for each main rating and save in a set
    main_var_set = set()
    for name in rating_names:
        main_var_set.update(find_common_var(name, search).head().index)

    #Create a dataframe with the tags/occupation and values for each main rating
    rating_main_var = defaultdict(list)
    for name in rating_names:
        current_rating_table = find_common_var(name, search).head()
        main_var = current_rating_table.index
        rating_main_var['rating'].append(name)
        for var in main_var_set:
            if var not in main_var:
                rating_main_var[var].append(0)
            else:
                rating_main_var[var].append(current_rating_table[var])
    return rating_main_var

rating_main_tags = main_tags_or_occupation('tags')
rating_main_tags = pd.DataFrame(rating_main_tags)  


# In[ ]:


#defining the rating column as the new dataframe index:
rating_main_tags = rating_main_tags.set_index('rating')
rating_main_tags.head()


# A heatmap will be created with the main tags information.

# In[ ]:


f, ax = plt.subplots(figsize=(15, 12))
cmap = sns.cubehelix_palette(light=1, as_cmap=True)
ax = sns.heatmap(rating_main_tags, annot = True, fmt = "d", cmap = cmap)


# Only the five most common tags for the talks of each main rating are shown in the heatmap.  
# The value zero doesn't mean that the tag was never used with talks that have this main rating, but it indicates that the tag is not one of the five most used for those talks.  
# 
# Some tags are common for different main ratings, such as technology, global issues and TEDx. These are very common tags in general for all TED talks. The rating was probably not influenced by them.   
# The most common tags for all talks:

# In[ ]:


tags_data['tags'].value_counts().head(10)


# Some of the ratings seem to be associated to different tags comparing to the most common ones and different from most of the other ratings. Some examples:  
# 1. Beautiful is associated with the tags art, performance and music.  
# 2. Funny with comedy, humor and entertainment.  
# 3. Ingenious with science, invention and technology.  
# 4. Courageous with activism, social change and society.
# 
# In these cases it seems the talk theme and the rating are related.
# 
# The same graph and analysis will be performed with the speakers occupation.

# In[ ]:


rating_main_occup = main_tags_or_occupation('speakers_occupation')
rating_main_occup = pd.DataFrame(rating_main_occup)  

#defining the rating column as the new dataframe index:
rating_main_occup = rating_main_occup.set_index('rating')


# In[ ]:


f, ax = plt.subplots(figsize=(15, 12))
cmap = sns.cubehelix_palette(light=1, as_cmap=True)
ax = sns.heatmap(rating_main_occup, annot = True, fmt = "d", cmap = cmap)


# The first thing it is possible to notice is that the speakers occupation are not as much concentrated as the tags. The values in the heatmap (that represent the number of talks with that speaker occupation) are much lower than in the case of tags. As it was already noticed before, the speakers occupation is very diversified.
# 
# However there are some interesting associations in the graph. For example, the speakers of the talks with main rating Beautiful are usually artists. The Ingenious talks are mostly presented by inventors, engineers and architects. The Funny ones by comedians and the Courageous ones by activists, as expected according to the tags analysis. 
# 
# 

# ## Conclusion

# This report contains an analysis of the TED Talks dataset focused on the ratings.  
# Many of the results observed in the analyses were already expected. However it was possible to notice some interesting behaviors and association between the variables.
# 
# First only the ratings behavior was analyzed. It was noticed that most of the talks have votes for all the types of ratings. The parallel coordinates graph gave an overview of the ratings votes for all talks. It was clear that in some cases, the is a strong agreement between the users about the talk classification and the percentage of votes for one type of rating is very high. However in many cases one rating did not stand out and the public opinion is divided among more than one option. For the following analyses only the rating with the highest number of votes was considered the main rating for each talk.  
# Most of the talks were classified as Inspiring and Informative, while only a few talks have a negative classification as the main rating. That shows most talks have great acceptance by the public.  
# 
# The rating was then compared with the other variables. Boxplots and a correlation matrix were built.  
# The number of comments and views may reflect the rating. The analysis showed that the jaw-dropping and funny talks are the ones with more views, while the talks with a negative rating have a very small number of views. And the persuasive and jaw-dropping talks are the ones with more comments.  
# It was also found some correlation between the ratings votes and the other basic features. The correlations were explored only through the heatmap, further calculations were not made. This is one example of future analysis that could be done with these data.  
# 
# Lastly the five most common tags and speakers occupations of the talks of each main rating were represented in two heatmaps. It seems from this analysis that the talk theme may influence the rating. The same occurred in some cases for the speakers occupation, but in this case it is more likely that the occupation influenced the talk theme, which influenced the rating.
# 
# The objective of this work was to understand a little about the TED Talks ratings. They are interesting metrics to evaluate a TED Talk and this work explored the ratings results, how people evaluate them and what may influence these results.
# 
# There are more analyses that could be performed with these data. Many of the correlations mentioned in this work could be calculated and tested. And the percentage values of the ratings could be further explored and a model built based on them.

# ## References
# 
# The references used in this project are listed in this section:
# 
# https://stackoverflow.com/questions/20965046/cumulative-sum-and-percentage-on-column   https://stackoverflow.com/questions/19384532/how-to-count-number-of-rows-in-a-group-in-pandas-group-by-object   https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.merge.html   https://pandas.pydata.org/pandas-docs/stable/merging.html  
# https://pandas.pydata.org/pandas-docs/stable/cookbook.html#cookbook-merge  
# https://plot.ly/python/heatmaps/  
# https://pandas.pydata.org/pandas-docs/stable/reshaping.html  
# https://stackoverflow.com/questions/39132742/groupby-value-counts-on-the-dataframe-pandas   https://github.com/mwaskom/seaborn/issues/202  
# https://stackoverflow.com/questions/26163702/how-to-change-figuresize-using-seaborn-factorplot  
# https://seaborn.pydata.org/examples/many_pairwise_correlations.html  
# https://seaborn.pydata.org/generated/seaborn.heatmap.html  
