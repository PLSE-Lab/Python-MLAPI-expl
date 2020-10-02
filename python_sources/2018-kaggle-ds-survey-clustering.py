#!/usr/bin/env python
# coding: utf-8

# ## Views on Data Science Expertise
# ## How Different Roles Evaluated Academic Achievements vs. Independent Projects
# 
# 
# Digging into Kaggle's 2018 MS and DS Survey, this document describes the process of creating a cluster map to evaluate how different professions answered the question: *'Which better demonstrates expertise in data science: academic achievements or independent projects?*
# 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


#Opening the file.
multipleChoice_df = pd.read_csv('../input/multipleChoiceResponses.csv')


# The survey has 50 questions, so next we'll drop the unnecesary columns for the intended task. We want to work only on Question 6 (*'Select the title most similar to your current role (or most recent title if retired)'*) and Question 40 (*'Which better demonstrates expertise in data science: academic achievements or independent projects?*').

# In[ ]:


#Create a list of all columns. Remove columns that you don't want to delete. Drop only the columns in the list.
target_df = multipleChoice_df
target_col_list = list(target_df.columns.values)
target_col_list.remove('Q6')
target_col_list.remove('Q40')
for col in target_col_list:
    target_df.drop([col], axis=1,inplace=True)


# The following dataframe is created:

# In[ ]:


target_df.head()


# Cleaning the dataframe. The null values will be removed.

# In[ ]:


for column in target_df.columns:
    print('Null values in',str(column),'?',target_df[column].isnull().any())


# In[ ]:


clean_target_df = target_df.dropna()


# In[ ]:


clean_target_df.isnull().any()


# The answers of Q6 point to the current or most recent profession of the respondent. The clustering will be done by taking the professions listed in Q6 and linking each profession to the counts of the answers in Q40.  In other words, we'll try to distribute the professions into indexes and the multiple choice answers in Q40 into columns .
# 
# First we'll split the current dataframe into groups using the .gropby() method and at the same time using the .reset_index() method to create a dataframe with column named 'Count', as shown below:

# In[ ]:


group_target_df = clean_target_df.groupby(['Q6','Q40']).size().reset_index(name='Count')
group_target_df.head(10)


# As you can see, the 'Count' column counts the answers of Q40 for each profession. Next, we'll pivot this dataframe, that is, create an index with each individual profession and move the answers in Q40 to create columns out of them and the 'Count' values will fill the body of the dataframe:

# In[ ]:


pivot_target_df = group_target_df.pivot(index='Q6',columns='Q40',values='Count')
pivot_target_df


# It's the dataframe that we wanted, but as seen, the formulated questions of both Q6 and Q40 are occupying  row 18  and a column 7 respectively. We need to remove that row and that column.
# 
# We can see as well that new NaN values were created in this dataframe. We'll assume that in such cells the count for the answer was zero, so we'll modify accordingly.

# In[ ]:


pivot_target_df.drop(['Which better demonstrates expertise in data science: academic achievements or independent projects? - Your views:'], axis=1,inplace=True)
pivot_target_df.drop(pivot_target_df.index[18],inplace=True)


# Ok, so now lets replace the NaN values with zeros:

# In[ ]:


give_me_cluster_df = pivot_target_df.fillna(0)
give_me_cluster_df


# We now have the final dataframe to create the cluster map graph. Below are the lines of code:

# In[ ]:


#sns.clustermap creates the desired plot. fmt='g' adjusts for scientific notation of digits in cells.
#The second line calls the .ax_heatmap to give a slight rotation to the x axis labels.
graph = sns.clustermap(give_me_cluster_df,cmap='magma', annot=True, fmt='g')
plt.setp(graph.ax_heatmap.xaxis.get_majorticklabels(), rotation=85)
plt.show()


# Finally, this map shows us a clustering in favor of independent projects as a measurement of success among students and Data Scientists. The darker side of the map to the left shows less respondents giving a positive response to academic achivements. 
