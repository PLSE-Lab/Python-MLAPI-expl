#!/usr/bin/env python
# coding: utf-8

# # Finding Clusters of Personality Types Based on OCEAN Model Online Study
# 
# Hello guys, this time we will try to examine the data collected from a study of Big 5 or OCEAN model online study. 
# You can find the link to the test here: https://openpsychometrics.org/tests/IPIP-BFFM/
# 
# In this notebook, we will try to
# 1. Find clusters of personality types of the participants of the study
# 2. Find the correlation between one trait and another
# 
# But before, what is Big Five / OCEAN model? It is a model of psychology which aims to classify a person's personality based on five major traits namely,
# * Openness (O) : Openness of the person to new things, experience and ideas 
# * Conscientiousness (C) : Dilligency and studiousness of the person
# * Extroversion (E) : The degree in which the person likes to meet people / new people 
# * Agreeableness (A) : The propensity of the person to agree to other's argument to avoid confrontation
# * Neuroticism (N) : Anxiety, restlessness of the person
# 
# If you are interested about the Big Five / OCEAN model, more details can be found in this link https://en.wikipedia.org/wiki/Big_Five_personality_traits
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


raw_data = pd.read_csv('/kaggle/input/big-five-personality-test/IPIP-FFM-data-8Nov2018/data-final.csv',sep='\t')


# In[ ]:


pd.options.display.max_columns = 200
print('First 5 values of raw_data')
raw_data.head()


# Above we can see raw data of the study that we are going to use in this notebook. 
# the columns EXT1 to OPN10 corresponds to the score chosen by the participants for the questions in the study
# 
# Here is the mapping for the values in EXT1 to OPN10 column:
# 
# * 1.0 corresponds to the answer of 'Strongly Disagree'
# * 2.0 corresponds to the answer of 'Disagree'
# * 3.0 corresponds to the answer of 'Neutral'
# * 4.0 corresponds to the answer of 'Agree'
# * 5.0 corresponds to the answer of 'Strongly Agree'
# 
# Columns ending with '_E' corresponds to the time it takes for the participant to asnwer the respective questions
# 
# For the purpose of our analysis, we will select only columns EXT to OPN as well as 'country' 

# In[ ]:


df = raw_data.copy()
df.drop(df.columns[108:], axis = 1, inplace = True)
df.dropna(inplace=True)
df.drop(df.columns[100:107],axis = 1, inplace = True)
df = df.loc[(df!=0).all(axis=1)] #remove entries with all 0 values on the questions
df.drop(df.columns[50:100],axis = 1, inplace = True)
df.drop(df[ df['country'] == 'NONE' ].index, inplace = True)
print('First 5 values in the data')
df.head()


# In[ ]:


# Groups and Questions
ext_questions = {'EXT1' : 'I am the life of the party',
                 'EXT2' : 'I dont talk a lot',
                 'EXT3' : 'I feel comfortable around people',
                 'EXT4' : 'I keep in the background',
                 'EXT5' : 'I start conversations',
                 'EXT6' : 'I have little to say',
                 'EXT7' : 'I talk to a lot of different people at parties',
                 'EXT8' : 'I dont like to draw attention to myself',
                 'EXT9' : 'I dont mind being the center of attention',
                 'EXT10': 'I am quiet around strangers'}

neu_questions = {'EST1' : 'I get stressed out easily',
                 'EST2' : 'I am relaxed most of the time',
                 'EST3' : 'I worry about things',
                 'EST4' : 'I seldom feel blue',
                 'EST5' : 'I am easily disturbed',
                 'EST6' : 'I get upset easily',
                 'EST7' : 'I change my mood a lot',
                 'EST8' : 'I have frequent mood swings',
                 'EST9' : 'I get irritated easily',
                 'EST10': 'I often feel blue'}

agr_questions = {'AGR1' : 'I feel little concern for others',
                 'AGR2' : 'I am interested in people',
                 'AGR3' : 'I insult people',
                 'AGR4' : 'I sympathize with others feelings',
                 'AGR5' : 'I am not interested in other peoples problems',
                 'AGR6' : 'I have a soft heart',
                 'AGR7' : 'I am not really interested in others',
                 'AGR8' : 'I take time out for others',
                 'AGR9' : 'I feel others emotions',
                 'AGR10': 'I make people feel at ease'}

csn_questions = {'CSN1' : 'I am always prepared',
                 'CSN2' : 'I leave my belongings around',
                 'CSN3' : 'I pay attention to details',
                 'CSN4' : 'I make a mess of things',
                 'CSN5' : 'I get chores done right away',
                 'CSN6' : 'I often forget to put things back in their proper place',
                 'CSN7' : 'I like order',
                 'CSN8' : 'I shirk my duties',
                 'CSN9' : 'I follow a schedule',
                 'CSN10' : 'I am exacting in my work'}

opn_questions = {'OPN1' : 'I have a rich vocabulary',
                 'OPN2' : 'I have difficulty understanding abstract ideas',
                 'OPN3' : 'I have a vivid imagination',
                 'OPN4' : 'I am not interested in abstract ideas',
                 'OPN5' : 'I have excellent ideas',
                 'OPN6' : 'I do not have a good imagination',
                 'OPN7' : 'I am quick to understand things',
                 'OPN8' : 'I use difficult words',
                 'OPN9' : 'I spend time reflecting on things',
                 'OPN10': 'I am full of ideas'}

# Group Names and Columns
Extroversion = [column for column in df if (column.startswith('EXT') and not(column.__contains__('_E')))]
Neuroticism = [column for column in df if (column.startswith('EST') and not(column.__contains__('_E')))]
Agreeableness = [column for column in df if (column.startswith('AGR') and not(column.__contains__('_E')))]
Conscientiousness = [column for column in df if (column.startswith('CSN') and not(column.__contains__('_E')))]
Openness = [column for column in df if (column.startswith('OPN') and not(column.__contains__('_E')))]


# # **Now let's visualize the Questions and Answers in the study.**
# 
# *The code for the visualization is inspired by Melih Akdag's notebook (https://www.kaggle.com/akdagmelih/five-personality-clusters-k-means).*

# In[ ]:


# Defining a function to visualize the questions and answers distribution
def vis_questions(groupname, questions, color):
    plt.figure(figsize=(40,60))
    for i in range(1, 11):
        plt.subplot(10,5,i)
        plt.hist(df[groupname[i-1]], bins=10, color= color, alpha=0.6)
        plt.title(str(questions[groupname[i-1]]), fontsize=18)
        plt.xticks([1,2,3,4,5])
        plt.xlabel('Strongly Disagree   -   Disagree   -   Neutral   -   Agree   -   Strongly Agree')
        plt.ylabel('Number of Respondents')
        plt.subplots_adjust(hspace = 0.4)


print('Q&As Related to Extroversion')
vis_questions(Extroversion, ext_questions, 'blue')


# In[ ]:


print('Q&As Related to Neuroticism')
vis_questions(Neuroticism, neu_questions, 'green')


# In[ ]:


print('Q&As Related to Agreeableness')
vis_questions(Agreeableness, agr_questions, 'red')


# In[ ]:


print('Q&As Related to Conscientiousness')
vis_questions(Conscientiousness, csn_questions, 'orange')


# In[ ]:


print('Q&As Related to Openness')
vis_questions(Openness, opn_questions, 'brown')


# In[ ]:


pd.options.display.max_rows = 250
print('Number of countries of the participants: ',len(df.country.value_counts()))
print(df.country.value_counts().head(20))
print('...')


# We can see that the participants come from 222 countries, with most of the participants coming from the US. 
# 
# For the purpose of our classification, we would only select the countries in which the number of participants is higher than 1000 for our sample.
# For each country we will be taking 1000 sample each.

# In[ ]:


change_scale =['EXT2','EXT4','EXT6','EXT10','EXT8','EST2','EST4','AGR1','AGR3','AGR5','AGR7','CSN2','CSN4','CSN6','CSN8','OPN2',               'OPN4','OPN6'] 
#we change the scale for those columns because for those questions, high score actually means the individual/participant is low on the trait that 
#the question is related to or vice versa, for example, for the question EST2 'I am relaxed most of the time', individuals who score high (4 or 5)
#on the question is actually showing traits of low neuroticsm

df_excl = df.groupby('country', as_index = False, group_keys = False).filter(lambda x: len(x) >= 1000) 
#remove countries where value count is less than 1000
sample = df_excl.groupby('country',as_index = False,group_keys=False).apply(lambda s: s.sample(1000,replace = True, random_state = 1))
#sample 1000 values from each country

sample[change_scale] = 6 - sample[change_scale]
sample.head()


# In[ ]:


sample_averaged = sample.groupby('country',as_index = True,group_keys=False).mean()
sample_stddev = sample.groupby('country',as_index = True,group_keys=False).std()

questions = {'EXT1' : 'I am the life of the party',
                 'EXT2' : 'I dont talk a lot',
                 'EXT3' : 'I feel comfortable around people',
                 'EXT4' : 'I keep in the background',
                 'EXT5' : 'I start conversations',
                 'EXT6' : 'I have little to say',
                 'EXT7' : 'I talk to a lot of different people at parties',
                 'EXT8' : 'I dont like to draw attention to myself',
                 'EXT9' : 'I dont mind being the center of attention',
                 'EXT10': 'I am quiet around strangers',
                 'EST1' : 'I get stressed out easily',
                 'EST2' : 'I am relaxed most of the time',
                 'EST3' : 'I worry about things',
                 'EST4' : 'I seldom feel blue',
                 'EST5' : 'I am easily disturbed',
                 'EST6' : 'I get upset easily',
                 'EST7' : 'I change my mood a lot',
                 'EST8' : 'I have frequent mood swings',
                 'EST9' : 'I get irritated easily',
                 'EST10': 'I often feel blue',
                 'AGR1' : 'I feel little concern for others',
                 'AGR2' : 'I am interested in people',
                 'AGR3' : 'I insult people',
                 'AGR4' : 'I sympathize with others feelings',
                 'AGR5' : 'I am not interested in other peoples problems',
                 'AGR6' : 'I have a soft heart',
                 'AGR7' : 'I am not really interested in others',
                 'AGR8' : 'I take time out for others',
                 'AGR9' : 'I feel others emotions',
                 'AGR10': 'I make people feel at ease',
                 'CSN1' : 'I am always prepared',
                 'CSN2' : 'I leave my belongings around',
                 'CSN3' : 'I pay attention to details',
                 'CSN4' : 'I make a mess of things',
                 'CSN5' : 'I get chores done right away',
                 'CSN6' : 'I often forget to put things back in their proper place',
                 'CSN7' : 'I like order',
                 'CSN8' : 'I shirk my duties',
                 'CSN9' : 'I follow a schedule',
                 'CSN10' : 'I am exacting in my work',
                 'OPN1' : 'I have a rich vocabulary',
                 'OPN2' : 'I have difficulty understanding abstract ideas',
                 'OPN3' : 'I have a vivid imagination',
                 'OPN4' : 'I am not interested in abstract ideas',
                 'OPN5' : 'I have excellent ideas',
                 'OPN6' : 'I do not have a good imagination',
                 'OPN7' : 'I am quick to understand things',
                 'OPN8' : 'I use difficult words',
                 'OPN9' : 'I spend time reflecting on things',
                 'OPN10': 'I am full of ideas'}

colors = ['blue', 'green', 'red', 'orange', 'brown']

plt.figure(figsize=(20,200))
for i in range(1,len(sample_averaged.columns)):
    plt.subplot(50,1,i)
    plt.plot(sample_averaged.index, sample_averaged[sample_averaged.columns[i]], color = colors[(i-1)//10],alpha = .7,             marker='s',linewidth=3, markersize=6)
    plt.errorbar(sample_averaged.index, sample_averaged[sample_averaged.columns[i]],sample_stddev[sample_stddev.columns[i]])
    plt.yticks([1,2,3,4,5])
    plt.xlabel('Country Code')
    plt.title(questions[sample_averaged.columns[i]], fontsize=18);
    plt.grid(axis='y',b=True, which='major', color='#666666', linestyle='-')
    plt.subplots_adjust(hspace = 0.4)
      
plt.savefig('QuestionsbyCountries.pdf')


# Above we can see the mean and standard deviation of the countries for the questions in the test. As we can see, for the purpose of this analysis, since the mean across countries for the answer to the questions are relatively similar, we would assume that countries would not be a determining variable to our clustering of the participants
# 
# # **Next, we would like to see the correlation between the questions in the study**

# In[ ]:


#we want to see the correlation between questions
questions_corr = pd.DataFrame.corr(sample)
print('Questions Mapping')
print('1 to 10 is regarding Extroversion')
print('11 to 20 is regarding Neuroticism')
print('21 to 30 is regarding Agreeableness')
print('31 to 40 is regarding Conscientiousness')
print('41 to 50 is regarding Openness')

   
plt.figure(figsize = (10,8.5))
plt.pcolor(questions_corr, cmap='plasma');
plt.grid(b=True, which='major', color='k', linestyle='-');
plt.colorbar();
plt.title('Questions Correlation Matrix', fontsize = 18)
plt.xlabel('Extroversion   -   Neuroticism   -   Agreeableness   -   Conscientiousness   -   Openness')
plt.ylabel('Extroversion   -   Neuroticism   -   Agreeableness   -   Conscientiousness   -   Openness')
plt.savefig('Question_correlation.pdf')


# **We can see that there is:**
# 
# A correlation of around 0.4 between Agreeableness and Extroversion (x axis of 21-30, y axis of 1-10), 
# 
# around -0.1 correlation between Conscientiousness and Neuroticism (x axis of 31-40, y axis of 11-20),
# 
# 0.1 correlation between Openness (x axis of 41 -50) and Extroversion (y axis of 1-10).
# 
# # **Now let's do the clustering using KNearest Neighbor clustering to find out the different types of the participants in this study**
# 
# But how many clusters are we expecting? let's find the optimum number of clusters

# In[ ]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
scaler = StandardScaler()
data = sample.drop('country', axis = 1)
data_scaled = scaler.fit_transform(data)

# fitting multiple k-means algorithms and storing the values in an empty list
SSE = []
for cluster in range(1,20):
    kmeans = KMeans(n_jobs = -1, n_clusters = cluster, init='k-means++')
    kmeans.fit(data_scaled)
    SSE.append(kmeans.inertia_)

# converting the results into a dataframe and plotting them
frame = pd.DataFrame({'Cluster':range(1,20), 'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Loss Function (Inertia)')


# You can think of the Loss Function as the accuracy of the classifier.
# 
# As we can see above, the optimal number of cluster is difficult to see since there is no distinct 'elbow' or point which signifies where the decrease of loss function become significantly lesser. However, we do see that after 9 cluster the change in Loss Function started to decrease less and became more linear, and 9 cluster corresponds to the 'bend' before it, therefore let's select the number of cluster to be 9

# In[ ]:


# k means using 9 clusters and k-means++ initialization
kmeans = KMeans(n_jobs = -1, n_clusters = 9, init='k-means++')
kmeans.fit(data_scaled)
pred = kmeans.predict(data_scaled)

data['Cluster'] = pred
data_clustered = data.groupby('Cluster').mean()

print('X-axis Mapping')
print('1 to 10 is the region of Extroversion')
print('11 to 20 is the region of Neuroticism')
print('21 to 30 is the region of Agreeableness')
print('31 to 40 is the region of Conscientiousness')
print('41 to 50 is the region of Openness')

plt.figure(figsize=(20,20))
for i in data_clustered.index:
    plt.subplot(len(data_clustered.index),1,i+1)
    plt.bar(range(1,51), data_clustered.iloc[i], color = colors[i%5],alpha = .6)
    plt.plot(range(1,51), data_clustered.iloc[i], color='black')
    plt.yticks([1,2,3,4,5])
    plt.title('Cluster ' + str(i+1), fontsize=18);
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.subplots_adjust(hspace = 0.4)
    
plt.savefig('Clusters.pdf')


# Above we can see the profile for the 9 types of participants we classify using the KNeighbor classifier, however, it is difficult to make out of the traits because of the numerous of questions (1-50 on the X-axis), hence let us sum those values above based on their questions grouping.
# 

# In[ ]:


col_list = list(data_clustered.columns)
ext = col_list[0:10]
est = col_list[10:20]
agr = col_list[20:30]
csn = col_list[30:40]
opn = col_list[40:50]

data_sums = pd.DataFrame()
data_sums['extroversion'] = data_clustered[ext].sum(axis=1)
data_sums['neuroticism'] = data_clustered[est].sum(axis=1)
data_sums['agreeableness'] = data_clustered[agr].sum(axis=1)
data_sums['conscientiousness'] = data_clustered[csn].sum(axis=1)
data_sums['openness'] = data_clustered[opn].sum(axis=1)
data_sums

# Visualizing the means for each cluster
plt.figure(figsize=(23,3.5))
for i in range(0, len(data_sums.index)):
    plt.subplot(1, len(data_sums.index),i+1)
    plt.bar(data_sums.columns, data_sums.iloc[i], color='green', alpha=0.2)
    plt.plot(data_sums.columns, data_sums.iloc[i], color='red')
    plt.title('Cluster ' + str(i+1))
    plt.xticks(rotation=45)
    plt.ylim(0,50);
plt.tight_layout()
    
plt.savefig('Cluster_traits.pdf');


# We can see above the Clusters of the participants of this study.
# 
# How to interpret the result?
# 
# You can think of the y-axis as the scale for the traits. With 50 being the highest, 10 being the lowest, and 30 as the mean.
# 
# For example: 
# 
# We can classify participants who belong in Cluster 7 as individuals who are low in Extroversion, average on Neuroticism, high on Agreeableness, Conscientiousness, and Openness.
# 

# In[ ]:




