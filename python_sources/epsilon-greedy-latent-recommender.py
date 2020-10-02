#!/usr/bin/env python
# coding: utf-8

# <h1><center>Epsilon-Greedy Latent Recommender</center></h1>
# 
# <center><a href="https://www.kaggle.com/hamzael1">Hamza El Bouatmani</a> on 14th April, 2019 </center>
# 

# *Last Update 23th April*: *Code Refactoring, Evaluation Section & more Documentation added*
# 
# ____

# # Introduction:
# <a href="https://www.careervillage.org/" target="_blank">CareerVillage.org</a> <span style="color: purple;">is a cloud-based solution for career advice</span>. It provides a platform where students with career-related questions meet professionals from the industry who help them by answering their questions.
# 
# The goal of <a href="https://www.kaggle.com/c/data-science-for-good-careervillage/overview" target="_blank">this competition</a>, is to develop a method to recommend relevant questions to the professionals who are most likely to answer them.
# 
# In this notebook, I propose a solution that addresses the problem in an efficient manner using a probabilistic approach (Epsilon-Greedy) combined with an *state-of-the-art technique (LSA)*. **This combination aims to balance between Exploration & Exploitation, targeting both the new and already-engaged professionals.**
# 
# The biggest strength that was noticed, is that **this solution behaves particularly well when encountering professionals with diverse interests. For example, when a professional follows a set of tags, and answers questions unrelated to those tags, the system still keeps recommending questions from both ends, adapts continuously to the interests of the professional along time and behaves in a resilient manner.**
# 
# Controlled randomness is inherent to the proposed approach, this has two advantages:
# * recommendations stay diverse.
# * unanswered new questions have a high chance to get answered because they get propritized. The model doesn't *over-focus* on the answered questions of the professionals.
# 
# Further, a basic framework for evaluating the system was proposed along with the most important metrics to measure. This helps fine-tune the model parameters locally before moving to production and gives and idea about the performance of the system.
# 
# This notebook is structured as follows:
# * First, we ask the question "[Why do we need a Recommender?](#why)" and answer it with some focused analytics.
# * Next, [techniques and concepts](#concepts) used in the proposed recommender are explained.
# * Then, we move to the actual [implementation of the proposed recommender system and explain its inner-workings](#implementation) after performing the necessary [proprocessings](#preproc).
# * We discuss the difficulty of evaluating Recommender Systems and [propose a very basic framework to evaluate the proposed system](#eval), along with the metrics that must be took into account.
# * Finally, some [advice and future suggestions for improving the system](#future) are listed, along with links to useful ressources.
# 

# # Why do we need a Recommender ? Let's ask the Data ! <a class="anchor" id="why"></a>

# <a href="https://www.kaggle.com/hamzael1/an-extensive-eda-for-careervillage" target="_blank">In a previous notebook</a> I made an overall Exploratory Data Analysis on the provided data. Here, I will be brief and focus on the most important statistics and metrics related to the recommendation problem.
# 
# *Note: some code snippets that are trivial are collapsed for better readability, feel free to expand them if you want to check the code*

# In[1]:


# Imports

import numpy as np  # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('max_colwidth', 200)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from plotly.offline import iplot, init_notebook_mode
init_notebook_mode(connected=True)

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import re
import string
import math
import random
from random import choice, choices
import time

import gc


from IPython.display import display

import warnings  
warnings.filterwarnings('ignore')

# Professionals Import

professionals = pd.read_csv('../input/professionals.csv', index_col='professionals_id')
professionals = professionals.rename(columns={'professionals_location': 'location', 'professionals_industry': 'industry', 'professionals_headline': 'headline', 'professionals_date_joined': 'date_joined'})
professionals['headline'] = professionals['headline'].fillna('')
professionals['industry'] = professionals['industry'].fillna('')

# Students Import

students = pd.read_csv('../input/students.csv', index_col='students_id')
students = students.rename(columns={'students_location': 'location', 'students_date_joined': 'date_joined'})

# Questions Import
questions = pd.read_csv('../input/questions.csv', index_col='questions_id', parse_dates=['questions_date_added'], infer_datetime_format=True)
questions = questions.rename(columns={'questions_author_id': 'author_id', 'questions_date_added': 'date_added', 'questions_title': 'title', 'questions_body': 'body', 'questions_processed':'processed'})

# Answers Import
answers = pd.read_csv('../input/answers.csv', index_col='answers_id', parse_dates=['answers_date_added'], infer_datetime_format=True)
answers = answers.rename(columns={'answers_author_id':'author_id', 'answers_question_id': 'question_id', 'answers_date_added': 'date_added', 'answers_body': 'body'})

# Tags Import
tags = pd.read_csv('../input/tags.csv',)
tags = tags.set_index('tags_tag_id')
tags = tags.rename(columns={'tags_tag_name': 'name'})

# Comments Import
comments = pd.read_csv('../input/comments.csv', index_col='comments_id')
comments = comments.rename(columns={'comments_author_id': 'author_id', 'comments_parent_content_id': 'parent_content_id', 'comments_date_added': 'date_added', 'comments_body': 'body' })


# School Memberships
school_memberships = pd.read_csv('../input/school_memberships.csv')
school_memberships = school_memberships.rename(columns={'school_memberships_school_id': 'school_id', 'school_memberships_user_id': 'user_id'})

# Groups Memberships
group_memberships = pd.read_csv('../input/group_memberships.csv')
group_memberships = group_memberships.rename(columns={'group_memberships_group_id': 'group_id', 'group_memberships_user_id': 'user_id'})

# Emails
emails = pd.read_csv('../input/emails.csv')
emails = emails.set_index('emails_id')
emails = emails.rename(columns={'emails_recipient_id':'recipient_id', 'emails_date_sent': 'date_sent', 'emails_frequency_level': 'frequency_level'})

#####################################################
print('Important numbers:')
print('\nThere are:')
print(f'- {len(students)} Students.', end="\t")
print(f'- {len(professionals)} Professionals.')
print(f'- {len(questions)} Questions.', end="\t")
print(f'- {len(answers)} Answers.')
print(f'- {len(tags)} Tags.', end="\t\t")
print(f'- {len(comments)} Comments.')
print(f'- {school_memberships["school_id"].nunique()} Schools.', end="\t\t")
print(f'- {len(pd.read_csv("../input/groups.csv"))} Groups.')
print(f'- {len(emails)} Emails were sent.')
#####################################################

# Questions-related stats
tag_questions = pd.read_csv('../input/tag_questions.csv',)
tag_questions = tag_questions.rename(columns={'tag_questions_tag_id': 'tag_id', 'tag_questions_question_id': 'question_id'})
count_question_tags = tag_questions.groupby('question_id').count().rename(columns={'tag_id': 'count_tags'}).sort_values('count_tags', ascending=False)
print('\nInteresting statistics: ')
print(f'- {(answers["question_id"].nunique()/len(questions))*100:.2f} % of the questions have at least 1 answer.')
print(f'\n- {(len(count_question_tags)/len(questions))*100:.2f}% of questions are tagged by at least {count_question_tags["count_tags"].tail(1).values[0]} tag.')
print(f'- Mean of tags per question: {count_question_tags["count_tags"].mean():.2f} tags per question.')

tag_users = pd.read_csv('../input/tag_users.csv',)
tag_users = tag_users.rename(columns={'tag_users_tag_id': 'tag_id', 'tag_users_user_id': 'user_id'})
users_who_follow_tags = list(tag_users['user_id'].unique())
nbr_pros_tags = len(professionals[professionals.index.isin(users_who_follow_tags)])
nbr_students_tags = len(students[students.index.isin(users_who_follow_tags)])
print(f'\n- {(nbr_pros_tags / len(professionals))*100:.2f} % of the professionals follow at least 1 Tag ({nbr_pros_tags}).')
print(f'- {(nbr_students_tags / len(students))*100:.2f} % of the students follow at least 1 Tag ({nbr_students_tags}).')

question_scores = pd.read_csv('../input/question_scores.csv')
nbr_questions_with_hearts = question_scores[question_scores['score'] > 0]['id'].nunique()
print(f'\n- {(nbr_questions_with_hearts/len(questions))*100:.2f} % of questions were upvoted ({nbr_questions_with_hearts}).')

answer_scores = pd.read_csv('../input/answer_scores.csv')
nbr_answers_with_hearts = answer_scores[answer_scores['score'] > 0]['id'].nunique()
print(f'- {(nbr_answers_with_hearts/len(questions))*100:.2f} % of answers were upvoted ({nbr_answers_with_hearts}).')


# School/Group Related Stats

def is_student(user_id):
    if user_id in students.index.values:
        return 1
    elif user_id in professionals.index.values:
        return 0
    else:
        raise ValueError('User ID not student & not professional')

school_memberships['is_student'] = school_memberships['user_id'].apply(is_student)
school_memberships['is_student'] = school_memberships['is_student'].astype(int)
count_students_professionals = school_memberships.groupby('is_student').count()[['school_id']].rename(columns={'school_id':'count'})
print(f'\n- Only {count_students_professionals.loc[1].values[0]/len(students):.2f} % of the students are members of schools ({count_students_professionals.loc[1].values[0]}).')
print(f'- Only {count_students_professionals.loc[0].values[0]/len(professionals):.2f} % of the professionals are members of schools ({count_students_professionals.loc[0].values[0]}).')

group_memberships['is_student'] = group_memberships['user_id'].apply(is_student)
group_memberships['is_student'] = group_memberships['is_student'].astype(int)
count_students_professionals = group_memberships.groupby('is_student').count()[['group_id']].rename(columns={'group_id':'count'})
print(f'\n- Only {count_students_professionals.loc[1].values[0]/len(students):.2f} % of the students are members of groups ({count_students_professionals.loc[1].values[0]}).')
print(f'- Only {count_students_professionals.loc[0].values[0]/len(professionals):.2f} % of the professionals are members of groups ({count_students_professionals.loc[0].values[0]}).')


print('')


# ## 1- Need to increase the number of active professionals:
# The following two graphs examine the degree activity of professionals in terms of number of posted answers.
# 
# * The first Pie Chart shows that most of the professionals still haven't posted their first answer.
# * The second Bar Graph compares the number of active (posted at least one answer) and inactive (didn't post any answer) professionals each year.
# 
# A good recommendation system can surely help professionals find relevant questions to answer and increase their activity on the platform.

# In[2]:


# Professionals with zero answers
nbr_pros_without_answers = len(professionals) - answers['author_id'].nunique()
#print(f'\n- {(nbr_pros_without_answers/len(professionals))*100:.2f} % of the professionals have Zero answers ({nbr_pros_without_answers}).')
fig = {
    'data': [{
        'type': 'pie',
        'labels': ['Zero answers', '> 0 answers'],
        'values': [nbr_pros_without_answers , len(professionals) - nbr_pros_without_answers],
        'textinfo': 'label+percent',
        'showlegend': False,
        'marker': {'colors': [ '#00FF66', '#D9BCDB',], 'line': {'width': 3, 'color': 'white'}},
    }],
    'layout': {
        'title': 'Professionals with Zero Answers'
    }
}
iplot(fig)


# In[3]:


# Answers Import
years = questions['date_added'].dt.year.unique()
years = sorted(years)
professionals['date_joined'] = pd.to_datetime(professionals['date_joined'])
activity_per_year = {}

for y in years:
#y = 2013
    limit_date = pd.to_datetime(f'{y}-12-31') - np.timedelta64(200, 'D')
    year_answers = answers[answers['date_added'].dt.year == y]
    professionals_up_to_year = professionals[professionals['date_joined'].dt.year <= y]
    
    nbr_active_pros = year_answers['author_id'].nunique()
    nbr_inactive_pros = len(professionals_up_to_year) - nbr_active_pros
    activity_per_year[y] = (nbr_active_pros, nbr_inactive_pros)


fig = {
    'data': [
        {
        'type': 'bar',
        'name': 'Number of Active Professionals',
        'x': years,
        'y': [e[0] for e in list(activity_per_year.values())],
        'marker': {'color': '#db2d43'}
        },
        {
        'type': 'bar',
        'name': 'Number of Inactive Professionals',
        'x': years,
        'y': [e[1] for e in list(activity_per_year.values())],
        'marker': {'color': '#906FA8'}
        }
    ],
    'layout': {
        'title': 'Number of Active vs Non-Active Professionals each year',
        'xaxis': {'title': 'Years'},
        'yaxis': {'title': 'Number of Professionals',},
        'barmode': 'stack',
        'legend': {'orientation': 'h'},
    }
}
iplot(fig)


# ## 2- Mean Time-To-First-Answer. Can we do better ?
# The following graph shows the evolution of the means of Time-to-First-Answer for questions of each year. A good recommendation system must minimize this metric.
# 
# **Mean Time-to-First-Answer**: $\frac{1}{nbr \thinspace questions}\sum_{q}^{questions}{nbr \thinspace days \thinspace between \thinspace question \thinspace q \thinspace was \thinspace posted \thinspace and \thinspace its \thinspace first \thinspace answer}$

# In[4]:


answers = answers.rename(columns={'date_added': 'answers_date_added'})
questions = questions.rename(columns={'date_added': 'questions_date_added'})
first_answers = answers[['question_id', 'answers_date_added']].groupby('question_id').min()
answers_questions = first_answers.join(questions[['questions_date_added']])
answers_questions['diff_days'] = (answers_questions['answers_date_added'] - answers_questions['questions_date_added'])/np.timedelta64(1,'D')
vals = [answers_questions[answers_questions['questions_date_added'].dt.year == y]['diff_days'].mean() for y in years]
LINE_COLOR = '#9250B0'
fig = {
    'data': [{
        'type': 'scatter',
        'x': years,
        'y': vals,
        'line': {'color': LINE_COLOR}
    }],
    'layout': {
        'title': 'Evolution of Time to First Response in days',
        'xaxis': {'title': 'Years'},
        'yaxis': {'title': 'Time to First Response'}
    }
}
iplot(fig)
answers = answers.rename(columns={'answers_date_added': 'date_added'})
questions = questions.rename(columns={'questions_date_added': 'date_added'})


# ## 3- Number of accurate recommendations:
# Next, we examine how many accurate recommendations are sent each year. How many of them were answered by the recipients of the emails. Again, this number must be maximized by our recommender system.

# In[5]:


# Number of accurate recommendations
emails['date_sent'] = pd.to_datetime(emails['date_sent'], infer_datetime_format=True)
matches = pd.read_csv('../input/matches.csv')
matches = matches.join(emails[['recipient_id', 'date_sent']], on='matches_email_id')

matches = matches.rename(columns={'matches_question_id': 'question_id', 'matches_email_id': 'email_id'})
all_recommendations_per_year = []
accurate_recommendations_per_year = []
matches['author_id'] = matches['recipient_id']
for y in years:
    year_answers = answers[answers['date_added'].dt.year == y]
    year_recommendations = matches[matches['date_sent'].dt.year == y]
    all_recommendations_per_year.append(len(year_recommendations))
    m = year_answers.reset_index().merge(year_recommendations, on=['question_id', 'author_id']).set_index('answers_id')
    nbr_accurate_recommendations = len(m)
    accurate_recommendations_per_year.append(nbr_accurate_recommendations)
    #print(f'- {(nbr_accurate_recommendations/len(matches))*100:.2f} % of recommended questions in emails were accurate (lead to professional answering the recommended question) ({nbr_accurate_recommendations})')

#print(accurate_recommendations_per_year)
LINE_COLOR = '#9250B0'
fig = {
    'data': [{
        'type': 'scatter',
        'x': years,
        'y': accurate_recommendations_per_year,
        'line': {'color': LINE_COLOR}
    }],
    'layout': {
        'title': 'Evolution of Number of Accurate recommendations',
        'xaxis': {'title': 'Years'},
        'yaxis': {'title': 'Time to First Response'}
    }
}
iplot(fig)


# ## 4- Proportion of accurate recommendations:
# 
# The following graph plots the the Ratio of number of accurate recommendations over all recommendations made. We can see that even though an increase of number of accurate recommendations occured in 2016 (previous graph), it was due to a significant increase in recommendations made. Ideally, a good recommender should maximize the number of accurate recommendations and minimize the number of incorrect ones in order to avoid churning.

# In[6]:


proportions_of_accurate_recommendations = np.array(accurate_recommendations_per_year)/np.array(all_recommendations_per_year)
proportions_of_accurate_recommendations = [0 if np.isnan(e) else e for e in proportions_of_accurate_recommendations]
#print(proportions_of_accurate_recommendations)
fig = {
    'data': [{
        'type': 'scatter',
        'x': years,
        'y': proportions_of_accurate_recommendations,
        'line': {'color': LINE_COLOR}
    },
    ],
    'layout': {
        'title': 'Percentage of Accurate recommendations',
        'xaxis': {'title': 'Years'},
        'yaxis': {'title': 'Proportion of Accurate Recommendations', 'tickformat': ',.0%'}
    }
}
iplot(fig)


# In[7]:


# Garbadge collect stuff we won't be using for building the recommender.

del m
del emails
del matches
del students
del school_memberships
del group_memberships
del count_question_tags
del users_who_follow_tags
del nbr_pros_tags
del nbr_students_tags
del nbr_pros_without_answers
del nbr_questions_with_hearts
del count_students_professionals
gc.collect()
print('')


# ### Takeaways:
# * Tags are heavily used by students in questions.
# * Most professionals follow tags to find questions related to their expertise.
# * **Most of the professionals (~63%) haven't answered any question yet.**
# * **Only a tiny proportion of recommended questions (~0.41%) in emails were accurate enough to probably lead the recipient to answer.**
# * For the moment, we can not rely on school/group memberships, because only a tiny portion of the users have used them.

# ## 5- Problem with the current System

# In the current system, emails containing recommended questions are sent to professionals on a daily basis by default. 
# 
# The possible frequencies that a professional can choose from are:
# * Immediate
# * Daily
# * Weekly
# * Turn off all notifications.
# 
# The 'daily' option is problematic. It is extremely difficult to to maintain a good quality of recommendations when the frequency is as high as 'Daily'. **We thus end up with a huge number of emails being sent daily with poor-quality recommendations. This can cause the professional to start ignoring emails and ultimately not returning to the site.**
# 
# In addition, having poor-quality recommendations increases the chance that the professional will choose to turn off all notifications, which is not desirable.
# 
# <span style="color: blue; font-weight: bold;">Quick Solution proposal: </span> Maintain good-quality recommendations by removing the 'Daily' option, and only keeping the 'Immediate' & 'Weekly' options.
# 
# Another *future* solution would be to leave it up to the system to decide when to email each professional depending on the interaction of the professional with the site.
# 

# # Basic concepts and techniques used in the Recommender System <a class="anchor" id="concepts"></a>
# The recommender system works in two "modes":
# * **Professional-to-Questions**: Recommend top K questions to a particular professional (needed for the professionals who choose a fixed frequency like 'Weekly' option )
# * **Question-to-Professionals**: Recommend top K professionals most likely to answer a particular question. (needed for the professional who choose the 'Immediate' option)
# 

# ## The Exploration-Exploitation Dilemma in Recommendations:
# 
# ![slots](https://i.imgur.com/pFO04zu.jpg?3)
# 
# 

# A recommender system's job is not that simple. If a recommender system keeps suggesting the same items to the same users, then in some cases, questions about fairness might be raised, in other cases, users might get bored getting the same type of content. In the case of Question-Answering platforms like CareerVillage, potential interests (other than the ones already expressed by the professional through tags) might be ignored and users might stop coming to the platform.
# 
# A recommendation system must not only recommend relevant questions to the professionals, **Occasionally, it should also introduce them to potentially new types of questions that might interest them**. It has to deal with the cold-start problem, where very little information about he professional is known.
# 
# In the ML litterature, finding the right tradeoff between these two components is called the **Exploration-Exploitation problem**.

# ## The Epsilon-Greedy Algorithm ( in a nutshell )
# 
# To tackle the Exploration-Exploitation problem, a popular algorithm called **'Epsilon-Greedy'** is used.
# 
# > It works by setting an Epsilon threshold, which represents the probability of 'Exploitation' .
# > 
# > A random number N between 0.0 and 1.0 is generated,
# > 
# > if N < Epsilon
# > 
# >     Exploit by searching similar questions based on the past
# > 
# > else
# > 
# >     Explore new questions
# 
# **The Epsilon-Greedy Algorithm is simple, easy to implement and does not need heavy computation, making it a great solution for the problem at hand. **
# 
# *( More details on the inner-workings in a later section )*
# 
# *Note: normaly Epsilon is used for exploration, in this implementation I used it for exploitation, but the idea is the same*

# ## LSA: Latent Semantic Analysis (in a nutshell)
# 
# Latent Semantic Analysis is a **simple**, yet **powerful** technique in Natural Language Processing. It captures the latent (hidden) topics of a corpus of text and represents each document by a vector of k dimensions, each pointing to one latent topic.
# 
# To do this, LSA relies on a robust mathematical technique called SVD (Singular-Value Decomposition), which factorizes a real matrix to a product of 3 matrices. ([More on LSA and SVD](#links))
# 
# 
# <span style="color: red; font-weight: bold;">Takeaway:</span> **Each question will be represented by a vector of length k. comparing the questions will be as easy as performing a cosine similarity between the vectors.**

# # Data Preprocessing is paramount ! <a class="anchor" id="preproc"></a>
# 
# 
# <div style="border: solid 1px blue; padding: 5px;"><h4><center><span style="color: red;">If we let Garbage In, we get Garbage Out ! (GIGO)</span><center></h4></div>
# 
# <br/>
# The most important data type in this project is Text (questions, tags ...). Unfortunately, if left unpreprocessed, it becomes extremely hard to extract useful information from it.
# 
# This section's goal, is to prepare the data by simplifying it and removing any noise that migh get in the way between us and the True Information that we want to extract.
# 
# This simple preprocessing can be easily done online in production, doesn't require a lot of computation.
# 

# ## 1- Tags:
# For some reason, there are many tags which are not used in any question (and they are also not followed by any user).

# In[8]:


# Drop tags that are not used in any question and not followed by any user (it will clean a lot of useless stuff)
useless_tags = tags[~tags.index.isin(tag_questions['tag_id'].unique())]
useless_tags = tags[ (tags.index.isin(useless_tags.index.values)) & (~tags.index.isin(tag_users['tag_id'].values)) ]
tags = tags.drop(useless_tags.index)

print(f'- {len(useless_tags)} useless tags were found and dropped.')


# Next, we make the following transformations to the tags:
# * make all tags lowercase.
# * create a new 'processed' column to hold the processed version of each tag
# * remove any special characters from the text.
# * correct some short words (yrs -> years)
# * lemmatize the tags ( eg. 'wolves' -> 'wolf' )
# * remove tags without any meaning that are just numbers, just preprositions, pronouns, stop-words ... ('where', 'and', 'the', '10', ...etc)
# 

# In[9]:


# Preprocessing Tags

nbr_tags = len(tags)

stop_words = set(stopwords.words('english'))
# some common words / mistakes to filter out too
stop_words.update(['want', 'go', 'like', 'aa', 'aaa', 'aaaaaaaaa', 
                   'good', 'best', 'would', 'get', 'as', 'th', 'k',
                   'become', 'know', 'us'])
special_characters = f'[{string.punctuation}]'
lm = WordNetLemmatizer()


tags['name'] = tags['name'].str.lower()
tags.fillna('', inplace=True)
tags['processed'] = tags['name'].str.replace(special_characters, '')
tags['processed'] = tags['processed'].str.replace('^\d+$', '') # tags that are just numbers :-/
tags['processed'] = tags['processed'].apply(lambda x: lm.lemmatize(x)) # avoid having plurals like 'career' and 'careers'
tags['processed'] = tags['processed'].str.replace('^\w$', '') # single letter tags :-/
tags['processed'] = tags['processed'].str.replace(r'(\d+)(yrs?)', r'\1year') #
tags['processed'] = tags['processed'].apply(lambda x: x if x not in stop_words else '')

# Drop tags which are prepositions, pronouns, determiners, wh-adverbs (where, ...)
tags_to_drop = []
for i, t in tags['processed'].iteritems():
    if len(t) > 0 and nltk.pos_tag([t])[0][1] in ['IN', 'PRP', 'WP$', 'PRP$', 'WP', 'DT', 'WRB']:
        tags_to_drop.append(i)
tag_questions = tag_questions.drop(tag_questions[tag_questions['tag_id'].isin(tags_to_drop)].index)
tags = tags.drop(tags_to_drop)

# Drop tags which are just numbers
tags_to_drop = tags[tags['name'].str.contains('^\d+$')].index
tag_questions = tag_questions.drop(tag_questions[tag_questions['tag_id'].isin(tags_to_drop)].index)
tags = tags.drop(tags_to_drop)

# Drop tags which are just stop words ( after, the , with , ...)
tags_to_drop = tags[tags['name'].isin(stop_words)].index
tag_questions = tag_questions.drop(tag_questions[tag_questions['tag_id'].isin(tags_to_drop)].index)
tags = tags.drop(tags_to_drop)

print(f'{nbr_tags - len(tags)} Tags were filtered out.')
tags.sample(2)


# **<span style="color: red">Result</span>**: We are now able to see that the following tags are the same: "information-technology", "#informationtechnology", "#information-technology", "information-technology-".
# 
# A future task might be to explore how to add to this list the word "IT" (using word2vec), but the preprocessing is always necessary.

# ## 2- Questions
# * We create a new column 'processed' containing both 'title' & 'body' text, and do the same transformations we did to tags ( remove special characters, lemmatize words and remove stop words ).
# * Create a new column 'count_answers'.

# In[10]:


# Questions Cleaning

questions['processed'] = questions['title'] + ' ' + questions['body']
questions['processed'] = questions['processed'].str.lower()
questions['processed'] = questions['processed'].str.replace('<.*?>', '') # remove html tags
questions['processed'] = questions['processed'].str.replace('[-_]', '') # remove separators
questions['processed'] = questions['processed'].str.replace(special_characters, ' ') # remove special characters

questions['processed'] = questions['processed'].str.replace('\d+\s?yrs?', ' years') # single letter tags :-/

def lem_question(q):
    return " ".join([lm.lemmatize(w) for w in q.split() if w not in stop_words])
questions['processed'] = questions['processed'].apply(lem_question)

questions['processed'] = questions['processed'].str.replace(r'(\d+)($|\s+)', r'\2') # remove numbers which are not part of words
questions['processed'] = questions['processed'].str.replace(r'(\d+)([th]|k)', r'\2') # remove numbers from before th and k


# Function to preprocess new questions
# TODO: update function to do like above
def preprocess_question(q):
    q = q.lower()
    q = re.sub("<.*?>", "", q)
    q = re.sub("[-_]", "", q)
    q = re.sub("\d+", "", q)
    q = q.translate(q.maketrans('', '', string.punctuation))
    q = " ".join([lm.lemmatize(t) for t in q.split()])
    return q

cnt_answers = answers.groupby('question_id').count()[['body']].rename(columns={'body': 'count_answers'})
questions = questions.join(cnt_answers)
questions['count_answers'] = questions['count_answers'].fillna(0)
questions['count_answers'] = questions['count_answers'].astype(int)

print('Questions preprocessed.')
questions.sample(1)[['title', 'body', 'processed', 'count_answers']]


# ## 3- Professionals
# * **Count Answers:** Create a new column 'count_answers' for professionals
# * **Cleaning the headlines**
# * **Follow Tags ?**: This is just a handy column I added to differentiate after between Pros who do and don't when evaluating the recommender
# * **Last Answer Date**: We will rely also on this new column to know if the professional is active or not.

# In[11]:



# Count Answers
print('Counting Answers ...')
pro_answers_count = answers.groupby('author_id').count()[['question_id']].rename(columns={'question_id': 'count_answers'})
professionals = professionals.join(pro_answers_count)
professionals['count_answers'] = professionals['count_answers'].fillna(0)
professionals['count_answers'] = professionals['count_answers'].astype(int)


# Cleaning the headlines
print('Cleaning Headlines ...')
professionals['headline'] = professionals['headline'].fillna('')
professionals['headline'] = professionals['headline'].str.lower()
professionals['headline'] = professionals['headline'].str.replace('--|hello|hello!|hellofresh', '')

# Check if follow tags or not
print('Creating "follow_tags" column ...')
professionals['follow_tags'] = False
followers = list(tag_users['user_id'].unique())
professionals.loc[professionals.index.isin(followers), 'follow_tags'] = True

# Create Last Answer Date Column
print('Creating "last_answer_date" column ... ')
professionals = professionals.join(answers[['author_id', 'date_added']].groupby('author_id').max().rename(columns={'date_added': 'last_answer_date'}))


print('Professionals preprocessed')
professionals.sample(3)


# # Start Modeling !
# 
# Now that we have pre-processed our data, we are ready for the modeling part.
# 
# The modeling steps are as follows:
# 
# * **Apply TF-IDF on the hole question corpus.**
# * **Apply SVD to reduce the dimensionality of the vectors.**
# * **Construct a Questions Similarity Matrix using The Cosine Similarity function.**
# 
# After some experimentation, I chose the number of topics ( new dimensionality of question vectors ) to be 1100 ( values between 900~1100 are ok ).
# 

# In[12]:


start = time.time()

tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words,)

NUM_TOPICS = 1100
def build_model(qs , nbr_topics=NUM_TOPICS):
    print('Building the Model ...')
    # TF-IDF Transformation

    qs_tfidf = tfidf_vectorizer.fit_transform(qs['processed'])
    terms = tfidf_vectorizer.get_feature_names()
    print(' (1/3) TF-IDF matrix shape: ', qs_tfidf.shape)

    # Dimensionality Reduction with SVD
    model = TruncatedSVD(n_components=nbr_topics)
    transformer_model = model.fit(qs_tfidf)
    qs_transformed = transformer_model.transform(qs_tfidf)
    print(' (2/3) Shape after Dimensionality Reduction:', qs_transformed.shape)

    # Construct Similarity Matrix
    sim_mat = cosine_similarity(qs_transformed, qs_transformed)
    print(' (3/3) Similarity Matrix Shape', sim_mat.shape, '\n')
    return transformer_model, qs_transformed, sim_mat

transformer_model, Qs_transformed, Qs_sim_matrix = build_model(questions)

end = time.time()

print(f'{(end-start)/60:.2f} minutes')


# <span style="color: blue;">Notes on production:</span>
# * *Here, we use the totality of the question corpus when constructing the Similarity Matrix, in practice though, the similarity matrix will only be constructed with relatively recent questions ( last 1 or 2 years ), since old questions will not be of any use. Its construction only takes ~ 40 seconds for a ~ 24k x 24k matrix (pretty quick).*
# * *When in production, the Similarity Matrix & Questions Transformed Matrix must be updated on regular basis, depending on the traffic.*

# # Building the Recommendations Engine <a class="anchor" id="implementation"></a>
# 
# In this section, we will build the recommendations engine from scratch using only the techniques previously talked about. Each sub-section will deal with a specific sub-problem.
# 
# There are two main data structures we will work with:
# * **The Transformed Questions Matrix**: a Matrix where each row represents a single question encoded in a K dimensional vector
# * **The Similarity Matrix**: a NxN Matrix (where N is the number of questions) rating the similarity between all pair of questions on a scale of 0 ~ 1.
# 
# Note that the recommender prioritizes **quality over quantity**. So, when asked for N recommendations, it will return the best k recommendations, where $k \leq N$. The reason for this, is that the cost of bad recommendations can be high. We don't want users to get bad recommendations and ignore our future emails.

# ## 1- Professional-to-Questions Mode:
# In this mode, three types of professionals can be distinguished:
# * **Hot**: The professional has posted at least one answer. But if no answer is posted for a defined period ( in code I set the variable min_date_for_answers=400 days ) then professional will be treated as Cold.
# * **Cold**: The professional has never posted an answer, but follows some tags.
# * **Freezing**: The professional never posted an answer and doesn't follow any tags.
# 
# When recommending K questions, each iteration can be either **Exploitation** or **Exploration**
# 
# **How to deal with the 'Freezing' professional ?**
# 
# The 'Freezing' professional has most probably registered recently, and doesn't follow any tags. We don't know much about him, the recommendations are more like 'suggestions' with the goal of taking him to the higher categories. 
# 
# We can suggest:
# * **Exploit:** session-based recommendations which recommend questions similar to questions already visited / upvoted or commented by the professional
# * **Explore:** popular questions on the platform and newly created ones.
# 
# ( session-based recommendations are only feasible in production, so in this implementation we stick with exploration for this type of users )
# 
# **How to deal with the 'Cold' professional ?**
# 
# Unlike the 'Freezing' professional, the 'Cold' one follows one more tags. This important hint must be fully exploited as followed:
# * **Exploit:** find relatively recent questions from the tags followed.
# * **Explore:** find tags similar to the followed tags and do the same. This is possible because of the simple *pre-processing of tags*.
# 
# **How to deal with the 'Hot' professional ?**
# 
# This type of professional has already expressed interest in one or more questions.
# * **Exploit:** suggest a similar question to one of the previously answered questions. To choose the answered question that will be the basis, we score all the questions and choose one question randomly based on the scores, which act as a probability distribution.
# * **Explore:** suggest most recent & similar questions from the tags followed as follows:
#     * Select the n most recent questions from each followed tag.
#     * from all those questions, select one the question which is most similar to one of the answered ones.
# 
# <span style="color: red">The Exploration/Exploitation approach has the advantage of not letting many questions unanswered by recommending often, and not over-focusing on the answered ones.</span>
# 
# ### Calculating the Exploit Threshold and Scoring Questions (for the 'Hot' professional):
# 
# Unlike the the two other types of professionals, the optimal Exploit Threshold for the 'Hot' professional is dynamic and changes from a professional to another. Some professionals have only answered one question, while others have answered many. Some professionals have answers which date to a relatively long time, while others have just recently answered a few. **Taking these parameters into consideration affects positively the quality of the recommendations**.
# 
# * We want our recommender to prioritize questions similar to questions recently answered by the professional
# * We also don't want to completely ignore older questions.
# 
# ### Question-Scoring Formula:
# The following formula scores the questions answered by the Hot professional while capturing the first note above:
# 
# $$ score(x) = \frac{log (\frac{x}{\epsilon})}{log (\frac{1}{\epsilon})} $$
# where:
# - x is the number of days elapsed between the date question was answered and today
# - $\epsilon$ is the maximum number of days after which we no longer consider the question to be relevant ( now it is set to 370 but can be changed with the variable "eps", see code below )
# 
# The formula gives a score between 0~1 where 1 means that the question is very relevant and should be used as a reference.
# 
# Below is the values-table of the formula (credits to [this online grapher](https://www.desmos.com/calculator)). The formula "rewards" questions that are recent ( x is small ). And as x gets smaller, the score drops drown in a logarithmic fashion until it interesects with the x axis exactly at the value $\epsilon$.
# 
# ![question_scoring_formula_table](https://i.imgur.com/HiHiQfA.png)
# 
# ### Exploit-Threshold Formula:
# 
# After the (answered) questions get scored, the Exploit Threshold is calculated as follows:
# 
# $$ threshold = log (\sqrt{x} + 1) \cdot \alpha +  \epsilon $$
# where:
# - x is the number of recently answered questions ( considering here only number of answers which have scores > 0 ).
# - $\alpha$ controls the exploitation intensity (1.35 in the implementation). Acceptable values range between 1.0 for low exploitation and 1.7 for high exploitation.
# - $\epsilon$ is an **optional** small value term (0.1 on the implementation) added if all answered questions are old ( meaning, that if x = 0, we still give a very small probability to $\epsilon$ for exploitation, see the implementation below ).
# 
# Below is values-table of the formula. The formula give a "bigger" exploit-threshold as more questions answered increase. It increases in a logarithmic fashion. ( there is a 1.35 in the left of the function in the value table but I couldn't get it to be visible)
# 
# ![threshold_formula_table](https://i.imgur.com/SqXgM5T.png)
# 
# Below is the implementation of the two formulas:

# In[13]:


def calculate_score_question_answered(days_elapsed_after_answer):
    eps = 370
    score = np.log10(days_elapsed_after_answer*(1/eps)) / (np.log10(1/eps))
    score = 0.001 if score < 0 else score # questions that got a score lower than 0 are still given a very low score
    return score

def calculate_exploit_threshold(answered_question_scores, nbr_recommendations, alpha=1.35):
    nbr_questions_answered = len([s for s in answered_question_scores if s > 0])
    eps = 0.1 if nbr_questions_answered == 0 else 0
    return np.log10(np.sqrt(nbr_questions_answered) + 1) * alpha + eps


# **The next snippet of code builds the recommendation engine using two main functions:**
# * **get_similar_questions**: returns similar questions to the one given using the similarity matrix ( the parameter "similarity_threshold" controls what similar means, I set it by default to be 0.4 as I found that value to work well for most cases ) .
# * **recommend_questions_to_professional**: given a professional ID, returns top K recommended questions.
# 
# The debug variable below, if set to True,  makes exploration / exploitation decisions visible.

# In[14]:


debug = False


# In[15]:


# Set current date as the last day of the data
def set_today(d_str):
    d = pd.to_datetime(d_str)
    
    min_for_questions = d - np.timedelta64(600, 'D') # used for the Freezing professional to select the latest questions and for the cold to select the latest questions in followed and suggested tags
    min_for_answers = d - np.timedelta64(400, 'D')   # used for hot professional to select his last answers. if no answers in this period, Hot professional will be treated as Cold
    return d, min_for_questions, min_for_answers

today, min_date_for_questions, min_date_for_answers = set_today('2019-01-31')


# *( Feel free to check the code below collapsed )*

# In[16]:



def choose_random_answered_question(question_score_dic):
    random_key = choices(list(question_score_dic.keys()), list(question_score_dic.values()))[0]
    return (random_key, question_score_dic[random_key])


def choose_random_followed_tag(pro_id):
    followed_tags = tag_users[tag_users['user_id'] == pro_id]
    return followed_tags.sample(1)['tag_id'].values[0]

def get_similar_questions(qid, nbr_questions=10, except_questions_ids=[], prioritize=False, similarity_threshold=0.4):
    recommendations = pd.DataFrame([])

    #print(len(except_questions_ids))
    #print()
    q_dists_row = list(Qs_sim_matrix[questions.index.get_loc(qid)])
    for eq_id in except_questions_ids:
        #print('removing ', eq_id)
        #print(len(q_dists_row), questions.index.get_loc(eq_id))
        q_dists_row[questions.index.get_loc(eq_id)] = -1
    q_dists_row = pd.Series(q_dists_row).sort_values(ascending=False)[:100]
    q_dists_row = q_dists_row[1:]

    if not prioritize:
        q_dists_row = q_dists_row[:nbr_questions]
        for i, d in q_dists_row.iteritems():
            qid = questions.index.values[i]
            recommendations = recommendations.append(questions.loc[qid])
    else:
        qid_to_score = {}
        for i, d in q_dists_row.iteritems():
            qid = questions.index.values[i]
            if d > similarity_threshold:
                #print(qid)
                q_added = questions.loc[qid, 'date_added']
                days_elapsed = (today - q_added) / np.timedelta64(1, 'D')
                qid_to_score[qid] = d * days_elapsed
        qid_scores = sorted(qid_to_score.items(), key=lambda x: x[1])[:nbr_questions]
        for qid, score in qid_scores:
            print(q_dists_row[questions.index.get_loc(qid)], qid_to_score[qid]) if debug else None
            recommendations = recommendations.append(questions.loc[qid])
    return recommendations



def recommend_questions_to_professional(pro_id, nbr_recommendations=10, silent=False, alpha=1.35):
    print('Professional ID:', pro_id ) if not silent else None

    # tags followed
    tags_followed = tag_users[tag_users['user_id'] == pro_id]['tag_id']
    tags_followed = tags[tags.index.isin(tags_followed)]
    print('Followed Tags: ', tags_followed['name'].values)  if not silent else None

    # Number of answered questions
    cnt_pro_answers = professionals.loc[pro_id, 'count_answers']
    if cnt_pro_answers > 0:
        pros_answers = answers[(answers['author_id'] == pro_id) & (answers['date_added'] < min_date_for_answers)]
        cnt_pro_answers = len(pros_answers)

    # Type of Start
    cold_start = (cnt_pro_answers == 0)
    freezing_start = (cold_start and len(tags_followed) == 0 )

    n = 3 # Nbr of questions per tag
    recommendations = pd.DataFrame([])


    # Freezing Start
    if freezing_start:
        print('Freezing ...')  if not silent else None
        recommendations = recommendations.append(questions[questions['date_added'] > min_date_for_questions].sample(10))

    # Cold Start
    elif cold_start:
        print('Cold', cnt_pro_answers)  if not silent else None

        qids_from_followed_tags  = tag_questions[tag_questions['tag_id'].isin(tags_followed.index.values)]['question_id'].values
        qids_from_followed_tags  = list(questions[(questions.index.isin(qids_from_followed_tags))   & (questions['date_added'] > min_date_for_questions)].sort_values('date_added', ascending=False).index.values)

        tags_suggested = tags[tags['processed'].isin(tags_followed['processed'].values)]
        tags_suggested = tags_suggested[~tags_suggested.index.isin(tags_followed.index.values)]
        print('Suggested Tags: ', tags_suggested['name'].values)  if not silent else None
        suggested_tags_available = len(tags_suggested) > 0
        # If there are suggested tags, we do explore on them while exploiting on the followed tags
        if suggested_tags_available:
            qids_from_suggested_tags = tag_questions[tag_questions['tag_id'].isin(tags_suggested.index.values)]['question_id'].values
            qids_from_suggested_tags = list(questions[(questions.index.isin(qids_from_followed_tags))  & (questions['date_added'] > min_date_for_questions)].sort_values('date_added', ascending=False).index.values)
            exploit_threshold = .6
        # If no suggested tags are available, we just exploit on the followed tags
        else:
            exploit_threshold = 1


        print('Exploit Threshold: ', exploit_threshold) if debug else None
        for i in range(1, nbr_recommendations+1):
            if np.random.rand() < exploit_threshold and len(qids_from_followed_tags) > 0:
                # Exploit followed tags
                print(f'{i}- Exploit followed tags') if debug else None
                random_index = choice(qids_from_followed_tags)
                q = questions.loc[random_index]
                recommendations = recommendations.append(q)
                qids_from_followed_tags.remove(random_index)
            elif suggested_tags_available and len(qids_from_suggested_tags) > 0:
                # Suggest from suggested tags
                print(f'{i}- Explore suggested tags') if debug else None
                random_index = choice(qids_from_suggested_tags)
                q = questions.loc[random_index]
                recommendations = recommendations.append(q)
                qids_from_suggested_tags.remove(random_index)
            else:
                # no more questions from the pool
                pass

    # Hot Start
    else:
        
        questions_answered_ids = list(pros_answers['question_id'].unique())
        questions_answered = questions[questions.index.isin(questions_answered_ids)].sort_values('date_added', ascending=False)
        questions_answered_locs = []
        for qid in questions_answered_ids:
            questions_answered_locs.append(questions.index.get_loc(qid))

        print('Hot, Answered Questions: ', cnt_pro_answers)  if not silent else None
        #print(questions_answered_locs)
        display(questions_answered[['date_added', 'title', 'body', 'count_answers']])  if not silent else None
        
        # calculate answered questions scores
        q_scores = {}
        for i, q in questions_answered.iterrows():
            answer_post_date = pros_answers[pros_answers['question_id'] == i]['date_added'].values[0]
            days_elapsed_after_answer = (today - answer_post_date)/np.timedelta64(1, 'D')
            q_scores[i] = calculate_score_question_answered(days_elapsed_after_answer)
        print('Question-Scores: ', q_scores) if debug else None

        # calculate exploit_threshold
        exploit_threshold = calculate_exploit_threshold(list(q_scores.values()), nbr_recommendations, alpha=alpha)
        print('Exploit Threshold:', exploit_threshold) if debug else None
        except_qs = []
        except_qs += questions_answered_ids
        for i in range(nbr_recommendations):

            if np.random.rand() < exploit_threshold:
                # Exploit
                random_q_score = choose_random_answered_question(q_scores)
                print('\nExploit Question', random_q_score) if debug else None
                recommendations = recommendations.append(get_similar_questions(random_q_score[0], nbr_questions=1, except_questions_ids=except_qs, prioritize=True))
            else:
                # Explore
                
                # Get Latest n questions from all followed tags
                n = 5
                latest_questions = pd.DataFrame([])
                for tid in tags_followed.index.values:
                    qids = tag_questions[tag_questions['tag_id'] == tid]['question_id'].values
                    tag_qs = questions[questions.index.isin(qids)]
                    tag_qs = tag_qs[~tag_qs.index.isin(except_qs)]
                    if len(tag_qs) > 0:
                        tag_qs = tag_qs.sort_values('date_added', ascending=False)
                        latest_questions = latest_questions.append(tag_qs.head(n))
                #display(latest_questions)
                
                # Select the most similar one to the ones answered using the similarity matrix
                best_question_id = 0
                best_distance = float('-inf')
                for qid, r in latest_questions.iterrows():
                    qloc = questions.index.get_loc(qid)
                    for aqloc in questions_answered_locs:
                        d = Qs_sim_matrix[qloc, aqloc]
                        if best_question_id == 0 or d > best_distance:
                            best_question_id = qid
                            best_distance = d

                print('\nExplore Tags', best_question_id, best_distance) if debug else None
                if best_question_id != 0:
                    recommendations = recommendations.append(questions.loc[best_question_id])
            except_qs = list(recommendations.index.values)
            except_qs += questions_answered_ids

    return recommendations


# ### Testing the recommender Part 1: standard random cases
# 
# Here I'll let the recommender run on two randomly chosen professionals ( cold & Hot ).
# 

# In[17]:


# Random Hot Professional
random_hot_pro_id = professionals[(professionals['count_answers'] > 2) & (professionals['count_answers'] < 5)].sample(1).index.values[0]

# Random Cold Professional ( check if he follows some tag )
random_cold_pro_id = professionals[(professionals['count_answers'] == 0) & (professionals['follow_tags'] == True)].sample(1).index.values[0]


#for random_pro_id in [random_hot_pro_id, random_cold_pro_id]:
for random_pro_id in [random_hot_pro_id, random_cold_pro_id]:
    recs = recommend_questions_to_professional(random_pro_id, nbr_recommendations=10)
    print('Recommendations: ')
    display(recs[['date_added', 'title', 'body', 'count_answers']]) if len(recs) > 0 else None


# * **Testing the recommender Part 2: Difficult Case **
# 
# This is the case of a professional who follows 'cooking' and 'computer-games', but has answered a question about 'journalism' and 'scholarships'. We can see that the system balances nicely between the topics.

# In[18]:


random_hot_pro_id = 'fbd6566ddf36402abeb031c088096ae4'
recs = recommend_questions_to_professional(random_hot_pro_id, nbr_recommendations=10)
print('Recommendations:')
display(recs[['date_added', 'title', 'body', 'count_answers']])


# ## 2- Question-to-Professionals mode:
# 
# Given a question, recommend the top K professionals to answer. This mode is used for the 'Immediate' professionals.
# 
# The overall approach taken is straightforward: recommend professionals who answered similar questions to the one given. 
# 
# The 'Hot' professionals have the biggest probability to answer and are recommended, since a **key requirement of the recommender is to get a good quality answer as soon as possible.**
# 
# 
# ### Is a professional active ?
# It is important to know the answer to this question. Maybe the professional has answered numerous questions similar to the query but is no longer active on the platform ! So out of N candidates, we will select our k recommended ones based on their activity.
# 
# 
# Here, we select the top k candidates using two metrics:
# * How active they are: we determine this by using the 'last_answer_date' column, if the professional has answered a question in the last n days he gets priority (n = 60 by default in implementation) .
# * If the number candidates is still smaller than the required k, we add candidates based on the number of similar questions they answered even if they weren't active in the last n days. ( of course they have all answered similar questions, but in the first step, we prioritized the active ones, in the second step, which is optional, we fill in the missing places with professionals who answered most similar questions ). 
# 
# You can check the small piece of code below: 

# In[19]:


def recommend_professionals_for_question(qid, nbr_recommendations=10, inactivity_period=60):
    #print(len(questions), len(answers))
    similar_questions = get_similar_questions(qid, nbr_questions=10, except_questions_ids=[], prioritize=False)
    #display(similar_questions)
    answer_author_ids = answers[answers['question_id'].isin(similar_questions.index.values)]['author_id'].values
    answer_author_ids = pd.Series(answer_author_ids).value_counts()
    
    # Step 1: Check how active the the candidates are
    min_last_answer_date = today - np.timedelta64(inactivity_period, 'D')
    candidates = professionals[(professionals.index.isin(answer_author_ids.index.values)) & (professionals['last_answer_date'] > min_last_answer_date)].sort_values('last_answer_date', ascending=False)
    answer_author_ids = answer_author_ids.drop(candidates.index)
    
    # Step 2: if number of candidates is still smaller than nbr_recommendations, fill in with other authors based on how many similar questions they answered.
    if len(candidates) < nbr_recommendations:
        others = answer_author_ids.head(nbr_recommendations-len(candidates)).index.values
        candidates = candidates.append(professionals[professionals.index.isin(others)])
    return candidates[:nbr_recommendations]


# ### Testing the recommender

# In[20]:


random_question_index = choice(questions.index.values)

print('Random Question: ', random_question_index,  questions.loc[random_question_index]['date_added'])
print(questions.loc[random_question_index]['title'])
print(questions.loc[random_question_index]['body'])
recommend_professionals_for_question(random_question_index, nbr_recommendations=8)[['location', 'industry', 'headline', 'count_answers', 'last_answer_date']]


# ## Helper functions in production:

# ### Get Tag Suggestions for a new question:
# It is very important to control tags, and use existing ones when possible. The following function suggests tags for a new question:

# In[21]:


# Analyze processed question and extracts implicit tags ( eg. 'computer science' => 'computerscience')
def get_tag_suggestions(q_p):
    #q_p = preprocess_question(q)
    #print(q_p)
    q_tokens = nltk.word_tokenize(q_p)
    q_tokens_cpy = q_tokens.copy()
    
    qp_tagged = nltk.pos_tag(q_tokens)
    important = []
    for t,pos in qp_tagged:
        if t not in stop_words and pos == 'NN' and len(tags[tags['processed'] == t]) > 0 :
            i = q_tokens.index(t)
            #print(len(q_p), t, i)
            poses_before_after = []
            if i > 0:
                poses_before_after.append(nltk.pos_tag([q_tokens[i-1]])[0])
            if i < (len(q_tokens)-1):
                poses_before_after.append(nltk.pos_tag([q_tokens[i+1]])[0])
            for i, bf in enumerate(poses_before_after):
                #print(t, bf)
                if bf[1] in ['NN', 'NNS', 'JJ', 'JJR', 'VBG']:
                    s = f'{t}{bf[0]}' if i == 1 else f'{bf[0]}{t}'
                    important.append(s)
            q_tokens.remove(t)
    important = set(important)
    for i in set(important):
        if i not in tags['processed'].values or i in q_tokens_cpy:
            important.remove(i)
    #print(len(important),important)
    
    return tags[tags['processed'].isin(important)]


# * Example: 
# 
# The following example illustrates how many tags this question is related to.

# In[22]:


new_question = 'I am a student in computer science and I want to be a data scientist but I dont know how to study machine learning and artificial intelligence. Can anyone give some advice ?' 
p_q = preprocess_question(new_question)
suggestions = get_tag_suggestions(p_q)
print('Question: ', new_question)
print('\nTag Suggestions: ')
suggestions[['name']]


# ### Function to add a new question to DB:
# 
# When a new question enters a DB, it should be converted and added to our model.
# * The following function transforms the new question to a vector and adds it to the matrix
# * adds an entry ( one row and one column ) to the similarity matrix

# In[23]:


# Generate a random index for adding a question to DB
def gen_test_index():
    length = np.random.randint(10,15)
    letters_digits = string.ascii_lowercase + string.digits
    return ''.join(random.sample(letters_digits, length))


def add_question_to_db(title, body):
    global questions
    global Qs_transformed
    global Qs_sim_matrix
    
    q = title + ' ' + body
    q_p = preprocess_question(q)
    
    tag_suggestions = get_tag_suggestions(q_p)
    q_p = q_p + ' ' + ' '.join(tag_suggestions)
    
    print(q_p)  if debug else None
    
    author_id = 1 # special if for test ( doesn't exist in DB )
    index = gen_test_index()
    questions = questions.append(pd.Series({'author_id': author_id,'date_added': pd.to_datetime('now'), 
                                                  'title': title,
                                                  'body': body, 
                                                  'processed': q_p, 
                                                  'count_answers': 0}, name=index))
    print('Qs Transformed before', qs_transformed.shape) if debug else None
    q_transformed = transformer_model.transform(tfidf_vectorizer.transform([q_p]))
    Qs_transformed = np.append(Qs_transformed, [Qs_transformed[0]], axis=0)
    print('Qs Transformed after', Qs_transformed.shape)  if debug else None
    
    sim_mat_shape = Qs_sim_matrix.shape
    print('Similarity Matrix shape before', sim_mat_shape)  if debug else None
    new_sims = cosine_similarity(Qs_transformed[-1].reshape(1,-1),Qs_transformed)[0]
    print('new_sims', new_sims.shape)  if debug else None
    Qs_sim_matrix = np.hstack((Qs_sim_matrix, np.zeros((sim_mat_shape[0], 1))))
    Qs_sim_matrix = np.vstack((Qs_sim_matrix, np.zeros((sim_mat_shape[0]+1))))
    Qs_sim_matrix[-1] = new_sims
    Qs_sim_matrix[:, -1] = new_sims
    print('Similarity Matrix shape before', sim_mat_shape)  if debug else None
    print('Question Added to DB.')  if debug else None
    return index


# # Evaluating the Recommender System using a Time Machine ! <a class="anchor" id="eval"></a>
# 
# Recommender Systems are trickier to evaluate than - for example - a machine learning classifier. The reason is that the current recommender system influences the data that we have at hand ( that we are using for training and testing ).
# 
# However, we can still get a idea about the performance of the model we're testing if we choose the good metrics and methodology for our project.
# 
# 
# We can distinguish between two types of evaluation: **Offline & Online Evaluation**.
# 
# *The Offline Evaluation* is used before deploying the model, to check its accuracy on the existing data. However, this method is **NOT** enough.
# Usually, only it's only after the model is deployed, and A/B tests are that we can draw conclusions about the actual performance of our system. This second step is referred to as *Online Evaluation*
# 
# <span style="color: purple">Another important thing about having a fixed offline evaluation framework, even though it's not accurate, is that we can use it as basis evaluate multiple recommender systems, or to fine-tune parameters related to our model, and observe how the results change ( before pushing the changes to prod ) .</span>
# 
# 
# 
# 
# ## Methodology
# 
# For offline evaluation, I chose to use a real-world **Split Validation** method. It works as follows:
# 
# * First, I extract a small amount of data ( the most recent ) for example the last 6 months (let's say October 2018 ~ November 2018), this will be the **Test Set**
# * The model is built using only the other part of the data ( the oldest ), this is the **Traning Set** .
# * Edit all data so that we can simulate exactly how the system was at that particular point in time : questions, answers and professionals that were added after the date ( July 2018 ) are removed.
# 
# Now that the system looks exactly as  it was (let's say at October 2018), the tests occur weekly:
# - Each week, new questions, answers and professionals are added. 
# - **Hide all answers that occured that week from the recommender.**
# - The recommender system makes predictions about:
#     * **the professionals who answered the questions: Given that a question "Q" was answered this week, use Question-to-Professionals mode and see if the recommendations generated for this question contain the professionals who actually answered the question. ( do that for all questions answered this week and each week of the test period )**
#     * **the questions that were answered by the professionals: Given that a professional "P" answered some question(s), use Professional-to-Questions mode and see if the recommendations generated for that professional contain the questions that the professional actually answered. ( do that for all professionals who posted answers this week and each week of the test period )**
# - The goal is to make as many accurate recommendations as possible. Meaning, send recommendations to professionals who actually answered the questions that week, or recommend the answered questions to the right professionals.
# - Note here, that the recommender can perform much better in production, because when evaluating it here, we don't account for potential answers coming as a result of the recommender itself.
# - The test system makes the following assumption: **if the model made accurate offline recommendations about the answered questions, it would do so for the unanswered ones in production.**
# 
# 
# The following code snippet builds our *"Time Machine"* :

# In[24]:


# Backup
questions_full = questions.copy()
answers_full = answers.copy()
professionals_full = professionals.copy()
tag_users_full = tag_users.copy()
tag_questions_full = tag_questions.copy()

def run_time_machine(today_str):
    global today
    global min_date_for_questions
    global min_date_for_answers
    global professionals
    global questions
    global answers
    global tag_users
    global tag_questions
    global tfidf_vectorizer
    global transformer_model
    global Qs_transformed
    global Qs_sim_matrix

    
    first_date = pd.to_datetime('2012-01-01') 

    today, min_date_for_questions, min_date_for_answers = set_today(today_str)
    print('Running Time Machine ....', 'Going to', today.strftime('%B %d %Y'), '................\n')

    professionals = professionals_full[professionals_full['date_joined'] < today].copy()
    assert (professionals['date_joined'].max() < today), "Professionals have date_joined > today !"
    questions = questions_full[(questions_full['date_added'] > first_date) & (questions_full['date_added'] < today)].copy()
    answers = answers_full[(answers_full['date_added'] > first_date) & (answers_full['date_added'] < today) & (answers_full['question_id'].isin(questions.index.values))].copy()

    #test_questions = questions_full[(questions_full['date_added'] > today) & (questions_full['count_answers'] > 0)].copy()
    #test_answers   = answers_full[(answers_full['question_id'].isin(test_questions.index.values)) & (answers_full['author_id'].isin(professionals.index.values))].copy()
    #print(len(test_questions), 'Test questions (with at least one answer)')
    #print(len(test_answers), 'Answers were posted for the test questions (from professionals who joined before that date)')

    cnt_answers = answers.groupby('question_id').count()[['body']].rename(columns={'body': 'count_answers'})
    questions = questions.drop('count_answers', axis=1)
    questions = questions.join(cnt_answers)
    questions['count_answers'] = questions['count_answers'].fillna(0)
    questions['count_answers'] = questions['count_answers'].astype(int)


    cnt_answers = answers.groupby('author_id').count()[['question_id']].rename(columns={'question_id': 'count_answers'})
    professionals = professionals.drop('count_answers', axis=1)
    professionals = professionals.join(cnt_answers)
    professionals['count_answers'] = professionals['count_answers'].fillna(0)
    professionals['count_answers'] = professionals['count_answers'].astype(int)

    # Create Last Answer Date Column
    professionals = professionals.drop('last_answer_date', axis=1)
    professionals = professionals.join(answers[['author_id', 'date_added']].groupby('author_id').max().rename(columns={'date_added': 'last_answer_date'}))


    tag_users = tag_users_full[tag_users_full['user_id'].isin(professionals.index.values)].copy()
    tag_questions = tag_questions_full[tag_questions_full['question_id'].isin(questions.index.values)].copy()

    transformer_model, Qs_transformed, Qs_sim_matrix = build_model(questions)
    gc.collect()
    print('##################################\n')


# Running this *"Time Machine"* is slow. This is because to simulate the exact state of the system at a particular date (each week) I rebuild the model each time ( I tried adding just what changed, but it was not better ). I chose a relatively short period for the test ( 2018-08-01 ~ 2018-08-22 ).
# 

# In[25]:


# Start Date and End Date of the Simulation

test_start_date = pd.to_datetime('2018-08-01')
test_end_date = pd.to_datetime('2018-08-22')
nbr_weeks = math.floor((test_end_date - test_start_date)/ np.timedelta64(1, 'W'))


# ## Parameters of the model:
# 
# We can fine-tune the following parameters:
# * **Number of recommendations to generate each time**: 5 for Professional -> Questions Mode and 10 for Question -> Professionals Mode
# * **Number of days without answers to consider a professional not active ( for selection of professionals to answer a question )**: 60 days
# * **Exploitation Intensity ($\alpha$)**: 1.35

# In[26]:



# Number of recommendations to generate for each professional. ( for the other mode, Question->Professional, it's nbr_recs*2)
nbr_recs_pro_to_qs = 5
nbr_recs_q_to_pros = nbr_recs_pro_to_qs * 2

# Exploitation Intensity ( 1.0 ~ 1.7 )
alpha_arg=1.35

# Number of days to consider professional as inactive
inactivity_period_arg=60


# In[27]:


## %%time

print('Start First Simulation', test_start_date.strftime('%B %d, %Y'), '--->', test_end_date.strftime('%B %d, %Y'), '(', nbr_weeks, ' weeks )', '\n')
start = time.time()
#run_time_machine(test_start_date.strftime('%Y-%m-%d'))


question_to_pros_recs = {}
pro_to_questions_recs = {}

nbr_accurate_q_to_pros = 0
nbr_accurate_pro_to_qs = 0
nbr_all_q_to_pros = 0
nbr_all_pro_to_qs = 0


correct_question_ids = set([])
all_question_ids = set([])

today = test_start_date
for i in range(1, nbr_weeks+1):
    print('\n----------- ', 'Week', i, ' -----------')
    d_old = today
    run_time_machine((today + np.timedelta64(1, 'W')).strftime('%Y-%m-%d'))

    week_questions= questions[(questions['date_added'] > d_old) & (questions['date_added'] < today)]
    week_answers = answers[(answers['date_added'] > d_old) & (answers['date_added'] < today) & (answers['author_id'].isin(professionals.index.values))].copy()
    
    if len(week_answers) == 0:
        continue
    
    target_questions = week_questions[week_questions.index.isin(week_answers)]
    qs_answered_this_week = list(week_answers['question_id'].unique())
    authors_this_week = week_answers['author_id'].unique()
    all_question_ids.update(qs_answered_this_week)

    print( d_old, ' ~ ', today, ' - Number of answers: ', len(week_answers), ' - Number answered questions: ', len(qs_answered_this_week), ' - Number authors: ', len(authors_this_week))
    
    # Hide answers from the system !!!
    answers = answers.drop(week_answers.index)
    for auth_id in authors_this_week:
        professionals.at[auth_id, 'count_answers'] = len(answers[answers['author_id'] == auth_id])

    # some tests to check if everything is ok
    assert (len(answers[(answers['date_added'] < today) & (answers['date_added'] > d_old) & (answers['author_id'].isin(professionals.index.values)) ]) == 0), "The answers of this week were not all removed"
    random_auth_id = authors_this_week[0]
    assert (professionals.loc[random_auth_id, 'count_answers'] == len(answers[answers['author_id'] == random_auth_id])), "Problem with count of answers for professional"

    print('Making Predictions for the week\'s answered questions ...')
    # Predict pros for questions that were answered ( Question->Pros )
    for qid in qs_answered_this_week:
        question_to_pros_recs[qid] = recommend_professionals_for_question(qid, nbr_recommendations=nbr_recs_q_to_pros, inactivity_period=inactivity_period_arg)
        recommended_pro_ids = set(question_to_pros_recs[qid].index.values)
        nbr_all_q_to_pros += len(recommended_pro_ids)
        target_pro_ids = set(week_answers[week_answers['question_id'] == qid]['author_id'].unique())
        union_len = len(target_pro_ids.union(recommended_pro_ids))
        sum_len = len(recommended_pro_ids) + len(target_pro_ids)
        if union_len < sum_len:
            nbr_accurate_q_to_pros += (sum_len - union_len)
            correct_question_ids.update([qid])
            
    # Predict questions for pros who answered ( Pro->Questions )
    for auth_id in authors_this_week:
        pro_to_questions_recs[auth_id] = recommend_questions_to_professional(auth_id, nbr_recommendations=nbr_recs_pro_to_qs, silent=True, alpha=alpha_arg)
        recommended_question_ids = set(pro_to_questions_recs[auth_id].index.values)
        nbr_all_pro_to_qs += len(recommended_question_ids)
        target_question_ids = set(week_answers[week_answers['author_id'] == auth_id]['question_id'].unique())
        union_len = len(target_question_ids.union(recommended_question_ids))
        sum_len = len(recommended_question_ids) + len(target_question_ids)
        if union_len < sum_len:
            nbr_accurate_pro_to_qs += (sum_len - union_len)
            correct_question_ids.update([e for e in recommended_question_ids if e in target_question_ids])
    
    #print('Number of Accurate Recommendations (Question -> Pros): ', nbr_accurate_q_to_pros)
    #print('Number of Accurate Recommendations (Pro -> Questions): ', nbr_accurate_pro_to_qs)

end = time.time()
print(f'\n-------- End of Simulation ({(end-start)/60:.2f} minutes) --------')


# ## Metrics & Results of the Evaluation
# 
# 
# Two principal metrics are used:
# * **Proportion of answered questions got right**: meaning, out of all the questions that were answered, how many of them did we get right when making recommendations
# * **Proportion of accurate recommendations**: out of all recommendations made, how many did we get right
# 
# Now, the proposed recommender system is compared to the legacy-one. Let's check the same metrics for recommendations made by the previous legacy-system in the same test period.
# 
# *<span style="color: red;">REMINDER:</span>* **This evaluation is actually not fair !** Because as mentioned before, the legacy recommender is actually influencing the answers we have at hand. **But ouf of curiosity**, we want to check how many accurate recommendations the proposed system is able to make, even without being deployed ! This must be taken into consideration when looking at the following numbers.
# 
# Furthermore, many answers are just the product of users visiting the site "organically", and not the product of any recommendations.
# 

# In[28]:


print('Results of the Test:')
print(f"- Percentage of Answered Questions that got accurate recommendations: {len(correct_question_ids)/len(all_question_ids)*100:.2f}%", f'( {len(correct_question_ids)} out of {len(all_question_ids)} questions  )' )
print('\n- Percentage of accurate recommendations ( out of all sent ones ):')
print(f'\t- Question-to-Professionals Mode:  {(nbr_accurate_q_to_pros/nbr_all_q_to_pros)*100:.2f}% ',  f'( {nbr_accurate_q_to_pros} out of {nbr_all_q_to_pros} recommendations were accurate )')
print(f'\t- Professional-to-Questions Mode: {(nbr_accurate_pro_to_qs/nbr_all_pro_to_qs)*100:.2f}% ', f'( {nbr_accurate_pro_to_qs} out of {nbr_all_pro_to_qs} recommendations were accurate )')


# ## The only True Evaluation
# 
# The only True Evaluation is done **online** with **good A/B Tests**, and with **measuring the right metrics**. I list here some metrics that should be considered:
# * **Number of active professionals**: we decide that a professional is active at time t if he answered a question in the last n days ( n = 100 in implementation )
# * **Mean Time-to-First-Answer**: $\frac{1}{nbr \thinspace questions}\sum_{q}^{questions}{nbr \thinspace days \thinspace between \thinspace question \thinspace q \thinspace was \thinspace posted \thinspace and \thinspace its \thinspace first \thinspace answer}$
# * **Number of accurate recommendations**
# * **Percentage of accurate recommendations**: Ratio $\frac{Number \thinspace of \thinspace accurate \thinspace recommendations}{Number \thinspace of \thinspace all \thinspace recommendations}$. The higher the ratio, the better. It will also help avoid churning.
# 
# Running A/B Tests on the parameters of the model will help to find the best combination of values that maximizes the metrics.

# # Summary and Future Explorations <a class="anchor" id="future"></a>
# 
# The proposed system's strengths are **its effectiveness, ease of implementation and ease of maintainance in production.**
# It uses controlled randomness to encourage new users while keeping engaged professionals in the platform.
# 
# The proposed system is designed to be very **resilient** when it comes to difficult cases like professionals with **various interests**. In addition to recommending questions based on semantic similarity, the proposed system also recommends relevant questions from tags, and from **tags which are textually similar to the followed ones** (eg. 'computer-science' and 'computerscience', 'information-technology' and 'informationtechnology' ) .
# 
# Further, a basic framework for evaluating the system was proposed along with the most important metrics to measure.
# 
# Some recommendations for future improvements:
# 
# * **Meta-data based Recommendations:** 
# 
# It is possible to use valuable information obtained when users open a browsing session ( viewed and visited questions, ... ).
# 
# * **Controlling Tags:**
# 
# It is important to preprocess tags and prevent the students from using tags which already exist. A powerful model possibly based on Word2Vec, could be used to model the relationships and similarities between tags. For example, capturing the similarity between 'information-technology' and 'IT' would hugely boost the performance of the recommender.
# 
# * **Controlling Typos**:
# 
# Using a spell-checking engine like [Hunspell](https://pypi.org/project/hunspell/) would increase the quality of the data and help the engine make better recommendations
# 
# * **Using the answers and comments in the model**:
# 
# In this model, only the text of the questions was used when encoding the questions. Another model should be explored, where we also make use of the answers and comments of each question to encode the questions.
# 
# * **Making use of upvotes**:
# 
# Unfortunately, The data provided about upvotes (questions & answers) wasn't specific as to who upvoted what. If this data was available, it could be used to better understand the interests of both professionals and students, and thus making better predictions.
# 
# * **Using other Exploration Techniques**:
# 
# I decided to go with the $\epsilon$ Greedy Algorithm for its simpleness and ease of use. Other algorithms can be further explored like "Optismism in face of uncertainty" and "Probability Matching" ([See Links for more](#links))
# 
# * **Making use of serious Reinforcement Learning**
# 
# Finally, after exploring the previous suggestions, a very insteresting project would be to to make use of RL techniques, where the recommender uses the feedback of its actions (recommendations) to update and improve its future behaviour (recommendations). [See Links for more](#links)

# <h4>I hope that this Kernel was useful, and see you in the <a href="https://www.kaggle.com/hamzael1/kernels" target="_blank">next one</a> !</h4>
# 
# *PS: upvotes & feedback are welcome !*
# 

# # Links to useful Ressources: <a class="anchor" id="links"></a>
# <h4>About Recommender systems</h4>
# * [A whole Coursera Specialization about Recommender Systems](https://www.coursera.org/specializations/recommender-systems)
# * [Recommender Systems in Practice](https://towardsdatascience.com/recommender-systems-in-practice-cef9033bb23a)
# * [An amazing Playlist of Stanford University Videos about Mining Datasets](https://www.youtube.com/watch?v=1JRrCEgiyHM&list=PLLssT5z_DsK9JDLcT8T62VtzwyW9LNepV&index=42&t=0s)
# 
# <h4>About Epsilon Greedy and Exploration/Exploitation</h4>
# * [The Multi-armed Bandit Problem](https://en.wikipedia.org/wiki/Multi-armed_bandit)
# * [A nice article About Epsilon-Greedy Algorithm](https://imaddabbura.github.io/post/epsilon_greedy_algorithm/)
# * [Some great slides about how to solve the Exploration/Exploitation dilemma](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/XX.pdf)
# 
# <h4>About Latent Semantic Analysis and SVD</h4>
# * [A nice short video about LSA](https://www.youtube.com/watch?v=OvzJiur55vo)
# * [A great explanation of SVD by Professor Jure Leskovec](https://www.youtube.com/watch?v=P5mlg91as1c&list=PLLssT5z_DsK9JDLcT8T62VtzwyW9LNepV&index=47)
# 
# <h4>About Reinforcement Learning in Recommender Systems</h4>
# * [Reinforcement Learning for Recommender Systems: A Case Study on Youtube](https://www.youtube.com/watch?v=HEqQ2_1XRTs)
# * [Contextualized Bandits for Recommendation Systems](https://towardsdatascience.com/bandits-for-recommender-system-optimization-1d702662346e)
# * [Netflix using using Contextualized Bandits for personalizing the selection of artwork](https://medium.com/netflix-techblog/artwork-personalization-c589f074ad76)
# * [A Multi-Armed Bandit Framework for Recommendations at Netflix](https://www.youtube.com/watch?v=kY-BCNHd_dM)

# In[ ]:




