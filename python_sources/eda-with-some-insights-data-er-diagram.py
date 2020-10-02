#!/usr/bin/env python
# coding: utf-8

# <h1><center><font size="6">EDA of CareerVillage.org data</font></center></h1>
# 
# <h2><center><font size="4">Dataset used: Data Science for Good: CareerVillage.org</font></center></h2>
# 
# <img src="https://www.ffwd.org/wp-content/uploads/CareerVillage-logo.png" width="500"></img>
# 
# <br>
# 
# # <a id='0'>Content</a>
# 
# - <a href='#1'>Introduction</a>
# - <a href='#2'>All imports necessary</a>
# - <a href='#3'>A bit of configuration</a>
# - <a href='#4'>Auxiliary methods</a>
# - <a href='#5'>List all files available</a>
# - <a href='#6'>Read the data</a>
# - <a href='#7'>Quickly go through the files</a>
#     - <a href='#71'>professionals</a>
#     - <a href='#72'>tag_users</a>
#     - <a href='#73'>students</a>
#     - <a href='#74'>tag_questions</a>
#     - <a href='#75'>groups</a>
#     - <a href='#76'>emails</a>
#     - <a href='#77'>group_memberships</a>
#     - <a href='#78'>answers</a>
#     - <a href='#79'>comments</a>
#     - <a href='#710'>matches</a>
#     - <a href='#711'>tags</a>
#     - <a href='#712'>questions</a>
#     - <a href='#713'>school_memberships</a>
# - <a href='#8'>Data quality</a>
#     - <a href='#81'>Null-values, unique values counts across tables</a>
#         - <a href='#811'>professionals</a>
#         - <a href='#812'>tag_users</a>
#         - <a href='#813'>students</a>
#         - <a href='#814'>tag_questions</a>
#         - <a href='#815'>groups</a>
#         - <a href='#816'>emails</a>
#         - <a href='#817'>group_memberships</a>
#         - <a href='#818'>answers</a>
#         - <a href='#819'>comments</a>
#         - <a href='#8110'>matches</a>
#         - <a href='#8111'>tags</a>
#         - <a href='#8112'>questions</a>
#         - <a href='#8113'>school_memberships</a>
#     - <a href='#82'>What columns in what tables?</a>
#     - <a href='#83'>Tables mapping</a>
#         - <a href='#831'>Students intersected with Questions by author_id</a>
#         - <a href='#832'>Professionals intersected with Questions by author_id</a>
#         - <a href='#833'>Professionals intersected with Answers by author_id</a>
#         - <a href='#834'>Students intersected with Answers by author_id</a>
#         - <a href='#835'>Questions intersected with Answers by question_id</a>
#         - <a href='#836'>Questions intersected with Comments by question_id</a>
#         - <a href='#837'>Answers intersected with Comments by answer_id</a>
#         - <a href='#838'>Professionals intersected with Comments by author_id</a>
#         - <a href='#839'>Students intersected with Comments by author_id</a>
#         - <a href='#8310'>Students intersected with Group_memberships by user_id</a>
#         - <a href='#8311'>Professionals intersected with Group_memberships by user_id</a>
#         - <a href='#8312'>Students intersected with School_memberships by user_id</a>
#         - <a href='#8313'>Professionals intersected with School_memberships by user_id</a>
#         - <a href='#8314'>Students intersected with Tag_users by user_id</a>
#         - <a href='#8315'>Professionals intersected with Tag_users by user_id</a>
#         - <a href='#8316'>Questions intersected with Tag_questions by question_id</a>
#         - <a href='#8317'>Students intersected with Emails by recipient_id</a>
#         - <a href='#8318'>Professionals intersected with Emails by recipient_id</a>
#         - <a href='#8319'>Emails intersected with Matches by email_id</a>
#         - <a href='#8320'>Questions intersected with Matches by question_id</a>
#     - <a href='#84'>ER-diagram of data</a>
# - <a href='#9'>Deeper analysis</a>
#     - <a href='#91'>Cumulative community growth</a>
#         - <a href='#911'>Yearly</a>
#         - <a href='#912'>Monthly</a>
#         - <a href='#913'>Weekly</a>
#         - <a href='#914'>Daily</a>
#     - <a href='#92'>Community growth dynamic</a>
#         - <a href='#921'>Yearly</a>
#         - <a href='#922'>Monthly</a>
#         - <a href='#923'>Weekly</a>
#         - <a href='#924'>Daily</a>
#     - <a href='#93'>Cumulative questions/answers/comments growth</a>
#         - <a href='#931'>Yearly</a>
#         - <a href='#932'>Monthly</a>
#         - <a href='#933'>Weekly</a>
#         - <a href='#934'>Daily</a>
#     - <a href='#94'>Questions/answers/comments growth dynamic</a>
#         - <a href='#941'>Yearly</a>
#         - <a href='#942'>Monthly</a>
#         - <a href='#943'>Weekly</a>
#         - <a href='#944'>Daily</a>
#     - <a href='#95'>Professionals</a>
#         - <a href='#951'>Professionals by answered_questions/asked_questions/wrote_comments flags</a>
#         - <a href='#952'>Distribution of professionals by questions/answers/comments count</a>
#         - <a href='#952'>Professionals locations, industries, headlines</a>
#     - <a href='#96'>Students</a>
#         - <a href='#961'>Students by answered_questions/asked_questions/wrote_comments flags</a>
#         - <a href='#962'>Students locations</a>
#     - <a href='#97'>Tags</a>
#     - <a href='#98'>Emails</a>
#     - <a href='#99'>Matches</a>

# # <a id='1'>Introduction</a> [<a href='#0'>back to content</a>]

# Hello everyone!
# 
# This notebook is a general attempt to tell about the data that CareerVillage has provided for analysis.
# 
# Here is description of tables that are available:
# 
# - **answers.csv:** Answers are what this is all about! Answers get posted in response to questions. Answers can only be posted by users who are registered as Professionals. However, if someone has changed their registration type after joining, they may show up as the author of an Answer even if they are no longer a Professional.
# 
# - **comments.csv:** Comments can be made on Answers or Questions. We refer to whichever the comment is posted to as the "parent" of that comment. Comments can be posted by any type of user. Our favorite comments tend to have "Thank you" in them :)
# 
# - **emails.csv:** Each email corresponds to one specific email to one specific recipient. The frequency_level refers to the type of email template which includes immediate emails sent right after a question is asked, daily digests, and weekly digests.
# 
# - **group_memberships.csv:** Any type of user can join any group. There are only a handful of groups so far.
# 
# - **groups.csv:** Each group has a "type". For privacy reasons we have to leave the group names off.
# 
# - **matches.csv:** Each row tells you which questions were included in emails. If an email contains only one question, that email's ID will show up here only once. If an email contains 10 questions, that email's ID would show up here 10 times.
# 
# - **professionals.csv:** We call our volunteers "Professionals", but we might as well call them Superheroes. They're the grown ups who volunteer their time to answer questions on the site.
# 
# - **questions.csv:** Questions get posted by students. Sometimes they're very advanced. Sometimes they're just getting started. It's all fair game, as long as it's relevant to the student's future professional success.
# 
# - **school_memberships.csv:** Just like group_memberships, but for schools instead.
# 
# - **students.csv:** Students are the most important people on CareerVillage.org. They tend to range in age from about 14 to 24. They're all over the world, and they're the reason we exist!
# 
# - **tag_questions.csv:** Every question can be hashtagged. We track the hashtag-to-question pairings, and put them into this file.
# 
# - **tag_users.csv:** Users of any type can follow a hashtag. This shows you which hashtags each user follows.
# 
# - **tags.csv:** Each tag gets a name.

# # <a id='2'>All imports necessary</a> [<a href='#0'>back to content</a>]

# In[ ]:


import numpy as np

import pandas as pd

import warnings

import os

import seaborn as sns

import matplotlib.pyplot as plt

import wordcloud

import matplotlib_venn

import matplotlib.image as mpimg


# # <a id='3'>A bit of configuration</a> [<a href='#0'>back to content</a>]

# In[ ]:


warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')

pd.options.display.max_rows = 10000


# # <a id='4'>Auxiliary methods</a> [<a href='#0'>back to content</a>]

# In[ ]:


def plot_null_and_unique_values(df):
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(30, 10))
    columns = df.columns
    sns.set(font_scale=1.5)
    res = pd.DataFrame(
        {
            'unique_counts': df[columns].nunique(),
            'null_counts': df[columns].isnull().sum()
        }
    )
    res.sort_values(by='unique_counts', ascending=False, inplace=True)
    sns.barplot(
        y=res.index,
        x=res['unique_counts'].values,
        orient='h',
        ax=ax1
    )
    sns.barplot(
        y=res.index,
        x=res['null_counts'].values,
        orient='h',
        ax=ax2
    )
    ax1.axvline(x=len(df), color='red')
    ax2.axvline(x=len(df), color='red')
    plt.suptitle(
        'The general look of columns\n\
        (vertical red line shows the number of records in the dataset)'
    )
    ax1.set_title('The number of unique values per column')
    ax2.set_title('The number of null values per column')
    plt.show()


# In[ ]:


def plot_intersections(
    df1,
    df2,
    col1,
    col2,
    venn_label_1,
    venn_label_2,
    venn_title_1,
    pie_label_1,
    pie_label_2,
    pie_label_3,
    pie_title
):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    matplotlib_venn.venn2(
        subsets=[set(df1[col1]), set(df2[col2])],
        set_labels=[venn_label_1, venn_label_2],
        ax=ax1
    )
    ax1.set_title(venn_title_1)
    ax2.pie(
        x=[
            len(set(df1[col1]) - set(df2[col2])),
            len(set(df1[col1]) & set(df2[col2])),
            len(set(df2[col2]) - set(df1[col1]))
        ],
        labels=[
            pie_label_1,
            pie_label_2,
            pie_label_3
        ],
        autopct='%1.2f%%'
    )
    ax2.set_title(pie_title)
    plt.tight_layout()
    plt.show()


# In[ ]:


def get_meta_info_about_columns_and_tables(df_arr, df_name_arr):
    tables = []
    columns = []
    for df, name in zip(df_arr, df_name_arr):
        columns.extend(df.columns.values)
        tables.extend([name] * len(df.columns))
    return pd.DataFrame({'table': tables, 'column': columns})


# In[ ]:


def remove_table_name_prefixes(df_arr, df_name_arr):
    for df, name in zip(df_arr, df_name_arr):
        new_columns_mapping = dict(zip(df.columns, list(map(lambda column: column.replace('{}_'.format(name), ''), df.columns))))
        df.rename(new_columns_mapping, axis=1, inplace=True)


# In[ ]:


def extract_year_month_week_day_hour(df, datetime_column):
    df[datetime_column] = df[datetime_column].astype('datetime64')
    df['year'] = df[datetime_column].dt.year
    df['month'] = df[datetime_column].dt.month
    df['week'] = df[datetime_column].dt.week
    df['day'] = df[datetime_column].dt.day
    df['hour'] = df[datetime_column].dt.hour


# In[ ]:


def calculate_yearly_cumsum(df, col_name):
    df_yearly_cumcount = df.groupby('year')['id'].count().reset_index()
    df_yearly_cumcount.id = df_yearly_cumcount.id.cumsum()
    df_yearly_cumcount.rename({'id': '{}_count'.format(col_name)}, axis=1, inplace=True)
    df_yearly_cumcount.set_index('year', inplace=True)
    return df_yearly_cumcount


# In[ ]:


def calculate_monthly_cumsum(df, col_name):
    df_monthly_cumcount = df.groupby(['year', 'month'])['id'].count().reset_index()
    df_monthly_cumcount.sort_values(by=['year', 'month'], inplace=True)
    df_monthly_cumcount.id = df_monthly_cumcount.id.cumsum()
    df_monthly_cumcount.rename({'id': '{}_count'.format(col_name)}, axis=1, inplace=True)
    return df_monthly_cumcount


# In[ ]:


def calculate_weekly_cumsum(df, col_name):
    df_weekly_cumcount = df.groupby(['year', 'week'])['id'].count().reset_index()
    df_weekly_cumcount.sort_values(by=['year', 'week'], inplace=True)
    df_weekly_cumcount.id = df_weekly_cumcount.id.cumsum()
    df_weekly_cumcount.rename({'id': '{}_count'.format(col_name)}, axis=1, inplace=True)
    return df_weekly_cumcount


# In[ ]:


def calculate_daily_cumsum(df, col_name):
    df_daily_cumcount = df.groupby(['year', 'month', 'day'])['id'].count().reset_index()
    df_daily_cumcount.sort_values(by=['year', 'month', 'day'], inplace=True)
    df_daily_cumcount.id = df_daily_cumcount.id.cumsum()
    df_daily_cumcount.rename({'id': '{}_count'.format(col_name)}, axis=1, inplace=True)
    return df_daily_cumcount


# In[ ]:


def calculate_yearly_count(df, col_name):
    df_yearly_cumcount = df.groupby('year')['id'].count().reset_index()
    df_yearly_cumcount.rename({'id': '{}_count'.format(col_name)}, axis=1, inplace=True)
    df_yearly_cumcount.set_index('year', inplace=True)
    return df_yearly_cumcount


# In[ ]:


def calculate_monthly_count(df, col_name):
    df_monthly_cumcount = df.groupby(['year', 'month'])['id'].count().reset_index()
    df_monthly_cumcount.sort_values(by=['year', 'month'], inplace=True)
    df_monthly_cumcount.rename({'id': '{}_count'.format(col_name)}, axis=1, inplace=True)
    return df_monthly_cumcount


# In[ ]:


def calculate_weekly_count(df, col_name):
    df_weekly_cumcount = df.groupby(['year', 'week'])['id'].count().reset_index()
    df_weekly_cumcount.sort_values(by=['year', 'week'], inplace=True)
    df_weekly_cumcount.rename({'id': '{}_count'.format(col_name)}, axis=1, inplace=True)
    return df_weekly_cumcount


# In[ ]:


def calculate_daily_count(df, col_name):
    df_daily_cumcount = df.groupby(['year', 'month', 'day'])['id'].count().reset_index()
    df_daily_cumcount.sort_values(by=['year', 'month', 'day'], inplace=True)
    df_daily_cumcount.rename({'id': '{}_count'.format(col_name)}, axis=1, inplace=True)
    return df_daily_cumcount


# # <a id='5'>List all files available</a> [<a href='#0'>back to content</a>]

# In[ ]:


data_folder = "../input/data-science-for-good-careervillage/"


# In[ ]:


for file in os.listdir(data_folder):
    print(file)


# # <a id='6'>Read the data</a> [<a href='#0'>back to content</a>]

# In[ ]:


professionals = pd.read_csv(os.path.join(data_folder, 'professionals.csv'))
tag_users = pd.read_csv(os.path.join(data_folder, 'tag_users.csv'))
students = pd.read_csv(os.path.join(data_folder, 'students.csv'))
tag_questions = pd.read_csv(os.path.join(data_folder, 'tag_questions.csv'))
groups = pd.read_csv(os.path.join(data_folder, 'groups.csv'))
emails = pd.read_csv(os.path.join(data_folder, 'emails.csv'))
group_memberships = pd.read_csv(os.path.join(data_folder, 'group_memberships.csv'))
answers = pd.read_csv(os.path.join(data_folder, 'answers.csv'))
comments = pd.read_csv(os.path.join(data_folder, 'comments.csv'))
matches = pd.read_csv(os.path.join(data_folder, 'matches.csv'))
tags = pd.read_csv(os.path.join(data_folder, 'tags.csv'))
questions = pd.read_csv(os.path.join(data_folder, 'questions.csv'))
school_memberships = pd.read_csv(os.path.join(data_folder, 'school_memberships.csv'))


# # <a id='7'>Quickly go through the files</a> [<a href='#0'>back to content</a>]

# ## <a id='71'>professionals</a> [<a href='#0'>back to content</a>]

# In[ ]:


professionals.head()


# In[ ]:


professionals.info(verbose=True, null_counts=True)


# ## <a id='72'>tag_users</a> [<a href='#0'>back to content</a>]

# In[ ]:


tag_users.head()


# In[ ]:


tag_users.info(verbose=True, null_counts=True)


# ## <a id='73'>students</a> [<a href='#0'>back to content</a>]

# In[ ]:


students.head()


# In[ ]:


students.info(verbose=True, null_counts=True)


# ## <a id='74'>tag_questions</a> [<a href='#0'>back to content</a>]

# In[ ]:


tag_questions.head()


# In[ ]:


tag_questions.info(verbose=True, null_counts=True)


# ## <a id='75'>groups</a> [<a href='#0'>back to content</a>]

# In[ ]:


groups.head()


# In[ ]:


groups.info(verbose=True, null_counts=True)


# ## <a id='76'>emails</a> [<a href='#0'>back to content</a>]

# In[ ]:


emails.head()


# In[ ]:


emails.info(verbose=True, null_counts=True)


# ## <a id='77'>group_memberships</a> [<a href='#0'>back to content</a>]

# In[ ]:


group_memberships.head()


# In[ ]:


group_memberships.info(verbose=True, null_counts=True)


# ## <a id='78'>answers</a> [<a href='#0'>back to content</a>]

# In[ ]:


answers.head()


# In[ ]:


answers.info(verbose=True, null_counts=True)


# ## <a id='79'>comments</a> [<a href='#0'>back to content</a>]

# In[ ]:


comments.head()


# In[ ]:


comments.info(verbose=True, null_counts=True)


# ## <a id='710'>matches</a> [<a href='#0'>back to content</a>]

# In[ ]:


matches.head()


# In[ ]:


matches.info(verbose=True, null_counts=True)


# ## <a id='711'>tags</a> [<a href='#0'>back to content</a>]

# In[ ]:


tags.head()


# In[ ]:


tags.info(verbose=True, null_counts=True)


# ## <a id='712'>questions</a> [<a href='#0'>back to content</a>]

# In[ ]:


questions.head()


# In[ ]:


questions.info(verbose=True, null_counts=True)


# ## <a id='713'>school_memberships</a> [<a href='#0'>back to content</a>]

# In[ ]:


school_memberships.head()


# In[ ]:


school_memberships.info(verbose=True, null_counts=True)


# # <a id='8'>Data quality</a> [<a href='#0'>back to content</a>]

# ## <a id='81'>Null-values, unique values counts across tables</a> [<a href='#0'>back to content</a>]

# Here is the list of tables:
# - professionals;
# - tag_users;
# - students;
# - tag_questions;
# - groups;
# - emails;
# - group_memberships;
# - answers;
# - comments;
# - matches;
# - tags;
# - questions;
# - school_memberships.

# ### <a id='811'>professionals</a> [<a href='#0'>back to content</a>]

# In[ ]:


plot_null_and_unique_values(professionals)


# ### <a id='812'>tag_users</a> [<a href='#0'>back to content</a>]

# In[ ]:


plot_null_and_unique_values(tag_users)


# ### <a id='813'>students</a> [<a href='#0'>back to content</a>]

# In[ ]:


plot_null_and_unique_values(students)


# ### <a id='814'>tag_questions</a> [<a href='#0'>back to content</a>]

# In[ ]:


plot_null_and_unique_values(tag_questions)


# ### <a id='815'>groups</a> [<a href='#0'>back to content</a>]

# In[ ]:


plot_null_and_unique_values(groups)


# ### <a id='816'>emails</a> [<a href='#0'>back to content</a>]

# In[ ]:


plot_null_and_unique_values(emails)


# ### <a id='817'>group_memberships</a> [<a href='#0'>back to content</a>]

# In[ ]:


plot_null_and_unique_values(group_memberships)


# ### <a id='818'>answers</a> [<a href='#0'>back to content</a>]

# In[ ]:


plot_null_and_unique_values(answers)


# ### <a id='819'>comments</a> [<a href='#0'>back to content</a>]

# In[ ]:


plot_null_and_unique_values(comments)


# ### <a id='8110'>matches</a> [<a href='#0'>back to content</a>]

# In[ ]:


plot_null_and_unique_values(matches)


# ### <a id='8111'>tags</a> [<a href='#0'>back to content</a>]

# In[ ]:


plot_null_and_unique_values(tags)


# ### <a id='8112'>questions</a> [<a href='#0'>back to content</a>]

# In[ ]:


plot_null_and_unique_values(questions)


# ### <a id='8113'>school_memberships</a> [<a href='#0'>back to content</a>]

# In[ ]:


plot_null_and_unique_values(school_memberships)


# ## <a id='82'>What columns in what tables?</a> [<a href='#0'>back to content</a>]

# In[ ]:


tables_columns_info = get_meta_info_about_columns_and_tables(
    [
        professionals,
        tag_users,
        students,
        tag_questions,
        groups,
        emails,
        group_memberships,
        answers,
        comments,
        matches,
        tags,
        questions,
        school_memberships
    ],
    [
        'professionals',
        'tag_users',
        'students',
        'tag_questions',
        'groups',
        'emails',
        'group_memberships',
        'answers',
        'comments',
        'matches',
        'tags',
        'questions',
        'school_memberships'
    ]
)


# In[ ]:


tables_columns_info.head(10)


# In[ ]:


plt.figure(figsize=(10, 20))
sns.countplot(y=tables_columns_info.table)
plt.title('How many columns there are in different tables')
plt.show()


# In[ ]:


plt.figure(figsize=(10, 20))
sns.countplot(y=tables_columns_info.column)
plt.title('How many tables contain particular column')
plt.show()


# In[ ]:


tables_columns_info[tables_columns_info.column.str.contains('date')]


# In[ ]:


tables_columns_info[tables_columns_info.column.str.contains('_id')]


# As you have (most probably) guessed the name on each column starts with the name of the table that column belongs to.
# 
# This may be a bit confusing especially when we want to understand which tables can be merged together.
# 
# So let's remove this prefixes from column names and see how many columns will be in several tables:

# In[ ]:


remove_table_name_prefixes(
    [
        professionals,
        tag_users,
        students,
        tag_questions,
        groups,
        emails,
        group_memberships,
        answers,
        comments,
        matches,
        tags,
        questions,
        school_memberships
    ],
    [
        'professionals',
        'tag_users',
        'students',
        'tag_questions',
        'groups',
        'emails',
        'group_memberships',
        'answers',
        'comments',
        'matches',
        'tags',
        'questions',
        'school_memberships'
    ]
)


# In[ ]:


tables_columns_info = get_meta_info_about_columns_and_tables(
    [
        professionals,
        tag_users,
        students,
        tag_questions,
        groups,
        emails,
        group_memberships,
        answers,
        comments,
        matches,
        tags,
        questions,
        school_memberships
    ],
    [
        'professionals',
        'tag_users',
        'students',
        'tag_questions',
        'groups',
        'emails',
        'group_memberships',
        'answers',
        'comments',
        'matches',
        'tags',
        'questions',
        'school_memberships'
    ]
)


# In[ ]:


tables_columns_info.head(10)


# In[ ]:


plt.figure(figsize=(10, 20))
sns.countplot(y=tables_columns_info.column)
plt.title('How many tables contain particular column')
plt.show()


# Now we can see that there are some columns that can be used to merge tables and generate more sophisticated insights.

# In[ ]:


tables_columns_info[tables_columns_info.column.str.contains('id')].sort_values(by='table')


# ## <a id='83'>Tables mapping</a> [<a href='#0'>back to content</a>]

# Okay, we've got lots of tables that can be merged together but the question is: how many rows of one table can be merged with rows from another table?
# 
# For example there can be:
# - students that did not ask any question yet;
# - professionals that did not answered any question yet;
# - questions without any answers or comments;
# - answers without any comments;
# - questions without any tags;
# - users without any tags;
# - users without any group memberships;
# - users without any school memberships.
# - etc.
# 
# All these kinds of questions can give an understanding of the users activity.

# ### <a id='831'>Students intersected with Questions by author_id</a> [<a href='#0'>back to content</a>]

# In[ ]:


plot_intersections(
    students,
    questions,
    'id',
    'author_id',
    'Students',
    'Questions',
    'How many students have been writing questions',
    'Students without questions',
    'Students with questions',
    'Not students',
    'What is the percentage'
)


# ### <a id='832'>Professionals intersected with Questions by author_id</a> [<a href='#0'>back to content</a>]

# In[ ]:


plot_intersections(
    professionals,
    questions,
    'id',
    'author_id',
    'Professionals',
    'Questions',
    'How many professionals have been writing questions',
    'Professionals without questions',
    'Professionals with questions',
    'Not professionals',
    'What is the percentage'
)


# Hmm... There are some questions that were asked neither by students nor professionals.
# 
# How many?

# In[ ]:


print('Number of questions asked neither by professionals nor students: ', len(questions[
    questions.author_id.isin(
        set(questions.author_id) - (set(students.id) | set(professionals.id))
    )
]))


# ### <a id='833'>Professionals intersected with Answers by author_id</a> [<a href='#0'>back to content</a>]

# In[ ]:


plot_intersections(
    professionals,
    answers,
    'id',
    'author_id',
    'Professionals',
    'Answers',
    'How many professionals have been writing answers',
    'Professionals without answers',
    'Professionals with answers',
    'Not professionals',
    'What is the percentage'
)


# ### <a id='834'>Students intersected with Answers by author_id</a> [<a href='#0'>back to content</a>]

# In[ ]:


plot_intersections(
    students,
    answers,
    'id',
    'author_id',
    'Students',
    'Answers',
    'How many students have been writing answers',
    'Students without answers',
    'Students with answers',
    'Not students',
    'What is the percentage'
)


# And it looks like there are answers that were written neither by professionals nor students:

# In[ ]:


print('Number of answers written neither by professionals nor students: ', len(answers[
    answers.author_id.isin(
        set(answers.author_id) - (set(students.id) | set(professionals.id))
    )
]))


# Okay, what about questions?
# 
# How many questions stayed unanswered?

# ### <a id='835'>Questions intersected with Answers by question_id</a> [<a href='#0'>back to content</a>]

# In[ ]:


plot_intersections(
    questions,
    answers,
    'id',
    'question_id',
    'Questions',
    'Answers',
    'How many questions have been answered',
    'Questions without answers',
    'Questions with answers',
    'Answers without questions',
    'What is the percentage'
)


# 821 questions stayed unanswered.
# 
# Not good but at least we don't have answers without questions)))
# 
# Now let's look at comments

# ### <a id='836'>Questions intersected with Comments by question_id</a> [<a href='#0'>back to content</a>]

# In[ ]:


plot_intersections(
    questions,
    comments,
    'id',
    'parent_content_id',
    'Questions',
    'Comments',
    'How many questions have been commented',
    'Questions without comments',
    'Questions with comments',
    'Comments for answers',
    'What is the percentage'
)


# Okay, 1875 questions have been commented.
# 
# What about answers?

# ### <a id='837'>Answers intersected with Comments by answer_id</a> [<a href='#0'>back to content</a>]

# In[ ]:


plot_intersections(
    answers,
    comments,
    'id',
    'parent_content_id',
    'Answers',
    'Comments',
    'How many answers have been commented',
    'Answers without comments',
    'Answers with comments',
    'Comments for questions',
    'What is the percentage'
)


# In[ ]:


print(
    'Number of comments written neither for answers nor for questions: ',
    len(set(comments.parent_content_id) - (set(answers.id) | set(questions.id)))
)


# Well at least we do not have comments that would not be related neither to questions no answers.
# 
# Yep, by the way, who writes these comments?

# ### <a id='838'>Professionals intersected with Comments by author_id</a> [<a href='#0'>back to content</a>]

# In[ ]:


plot_intersections(
    professionals,
    comments,
    'id',
    'author_id',
    'Professionals',
    'Comments',
    'How many professionals have been writing comments',
    'Professionals without comments',
    'Professionals with comments',
    'Not professionals',
    'What is the percentage'
)


# In[ ]:


print(
    'Number of comments written by professionals: ',
    len(comments[comments.author_id.isin(set(professionals.id))])
)


# ### <a id='839'>Students intersected with Comments by author_id</a> [<a href='#0'>back to content</a>]

# In[ ]:


plot_intersections(
    students,
    comments,
    'id',
    'author_id',
    'Students',
    'Comments',
    'How many students have been writing comments',
    'Students without comments',
    'Students with comments',
    'Not students',
    'What is the percentage'
)


# In[ ]:


print(
    'Number of comments written by students: ',
    len(comments[comments.author_id.isin(set(students.id))])
)


# So students much more active in commenting which seems quite logical.

# ### <a id='8310'>Students intersected with Group_memberships by user_id</a> [<a href='#0'>back to content</a>]

# In[ ]:


plot_intersections(
    students,
    group_memberships,
    'id',
    'user_id',
    'Students',
    'Groups',
    'How many students participate in groups',
    'Students without groups',
    'Students with groups',
    'Not students',
    'What is the percentage'
)


# ### <a id='8311'>Professionals intersected with Group_memberships by user_id</a> [<a href='#0'>back to content</a>]

# In[ ]:


plot_intersections(
    professionals,
    group_memberships,
    'id',
    'user_id',
    'Professionals',
    'Groups',
    'How many professionals participate in groups',
    'Professionals without groups',
    'Professionals with groups',
    'Not professionals',
    'What is the percentage'
)


# In[ ]:


print(
    'Number of group members which are neither students nor professionals: ',
    len(set(group_memberships.user_id) - (set(students.id) | set(professionals.id)))
)


# ### <a id='8312'>Students intersected with School_memberships by user_id</a> [<a href='#0'>back to content</a>]

# In[ ]:


plot_intersections(
    students,
    school_memberships,
    'id',
    'user_id',
    'Students',
    'Schools',
    'How many students are related to some school',
    'Students without some school',
    'Students with some school',
    'Not students',
    'What is the percentage'
)


# ### <a id='8313'>Professionals intersected with School_memberships by user_id</a> [<a href='#0'>back to content</a>]

# In[ ]:


plot_intersections(
    professionals,
    school_memberships,
    'id',
    'user_id',
    'Professionals',
    'Schools',
    'How many professionals are related to some school',
    'Professionals without some school',
    'Professionals with some school',
    'Not professionals',
    'What is the percentage'
)


# In[ ]:


print(
    'Number of school members which are neither students nor professionals: ',
    len(set(school_memberships.user_id) - (set(students.id) | set(professionals.id)))
)


# As we can see professionals are more active in various groups participation.

# ### <a id='8314'>Students intersected with Tag_users by user_id</a> [<a href='#0'>back to content</a>]

# In[ ]:


plot_intersections(
    students,
    tag_users,
    'id',
    'user_id',
    'Students',
    'Tags',
    'How many students follow tags',
    'Students not following tags',
    'Students following tags',
    'Not students',
    'What is the percentage'
)


# ### <a id='8315'>Professionals intersected with Tag_users by user_id</a> [<a href='#0'>back to content</a>]

# In[ ]:


plot_intersections(
    professionals,
    tag_users,
    'id',
    'user_id',
    'Professionals',
    'Tags',
    'How many professionals follow tags',
    'Professionals not following tags',
    'Professionals following tags',
    'Not professionals',
    'What is the percentage'
)


# In[ ]:


print(
    'Number of tags that were given neither to students nor professionals: ',
    len(set(tag_users.user_id) - (set(students.id) | set(professionals.id)))
)


# ### <a id='8316'>Questions intersected with Tag_questions by question_id</a> [<a href='#0'>back to content</a>]

# In[ ]:


plot_intersections(
    questions,
    tag_questions,
    'id',
    'question_id',
    'Questions',
    'Tags',
    'How many questions have been marked with tags',
    'Questions without tags',
    'Questions with tags',
    'Not questions',
    'What is the percentage'
)


# ### <a id='8317'>Students intersected with Emails by recipient_id</a> [<a href='#0'>back to content</a>]

# In[ ]:


plot_intersections(
    students,
    emails,
    'id',
    'recipient_id',
    'Students',
    'Emails',
    'How many students receive emails',
    'Students without emails',
    'Students with emails',
    'Not students',
    'What is the percentage'
)


# ### <a id='8318'>Professionals intersected with Emails by recipient_id</a> [<a href='#0'>back to content</a>]

# In[ ]:


plot_intersections(
    professionals,
    emails,
    'id',
    'recipient_id',
    'Professionals',
    'Emails',
    'How many professionals receive emails',
    'Professionals without emails',
    'Professionals with emails',
    'Not professionals',
    'What is the percentage'
)


# ### <a id='8319'>Emails intersected with Matches by email_id</a> [<a href='#0'>back to content</a>]

# In[ ]:


plot_intersections(
    emails,
    matches,
    'id',
    'email_id',
    'Emails',
    'Matches',
    'How many emails matched',
    'Emails without matches',
    'Emails with matches',
    'Not emails',
    'What is the percentage'
)


# ### <a id='8320'>Questions intersected with Matches by question_id</a> [<a href='#0'>back to content</a>]

# In[ ]:


plot_intersections(
    questions,
    matches,
    'id',
    'question_id',
    'Questions',
    'Matches',
    'How many questions matched',
    'Not matched questions',
    'Matched questions',
    'Unknown matches',
    'What is the percentage'
)


# ## <a id='84'>ER-diagram of data</a> [<a href='#0'>back to content</a>]

# Here is the result of the preliminary EDA.
# 
# It will probably help to understand the nature of the data.

# In[ ]:


plt.figure(figsize=(20, 30))
plt.imshow(mpimg.imread("../input/erdiagram/EntityRelationshipDiagramExample.jpeg"), interpolation='bilinear', aspect='auto')
plt.axis("off")
plt.show()


# # <a id='9'>Deeper analysis</a> [<a href='#0'>back to content</a>]

# <img src="https://i.kym-cdn.com/photos/images/newsfeed/000/531/557/a88.jpg" width="500"></img>
# 
# Yep, after some data quality analysis we can go deeper and try to collect more information about the domain.
# 
# For example, if we work with time series we can plot the dynamic of, say:
# - professionals or students community growth (yearly, monthly, meekly, daily, hourly);
# - community growth in certain professional area;
# - question/answer/comment dynamic (how many unanswered/uncommented do we have? Is this number increasing/decreasing/stays the same over time?).

# In[ ]:


tables_columns_info[tables_columns_info.column.str.contains('date')]


# In[ ]:


extract_year_month_week_day_hour(professionals, 'date_joined')


# In[ ]:


extract_year_month_week_day_hour(students, 'date_joined')


# In[ ]:


extract_year_month_week_day_hour(emails, 'date_sent')


# In[ ]:


extract_year_month_week_day_hour(answers, 'date_added')


# In[ ]:


extract_year_month_week_day_hour(comments, 'date_added')


# In[ ]:


extract_year_month_week_day_hour(questions, 'date_added')


# In[ ]:


print('Professionals year range is: ', professionals.year.min(), professionals.year.max())
print('Students year range is: ', students.year.min(), students.year.max())
print('Emails year range is: ', emails.year.min(), emails.year.max())
print('Answers year range is: ', answers.year.min(), answers.year.max())
print('Comments year range is: ', comments.year.min(), comments.year.max())
print('Questions year range is: ', questions.year.min(), questions.year.max())


# In[ ]:


print('Professionals month range is: ', professionals.month.min(), professionals.month.max())
print('Students month range is: ', students.month.min(), students.month.max())
print('Emails month range is: ', emails.month.min(), emails.month.max())
print('Answers month range is: ', answers.month.min(), answers.month.max())
print('Comments month range is: ', comments.month.min(), comments.month.max())
print('Questions month range is: ', questions.month.min(), questions.month.max())


# In[ ]:


print('Professionals week range is: ', professionals.week.min(), professionals.week.max())
print('Students week range is: ', students.week.min(), students.week.max())
print('Emails week range is: ', emails.week.min(), emails.week.max())
print('Answers week range is: ', answers.week.min(), answers.week.max())
print('Comments week range is: ', comments.week.min(), comments.week.max())
print('Questions week range is: ', questions.week.min(), questions.week.max())


# In[ ]:


print('Professionals day range is: ', professionals.day.min(), professionals.day.max())
print('Students day range is: ', students.day.min(), students.day.max())
print('Emails day range is: ', emails.day.min(), emails.day.max())
print('Answers day range is: ', answers.day.min(), answers.day.max())
print('Comments day range is: ', comments.day.min(), comments.day.max())
print('Questions day range is: ', questions.day.min(), questions.day.max())


# In[ ]:


print('Professionals hour range is: ', professionals.hour.min(), professionals.hour.max())
print('Students hour range is: ', students.hour.min(), students.hour.max())
print('Emails hour range is: ', emails.hour.min(), emails.hour.max())
print('Answers hour range is: ', answers.hour.min(), answers.hour.max())
print('Comments hour range is: ', comments.hour.min(), comments.hour.max())
print('Questions hour range is: ', questions.hour.min(), questions.hour.max())


# ## <a id='91'>Cumulative community growth</a> [<a href='#0'>back to content</a>]

# ### <a id='911'>Yearly</a> [<a href='#0'>back to content</a>]

# In[ ]:


professionals_yearly_cumcount = calculate_yearly_cumsum(professionals, 'professionals')
students_yearly_cumcount = calculate_yearly_cumsum(students, 'students')


# In[ ]:


plt.figure(figsize=(20, 10))
plt.plot(professionals_yearly_cumcount, label='Professionals', color='green')
plt.plot(students_yearly_cumcount, label='Students', color='yellow')
plt.xlabel('year')
plt.ylabel('count')
plt.title('Yearly cumulative community growth')
plt.legend()
plt.show()


# ### <a id='912'>Monthly</a> [<a href='#0'>back to content</a>]

# In[ ]:


professionals_monthly_cumcount = calculate_monthly_cumsum(professionals, 'professionals')
students_monthly_cumcount = calculate_monthly_cumsum(students, 'students')


# In[ ]:


plt.figure(figsize=(20, 10))
plt.plot(professionals_monthly_cumcount.professionals_count, label='Professionals', color='green')
plt.plot(students_monthly_cumcount.students_count, label='Students', color='yellow')
plt.xlabel('month')
plt.ylabel('count')
plt.title('Monthly cumulative community growth')
plt.legend()
plt.show()


# ### <a id='913'>Weekly</a> [<a href='#0'>back to content</a>]

# In[ ]:


professionals_weekly_cumcount = calculate_weekly_cumsum(professionals, 'professionals')
students_weekly_cumcount = calculate_weekly_cumsum(students, 'students')


# In[ ]:


plt.figure(figsize=(20, 10))
plt.plot(professionals_weekly_cumcount.professionals_count, label='Professionals', color='green')
plt.plot(students_weekly_cumcount.students_count, label='Students', color='yellow')
plt.xlabel('week')
plt.ylabel('count')
plt.title('Weekly cumulative community growth')
plt.legend()
plt.show()


# ### <a id='914'>Daily</a> [<a href='#0'>back to content</a>]

# In[ ]:


professionals_daily_cumcount = calculate_daily_cumsum(professionals, 'professionals')
students_daily_cumcount = calculate_daily_cumsum(students, 'students')


# In[ ]:


plt.figure(figsize=(20, 10))
plt.plot(professionals_daily_cumcount.professionals_count, label='Professionals', color='green')
plt.plot(students_daily_cumcount.students_count, label='Students', color='yellow')
plt.xlabel('day')
plt.ylabel('count')
plt.title('Daily cumulative community growth')
plt.legend()
plt.show()


# ## <a id='92'>Community growth dynamic</a> [<a href='#0'>back to content</a>]

# ### <a id='921'>Yearly</a> [<a href='#0'>back to content</a>]

# In[ ]:


professionals_yearly_count = calculate_yearly_count(professionals, 'professionals')
students_yearly_count = calculate_yearly_count(students, 'students')


# In[ ]:


plt.figure(figsize=(20, 10))
plt.plot(professionals_yearly_count, label='Professionals', color='green')
plt.plot(students_yearly_count, label='Students', color='yellow')
plt.xlabel('year')
plt.ylabel('count')
plt.title('Yearly growth dynamic (how many new professionals/students per year do we have)')
plt.legend()
plt.show()


# ### <a id='922'>Monthly</a> [<a href='#0'>back to content</a>]

# In[ ]:


professionals_monthly_count = calculate_monthly_count(professionals, 'professionals')
students_monthly_count = calculate_monthly_count(students, 'students')


# In[ ]:


plt.figure(figsize=(20, 10))
plt.plot(professionals_monthly_count.professionals_count, label='Professionals', color='green')
plt.plot(students_monthly_count.students_count, label='Students', color='yellow')
plt.xlabel('month')
plt.ylabel('count')
plt.title('Monthly growth dynamic (how many new professionals/students per month do we have)')
plt.legend()
plt.show()


# ### <a id='923'>Weekly</a> [<a href='#0'>back to content</a>]

# In[ ]:


professionals_weekly_count = calculate_weekly_count(professionals, 'professionals')
students_weekly_count = calculate_weekly_count(students, 'students')


# In[ ]:


plt.figure(figsize=(20, 10))
plt.plot(professionals_weekly_count.professionals_count, label='Professionals', color='green')
plt.plot(students_weekly_count.students_count, label='Students', color='yellow')
plt.xlabel('week')
plt.ylabel('count')
plt.title('Weekly growth dynamic (how many new professionals/students per week do we have)')
plt.legend()
plt.show()


# ### <a id='924'>Daily</a> [<a href='#0'>back to content</a>]

# In[ ]:


professionals_daily_count = calculate_daily_count(professionals, 'professionals')
students_daily_count = calculate_daily_count(students, 'students')


# In[ ]:


plt.figure(figsize=(20, 10))
plt.plot(professionals_daily_count.professionals_count, label='Professionals', color='green')
plt.plot(students_daily_count.students_count, label='Students', color='yellow')
plt.xlabel('day')
plt.ylabel('count')
plt.title('Daily growth dynamic (how many new professionals/students per day do we have)')
plt.legend()
plt.show()


# ## <a id='93'>Cumulative questions/answers/comments growth</a> [<a href='#0'>back to content</a>]

# ### <a id='931'>Yearly</a> [<a href='#0'>back to content</a>]

# In[ ]:


questions_yearly_cumcount = calculate_yearly_cumsum(questions, 'questions')
answers_yearly_cumcount = calculate_yearly_cumsum(answers, 'answers')
comments_yearly_cumcount = calculate_yearly_cumsum(comments, 'comments')


# In[ ]:


plt.figure(figsize=(20, 10))
plt.plot(questions_yearly_cumcount, label='Questions', color='orange')
plt.plot(answers_yearly_cumcount, label='Answers', color='purple')
plt.plot(comments_yearly_cumcount, label='Comments', color='pink')
plt.xlabel('year')
plt.ylabel('count')
plt.title('Yearly cumulative questions/answers/comments/emails count growth')
plt.legend()
plt.show()


# ### <a id='932'>Monthly</a> [<a href='#0'>back to content</a>]

# In[ ]:


questions_monthly_cumcount = calculate_monthly_cumsum(questions, 'questions')
answers_monthly_cumcount = calculate_monthly_cumsum(answers, 'answers')
comments_monthly_cumcount = calculate_monthly_cumsum(comments, 'comments')


# In[ ]:


plt.figure(figsize=(20, 10))
plt.plot(questions_monthly_cumcount.questions_count, label='Questions', color='orange')
plt.plot(answers_monthly_cumcount.answers_count, label='Answers', color='purple')
plt.plot(comments_monthly_cumcount.comments_count, label='Comments', color='pink')
plt.xlabel('month')
plt.ylabel('count')
plt.title('Monthly cumulative questions/answers/comments count growth')
plt.legend()
plt.show()


# ### <a id='933'>Weekly</a> [<a href='#0'>back to content</a>]

# In[ ]:


questions_weekly_cumcount = calculate_weekly_cumsum(questions, 'questions')
answers_weekly_cumcount = calculate_weekly_cumsum(answers, 'answers')
comments_weekly_cumcount = calculate_weekly_cumsum(comments, 'comments')


# In[ ]:


plt.figure(figsize=(20, 10))
plt.plot(questions_weekly_cumcount.questions_count, label='Questions', color='orange')
plt.plot(answers_weekly_cumcount.answers_count, label='Answers', color='purple')
plt.plot(comments_weekly_cumcount.comments_count, label='Comments', color='pink')
plt.xlabel('week')
plt.ylabel('count')
plt.title('Weekly cumulative questions/answers/comments count growth')
plt.legend()
plt.show()


# ### <a id='934'>Daily</a> [<a href='#0'>back to content</a>]

# In[ ]:


questions_daily_cumcount = calculate_daily_cumsum(questions, 'questions')
answers_daily_cumcount = calculate_daily_cumsum(answers, 'answers')
comments_daily_cumcount = calculate_daily_cumsum(comments, 'comments')


# In[ ]:


plt.figure(figsize=(20, 10))
plt.plot(questions_daily_cumcount.questions_count, label='Questions', color='orange')
plt.plot(answers_daily_cumcount.answers_count, label='Answers', color='purple')
plt.plot(comments_daily_cumcount.comments_count, label='Comments', color='pink')
plt.xlabel('week')
plt.ylabel('count')
plt.title('Daily cumulative questions/answers/comments count growth')
plt.legend()
plt.show()


# ## <a id='94'>Questions/answers/comments growth dynamic</a> [<a href='#0'>back to content</a>]

# ### <a id='941'>Yearly</a> [<a href='#0'>back to content</a>]

# In[ ]:


questions_yearly_count = calculate_yearly_count(questions, 'questions')
answers_yearly_count = calculate_yearly_count(answers, 'answers')
comments_yearly_count = calculate_yearly_count(comments, 'comments')


# In[ ]:


plt.figure(figsize=(20, 10))
plt.plot(questions_yearly_count, label='Questions', color='orange')
plt.plot(answers_yearly_count, label='Answers', color='purple')
plt.plot(comments_yearly_count, label='Comments', color='pink')
plt.xlabel('year')
plt.ylabel('count')
plt.title('Yearly questions/answers/comments count growth dynamic')
plt.legend()
plt.show()


# ### <a id='942'>Monthly</a> [<a href='#0'>back to content</a>]

# In[ ]:


questions_monthly_count = calculate_monthly_count(questions, 'questions')
answers_monthly_count = calculate_monthly_count(answers, 'answers')
comments_monthly_count = calculate_monthly_count(comments, 'comments')


# In[ ]:


plt.figure(figsize=(20, 10))
plt.plot(questions_monthly_count.questions_count, label='Questions', color='orange')
plt.plot(answers_monthly_count.answers_count, label='Answers', color='purple')
plt.plot(comments_monthly_count.comments_count, label='Comments', color='pink')
plt.xlabel('month')
plt.ylabel('count')
plt.title('Monthly questions/answers/comments count growth dynamic')
plt.legend()
plt.show()


# ### <a id='943'>Weekly</a> [<a href='#0'>back to content</a>]

# In[ ]:


questions_weekly_count = calculate_weekly_count(questions, 'questions')
answers_weekly_count = calculate_weekly_count(answers, 'answers')
comments_weekly_count = calculate_weekly_count(comments, 'comments')


# In[ ]:


plt.figure(figsize=(20, 10))
plt.plot(questions_weekly_count.questions_count, label='Questions', color='orange')
plt.plot(answers_weekly_count.answers_count, label='Answers', color='purple')
plt.plot(comments_weekly_count.comments_count, label='Comments', color='pink')
plt.xlabel('week')
plt.ylabel('count')
plt.title('Weekly questions/answers/comments count growth dynamic')
plt.legend()
plt.show()


# ### <a id='944'>Daily</a> [<a href='#0'>back to content</a>]

# In[ ]:


questions_daily_count = calculate_daily_count(questions, 'questions')
answers_daily_count = calculate_daily_count(answers, 'answers')
comments_daily_count = calculate_daily_count(comments, 'comments')


# In[ ]:


plt.figure(figsize=(20, 10))
plt.plot(questions_daily_count.questions_count, label='Questions', color='orange')
plt.plot(answers_daily_count.answers_count, label='Answers', color='purple')
plt.plot(comments_daily_count.comments_count, label='Comments', color='pink')
plt.xlabel('day')
plt.ylabel('count')
plt.title('Daily questions/answers/comments count growth dynamic')
plt.legend()
plt.show()


# ## <a id='95'>Professionals</a> [<a href='#0'>back to content</a>]
# 
# Okay, we've got some understanding of community growth.
# 
# But now let's try to see what types of members do we have.
# 
# For example:
# - how many professionals ask but not answer questions;
# - how many of them also comment questions/answers;
# - etc.

# ### <a id='951'>Professionals by answered_questions/asked_questions/wrote_comments flags</a> [<a href='#0'>back to content</a>]

# In[ ]:


professionals['answered_questions'] = professionals.id.isin(answers.author_id)
professionals['asked_questions'] = professionals.id.isin(questions.author_id)
professionals['wrote_comments'] = professionals.id.isin(answers.author_id)


# In[ ]:


professionals_crosstab_stats = pd.crosstab(
    professionals.year,
    [
        professionals.answered_questions,
        professionals.asked_questions,
        professionals.wrote_comments
    ],
    rownames=[
        'year'
    ],
    colnames=[
        'answered questions',
        'asked questions',
        'wrote comments'
    ]
).cumsum().apply(lambda row: 100 * row / sum(row), axis=1)


# In[ ]:


professionals_crosstab_stats


# So each year there is a big part of professionals community which does not ask questions, does not answer questions and does not write comments (from ~36.84% in 2011th to ~64.22% in 2019th).
# 
# Also there is another big part of professionals community which answer questions and writes comments (from ~35.71% in 2019th to ~59.65% in 2011th).

# ### <a id='952'>Distribution of professionals by questions/answers/comments count</a> [<a href='#0'>back to content</a>]

# In[ ]:


professionals_questions_answers_comments = pd.merge(
    professionals[['id', 'date_joined']],
    questions[['author_id', 'date_added']],
    left_on='id',
    right_on='author_id',
    how='left'
)
professionals_questions_answers_comments.rename({
    'author_id': 'question_author_id',
    'date_added': 'question_date_added'
}, axis=1, inplace=True)
professionals_questions_answers_comments = pd.merge(
    professionals_questions_answers_comments,
    answers[['author_id', 'date_added']],
    left_on='id',
    right_on='author_id',
    how='left'
)
professionals_questions_answers_comments.rename({
    'author_id': 'answer_author_id',
    'date_added': 'answer_date_added'
}, axis=1, inplace=True)
professionals_questions_answers_comments = pd.merge(
    professionals_questions_answers_comments,
    comments[['author_id', 'date_added']],
    left_on='id',
    right_on='author_id',
    how='left'
)
professionals_questions_answers_comments.rename({
    'author_id': 'comment_author_id',
    'date_added': 'comment_date_added'
}, axis=1, inplace=True)


# In[ ]:


professionals_questions_answers_comments_grouped = professionals_questions_answers_comments.groupby('id')[
    ['question_author_id', 'answer_author_id', 'comment_author_id']
].count().reset_index().rename({
    'question_author_id': 'questions_count',
    'answer_author_id': 'answers_count',
    'comment_author_id': 'comments_count'
}, axis=1)


# In[ ]:


professionals_questions_answers_comments_grouped.questions_count.nunique()


# In[ ]:


professionals_questions_answers_comments_grouped.answers_count.nunique()


# In[ ]:


professionals_questions_answers_comments_grouped.comments_count.nunique()


# In[ ]:


plt.figure(figsize=(20, 10))
plt.imshow(
    wordcloud.WordCloud(
        min_font_size=8,
        relative_scaling=0.1,
        background_color='white',
        width=2000,
        height=2000
    ).generate_from_frequencies(
        dict(
            zip(
                list(map(str, professionals_questions_answers_comments_grouped.questions_count.value_counts().index)),
                professionals_questions_answers_comments_grouped.questions_count.value_counts().values
            )
        )
    ),
    interpolation='bilinear'
)
plt.axis("off")
plt.show()


# In[ ]:


plt.figure(figsize=(20, 20))
plt.imshow(
    wordcloud.WordCloud(
        min_font_size=8,
        relative_scaling=0.1,
        background_color='white',
        width=4000,
        height=2000
    ).generate_from_frequencies(
        dict(
            zip(
                list(map(str, professionals_questions_answers_comments_grouped.answers_count.value_counts().index)),
                professionals_questions_answers_comments_grouped.answers_count.value_counts().values
            )
        )
    ),
    interpolation="bilinear"
)
plt.axis("off")
plt.show()


# In[ ]:


plt.figure(figsize=(20, 10))
plt.imshow(
    wordcloud.WordCloud(
        min_font_size=8,
        relative_scaling=0.1,
        background_color='white',
        width=4000,
        height=2000
    ).generate_from_frequencies(
        dict(
            zip(
                list(map(str, professionals_questions_answers_comments_grouped.comments_count.value_counts().index)),
                professionals_questions_answers_comments_grouped.comments_count.value_counts().values
            )
        )
    ),
    interpolation='bilinear'
)
plt.axis("off")
plt.show()


# ### <a id='953'>Professionals locations, industries, headlines</a> [<a href='#0'>back to content</a>]

# Okay, enough of questions and answers))
# 
# Not that I would stop asking questions and searching for answers but, hey, you know what I mean))
# 
# Let's now concentrate more on three other columns namely:
# - location;
# - industry;
# - headline.

# In[ ]:


print('Unique locations number: ', professionals.location.nunique())
print('Unique industries number: ', professionals.industry.nunique())
print('Unique headlines number: ', professionals.headline.nunique())


# To much of unique values so it will be inconvenient to use TOP-something or countplot, or pieplot.
# 
# But probably something like wordcloud would help.
# 
# Let's try:

# In[ ]:


plt.figure(figsize=(20, 10))
plt.imshow(
    wordcloud.WordCloud(
        min_font_size=8,
        relative_scaling=0.1,
        background_color='white',
        width=4000,
        height=2000
    ).generate(' '.join(professionals.location.dropna().values)),
    interpolation='bilinear'
)
plt.axis("off")
plt.show()


# In[ ]:


plt.figure(figsize=(20, 10))
plt.imshow(
    wordcloud.WordCloud(
        min_font_size=8,
        relative_scaling=0.1,
        background_color='white',
        width=4000,
        height=2000
    ).generate(' '.join(professionals.industry.dropna().values)),
    interpolation='bilinear'
)
plt.axis("off")
plt.show()


# In[ ]:


plt.figure(figsize=(20, 10))
plt.imshow(
    wordcloud.WordCloud(
        min_font_size=8,
        relative_scaling=0.1,
        background_color='white',
        width=4000,
        height=2000
    ).generate(' '.join(professionals.headline.dropna().values)),
    interpolation='bilinear'
)
plt.axis("off")
plt.show()


# ## <a id='96'>Students</a> [<a href='#0'>back to content</a>]

# ### <a id='961'>Students by answered_questions/asked_questions/wrote_comments flags</a> [<a href='#0'>back to content</a>]

# In[ ]:


students['answered_questions'] = students.id.isin(answers.author_id)
students['asked_questions'] = students.id.isin(questions.author_id)
students['wrote_comments'] = students.id.isin(answers.author_id)


# In[ ]:


students_crosstab_stats = pd.crosstab(
    students.year,
    [
        students.answered_questions,
        students.asked_questions,
        students.wrote_comments
    ],
    rownames=[
        'year'
    ],
    colnames=[
        'answered questions',
        'asked questions',
        'wrote comments'
    ]
).cumsum().apply(lambda row: 100 * row / sum(row), axis=1)


# In[ ]:


students_crosstab_stats


# Here we also see that at least half of students does not ask questions, does not answer questions and does not write comments.
# 
# But still there are students (and not that few) who ask questions.

# ### <a id='962'>Students locations</a> [<a href='#0'>back to content</a>]

# In[ ]:


print('Unique locations number: ', students.location.nunique())


# In[ ]:


plt.figure(figsize=(20, 10))
plt.imshow(
    wordcloud.WordCloud(
        min_font_size=8,
        relative_scaling=0.1,
        background_color='white',
        width=4000,
        height=2000
    ).generate(' '.join(students.location.dropna().values)),
    interpolation='bilinear'
)
plt.axis("off")
plt.show()


# ### <a id='97'>Tags</a> [<a href='#0'>back to content</a>]

# In[ ]:


tag_users_with_names = pd.merge(tag_users, tags, how='inner', on='tag_id')


# In[ ]:


professional_tags = tag_users_with_names[tag_users_with_names.user_id.isin(professionals.id)]


# In[ ]:


student_tags = tag_users_with_names[tag_users_with_names.user_id.isin(students.id)]


# In[ ]:


plt.figure(figsize=(20, 10))
plt.imshow(
    wordcloud.WordCloud(
        min_font_size=8,
        background_color='white',
        width=4000,
        height=2000
    ).generate(' '.join(professional_tags.tag_name.values)),
    interpolation='bilinear'
)
plt.axis("off")
plt.show()


# In[ ]:


plt.figure(figsize=(20, 10))
plt.imshow(
    wordcloud.WordCloud(
        min_font_size=8,
        background_color='white',
        width=4000,
        height=2000
    ).generate(' '.join(student_tags.tag_name.values)),
    interpolation='bilinear'
)
plt.axis("off")
plt.show()


# ### <a id='98'>Emails</a> [<a href='#0'>back to content</a>]

# In[ ]:


emails.head()


# In[ ]:


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
sns.countplot(x=emails.frequency_level, ax=ax1)
ax1.set_xticklabels(['Daily', 'immediate', 'weekly'])
ax2.pie(
    x=emails.frequency_level.value_counts().values,
    labels=['Daily', 'immediate', 'weekly'],
    autopct='%1.2f%%'
)
plt.tight_layout()
plt.show()


# ### <a id='99'>Matches</a> [<a href='#0'>back to content</a>]

# In[ ]:


matches.head()


# Here we've got a table that contains the information about what email matches what question.
# 
# But what does it mean?
# 
# Can we interpret it as a positive examples?
# 
# Do these questions somehow different from the questions that are not in in this "matches" table?
# 
# Let's try to fing it out!

# In[ ]:


questions['with_matches'] = questions.id.isin(matches.question_id)
questions['with_answers'] = questions.id.isin(answers.question_id)
questions['with_comments'] = questions.id.isin(comments.parent_content_id)


# In[ ]:


pd.crosstab(
    questions.with_matches,
    [
        questions.with_answers,
        questions.with_comments
    ],
    rownames=[
        'with matches'
    ],
    colnames=[
        'with answers',
        'with comments'
    ],
    normalize=True
) * 100


# In[ ]:


pd.crosstab(
    questions[questions.with_matches].with_matches,
    [
        questions[questions.with_matches].with_answers,
        questions[questions.with_matches].with_comments
    ],
    rownames=[
        'with matches'
    ],
    colnames=[
        'with answers',
        'with comments'
    ],
    normalize=True
) * 100


# In[ ]:


pd.crosstab(
    questions[~questions.with_matches].with_matches,
    [
        questions[~questions.with_matches].with_answers,
        questions[~questions.with_matches].with_comments
    ],
    rownames=[
        'with matches'
    ],
    colnames=[
        'with answers',
        'with comments'
    ],
    normalize=True
) * 100


# Well, as we can see from this crosstab:
# - ~82.15% of questions are with matches and they have answers;
# - ~6.56%  of questions are with matches and they have not only answers but also comments;
# - ~2.55% of questions are with matches AND YET THEY DO NOT HAVE ANSWERS;
# - ~7.08% of questions are without matches AND YET THEY HAVE ANSWERS;
# - ~83.16% of questions without matches DO HAVE ANSWERS and ~9.18% of questions without matches HAVE NOT ONLY ANSWERS BUT ALSO COMMENTS.
# 
# 
# It looks that the simple boolean flag of answer presence cannot separate questions with matches from questions without matches.
# 
# But what about the time difference between question and answer?

# In[ ]:


questions.head()


# In[ ]:


answers.head()


# In[ ]:


q_and_a = pd.merge(questions, answers, left_on='id', right_on='question_id', how='left', suffixes=('_q', '_a'))


# In[ ]:


q_and_a.sort_values(by=['date_added_q', 'date_added_a'], inplace=True)


# In[ ]:


q_and_a.head().T


# In[ ]:


q_and_a['q_and_a_time_diff'] = q_and_a.date_added_a - q_and_a.date_added_q


# In[ ]:


q_and_a.head().T


# In[ ]:


q_and_a.info(verbose=True, null_counts=True)


# In[ ]:


q_and_a[q_and_a.id_q.isin(matches.question_id)].q_and_a_time_diff.describe()


# Hmmm...
# 
# There are some really strange q&a pairs where the answer was written EARLIER THAT the question!!!
# 
# I do not really understand how is it possible, probably there were some issues in the system.
# 
# But, anyway, let's look at them:

# In[ ]:


q_and_a[
    (q_and_a.id_q.isin(matches.question_id)) &
    (q_and_a.q_and_a_time_diff < pd.Timedelta(0, unit='d'))
].head().T


# Okay, they look like usual ordinary questions and answers
# 
# Let's look at how many of them:

# In[ ]:


print('Number of q & a pairs with negative responce time: {}'.format(len(q_and_a[
    (q_and_a.id_q.isin(matches.question_id)) &
    (q_and_a.q_and_a_time_diff < pd.Timedelta(0, unit='d'))
])))


# Not so many of them.
# 
# I guess it's not a big deal if we just get rid of them because there is just too few of that kind of question to consider them as a serious issue.

# In[ ]:


strange_q_and_a_index = q_and_a[
    (q_and_a.id_q.isin(matches.question_id)) &
    (q_and_a.q_and_a_time_diff < pd.Timedelta(0, unit='d'))
].index


# In[ ]:


q_and_a[
    (q_and_a.id_q.isin(matches.question_id)) &
    (~q_and_a.index.isin(strange_q_and_a_index))
].q_and_a_time_diff.describe()


# In[ ]:


q_and_a[~q_and_a.id_q.isin(matches.question_id)].q_and_a_time_diff.describe()


# Cool!
# 
# As we can see, the mean responce time of questions that are in matches is almost twice less that the mean responce time of questions that are not in matches.
# 
# Good to see))
# 
# Moreover, the count of questions that are in matches is a lot bigger that the count of questions that are not in matches.
# 
# Definitely, it means that if the question is in matches table than it (most probably) does have one or more answers (because ~82.15% of questions, that are with matches do have answers).

# In[ ]:


q_and_a['in_matches'] = q_and_a.id_q.isin(matches.question_id)
q_and_a['days_to_answer'] = q_and_a.q_and_a_time_diff.dt.days


# In[ ]:


q_and_a_groupped = q_and_a.groupby('id_q')[
    ['in_matches', 'id_a']
].agg({'in_matches': 'first', 'id_a': 'count'}).reset_index().rename({'id_a': 'answers_count'}, axis=1)


# In[ ]:


q_and_a_groupped.head()


# In[ ]:


f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10), sharey=False, sharex=False)
sns.boxplot(y='in_matches', x='days_to_answer', orient='h', data=q_and_a, ax=ax1)
sns.boxplot(y='in_matches', x='answers_count', orient='h', data=q_and_a_groupped, ax=ax2)
sns.countplot(y='in_matches', data=q_and_a, ax=ax3)
ax3.set_xlabel('count of questions')
plt.suptitle('The plot here shows the difference between questions in/not in matches')
plt.show()


# # Please give me your upvote if you find my work interesting or informative))
# 
# # Also you are free to fork this kernel and modify it as you wish.
# 
# # I'll be glad to see appreciation of any kind)))
