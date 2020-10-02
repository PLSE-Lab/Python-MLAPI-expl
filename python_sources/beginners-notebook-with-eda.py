#!/usr/bin/env python
# coding: utf-8

# # Import libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns


# # Import Dataset

# In[ ]:


que = pd.read_csv('../input/data-science-for-good-careervillage/questions.csv')
ans = pd.read_csv('../input/data-science-for-good-careervillage/answers.csv')
prof = pd.read_csv('../input/data-science-for-good-careervillage/professionals.csv')
stud = pd.read_csv('../input/data-science-for-good-careervillage/students.csv')


# # Look into DataSet

# In[ ]:


ans.shape


# In[ ]:


que.shape


# In[ ]:


prof.shape


# In[ ]:


que.head()


# In[ ]:


stud.shape


# In[ ]:


stud.isnull().sum()


# In[ ]:


stud.head()


# In[ ]:


ans.head()


# In[ ]:


prof.head()


# In[ ]:


stud.students_location.unique()


# In[ ]:


n_locations = 20

users = [
    ('students', stud),
    ('professionals', prof)
]

for user, df in users:
    locations = df['{}_location'.format(user)].value_counts().sort_values(ascending=True).tail(n_locations)
    
    ax = locations.plot(kind='barh',figsize=(14, 10),width=0.8, fontsize=14) 
    ax.set_title('Top %s {} locations'.format(user) % n_locations, fontsize=20)
    ax.set_xlabel('Number of {}'.format(user), fontsize=14)
    for p in ax.patches:
        ax.annotate(str(p.get_width()), (p.get_width(), p.get_y()), color='w', fontsize=14)
    plt.show()


# In[ ]:


stud.isnull().sum()


# In[ ]:


ans.isnull().sum()


# In[ ]:


que.isnull().sum()


# # Datetime feature preprocessing

# In[ ]:


ans['answers_date_added'] = pd.to_datetime(ans['answers_date_added'], infer_datetime_format=True)
prof['professionals_date_joined'] = pd.to_datetime(prof['professionals_date_joined'], infer_datetime_format=True)
que['questions_date_added'] = pd.to_datetime(que['questions_date_added'], infer_datetime_format=True)
stud['students_date_joined'] = pd.to_datetime(stud['students_date_joined'], infer_datetime_format=True)


# # Users Growth

# In[ ]:


users = [
    ('students', stud),
    ('professionals', prof)
]

colors = {'students' : 'cyan', 'professionals' : 'mediumvioletred'}

for user, df in users:
    
    years = df['{}_date_joined'.format(user)].dt.year.unique()
    years.sort()
    
    min_date = df['{}_date_joined'.format(user)].min()
    min_date = min_date.strftime("%B %Y")
    
    max_date = df['{}_date_joined'.format(user)].max()
    max_date = max_date.strftime("%B %Y")
    
    
    amounts = [len(df[df['{}_date_joined'.format(user)].dt.year == y]) for y in years]
    
    for i in range(len(amounts)):
        if i > 0:
            amounts[i] += amounts[i - 1]
    to_plot = pd.DataFrame({'years': years, 'users': amounts})
    plt.figure(figsize=(14, 5))
    
    plt.plot('years', 'users', data=to_plot, marker='o', color=colors[user])
    x = to_plot['years']
    y = to_plot['users']
    plt.fill_between(x, y, color=colors[user], alpha = 0.4)
    
    plt.ylabel('Users', fontsize=14)
    plt.title('Growth of {}'.format(user), fontsize=20)
    plt.show()


# # Number of Answers & Questions added per year

# In[ ]:


entities = [
    ('questions', que),
    ('answers', ans)
]

colors = {'questions' : 'cyan', 'answers' : 'mediumvioletred'}

for entity, df in entities:
    min_date = df['{}_date_added'.format(entity)].min().strftime("%B %Y")
    max_date = df['{}_date_added'.format(entity)].max().strftime("%B %Y")

    df['year'] = df['{}_date_added'.format(entity)].dt.year
    plt_data = df.groupby('year').size()
    plt_data.plot(figsize=(14, 5), color=colors[entity],  marker='o')

    x = plt_data.reset_index()['year']
    y = plt_data.reset_index()[0]
    plt.fill_between(x, y, color=colors[entity], alpha = 0.4)

    plt.xlabel('Year', fontsize=15)
    plt.ylabel('{} Count'.format(entity.capitalize()), fontsize=15)
    plt.title('Number of {} asked per year ({}-{})'.format(entity.capitalize(), min_date, max_date), fontsize=20)
    plt.show()


# # Professionals Answers & Students Questions amounts

# In[ ]:


from collections import Counter 

ll = [
    ('students', 'questions', stud, que),
    ('professionals', 'answers', prof, ans)
]

colors = {'students' : 'cyan', 'professionals' : 'mediumvioletred'}

for user, entity, user_df, entity_df in ll:
    tm = dict(sorted(Counter(pd.merge(user_df, entity_df, left_on='{}_id'.format(user), right_on='{}_author_id'.format(entity), how='inner').groupby('{}_id'.format(user)).size().values).items())) 
    t_d = {}
    t_d['{}_amount'.format(entity)] = list(tm.keys())
    t_d['{}_amount'.format(user)] = list(tm.values())

    plt_data = pd.DataFrame(t_d)

    plt_data.plot(x='{}_amount'.format(entity), y='{}_amount'.format(user), kind='bar', figsize=(14, 5), color=colors[user])
    plt.xlim(-1, 30)
    plt.xlabel('{} Count'.format(entity.capitalize()), fontsize=15)
    plt.ylabel('{} Count'.format(user.capitalize()), fontsize=15)
    plt.title('{} {} Diagram'.format(user.capitalize(), entity.capitalize()), fontsize=20)
    plt.show()


# # Top n Professionals with most Answers & Students with most Questions

# In[ ]:


n = 10

ll = [
    ('students', 'questions', stud, que),
    ('professionals', 'answers', prof, ans)
]

colors = {'students' : '#549da8', 'professionals' : '#852ab2'} 

for user, entity, user_df, entity_df in ll:
    top_n = pd.DataFrame(pd.merge(user_df, entity_df, left_on='{}_id'.format(user), right_on='{}_author_id'.format(entity), how='inner').groupby('{}_id'.format(user)).size().reset_index())
    plt_data = top_n.rename(index=str, columns={0: '{}_amount'.format(entity)}).sort_values(by=['{}_amount'.format(entity)], ascending=False)[:n]

    plt_data.plot(kind='bar', figsize=(14, 5), color=colors[user])
    plt.xticks(np.arange(len(plt_data)), tuple(plt_data['{}_id'.format(user)]), rotation=90)
    plt.xlabel('{} Ids'.format(user.capitalize()), fontsize=15)
    plt.ylabel('{} Count'.format(entity.capitalize()), fontsize=15)
    plt.title('Top {} {} with most {}'.format(n, user.capitalize(), entity), fontsize=20)
    leg = plt.legend(loc='best', fontsize=15)
    for text in leg.get_texts():
        plt.setp(text, color = 'w')
    plt.show()


# # First activity after registration 
# ### There are two general types of users:
# 
# - Activity right after registration
# - No activity at all

# In[ ]:


answers = pd.read_csv('../input/data-science-for-good-careervillage/answers.csv')
answer_scores = pd.read_csv('../input/data-science-for-good-careervillage/answer_scores.csv')
comments = pd.read_csv('../input/data-science-for-good-careervillage/comments.csv')
emails = pd.read_csv('../input/data-science-for-good-careervillage/emails.csv')
groups = pd.read_csv('../input/data-science-for-good-careervillage/groups.csv')
group_memberships = pd.read_csv('../input/data-science-for-good-careervillage/group_memberships.csv')
matches = pd.read_csv('../input/data-science-for-good-careervillage/matches.csv')
professionals = pd.read_csv('../input/data-science-for-good-careervillage/professionals.csv')
questions = pd.read_csv('../input/data-science-for-good-careervillage/questions.csv')
question_scores = pd.read_csv('../input/data-science-for-good-careervillage/question_scores.csv')
school_memberships = pd.read_csv('../input/data-science-for-good-careervillage/school_memberships.csv')
students = pd.read_csv('../input/data-science-for-good-careervillage/students.csv')
tags = pd.read_csv('../input/data-science-for-good-careervillage/tags.csv')
tag_questions = pd.read_csv('../input/data-science-for-good-careervillage/tag_questions.csv')
tag_users = pd.read_csv('../input/data-science-for-good-careervillage/tag_users.csv')


# In[ ]:


answers['answers_date_added'] = pd.to_datetime(answers['answers_date_added'], infer_datetime_format=True)
comments['comments_date_added'] = pd.to_datetime(comments['comments_date_added'], infer_datetime_format=True)
emails['emails_date_sent'] = pd.to_datetime(emails['emails_date_sent'], infer_datetime_format=True)
professionals['professionals_date_joined'] = pd.to_datetime(professionals['professionals_date_joined'], infer_datetime_format=True)
questions['questions_date_added'] = pd.to_datetime(questions['questions_date_added'], infer_datetime_format=True)
students['students_date_joined'] = pd.to_datetime(students['students_date_joined'], infer_datetime_format=True)


# In[ ]:


# Last Answer
temp = answers.groupby('answers_author_id')['answers_date_added'].max()
professionals['date_last_answer'] = pd.merge(professionals, pd.DataFrame(temp.rename('last_answer')), left_on='professionals_id', right_index=True, how='left')['last_answer']
# First Answer
temp = answers.groupby('answers_author_id')['answers_date_added'].min()
professionals['date_first_answer'] = pd.merge(professionals, pd.DataFrame(temp.rename('first_answer')), left_on='professionals_id', right_index=True, how='left')['first_answer']
# Last Comment
temp = comments.groupby('comments_author_id')['comments_date_added'].max()
professionals['date_last_comment'] = pd.merge(professionals, pd.DataFrame(temp.rename('last_comment')), left_on='professionals_id', right_index=True, how='left')['last_comment']
# First Comment
temp = comments.groupby('comments_author_id')['comments_date_added'].min()
professionals['date_first_comment'] = pd.merge(professionals, pd.DataFrame(temp.rename('first_comment')), left_on='professionals_id', right_index=True, how='left')['first_comment']
# Last Activity
professionals['date_last_activity'] = professionals[['date_last_answer', 'date_last_comment']].max(axis=1)
# First Activity
professionals['date_first_activity'] = professionals[['date_first_answer', 'date_first_comment']].min(axis=1)
# Last activity (Question)
temp = questions.groupby('questions_author_id')['questions_date_added'].max()
students['date_last_question'] = pd.merge(students, pd.DataFrame(temp.rename('last_question')), left_on='students_id', right_index=True, how='left')['last_question']
# First activity (Question)
temp = questions.groupby('questions_author_id')['questions_date_added'].min()
students['date_first_question'] = pd.merge(students, pd.DataFrame(temp.rename('first_question')), left_on='students_id', right_index=True, how='left')['first_question']
# Last activity (Comment)
temp = comments.groupby('comments_author_id')['comments_date_added'].max()
students['date_last_comment'] = pd.merge(students, pd.DataFrame(temp.rename('last_comment')), left_on='students_id', right_index=True, how='left')['last_comment']
# First activity (Comment)
temp = comments.groupby('comments_author_id')['comments_date_added'].min()
students['date_first_comment'] = pd.merge(students, pd.DataFrame(temp.rename('first_comment')), left_on='students_id', right_index=True, how='left')['first_comment']
# Last activity (Total)
students['date_last_activity'] = students[['date_last_question', 'date_last_comment']].max(axis=1)
# First activity (Total)
students['date_first_activity'] = students[['date_first_question', 'date_first_comment']].min(axis=1)


# In[ ]:


pro_emails = pd.merge(professionals, emails, how='inner', left_on='professionals_id', right_on='emails_recipient_id')
pro_emails = pro_emails[pro_emails['emails_frequency_level'] == 'email_notification_immediate']
pro_emails = pro_emails[['professionals_id', 'emails_id', 'emails_date_sent']]

pro_email_ques = pro_emails.merge(matches, left_on='emails_id', right_on='matches_email_id')
pro_email_ques = pro_email_ques.drop(columns=['emails_id', 'matches_email_id'])                  .set_index('professionals_id').rename(columns={'matches_question_id': 'questions_id'})


# In[ ]:


users = [
    ('students', students),
    ('professionals', professionals)
]

min_rel_date = '01-01-2016'
max_rel_date = '01-01-2019'

plt_data = {}

for user, df in users:
    df = df[(df['{}_date_joined'.format(user)] >= min_rel_date) & (df['{}_date_joined'.format(user)] <= max_rel_date)]
    df = (df['date_first_activity'] - df['{}_date_joined'.format(user)]).dt.days.fillna(10000).astype(int)
    df = df.groupby(df).size()/len(df)
    df = df.rename(lambda x: 0 if x < 0 else x)
    df = df.rename(lambda x: x if x <= 1 or x == 10000 else '> 1')
    df = df.rename({10000: 'NaN'})
    df = df.groupby(level=0).sum()

    plt_data[user] = df

plt_data = pd.DataFrame(plt_data)

plt_data.plot(kind='bar', figsize=(14, 5), colors=('#852ab2', '#21b7f2'))
plt.xlabel('Days', fontsize=15)
plt.ylabel('Ration', fontsize=15)
plt.title('Days before first activity after registration', fontsize=20)
leg = plt.legend(bbox_to_anchor=(1, 0.5), fontsize=15)
for text in leg.get_texts():
    plt.setp(text, color = 'w')
plt.show()


# # Last activity 
# Depending on the last comment, question or answer of a user, we have extracted the last activity date. On the previous plot we have seen, that many users haven't done any activity yet. For the 'last activity' plot we take a look only on users with already have one activity (dropna).

# In[ ]:


import datetime
from datetime import datetime


# In[ ]:


# Date of export
current_date = datetime(2019, 2 ,1)

users = [
    ('students', students),
    ('professionals', professionals)
]

plt_data = {}

for user, df in users:
    df = ((current_date - df['date_last_activity']).dt.days/30).dropna().astype(int)
    df = df.groupby(df).size()/len(df)
    df = df.rename(lambda x: 0 if x < 0 else x).rename(lambda x: x if x <= 30 or x == 10000 else '> 30').rename({10000:'NaN'})
    df = df.groupby(level=0).sum()

    plt_data[user] = df

plt_data = pd.DataFrame(plt_data)

plt_data.plot(kind='bar', figsize=(14, 5), colors=('#852ab2', '#21b7f2'))
plt.xlabel('Months', fontsize=15)
plt.ylabel('Ratio', fontsize=15)
plt.title('Last activity by Month', fontsize=20)
leg = plt.legend(loc='best', fontsize=15)
for text in leg.get_texts():
    plt.setp(text, color = 'w')
plt.show()


# # Number of Emails sent per year
# The number of emails sent yearly tends to grow each year.

# In[ ]:


min_date = emails['emails_date_sent'].min().strftime("%B %Y")
max_date = emails['emails_date_sent'].max().strftime("%B %Y")

emails['year'] = emails['emails_date_sent'].dt.year
plt_data = emails.groupby('year').size()

plt_data.plot(figsize=(14, 5), color='#f4d641',  marker='o')

x = plt_data.reset_index()['year']
y = plt_data.reset_index()[0]
plt.fill_between(x, y, color='#f4d641', alpha = 0.4)

plt.xlabel('Year', fontsize=20)
plt.ylabel('Emails Amount', fontsize=20)
plt.title('Number of Emails sent per year ({0}, {1})'.format(min_date, max_date), fontsize=20)
leg = plt.legend(loc='best', fontsize=15)
for text in leg.get_texts():
    plt.setp(text, color = 'w')
plt.show()


# # How many questions are contained in each email
# Most emails contain 1-3 questions, so an average amount of questions per email is 2.33.
# 
# Accessing questions is also possible directly from the Career Village website, so professionals are not restricted to answering emails, so contact method should not be assumed. However, we need inferred links between questions and professionals to build a recommender.

# In[ ]:


e_m = pd.DataFrame(pd.merge(emails, matches, how='inner', left_on='emails_id', right_on='matches_email_id').groupby('emails_id').size().reset_index()).rename(index=str, columns={0: "questions_amount"}).sort_values(by=['questions_amount'], ascending=False)
plt_data = e_m.groupby('questions_amount').size().reset_index().rename(index=str, columns={0: "emails_amount"})

mapping = {
    1: '1',
    2: '2',
    3: '3',
    4: '4 - 7',
    8: '8 - 10',
}

def get_key(x):
    for i in range(x, 0, -1):
        if i in mapping:
            return mapping[i]


plt_data['groups'] = plt_data['questions_amount'].apply(lambda x: '>10' if x >= 11 else get_key(x))
plt_data = pd.DataFrame({'groups' :['0'], 'emails_amount' : [len(emails) - len(e_m)]}).append(plt_data.groupby('groups').sum().reset_index()[['groups', 'emails_amount']])

plt_data.plot(kind='bar', figsize=(14, 5), color='#57c6b1')

plt.xticks(np.arange(len(plt_data)), tuple(plt_data['groups']))
plt.xlabel('Questions Count', fontsize=15)
plt.ylabel('Emails Count', fontsize=15)
plt.title('Questions contained in each email', fontsize=20)
leg = plt.legend(loc='best', fontsize=15)
for text in leg.get_texts():
    plt.setp(text, color = 'w')
plt.show()


# # Tags Wordclouds 
# In most of the cases, students are not using tags. Student tags are similar to questions tags. The current system is recommending questions tags, and they are not that similar to those which professionals are following.
# 
# Tags of questions and students and more generalized comparing to professionals tags. It means that even if we apply some processing and modeling techniques and deriving similarities out of it, there still be unmatched student and professionals using tags due to generalized vs. specialized tags problem.
# 
# Our model also solves this issue.

# In[ ]:


from wordcloud import WordCloud


# In[ ]:


entities = [
    ('students', students),
    ('professionals', professionals),
    ('questions', questions)
]

dfs = []

for entity, df in entities:
    if entity == 'questions':
        df = tag_questions
        df = pd.merge(df, tags, left_on='tag_questions_tag_id', right_on='tags_tag_id')
    else:
        df = tag_users[tag_users['tag_users_user_id'].isin(df['{}_id'.format(entity)])]
        df = pd.merge(df, tags, left_on='tag_users_tag_id', right_on='tags_tag_id')

    df['entity_type'] = entity

    dfs.append(df)


plt_data = pd.concat(dfs)

plt_data = plt_data[['tags_tag_name', 'entity_type']].pivot_table(index='tags_tag_name', columns='entity_type', aggfunc=len, fill_value=0)

for entity, df in entities:
    plt_data[entity] = plt_data[entity] / len(df)

plt_data['sum'] = (plt_data['professionals'] + plt_data['students'] + plt_data['questions'])
plt_data = plt_data.sort_values(by='sum', ascending=False).drop(['sum'], axis=1).head(100)


# Wordcloud
plt.figure(figsize=(20, 20))
wordloud_values = ['students', 'professionals', 'questions']
axisNum = 1
for wordcloud_value in wordloud_values:
    wordcloud = WordCloud(margin=0, max_words=20, random_state=42).generate_from_frequencies(plt_data[wordcloud_value])
    ax = plt.subplot(1, 3, axisNum)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(wordcloud_value)
    plt.axis("off")
    axisNum += 1
plt.show()    

