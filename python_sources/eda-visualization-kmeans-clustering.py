#!/usr/bin/env python
# coding: utf-8

# # Table Of Content

# #### [1. Data preparation](#data_preparation)
# #### [2. Professionals](#professionals)
# #### [3. Students](#students)
# #### [4. Answers](#answers)
# #### [5. Questions](#questions)
# #### [6. Tags](#tags)
# #### [7. Questions + Answers + Tags Analysis](#q_a_t)
# #### [8. Groups](#groups)
# #### [9. Schools](#schools)
# #### [10. Emails](#emails)
# #### [11. Matches](#matches)
# ### [12. Recommender](#recommender)

# # Data preparation <a class="anchor" id="data_preparation"></a>

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sb
import os
import re
from wordcloud import WordCloud
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly
import plotly.graph_objs as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

init_notebook_mode(connected=True)
get_ipython().run_line_magic('matplotlib', 'inline')

# load dataset
questions = pd.read_csv(os.path.join(os.getcwd(), '../input//questions.csv')) 
answers = pd.read_csv(os.path.join(os.getcwd(), '../input//answers.csv')) 
students = pd.read_csv(os.path.join(os.getcwd(), "../input//students.csv")) 
professionals = pd.read_csv(os.path.join(os.getcwd(), '../input//professionals.csv')) 
tag_questions = pd.read_csv(os.path.join(os.getcwd(), '../input//tag_questions.csv')) 
tags = pd.read_csv(os.path.join(os.getcwd(), '../input//tags.csv'))  
groups = pd.read_csv(os.path.join(os.getcwd(), '../input//groups.csv'))  
group_memberships = pd.read_csv(os.path.join(os.getcwd(), '../input//group_memberships.csv'))  
school_memberships = pd.read_csv(os.path.join(os.getcwd(), '../input//school_memberships.csv')) 
emails = pd.read_csv(os.path.join(os.getcwd(), '../input//emails.csv')) 
matches = pd.read_csv(os.path.join(os.getcwd(), '../input//matches.csv')) 
comments = pd.read_csv(os.path.join(os.getcwd(), '../input//comments.csv'))


# In[ ]:


# define some utils methods
def draw_bar(dt, s, head=20, sort=True):
    if sort:
        dt.groupby([s])[s].count().sort_values(ascending=False).head(head).plot(kind='bar', figsize=(15,8))
    else:
        dt.groupby([s])[s].count().head(head).plot(kind='bar', figsize=(15,8))
    
def draw_map(dt, s1, s2, sort=False):
    d1 = dt[s1]
    d2 = dt[s2]
    fig, ax = plt.subplots(figsize=(12,12))
    sb.heatmap(pd.crosstab(d1, d2).head(20), annot=True, ax=ax, fmt='d', linewidths=0.1)
    
def clean_body(raw_html):
    cleanr = re.compile('<.*?>|\\W|\\n|\\r')
    cleantext = re.sub(cleanr, ' ', raw_html)
    cleantext = re.sub('\\s{2,}', ' ', cleantext)
    return cleantext

def draw_cloud(s, n=100, what='str'):
    if what == 'str':
        cloud = WordCloud(width=1440, height=1080, max_words=n).generate(" ".join(s.astype(str)))
    else:
        cloud = WordCloud(width=1440, height=1080, max_words=n).generate_from_frequencies(s)
    plt.figure(figsize=(20, 15))
    plt.imshow(cloud)
    
def draw_plotly(x_1, y_1, x_2, y_2, l_name, r_name, title):
    trace0 = go.Bar(
        x=x_1,
        y=y_1,
        name=l_name
    )
    trace1 = go.Bar(
        x=x_2,
        y=y_2,
        name=r_name
    )

    data = [trace0, trace1]
    layout = {'title': title, 'xaxis': {'tickangle': 45}}

    fig = go.Figure(data=data, layout=layout)
    iplot(fig, show_link=False)


# # Professionals <a class="anchor" id="professionals"></a>

# **Let's take a look at the industry, location and headlines of professionals. As we can see, this is a free format and we can have the same elements, but with different names - in the lower and upper cases.**

# In[ ]:


professionals_lower = professionals.copy()


# In[ ]:


professionals_industry = professionals.groupby('professionals_industry')['professionals_industry'].count().sort_values(ascending=False).head(20)
professionals_lower['professionals_industry'] = professionals_lower['professionals_industry'].str.lower()
professionals_industry_lower = professionals_lower.groupby('professionals_industry')['professionals_industry'].count().sort_values(ascending=False).head(20)
draw_plotly(professionals_industry_lower.index, professionals_industry, professionals_industry_lower.index,
            professionals_industry_lower, 'Professionals industry', 'Professionals industry lowercas', 'Professionals Industry')


# In[ ]:


professionals_location = professionals.groupby('professionals_location')['professionals_location'].count().sort_values(ascending=False).head(20)
professionals_lower['professionals_location'] = professionals_lower['professionals_location'].str.lower()
professionals_location_lower = professionals_lower.groupby('professionals_location')['professionals_location'].count().sort_values(ascending=False).head(20)
draw_plotly(professionals_location_lower.index, professionals_location, professionals_location_lower.index,
           professionals_location_lower, 'Professionals location', 'Professionals location lowercase', 'Professionals Location')


# In[ ]:


professionals_headline = professionals.groupby('professionals_headline')['professionals_headline'].count().sort_values(ascending=False).head(20)
professionals_lower['professionals_headline'] = professionals_lower['professionals_headline'].str.lower()
professionals_headline_lower = professionals_lower.groupby('professionals_headline')['professionals_headline'].count().sort_values(ascending=False).head(20)
draw_plotly(professionals_headline_lower.index, professionals_headline, professionals_headline_lower.index,
           professionals_headline_lower, 'Professionals headlite', 'Professionals headline lowercase',
           'Professionals Headline')


# **It would be better to have some standard for these fields. This is not useful for students, but may be useful for future data analysis.**

# **Let's look at the distribution of the joining date by year and month**

# In[ ]:


professionals['year_joined'] = pd.to_datetime(professionals['professionals_date_joined']).dt.year


# In[ ]:


professionals['month_joined'] = pd.to_datetime(professionals['professionals_date_joined']).dt.month


# In[ ]:


professionals.shape


# In[ ]:


draw_map(professionals, 'year_joined', 'month_joined')


# In[ ]:


draw_bar(professionals, 'year_joined', sort=False)


# # Students <a class="anchor" id="students"></a>

# **Let's look at the distribution of the joining date by year and month**

# In[ ]:


students['year_joined'] = pd.to_datetime(students['students_date_joined']).dt.year


# In[ ]:


students['month_joined'] = pd.to_datetime(students['students_date_joined']).dt.month


# In[ ]:


draw_map(students, 'year_joined', 'month_joined')


# **And by year**

# In[ ]:


draw_bar(students, 'year_joined', sort=False)


# **Let's take a look at the students location. Seems to look good.**

# In[ ]:


students_location = students.groupby('students_location')['students_location'].count().sort_values(ascending=False).head(20)
students_lower = students.copy()
students_lower['students_location'] = students_lower['students_location'].str.lower()
students_location_lower = students_lower.groupby('students_location')['students_location'].count().sort_values(ascending=False).head(20)
draw_plotly(students_location_lower.index, students_location, students_location_lower.index, students_location_lower,
           'Students location', 'Students location lowercase', 'Students Location')


# # Answers <a class="anchor" id="answers"></a>

# **Let's look at the number of answers by months and years**

# In[ ]:


answers['year_added'] = pd.to_datetime(answers['answers_date_added']).dt.year


# In[ ]:


answers['month_added'] = pd.to_datetime(answers['answers_date_added']).dt.month


# In[ ]:


draw_map(answers, 'year_added', 'month_added')


# In[ ]:


draw_bar(answers, 'year_added', sort=False)


# **Combine proffessionals with answers for analysis**

# In[ ]:


prof_answ = pd.merge(answers, professionals, right_on='professionals_id', left_on='answers_author_id', how='left')


# **Let's look to most active professionals**

# In[ ]:


draw_bar(prof_answ, 'answers_author_id')


# **Let's see how many professionals have answers**

# In[ ]:


prof_answ['answers_author_id'].unique().size


# In[ ]:


professionals['professionals_id'].unique().size


# In[ ]:


u1 = prof_answ.drop(['answers_id', 'answers_author_id', 'answers_question_id', 'answers_date_added', 'answers_body', 'month_added', 'professionals_date_joined', 'year_added'], axis=1).drop_duplicates()


# In[ ]:


profs = professionals.groupby('year_joined')[['professionals_id']].count().join(
    u1.groupby('year_joined')[['professionals_id']].count(), lsuffix='_left', rsuffix='_right')

draw_plotly(profs.index, profs.professionals_id_left, profs.index, profs.professionals_id_right, 
            'All professionals', 'Active professionals', 'Professionals')


# **Doesn't look good. Most of the professionals are not active. Perhaps you should do a survey for inactive professionals and find out the reason.**

# **Let's look top 20 active and not active professionais by location** 

# In[ ]:


profs = professionals.groupby('professionals_location')[['professionals_id']].count().nlargest(30, columns=['professionals_id']).join(
    u1.groupby('professionals_location')[['professionals_id']].count().nlargest(30, columns=['professionals_id']), lsuffix='_left', rsuffix='_right')
draw_plotly(profs.index, profs.professionals_id_left, profs.index, profs.professionals_id_right, 'All professionals', 
           'Active professionals', "Top 30. Professional's activity by location")


# In[ ]:


p = professionals.groupby('professionals_location')[['professionals_id']].count().join(
    u1.groupby('professionals_location')[['professionals_id']].count(), lsuffix='_left', rsuffix='_right')
profs = p[p['professionals_id_right'].isna()].sort_values(by=['professionals_id_left'], ascending=False).head(30)
draw_plotly(profs.index, profs.professionals_id_left, profs.index, profs.professionals_id_right, 'All professionals', 
           'Active professionals', "Top 30 not active professionals by location")


# **Let's look how many active and not active professionals by industry**

# In[ ]:


profs = professionals.groupby('professionals_industry')[['professionals_id']].count().nlargest(30, columns=['professionals_id']).join(
    u1.groupby('professionals_industry')[['professionals_id']].count().nlargest(30, columns=['professionals_id']), lsuffix='_left', rsuffix='_right')
draw_plotly(profs.index, profs.professionals_id_left, profs.index, profs.professionals_id_right, 'All professionals', 
           'Active professionals', "Top 30. Professional's activity by industry")


# In[ ]:


p = professionals.groupby('professionals_industry')[['professionals_id']].count().join(
    u1.groupby('professionals_industry')[['professionals_id']].count(), lsuffix='_left', rsuffix='_right')
profs = p[p['professionals_id_right'].isna()].sort_values(by=['professionals_id_left'], ascending=False).head(30)
draw_plotly(profs.index, profs.professionals_id_left, profs.index, profs.professionals_id_right, 'All professionals', 
           'Active professionals', "Top 30 not active professionals by industry")


# **Let's look how many active and not active professionals by headline**

# In[ ]:


profs = professionals.groupby('professionals_headline')[['professionals_id']].count().nlargest(30, columns=['professionals_id']).join(
    u1.groupby('professionals_headline')[['professionals_id']].count().nlargest(30, columns=['professionals_id']), lsuffix='_left', rsuffix='_right')
draw_plotly(profs.index, profs.professionals_id_left, profs.index, profs.professionals_id_right, 'All professionals', 
           'Active professionals', "Top 30. Professional's activity by headline")


# In[ ]:


p = professionals.groupby('professionals_headline')[['professionals_id']].count().join(
    u1.groupby('professionals_headline')[['professionals_id']].count(), lsuffix='_left', rsuffix='_right')
profs = p[p['professionals_id_right'].isna()].sort_values(by=['professionals_id_left'], ascending=False).head(30)
draw_plotly(profs.index, profs.professionals_id_left, profs.index, profs.professionals_id_right, 'All professionals', 
           'Active professionals', "Top 30 not active professionals by headline")


# **Let's look to word cloud for answers.**

# In[ ]:


prof_answ['answers_body'] = prof_answ['answers_body'].apply(lambda x: clean_body(str(x)))
draw_cloud(prof_answ['answers_body'])


# # Questions <a class="anchor" id="questions"></a>

# **Lets check the duplicates of body and title**

# In[ ]:


draw_bar(questions, 'questions_body')


# In[ ]:


draw_bar(questions, 'questions_title')


# **We need more details.**

# In[ ]:


questions.groupby(['questions_body', 'questions_author_id'])['questions_body'].count().sort_values(ascending=False).head(10).T


# In[ ]:


questions[questions['questions_author_id'] == 'c17fb778ae734737b08f607e75a87460'].sort_values(by='questions_date_added').head(20).T


# **So, our hero 'c17fb778ae734737b08f607e75a87460' is very impatient. The question was duplicated 10 times in the interval 2017-03-14 22:01:34 - 2017-03-14 22:29:50 with a frequency of ~30 seconds and 10 times in the interval 2017-03-20 14:21:15 - 2017-03-20 14:29:12 with the same frequency. Interestingly, the title of the question changes, but not the body. I would suggest that in this case someone used (perhaps for the test) some kind of bot. Since we don't have many duplicates, it is not very important, but it may be useful for behavior analysis.**

# **Let's look at the number of questions by months and years**

# In[ ]:


questions['year_added'] = pd.to_datetime(questions['questions_date_added']).dt.year


# In[ ]:


questions['month_added'] = pd.to_datetime(questions['questions_date_added']).dt.month


# In[ ]:


draw_map(questions, 'month_added', 'year_added')


# **And by year**

# In[ ]:


draw_bar(questions, 'year_added', sort=False)


# **Let's see how many questions and answers for years.**

# In[ ]:


a = answers.groupby('year_added')['year_added'].count().head(20)
q = questions.groupby('year_added')['year_added'].count().head(20)
draw_plotly(a.index, a, q.index, q, 'Answers', 'Questions', 'Questions vs Answers')


# **Seems to look good. But we'll check later to see if we have any unanswered questions**

# **Let's look to most used words in the question's title and body.**

# In[ ]:


draw_cloud(questions['questions_title'])


# In[ ]:


draw_cloud(questions['questions_body'])


# **Well, most of questions about work, career and job. Unexpected ))**

# **Let's look at the most curious students.**

# In[ ]:


stud_quest = pd.merge(questions, students, right_on='students_id', left_on='questions_author_id', how='inner')


# In[ ]:


draw_bar(stud_quest, 'students_id', 40)


# # Tags <a class="anchor" id="tags"></a>

# **Let's see how many tags we have**

# In[ ]:


tags.shape


# In[ ]:


tags['tags_tag_name'].unique().size


# In[ ]:


tag_questions.shape


# In[ ]:


tags_with_name = pd.merge(tags, tag_questions, left_on='tags_tag_id', right_on='tag_questions_tag_id', how='inner')


# In[ ]:


tags_with_name['tags_tag_name'].unique().size


# **So, we have 16269 tags, but for questions used 7091 tags only. Seems, a lot of tags are used for comments and answers, but we don't have such data for analysis.**

# **Let's see most used tags**

# In[ ]:


draw_bar(tags_with_name, 'tags_tag_name')


# In[ ]:


draw_cloud(tags_with_name['tags_tag_name'])


# ### Unused tags for questions

# In[ ]:


unused_tags_id = set(tags['tags_tag_id'].tolist()) - set(tag_questions['tag_questions_tag_id'].tolist())


# In[ ]:


len(unused_tags_id)


# In[ ]:


unused_tags = tags[tags['tags_tag_id'].isin(unused_tags_id)]


# In[ ]:


draw_cloud(unused_tags['tags_tag_name'])


# **It's strange, seems we have the same used and unused tags. Need more details. Lets check tag 'college'.**

# In[ ]:


tags[tags['tags_tag_name'].str.contains('college', regex=False, na=False)].head(20)


# In[ ]:


unused_tags[unused_tags['tags_tag_name'].str.contains('college', regex=False, na=False)].head(20)


# **We have tag 'college' in the used tags and tags '#college', '#-college', 'college-' and 'college#' in the unused tags. In my opinion, there is not enough formatting for tags. It may be better to define the format for the tags, it may be useful for searching by tags.**

# # Questions + Answers + Tags Analysis <a class="anchor" id="q_a_t"></a>

# **Let's merge questions, answers and tags**

# In[ ]:


answ_quest = pd.merge(answers, questions, left_on='answers_question_id', right_on='questions_id', how='outer', suffixes=('_answ', '_quest'))
answ_quest_tags = pd.merge(answ_quest, tags_with_name, left_on='questions_id', right_on='tag_questions_question_id', how='outer')


# **Let's group and sort the questions by number of answers**

# In[ ]:


draw_bar(answ_quest, 'questions_id')


# **Let's check questions without answers**

# In[ ]:


without_answer = answ_quest[answ_quest['answers_id'].isna()]


# In[ ]:


draw_map(without_answer, 'month_added_quest', 'year_added_quest')


# **As we can see, the largest number of unanswered questions were in 2018**

# In[ ]:


unanswered = answ_quest_tags[answ_quest_tags['answers_id'].isna()]


# **Let's look at the tags for unanswered questions**

# **In terms of words**

# In[ ]:


draw_cloud(unanswered['tags_tag_name'])


# **And in terms of tags**

# In[ ]:


unansw_tag_list = unanswered['tags_tag_name'].tolist()
freq_unansw_tags = {str(x): unansw_tag_list.count(x) for x in unansw_tag_list}
draw_cloud(freq_unansw_tags, what='freq')


# **As we can see, most unanswered questions have medical or biological tags**

# **Let's look at the title and body for unanswered questions**

# In[ ]:



draw_cloud(unanswered['questions_title'])


# In[ ]:


draw_cloud(unanswered['questions_body'])


# **As we can see, most of the unanswered questions have a medical subject**

# **Seems like you need to invite more medical professionals**

# # Groups <a class="anchor" id="groups"></a>

# **Let's see what types of groups are represented in the data**

# In[ ]:


groups['groups_group_type'].unique()


# **How many groups in each type**

# In[ ]:


draw_bar(groups, 'groups_group_type')


# **How many groups has each member**

# In[ ]:


draw_bar(group_memberships, 'group_memberships_user_id')


# **Let's see how many professionals and students in the groups**

# In[ ]:


prof_group = pd.merge(group_memberships, professionals, left_on='group_memberships_user_id', right_on='professionals_id')


# In[ ]:


stud_group = pd.merge(group_memberships, students, left_on='group_memberships_user_id', right_on='students_id')


# In[ ]:


draw_bar(prof_group, 'year_joined')


# In[ ]:


draw_bar(stud_group, 'year_joined')


# **How many unanswered questions in each group type**

# In[ ]:


group_answ = pd.merge(prof_group, answers, left_on='professionals_id', right_on='answers_author_id', how='inner')
prof_group_unansw = set(prof_group['professionals_id'].tolist()) - set(group_answ['answers_author_id'].tolist())
groups_members_and_type = pd.merge(groups, group_memberships, left_on='groups_id', right_on='group_memberships_group_id')


# In[ ]:


draw_bar(groups_members_and_type[groups_members_and_type['group_memberships_user_id'].isin(prof_group_unansw)], 'groups_group_type')


# # Schools <a class="anchor" id="schools"></a>

# **Let's see how many members per school id**

# In[ ]:


draw_bar(school_memberships, 'school_memberships_school_id')


# **Let's see how many professionals and students indicated school**

# In[ ]:


prof_school = pd.merge(school_memberships, professionals, left_on='school_memberships_user_id', right_on='professionals_id')


# In[ ]:


stud_school = pd.merge(school_memberships, students, left_on='school_memberships_user_id', right_on='students_id')


# In[ ]:


prof_school.shape


# In[ ]:


stud_school.shape


# **4283 professionals and 1355 students**

# **Let's see how many professionals and students who have indicated the school has answers**

# In[ ]:


school_answ = pd.merge(school_memberships, answers, left_on='school_memberships_user_id', right_on='answers_author_id', how='inner')
school_answ.shape


# In[ ]:


prof_school_unansw = school_answ[school_answ['answers_author_id'].isna()]
prof_school_unansw.shape


# **Interesting observation - all those who have indicated the school are active participants**

# **Let's see how many answers has each school(top 20)**

# In[ ]:


draw_bar(school_answ, 'school_memberships_school_id')


# # Emails <a class="anchor" id="emails"></a>

# In[ ]:


emails.shape


# **Let's look how many e-mails sent by year/month and by year**

# In[ ]:


emails['year_sent'] = pd.to_datetime(emails['emails_date_sent']).dt.year


# In[ ]:


emails['month_sent'] = pd.to_datetime(emails['emails_date_sent']).dt.month


# In[ ]:


draw_map(emails, 'month_sent', 'year_sent')


# In[ ]:


draw_bar(emails, 'year_sent')


# **What is emails_frequency_level?**

# In[ ]:


emails['emails_frequency_level'].unique()


# **Let's look how many e-mails sent by frequency level**

# In[ ]:


draw_bar(emails, 'emails_frequency_level')


# **Let's look how many e-mails sent by emails_recipient_id**

# In[ ]:


draw_bar(emails, 'emails_recipient_id')


# **Let's take a look at the top five email recipients**

# In[ ]:


top_5 = emails.groupby('emails_recipient_id')['emails_recipient_id'].count().sort_values(ascending=False).head(5).index


# In[ ]:


professionals[professionals['professionals_id'].isin(top_5)].head()


# **And how many answers do they have**

# In[ ]:


answers[answers['answers_author_id'].isin(top_5)].groupby('answers_author_id')['answers_author_id'].count()


# **There is our hero 36ff3b3666df400f956f8335cf53e09e with biggest count of answers, but why do others get so many emails? And the hero-receiver of e-mails 0079e89bf1544926b98310e81315b9f1  has no answer at all. I think, you need check last user activity and not to send emails to people who haven't logged in a long time. Perhaps because of the number of emails they are blocked by some spam filter. It is better to send some invitation.**

# # Matches <a class="anchor" id="matches"></a>

# **Let's look at the number of e-mails sent by year and month**

# In[ ]:


draw_map(questions[questions['questions_id'].isin(matches['matches_question_id'].unique())], 'month_added', 'year_added')


# **How many questions were sent for each email (top 20)** 

# In[ ]:


draw_bar(matches, 'matches_email_id')


# **How many emails were sent for each question (top 20)**

# In[ ]:


draw_bar(matches, 'matches_question_id')


# **Let's check that all questions were sent by email**

# In[ ]:


questions.shape


# In[ ]:


matches['matches_question_id'].unique().size


# **Looks like not all the questions were sent out. We need to do a deeper investigation**

# In[ ]:


quest_tags = pd.merge(questions, tags_with_name, left_on='questions_id', right_on='tag_questions_question_id', how='inner')
not_sent_ids = set(quest_tags['questions_id'].tolist()) - set(matches['matches_question_id'].tolist())
not_sent_quest = quest_tags[quest_tags['questions_id'].isin(not_sent_ids)].dropna()


# In[ ]:


not_sent_quest.head(10).T


# **Let's look tags.**

# In[ ]:



not_sent_quest.groupby('tags_tag_name')['tags_tag_name'].count().sort_values(ascending=False).head(20)


# **Unanswered questions seem to relate to areas where you do not have professionals - medicine, law, art, etc. Or just we have not full data**

# # Reocommender <a class="anchor" id="recommender"></a>

# **I'll use clustering. First we need to prepare the data. To identify the most suitable professionals, we need the body of the question, the tags and the professionals themselves. 
# Let's prepare the data. **

# In[ ]:


q_copy = questions.copy()
a_copy = answers.copy()
p_copy = professionals.copy()
q_copy.drop(['questions_date_added', 'questions_author_id', 'questions_title'], axis=1, inplace=True)
a_copy.drop(['answers_date_added', 'answers_body'], axis=1, inplace=True)
p_copy.drop(['professionals_location', 'professionals_industry', 'professionals_headline', 'professionals_date_joined'], axis=1, inplace=True)
a_p = pd.merge(a_copy, p_copy, left_on='answers_author_id', right_on='professionals_id')
a_p.drop(['answers_id', 'answers_author_id'], axis=1, inplace=True)
t = pd.merge(tags, tag_questions, left_on='tags_tag_id', right_on='tag_questions_tag_id')
t.drop(['tags_tag_id', 'tag_questions_tag_id'], axis=1, inplace=True)


# **I also propose to exclude from the analysis some tags that do not define the subject matter of the question. You can increase or decrease this list**

# In[ ]:


stop_tags = ['college', 'career', 'college-major ', 'career-counseling', 'scholarships', 'jobs', 'college-advice', 
             'double-major', 'chef', 'college-minor', 'college-applications', 'college-student', 'school', 
             'college-admissions', 'career-choice', 'university', 'job', 'college-major', 'any', 'student', 
             'professional', 'graduate-school', 'career-path', 'career-paths', 'college-majors', 'career-details', 
             'work', 'college-bound', 'success', 'studying', 'first-job', 'life', 'classes', 'resume', 'job-search']


# **And we will take the 200 most used tags. I tried to increase and decrease this value - 200 tags give the best result.**

# In[ ]:


most_used_tags = t[~t['tags_tag_name'].isin(stop_tags)].groupby('tags_tag_name')['tags_tag_name'].count().nlargest(200).index


# **Now we need to combine and clear the data we need**

# In[ ]:


a_q_p = pd.merge(a_p, questions, left_on='answers_question_id', right_on='questions_id')
a_q_p.drop(['answers_question_id'], axis=1, inplace=True)
a_q_p_t = pd.merge(t, a_q_p, left_on='tag_questions_question_id', right_on='questions_id', how='inner')
a_q_p_t.drop(['tag_questions_question_id', 'questions_id'], axis=1, inplace=True)
filtered = a_q_p_t[a_q_p_t['tags_tag_name'].isin(most_used_tags)]
filtered = filtered.copy()
filtered.loc[:, 'questions_body'] = filtered['questions_body'].map(clean_body)
filtered = filtered.fillna('')


# **We need data, labels and number of clusters**

# In[ ]:


labels = filtered['tags_tag_name']
data = filtered['questions_body']
n_clusters = np.unique(labels).shape[0]


# **Suppose we have new questions and tags for them. And we want to identify the professionals who will answer these questions. And we have prepared data from previous questions. We need to add new questions to the existing ones, cluster the questions and get a list of the most relevant professionals to the topic of the question.**

# **Let these be the new questions we've added to our data.**

# In[ ]:


test_data_1 = data.iloc[11]
test_tag_1 = labels.iloc[11]
test_data_2 = data.iloc[40]
test_tag_2 = labels.iloc[40]
print(test_data_1)
print(test_tag_1)
print()
print(test_data_2)
print(test_tag_2)


# In[ ]:


vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000, min_df=2, stop_words='english',use_idf=True)
X = vectorizer.fit_transform(data)


# In[ ]:


km = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, n_init=1)


# In[ ]:


print("Clustering sparse data with %s" % km)
get_ipython().run_line_magic('time', 'km.fit(X)')


# **Now we need to combine the clusters we've got with the professionals**

# In[ ]:


clusters = km.labels_.tolist()
quest = { 'tags': filtered['tags_tag_name'].tolist(), 'professionals_id': filtered['professionals_id'].tolist(), 'question': filtered['questions_body'].tolist(), 'cluster': clusters }
frame = pd.DataFrame(quest, index = [clusters] , columns = ['tags', 'professionals_id', 'question', 'cluster'])


# **And find new questions in the clusters**

# In[ ]:


result = dict()

for i in range(n_clusters):
    tags = [tag for tag in frame.loc[i]['tags'].unique()]
    profs = [prof for prof in frame.loc[i]['professionals_id'].unique()]
    quests = [quest for quest in frame.loc[i]['question'].unique()]
        
    if test_data_1 in quests:
        result['test_data_1'] = test_data_1
        result['test_tag_1'] = test_tag_1
        result['tags_1'] = tags
        result['profs_1'] = profs
    if test_data_2 in quests:
        result['test_data_2'] = test_data_2
        result['test_tag_2'] = test_tag_2
        result['tags_2'] = tags
        result['profs_2'] = profs


# **Let's look at the result**

# **Question #1**

# In[ ]:


print(result.get('test_data_1'))
print()
print(result.get('test_tag_1'))
print(len(result.get('profs_1')))


# **To check the result, I filtered out the professional ID from the resulting cluster and chose the top 10 questions, which are answered more.**

# In[ ]:


a_q_p_t[a_q_p_t['professionals_id'].isin(result.get('profs_1'))].groupby('questions_body')['questions_body'].count().sort_values(ascending=False).head(10)


# **Question #2**

# In[ ]:


print(result.get('test_data_2'))
print()
print(result.get('test_tag_2'))
print(len(result.get('profs_2')))


# In[ ]:


a_q_p_t[a_q_p_t['professionals_id'].isin(result.get('profs_2'))].groupby('questions_body')['questions_body'].count().sort_values(ascending=False).head(10)


# **Let's see how many proffesionals we have for each question.**

# In[ ]:


print(len(result.get('profs_1')))
print(result.get('profs_1'))
print()
print(len(result.get('profs_2')))
print(result.get('profs_2'))


# **So, you just can send emails for these proffesionals**

# **It's very easy solution and you can play with tags and cluster size. Looks not bad, though it doesn't give 100% accuracy. 
# Note: A lot of questions that don't make sense, but completely consist of tags.**

# In[ ]:




