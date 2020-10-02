#!/usr/bin/env python
# coding: utf-8

# **Thanks for viewing my Kernel! If you like my work and find it useful, please leave an upvote! :)**
# 
# CareerVillage.org is a nonprofit that crowdsources career advice for underserved youth. The platform uses a Q&A style similar to StackOverflow or Quora to provide students with answers to any question about any career. Underserved youth lack the network to find their career role models, making CareerVillage.org the only option for millions of young people in America and around the globe with nowhere else to turn. 
# 
# The objective of this competition is to recommend questions to appropriate volunteers using 5 years of data.

# In[ ]:


from IPython.display import Image
Image(filename="../input/cv-logo/cv.jpg")


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from collections import defaultdict
from wordcloud import WordCloud, STOPWORDS
import random

from nltk.corpus import stopwords
stop = stopwords.words('english')

import os
for file in os.listdir("../input/data-science-for-good-careervillage"):
    print(file)


# In[ ]:


def ngram_extractor(text, n_gram):
    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]

# Function to generate a dataframe with n_gram and top max_row frequencies
def generate_ngrams(df, col, n_gram, max_row):
    temp_dict = defaultdict(int)
    for question in df[col].dropna():
        for word in ngram_extractor(question, n_gram):
            temp_dict[word] += 1
    temp_df = pd.DataFrame(sorted(temp_dict.items(), key=lambda x: x[1])[::-1]).head(max_row)
    temp_df.columns = ["word", "wordcount"]
    return temp_df


# Let's analyze the matches first. It contains the data on which professionals receive which question over e-mail. 

# In[ ]:


from IPython.display import Image
Image(filename="../input/cv-matches/matches.png")


# In[ ]:


professionals = pd.read_csv('../input/data-science-for-good-careervillage/professionals.csv')
print('Professionals data: \nRows: {}\nCols: {}\n'.format(professionals.shape[0],professionals.shape[1]))

emails = pd.read_csv('../input/data-science-for-good-careervillage/emails.csv')
print('E-Mails data: \nRows: {}\nCols: {}\n'.format(emails.shape[0],emails.shape[1]))

matches = pd.read_csv('../input/data-science-for-good-careervillage/matches.csv')
print('Matches data: \nRows: {}\nCols: {}\n'.format(matches.shape[0],matches.shape[1]))

students = pd.read_csv('../input/data-science-for-good-careervillage/students.csv')
print('Students data: \nRows: {}\nCols: {}\n'.format(students.shape[0],students.shape[1]))

questions = pd.read_csv('../input/data-science-for-good-careervillage/questions.csv')
print('Questions data: \nRows: {}\nCols: {}\n'.format(questions.shape[0],questions.shape[1]))

emails_professionals = pd.merge(emails, 
                                professionals,
                                how='left',
                                left_on='emails_recipient_id', 
                                right_on='professionals_id')

matches_new = pd.merge(matches, 
                       emails_professionals,
                       how='left',
                       left_on='matches_email_id',
                       right_on='emails_id')

questions_students = pd.merge(questions, 
                              students,
                              how='left',
                              left_on='questions_author_id', 
                              right_on='students_id')

matches_new = pd.merge(matches_new, 
                       questions_students,
                       how='left',
                       left_on='matches_question_id',
                       right_on='questions_id')

matches_new.drop(columns=['matches_email_id', 'matches_question_id', 'emails_recipient_id','questions_author_id'], inplace=True)

print('New matches data: \nRows: {}\nCols: {}'.format(matches_new.shape[0],matches_new.shape[1]))


# In[ ]:


# Lowercasing
questions['questions_title'] = questions['questions_title'].apply(lambda x: " ".join(x.lower() for x in x.split()))

# Removing punctuations
questions['questions_title'] = questions['questions_title'].str.replace('[^\w\s]','')

# Removing stop words
questions['questions_title'] = questions['questions_title'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))


# **Dataset: Professionals**
# 
# There are 28,152 professionals. We have data related to their location, industry, headline and when they joined. Almost 10% of their data is unknown. 

# In[ ]:


print("Missing values in Professionals data")
for x in professionals.columns:
    if professionals[x].isnull().values.ravel().sum() > 0:
        print('{} - {}'.format(x,professionals[x].isnull().values.ravel().sum()))


# New York tops the location of professionals

# In[ ]:


temp = professionals['professionals_location'].value_counts(normalize=True) * 100
temp = temp.reset_index().head(10)

f, ax = plt.subplots(figsize=(10, 4))
sns.barplot(x="professionals_location", y="index", data=temp, label="index", color="palegreen")

for p in ax.patches:
    ax.text(p.get_width()+.15,
            p.get_y() + (p.get_height()/2) + .1,
            '{:1.1f}%'.format(p.get_width()),
            ha="center")

ax.set_xlabel('% of professionals', size=10, color="green")
ax.set_ylabel('Location', size=10, color="green")
ax.set_title('[Horizontal Bar Plot] % of professionals from top 10 locations', size=12, color="green")
plt.show()


# Telecommunications, Information Technology & Services and Computer Software are the industries with high number of the professionals

# In[ ]:


temp = professionals['professionals_industry'].value_counts(normalize=True) * 100
temp = temp.reset_index().head(10)

f, ax = plt.subplots(figsize=(10, 4))
sns.barplot(x="professionals_industry", y="index", data=temp, label="index", color="palegreen")

for p in ax.patches:
    ax.text(p.get_width()+.4,
            p.get_y() + (p.get_height()/2) + .1,
            '{:1.1f}%'.format(p.get_width()),
            ha="center")

ax.set_xlabel('% of professionals', size=10, color="green")
ax.set_ylabel('Industry', size=10, color="green")
ax.set_title('[Horizontal Bar Plot] % of professionals from top 10 industries', size=12, color="green")
plt.show()


# Top headlines used are student and director.

# In[ ]:


professionals["professionals_headline"] = professionals['professionals_headline'].str.replace('[^\w\s]','')

def grey_color_func(word, font_size, position, orientation, random_state=None,**kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(0, 100)

word_string = professionals["professionals_headline"].str.cat(sep=' ')

wordcloud = WordCloud(
    stopwords=STOPWORDS,
    background_color='white',
    width=3000,
    height=1000).generate(word_string)

plt.figure(figsize=(20,40))
plt.imshow(wordcloud.recolor(color_func=grey_color_func, random_state=3),interpolation="bilinear")
plt.axis('off')
font = {'family': 'sans-serif',
        'color':  'brown',
        'weight': 'normal',
        'size': 32
        }
plt.title('Word Cloud - Professionals Headline', fontdict=font)
plt.show()


# **Dataset: Groups**
# 
# There are 49 groups. 

# In[ ]:


groups = pd.read_csv('../input/data-science-for-good-careervillage/groups.csv')

print('Groups data: \nRows: {}\nCols: {}'.format(groups.shape[0],groups.shape[1]))
print(groups.columns)


# There are no missing values.

# In[ ]:


print("Missing values in Groups data")
for x in groups.columns:
    if groups[x].isnull().values.ravel().sum() > 0:
        print('{} - {}'.format(x,groups[x].isnull().values.ravel().sum()))


# 2/3rd of the groups are youth programs. 1/6th of the groups are professional networks.

# In[ ]:


temp = groups['groups_group_type'].value_counts(normalize=True) * 100
temp = temp.reset_index()

f, ax = plt.subplots(figsize=(10, 4))
sns.barplot(x="groups_group_type", y="index", data=temp, label="index", color="cyan")

for p in ax.patches:
    ax.text(p.get_width()+2,
            p.get_y() + (p.get_height()/2) + .1,
            '{:1.1f}%'.format(p.get_width()),
            ha="center")

ax.set_xlabel('% of groups', size=10, color="darkcyan")
ax.set_ylabel('Group Type', size=10, color="darkcyan")
ax.set_title('[Horizontal Bar Plot] % of groups', size=12, color="darkcyan")
plt.show()


# **Dataset: Comments**
# 
# There are 14,966 comments in the dataset. 

# In[ ]:


comments = pd.read_csv('../input/data-science-for-good-careervillage/comments.csv')

comments['comments_date'] = pd.to_datetime(comments['comments_date_added'])
comments['day'] = comments['comments_date'].dt.day
comments['month'] = comments['comments_date'].dt.month
comments['year'] = comments['comments_date'].dt.year
comments['hour'] = comments['comments_date'].dt.hour
comments['minute'] = comments['comments_date'].dt.minute
comments['second'] = comments['comments_date'].dt.second
comments['day_of_week'] = comments['comments_date'].dt.dayofweek

print('Comments data: \nRows: {}\nCols: {}'.format(comments.shape[0],comments.shape[1]))
print(comments.columns)


# In[ ]:


print("Missing values in Comments data")
for x in comments.columns:
    if comments[x].isnull().values.ravel().sum() > 0:
        print('{} - {}'.format(x,comments[x].isnull().values.ravel().sum()))


# Thank(s), much, will, answer, advice, great, school, work and good are the most present words in the comments. It's pretty obvious though. 

# In[ ]:


comments["comments_body"] = comments['comments_body'].str.replace('[^\w\s]','')

temp = generate_ngrams(comments,'comments_body',1,10)

f, ax = plt.subplots(figsize=(10, 4))
sns.barplot(x="wordcount", y="word", data=temp, label="wordcount", color="silver")

for p in ax.patches:
    ax.text(p.get_width() + 170,
            p.get_y() + (p.get_height()/2) + .1,
            '{:1.0f}'.format(p.get_width()),
            ha="center")

ax.set_xlabel('Count of word', size=10, color="black")
ax.set_ylabel('Word in comments', size=10, color="black")
ax.set_title('[Horizontal Bar Plot] Count of words in comments', size=12, color="black")
plt.show()


# A quick word cloud on the same data. 

# In[ ]:


def grey_color_func(word, font_size, position, orientation, random_state=None,**kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(0, 100)

word_string = comments["comments_body"].str.cat(sep=' ')

wordcloud = WordCloud(
    stopwords=STOPWORDS,
    background_color='white',
    width=3000,
    height=1000).generate(word_string)

plt.figure(figsize=(20,40))
plt.imshow(wordcloud.recolor(color_func=grey_color_func, random_state=3),interpolation="bilinear")
plt.axis('off')
font = {'family': 'sans-serif',
        'color':  'brown',
        'weight': 'normal',
        'size': 32
        }
plt.title('Word Cloud - Comment Body', fontdict=font)
plt.show()


# Comments (and in turn overall activity) is higher towards the end of the month. It's an expected behaviour. 

# In[ ]:


temp = comments['day'].value_counts()
temp = temp.reset_index()

f, ax = plt.subplots(figsize=(12, 4))
sns.barplot(x="index", y="day", data=temp, label="index", color="silver", orient='v')

for p in ax.patches:
    ax.text(p.get_x() + (p.get_width()/2),
            p.get_height() + 10,
            '{:1.0f}'.format(p.get_height()),
            ha="center")

ax.set_xlabel('Day', size=10, color="black")
ax.set_ylabel('Count of Comments', size=10, color="black")
ax.set_title('[Vertical Bar Plot] Count of Comments over each day', size=12, color="black")
plt.show()


# May month, which is the end of the academic year, is when the comments (activity) are higher.

# In[ ]:


temp = comments['month'].value_counts()
temp = temp.reset_index()

f, ax = plt.subplots(figsize=(12, 4))
sns.barplot(x="index", y="month", data=temp, label="index", color="silver", orient='v')

for p in ax.patches:
    ax.text(p.get_x() + (p.get_width()/2),
            p.get_height() + 30,
            '{:1.0f}'.format(p.get_height()),
            ha="center")

ax.set_xlabel('Month', size=10, color="black")
ax.set_ylabel('Count of Comments', size=10, color="black")
ax.set_title('[Vertical Bar Plot] Count of Comments over each month', size=12, color="black")
plt.show()


# 2016 is the year when there was a lot of activity. Is there a specific reason to it? It should be explored.

# In[ ]:


temp = comments['year'].value_counts()
temp = temp.reset_index()

f, ax = plt.subplots(figsize=(12, 3))
sns.barplot(x="year", y="index", data=temp, label="index", color="silver", orient='h')

for p in ax.patches:
    ax.text(p.get_width() + 120,
            p.get_y() + (p.get_height()/1.2),
            '{:1.0f}'.format(p.get_width()),
            ha="center")

ax.set_xlabel('Count of Comments', size=10, color="black")
ax.set_ylabel('Year', size=10, color="black")
ax.set_title('[Horizontal Bar Plot] Count of Comments over each year', size=12, color="black")
plt.show()


# The comments rise from Monday to peak at Weednesday and then drops off. 

# In[ ]:


temp = comments['day_of_week'].value_counts()
temp = temp.reset_index()

f, ax = plt.subplots(figsize=(12, 3))
sns.barplot(x="day_of_week", y="index", data=temp, label="index", color="silver", orient='h')

for p in ax.patches:
    ax.text(p.get_width() + 80,
            p.get_y() + (p.get_height()/1.4),
            '{:1.0f}'.format(p.get_width()),
            ha="center")

ax.set_xlabel('Count of Comments', size=10, color="black")
ax.set_ylabel('Day of week', size=10, color="black")
ax.set_title('[Horizontal Bar Plot] Count of Comments over the weekday', size=12, color="black")
plt.show()


# Comments peak at 17:00. Peak hours are from 15:00 to 19:00. 

# In[ ]:


temp = comments['hour'].value_counts()
temp = temp.reset_index()

f, ax = plt.subplots(figsize=(12, 4))
sns.barplot(x="index", y="hour", data=temp, label="index", color="silver", orient='v')

for p in ax.patches:
    ax.text(p.get_x() + (p.get_width()/2),
            p.get_height() + 10,
            '{:1.0f}'.format(p.get_height()),
            ha="center")

ax.set_xlabel('Hour of the day', size=10, color="black")
ax.set_ylabel('Count of Comments', size=10, color="black")
ax.set_title('[Vertical Bar Plot] Count of Comments over hour of the day', size=12, color="black")
plt.show()


# **Dataset: school_memberships**
# 
# There are 5638 school membership entries

# In[ ]:


school_memberships = pd.read_csv('../input/data-science-for-good-careervillage/school_memberships.csv')

print('School Memberships data: \nRows: {}\nCols: {}'.format(school_memberships.shape[0],school_memberships.shape[1]))
print(school_memberships.columns)


# **Dataset: tags**
# 
# There are 16,269 tags. 

# In[ ]:


tags = pd.read_csv('../input/data-science-for-good-careervillage/tags.csv')

print('Tags data: \nRows: {}\nCols: {}'.format(tags.shape[0],tags.shape[1]))
print(tags.columns)


# **Dataset: group_memberships**
# 
# 1038 entries are present.

# In[ ]:


group_memberships = pd.read_csv('../input/data-science-for-good-careervillage/group_memberships.csv')

print('Group memberships data: \nRows: {}\nCols: {}'.format(group_memberships.shape[0],group_memberships.shape[1]))
print(group_memberships.columns)


# **Dataset: answers**
# 
# There are 51,123 answers. 

# In[ ]:


answers = pd.read_csv('../input/data-science-for-good-careervillage/answers.csv')

print('Answers data: \nRows: {}\nCols: {}'.format(answers.shape[0],answers.shape[1]))
print(answers.columns)


# In[ ]:


def grey_color_func(word, font_size, position, orientation, random_state=None,**kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(0, 100)

word_string = answers['answers_body'].str.cat(sep=' ')

wordcloud = WordCloud(
    stopwords=STOPWORDS,
    background_color='white',
    width=3000,
    height=1000).generate(word_string)

plt.figure(figsize=(20,40))
plt.imshow(wordcloud.recolor(color_func=grey_color_func, random_state=3),interpolation="bilinear")
plt.axis('off')
font = {'family': 'sans-serif',
        'color':  'brown',
        'weight': 'normal',
        'size': 32
        }
plt.title('Word Cloud - Answers Body', fontdict=font)
plt.show()


# **Dataset: Students** 
# 
# 30,971 students are on the platform

# In[ ]:


print("Missing values in Students data")
for x in students.columns:
    if students[x].isnull().values.ravel().sum() > 0:
        print('{} - {}'.format(x,students[x].isnull().values.ravel().sum()))


# New York, Bengaluru and Los Angeles are the top locations in which the platform has a high student base. 

# In[ ]:


temp = students['students_location'].value_counts()
temp = temp.reset_index().head(10)

f, ax = plt.subplots(figsize=(12, 4))
sns.barplot(x="students_location", y="index", data=temp, label="index", color="silver", orient='h')

for p in ax.patches:
    ax.text(p.get_width() + 20,
            p.get_y() + (p.get_height()/1.4),
            '{:1.0f}'.format(p.get_width()),
            ha="center")

ax.set_xlabel('Number of students', size=10, color="black")
ax.set_ylabel('Location', size=10, color="black")
ax.set_title('[Horizontal Bar Plot] Number of students across cities', size=12, color="black")
plt.show()


# **Dataset: questions**
# 
# There are 23,931 questions.

# In[ ]:


def grey_color_func(word, font_size, position, orientation, random_state=None,**kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(0, 100)

word_string = questions['questions_title'].str.cat(sep=' ')

wordcloud = WordCloud(
    stopwords=STOPWORDS,
    background_color='white',
    width=3000,
    height=1000).generate(word_string)

plt.figure(figsize=(20,40))
plt.imshow(wordcloud.recolor(color_func=grey_color_func, random_state=3),interpolation="bilinear")
plt.axis('off')
font = {'family': 'sans-serif',
        'color':  'brown',
        'weight': 'normal',
        'size': 32
        }
plt.title('Word Cloud - Questions Title', fontdict=font)
plt.show()


# In[ ]:


def grey_color_func(word, font_size, position, orientation, random_state=None,**kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(0, 100)

word_string = questions['questions_body'].str.cat(sep=' ')

wordcloud = WordCloud(
    stopwords=STOPWORDS,
    background_color='white',
    width=3000,
    height=1000).generate(word_string)

plt.figure(figsize=(20,40))
plt.imshow(wordcloud.recolor(color_func=grey_color_func, random_state=3),interpolation="bilinear")
plt.axis('off')
font = {'family': 'sans-serif',
        'color':  'brown',
        'weight': 'normal',
        'size': 32
        }
plt.title('Word Cloud - Questions Body', fontdict=font)
plt.show()


# **Dataset: tag_users**

# In[ ]:


tag_users = pd.read_csv('../input/data-science-for-good-careervillage/tag_users.csv')

print('Tag Users data: \nRows: {}\nCols: {}'.format(tag_users.shape[0],tag_users.shape[1]))
print(tag_users.columns)


# **Dataset: tag_questions**

# In[ ]:


tag_questions = pd.read_csv('../input/data-science-for-good-careervillage/tag_questions.csv')

print('Tag Questions data: \nRows: {}\nCols: {}'.format(tag_questions.shape[0],tag_questions.shape[1]))
print(tag_questions.columns)


# **Join datasets**

# In[ ]:


matches_new = pd.merge(matches, 
                       emails, 
                       how='left', 
                       left_on='matches_email_id', 
                       right_on='emails_id')


# In[ ]:


matches_new = pd.merge(matches_new, 
                       questions, 
                       how='left', 
                       left_on='matches_question_id', 
                       right_on='questions_id')


# In[ ]:


matches_new = pd.merge(matches_new, 
                       professionals, 
                       how='left', 
                       left_on='emails_recipient_id', 
                       right_on='professionals_id')


# In[ ]:


matches_new = pd.merge(matches_new, 
                       students, 
                       how='left', 
                       left_on='questions_author_id', 
                       right_on='students_id')


# In[ ]:


matches_new.head()


# ***More to come!!!***
