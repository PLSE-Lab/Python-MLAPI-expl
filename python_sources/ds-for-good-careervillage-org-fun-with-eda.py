#!/usr/bin/env python
# coding: utf-8

# **Overview**  
# 
# ![](https://www.ffwd.org/wp-content/uploads/CareerVillage-logo.png)
# 
# CareerVillage.org is a nonprofit that crowdsources career advice for underserved youth. Founded in 2011 in four classrooms in New York City, the platform has now served career advice from 25,000 volunteer professionals to over 3.5M online learners. The platform uses a Q&A style similar to StackOverflow or Quora to provide students with answers to any question about any career.
# 
# In this Data Science for Good challenge, CareerVillage.org, in partnership with Google.org, is inviting fellow Data Scientist around the world to help recommend questions to appropriate volunteers. To support this challenge, CareerVillage.org has supplied five years of data.
# 
# **Problem Statement**  
# The U.S. has almost 500 students for every guidance counselor. Underserved youth lack the network to find their career role models, making CareerVillage.org the only option for millions of young people in America and around the globe with nowhere else to turn.
# 
# To date, 25,000 volunteers have created profiles and opted in to receive emails when a career question is a good fit for them. This is where your skills come in. To help students get the advice they need, the team at CareerVillage.org needs to be able to send the right questions to the right volunteers. The notifications sent to volunteers seem to have the greatest impact on how many questions are answered.
# 
# **Our objective is to develop a method to recommend relevant questions to the professionals who are most likely to answer them.**
# 
# **Approach**  
# The one that never changes in Data Science related analysis - **Exploratory Data Analysis**. I will start with EDA and based on the insights will decided the recommendations.

# **Listing the files**   
# ![](https://media.giphy.com/media/ig3CJoXHgcJ5m/giphy.gif)  
# Let's first list all the files that are given for analysis and know the data provided inside those files. 

# In[ ]:


get_ipython().system('ls ../input/')


# **Details on the Files**  
# ![](https://media.giphy.com/media/tQliIp3sn1T44/giphy.gif)    
# Let's look at the details of files that are provided for the analysis.  
# 
# * answers.csv - Answers are what this is all about! Answers get posted in response to questions. Answers can only be posted by users who are registered as Professionals. However, if someone has changed their registration type after joining, they may show up as the author of an Answer even if they are no longer a Professional.  
# * comments.csv - Comments can be made on Answers or Questions. We refer to whichever the comment is posted to as the "parent" of that comment. Comments can be posted by any type of user. Our favorite comments tend to have "Thank you" in them :)  
# * emails.csv - Each email corresponds to one specific email to one specific recipient. The frequency_level refers to the type of email template which includes immediate emails sent right after a question is asked, daily digests, and weekly digests.  
# * group_memberships.csv - Any type of user can join any group. There are only a handful of groups so far.  
# * groups.csv - Each group has a "type". For privacy reasons we have to leave the group names off.  
# * matches.csv - Each row tells you which questions were included in emails. If an email contains only one question, that email's ID will show up here only once. If an email contains 10 questions, that email's ID would show up here 10 times.  
# * professionals.csv - We call our volunteers "Professionals", but we might as well call them Superheroes. They're the grown ups who volunteer their time to answer questions on the site.  
# * questions.csv - Questions get posted by students. Sometimes they're very advanced. Sometimes they're just getting started. It's all fair game, as long as it's relevant to the student's future professional success.  
# * school_memberships.csv - Just like group_memberships, but for schools instead.  
# * students.csv - Students are the most important people on CareerVillage.org. They tend to range in age from about 14 to 24. They're all over the world, and they're the reason we exist!  
# * tag_questions.csv - Every question can be hashtagged. We track the hashtag-to-question pairings, and put them into this file.  
# * tag_users.csv - Users of any type can follow a hashtag. This shows you which hashtags each user follows.  
# * tags.csv - Each tag gets a name.

# **Getting hands dirty with EDA**  
# 
# ![](https://media.giphy.com/media/3og0ITQOC5wlyk8ffy/giphy.gif)
# 
# Now, Let's analyse each of the datasets that is provided one after another and then try to join all the dataset based on the analysis results to provide a valid recommendation.

# **Fun with answers.csv dataset**  
# The answers.csv dataset consists of answers for the questions that are posted. The dataset has the below list of columns,
# * answers_id
# * answers_author_id
# * answers_question_id
# * answers_date_added
# * answers_body

# In[ ]:


import pandas as pd
import datetime as dt
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199
# pd.describe_option('display') #Full list of useful options
answers = pd.read_csv("../input/answers.csv")
answers['date_format'] = pd.to_datetime(answers['answers_date_added'], format='%d%b%Y:%H:%M:%S.%f', infer_datetime_format=True)
answers['year_month'] = answers.date_format.dt.to_period('M')
answers.head()


# **Missing data check**  
# Let us check whether there is any missing data in the answers dataset.

# In[ ]:


def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
    
missing_values_table(answers)


# **Answers dataset with no missing values**  
# The answers dataset has one row as missing value. So, let us drop the missing value row from the dataset.

# In[ ]:


#Selecting all the rows which has the data
answers = answers[pd.notnull(answers['answers_body'])]
qts = answers['answers_question_id'].nunique()
aut = answers['answers_author_id'].nunique()
print('There are {0} unique questions that are answered by {1} unique responders.'.format(qts, aut))


# **Top 10 authors who answered the questions by count**  
# ![](https://media.giphy.com/media/d3g1lkA1srw1c3Di/giphy.gif)
# Let's, visualize the top 10 authors who answered by the number of questions that are posted.

# In[ ]:


ans_aut_df = answers['answers_author_id'].value_counts().reset_index()
ans_aut_df.columns = ['Author ID', 'Count']
ans_aut_10 = ans_aut_df.nlargest(10, 'Count')

#Plotting top 10 authors
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.plotly as py
import plotly.graph_objs as go
init_notebook_mode(connected=True)


trace = go.Bar(
    x=ans_aut_10['Count'],
    y=ans_aut_10['Author ID'],
    orientation = 'h',
    marker=dict(
        color=ans_aut_10['Count'],
        colorscale = 'Viridis',
        reversescale = True
    ),
)

layout = go.Layout(
    title='Top 10 author ID''s who responded for the post', 
    margin=dict(
        l=320,
        r=10,
        t=140,
        b=80
    )
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename="Unique_val")


# **Average Length of Top 10 responded Vs rest**  

# In[ ]:


import numpy as np
top_10_res = ans_aut_10['Author ID']
top_10 = answers.loc[answers['answers_author_id'].isin(top_10_res)]
rest_answers = answers.loc[~answers['answers_author_id'].isin(top_10_res)]

print('Average word length of response by Top 10 responders is {0:.0f}.'.format(np.mean(top_10['answers_body'].apply(lambda x: len(x.split())))))
print('Average word length of response by rest is {0:.0f}.'.format(np.mean(rest_answers['answers_body'].apply(lambda x: len(x.split())))))


# There is approximately 20 word difference between the average word length of response by top 10 responders and the average length of response by rest of the responders. 

# **Answer trend**  
# ![](https://media.giphy.com/media/dfEYhn5LpEezu/giphy.gif)  
# Let us see if there is any pattern in answering the questions posted over a period of time.

# In[ ]:


ans_tnd_df = pd.DataFrame({'Count' : answers.groupby([ "year_month"]).size()}).reset_index()
ans_tnd_df = ans_tnd_df.sort_values(by='year_month', ascending=True)

import matplotlib.pyplot as plt
f,ax1=plt.subplots(figsize=(25,10))
import seaborn as sns
sns.pointplot(x=ans_tnd_df['year_month'],
              y=ans_tnd_df['Count'],color='lime',alpha=0.8)
plt.xlabel('Period')
plt.ylabel('Count of questions answered by responders')
plt.title('Trend of questions answered by the responders')
plt.xticks(rotation=90)
plt.grid()
plt.show()


# The questions got answered in an increasing trend from August 2014. Till February 2016 the number of questions answered were below the count of 500. After February 2016 the number of questions answered were all above 500. Further, from April 2016 to December 2017 there seems a repeating trend in answering the questions.

# **Meta Features on answers dataset**

# In[ ]:


import string
def add_metafeatures(dataframe):
    mf_df = dataframe.copy()
    ans_pts = answers['answers_body']
    n_charac = pd.Series([len(t) for t in ans_pts])
    n_punctuation = pd.Series([sum([1 for x in text if x in set(string.punctuation)]) for text in ans_pts])
    n_upper = pd.Series([sum([1 for c in text if c.isupper()]) for text in ans_pts])
    mf_df['n_charac'] = n_charac
    mf_df['n_punctuation'] = n_punctuation
    mf_df['n_upper'] = n_upper
    return mf_df

ans_meta = add_metafeatures(answers)


# **Plotting Meta features**  
# We will plot the meta features 'n_charac', 'n_punctuation' and 'n_upper' with its average and see how its varied over the period of time.

# In[ ]:


ans_meta_df = pd.DataFrame(ans_meta.groupby('date_format').                           mean()[['n_charac','n_punctuation','n_upper']])

f,ax1=plt.subplots(figsize=(25,10))
sns.lineplot(data=ans_meta_df, palette="tab10", linewidth=2.5)


# **Fun with response**  
# Now, its time to explore the responses provided by the responders for the questions that are posted.

# In[ ]:


# Tokenizing the response
import nltk
from nltk.tokenize import RegexpTokenizer
import re
resp_det = pd.DataFrame(answers['answers_body'])
resp_det.reset_index(drop=True, inplace=True)
res_lt = []
tokenizer = RegexpTokenizer(r'\w+')
for rows in range(0, resp_det.shape[0]):
    res_txt = " ".join(re.findall("[a-zA-Z]+", resp_det.answers_body[rows]))
    res_txt = tokenizer.tokenize(res_txt)
    res_lt.append(res_txt)

#Converting into single list of response
import itertools
res_list = list(itertools.chain(*res_lt))

#Removing Non sense words - I
words = set(nltk.corpus.words.words())
def clean_sent(sent):
    return " ".join(w for w in nltk.wordpunct_tokenize(sent)      if w.lower() in words or not w.isalpha())

res_list_cl = [clean_sent(item) for item in res_list]

#Removing Space, Tab, CR and Newline
res_list_cl = [re.sub(r'\s+', '', item) for item in res_list_cl]

#import nltk
stopwords = nltk.corpus.stopwords.words('english')
res_list_cl = [word for word in res_list_cl if word.lower() not in stopwords]


#Removing non sense words - II
tp = pd.DataFrame(list(set(res_list_cl))).reset_index(drop=True)
tp.columns = ['uniq_lt']
tp['length'] = tp['uniq_lt'].apply(lambda x: len(x))
non_sense = list(tp['uniq_lt'][tp.length <= 2])
res_list_cl = [item for item in res_list_cl if item not in non_sense]

print("Length of original response list: {0} words\n"
      "Length of response list after stopwords removal: {1} words"
      .format(len(res_list), len(res_list_cl)))


# **Top 30 Occuring words after removing Stopwords from response**

# In[ ]:


#Data cleaning for getting top 30
from collections import Counter
resp_cnt = Counter(res_list_cl)

#Dictonary to Dataframe
resp_cnt_df = pd.DataFrame(list(resp_cnt.items()), columns = ['Words', 'Freq'])
resp_cnt_df = resp_cnt_df.sort_values(by=['Freq'], ascending=False)

#Top 30
resp_cnt_df_l30 = resp_cnt_df.nlargest(30, 'Freq')
resp_cnt_df_s30 = resp_cnt_df.nsmallest(30, 'Freq')


# In[ ]:


#Plotting the top 30 largest Vs smallest
from plotly import tools
lr_tr  = go.Bar(
    x=resp_cnt_df_l30['Freq'],
    y=resp_cnt_df_l30['Words'],
    name='Most used',
    marker=dict(
        color='rgba(88, 214, 141, 0.6)',
        line=dict(
            color='rgba(88, 214, 141, 1.0)',
            width=.3,
        )
    ),            
           orientation='h',
    opacity=0.6
)

sm_tr = go.Bar(
    x=resp_cnt_df_s30['Freq'],
    y=resp_cnt_df_s30['Words'],
    name='Least used',
    marker=dict(
        color='rgba(155, 89, 182, 0.6)',
        line=dict(
            color='rgba(155, 89, 182, 1.0)',
            width=.3,
        )
    ),
    orientation='h',
    opacity=0.6
)

fig = tools.make_subplots(rows=2, cols=1, subplot_titles=('Top 30 Most occuring words in response',
                                                          'Top 30 Least occuring words in response'))

fig.append_trace(lr_tr, 1, 1)
fig.append_trace(sm_tr, 2, 1)


fig['layout'].update(height=1200, width=800)

iplot(fig, filename='lr_vs_sm')


# Both the Top 30 most occuring and least occuring have entirely different pattern of word list. In the top 30 most of the words represent the positive sentiment and the least 30 words represent the negative sentiment.

# **Fun with Comments.csv dataset**  
# The comments dataset consists of comments posted in response to the answers answered by the responders to the questions that are posted. The comments consists of list of below list of columns,
# * comments_id
# * comments_author_id
# * comments_parent_content_id
# * comments_date_added
# * comments_body  

# In[ ]:


comments = pd.read_csv("../input/comments.csv")
comments['date_format'] = pd.to_datetime(comments['comments_date_added'], format='%d%b%Y:%H:%M:%S.%f', infer_datetime_format=True)
comments['year_month'] = comments.date_format.dt.to_period('M')
comments.head()


# **Missing value check**  
# Let, us check the quality of data by checking whether the data has any missing value and its proportion.

# In[ ]:


#Missing value check
missing_values_table(comments)


# As we can see only 4 rows of data is missing, we will drop all the 4 rows from the comments dataset.

# **Comments dataset with no missing values**  
# The comments dataset has 4 rows as missing value. So, let us drop the missing value row from the dataset.

# In[ ]:


#Selecting all the rows which has the data
comments = comments[pd.notnull(comments['comments_body'])]
cntid = comments['comments_parent_content_id'].nunique()
autid = comments['comments_author_id'].nunique()
print('There are {0} unique contents that are commented by {1} unique authors.'.format(cntid, autid))


# **Top 10 authors who commented on the post by count**  
# ![](https://media.giphy.com/media/2tSodgDfwCjIMCBY8h/200w_d.gif)
# Let's, visualize the top 10 authors who commented by the number of questions that are posted.

# In[ ]:


comm_aut_df = comments['comments_author_id'].value_counts().reset_index()
comm_aut_df.columns = ['Author ID', 'Count']
comm_aut_10 = comm_aut_df.nlargest(10, 'Count')


trace = go.Bar(
    x=comm_aut_10['Count'],
    y=comm_aut_10['Author ID'],
    orientation = 'h',
    marker=dict(
        color=comm_aut_10['Count'],
        colorscale = 'Viridis',
        reversescale = True
    ),
)

layout = go.Layout(
    title='Top 10 author ID''s who commented for the post', 
    margin=dict(
        l=320,
        r=10,
        t=140,
        b=80
    )
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename="Unique_val_comm")


# **Blogger Addict**  
# 
# ![](https://media.giphy.com/media/ql2lUYvISjpaE/giphy.gif)
# 
# Let us see if any of the authors who responded to the questions which are posted have commented on the post as well.

# In[ ]:


ans_10 = ans_aut_10['Author ID'].unique()
comm_10 = comm_aut_10['Author ID'].unique()
ans_comm_10 = [comm for comm in comm_10 if comm in ans_10]
print("There are 3 authors who responded most of the questions who were also present in the top 10 commenters list.\n"        "The Author ID's are {0}, {1} and {2}.".format(ans_comm_10[0], ans_comm_10[1], ans_comm_10[2]))


# There are 3 authors namely '36ff3b3666df400f956f8335cf53e09e',  '05444a2f42454327b2ac4b463c0adbe0' and '58fa5e95fe9e480a9349bbb1d7faaddb' who responded most of the questions were also present in the top 10 commented list.

# **Comment Trend**  
# ![](https://media.giphy.com/media/1vZfYD1L2lwhLC7I9t/200w_d.gif)  
# Let us see if there is any pattern in comments for the responses over a period of time.

# In[ ]:


comm_tnd_df = pd.DataFrame({'Count' : comments.groupby([ "year_month"]).size()}).reset_index()
comm_tnd_df = comm_tnd_df.sort_values(by='year_month', ascending=True)

import matplotlib.pyplot as plt
f,ax1=plt.subplots(figsize=(25,10))
import seaborn as sns
sns.pointplot(x=comm_tnd_df['year_month'],
              y=comm_tnd_df['Count'],color='blue',alpha=0.8)
plt.xlabel('Period')
plt.ylabel('Count of comments by responders')
plt.title('Comments trend based on comments by the responders')
plt.xticks(rotation=90)
plt.grid()
plt.show()


# Both the response and the comments have the same peak on the month of May-2016. But, with respect to count the comments count is approximately half of the response count. Further, there was major spike on the month of April 2014 with respect to comments with a count of 800 and  430 in count of response which has the reverse trend in count.
# In addition, after May - 2106 there are multiple spikes in both response and comments. 

# **Fun with comments**  
# Now, its time to explore the comments provided by the responders for the responses posted for the questions..

# In[ ]:


# Tokenizing the response
comm_det = pd.DataFrame(comments['comments_body'])
comm_det.reset_index(drop=True, inplace=True)
comm_lt = []
tokenizer = RegexpTokenizer(r'\w+')
for rows in range(0, comm_det.shape[0]):
    comm_txt = " ".join(re.findall("[a-zA-Z]+", comm_det.comments_body[rows]))
    comm_txt = tokenizer.tokenize(comm_txt)
    comm_lt.append(comm_txt)
    
#Converting into single list of comments
import itertools
comm_list = list(itertools.chain(*comm_lt))

#Removing Non sense words - I
words = set(nltk.corpus.words.words())
def clean_sent(sent):
    return " ".join(w for w in nltk.wordpunct_tokenize(sent)      if w.lower() in words or not w.isalpha())

comm_list_cl = [clean_sent(item) for item in comm_list]

#Removing Space, Tab, CR and Newline
comm_list_cl = [re.sub(r'\s+', '', item) for item in comm_list_cl]

#import nltk
stopwords = nltk.corpus.stopwords.words('english')
comm_list_cl = [word for word in comm_list_cl if word.lower() not in stopwords]


#Removing non sense words - II
tp = pd.DataFrame(list(set(comm_list_cl))).reset_index(drop=True)
tp.columns = ['uniq_lt']
tp['length'] = tp['uniq_lt'].apply(lambda x: len(x))
non_sense = list(tp['uniq_lt'][tp.length <= 2])
comm_list_cl = [item for item in comm_list_cl if item not in non_sense]

print("Length of original comments list: {0} words\n"
      "Length of comments list after stopwords removal: {1} words"
      .format(len(comm_list), len(comm_list_cl)))


# **Top 30 Occuring words after removing Stopwords from comments**

# In[ ]:


#Data cleaning for getting top 30
from collections import Counter
comm_cnt = Counter(comm_list_cl)

#Dictonary to Dataframe
comm_cnt_df = pd.DataFrame(list(comm_cnt.items()), columns = ['Words', 'Freq'])
comm_cnt_df = comm_cnt_df.sort_values(by=['Freq'], ascending=False)

#Top 30
comm_cnt_df_l30 = comm_cnt_df.nlargest(30, 'Freq')
comm_cnt_df_s30 = comm_cnt_df.nsmallest(30, 'Freq')


#Plotting the top 30 largest Vs smallest
from plotly import tools
lr_tr  = go.Bar(
    x=comm_cnt_df_l30['Freq'],
    y=comm_cnt_df_l30['Words'],
    name='Most used',
    marker=dict(
        color='rgba(88, 214, 141, 0.6)',
        line=dict(
            color='rgba(88, 214, 141, 1.0)',
            width=.3,
        )
    ),            
           orientation='h',
    opacity=0.6
)

sm_tr = go.Bar(
    x=comm_cnt_df_s30['Freq'],
    y=comm_cnt_df_s30['Words'],
    name='Least used',
    marker=dict(
        color='rgba(155, 89, 182, 0.6)',
        line=dict(
            color='rgba(155, 89, 182, 1.0)',
            width=.3,
        )
    ),
    orientation='h',
    opacity=0.6
)

fig = tools.make_subplots(rows=2, cols=1, subplot_titles=('Top 30 Most occuring words in Comments',
                                                          'Top 30 Least occuring words in Comments'))

fig.append_trace(lr_tr, 1, 1)
fig.append_trace(sm_tr, 2, 1)


fig['layout'].update(height=1200, width=800)

iplot(fig, filename='lr_vs_sm')


# **Fun with emails.csv dataset**  
# The emails dataset consists of email ID's of sender and receiver with email sent date and frequency of emails . The email dataset  consists of list of below list of columns,
# * emails_id
# * emails_recipient_id
# * emails_date_sent
# * emails_frequency_level

# In[ ]:


email = pd.read_csv("../input/emails.csv")
email['date_format'] = pd.to_datetime(email['emails_date_sent'], format='%d%b%Y:%H:%M:%S.%f', infer_datetime_format=True)
email['year_month'] = email.date_format.dt.to_period('M')
email.head()


# **Missing value check**  
# Let, us check the quality of data by checking whether the data has any missing value and its proportion.

# In[ ]:


#Missing value check
missing_values_table(email)


# There is no missing values in the emails dataset. So, we will move ahead with the analysis.

# **Proportion of Email Frequency**  
# Let us see the how often people send mails with respect to email frequency which helps us to determine how fast people respond to the queries that are posted.

# In[ ]:


#Summarising the email frequency
email_freq = email.groupby('emails_frequency_level', as_index=False).agg({"emails_id": "count"})
email_freq.columns = ['Email_Freq', 'Count_of_Freq']

e_trace = go.Pie(labels=email_freq.Email_Freq, values=email_freq.Count_of_Freq)

data = [e_trace]
layout = go.Layout(title = "Proportion of Email Frequency")

fig = go.Figure(data = data, layout = layout)
iplot(fig)


# From the pie chart its clearly seen that almost 18% replying immediately and 80% of the people respond on daily basis which seems reasonably good number. Its great as people help out other people in need of guidance. 

# **Trend of response frequency**  
# Now, let us see as how the trend of response frequency varies with each other category of frequencies.

# In[60]:


res_freq_trend = email.groupby(['year_month', 'emails_frequency_level']).size().reset_index(name="Count")
res_freq_trend.sort_values(['emails_frequency_level', 'year_month'], ascending=[True, True])
res_freq_trend.head()


# **Stay Tuned.....**
