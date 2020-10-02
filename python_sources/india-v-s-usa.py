#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# This notebook is a work to see what is the difference in the taste of different countires for technology. This notebook explains the differences between the countries who has most of the respondents - India and USA. <br>
# 
# **Let's see how they are different**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import os
import numpy as np 
import pandas as pd
pd.set_option('display.max_columns', 5000)
pd.set_option('max_colwidth', -1)
import seaborn as sns
import cufflinks as cf
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
from IPython.display import Markdown, display

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings 
warnings.filterwarnings("ignore")


# In[ ]:


def read_csv(file_name):
    df = pd.read_csv(file_name)
    return df

mcq_responses_df = read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv')
other_txt_df = read_csv('/kaggle/input/kaggle-survey-2019/other_text_responses.csv')
questions_df = read_csv('/kaggle/input/kaggle-survey-2019/questions_only.csv')
survey_schema_df = read_csv('/kaggle/input/kaggle-survey-2019/survey_schema.csv')


# In[ ]:


mcq_responses_df.drop(mcq_responses_df.index[0], inplace=True)


# In[ ]:


mcq_responses_df.head()


# In[ ]:


questions_df


# ### Which Country has more aspiring Data Scientists ?

# In[ ]:


display(Markdown('**{}**'.format(questions_df['Q3'][0])))
mcq_responses_df['Q3'].value_counts()[0:30].plot.barh(figsize=(5, 8))
plt.title('Top Countries with more Respondents')
plt.show()


# So, India and United States of America are the top countries which has most of the respondents. <br><br>
# 1) So, Does that mean that we have more opportunities in these countries? <br>
# 2) Do they have more Young aspiring Data Scientists? <br>
# 3) Are they demanding more AI experts? <br>
# 4) What is the difference between these groups for the taste of technology?<br>**Let's look more into it.**

# ### Do these countries have more Young Respondents of experienced ones?

# In[ ]:


india_df = mcq_responses_df[mcq_responses_df['Q3']=='India']
usa_df = mcq_responses_df[mcq_responses_df['Q3']=='United States of America']


# In[ ]:


def plot_grouped_graph(col_name, india_df=india_df, usa_df=usa_df):
    display(Markdown('**{}**'.format(questions_df[col_name][0])))
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 8))
    sns.countplot(y=col_name, data=india_df, ax=ax[0], order = india_df[col_name].value_counts().index)
    sns.countplot(y=col_name, data=usa_df, ax=ax[1], order = usa_df[col_name].value_counts().index)
    
    ax[0].tick_params(labelsize=10)
    ax[0].set_ylabel('')
    ax[0].set_xlabel('')
    ax[1].set_ylabel('')
    ax[1].set_xlabel('')
    ax[1].tick_params(labelsize=10)
    
    ax[0].set_title('India', fontsize=20)
    ax[1].set_title('USA', fontsize=20)
    
    plt.show()
    return None


# In[ ]:


plot_grouped_graph('Q1')


# India seems to be getting more respondents from younger ages. While USA got more respondents from the middles age. So USA has more people above the age of 25 while India has more below the age of 25. <br> <br>
# 1) Does that mean that USA has more experienced developers? <br>
# 2) If they are more into Machine Learning Engineers?<br>
# 3) Which country is aspiring more younger Data Scientists and hence wants more AI environment?<br>
# 4) Which country has more female geeks?

# ### What is the Age and Gender Distribution of these countries ?

# In[ ]:


fig, ax = plt.subplots(2, 1, sharex=True, figsize=(20, 10))


sns.countplot(x='Q1', hue='Q2', data=india_df, 
              order = india_df['Q1'].value_counts().sort_index().index, 
              ax=ax[0])

ax[0].set_title('Age & Gender Distribution of India', size=15)


sns.countplot(x='Q1', hue='Q2', data=usa_df, 
              order = usa_df['Q1'].value_counts().sort_index().index, 
              ax=ax[1])

ax[1].set_title('Age & Gender Distribution of USA', size=15)
ax[0].set_ylabel('')
ax[0].set_xlabel('')
ax[1].set_ylabel('')
ax[1].set_xlabel('')
plt.show()


# Both countries has more Male Developers than female Developers.<br>
# 1) What's the pattern of people in AI? <br>
# 2) For how many years people are there in Machine Learning? <br>
# 3) What is their highest formal education ?

# ### What is their highest formal education ?

# In[ ]:


fig, ax = plt.subplots(2, 1, sharex=True, figsize=(20, 12))


sns.countplot(x='Q4', hue='Q2', data=india_df, 
              order = india_df['Q4'].value_counts().sort_index().index, 
              ax=ax[0])

ax[0].set_title('India', size=15)


sns.countplot(x='Q4', hue='Q2', data=usa_df, 
              order = usa_df['Q4'].value_counts().sort_index().index, 
              ax=ax[1])

ax[1].set_title('USA', size=15)
ax[0].set_ylabel('')
ax[0].legend(loc=1)
ax[0].set_xlabel('')
ax[1].set_ylabel('')
ax[1].set_xlabel('')
ax[1].legend(loc=1)
plt.xticks(rotation=45, size=12)
plt.show()


# We have seen that many young aspirants are from India and that makes sense as most of the individuals from india are persuing or completed Bachlor's degree but the male to female ratio is quite high.<br>
# USA has more doctrate Degree holders in comparison to India and we have seen this that USA has more people between the age of 25 to 30.<br>
# Let's see how long they are into Machine Learning?

# ### How long people are into Machine Learning?

# In[ ]:


plot_grouped_graph('Q23')


# 1) India has more people with less than 2 years of experience and that perfectly makes sense because it has more respndents from younger age and are in their bachlor's or may be completed their bachlor's. **But** USA has also more people with experience less than 2 years. **Does that mean people are now shifting their career in machine learning ?** <br>
# 2) But in comparison to India, USA has almost 5 times more people with more than 5 years of experience. **This means that USA went into AI much early than India.** But as India has almost 2.6 times more people with less than 1 year of experience, does that mean India is moving much faster towards AI and India is going to open more opportunities in future?

# ### What is the salary differences ?

# In[ ]:


plot_grouped_graph('Q10')


# This graph shows that many Indians have salary less than 1,000 (USD) but in USA, people have salary greater than $100,000. This is clear from the fact that USA has more experienced people and hence more salary while india has many young engineers.

# ### How their taste for IDE different ?

# In[ ]:


def multiple_responses_question_to_df(col_name, df):
    option_df = df.loc[:, india_df.columns.str.startswith(col_name)]

    temp_df = {}
    for col in option_df.columns:
        frame = option_df[col].value_counts().to_frame()
        name = frame.index.tolist()[0]

        if isinstance(name, int):
            continue
        else:
            temp_df[name.split('(')[0]] = frame[col][name]
    return pd.DataFrame(temp_df, index=[0]).transpose()
 



def plot_mcq_df(india_mcq_df, usa_mcq_df):
    
    fig, ax = plt.subplots(1, 2, figsize=(30, 12))
    india_mcq_df.sort_values(by=0).plot.barh(legend=False, ax=ax[0])
    usa_mcq_df.sort_values(by=0).plot.barh(legend=False,ax=ax[1])
    
    ax[0].tick_params(labelsize=14)
    ax[0].set_ylabel('')
    ax[0].set_xlabel('')
    ax[1].set_ylabel('')
    ax[1].set_xlabel('')
    ax[1].tick_params(labelsize=14)
    
    ax[0].set_title('India', fontsize=20)
    ax[1].set_title('USA', fontsize=20)
    
    plt.show()
    return None


# In[ ]:


india_mcq_df = multiple_responses_question_to_df('Q16', india_df)
usa_mcq_df = multiple_responses_question_to_df('Q16', usa_df)

plot_mcq_df(india_mcq_df, usa_mcq_df)


# So, Jupyter is most lovable among people of India and USA but India has more jupyter lovers than USA and India has more young engineers too. But both the countries has slight difference in people who like RStudio.<br>
# 1) Does that mean that new commers in machine learning prefer Jupyter more?<br>
# 2) Does that mean people in both countries prefer Python more than any other language?<br>
# 3) What about R language in boththe countries?
# <br>
# 

# In[ ]:


india_mcq_df = multiple_responses_question_to_df('Q18', india_df)
usa_mcq_df = multiple_responses_question_to_df('Q18', usa_df)

plot_mcq_df(india_mcq_df, usa_mcq_df)


# So, to not much surprise, people in both countries mostly prefer Python over other languages. It can be due to simple, flexible style of programming. And people also like R but m=most people recommend Python and Jupyter as their IDE.

# ### How they learn Data Science ?

# In[ ]:


india_mcq_df = multiple_responses_question_to_df('Q13', india_df)
usa_mcq_df = multiple_responses_question_to_df('Q13', usa_df)

plot_mcq_df(india_mcq_df, usa_mcq_df)


# More people are moving towards online Courses instead of university degree. But USA has almost double the number of respondents who has University degree in comparison to India. It has been observed earlier that people in USA mostly have Master's Degree which adds on this observation that people like going to college and have degree. <br>
# 
# But does that mean people in USA are less active to knowledge from online sources than India ? Let's see about that.

# In[ ]:


india_mcq_df = multiple_responses_question_to_df('Q12', india_df)
usa_mcq_df = multiple_responses_question_to_df('Q12', usa_df)

plot_mcq_df(india_mcq_df, usa_mcq_df)


# From this, it can not be denied that people from both regions are highly active on different social sites for Data science topics. So many poeple are relying on **Kaggle, blogs and Youtube channels**. This can be seen as a very good opportunity for experienced ones to spend time on making online profile because new commers will be looking at them through their Kaggle notebooks, blogs, youtube channels. It can help many to learn from their experiences.

# ### What hardware type is being used ?

# In[ ]:


india_mcq_df = multiple_responses_question_to_df('Q21', india_df)
usa_mcq_df = multiple_responses_question_to_df('Q21', usa_df)

plot_mcq_df(india_mcq_df, usa_mcq_df)


# As India has more young or aspiring data engineers, India has more CPU users than USA. But as the data is growing tremendously daya by day, more people find the GPUs as necessity for various kinds of work. And the results show that India has more GPU users than USA. Let's see the differences for TPU's

# In[ ]:


plot_grouped_graph('Q22')


# ### Summary
# 
# So, all in short, both the countries are moving fastly towards using Machine Learning but USA has advanced early than India and has got more experienced Engineers but India is also heading towards AI. This may give an indication of rising demand for AI in both countires and also we can see that as India has more young data scientists, may be, we will see a tremendous increase in AI in India in the comming years. So that's a big path for young ones to invest their time in it.

# Still India is getting more hits. Although in both countries majority of the resppndents have never used GPU but those minority users are more in India than USA. This may indicate more data work in India.

# In[ ]:




