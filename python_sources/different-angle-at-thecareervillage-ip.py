#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing workers
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import datetime as DT #for today's date
import wordcloud
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


get_ipython().system('ls ../input')


# In[ ]:


data_folder = "../input/data-science-for-good-careervillage"


# In[ ]:


get_ipython().system('ls $data_folder')


# In[ ]:


#Reading data
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


# > <h1 style="color: #5499C7"> Understanding the data dependency {**ER - Diagram**} </h1>
# <div style="color: #1A4A60">Credit <a href="https://www.kaggleusercontent.com/kf/11221422/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..Nmw2WJoXNQ3Q_zjx8ER0LQ.1tFqNXuIgULuQ1pPIkojG_HLCzjmiLsCn2vDcDuv8z1hMUSgnRMJchz0Ej_GSdlJTtf7aLSjIGjtOH82Acz0Hi9oiWx2HbqArt64Hm5JZZpPKf1NZ8C5E6WbDRCb5WAH1rw4thlc0xrAxfyrJyj4ocujTtIa0zlGWN1DDqsl5NY.gexZBxV77cQEAYliDOG4mw/__results__.html#84">@ioohooi</a></div>

# In[ ]:


plt.figure(figsize=(15, 15))
plt.imshow(plt.imread('../input/er-diagram/erd.png'), interpolation='bilinear', aspect='auto')
plt.axis("off")
plt.show()


# ><h2>To begin with, My focus is on..</h2>
# ><h3>
# 1. Professionals<br>
# 2. Students<br>
# 3. Questions<br>
# 4. Answers
# </h3>
# 1. Questions can be Asked by both <b>Professionals</b> AND <b>Students</b>
# 1. Questions can be answered by both <b>Professionals</b> AND <b>Students</b>

# In[ ]:


professionals.head()


# 1. Replace Nan with 0 or constant word for all or blank ''
# 2. calculate years of exp -> today - joined date

# In[ ]:


professionals = professionals.replace(np.nan, '', regex=True)


# ><div style="color:#808502; font-size:18px"> 
# Converting the timestamp to Exeperience. </div>
# <div style="color:#101442; font-size:15px"> 
# 1. Read the system current timestamp<br>
# 2. Take the difference between current timestamp and date of joined timestamp
# <br>
# 3. Convert the difference to year
# </div>
#  

# In[ ]:


#timestamp to year converter
def tm_stamp_year(data_joined):
    now = pd.Timestamp(DT.datetime.now())
    time_diff = now.tz_localize('UTC').tz_convert('Asia/Kolkata') - pd.Timestamp(data_joined).tz_convert('Asia/Kolkata')
    return round(time_diff.components.days/365,1)


# In[ ]:


#professionals exeperience
professionals['Exeperience(year)'] = professionals['professionals_date_joined'].apply(tm_stamp_year)


# In[ ]:


professionals.head()


# <div style="color:#808502; font-size:18px"> 
# Histogram plot of professional's experience</div>

# In[ ]:


plt.hist(professionals['Exeperience(year)'], normed=True, bins=8)
plt.xlabel('Exeperience')


# <h3 style="color: #11aa11">most of the professionals fall under 0 - 1 years of experience</h3>
# >Very Less Professionals are highly experienced<br>
# i.e #num Professional's = 1 / year of experience

# In[ ]:


#focus on highly exeperienced professionals
plt.figure(figsize=(20, 10))
plt.imshow(
    wordcloud.WordCloud(
        min_font_size=6,
        background_color='white',
        width=4000,
        height=2000
    ).generate(' '.join(professionals[professionals['Exeperience(year)'] > 6]['professionals_location'].values)),
    interpolation='bilinear'
)
plt.axis("off")
plt.show()


# <h1 style="color:#aa11aa"> Location's for TOP 1 % professionals [experience wise]</h1>

# In[ ]:


students.head()


# 1. Replace Nan with 0 or constant word or blank ''
# 2. calculate years of exp -> today - joined date

# In[ ]:


students = students.replace(np.nan, '', regex=True)


# In[ ]:


#Students exeperience
students['Exeperience(year)'] = students['students_date_joined'].apply(tm_stamp_year)


# In[ ]:


students.head()


# <div style="color:#808502; font-size:18px"> 
# Histogram plot of student's experience</div>

# In[ ]:


plt.hist(students['Exeperience(year)'], normed=True, bins=8)
plt.xlabel('Exeperience')


# <h3 style="color: #11aa11">Most of students fall under 2 - 3 years of experience <br> which is higher than the professionals</h3>

# In[ ]:


#focus on highly exeperienced professionals
plt.figure(figsize=(20, 10))
plt.imshow(
    wordcloud.WordCloud(
        min_font_size=6,
        background_color='white',
        width=4000,
        height=2000
    ).generate(' '.join(students[students['Exeperience(year)'] > 7]['students_location'].values)),
    interpolation='bilinear'
)
plt.axis("off")
plt.show()


# <h1 style="color:#aa11aa"> Location's for TOP 1 % students [experience wise]</h1>

# <h2 style="color: #55a555"> Thank you for your support </h2>
# <h4 style="color: #aaaa00">Work is still inproces, stay tune :-)</h4>
