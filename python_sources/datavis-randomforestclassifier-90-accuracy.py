#!/usr/bin/env python
# coding: utf-8

# # OSMI Mental Health in Tech (2016, ongoing)

# ## Context
# 
# * Mental Health (MH) has been an increasingly prominent topic in recent years.
# * According to the Singapore Association for Mental Health (SAMH), a mental illess "...is a disturbance of the mind that impairs the way we think, feel and behave. It affects our daily activities, as well as impact the lives of family members and friends."
# * SAMH states that 1 in 7 Singaporeans have experienced a mood or anixety disorder (which are common mental illnesses).
# * SAMH also states that "Despite their prevalence, persons with mental illness still face considerable stigma and discrimination. Many such individuals are thus reluctant to seek help and treatment."
# * It is supposed to be beneficial for people suffering from mental illness to firstly recognise it, and secondly, to seek help for it.
# * Tech companies are anecdotally notorious for endless and high-stress work environment.
# * As a tech student, I'm personally interested to find out the prevalent attitudes regarding MH in the tech workplace.
# * I am also interested to identify what company features are most critical in inclining employees towards recognising their Mental Health Disorders (MHD) and seeking help for it.
# 
# ## Objectives
# 1. Explore the data provided in survey for insights into Mental Health.
# 2. Attempt to build Classifiers to predict whether an employee:
#     1. will be willing to raise MH issues with future employers
#     2. will seek professional diagnosis for MHD
#     3. will seek treatment for MHD
# 
# SAMH: https://www.samhealth.org.sg/understanding-mental-health/what-is-mental-illness/
# 
# ### Datasets required:
# * `mental-heath-in-tech-2016_20161114.csv` from [Kaggle (OSMI Mental Health in Tech Survey 2016)](https://www.kaggle.com/osmi/mental-health-in-tech-2016)
# 
# ### Outline
# 1. Preview Data
# 2. Data Cleaning
#     1. Filter out self-employed
#     2. Drop empty or irrelevant columns
#     3. Preview Column Values & General Pre-Process
#     4. Employer Type & Role
#     5. Age
#     6. Gender
#     7. Country of Employment
#     8. Work Position
#     9. Mass Rename Columns
# 3. Data Visualisations
#     1. Mental Health Status across Ages and Genders
#     2. Mental Health Disorders (MHD)
#     3. Mental Health Status and Company Size
#     4. Correlation Matrix of Features  
# 4. Classifiers
#     1. RFC: Openness about MH with Future Employer (acc: 93%)
#     2. RFC: Seeking Professional Diagnosis (acc: 90+%)
#     3. RFC: Seeking Treatment (acc: 90+%)
# 5. Credits
#     1. Data Visualisation Python Seaborn kernel [https://www.kaggle.com/jchen2186/data-visualization-with-python-seaborn]
#     2. Preprocessing and Random Forest with 87% accuracy kernel [https://www.kaggle.com/h1rshit/preprocessing-and-random-forest-with-87-accuracy]

# In[3]:


import pandas as pd
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

pd.options.display.float_format = '{:.2f}'.format
pd.set_option('display.max_columns', 64) #63 columns in original data
sns.set()

dim = (16,8)
fs=16


# ## 1. Preview Data

# * There are 63 features (survey questions) and a total of 1433 respondents.
# * There are many nan values, which appear to be associated with a respondent being self-employed (see row idx 3).
# * Respondents who are self-employed will be dropped from this study.
# * 

# In[4]:


import os
print(os.listdir("../input"))
csv_file = '../input/mental-heath-in-tech-2016_20161114.csv'
df = pd.read_csv(csv_file)
df.shape


# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


df.describe(include=['O'])


# ## 2. Data Cleaning

# ### 2.A Filter out self-employed

# In[ ]:


df["Are you self-employed?"].value_counts()
# there are 287 self-employed and 1146 employed respondents.


# In[ ]:


# drop self-employed respondents
df2 = df.copy()
df2 = df2[df2["Are you self-employed?"]==0]
df2.shape 


# In[ ]:


# relook df2
df2.head()
# notice several columns with many nan values - may need to drop.


# In[ ]:


df2.describe()
# notice that col
# "Do you have medical coverage (private insurance or state-provided) which includes treatment of  mental health issues?"
# is completely empty and will be dropped


# In[ ]:


df2.describe(include=['O'])
# some columns are completely empty and will be dropped
# some columns are irrelevant and will be dropped


# ### 2.B Drop empty or irrelevant columns

# In[ ]:


def col_ls(df):
    '''
    Prints out column names of df and its column index
    for ease of reading.
    
    Also returns a list of column names.'''
    ls = []
    for i, c in enumerate(df.columns):
        print(i, c)
        ls.append(c)
    return ls


# In[ ]:


df2_col = col_ls(df2)


# In[ ]:


empty_col = [
            "Do you know local or online resources to seek help for a mental health disorder?",
            "If you have been diagnosed or treated for a mental health disorder, do you ever reveal this to clients or business contacts?",
            "If you have revealed a mental health issue to a client or business contact, do you believe this has impacted you negatively?",
            "If you have been diagnosed or treated for a mental health disorder, do you ever reveal this to coworkers or employees?",
            "If you have revealed a mental health issue to a coworker or employee, do you believe this has impacted you negatively?",
            "Do you believe your productivity is ever affected by a mental health issue?",
            "If yes, what percentage of your work time (time performing primary or secondary job functions) is affected by a mental health issue?"]
empty_col += [df2_col[16]]
irre_col = ["Are you self-employed?",
            "What US state or territory do you work in?",
           "What US state or territory do you live in?",
           "What country do you live in?",
           "Why or why not?",
           "Why or why not?.1"]
df3 = df2.copy()
df3 = df3.drop(empty_col+irre_col,axis=1)


# In[ ]:


df3_col = col_ls(df3)


# In[ ]:


df3.describe()


# In[ ]:


df3.describe(include=['O'])


# ### 2.C Preview Column Values & General Pre-Process
# * Replace category responses with numeric labels.
#     * The more affirmative the response is (wrt qn), closer to 1.
#     * Otherwise, in decreasing order of affirmation, larger integer.
#     * Non-Applicable & nan responses will be marked -1.
#     * -1 responses will be dropped from data visualisations, if relevant.
# * Identify weird columns to drop.
#     * Such as columns with many missing & unaccounted responses.
# * Identify potential target columns.
# * Identify columns that require special attention to process.

# In[ ]:


def val(df):
    '''
    Prints out columns, its unique values and value counts,
    as well as null value count.
    To aid in data cleaning.'''
    for i, c in enumerate(df.columns):
        print(i,c)
        print()
        unique = df[c].unique()
        if len(unique) > 10:
            print("### More than 10 unique values. ###")
            print('### Special attention required for col\n{} {}\n###'.format(i,c))
            print()
        else:
            print(unique)
            print()
            print(df[c].value_counts())
            vc = df[c].value_counts().sum()
            print()
            print("Value count: ",vc)
        nc = df[c].isna().sum()
        print("Null value count: ",nc)
        print("Null %: {:.2f}%".format(100*nc/(nc+vc)))            
        print()


# In[ ]:


# preview columns, values, counts, etc
# identify columns that require special attention
# i.e. more than 10 unique values
val(df3)


# In[ ]:


df4 = df3.copy()

###
rp_col = "How many employees does your company or organization have?"
# fill na
# df4[cdf4[rp_col]] = df4[cdf4[rp_col]].fillna(-1)
# replace labels with
rp_dt = {'1-5':1,
        '6-25':6,
        '26-100':26,
        '100-500':101,
        '500-1000':501,
        'More than 1000':1001}

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Is your primary role within your company related to tech/IT?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA

rp_col = "Does your employer provide mental health benefits as part of healthcare coverage?"
rp_dt = {'Yes':1, # positive/yes response to qn will be 1
        "I don't know":2, # responses in increasing negativity will be 2 onwards
        'No':3,
        'Not eligible for coverage / N/A':-1
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Do you know the options for mental health care available under your employer-provided coverage?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes':1, # positive/yes response to qn will be 1
        'I am not sure':2, # responses in increasing negativity will be 2 onwards
        'No':3,
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes':1, # positive/yes response to qn will be 1
        "I don't know":2, # responses in increasing negativity will be 2 onwards
        'No':3,
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Does your employer offer resources to learn more about mental health concerns and options for seeking help?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes':1, # positive/yes response to qn will be 1
        "I don't know":2, # responses in increasing negativity will be 2 onwards
        'No':3,
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes':1, # positive/yes response to qn will be 1
        "I don't know":2, # responses in increasing negativity will be 2 onwards
        'No':3,
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {"Very easy":1, # positive/yes response to qn will be 1
        "Somewhat easy":2, # responses in increasing negativity will be 2 onwards
        "Neither easy nor difficult":3,
         "I don't know":3,
         "Somewhat difficult":4,
         "Very difficult":5
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Do you think that discussing a mental health disorder with your employer would have negative consequences?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes':1, # positive/yes response to qn will be 1
        'Maybe':2, # responses in increasing negativity will be 2 onwards
         'No':3,
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Do you think that discussing a physical health issue with your employer would have negative consequences?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes':1, # positive/yes response to qn will be 1
        'Maybe':2, # responses in increasing negativity will be 2 onwards,
         'No':3,
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Would you feel comfortable discussing a mental health disorder with your coworkers?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes':1, # positive/yes response to qn will be 1
        'Maybe':2, # responses in increasing negativity will be 2 onwards,
         'No':3,
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes':1, # positive/yes response to qn will be 1
        'Maybe':2, # responses in increasing negativity will be 2 onwards,
         'No':3,
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Do you feel that your employer takes mental health as seriously as physical health?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes':1, # positive/yes response to qn will be 1
        "I don't know":2, # responses in increasing negativity will be 2 onwards,
         'No':3,
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Have you heard of or observed negative consequences for co-workers who have been open about mental health issues in your workplace?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes':1, # positive/yes response to qn will be 1
        'No':2, # responses in increasing negativity will be 2 onwards,
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Do you have previous employers?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {0:2 # replace 0 (no) with 2 for consistency
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Have your previous employers provided mental health benefits?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes, they all did':1, # positive/yes response to qn will be 1
        'Some did':2, # responses in increasing negativity will be 2 onwards,
        "I don't know":3,
         'No, none did':4
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Were you aware of the options for mental health care provided by your previous employers?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes, I was aware of all of them':1, # positive/yes response to qn will be 1
        'I was aware of some':2, # responses in increasing negativity will be 2 onwards,
        'No, I only became aware later':3,
         'N/A (not currently aware)':4
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Did your previous employers ever formally discuss mental health (as part of a wellness campaign or other official communication)?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes, they all did':1, # positive/yes response to qn will be 1
        'Some did':2, # responses in increasing negativity will be 2 onwards,
        "I don't know":3,
         'None did':4
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Did your previous employers provide resources to learn more about mental health issues and how to seek help?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes, they all did':1, # positive/yes response to qn will be 1
        'Some did':2, # responses in increasing negativity will be 2 onwards,
        "I don't know":3,
         'None did':4
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Was your anonymity protected if you chose to take advantage of mental health or substance abuse treatment resources with previous employers?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes, always':1, # positive/yes response to qn will be 1
        'Sometimes':2, # responses in increasing negativity will be 2 onwards,
        "I don't know":3,
         'No':4
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Do you think that discussing a mental health disorder with previous employers would have negative consequences?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes, all of them':1, # positive/yes response to qn will be 1
        'Some of them':2, # responses in increasing negativity will be 2 onwards,
        "I don't know":3,
         'None of them':4
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Do you think that discussing a physical health issue with previous employers would have negative consequences?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes, all of them':1, # positive/yes response to qn will be 1
        'Some of them':2, # responses in increasing negativity will be 2 onwards,
         'None of them':3
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Would you have been willing to discuss a mental health issue with your previous co-workers?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes, at all of my previous employers':1, # positive/yes response to qn will be 1
        'Some of my previous employers':2, # responses in increasing negativity will be 2 onwards,
         'No, at none of my previous employers':3
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Would you have been willing to discuss a mental health issue with your direct supervisor(s)?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes, at all of my previous employers':1, # positive/yes response to qn will be 1
        'Some of my previous employers':2, # responses in increasing negativity will be 2 onwards,
         "I don't know":3,
         'No, at none of my previous employers':4
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Did you feel that your previous employers took mental health as seriously as physical health?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes, they all did':1, # positive/yes response to qn will be 1
        'Some did':2, # responses in increasing negativity will be 2 onwards,
        "I don't know":3,
         'None did':4
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Did you hear of or observe negative consequences for co-workers with mental health issues in your previous workplaces?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes, all of them':1, # positive/yes response to qn will be 1
        'Some of them':2, # responses in increasing negativity will be 2 onwards,
         'None of them':3
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Would you be willing to bring up a physical health issue with a potential employer in an interview?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes':1, # positive/yes response to qn will be 1
        'Maybe':2, # responses in increasing negativity will be 2 onwards,
         'No':3,
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Would you bring up a mental health issue with a potential employer in an interview?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes':1, # positive/yes response to qn will be 1
        'Maybe':2, # responses in increasing negativity will be 2 onwards,
         'No':3,
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Do you feel that being identified as a person with a mental health issue would hurt your career?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes, it has':1, # positive/yes response to qn will be 1
        'Yes, I think it would':2, # responses in increasing negativity will be 2 onwards,
        'Maybe':3,
         "No, I don't think it would":4,
         'No, it has not':5
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Do you think that team members/co-workers would view you more negatively if they knew you suffered from a mental health issue?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes, they do':1, # positive/yes response to qn will be 1
         'Yes, I think they would':2, # responses in increasing negativity will be 2 onwards,
        'Maybe':3,
         "No, I don't think they would":4,
         'No, they do not':5
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "How willing would you be to share with friends and family that you have a mental illness?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Very open':1, # positive/yes response to qn will be 1
         'Somewhat open':2, # responses in increasing negativity will be 2 onwards,
        'Neutral':3,
         'Somewhat not open':4,
         'Not open at all':5,
         'Not applicable to me (I do not have a mental illness)':-1
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Have you observed or experienced an unsupportive or badly handled response to a mental health issue in your current or previous workplace?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes, I experienced':1, # positive/yes response to qn will be 1
         'Yes, I observed':2, # responses in increasing negativity will be 2 onwards,
        'Maybe/Not sure':3,
         'No':4,
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Have your observations of how another individual who discussed a mental health disorder made you less likely to reveal a mental health issue yourself in your current workplace?"
# nan values is 55.41%; unsure what is the cause of nan values
# drop column
df4 = df4.drop([rp_col],axis=1)

###
rp_col = "Do you have a family history of mental illness?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes':1, # positive/yes response to qn will be 1
        "I don't know":2, # responses in increasing negativity will be 2 onwards
        'No':3,
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Have you had a mental health disorder in the past?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes':1, # positive/yes response to qn will be 1
        'Maybe':2, # responses in increasing negativity will be 2 onwards,
         'No':3,
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Do you currently have a mental health disorder?"
# potential target column or key X column
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes':1, # positive/yes response to qn will be 1
        'Maybe':2, # responses in increasing negativity will be 2 onwards,
         'No':3,
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Have you been diagnosed with a mental health condition by a medical professional?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes':1, # positive/yes response to qn will be 1
        'No':2, # responses in increasing negativity will be 2 onwards,
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Have you ever sought treatment for a mental health issue from a mental health professional?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {1:1, # positive/yes response to qn will be 1
        0:2, # responses in increasing negativity will be 2 onwards,
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "If you have a mental health issue, do you feel that it interferes with your work when being treated effectively?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Often':1, # positive/yes response to qn will be 1
        'Sometimes':2, # responses in increasing negativity will be 2 onwards,
        'Rarely':3,
        'Never':4,
        'Not applicable to me':-1}

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Often':1, # positive/yes response to qn will be 1
        'Sometimes':2, # responses in increasing negativity will be 2 onwards,
        'Rarely':3,
        'Never':4,
        'Not applicable to me':-1}

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Do you work remotely?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Always':1, # positive/yes response to qn will be 1
        'Sometimes':2, # responses in increasing negativity will be 2 onwards,
        'Never':3,
       }

df4[rp_col] = df4[rp_col].replace(rp_dt)

#####
df4.describe(include='all')


# ### 2.D Employer Type & Role

# In[ ]:


df4["Is your employer primarily a tech company/organization?"].value_counts()
# employer of 263 respondents is not primarily tech
# there are 883 respondents whose employers are primarily tech


# In[ ]:


df4["Is your primary role within your company related to tech/IT?"].value_counts()
# 883 '-1' responses correspond to num of respondents whose employers are primarily tech
# 15 respondents are neither in a tech company nor in a tech role
# these 15 respondents will be dropped
# expect 883+248 = 1131 rows left


# In[ ]:


# drop 15 respondents who are neither in tech company nor in tech role
df5 = df4.copy()
df5 = df5[df5["Is your primary role within your company related to tech/IT?"].isin([-1,1])]
df5.shape
# (1131, 48)


# In[ ]:


# rename first column, drop other column
err_msg = "Column has been dropped, renamed, or does not exist."
try: 
    df5 = df5.rename(columns={"Is your employer primarily a tech company/organization?":"tech_company_or_role"})
    df5 = df5.drop(["Is your primary role within your company related to tech/IT?"],axis=1)
except:
    print(err_msg)
    
df5.head(1)    


# ### 2.E Age
# * The IQR method of identifying outliers may not work well in this situation.
# * Instead, common sense will be applied.
# * Values below minimum legal working age of 13 (SG: 13; US: 14) will be replaced with mode age of 32.
# * Values more than a decade above retirement age of 62 (SG: 62; US: 66) will be replaced with mode age of 32.
# * This approach assumes that ages below 13 and above 72 are incorrect - either due to typos or intentional obscuration.

# In[ ]:


# brief detour:
# drop both MHDD and MHDS
# these columns will be dealt with in a later section 3.B
df6 = df5.copy()
df6 = df6.drop(["If yes, what condition(s) have you been diagnosed with?",
         "If maybe, what condition(s) do you believe you have?",
               "If so, what condition(s) were you diagnosed with?"],axis=1)
df6.head(1)


# In[ ]:



df6['What is your age?'].describe()
# min: 3, max 323 - weird values!


# In[ ]:


def rp_age(age):
    '''
    Replaces age below min or age above max with mode age.
    Else, returns age.'''
    mode = 32
    low,up = 13,72
    if age < 13: return mode
    elif age > 72: return mode
    else: return int(age)


# In[ ]:


# replace age
df6['What is your age?'] = df6['What is your age?'].apply(rp_age)


# ### 2.F Gender
# * The survey made gender a freeform field for gender input.
# * There were different ways of indicating the same gender, and a myriad of other gender types.
# * As much as possible, each individual's response will be respected. Obvious male or female responses will be mapped to 1 and 2 respectively, while any other response will be mapped to 3.
#     * eg a transgender male will be assigned 1 (male)

# In[ ]:


df6["What is your gender?"].value_counts()
# many ways to say same thing
# 'Others' category needed to hold responses that are not obviously 'Male' or 'Female'


# In[ ]:


df7 = df6.copy()
# prepare replacement lists
male_ls = ['Male','male', 'Male ', 'M', 'm', 'man', 'Cis male',
           'Male.', 'Male (cis)', 'Man', 'Sex is male',
           'cis male', 'Malr', 'Dude', "I'm a man why didn't you make this a drop down question. You should of asked sex? And I would of answered yes please. Seriously how much text can this take? ",
           'mail', 'M|', 'male ', 'Cis Male', 'Male (trans, FtM)',
           'cisdude', 'cis man', 'MALE']
# FYI: cisgender: describes a person who identifies as the same gender assigned at birth
female_ls = ['Female','female', 'I identify as female.', 'female ',
             'Female assigned at birth ', 'F', 'Woman', 'fm', 'f',
             'Cis female', 'Transitioned, M2F', 'Female or Multi-Gender Femme',
             'Female ', 'woman', 'female/woman', 'Cisgender Female', 
             'mtf', 'fem', 'Female (props for making this a freeform field, though)',
             ' Female', 'Cis-woman', 'AFAB', 'Transgender woman',
             'Cis female ']
# FYI: AFAB: assigned female at birth
other_ls = ['Bigender', 'non-binary,', 'Genderfluid (born female)',
            'Other/Transfeminine', 'Androgynous', 'male 9:1 female, roughly',
            'nb masculine', 'genderqueer', 'Human', 'Genderfluid',
            'Enby', 'genderqueer woman', 'Queer', 'Agender', 'Fluid',
            'Genderflux demi-girl', 'female-bodied; no feelings about gender',
            'non-binary', 'Male/genderqueer', 'Nonbinary', 'Other', 'none of your business',
            'Unicorn', 'human', 'Genderqueer']

# replace gender values with numberic labels
df7["What is your gender?"] = df7["What is your gender?"].replace(male_ls,1)
df7["What is your gender?"] = df7["What is your gender?"].replace(female_ls,2)
df7["What is your gender?"] = df7["What is your gender?"].replace(other_ls,3)
df7["What is your gender?"] = df7["What is your gender?"].fillna(3)
df7["What is your gender?"].unique()


# ### 2.G Country of Employment
# * Each unique country is replaced with a numeric label.
# * Country can be identified with country_rp_dt with key:value of label:country

# In[ ]:


df8 = df7.copy()
country_rp_dt = {}
for idx, name in enumerate(df8['What country do you work in?'].unique()):
#     print(idx, name)
    country_rp_dt[name] = idx
# country_rp_dt
df8['What country do you work in?'] = df8['What country do you work in?'].replace(country_rp_dt)


# ### 2.H Work Position
# * I'm not too interested in the work positions of respondents, although it may be interesting to identify whether any particular current work position is correlated to having a particular current MHD.
# * For now I will extract the work position column into a separate Series and drop it from the main df8 to form df9.

# In[ ]:


df8.head()


# In[ ]:


# extract column
work_pos_s = df8['Which of the following best describes your work position?']


# In[ ]:


df9 = df8.copy()
df9 = df9.drop(["Which of the following best describes your work position?"],axis=1)
df9.head(1)


# ### 2.I Mass Rename Columns
# * for easy handling when doing classification

# In[ ]:


# for easy viewing
cdf9 = col_ls(df9)


# In[ ]:


df10 = df9.copy()

df_rn_dt = {
    "How many employees does your company or organization have?":"num_employees",
    "Does your employer provide mental health benefits as part of healthcare coverage?":"cep_benefits",
    "Do you know the options for mental health care available under your employer-provided coverage?":"cep_know_options",
    "Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?":"cep_discuss",
    "Does your employer offer resources to learn more about mental health concerns and options for seeking help?":"cep_learn",
    "Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?":"cep_anon",
    "If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:":"cep_mh_leave",
    "Do you think that discussing a mental health disorder with your employer would have negative consequences?":"cep_mh_ncsq",
    "Do you think that discussing a physical health issue with your employer would have negative consequences?":"cep_ph_ncsq",
    "Would you feel comfortable discussing a mental health disorder with your coworkers?":"cep_comf_cw",
    "Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?":"cep_comf_sup",
    "Do you feel that your employer takes mental health as seriously as physical health?":"cep_serious",
    "Have you heard of or observed negative consequences for co-workers who have been open about mental health issues in your workplace?":"cep_others_ncsq",
    "Do you have previous employers?":"pep_have",
    "Have your previous employers provided mental health benefits?":"pep_benefits",
    "Were you aware of the options for mental health care provided by your previous employers?":"pep_know_options",
    "Did your previous employers ever formally discuss mental health (as part of a wellness campaign or other official communication)?":"pep_discuss",
    "Did your previous employers provide resources to learn more about mental health issues and how to seek help?":"pep_learn",
    "Was your anonymity protected if you chose to take advantage of mental health or substance abuse treatment resources with previous employers?":"pep_anon",
    "Do you think that discussing a mental health disorder with previous employers would have negative consequences?":"pep_mh_ncsq",
    "Do you think that discussing a physical health issue with previous employers would have negative consequences?":"pep_ph_ncsq",
    "Would you have been willing to discuss a mental health issue with your previous co-workers?":"pep_comf_cw",
    "Would you have been willing to discuss a mental health issue with your direct supervisor(s)?":"pep_comf_sup",
    "Did you feel that your previous employers took mental health as seriously as physical health?":"pep_serious",
    "Did you hear of or observe negative consequences for co-workers with mental health issues in your previous workplaces?":"pep_others_ncsq",
    "Would you be willing to bring up a physical health issue with a potential employer in an interview?":"fep_ph_willing",
    "Would you bring up a mental health issue with a potential employer in an interview?":"fep_mh_willing",
    "Do you feel that being identified as a person with a mental health issue would hurt your career?":"hurt_career",
    "Do you think that team members/co-workers would view you more negatively if they knew you suffered from a mental health issue?":"cw_view_neg",
    "How willing would you be to share with friends and family that you have a mental illness?":"comf_ff",
    "Have you observed or experienced an unsupportive or badly handled response to a mental health issue in your current or previous workplace?":"neg_response",
    "Do you have a family history of mental illness?":"mh_fam_hist",
    "Have you had a mental health disorder in the past?":"mh_hist",
    "Do you currently have a mental health disorder?":"mh_cur",
    "Have you been diagnosed with a mental health condition by a medical professional?":"mh_diag_pro",
    "Have you ever sought treatment for a mental health issue from a mental health professional?":"sought_treat",
    "If you have a mental health issue, do you feel that it interferes with your work when being treated effectively?":"work_affect_effect",
    "If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?":"work_affect_ineffect",
    "What is your age?":"age",
    "What is your gender?":"gender",
    "What country do you work in?":"work_country",
    "Do you work remotely?":"work_remote"
}

df10=df10.rename(columns=df_rn_dt)


# In[ ]:


df10.head(1)


# ## 3. Data Visualisations

# ### 3.A Mental Health Status across Ages and Genders
# * The largest proportion of genders is held by males, followed by females and then others.
# * The proportion of males who do not currently have MHD is larger than the proportion of males who do.
# * This is different from that observed among females. The proportion of females who currently have MHD is larger than the proportion of females who do.
# * For others, most have MHD while none do not currently have MHD.

# In[ ]:


g = sns.FacetGrid(df10,row="gender",col="mh_cur",size=4)
g.map(plt.hist,'age',alpha=0.5,bins=15)
g.add_legend()
plt.show()


# ### 3.B Mental Health Disorders (MHD)
# Notes on cleaning:
# * Some questions ask for MHD Diagnosed (MHDD) or Suspected (MHDS) and MHD diagnosed by professional (MHDDP).
# * MHDDP refer to MHDs that respondents have in the past and presently.
# * While MHDD and MHDS refer to MHDs that respondents have or suspect present.
# * I will clean up MHDD into a separate df and visualise it.
# * All 3 columns (MHDS, MHDDP) will be dropped from df5 to form df6.
# * Prior to MHD columns, respondents answer "Do you currently have a mental health disorder?" with Yes, No and Maybe.
#     * If yes, they largely answered MHDD.
#     * If maybe, they largely answered MHDS.
# * Answers to MHDD can include more than 1 MHD, separated by '|'.
# * Answeres to MHDD, for each MHD, includes a category name and sub disorders in '(...)'
# * Multiple disorders will be accounted for, but only the category name will be noted.
# 
# Insights will be mentioned below the graph.

# #### But first, some checks and cleaning up.

# In[ ]:


df5["If so, what condition(s) were you diagnosed with?"].describe()


# In[ ]:


# df5.describe(include='all')
print("Responses to 'Do you have MHD?':")
print(df5["Do you currently have a mental health disorder?"].value_counts())
df5[["Do you currently have a mental health disorder?",
     "If yes, what condition(s) have you been diagnosed with?",
    "If maybe, what condition(s) do you believe you have?"]].describe(include='all').iloc[0,:]


# In[ ]:


# extract column
mhd_ser = df5["If yes, what condition(s) have you been diagnosed with?"]
# convert to dict
mhd_dt = dict(mhd_ser.value_counts())
mhd_dt


# In[ ]:


# create new dict to split and count diagnosis
mhd = {}
for dia, count in  mhd_dt.items():
    dia_ls = dia.split('|')
    for d in dia_ls:
        d = d.split(' (')[0]
        mhd[d] = mhd.get(d,0) + count
mhd


# In[ ]:


# convert counter dict into df
# append one entry for each count
mhd_df = pd.DataFrame()
for d in mhd:
    mhd_df = mhd_df.append([d]*mhd[d])
    
# mass replacing values: group similar conditions under one category name
mhd_df = mhd_df.rename(columns={0:"Diagnosis"})
mhd_df["Diagnosis"] = mhd_df["Diagnosis"].replace(['Autism Spectrum Disorder',
                            'Autism','Autism - while not a "mental illness", still greatly affects how I handle anxiety',
                            'PDD-NOS','autism spectrum disorder','Autism spectrum disorder'],"Austism")

mhd_df["Diagnosis"] = mhd_df["Diagnosis"].replace(['posttraumatic stress disourder','PTSD'],
                                                  'Post-traumatic Stress Disorder')

mhd_df["Diagnosis"] = mhd_df["Diagnosis"].replace(['Attention Deficit Disorder',
                                                   'attention deficit disorder',
                                                   'ADD'],'Attention Deficit Hyperactivity Disorder')

mhd_df["Diagnosis"] = mhd_df["Diagnosis"].replace(['Schizotypal Personality Disorder'],
                                                   'Personality Disorder')
mhd_df["Diagnosis"] = mhd_df["Diagnosis"].replace(['Depression'],'Mood Disorder')
mhd_df["Diagnosis"] = mhd_df["Diagnosis"].replace(['Depression'],'Mood Disorder')
# mhd_df                                               


# In[ ]:


# observe value count
mhd_df['Diagnosis'].value_counts()


# In[ ]:


# place rare MHD under Others
mhd_df['Diagnosis']=mhd_df['Diagnosis'].replace([
    'Seasonal Affective Disorder','Asperger Syndrome','Asperges',
    'Suicidal Ideation','Gender Identity Disorder',
    'Psychotic Disorder','Dissociative Disorder',
    'Austism','Traumatic Brain Injury','Sleeping Disorder',
'Pervasive Developmental Disorder','Sexual addiction',
'Transgender'],'Others')
mhd_df['Diagnosis'].value_counts()


# In[ ]:


# plot
plot_df = mhd_df['Diagnosis'].value_counts()

fig = plt.figure(figsize=(16,8))

ax1 = fig.add_subplot(111)

plot_df.plot(kind='barh',ax=ax1)
ax1.set_title("Mental Health Disorder Distribution",fontsize=fs*2)
ax1.set_ylabel("Disorder",fontsize=fs)
ax1.set_xlabel("Frequency",fontsize=fs)

total=plot_df.sum()
[ax1.text(v+3,i-0.2,
          '{}\n{:.1f}%'.format(str(v),v/total*100),
          fontsize=fs) for i, v in enumerate(plot_df)]

plt.show()


# * Mood Disorders form the largest proportion of MHDs among the surveyed, at 35.3% or 322 out of 912 instances of MHDs.
#     * Mood Disorder causes one's general emotional state or mood to be distorted or inconsistent with circumstances, interfering with the one's ability to function.
# * Anxiety Disorder is 2nd at 29.3%.
#     * Anxiety Disorder also affects one's mood and often occurs alongside depression, which falls under Mood Disorders.
#     * This may be why Anxiety Disorder is the 2nd most frequent MHD among the surveyed, slightly less than the frequency of Mood Disorders but much more than the 3rd most frequent disorder.
# * ADHD is 3rd at 10.2%.
# * A myriad of other disorders follow at around 2.0-5.6% in frequency, or less than 56 persons among 1000.
# * Given that the most frequent disorders are Mood and Anxiety, which appear to largely affect the mood, it may not be a surprise that employers find it hard to accept or treat employees with these MHDs kindly.
# * That is, an employer may perceive employee to simply be moody or difficult than actually suffering from a disorder that requires medical attention.
# * Disorders like substance-abuse, PTSD, addictive might be more likely to raise medical concern, but these appear infrequently.

# ### 3.C Mental Health Status and Company Size
# * The varying proportions across company size is largely due to the distribution of company sizes.
# * For companies with 26-100 employees, there is a noticeably larger proportion of employees with current MHD than employees without.
#     * Furthermore, there is relatively less uncertainty (mh_cur = 2) for these companies.
#     * See how the proportion of people who are unsure is comparable across companies of sizes 6-25, 26-100, and 101-500, although the frequency of these companies differ.
# * Similarly for companies with 101-500 employees, although the increment is less obvious.
# * There is noticeably more people who do not have MHD in companies of size 501-1000 than people who do.

# In[ ]:


# Company Size Distribution
plot_df = df10['num_employees'].value_counts().reset_index(name='n').rename(columns={"index":"num_employees"}).sort_values("num_employees",ascending=False)
fig = plt.figure(figsize=(16,8))
ax1 = fig.add_subplot(111)
plot_df.plot(kind='barh',x = "num_employees", y = "n",ax=ax1)
ax1.set_title("Company Size by Employee Count Distribution",fontsize=fs*2)
ax1.set_ylabel("Employee Count",fontsize=fs)
ax1.set_xlabel("Frequency",fontsize=fs)
total=plot_df["n"].sum()
[ax1.text(v+3,i-0.2,
          '{}\n{:.1f}%'.format(str(v),v/total*100),
          fontsize=fs) for i, v in enumerate(plot_df["n"])]

# Prevalence of MHD across Company Size and Age
g = sns.FacetGrid(df10,row="num_employees",col="mh_cur",size=4)
g.map(plt.hist,'age',alpha=0.5,bins=15)
g.add_legend()

plt.show()


# ### 3.D Correlation Matrix of Features

# In[ ]:


# obtain list of feat names
cdf10 = col_ls(df10)


# In[ ]:


# create df where all -1 (N.A. responses) are replaced with np.NaN
df10_nan=df10.replace({-1:np.NaN})
df10_nan.describe()


# In[ ]:


# plot correlatin matrix
corr = df10_nan.corr() # use df with NaN values
fig = plt.figure(figsize=(16,16))
ax = fig.add_subplot(111)
cax = ax.matshow(corr,vmin=-1,vmax=1,
                 cmap=sns.diverging_palette(10, 500, as_cmap=True))
fig.colorbar(cax)
names = cdf10
ticks = np.arange(0,len(names),1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names,fontsize=fs,rotation='vertical')
ax.set_yticklabels(names, fontsize=fs)
ax.set_title("Correlation Matrix of Features",fontsize=fs*2,y=1.2)

plt.show()


# There are a zones where correlation is stronger.
# * Top left: features regarding current employment.
# * Middle: features regarding past employment.
# * Middle, Bottom Right: features regarding future employment and views of others.
# * Bottom right: features regarding mental health history, diagnosis, treatment and effect upon work.
# * Bottom left (& Top right): features regarding sentiment towards mental health and comfort in sharing about it.
# 
# NaN values are accounted for.

# In[ ]:


# top left: current employment
cep_ls=cdf10[2:14]
corr = df10_nan[cep_ls].corr()
fig = plt.figure(figsize=(20,8))
ax1 = fig.add_subplot(121)
cax = ax1.matshow(corr,vmin=-1,vmax=1,
                 cmap=sns.diverging_palette(10, 500, as_cmap=True))

names = cep_ls
ticks = np.arange(0,len(names),1)
ax1.set_xticks(ticks)
ax1.set_yticks(ticks)
ax1.set_xticklabels(names,fontsize=fs,rotation='vertical')
ax1.set_yticklabels(names, fontsize=fs)
ax1.set_title("Correlation Matrix of Current Employment Features",fontsize=fs,y=1.4)

# fig.colorbar(cax) # intentionally left out for easier comparison between CM graphs

# middle: previous employment
pep_ls=cdf10[15:26]
df_corr = df10_nan[pep_ls]
corr = df_corr.corr()
ax2 = fig.add_subplot(122)
cax = ax2.matshow(corr,vmin=-1,vmax=1,
                 cmap=sns.diverging_palette(10, 500, as_cmap=True))
names = pep_ls
ticks = np.arange(0,len(names),1)
ax2.set_xticks(ticks)
ax2.set_yticks(ticks)
ax2.set_xticklabels(names,fontsize=fs,rotation='vertical')
ax2.set_yticklabels(names, fontsize=fs)
ax2.set_title("Correlation Matrix of Previous Employment Features",fontsize=fs,y=1.4)

plt.show()


# Notable correlations from Current Employment Features:
# * The lesser sense of threat from bringing up MH, the easier to obtain leave for MH issues.
# * The greater the sense of threat from bringing up PH, the greater the sense of threat from bring up MH.
# * The greater the comfort in sharing about MH with coworkers or supervisors, the easier to obtain leave for MH issues.
# * The more employer views MH as seriously as PH, the less the sense of threat from bringing up MH.
# 
# Notable correlations from Previous Employment Features:
# * Features of previous employment related to company policy (e.g. availability of MH benefits) is more strongly positively related than compared to features of current employment.
# * However, the negative correlation between comfort of sharing about MH issues with supervisors & coworkers and sense of threat of raising MH issues is milder for previous employment than current employment.
# * The general direction of correlation is similar for both previous and current employment features.

# In[ ]:


# middle, bottom right: future employment and perceptions
fep_ls=cdf10[26:31]
corr = df10_nan[fep_ls].corr()
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
cax = ax.matshow(corr,vmin=-1,vmax=1,
                 cmap=sns.diverging_palette(10, 500, as_cmap=True))
fig.colorbar(cax)
names = fep_ls
ticks = np.arange(0,len(names),1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names,fontsize=fs,rotation='vertical')
ax.set_yticklabels(names, fontsize=fs)
ax.set_title("Correlation Matrix of Future Employment Features",fontsize=fs,y=1.4)

plt.show()


# Notable correlations from Future Employment Features:
# * The more willing one is with sharing about PH, the more willing one is with sharing about MH.
# * The more one feels that being identified as having MHD will hurt one's career, the less willing one is to share about MH.
# * The more one feels that coworkers will view one more negatively if one has MHD, the more one feels that being identified as having MHD will hurt one's career.

# In[ ]:


# bottom right: mental health history
mhh_ls=cdf10[30:39]
corr = df10_nan[mhh_ls].corr()
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
cax = ax.matshow(corr,vmin=-1,vmax=1,
                 cmap=sns.diverging_palette(10, 500, as_cmap=True))
fig.colorbar(cax)
names = mhh_ls
ticks = np.arange(0,len(names),1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names,fontsize=fs,rotation='vertical')
ax.set_yticklabels(names, fontsize=fs)
ax.set_title("Correlation Matrix of Mental Health History Features",fontsize=fs,y=1.4)

plt.show()


# Notable correlations from Mental Health History Features:
# * The strongest correlations are found in this slice of the general CM.
# * One's comfort in sharing with family and friends is surprisingly only mildly or not correlated with features like
#     * one's experience or observation of negative response to opening up about MH issues
#     * one's family history of MH issues
#     * one's current or historical MH status
# * As expected, if one has a history of MHD or has been professionally diagnosed, one is more likely to have sought treatment for MHD.
# * There seems to be no correlation between the degree of MHD's interference with work (when treated effectively) and whether one has sought treatment for MHD.
#     * However, the more one finds that work is interfered when MHD is not effectively treated, the more likely one has sought treatment for MHD.

# In[ ]:


# bottomn left: current employment and future employment
cfep_ls = cdf10[7:13] + cdf10[26:31]
corr = df10_nan[cfep_ls].corr()
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
cax = ax.matshow(corr,vmin=-1,vmax=1,
                 cmap=sns.diverging_palette(10, 500, as_cmap=True))
fig.colorbar(cax)
names = cfep_ls
ticks = np.arange(0,len(names),1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names,fontsize=fs,rotation='vertical')
ax.set_yticklabels(names, fontsize=fs)
ax.set_title("Correlation Matrix of Current and Future Employment Features",fontsize=fs,y=1.4)

plt.show()


# Notable correlations from Current and Future Employment Features:
# * Current experiences affect future willingness.
# * The more one senses a threat of negative consequences when raising MH concerns with current employer, the less likely one is willing to raise MH issues with future employers.
#     * And more likely one will feel that being identified as having MHD will hurt one's career and cause coworkers to view one negatively.
# * The more willing one is with sharing about PH with future employers, the more willing one is with sharing about MH as well.
# * The more one feels that coworkers will view one negatively for having MHD, the more one feels that being identified as having MHD will hurt one's career.
# 

# ## 4. Classifiers

# * Predict degree of comfort in sharing about MH issues with Future Employers ("fep_mh_willing") based on profile and current & past experiences.
# * Predict whether one will seek professional diagnosis ("mh_diag_pro") based on profile and current & past experiences.
# * Predict whether one will seek treatment ("sought_treatment") based on profile and current & past experiences.

# In[ ]:


# prepare x variables
profile_ls = cdf10[0:2] + cdf10[30:31] + cdf10[32:35] + cdf10[39:]
cpep_ls = cdf10[2:14] + cdf10[14:26] + cdf10[28:30] + cdf10[31:32] + cdf10[37:39]
# profile_ls
# cpep_ls
x_col = profile_ls + cpep_ls
x_col


# In[ ]:


# prepare y variables
y1_col = "fep_mh_willing"
y2_col = "mh_diag_pro"
y3_col = "sought_treat"


# In[ ]:


X = df10[x_col]
y1 = df10[y1_col]
y2 = df10[y2_col]
y3 = df10[y3_col]


# ### 4.A RFC: Openness about MH with Future Employer (93%)

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


# prep for y1
X_train, X_test, y_train, y_test = train_test_split(X, y1, 
                                                    test_size = 0.25,
                                                   random_state=42)
print(X_train.count())
print(y_train.count())
print(X_test.count())
print(y_test.count())


# In[ ]:


clf = RandomForestClassifier(n_estimators=200, random_state=0)  
clf.fit(X_train, y_train)  
# y_pred = clf.predict(X_test)


# In[ ]:


# Returns the mean accuracy on the given test data and labels.
clf.score(X,y1)
# 0.9221927497789567


# In[ ]:


print(clf.feature_importances_)


# In[ ]:


y1_feat_impt = list(clf.feature_importances_)
y1_feat_impt
y1_df = pd.DataFrame({"Feature":x_col,"Importance":y1_feat_impt})
y1_sort_df = y1_df.sort_values("Importance")


# In[ ]:


fig = plt.figure(figsize=dim)
ax1 = fig.add_subplot(111)

y1_sort_df.plot(kind='barh',ax=ax1,x="Feature",y="Importance")

ax1.set_title("Willingess to Share with Future Employer: Feature Importance",fontsize=fs)
ax1.set_ylabel("Feature",fontsize=fs)
ax1.set_xlabel("Importance",fontsize=fs)

plt.show()


# * Age is the most important predictor, although there appears to be no linear correlation with willingess to share about MH with future employer, as observed in the Correlation MAtrix (CM) below.
# * The sentiment that being identified as having MHD as detrimental to one's career is the next most mportant predictor.
# * Intuitively, the more one feels that being identified with MHD will hurt one's career, the less likely one will bring it up with a future employer.

# In[ ]:


# To study the links between openness with future employer, age and career prospect
tmp = cdf10[27:29] + cdf10[39:40]
corr = df10_nan[tmp].corr()
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
cax = ax.matshow(corr,vmin=-1,vmax=1,
                 cmap=sns.diverging_palette(10, 500, as_cmap=True))
fig.colorbar(cax)
names = tmp
ticks = np.arange(0,len(names),1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names,fontsize=fs,rotation='vertical')
ax.set_yticklabels(names, fontsize=fs)
ax.set_title("CM of Age, Career Prospects and Openness with Future Employment",fontsize=fs,y=1.4)

plt.show()


# ### 4.B RFC: Seeking Professional Diagnosis (90+%)

# In[ ]:


# prep for y1
X_train, X_test, y_train, y_test = train_test_split(X, y2, 
                                                    test_size = 0.25,
                                                   )
print(X_train.count())
print(y_train.count())
print(X_test.count())
print(y_test.count())


# In[ ]:


clf2 = RandomForestClassifier(n_estimators=200, random_state=0)  
clf2.fit(X_train, y_train)  
# y_pred = clf.predict(X_test)


# In[ ]:


# Returns the mean accuracy on the given test data and labels.
clf2.score(X,y2)
# 0.9664014146772767


# In[ ]:


print(clf2.feature_importances_)


# In[ ]:


y2_feat_impt = list(clf2.feature_importances_)
# y2_feat_impt
y2_df = pd.DataFrame({"Feature":x_col,"Importance":y2_feat_impt})
y2_sort_df = y2_df.sort_values("Importance")


# In[ ]:


fig = plt.figure(figsize=dim)
ax1 = fig.add_subplot(111)

y2_sort_df.plot(kind='barh',ax=ax1,x="Feature",y="Importance")

ax1.set_title("Seeking Professional Diagnosis: Feature Importance",fontsize=fs)
ax1.set_ylabel("Feature",fontsize=fs)
ax1.set_xlabel("Importance",fontsize=fs)

plt.show()


# #### Drop some features that presuppose Diagnosis
# * It might be better to drop the features mh_hist, mh_cur, work_affect_effect, work_affect_ineffect as these features are dependent on one having sought professional diagnosis.
# * This dependence led to overly high feature importance.

# In[ ]:


# drop features and retry classifier
try:
    X = X.drop(["mh_hist","mh_cur","work_affect_effect", "work_affect_ineffect"],axis=1)
except: print(err_msg)


# In[ ]:


# prep for y1
X_train, X_test, y_train, y_test = train_test_split(X, y2, 
                                                    test_size = 0.25)
print(X_train.count())
print(y_train.count())
print(X_test.count())
print(y_test.count())


# In[ ]:


clf2 = RandomForestClassifier(n_estimators=200, random_state=0)  
clf2.fit(X_train, y_train)  
# y_pred = clf.predict(X_test)


# In[ ]:


# Returns the mean accuracy on the given test data and labels.
clf2.score(X,y2)
# 0.9310344827586207, drop from 0.9752431476569408
# but still pretty accurate


# In[ ]:


print(clf2.feature_importances_)


# In[ ]:


y2_feat_impt = list(clf2.feature_importances_)
# y2_feat_impt
y2_df = pd.DataFrame({"Feature":X.columns.tolist(),"Importance":y2_feat_impt})
y2_sort_df = y2_df.sort_values("Importance")


# In[ ]:


fig = plt.figure(figsize=dim)
ax1 = fig.add_subplot(111)

y2_sort_df.plot(kind='barh',ax=ax1,x="Feature",y="Importance")

ax1.set_title("Seeking Professional Diagnosis: Feature Importance",fontsize=fs)
ax1.set_ylabel("Feature",fontsize=fs)
ax1.set_xlabel("Importance",fontsize=fs)

plt.show()


# * Having a family history of MH issues is the most important predictor of seeking professional diagnosis.
# * Age and openness in sharing about MHD with family or friends are also important predictors over the remaining features.
#     * The older, the more likely to seek professional diagnosis.
#     * The more comfortable with family or friends, the more likely to seek diagnosis.
# * The most important predictors related to current employment are knowing the MH care options provided by current employment as well as ease of obtaining leave for MH.
#     * The more one knows one's options with current employer, the more one may seek diagnosis.
#     * Curiously, the harder to obtain leave for MH, the more likely one may seek diagnosis.
# * The experience of negative responses to MH issues is a similarly important predictor.
#     * The less negative response experienced or observed, the more one may seek diagnosis. 

# In[ ]:


# To study the links between openness with future employer, age and career prospect
tmp = cdf10[35:36] + cdf10[32:33] + cdf10[39:40] + cdf10[30:31] + cdf10[3:4] + cdf10[7:8] + cdf10[31:32]
corr = df10_nan[tmp].corr()
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
cax = ax.matshow(corr,vmin=-1,vmax=1,
                 cmap=sns.diverging_palette(10, 500, as_cmap=True))
fig.colorbar(cax)
names = tmp
ticks = np.arange(0,len(names),1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names,fontsize=fs,rotation='vertical')
ax.set_yticklabels(names, fontsize=fs)
ax.set_title("CM of Age, Career Prospects and Openness with Future Employment",fontsize=fs,y=1.4)

plt.show()


# ### 4.C RFC: Seeking Treatment (90+%)

# In[ ]:


# reset X
X = df10[x_col]


# In[ ]:


# prep for y1
X_train, X_test, y_train, y_test = train_test_split(X, y3, 
                                                    test_size = 0.25,
                                                   )
print(X_train.count())
print(y_train.count())
print(X_test.count())
print(y_test.count())


# In[ ]:


clf3 = RandomForestClassifier(n_estimators=200, random_state=0)  
clf3.fit(X_train, y_train)  
# y_pred = clf.predict(X_test)


# In[ ]:


# Returns the mean accuracy on the given test data and labels.
clf3.score(X,y3)
# 0.969053934571176


# In[ ]:


print(clf3.feature_importances_)


# In[ ]:


y3_feat_impt = list(clf.feature_importances_)
# y3_feat_impt
y3_df = pd.DataFrame({"Feature":x_col,"Importance":y3_feat_impt})
y3_sort_df = y3_df.sort_values("Importance")


# In[ ]:


fig = plt.figure(figsize=dim)
ax1 = fig.add_subplot(111)

y3_sort_df.plot(kind='barh',ax=ax1,x="Feature",y="Importance")

ax1.set_title("Seeking Treatment: Feature Importance",fontsize=fs)
ax1.set_ylabel("Feature",fontsize=fs)
ax1.set_xlabel("Importance",fontsize=fs)

plt.show()


# * Age is the most important predictor (again). The older one is, the more likely one will have sought treatment.
# * The sentiment that being identified as having MHD as detrimental to one's career is the next most mportant predictor.
# * Intuitively, the more one feels that being identified with MHD will hurt one's career, the more likely one would have sought treatment for it.

# In[ ]:


# To study the links between openness with future employer, age and career prospect
tmp = cdf10[36:37] + cdf10[39:40] + cdf10[28:29]
corr = df10_nan[tmp].corr()
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
cax = ax.matshow(corr,vmin=-1,vmax=1,
                 cmap=sns.diverging_palette(10, 500, as_cmap=True))
fig.colorbar(cax)
names = tmp
ticks = np.arange(0,len(names),1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names,fontsize=fs,rotation='vertical')
ax.set_yticklabels(names, fontsize=fs)
ax.set_title("CM of Age, Career Prospects and Openness with Future Employment",fontsize=fs,y=1.4)

plt.show()


# ## 5. Credits
# * Dataset: https://www.kaggle.com/osmi/mental-health-in-tech-2016
# * With some reference and inspiration from:
#     * Data Visualisation Python Seaborn kernel [https://www.kaggle.com/jchen2186/data-visualization-with-python-seaborn]
#     * Preprocessing and Random Forest with 87% accuracy kernel [https://www.kaggle.com/h1rshit/preprocessing-and-random-forest-with-87-accuracy]
