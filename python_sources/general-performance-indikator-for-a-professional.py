#!/usr/bin/env python
# coding: utf-8

# ## General performance indikators for a professional
# 
# For each professional I will calculate a "answered_questions_per_10days_quote". So I have a comparable answering/activity value for all professionals.
# 
# (For later improvements: This is just an quantitativ value. We don't now the quality of the answers)
# 
# 
# 
# 
# 
# ---
# 
# This is just a simple first solution. At this point I don't use machine learning but there are some ways to improve this solution later.
# 
# 
# 
# Of course I looked in all CSVs first, and for the first section I only will use the answers, questions and professionals data.

# In[ ]:


# Load some modules...
import pandas as pd
import numpy as np
import calendar 
import time 


# ### Section 1 - the "answered_questions_per_10days_quote"
# 

# In[ ]:


# Load the data in dataframes...
df_professionals = pd.read_csv("../input/professionals.csv")
df_answers = pd.read_csv("../input/answers.csv")
df_questions = pd.read_csv("../input/questions.csv")


# Now I merge questions an answers by ID and add a column "question count" with value 1 to all question-answer combinations. 
# 
# After that I can group our question-answer combinations by "answers_author_id" and sum up the "question count"-values.

# In[ ]:


df_questions_answers = df_questions.merge(right=df_answers, how='inner', 
                                         left_on='questions_id', 
                                         right_on='answers_question_id')

df_questions_answers["question_count"] = 1
df_questions_count = df_questions_answers.groupby("answers_author_id").sum()


# Let's take a look what we created...

# In[ ]:


print(df_questions_count.head())


# Looks good! Now we know how many questions a professional has answered
# 
# ---
# 
# Let's get the days since a professional is registered... a short look in the professionals dataframe

# In[ ]:


df_professionals.head()


# We see that we can use the column "professionals_date_joined" to get a timestamp from the registration date. We only need the date and not the time, so we will split the value at the first white-space and convert the date to a timestamp.
# 
# Then we can substract the registration timestamp from todays timestamp to get the time between registration and today in secondsband convert the seconds to days. We write the value in a new column 

# In[ ]:



def calc_days_since_joined(x):
    registration_timestamp = x.split(" ", 1)[0]
    registration_timestamp = calendar.timegm(time.strptime(registration_timestamp, '%Y-%m-%d'))
    timestamp = time.time()    
    
    result = int((timestamp - registration_timestamp)/60/60/24)
    
    return result


# We use copy of the professionals dataframe, so we can drop some columns without losing them
df_professionals_custom = df_professionals.copy()

# Here is the real action in....
df_professionals_custom["days_since_join"] = df_professionals["professionals_date_joined"].apply(calc_days_since_joined)

# We drop some columns, we don't need them in this dataframe
df_professionals_custom.drop(["professionals_location", "professionals_industry", "professionals_headline", "professionals_date_joined"], axis=1, inplace=True)



# ok, let's take a look what we created...

# In[ ]:


df_professionals_custom.head()


# Perfect! Now we know how many days a professional is registered
# 
# --- 
# 
# So let's merge both together
# 

# In[ ]:


df_professionals_custom = df_professionals_custom.merge(right=df_questions_count, how='inner', 
                                                 left_on='professionals_id', 
                                                 right_on='answers_author_id')

print(df_professionals_custom.head())


# Our dataframe is ready to calculate our "answered_questions_per_10days_quote" 

# In[ ]:


# 
def answered_questions_quote(x):
    result = x["question_count"] / (x["days_since_join"]/10)
    return result
    
    
df_professionals_custom["answered_questions_per_10days_quote"] = df_professionals_custom.apply(answered_questions_quote, axis=1)



df_professionals_custom.head()


# Awesome!! We have a comparable answering/activity value for all professionals
# 

# I will use this indikator in my solution later... and of course I will publish it here.
# 
# 
# **Thanks for reading!**

# In[ ]:




