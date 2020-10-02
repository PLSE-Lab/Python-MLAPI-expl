#!/usr/bin/env python
# coding: utf-8

# Copyright [jeeu] [name of copyright owner]
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# # Simple Solution in CareerVilage

# ## Abstract

# <b> This is a very simple solution that uses similarities to recommend the appropriate professionals. </b>

# ## Mechanism

# ### About Professional Recognition

# <b> First of all, What we need to do is identify experts.</b> <br>
# Analyze the answers on each post to identify the keywords of the expert who gave the answers.

# In[ ]:


import pandas as pd
import numpy as np
import os
import sys
from gensim.summarization import summarize 
from gensim.summarization import keywords
import re


# In[ ]:


print(os.listdir("../input"))
PATH = "../input/"


# In[ ]:


answers = pd.read_csv(PATH + 'answers.csv')
comments = pd.read_csv(PATH + 'comments.csv')
emails = pd.read_csv(PATH + 'emails.csv')
group_memberships = pd.read_csv(PATH + 'group_memberships.csv')
groups = pd.read_csv(PATH + 'groups.csv')
matches = pd.read_csv(PATH + 'matches.csv')
professionals = pd.read_csv(PATH + 'professionals.csv')
questions = pd.read_csv(PATH + 'questions.csv')
school_memberships = pd.read_csv(PATH + 'school_memberships.csv')
students = pd.read_csv(PATH + 'students.csv')
tag_questions = pd.read_csv(PATH + 'tag_questions.csv')
tag_users = pd.read_csv(PATH + 'tag_users.csv')
tags = pd.read_csv(PATH + 'tags.csv')


# <b> As you can see below, you need to pre-process answer. </b>

# In[ ]:


answers.head()


# In[ ]:


answers['answers_body'].head()


# <b> Merge 'Professionals' and 'answers' </b>

# In[ ]:


Professionals_ID = pd.merge(professionals[['professionals_id','professionals_industry','professionals_headline']], answers[['answers_author_id','answers_id','answers_body']], left_on='professionals_id', right_on='answers_author_id', how='inner')
Professionals_ID.sort_values('professionals_id')

Professionals_ID.head()


# <b> Check the length.

# In[ ]:


len(Professionals_ID.answers_body)


# At this time, We can see that there are 10,000 differences.(If you use to pd.merge option "how=outer", you can see 60,000)<br>
# <b> We're going to rule out 10,000 ghost users.

# Add "answers_keywords" column

# In[ ]:


Professionals_ID["answers_keywords"] = ""

Professionals_ID.head()


# We execute to preprocess 'answers_body' (Remove Na/NaN, Sub, etc..)

# In[ ]:


Professionals_ID['answers_body'].fillna("No Answer", inplace = True)
for num in range(len(Professionals_ID.answers_body)):
    Answers_Text = Professionals_ID['answers_body'][num]
    Answers_Text = re.sub('<.+?>', '', Answers_Text, 0, re.I|re.S)
    Answers_Text = re.split('\n', Answers_Text)
    for n in Answers_Text:
        if (n.startswith('http')):
            del Answers_Text[Answers_Text.index(n)]
    
    result = ""
    for n in Answers_Text:
        result += n + ". "
    
    if (num%1000 == 0 ):
        try:
            print("iteration : " + str(num))
            print("1. Summarizing :", '\n', summarize(str(result)), '\n')
            print("######################################################################################################", '\n')
            print("2. Keywords :", '\n', keywords(str(result)))
        except ValueError:
            pass  # do nothing!
    
    keywords1 = re.split('\n', keywords(str(result)))
    
    Professionals_ID["answers_keywords"][num] = keywords1


# Check this Dataframe.

# In[ ]:


Professionals_ID[['professionals_id','answers_body','answers_keywords']].tail(10)


# In[ ]:


Professionals_ID.to_csv("Professinals_kewords", sep='\t', encoding='utf-8')


# ## What's Next?

# If I have more time, I'll try to something below.

# ### Make similarities

# Use Jaccard similarity, then compare to new questions that have new keywords

# ### Make Model

# It's quite simple model, but I'am assured that it makes sense.
