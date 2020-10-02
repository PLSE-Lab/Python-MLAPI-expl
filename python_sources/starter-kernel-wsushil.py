#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


get_ipython().system('ls ../input')


# ### Howdy
# 
# Your job in this Data Science for Good competition is to develop a recommendation engine that will suggest relevant questions to professionals via email. Students ask questons on the CareerVillage.org platform and Professionals answer them. CareerVillage.org has a pretty good recommendation system inplace that's based on hard-coded rules. They would love to make it more efficient and improve it's performance. That's where you come in :)
# 
# Since questions and answers are the main focus of this competition, I thought it would be helpful to join a few tables together to help get things started. This is totally not an exhaustive list of things to do and there's a lot more data to explore so go get wild!
# 

# It's all about questions and answer and how to make proper recommendations so these tables might be a good palce to start:
# 
# - questions.csv
# - answers.csv
# - professionals.csv

# In[ ]:


questions = pd.read_csv('../input/questions.csv')
answers = pd.read_csv('../input/answers.csv')
professionals = pd.read_csv('../input/professionals.csv')


# In[ ]:


questions.head()


# In[ ]:


answers.head()


# In[ ]:


professionals.head()


# First I'll join questions and answers together. Notice that each column name is prepended with the table name. 
# 
# 

# In[ ]:


question_answers = questions.merge(right=answers, how='inner', left_on='questions_id', right_on='answers_question_id')


# In[ ]:


question_answers.head()


# The next thing I will do add the professionals table, because at the end of the day we'll probably need to know who answers which kinds of questions.

# In[ ]:


qa_professionals = question_answers.merge(right=professionals, left_on='answers_author_id', right_on='professionals_id')
qa_professionals.head()


# Ok friends, there's now a table with questions, answers, and the professionals that answered them. This may or may not be helpful, I'll leave that up to you to decide!
# 
# 
# Good luck!
# 
# 
# 
