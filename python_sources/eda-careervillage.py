#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Big thanks to [GaneshN](https://www.kaggle.com/ganeshn88) for contributing the wordcloud code below!

# # Students

# In[ ]:


students = pd.read_csv('../input/students.csv')
students.head()


# In[ ]:


print('max:',students.students_date_joined.max(), '\n' + 'min:',students.students_date_joined.min())


# Students' join dates range from December 2011 to as late as January of this year!

# In[ ]:


print(students.students_location.shape)
print(students.students_location.isnull().sum())


# In[ ]:


locs = pd.Series(students.students_location.tolist(), index=students.students_location).apply(lambda x: len(str(x)))
locs.sort_values().head()


# In[ ]:


locs.sort_values(ascending=False).head()


# In[ ]:


students.students_location.value_counts().head(5)


# The good news is that we have locations for the vast majority of students, and these span a wide variety of states and countries. The bad news is that the locations aren't standardized, so there will be some preprocessing of the data required.
# 
# Sometimes they're in `City, State` format, other times `City, Country`, and in one case I noticed `City, State, Country`. 
# 
# The top student locations are primarily large cities in the United States and India: New York (pop 8.6mm), Bengaluru (pop 12.34mm), and Los Angeles (pop: 4mm).
# 
# 

# # Professionals

# In[ ]:


professionals = pd.read_csv('../input/professionals.csv')
professionals.head()


# In[ ]:


professionals.professionals_location.value_counts().head(5)


# In[ ]:


professionals.professionals_industry.value_counts().head(5)


# In[ ]:


professionals.professionals_headline.value_counts().head(5)


# In[ ]:


print('max:',professionals.professionals_date_joined.max(), '\n' + 'min:',professionals.professionals_date_joined.min())


# In[ ]:


print(professionals.professionals_location.shape)
print(professionals.professionals_location.isnull().sum())


# In[ ]:


locs = pd.Series(professionals.professionals_location.tolist(), index=professionals.professionals_location).apply(lambda x: len(str(x)))
locs.loc[professionals.professionals_location.notnull().tolist()].sort_values().head()


# In[ ]:


locs.sort_values(ascending=False).head()


# The site seems to skew heavily toward the computer industry, which I suppose isn't unexpected!
# 
# It's a bit odd that one of the top headlines is "Assurance Associate at PwC". Looking at CareerVillage's [website](https://www.careervillage.org/partners/), I noticed that PwC is a corporate partner. Problem solved!

# # Questions

# In[ ]:


questions = pd.read_csv('../input/questions.csv')


# In[ ]:


questions.head()


# **QUESTIONS TITLE**

# In[ ]:


from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
words = ' '.join(questions['questions_title'])
wordcloud = WordCloud(stopwords=STOPWORDS,max_words=500,
                      background_color='white',min_font_size=6,
                      width=3000,collocations=False,
                      height=2500
                     ).generate(words)
plt.figure(1,figsize=(20, 20))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# **questions_body**

# In[ ]:


from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
words = ' '.join(questions['questions_body'])
wordcloud = WordCloud(stopwords=STOPWORDS,max_words=500,
                      background_color='black',min_font_size=6,
                      width=3000,collocations=False,
                      height=2500
                     ).generate(words)
plt.figure(1,figsize=(20, 20))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
wordcloud_words = wordcloud.words_
s  = pd.Series(wordcloud_words,index=wordcloud_words.keys())
s.head(5)


# In[ ]:


answers = pd.read_csv('../input/answers.csv')


# In[ ]:


answers.head()


# In[ ]:


comments = pd.read_csv('../input/comments.csv')


# In[ ]:


comments.head()


# In[ ]:


emails = pd.read_csv('../input/emails.csv')


# In[ ]:


emails.head()

