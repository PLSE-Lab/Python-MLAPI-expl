#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


multi_choice = pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv')
others = pd.read_csv('/kaggle/input/kaggle-survey-2019/other_text_responses.csv')
questions = pd.read_csv('/kaggle/input/kaggle-survey-2019/questions_only.csv')


# # Introduction
# 
# If you were looking to becoming a data scientist, you are definetly at the right place. This survey provides insights regarding to being in the ML / DS Field. **I'm still new to data science** and I'm also exploring the survey same as you do to get more data or information about it.
# 
# This kernel brings you information regarding to the current data science field and answer the one question that most people have been wondering... **Is it hard to become a data scientist?**

# # Different perspectives
# 
# We start off by looking at the age distribution among the survey participants. 

# In[ ]:


a = plt.bar(multi_choice.Q1.drop(0).value_counts().keys(), multi_choice.Q1.drop(0).value_counts())
a[0].set_color('r')
a[1].set_color('r')
a[2].set_color('r')
a[3].set_color('r')
plt.tight_layout()
plt.title('Age distribution of survey participants')
plt.show()


# Clearly, this survey shows that most of the participant's age range from 18 - 34. We will seperate these participants into junior data scientist and senior data scientist based on their age range. Different age group might provide different perspectives on each questions given. Junior data scientist might provide some insight on how to start your data scientist journey while senior data scientist might give you tips on how to improve your skillset. 

# In[ ]:


junior = multi_choice.loc[multi_choice.Q1.isin(multi_choice.Q1.value_counts().nlargest(4).keys())]


# In[ ]:


print('This survey contains {} junior data scientist'.format(junior.shape[0]))


# In[ ]:


senior = multi_choice.loc[~multi_choice.Q1.isin(multi_choice.Q1.value_counts().nlargest(4).keys())]


# In[ ]:


print('We got around {} senior data scientist'.format(senior.shape[0]))


# # Does gender matters? 

# In[ ]:


junior_age = junior.Q2.value_counts()
junior_age.plot.pie(y=junior_age.index,
           shadow=False,
           explode=(0, 0, 0.2, 0.5),
           startangle=45,
           autopct='%1.1f%%')

plt.axis('equal')
plt.title('Gender distribution based on Junior DS')
plt.tight_layout()
plt.show()


# In[ ]:


senior_age = senior.Q2.drop(0).value_counts()
senior_age.plot.pie(y=senior_age.index,
           shadow=False,
           explode=(0, 0, 0.2, 0.5),
           startangle=45,
           autopct='%1.1f%%')

plt.axis('equal')
plt.title('Gender distribution based on Senior DS')
plt.tight_layout()
plt.show()


# There are might be more female programmers / data scientist out there that didn't took part in this survey so we can't say that gender affects in your roadmap towards being a data scientist. However, based on this two pie chart, we can see that there's a slight decrease (~ 4%) in promoted from junior programmer / data scientist towards senior data scientist. I personally think that passion and hardwork are the key ingredients in being a successful programmer / data scientist. 

# # Do you need a degree to become a data scientist?

# In[ ]:


ax = senior.Q4.value_counts(normalize=True).nlargest(3).plot(kind='barh')
plt.title('What is the highest level of formal education \n             that you have attained or plan to attain \n within the next 2 years? \n(Senior)')
ax.patches[2].set_facecolor('r')


# In[ ]:


ax = junior.Q4.value_counts(normalize=True).nlargest(3).plot(kind='barh')
plt.title('What is the highest level of formal education \n             that you have attained or plan to attain \nwithin the next 2 years? \n(Junior)')
ax.patches[1].set_facecolor('r')


# One of the most common questions that has been asked. 
# 
# The survey shows that if you were to become a senior data scientist, YES you need one and at least a Master's Degree. Over 70% of the survery participants that were a senior data scientist requires a Master's degree of above.
# 
# Being a junior data scientist requires you to have at least a Bachelor's Degree or persuing one. The reason why this is true for most people is because that university of college will provide you some sort of basic math understanding and other related programming skills in order to solidify your core data science skill. 
# 
# #### Personal Comment: 
# Nowadays, huge number of online courses have been quite popular to help you start your journey to become a data scientist. All these online courses fee cost less than being in college or university. The only tradeoff to me is that you are required to be self descipline. In unversity, you have deadlines and assignments etc. But for online courses, they provide you the flexibility to learn whenever and wherever you want. So you might procastinate and forget about it soon or later.

# # Getting some insights on working in real world company
# ## ML skills required?

# In[ ]:


ax = multi_choice[multi_choice.Q5 == 'Data Scientist'].Q8.value_counts(normalize=True).plot(kind='barh')
plt.title('(Overall) Does your current employer incorporate machine learning methods into their business?')
ax.patches[1].set_facecolor('r')


plt.show()


# Do you really require ML skills to be able to work as a data scientist? Survey shows only 29% of the data scientist works in company that are uses well established ML methods. The rest are either still in infant stage of becoming a well establish AI company or not planning to implement.

# In[ ]:


ax = junior[junior.Q5 == 'Data Scientist'].Q8.value_counts(normalize=True).plot(kind='barh')
plt.title('(Junior) Does your current employer incorporate machine learning methods into their business?')
ax.patches[0].set_facecolor('r')
ax.patches[1].set_facecolor('r')
ax.patches[2].set_facecolor('grey')
ax.patches[3].set_facecolor('grey')
ax.patches[4].set_facecolor('grey')
ax.patches[5].set_facecolor('grey')


    

model = mpatches.Patch(color='red', label='model in production')
no_model = mpatches.Patch(color='grey', label='no model in production')
plt.legend(handles=[model, no_model])

plt.show()


# However, 59% of the **junior data scientist** are working in a company that have models in production. This shows that you are required to at least be familiar with ML methods in order to work as a data scientist. 
# 
# Personal Thoughts:
# You should at least be familiar with the basic of most of the ML methods (Linear Regression, Random Forest etc..) in order to become a data scientist. On the other hand, if you knew what the company is specialised in, you can just dive deep into that field (NLP, Computer Vision etc...)

# # Daily Job of A Data Scientist

# In[ ]:


job_scope = others.loc[others.Q9_OTHER_TEXT.notnull()].drop([0,1]).Q9_OTHER_TEXT.tolist()
join_job_scope = ' '.join(job_scope)

stopwords = set(STOPWORDS)
stopwords.update(["ML", "machine learning", "new", "end", "data science"])

wordcloud = WordCloud(stopwords=stopwords, max_font_size=100, max_words=100, background_color="black").generate(join_job_scope)

plt.figure(figsize=(15,15))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# We can't really split the job scope for junior and senior so we will inspect the overall job scope of being a DS. Based on the word cloud, most of these data scientist jobs are related to research, data and model. Data is the most common word that was found in everyone's answer because dealing with data is the main task for a data scientist. Task such as **data visualisation, data analysis, preparing dataset, cleaning data** etc... are what data scientist do most of the time. Research comes next is because data scientist job is more to experimenting model that are suitable for the company to deploy in production. 

# # Years of experience in writing code

# In[ ]:


ax = junior.Q15.value_counts().plot(kind='bar')
ax.patches[0].set_facecolor('r')
ax.patches[1].set_facecolor('r')
plt.xticks(rotation='50')
plt.title('years of writing code (Junior)')
plt.show()


# It's surprising that you don't even have to write code for more than 2 years to become a junior data scientist. 

# In[ ]:


ax = senior.Q15.drop(0).value_counts().plot(kind='bar')
ax.patches[0].set_facecolor('r')
ax.patches[1].set_facecolor('r')
plt.xticks(rotation='50')
plt.title('years of writing code (Senior)')
plt.show()


# However, to become a senior data scientist, experience in writing code is extreme important. One of the main reason is that people requires you to be efficient and effective. You can't just keep google on how to do the basic stuff and learn as you go. Basic analysis or plotting should be memorized as the first step to become a senior data scientist. 

# # Which media sources are popular among data scientist?

# In[ ]:


job_scope = others.loc[others.Q12_OTHER_TEXT.notnull()].drop([0,1]).Q12_OTHER_TEXT.tolist()
join_job_scope = ' '.join(job_scope)

stopwords = set(STOPWORDS)
stopwords.update(["site", "email", 'https'])


wordcloud = WordCloud(stopwords=stopwords, max_font_size=100, max_words=50, 
                      background_color="black").generate(join_job_scope)

plt.figure(figsize=(15,15))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# Being a data scientist requires constant update on hot topics or discussion at the moment. This survey provides us a clear idea where to look for these media sources. As you can see that LinkedIn is the most frequent source for data scientist to visit despite most people including me thought that LinkedIn is only for job searching/application platform. On the other hand, Medium is an obvious source for data science topics as I constantly find myself looking through medium to learn most of my data science skill sets. Besides tutorials provided, most of the social media have their own specific groups for data scientist to gather and communicate with each other to learn the hottest or latest topics. You can even ask questions or get help from there. 
# 
# *Don't be shy!*

# # Take your first step! (Tips on becoming a data scientist)

# ## 1. Where do you learn? Where can you start?

# In[ ]:


ax = others.loc[others.Q13_OTHER_TEXT.notnull()].Q13_OTHER_TEXT.value_counts().nlargest(7).plot(kind='barh')
plt.title('On which platforms have you begun or completed data science courses?')
ax.patches[0].set_facecolor('r')
plt.show()


# I'm still new to data science and I'm surprise to find that there are so many good resources out there. I usually go to youtube or medium to learn data science. Those who haven't start or still half way through, these platforms are well chosen by our fellow survey participants. This means they most probably provide a high quality content. After looking at this bar graph, I can say that the mlcourse.ai are one of the top platforms to learn data science. No wonder most people go for that!

# ## 2. Tools/Framework for data analysis

# In[ ]:


tool_scope = others.loc[others.Q14_OTHER_TEXT.notnull()].drop([0]).Q14_OTHER_TEXT.tolist()
join_tool_scope = ' '.join(tool_scope)
join_tool_scope = join_tool_scope.lower().replace('none', '')

stopwords = set(STOPWORDS)
stopwords.add('software')
wordcloud = WordCloud(stopwords=stopwords, max_font_size=100, max_words=50, 
                      background_color="black").generate(join_tool_scope)

plt.figure(figsize=(15,15))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# Based on the word cloud above, there's a mixture of programming languages and frameworks. Python, Matlab and SQL are the most common programming languages for data science. Jupyter or Jupyter Notebook is the place where you work on your code or analyse your data. I can explain all these frameworks one by one and so on but that would not provide you the exictment to explore your favourite framework. So my personal favourites are:
# 1. Python
# 2. Jupyter Notebook
# 3. Pandas
# 4. Matplotlib
# 5. Numpy

# ## 3. Which language should you choose?

# In[ ]:


multi_choice.Q19.value_counts(normalize=True).nlargest(3).plot(kind='bar')
plt.title('Recommended Programming Language for aspiring data scientist')
plt.show()


# As you can see from the two graph above, the main 3 programming language that **recommended by both junior data scientist and senior data scientist** are Python, R and SQL. In addition, these languages have various tutorial all over the internet including books. They are easy to pick up and there is plenty of forums and discussion on problems that you might get stuck while learning. So, don't be afraid to start. Helps are everywhere! As long as you pick one and start now, nothing is too late!

# ## 4. Core algorithm that most data scientist used

# In[ ]:


algo_scope = others.loc[others.Q24_OTHER_TEXT.notnull()].drop([0]).Q24_OTHER_TEXT.tolist()
join_algo_scope = ' '.join(algo_scope)
join_algo_scope = join_algo_scope.lower().replace('none', '')

stopwords = set(STOPWORDS)
stopwords.add('software')
wordcloud = WordCloud(stopwords=stopwords, max_font_size=100, max_words=100, 
                      background_color="black").generate(join_algo_scope)

plt.figure(figsize=(15,15))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# There could be tons of algorithm to learn and you might be overwhelmed. Based on the survey above, we can see that SVM (support vector machine), clustering algorithm (k-means) and regression are among the most popular algorithm. Perhaps you can start from here because these algorithms are proven to be generate effective results and easy for beginners.

# # Conclusion
# 
# I hope that based on the 4 tips above, you can take your first step to becoming a data scientist without any fear. Being a data science is not rocket science, you just need to put some effort to it. All the best mates!
# 
# This is my first analysis kernel and I hope you liked it. Thank you for your time and any suggestion would be much apperciated.
