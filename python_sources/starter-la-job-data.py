#!/usr/bin/env python
# coding: utf-8

# # Objective
# I will try to visualize the data of job bulletins in Los Angeles.
# 
# Thanks to the Shahules786's [Kernel](https://www.kaggle.com/shahules/discovering-opportunities-at-la) and I refer to the preprocessing in this kernel.

# # Importing Libraries

# In[ ]:


import numpy as np
import pandas as pd
import re
import os
import numpy as np
from datetime import datetime
from collections  import Counter
from nltk import word_tokenize
import seaborn as sns
import matplotlib.pyplot as plt
import calendar
from wordcloud import WordCloud ,STOPWORDS
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import word2vec
import gensim
import matplotlib.colors as mcolors
plt.style.use('ggplot')


# # Getting all the csv files

# In[ ]:


csvfiles=[]
additional=os.listdir("../input/cityofla/CityofLA/Additional data/")
for file in additional:
    if file.endswith('.csv'):
        print(file)
        csvfiles.append("../input/cityofla/CityofLA/Additional data/"+file)

sample_job=pd.read_csv(csvfiles[0])
kaggle_data=pd.read_csv(csvfiles[1])
job_title=pd.read_csv(csvfiles[2])


# # Convert all the files in Job Bulletins directory to DataFrame

# In[ ]:


def to_dataframe(num,df):

    opendate=re.compile(r'(Open [D,d]ate:)(\s+)(\d\d-\d\d-\d\d)')
    salary=re.compile(r'\$(\d+,\d+)((\s(to|and)\s)(\$\d+,\d+))?')
    requirements=re.compile(r'(REQUIREMENTS?/\s?MINIMUM QUALIFICATIONS?)(.*)(PROCESS NOTE)')
    
    for no in range(0,num):
        with open("../input/cityofla/CityofLA/Job Bulletins/"+bulletins[no],encoding="ISO-8859-1") as f:
                try:
                    file=f.read().replace('\t','')
                    data=file.replace('\n','')
                    headings=[heading for heading in file.split('\n') if heading.isupper()]     
                    sal=re.search(salary,data)
                    try:
                        req=re.search(requirements,data).group(2)
                    except Exception as e:
                        req=re.search('(.*)NOTES?',re.findall(r'(REQUIREMENTS?)(.*)(NOTES?)',
                                                              data)[0][1][:1200]).group(1)
                    
                    
                    df=df.append({'File Name':bulletins[no],'Position':headings[0].lower(),'salary_start':sal.group(1),"requirements":req},ignore_index=True)
                    
                except Exception as e:
                    print('umatched')  
    return df
bulletins=os.listdir("../input/cityofla/CityofLA/Job Bulletins/")
df=pd.DataFrame(columns=['File Name','Position','salary_start'])
df=to_dataframe(len(bulletins),df)


# Confirming new dataframe.

# In[ ]:


df.head()


# # What are the words that frequently appear in job offers?

# In[ ]:


plt.figure(figsize=(10,5))
text=' '.join(job for job in df['Position'])
text=word_tokenize(text)
jobs=Counter(text)
jobs_class=[job for job in jobs.most_common(20) if len(job[0])>3]
a,b=map(list, zip(*jobs_class))
sns.barplot(b,a,palette='deep')                                           
plt.title('Words that frequently appear in job offers')
plt.xlabel("number of occurrences")
plt.ylabel('job offers')


# # Top 5 and Bottom 5 job Types of Annual Salary in Job Bulletins in the City of Los Angeles

# In[ ]:


df['salary_start']=[int(sal.split(',')[0]+sal.split(',')[1] ) for sal in df['salary_start']]   


# In[ ]:


worst = df[['Position','salary_start']].sort_values(by='salary_start',ascending=True)[:5]
best = df[['Position','salary_start']].sort_values(by='salary_start',ascending=False)[:5]
worst_best = pd.concat([best,worst]).sort_values(by='salary_start',ascending=False)
plt.figure(figsize=(8,5))
sns.barplot(y=worst_best['Position'], x=worst_best['salary_start'], palette='deep')
plt.title('Top 5 and Bottom 5 job Types of Annual Salary')
plt.xlabel('Annual Salary')
plt.ylabel('job offers')


# # What are the average annual salary jobs in the city of Los Angeles?

# In[ ]:


average_salary = df[(df['salary_start'] >79900) & (df['salary_start'] < 81100)].sort_values(by='salary_start',ascending=False)
sns.barplot(x=average_salary['salary_start'],y=average_salary['Position'],palette='deep')
plt.title("tAverage annual salary jobs (80,000$) in the city of Los Angeles?",loc='right')
plt.xlabel('Annual salary')
plt.ylabel('job offers')


# # Classification of Annual Salary in job bulletins

# In[ ]:


a = pd.cut(df['salary_start'],bins=[-1,49999,99999,149999,300000],labels=['<50,000$','>=50,000$, <100,000$','>=100,000$, <150,000$','>150,000$']).value_counts()
plt.figure(figsize=(5,5))
plt.rcParams['font.size'] = 14.0
plt.pie(a,counterclock=False, startangle=90,        wedgeprops={'linewidth': 2, 'edgecolor':"white"},autopct="%1.1f%%",pctdistance=0.7)
plt.legend(a.index,fancybox=True,loc='upper left',bbox_to_anchor=(0.5,0.35))
plt.title("Classification of Annual Salary in job bulletins")


# # What are the words that have relation to "Experience"?

# In[ ]:


def build_corpus(df):

    lem=WordNetLemmatizer()
    corpus= []
    for x in df:

        words=word_tokenize(x)
        corpus.append([lem.lemmatize(w.lower()) for w in words if len(w)>3])
    return corpus

a = df['Position'] + " " + df['requirements']
corpus=build_corpus(a)
model = word2vec.Word2Vec(corpus)


# In[ ]:


from sklearn import preprocessing

vector = model.wv["experience"]
word = model.most_similar( [ vector ], [], 20)
rank = [data[1] for data in word]
name = [data[0] for data in word]
comp = name + rank


# In[ ]:


def show_wordcloud(data, title = None):
    
    wordcloud = WordCloud(
        background_color='white',
        stopwords=set(STOPWORDS),
        max_words=250,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()
show_wordcloud(comp,'REQUIREMENTS')


# More to come. Stay tuned!
