#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


get_ipython().system('pip install ndjson')


# In[ ]:


import ndjson

# load from file-like objects
with open('/kaggle/input/celebrityprofiling/training-dataset/celebrity-feeds.ndjson') as f:
    data1 = ndjson.load(f)
    
with open('/kaggle/input/celebrityprofiling/supplement-dataset/celebrity-feeds.ndjson') as f:
    data2 = ndjson.load(f)


# In[ ]:


tweets=data1+data2


# In[ ]:


# load from file-like objects
with open('/kaggle/input/celebrityprofiling/supplement-dataset/labels.ndjson') as f:
    data1 = ndjson.load(f)
    
# load from file-like objects
with open('/kaggle/input/celebrityprofiling/training-dataset/labels.ndjson') as f:
    data2 = ndjson.load(f)    


# In[ ]:


labels=data2+data1


# In[ ]:


len(labels)


# In[ ]:


#Number of Tweets
print("Number of Celebrities:     "+str(len(tweets)))


# In[ ]:


tweet=[]
birthyear=[]
gender=[]
occupation=[]
average_tweet=[]
number_words=[]
ids=[]
for i in range(len(tweets)):
    for j in range(len(tweets[i]["text"])):
        tweet.append(tweets[i]["text"][j])
        birthyear.append(labels[i]['birthyear'])
        gender.append(labels[i]['gender'])
        occupation.append(labels[i]['occupation'])
        ids.append(labels[i]['id'])

    average_tweet.append(len(tweets[i]["text"]))
    for k in range(len(tweets[i]["text"])):
        number_words.append(len(tweets[i]["text"][k].split()))


# In[ ]:


total_tweet=sum(average_tweet)


# In[ ]:


celebrities=len(average_tweet)


# In[ ]:


print("Number of Celebrities:     "+str(len(tweets)))
print("Number Of Tweets/Celebrity:     "+str(total_tweet/celebrities))
print("Total Number Of Tweets:     "+str(total_tweet))
print("Total Number Of Words/Tweet:     "+str(sum(number_words)/total_tweet))


# In[ ]:


male=0
female=0

for k in range(len(labels)):
    if labels[k]['gender']=="male":
        male=male+1
        
    if labels[k]['gender']=="female":
        female=female+1
        


# In[ ]:


print("Number of Celebrities{male}:     "+str(male)+"  |  Percentage of Celebrities{male}:       "+str(male/(male+female))+"%")
print("Number of Celebrities{female}:     "+str(female)+"  |   Percentage of Celebrities{female}:     "+str(female/(male+female))+"%")


# In[ ]:


birthyear_list=[]

for k in range(len(labels)):
    birthyear_list.append(labels[k]['birthyear'])
        


# In[ ]:


average_age=0
this_year=2020
for k in range(len(birthyear_list)):
    average_age=average_age+this_year-int(birthyear_list[k])
        
average_age=average_age/len(birthyear_list)


# In[ ]:


print("Average Age Of Celebrities:     "+str(average_age))


# In[ ]:


set(occupation)


# In[ ]:


creator=0
performer=0
politics=0
sports=0

for k in range(len(labels)):
    if labels[k]['occupation']=="creator":
        creator=creator+1
        
    if labels[k]['occupation']=="performer":
        performer=performer+1

                
    if labels[k]['occupation']=="politics":
        politics=politics+1
        
                
    if labels[k]['occupation']=="sports":
        sports=sports+1
                


# In[ ]:


print("Number of creator in the dataset:     "+str(creator))
print("Number of performer in the dataset:     "+str(performer))
print("Number of politics in the dataset:    "+str(politics))
print("Number of sports in the dataset:     "+str(sports))


# In[ ]:


from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import pandas as pd 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


celebrity = {
    'id':ids,
    'text': tweet,
        'birthyear': birthyear,
         'gender':gender,
           'occupation':occupation
        }


df = pd.DataFrame(celebrity)


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.to_csv("celebrity_profiling.csv")


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


del data1
del data2


# In[ ]:


def words_without_stops(df1):
    comment_words = '' 
    stopwords = set(STOPWORDS) 

    # iterate through the csv file 
    for val in df1.text: 

        # typecaste each val to string 
        val = str(val) 

        # split the value 
        tokens = val.split() 

        # Converts each token into lowercase 
        for i in range(len(tokens)): 
            tokens[i] = tokens[i].lower() 

        comment_words += " ".join(tokens)+" "
        
    return comment_words


# In[ ]:


occupation_list=list(set(occupation))


# In[ ]:


df1=df.sample(n=1000)


# In[ ]:


comment_words=words_without_stops(df1)


# In[ ]:


stopwords = set(STOPWORDS)

wordcloud = WordCloud(width = 800, height = 800, 
            background_color ='white', 
            stopwords = stopwords, 
            min_font_size = 10).generate(comment_words) 



# In[ ]:



# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.title("creator") 
  
plt.show() 


# In[ ]:


df1=df[df['occupation']=="creator"]
df1=df1.sample(n=1000)


# In[ ]:


comment_words=words_without_stops(df1)


# In[ ]:



wordcloud = WordCloud(width = 800, height = 800, 
            background_color ='white', 
            stopwords = stopwords, 
            min_font_size = 10).generate(comment_words) 



# In[ ]:



# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.title("creator") 
  
plt.show() 


# In[ ]:


df1=df[df['occupation']=="performer"]
df1=df1.sample(n=1000)


# In[ ]:


comment_words=words_without_stops(df1)


# In[ ]:


wordcloud = WordCloud(width = 800, height = 800, 
            background_color ='white', 
            stopwords = stopwords, 
            min_font_size = 10).generate(comment_words) 



# In[ ]:



# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.title("performer") 
  
plt.show() 


# In[ ]:


df1=df[df['occupation']=="politics"]
df1=df1.sample(n=1000)


# In[ ]:


comment_words=words_without_stops(df1)


# In[ ]:


wordcloud = WordCloud(width = 800, height = 800, 
            background_color ='white', 
            stopwords = stopwords, 
            min_font_size = 10).generate(comment_words) 



# In[ ]:



# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.title("politics") 
  
plt.show() 


# In[ ]:


df1=df[df['occupation']=="sports"]
df1=df1.sample(n=1000)


# In[ ]:


comment_words=words_without_stops(df1)


# In[ ]:


wordcloud = WordCloud(width = 800, height = 800, 
            background_color ='white', 
            stopwords = stopwords, 
            min_font_size = 10).generate(comment_words) 



# In[ ]:



# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.title("sports") 
  
plt.show() 


# In[ ]:


df1=df[df['gender']=="male"]
df1=df1.sample(n=1000)


# In[ ]:


comment_words=words_without_stops(df1)


# In[ ]:


wordcloud = WordCloud(width = 800, height = 800, 
            background_color ='white', 
            stopwords = stopwords, 
            min_font_size = 10).generate(comment_words) 



# In[ ]:



# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.title("sports") 
  
plt.show() 


# In[ ]:


df1=df[df['gender']=="male"]
df1=df1.sample(n=1000)


# In[ ]:


comment_words=words_without_stops(df1)


# In[ ]:


wordcloud = WordCloud(width = 800, height = 800, 
            background_color ='white', 
            stopwords = stopwords, 
            min_font_size = 10).generate(comment_words) 



# In[ ]:



# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.title("male") 
  
plt.show() 


# In[ ]:


df1=df[df['gender']=="female"]
df1=df1.sample(n=1000)


# In[ ]:


comment_words=words_without_stops(df1)


# In[ ]:


wordcloud = WordCloud(width = 800, height = 800, 
            background_color ='white', 
            stopwords = stopwords, 
            min_font_size = 10).generate(comment_words) 



# In[ ]:



# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.title("male") 
  
plt.show() 

