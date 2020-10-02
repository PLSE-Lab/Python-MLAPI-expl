#!/usr/bin/env python
# coding: utf-8

# # In this notebook we are going to get some analytics out of the yelp_business data and do some NLP tasks on reviews.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS
from operator import itemgetter
import matplotlib as mpl
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(12,5)
yelp_business=pd.read_csv("../input/yelp_business.csv")


# # Lets find the top 40 Categories in Business

# In[ ]:


Categories={}

for x in yelp_business.categories:
    all_categories=x.split(";")
    for cat in all_categories:
        if cat not in Categories:
            Categories[cat]=1
        else:
            Categories[cat]+=1
All_categories=list(Categories.keys())
Cat_list=[[x,Categories[x]] for x in All_categories]

Cat_list=sorted(Cat_list, key=lambda x: x[1], reverse=True)
#LETS find the top 40 Categories of business
Cat_list=Cat_list[:40]
plt.bar(range(len(Cat_list)),[x[1] for x in Cat_list] ,align="center", color="bkmcgr")
plt.xticks(range(len(Cat_list)), [x[0] for x in Cat_list], rotation="vertical")
plt.show()


# ## As you can see, amongst the top40  categories for business, Top one is Restaurants followed by shopping and food
# .
# 
# .
# 
# .
# 
# ## Lets find out the top rated Category along with the least rates Category

# In[ ]:


Only_stars=[]
Categories_star={}
for i,x in yelp_business.iterrows():
    all_categories=x["categories"].split(";")
    Only_stars.append(int(round(x["stars"])))
    for cat in all_categories:
        if cat not in Categories_star:
            Categories_star[cat]=[]
        Categories_star[cat].append(x["stars"])
Star_list=[]
for x in list(Categories_star.keys()):
    Star_list.append([x, np.mean(Categories_star[x])])
    
Star_list=sorted(Star_list, key=lambda x: x[1], reverse=True)
Star_list=Star_list[:20] + Star_list[len(Star_list)-20:]

plt.bar(range(len(Cat_list)),[x[1] for x in Star_list] ,align="center",color="rgbkmc")
plt.xticks(range(len(Cat_list)), [x[0] for x in Star_list], rotation="vertical")
plt.show()
Only_stars=pd.DataFrame(Only_stars)
Only_stars.columns=["STARS"]


# 

# # Lets find out the count of stars for all businesses

# In[ ]:


Only_stars["STARS"].groupby(Only_stars["STARS"]).count().plot(kind="bar", sort_columns=True,color=[plt.cm.Paired(np.arange(len(Only_stars)))])


# # It is evident that Yelp have an abundance of 4 stars businesses folllowed by 2 stars.

# ## Now lets find the top 35 most common cities which business in the entries of yelp

# In[ ]:


yelp_business.city.groupby(yelp_business.city).count().sort_values()[::-1][:35].plot(kind="bar",color=[plt.cm.Paired(np.arange(len(yelp_business)))])


# ## Now lets find the most common words used in Naming the business

# In[ ]:


mpl.rcParams['font.size']=10
mpl.rcParams['figure.subplot.bottom']=.1 
word_string=" ".join(yelp_business["name"]).replace('"','').lower()
wordcloud = WordCloud(
                          background_color='black',
                          stopwords=set(STOPWORDS),
                          max_words=2500,
                          max_font_size=400, 
                          random_state=42
                         ).generate(word_string)
plt.imshow(wordcloud)
plt.axis('off')

plt.show()


# # Lets see the most common words used in Reviews. We will now find the most common words in
# ### 5 star reviews
# ### 4 star reviews
# ### 3 star reviews
# ### 2 star reviews
# ### 1 star reviews
# ### All reviews

# ## Since the Yelp_review data is too huge we need to process it in chunks and populate the dictionary with it, 
# ### We first read the data in chunk while populating the dictionary with the counts of words for 5 stars, 4 stars, 3 stars, 2 stars and 1 stars and finally for the whole review text
# 

# In[ ]:


from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import time
stop_words = set(stopwords.words('english'))
CountDictionary={}
for i in range(1,6):
    CountDictionary[i]={}
CountDictionary["all"]={}
def process(yelp_review):
    for i,x in yelp_review.iterrows():
        text=x["text"]
        tokenizer = RegexpTokenizer(r'\w+')
        text = tokenizer.tokenize(text.lower())
        text=[x for x in text if x not in stop_words]
        star=x["stars"]
        for val in text:
            if val in CountDictionary[star]:
                CountDictionary[star][val]+=1
            else:
                CountDictionary[star][val]=1
            if val in CountDictionary["all"]:
                CountDictionary["all"][val]+=1
            else:
                CountDictionary["all"][val]=1

chunksize = 10000
filename="../input/yelp_review.csv"
count=1
beg_ts = time.time()
avg_time_chunk=[]
for chunk in pd.read_csv(filename, chunksize=chunksize):
    ch_start=time.time()
    process(chunk)
    ch_end=time.time()
    avg_time_chunk.append(ch_end-ch_start)
    count+=1
end_ts=time.time()
print ("Total time taken to read 3.53 GB file is " + str(end_ts - beg_ts))
print ("Average time for processing one chunk of"+ str(count) + "chunks is "+ str( np.mean(avg_time_chunk)))
print ("Sucessfully Populated Dictionary")


# ### We saw the analyzing a 3.53 gb file takes a lot of time hence be broke it into chunks

# In[ ]:


def CreateCloud(attr):
    
    wordcloud = WordCloud(background_color='black',
                              stopwords=set(STOPWORDS),
                              random_state=42).generate_from_frequencies(frequencies=CountDictionary[attr])
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    


# ## WordCloud for all review irrespective of the stars given

# In[ ]:


CreateCloud("all")


# # Lets see the word cloud for 1 star to 5 stars

# In[ ]:


for i in range(1,6):
    print (("_")*90)
    print ("WORD CLOUD FOR "+ str(i)+" stars")
    CreateCloud(i)
    


# ## Now lets find out the most common day for a checkin

# In[ ]:


yelp_checkin=pd.read_csv("../input/yelp_checkin.csv")
yelp_checkin.weekday.groupby(yelp_checkin.weekday).count().sort_values()[::-1].plot(kind="bar",color=[plt.cm.Paired(np.arange(len(yelp_checkin)))])


# ## as expected its Saturday
# 
# 
# # Now lets find out the Business with most number of checkins on yelp
# ## The answer is pretty obvious though 

# In[ ]:


yelp_checkin=yelp_checkin.sort_values(by="checkins", ascending=False)[:15]
s1 = pd.merge(yelp_business, yelp_checkin, how='inner', on=['business_id'])
s1=s1.sort_values(by="checkins", ascending=False)
plt.style.use('ggplot')
ax = s1[['checkins']].plot(kind='bar',figsize=(15,10),legend=True, fontsize=12, color=[plt.cm.Paired(np.arange(len(yelp_checkin)))])
ax.set_xticklabels(s1.name, rotation=90)
ax.set_xlabel("Business Names",fontsize=12)
ax.set_ylabel("Checkins",fontsize=12)
plt.show()


# ## Airports Obviously will have most number of checkins, Morevoer "Kungfu tea " seems to be the most popular after airports.

# # To be Continued
# # Do give your suggestions and upvote if you find it useful
