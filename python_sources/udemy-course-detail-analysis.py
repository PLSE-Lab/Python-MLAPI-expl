#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# In[ ]:


df=pd.read_csv("../input/udemy-courses/udemy_courses.csv")


# In[ ]:


df.head()


# In[ ]:


df.columns


# In[ ]:


df.shape


# In[ ]:


print("Percentage of paid courses is: ",df[df["is_paid"] == True].shape[0]*100/df.shape[0])
print("Percentage of free courses is: ",100 - df[df["is_paid"] == True].shape[0]*100/df.shape[0])


# In[ ]:


df.subject.unique()


# As we can see there are 4 domains of courses listed in the dataset. They are
#     1. Business Finance', 
#     2. 'Graphic Design', 
#     3. 'Musical Instruments',
#     4. 'Web Development

# In[ ]:


df.level.unique()


# The courses have been divided into 4 levels.
# 1. All Levels
# 2. Intermediate Level
# 3. beginner level
# 4. expert level
# 

# In[ ]:


sns.catplot(x="level", col="is_paid",
                data=df, kind="count",
                height=8, aspect=.9);


# Its pretty evident that most of the free courses are aimed at beginners. most of the instructors make paid courses which are suitable for all levels.
# 

# In[ ]:


fig_dims = (9, 6)
fig, ax = plt.subplots(figsize=fig_dims)
sns.barplot(x="level",y="num_subscribers",data=df,ax=ax,palette="cubehelix")


# most of the people prefer beginner level courses or courses which are suited for all leveles.

# In[ ]:


df["price"]=df.price.replace("Free",0)
df["price"]=df.price.replace("TRUE","178")


# for analysis prices of free courses have b een replaced as 0 .also there is a course whose price is quoted as true. its real price is Rs.12480. assuming the prices are in dollars i have replaced it as 178

# In[ ]:


fig_dims = (6,9)
fig, ax = plt.subplots(figsize=fig_dims)
df['price'] = df['price'].astype(int)
plt.hist(df["price"])


# most of the courses are priced between 25 and 50 dollars excluding discounts.

# In[ ]:


fig_dims = (15, 6)
fig, ax = plt.subplots(figsize=fig_dims)
sns.barplot(x="subject",y="num_subscribers",data=df,ax=ax,palette="cubehelix",estimator=np.mean,hue=df.level)


# Maximum people prefer to opt for Web development courses . the trend is same for all the levels.

# In[ ]:


sns.catplot(x="subject",y="price",hue="level",data=df,kind="box",height=9, aspect=2.3)


# Most of the beginner level courses are overpriced except the web development course. the web development courses look well priced as there aren't any outliers in it for all the levels except the expert level. This can be one of the reason why web development courses have much more subscribers than the others.

# **Lets see which are the most used topics by the course creators in each subject. This will help a new course creator to know in which topic he can create a course to attract the audience.**

# In[ ]:


text_webdevelopment = " ".join(review for review in df[df["subject"] =="Web Development"]["course_title"])
wordcloud = WordCloud(max_font_size=50, max_words=100,background_color="black").generate(text_webdevelopment)
plt.figure(figsize = (10, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# For web development courses the most used words are javascript ,website, beginner ,wordpress. Javascript looks the most in demand course.

# In[ ]:


text_busfin = " ".join(review for review in df[df["subject"] =="Business Finance"]["course_title"])
wordcloud = WordCloud(max_font_size=50, max_words=100,background_color="black").generate(text_busfin)
plt.figure(figsize = (10, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# Most used words in the business finance domain are Trading, Accounting,Forex,stock and options.Trading looks the most prominent word.

# In[ ]:


text_graphdes = " ".join(review for review in df[df["subject"] =="Graphic Design"]["course_title"])
wordcloud = WordCloud(max_font_size=50, max_words=100,background_color="black").generate(text_graphdes)
plt.figure(figsize = (10, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# Most used words in graphic design domain are Design, Photoshop and illustrator. Design is the most prominent word 

# In[ ]:


text_musinstru = " ".join(review for review in df[df["subject"] =="Musical Instruments"]["course_title"])
wordcloud = WordCloud(max_font_size=50, max_words=100,background_color="black").generate(text_musinstru)
plt.figure(figsize = (10, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# Most used words in Musical Instruments domain are Piano and Guitar. both these words look prominent.

# **Thank You . more analysis coming later. pls do help me improve if you think something went wrong.this is my first data science kaggle project**.
