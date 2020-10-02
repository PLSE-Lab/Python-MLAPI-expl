#!/usr/bin/env python
# coding: utf-8

# # Importing the needed libraries 

# In[ ]:


import numpy as np
import pandas as pd 
import re
import emoji 
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image


# In[ ]:


##reading in the already data
df=pd.read_csv(r"C:\Users\User\Downloads\Discussing Football.csv",index_col=None)


# In[ ]:


##checking the data again, just to be sure 
df


# In[ ]:


#converting date and time to timsstamp
df['DateTime']=pd.to_datetime(df['Date']+ ' '+ df["Time"], dayfirst=True)


# In[ ]:


#indexing with the timestamp
df.index=df['DateTime']


# <h3> Checking the activity of members in the group </h3> 
# <p1> Like I said earlier, I'm not a football guy so you'll find me among the least 10 active members of the group. As you would see, my name is the 7th from the end. </p1>

# In[ ]:


member_activity=df.Sender.value_counts()
plt.figure(figsize=[15,5])
plt.xlabel('Group Members')
plt.ylabel('No. of Messages')
member_activity.plot.bar(color={'r', 'b','g','y','b'},figsize=[15,5])
plt.xticks(rotation=50)
plt.title('Members of the Group and their Relative Activity')
plt.show()


# In[ ]:


df["Sender"].value_counts().head(10).plot.bar(color={'r', 'b','g','y','b'},figsize=[15,5])
plt.title('Top 10 Most Active Members of the Group')
plt.xlabel('No. of messages')
plt.ylabel('Senders')
plt.show()


# In[ ]:


df["Sender"].value_counts().tail(10).plot.bar(color={'r', 'b','g','y','b'},figsize=[15,5])
plt.title(' 10 Least Active Members of the Group')
plt.xlabel('No. of messages')
plt.ylabel('Senders')
plt.show()


# # Comparing Activity across the hours of the day 
# from here you'll see that between the 6pm and 10pm are the best times to message the group between 1 am and 5am are the worst times to message the group(obviosuly, everyone would be asleep). which begs the question, who are those that message at night?

# In[ ]:


#Top 5 Active Hours of the Day
plt.figure(figsize=[15,5])
plt.xlabel('Hours of the day')
plt.ylabel('No. of media messages')
df.Hour.value_counts().head().sort_values().plot.bar(color={'r', 'b','g','y','b'})
plt.xticks(rotation=50)
plt.title('Top 5 Active Hours of the Day')
plt.show()


# In[ ]:


#5 Least Active Hours of the Day
plt.figure(figsize=[15,5])
plt.xlabel('Hours of the day')
plt.ylabel('No. of media messages')
df.Hour.value_counts().tail().sort_values(ascending=False).plot.bar(color={'r', 'b','g','y','b'})
plt.xticks(rotation=50)
plt.title('Least 5 Active Hours of the Day')
plt.show()


# # Who are those that message at night?

# In[ ]:


nocturnals=df[df['Hour']<6]
nocturnals.Sender.value_counts().head(10).plot.bar(color={'r', 'b','g','y','c'}, figsize=[15,5])
plt.xlabel('Names')
plt.ylabel('No. of Messages')
plt.title('Messengers at night')
plt.show()


# # Checking the ativity for the dates 
# Interstingly, the dates with the highest discussions seemed to be 
# 24th of February, that was a day after Deontay Wilder and Tyson Fury had a fight, also 
# Manchester United and Arsenal had won their games a night before. Liverpool had a game that night as well and won(the most active person in the group is a Liverpool fan). So there was a lot to discuss that day. None the less I'll check the data from that day to see what it looks like.
# 
# 25th of December 2019 is obviously christmas day, kind of explains itself.

# In[ ]:


top_10_active_days_of_the_group = df['Date'].value_counts().head(10).plot.bar(color={'r', 'b','g','y','b'},figsize=[15,5])
plt.title('top 10 active days of the group')
plt.xlabel('no. of messages')
plt.ylabel('dates')
plt.show()


# In[ ]:


least_10_active_days_of_the_group = df['Date'].value_counts().tail(10).plot.bar(color={'r', 'b','g','y','b'},figsize=[15,5] )


# <p> Turns out only one of my guesses was correst about the day with the highest  activity and it seems the match was the real reason for the spike. This I know cos the discussions spiked from 8pm when the match was supposed to start </p>

# In[ ]:


DHM=df[df['Date']=='24/02/2020']
DHM.Hour.value_counts().plot.bar(color={'r', 'b','g','y','b'},figsize=[15,5])
plt.xlabel('Hours of the Day')
plt.ylabel('No of messages')
plt.show()


# <P> To be sure, let's check those messaging on that day, since Korede is the only Liverpool fan on the group, it would only make sense that his messages are more on that day... 
#     lets see </p>

# In[ ]:


DHM=df[df['Date']=='24/02/2020']
DHM.Sender.value_counts().plot.bar(color={'r', 'b','g','y','b'},figsize=[15,5])
plt.xlabel('Names')
plt.ylabel('No of messages')
plt.show()


# # Inspecting the Messages 

# In[ ]:


#nUMBER OF MEDIA MESSAGES 
media=df[df["Message"]=='<Media omitted>']
len(media)


# In[ ]:


plt.figure(figsize=[15,5])
plt.xlabel('Group Members')
plt.ylabel('No. of media messages')
media.Sender.value_counts().head().plot.bar(color={'r', 'b','g','y','b'})
plt.xticks(rotation=50)
plt.title('Top 5 media senders')
plt.show()


# So what you are seeing is a bar graph showing the longest messages sent and the number of words in those sentences. If you would notice, a sender(okiki) has a mark, that's because he sent two very long sentences, so the mean was taken for the graph. The line on his bar shows the values for his two long sentences. 

# In[ ]:


#Top 5 longest sentences and their senders 
long_sents=df.sort_values(by=['Word_Count'],ascending=False)[1:7]
long_sents=long_sents[['Sender', "Word_Count"]]
plt.figure(figsize=[15,5])
sns.barplot( x="Sender",
    y="Word_Count",
    hue="Sender",
    data=long_sents)
plt.title('Longest Sentences Ever and their Senders')
plt.show()


# In[ ]:


word_count=df[['Sender', 'Letter_Count','Word_Count','Avg_Word_length']]
word_count.index=word_count['Sender']


# In[ ]:


#To check the those instances where group members sent sentences with long words 
word_count.sort_values(by=['Letter_Count'],ascending=False).head(5).plot.bar(figsize=[15,5])
plt.title('Comparing number of letters, word count and Average Word Length')
plt.xlabel('Names')
plt.ylabel('No of messages')
plt.show()


# In[ ]:


#However, let us check those who had very lenghty words... 
word_count2=df[['Sender', 'Letter_Count','Word_Count','Avg_Word_length']]
word_count2.groupby(df["Sender"]).sum().sort_values(by=['Letter_Count'],ascending=False).head(5).plot.bar(figsize=[15,5])
plt.title('Comparing number of letters, word count and Average Word Length')
plt.xlabel('Names')
plt.show()


# The values above are very suggestive as the names appearing are those that are very active. So let's check on the average, who is used to sending very long sentnces long words.
# Interestingly, some members of the group that are non active appears on the graph. Even though they do not talk, whenever they do, they make a mark in the group!

# In[ ]:


word_count2.groupby(df["Sender"]).mean().sort_values(by=['Word_Count'],ascending=False).head().plot.bar(figsize=[15,5])
plt.title('Comparing number of letters, word count and Average Word Length')
plt.xlabel('Names')
plt.show()


# # Comparing year 2019 and 2020

# In[ ]:


plt.figure(figsize=[15,5])
plt.xlabel('Year')
plt.ylabel('No. of Messages')
df.index.year.value_counts().plot.bar(color={'r', 'b'})
plt.title('Number of Messages Sent in 2019 and 2020')
plt.show()


# In[ ]:


#spliting the dataset into 2019 and 2020 groups 
dateG2019=df[df.index.year==2019]
dateG2020=df[df.index.year==2020]

##checking if the split is correct, just to be sure 
len(dateG2019)+len(dateG2020)==len(df)


# In[ ]:


dateG2020.describe()


# In[ ]:


dateG2019.describe()


# In[ ]:


dateG2019['Date'].value_counts().head(10).plot.bar(color={'r', 'b','g','y','b'},figsize=[15,5])
plt.title('10 days of highest activity in 2019')
plt.xlabel('No. of messages')
plt.ylabel('Dates')
plt.show()


# In[ ]:


dateG2019['Date'].value_counts().tail(10).plot.bar(color={'r', 'b','g','y','b'},figsize=[15,5])
plt.title('10 days of least activity in 2019')
plt.xlabel('No. of messages')
plt.ylabel('Dates')
plt.show()


# In[ ]:


dateG2020['Date'].value_counts().head(10).plot.bar(color={'r', 'b','g','y','b'},figsize=[15,5])
plt.title('10 days of highest activity in 2019')
plt.xlabel('No. of messages')
plt.ylabel('Dates')
plt.show()


# In[ ]:


dateG2020['Date'].value_counts().tail(10).plot.bar(color={'r', 'b','g','y','b'},figsize=[15,5])
plt.title('10 days of least activity in 2019')
plt.xlabel('No. of messages')
plt.ylabel('Dates')

plt.show()


# In[ ]:


print('Date with the lowest activity in 2019: {}' .format(dateG2019['Date'].value_counts().tail(1)))
print('Date with the highest activity in 2019: {}' .format(dateG2019['Date'].value_counts().head(1)))


# In[ ]:


print('Date with the lowest activity in 2020: {}' .format(dateG2020['Date'].value_counts().tail(1)))
print('Date with the highest activity in 2020: {}' .format(dateG2020['Date'].value_counts().head(1)))


# In[ ]:


dateG2019.groupby(dateG2019.index.month)["Message"].count().plot.line(color={'b'},figsize=[15,5])
dateG2020.groupby(dateG2020.index.month)["Message"].count().plot.line(color={'r'})
plt.title('Line Graph Cmparing The Activity In the Group for 2019 and 2020')
plt.xlabel('Number of Messages')
plt.ylabel('Months of the year')
plt.show()
###2019 is in blue 


# In[ ]:


dateG2019.groupby(dateG2019.index.month).sum().plot.line()
dateG2020.groupby(dateG2020.index.month).sum().plot.line()
plt.title('Line Graph Cmparing The Activity In the Group for 2019 and 2020')
plt.xlabel('no. of messages')
plt.ylabel('Months of the year')
plt.show()


# In[ ]:


df


# In[ ]:


def gen_text(col):
    col=col.dropna()
    txt=" ". join(message for message in col)
    txt=re.sub('..... omitted', '', txt)
    return txt


# In[ ]:


text2019=gen_text(dateG2019['Message'])
text2020=gen_text(dateG2020['Message'])


# In[ ]:


stopwords = set(STOPWORDS)
wordcloud19=WordCloud(max_font_size=50, max_words=100, background_color="white",stopwords=stopwords).generate(text=text2019)
wordcloud20=WordCloud(max_font_size=50, max_words=100, background_color="white",stopwords=stopwords).generate(text=text2020)


# # <h2> In summary, lets see the common words used in the group.</h2>
# I'm not really a football guy but then it's easy to see why Liverpool and arsenal are very prominent words in the group

# In[ ]:


plt.figure(figsize=[15,5])
plt.imshow(wordcloud19, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[ ]:


plt.figure(figsize=[15,5])
plt.imshow(wordcloud20, interpolation='bilinear')
plt.axis("off")
plt.show()


# # If you got through to this stage, I can only but say thank you for reading through my boring sermon. I'm still much learning so I welcome suggestions comments and critics.  

# In[ ]:




