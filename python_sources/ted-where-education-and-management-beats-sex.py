#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# Getting started:
# * I use this analysis on the TED Database as an excercise for the Udemy course "Python for Data Science and Machine Learning" (https://www.udemy.com/python-for-data-science-and-machine-learning-bootcamp/learn/v4/overview)
# * Therefore, please be forewarned and show mercy with my early beginnings into Data Analysis
# * I would be very thankful for any hints and tipps you may have for me to improve
#  ![](https://binvested.com.au/wp-content/uploads/getting-started-in-property-investment.jpg)
# 
# * About TED-Talks:
#     *![](http://www.instantoffices.com/blog/wp-content/uploads/2016/08/TED-Talks.jpg)
#     * https://en.wikipedia.org/wiki/TED_(conference)
#     * TED (Technology, Entertainment, Design) is a media organization which posts talks online for free distribution, under the slogan "ideas worth spreading". TED was founded in February 1984 as a conference, which has been held annually since 1990. TED's early emphasis was technology and design, consistent with its Silicon Valley origins, but it has since broadened its focus to include talks on many scientific, cultural, and academic topics. The main TED conference is held annually.   

# In[ ]:


ted=pd.read_csv("../input/ted_main.csv")


# In[ ]:


transcripts=pd.read_csv("../input/transcripts.csv")


# In[ ]:


ted.info()


# * Overall there are 2,550 TED talks in the database. 
# * Only deviation is 6 people that did not disclose their occupation. Let's check them out.

# In[ ]:


ted[ted["speaker_occupation"].isnull()]


# Nothing specifically interesting.
# 
# Let's get a first idea of the data:

# In[ ]:


ted.describe()


# **Observations:**
# * Comments and views widely spread. Standard deviation larger than mean.
# * I was suprised to see the spread of average duration. To my knowledge I thought that TED presentations are highly standardized in format. (Nice reading on giving a TED talk here: https://waitbutwhy.com/2016/03/my-ted-talk.html)
# ![](http://waitbutwhy.com/wp-content/uploads/2016/03/TED-Vid-FB.jpg)
# 
# Now let's look at some examples to get a better understanding of the available data:

# In[ ]:


ted.head()


# In[ ]:


ted.corr()


# **Observations**
# * Not really worth mentioning the nearly perfect correlation between film_date and published_date
# * Biggest correlation between views and comments. Not surprising.
# * Positive correlation between languages and comments/views. Also not surprising that most popular content is being most translated
# * Interestingly views and published dates are not correlated, meaning that popularity much more depends on topics than time. Probably talks are only being viewed a certain amount of after release, e.g. in the first year.
# * Slightly negative correlation between duration with both film_date and languages. 
#     * Film_date suggesting that over time the talks got shorter (consistent with later finding that # of talks per yearly main event increases).
#     * Languages implies that shorter talks are more likely to be translated. 
# 
# 
# So what are the most viewed talks?

# In[ ]:


ted.sort_values("views",ascending=False).head(5)


# * Impressive that talks #1 and 2 have nearly double the number of views from #4 or 5. The typical "scaling" possibility of online media. Surely there will be a very "long tail" of talks with comparably little views. 
# * Surprising that only 1 of 5 is about something sexual. And that business leadership and education scores higher.
# * I personally knew talks no 2 and 3. No. 1, 4, 5 now on my longlist for things to view.
# 
# So, what then are the least viewed talks?

# In[ ]:


ted.sort_values("views",ascending=True).head(5)


# Without analysing in detail:
# * The titles are really not strong. What content is to be expected behind "Kounandi"? or "The early birdwatchers"? At least to me its not at all clear why I should watch this talk and what I should expect to see or learn. Compare with titles from the top 5: "Do schools kill creativity?" or "How great leaders inspire action". Typical thing a journalist learns first thing: Write a convincing headline. 
# ![](https://cdn3.img.sputniknews.com/images/104130/49/1041304974.jpg)
# * This led me to rename this kernel from "exploring-the-ted-dataset" to "TED - Where education and management beats sex". Excited to see the effect :-)
# * The speakers are (at least to me) unknown
# * The events sound like they are small ones

# In[ ]:


ted.event.value_counts().head(10)


# In[ ]:


ted.event.value_counts().tail(10)


# Most talks are from the official TED talkevents with increasing # of talks per year 

# In[ ]:


ted.speaker_occupation.value_counts().head(10)


# Most ted speakers are freelancing and  creative occupations. Kind of not real life. teachers, office people,  medical stuff, ...

# Now lets look more detailled via graphs at content, categories, longtail, descriptions, ratings etc.

# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


fig=plt.figure()
axes=fig.add_axes([0,0,1,1])
axes.set_xlabel("views")
axes.set_ylabel("comments")
axes.plot(ted["views"],ted["comments"],ls="",marker=".")


# Some true outliers in views as well as comments.

# In[ ]:


fig=plt.figure()
axes=fig.add_axes([0,0,1,1])
axes.set_xlabel("views")
axes.set_xlim(2200000)
axes.set_ylabel("comments")
axes.set_ylim(200)
axes.plot(ted["views"],ted["comments"],ls="",marker=".")


# Except the previously seen outliers, from this scattered plot it can be seen, that the relation between views and comments depends on other factors.

# In[ ]:


labels = ted.num_speaker.unique()
sizes = ted.num_speaker.value_counts()

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax1.set_title("Number of presenters")

plt.show()


# * Nearly all talks are being held by one presenter.
# * Rather obvious that further quantitative analysis of 2% of dataset would result in deep insights.

# In[ ]:


ted.num_speaker.unique()
ted.num_speaker.value_counts()


# In[ ]:


ted.columns


# Now lets look at the ratings.

# In[ ]:


ted.ratings[0]


# Ratings are stored in a string. In the string is a list of dictionaries. Each dictionary has categories:
#     "ID"
#     "Name"
#     "Count"

# Lets figure out how to get rid of the string:

# In[ ]:


# ast.literal_eval evaluates the content of a string as python code - in a safe way (https://docs.python.org/3/library/ast.html#ast.literal_eval)
import ast
ted.ratings=ted.ratings.apply(lambda x:ast.literal_eval(x))
# Now ted.ratings is of type list


# In[ ]:


ted.ratings[0]


# In[ ]:


#Now I have a lists of dictionaries
ted.ratings[0][1]


# How do I evaluate, e.g. sum, over these dictionaries?
# 
# 1) Lets try to find all different categories.

# In[ ]:


ted.ratings[0][1]["name"]


# In[ ]:


interim=pd.DataFrame(ted.ratings[0])
interim


# Ok. Every rating cell can be interpreted as a dataframe.
# 
# Let's try to add all these dataframes up to see what are the most and least used names and how many there are.

# In[ ]:


interim2=pd.DataFrame(ted.ratings.sum())
interim2.name.value_counts()


# Seems like there are 14 categories and each categorie exists for every talk. I wonder why there are more IDs, but I won't bother. Let's see what are the sums of all rating categories.

# In[ ]:


interim3=interim2.groupby("name").sum()["count"].sort_values(ascending=False)
interim3


# **Observations**
# * TED talks, unsurprisingly, are considered very inspiring, informative and fascinating.
# * Raters seem to be in general a very positive bunch. The "negative" ratings exclusively form the end of the list: "unconvincing", "longwinded", "obnoxious", "confusing". 
# ![](http://img09.deviantart.net/9210/i/2013/196/a/c/rainbow_unicorn_by_iridalaoi-d6djq2a.jpg)

# Could be interesting to see correlations between these categories and views. Check if average view per rating-count differs.

# In[ ]:


Categories=["Inspiring","Informative","Fascinating","Persuasive","Beautiful","Courageous","Funny","Ingenious","Jaw-dropping","OK","Unconvincing","Longwinded","Obnoxious","Confusing"]
Categories


# In order to analyze further I needed to construct columns per rating category.

# In[ ]:


"""
This took me a very long time to somehow solve it. I am quite sure in a very, very ugly way.
If you happen to be able to point me in a direction how to do this much more elegant please do so!
"""
def normalized_count_of(i,x):
    # x is the full ratings dataframe
    # i is the Category name which is now the column to be filled
    for count in range(14):
        if pd.DataFrame(x)["name"][count]==Categories[i]:
            return pd.DataFrame(x).loc[count,"count"]

def total_count(x):
    total_votes=0
    for count in range(14):
        total_votes+=pd.DataFrame(x).loc[count,"count"]
    return total_votes
    
ted["Total_Votes"]=ted["ratings"].apply(lambda x:total_count(x))
        
for i in range(14):
    ted[Categories[i]]=ted["ratings"].apply(lambda x:normalized_count_of(i,x))/ted["Total_Votes"]
ted.head(2)


# In[ ]:


import seaborn as sns
sns.heatmap(ted.corr().iloc[6:33,6:33])


# **Observations:**
# * Unsurprisingly largest positive correlation between views and total_votes
# * Nice to see the square cluster of "negative" ratings which are positive correlated to each other
# * Inspiring (remember, most common category) talks are, in relation, not informative or fascinating
# * "Informative" and "Beautiful" correlate negatively
# * "Fascinating" correlates negatively with both "Persuative" and "Courageous"
# * Number of "Views" does not seem to be specifically tied to a certain category. "negative ratings" to correlate negatively.

# And now welcome to the "Oscars of TED"!
# ![](http://www.worldblaze.in/wp-content/uploads/2015/12/Oscar-award.jpg)
# The price for the most inspiring talk goes to:

# In[ ]:


ted[ted["Inspiring"]==ted["Inspiring"].max()]


# https://www.ted.com/talks/susan_lim?language=en
# Interestingly a talk from a minor event with very little views (625,415)
# 
# The most "Funny" talk of all times:
# ![](http://www.reactiongifs.com/wp-content/uploads/2013/11/baby-dance.gif)

# In[ ]:


ted[ted["Funny"]==ted["Funny"].max()]


# Wow. 70% Funny.
# https://www.ted.com/talks/julia_sweeney_has_the_talk
# 
# Excuse me, I have to watch a talk now. :-)
# 
# Finishing this descriptive Kernel now to resume my learning. Will revert back to a Kaggle dataset, probably even this one, when I have worked myself into sklearn. 
# 
# If you made it till here: hope you found this Kernel interesting. As said I would be very happy if you have tipps and hints for me.

# In[ ]:




