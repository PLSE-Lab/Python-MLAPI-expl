#!/usr/bin/env python
# coding: utf-8

# # Doctor Who ?
# 
# In this notebook there is a gentle introduction to the dataset associated to Doctor who that I just build. Nothing really fancy just an exploration of the data and some basic usage of the NLTK library to find some informations.

# In[ ]:


from datetime import datetime
import itertools
import ast
import re
import itertools
from collections import Counter

import pandas as pd
import numpy as np
import nltk
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
# nltk.download('vader_lexicon')
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
sns.set(color_codes=True)


# In[ ]:


#folder = "data/kaggle"
folder = "../input"


# ## IMDB (Modern age)
# 
# In this part, an analysis of the rating of the different seasons will be done to see if there is general trends in function of the new doctor etc.

# In[ ]:


# Collect the data from imdb
imdb_details = pd.read_csv(f"{folder}/imdb_details.csv")


# In[ ]:


imdb_details.head()


# In[ ]:


# Let's just draw the rating in function of the episode
fig,ax=plt.subplots(figsize=[12,6])

imdb_details.plot(ax = ax,kind = "line", y="rating", legend=None)

plt.xlabel("Episodes")
plt.ylabel("IMDB rating")

plt.show()


# *The rating on IMDB is between 0 and 10, in the dataset this is the average rating display on the main page of the episode (maybe I should collect individual rating and comments)*
# 
# **Notes**: As we can see the rating of the IMDB user seems very different in function of the episode but it never fall under 5. From this perspective it can be noticed that the last episode seems to be less appreciate than the others.
# 

# In[ ]:


# Let's draw the avergae rating per season
fig,ax=plt.subplots(figsize=[12,6])

imdb_details.groupby(["season"]).mean()["rating"].plot(ax = ax,kind = "line", legend=None)

plt.xlabel("Season", fontsize = 15)
plt.ylabel("Average IMDB rating", fontsize = 15)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)

plt.ylim([0,10])

plt.show()
fig.savefig('imdb_season_rating.png')


# **Notes**:In this case in avergae all the season seems to be appreciate with an average rating around 8 but the last season seems to have find his audience with an average rating around 6. This difference between the previous season can be explained by
# * the actor who was playing the doctor changed from Peter Capaldi to Jodie Whittaker, the new actress will be the first woman to interpret the Doctor (so that could have impacted the IMDB reviewers)
# * the showrunner is not Steven Moffat that relaunch the show with the modern area, so the tone and rythm could be different and the IMDB reviewers wil not like too
# 
# I have no opinion on this last season because I still didn't finsih the last season with Peter Capaldi, but I was living in the UK and I saw all the hate on the newspapers about the choice of Jodie Whittaker to be the Doctor so maybe there is on the number of reviews a trace of this hate by seeing a very important number of reviews to do some bashing.
# 

# In[ ]:


# Let's draw the number of review per season
fig,ax=plt.subplots(figsize=[12,6])

imdb_details.groupby(["season"]).mean()["nbr_votes"].plot(ax = ax,kind = "line", legend=None)

plt.xlabel("Season", fontsize = 15)
plt.ylabel("Average count of reviews", fontsize = 15)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)

plt.ylim([0,6000])


plt.show()

fig.savefig('imdb_season_ratingcount.png')


# **Notes**: It doesn't seems that there is some review bombing on the last season of ther show, so it seems to be a dislike of this new season.

# ## Analysis of the docto who guide data 
# 
# In this part, I am going to see some details on the show in terms of broadcast and distribution.

# In[ ]:


dw_guide = pd.read_csv(f"{folder}/dwguide.csv")


# In[ ]:


# Make some cleaning
dw_guide["AI"] = dw_guide.apply(lambda row: float(row["AI"]),axis=1)
dw_guide["views"] = dw_guide.apply(lambda row: float(row["views"].replace("m","")),axis=1)
dw_guide["date"] = dw_guide.apply(lambda row: row['broadcastdate'] + " " + row['broadcasthour'] ,axis=1)
dw_guide["broadcastdate_datetime"] = dw_guide.apply(lambda row: datetime.strptime(row['date'],"%d %b %Y %I:%M%p") ,axis=1)
dw_guide["broadcastdate_hour"] = dw_guide.apply(lambda row: row['broadcastdate_datetime'].hour,axis=1)
dw_guide["broadcastdate_year"] = dw_guide.apply(lambda row: row['broadcastdate_datetime'].year,axis=1)

#Works on the title of the episode (for the classic area)
dw_guide["title2"] = dw_guide["title"].apply(lambda x:x.split(":")[0])

# Clean the dirty string list
dw_guide["cast"] = dw_guide["cast"].apply(lambda x:ast.literal_eval(x))
dw_guide["crew"] = dw_guide["crew"].apply(lambda x:ast.literal_eval(x))


# In[ ]:


# Estimate if the episode was in the classic era or the modern era
def is_classicperiod(x):
    if x>=2005:
        return False
    return True

dw_guide["is_classicperiod"] = dw_guide["broadcastdate_year"].apply(lambda x:is_classicperiod(x))


# In[ ]:


dw_guide.sort_values(["episodenbr"], ascending = True, inplace = True)
dw_guide.reset_index(inplace = True, drop = True)


# In[ ]:


dw_guide.head()


# ### Focus on the casting of actors

# In[ ]:


# Collect the right columns for the analysis
dw_guide_cast = dw_guide[["episodenbr","title","title2","broadcastdate_datetime","broadcastdate_year","is_classicperiod","cast"]].reset_index()


# In[ ]:


# Rebuild the casting of the show
#casting = dw_guide_cast["cast"].explode().to_frame().reset_index() # Run with pandas 0.25

# Use an old trick to do it
casting = pd.DataFrame({'index':dw_guide_cast["index"].repeat(dw_guide_cast["cast"].str.len()),'cast':np.concatenate(dw_guide_cast["cast"].values)})


# In[ ]:


# Rebuild the casting of the show
casting["name"] = casting["cast"].apply(lambda x:x["name"])
casting["role"] = casting["cast"].apply(lambda x:x["role"])

# Get an uncredited flag
def is_uncredited(x):
    if "uncredited" in x:
        return True
    return False
casting["is_uncredited"] = casting["name"].apply(lambda x:is_uncredited(x))
# Drop the uncredited tag in the name
casting["name"] = casting["name"].apply(lambda x:x.replace(" (uncredited)",""))


del casting["cast"]
del dw_guide_cast["cast"]


# In[ ]:


casting.head()


# In[ ]:


# Upgrade the general informations on the casting
dw_guide_cast = dw_guide_cast.reset_index().merge(casting,on = ["index"])
del dw_guide_cast["index"]


# In[ ]:


dw_guide_cast.head()


# In[ ]:


# Build some statistic on the actor who played in the show (if they are bask or not etc)
agg_func = {
    "broadcastdate_year":["min","max"],
    "episodenbr":[pd.Series.nunique]
}
stats_cast = dw_guide_cast[dw_guide_cast["is_uncredited"] == False].groupby(["name"]).agg(agg_func).reset_index()
stats_cast["deltatime"] = stats_cast["broadcastdate_year","max"] - stats_cast["broadcastdate_year","min"]

stats_cast.sort_values(["deltatime"], ascending = False, inplace = True)

old_columns = stats_cast.columns
new_columns = []
for column in old_columns:
    new_columns.append(f"{column[0]}_{column[1]}")
stats_cast.columns = new_columns


# In[ ]:


# Determine if the actor start or end in the modern age of the show
def is_threshold(x, limit, is_superior = True):
    if is_superior:
        if x >= limit:
            return True
        return False
    else:
        if x <= limit:
            return True
        return False
    
stats_cast['firstappereance_ismodern'] = stats_cast["broadcastdate_year_min"].apply(lambda x:is_threshold(x, 2005))
stats_cast['lastappereance_ismodern'] = stats_cast["broadcastdate_year_max"].apply(lambda x:is_threshold(x, 2005))

stats_cast["appereance_modernage"] = stats_cast.apply(lambda x:f"FA:{x['firstappereance_ismodern']} / LA:{x['lastappereance_ismodern']}",axis = 1)


# In[ ]:


stats_cast.head(5)


# In[ ]:


# Number of actors in the show
len(stats_cast)


# In[ ]:


# Determine the proportion of episode played in function of the profile of the actors
stats = stats_cast[["episodenbr_nunique","appereance_modernage"]].groupby(["appereance_modernage"]).describe().unstack(1).to_frame().reset_index()
study_episodes = pd.pivot_table(stats, columns = ["level_1"], index = ["appereance_modernage"], values = [0])


# In[ ]:


# Renaming and selection of the right columns
old_columns = study_episodes.columns
new_columns = []
for column in old_columns:
    new_columns.append(f"{column[1]}")
study_episodes.columns = new_columns
study_episodes = study_episodes[["min","25%","50%","75%","max","count"]]


# In[ ]:


study_episodes


# **Notes**: This analyse is illustrating the fact that the actors of the classic era , where doing more than 1 episode it could be explained by the fact that the story was divide in multiple episodes (so you need to keep the same actors). For the modern age, most of the actors are doing only one episode.
# 
# The actors that can be find on the two eras have an important number of episodes, that could be explained by the fact that some actors that are playing the doctor in the classic era (so a high number of episodes).

# In[ ]:


# Let's see now the actor and their count of role
count_role = dw_guide_cast[dw_guide_cast["is_uncredited"] == False].groupby(["name"]).nunique()["role"].reset_index()
count_role.sort_values(["role"], ascending = False, inplace = True)


# In[ ]:


count_role.head()


# **Notes**: This study highlight the actor that are playing some characters in the costume like the daleks, cybermen . It's highlighting the voice actor behind the Dalek for exemple.

# In[ ]:


dw_guide_cast[dw_guide_cast["name"] == "Nicholas Briggs"]["role"].unique()


# ## Analyse of the scripts

# In[ ]:


# Collect the scripts
all_scripts = pd.read_csv(f"{folder}/all-scripts.csv")
all_scripts.sort_values(["doctorid","episodeid"],ascending = True,inplace = True)


# In[ ]:


all_scripts.head()


# In[ ]:


# Add details on the type of doctor
def clean_name(row):
    clean_name = str(row["details"])
    for piece in [" [OC]","[OC]"]:
        if piece in clean_name:
            clean_name = clean_name.replace(piece,"")
    
    if "DOCTOR" in clean_name:
        return f"DOCTOR_{row['doctorid']}"
    
    return clean_name

all_scripts["details"] = all_scripts.apply(lambda row: clean_name(row) ,axis=1)


# ### Dive in NLTK

# In[ ]:


# Focus on the talks of the episodes
all_talks = all_scripts[all_scripts["type"] == "talk"]
all_talks.reset_index(inplace = True, drop = True)


# In[ ]:


#Get max idx for each episode (to estimate the progression during a speech of the episode)
max_idx = all_scripts.groupby(["episodeid"]).max()["idx"].to_frame()
max_idx.reset_index(inplace = True)
max_idx.columns= ["episodeid","max_idx"]


# In[ ]:


# upgrade the talks 
all_talks = all_talks.merge(max_idx,on = ["episodeid"])


# In[ ]:


# Estimate the progression and make some kind of buckets of progression
all_talks["progression"] = 100.0 * all_talks["idx"] / all_talks["max_idx"]
all_talks["int_progression"] = all_talks["progression"].astype(int)

for block in [2,5,10]:
    all_talks[f"progression_{block}%"] = all_talks["int_progression"] / block
    all_talks[f"progression_{block}%"] = all_talks[f"progression_{block}%"].astype(int)

del all_talks["max_idx"]


# In[ ]:


# Sorting time
all_talks.sort_values(["episodeid","idx"], inplace = True)


# In[ ]:


all_talks.head()


# ### Let's tokenise the scripts

# In[ ]:


# Make some string cleaning
all_talks["text"] = all_talks["text"].apply(lambda x:re.sub("[^a-zA-Z]"," ", str(x)))
all_talks["text"] = all_talks["text"].apply(lambda row: row.lower())


# In[ ]:


# Make tokenisation of the scripts
all_talks["tokenised_text"] = all_talks["text"].apply(lambda x:nltk.sent_tokenize(x) )


# In[ ]:


# Quick check
all_talks.sample(frac = 0.1).head()


# In[ ]:


# Make word tokenisation
def get_wordtokenise(container):
    new_container = []
    for elt in container:
        new_container.append(nltk.tokenize.word_tokenize(elt))
    return new_container

all_talks["word_tokenised_text"] = all_talks["tokenised_text"].apply(lambda x:get_wordtokenise(x))
all_talks["merged_word_tokenised_text"] = all_talks["word_tokenised_text"].apply(lambda x:list(itertools.chain.from_iterable(x)))


# In[ ]:


all_talks.sample(frac = 0.1).head()


# In[ ]:


# Collect all the words used (after tokenisation)
all_words = list(itertools.chain.from_iterable(all_talks["merged_word_tokenised_text"].tolist()))


# In[ ]:


# Estimate the most common word used in all the script
fdist = nltk.probability.FreqDist(all_words)
pd.DataFrame(fdist.most_common(10), columns = ["word","count"])


# **Notes**: The most common words seems the small words , so we need to make some cleaning

# ### Nake some cleaning of the small words

# In[ ]:


# Get the stopwords
stopwords_list = nltk.corpus.stopwords.words("english")
# But don't drop the who word please
del stopwords_list[stopwords_list.index("who")]


# In[ ]:


def drop_stopwords(container,stop_words):
    new_container = []
    for subcontainer in container:
        new_subcontainer = []
        for elt in subcontainer:
            if elt not in stopwords_list:
                new_subcontainer.append(elt)
        new_container.append(new_subcontainer)
    return new_container

all_talks["word_tokenised_text_nostopword"] = all_talks["word_tokenised_text"].apply(lambda x:drop_stopwords(x,stopwords_list))
all_talks["merged_word_tokenised_text_nostopword"] = all_talks["word_tokenised_text_nostopword"].apply(lambda x:list(itertools.chain.from_iterable(x)))


# In[ ]:


# Collect all the words used (after tokenisation and drop of the stopwords)
all_words = list(itertools.chain.from_iterable(all_talks["merged_word_tokenised_text_nostopword"].tolist()))


# In[ ]:


# Estimate the most common word used in all the script
fdist = nltk.probability.FreqDist(all_words)
pd.DataFrame(fdist.most_common(10), columns = ["word","count"])


# **Notes**: Now we can start to have some interesting trends on the most used word, and guess what doctor is the most common !!!

# ### Works on the stemmer

# In[ ]:


# Define the stemmer (to just get the root )
ps = nltk.stem.PorterStemmer()


# In[ ]:


# It's time to find the root of the word
def get_stem(container,ps):
    new_container = []
    for subcontainer in container:
        new_subcontainer = []
        for elt in subcontainer:
            new_subcontainer.append(ps.stem(elt))
        new_container.append(new_subcontainer)
    return new_container

all_talks["word_tokenised_text_nostopword_stem"] = all_talks["word_tokenised_text_nostopword"].apply(lambda x:get_stem(x,ps))
all_talks["merged_word_tokenised_text_nostopword_stem"] = all_talks["word_tokenised_text_nostopword_stem"].apply(lambda x:list(itertools.chain.from_iterable(x)))


# In[ ]:


all_talks.sample(frac = 0.1).head()


# In[ ]:


# Collect all the words used (after tokenisation and drop of the stopwords)
all_stems = list(itertools.chain.from_iterable(all_talks["merged_word_tokenised_text_nostopword_stem"].tolist()))


# In[ ]:


# Estimate the most common word used in all the script
fdist = nltk.probability.FreqDist(all_stems)
pd.DataFrame(fdist.most_common(10), columns = ["word","count"])


# ## Get details on the doctor word concordance

# In[ ]:


# Block to determine the word that are arriving before or after the word doctor
count_starter = 0
count_end = 0
count_only = 0

dict_count = {
    "before":[],
    "after":[]
}

for elt in all_talks["word_tokenised_text_nostopword"]:
    for quote in elt:
        if "doctor" in quote:
            quote_s = quote
            concordance = nltk.ConcordanceIndex(quote_s)
            detection = concordance.offsets('doctor')
            if len(quote_s) != 1:
                # Works on what's before doctor
                for idx in detection:
                    if idx == 0:
                        count_starter += 1
                        dict_count["after"].append(quote_s[idx+1])
                    elif idx == len(quote) - 1:
                        count_end += 1
                        dict_count["before"].append(quote_s[idx-1])
                    else:
                        dict_count["after"].append(quote_s[idx+1])
                        dict_count["before"].append(quote_s[idx-1])
            else:
                count_only += 1
                                        
print("Count first word", count_starter)
print("Count last word", count_end)
print("Count only", count_only)


# In[ ]:


list(Counter(dict_count["before"]).items())


# In[ ]:


#Count_occurence in each list
count_before = pd.DataFrame(list(Counter(dict_count["before"]).items()), columns = ["word","count_before"])
count_after = pd.DataFrame(list(Counter(dict_count["after"]).items()), columns = ["word","count_after"])

# Merge the count
count = pd.merge(count_before,count_after,on = "word",how='outer').fillna(0)
count["count_all"] = count["count_before"] + count["count_after"]


# In[ ]:


count.sort_values(["count_all"],ascending = False,inplace = True)
count.reset_index(inplace = True, drop = True)


# In[ ]:


count.head(20)


# **Notes**: The word doctor seems to be the one that is the more connect to doctor, the word who is just the 20th word the more related to doctor (on 4298 so that's not too bad)

# ## Get details on the wordcloud of some companions

# In[ ]:


stopwords = set(STOPWORDS)
#fig,ax=plt.subplots(figsize=[12,12])
cnt = 1
for character in ["ROSE","MARTHA","DONNA","AMY","CLARA","BILL"]:
    print(character)
    speech = all_talks[all_talks["details"] == character]
    text = "".join(sentence for sentence in speech["text"].tolist())
    wordcloud = WordCloud(stopwords=stopwords_list,background_color="white").generate(text)

    
    all_words = list(itertools.chain.from_iterable(speech["merged_word_tokenised_text_nostopword"].tolist()))
    fdist = nltk.probability.FreqDist(all_words)
    print(pd.DataFrame(fdist.most_common(10), columns = ["word","count"]))
    
    fig,ax=plt.subplots(figsize=[12,12])
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    #plt.title(f"Wordcloud : {character}", fontsize = 15)
    plt.show()
    
    fig.tight_layout()
    fig.savefig(f"wordcloud_{character}.png")


# **Notes**: The word doctor

# ### Let's make a small sentiment analysis on the doctor speech

# In[ ]:


# Define the sentiment analyser
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()


# In[ ]:


# Let's do the analysis of all the scripts
data = []
for i,row in all_talks.iterrows():
    sentence = row["text"]
    sa = sid.polarity_scores(sentence)
    sa["index"] = i
    data.append(sa)

sentiment_analysis = pd.DataFrame(data) 
sentiment_analysis.set_index(["index"],inplace = True)
sentiment_analysis.columns = [f"ss_{column}"for column in sentiment_analysis.columns]


# In[ ]:


all_talks_sa = pd.concat([all_talks,sentiment_analysis],axis = 1)
all_talks_sa.sort_values(["doctorid","episodeid","idx"],inplace = True)


# In[ ]:


# Determine the text said by the doctor in general
def is_containingword(x,word):
    if word in x:
        return True
    return False

all_talks_sa["is_doctorspeech"] = all_talks_sa["details"].apply(lambda x:is_containingword(x,"DOCTOR"))


# In[ ]:


# Let's make a simple wordcloud to detect the negative word said by the doctor

#Focus on the doctor talk
doctor_speech = all_talks_sa[all_talks_sa["is_doctorspeech"]]
# focus on the negative speech of the doctor
speech = doctor_speech[doctor_speech["ss_neg"] >= 0.5]

all_words = list(itertools.chain.from_iterable(speech["merged_word_tokenised_text_nostopword"].tolist()))
fdist = nltk.probability.FreqDist(all_words)
print(pd.DataFrame(fdist.most_common(10), columns = ["word","count"]))

text = "".join(sentence for sentence in speech["text"].tolist())
wordcloud = WordCloud(stopwords=stopwords_list,background_color="white").generate(text)

fig,ax=plt.subplots(figsize=[12,12])
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
#plt.title(f"Wordcloud : {character}", fontsize = 15)
plt.show()

fig.tight_layout()
fig.savefig(f"wordcloud_negative_doctor.png")


# In[ ]:




