#!/usr/bin/env python
# coding: utf-8

# ## Sentiment analysis on Trump and Hillary tweets

# In[ ]:


#libraries
# This Python 3 environment
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
import matplotlib.pyplot as plt#visualization
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import plotly.offline as py#visualization
py.init_notebook_mode(connected=True)#visualization
import plotly.graph_objs as go#visualization
import plotly.tools as tls#visualization
import plotly.figure_factory as ff#visualization
# Any results you write to the current directory are saved as output.


# ## Import Data

# In[ ]:


#import data
tweets = pd.read_csv(r"../input/clinton-trump-tweets/tweets.csv")
#select columns
tweets = tweets[[ 'handle', 'text', 'is_retweet', 'original_author', 
                 'time', 'lang', 'retweet_count', 'favorite_count']]
    
tweets.head()


# ## Data manipulation

# In[ ]:



#convert to date format and extract hour
from datetime import datetime
date_format = "%Y-%m-%dT%H:%M:%S" 
tweets["time"]   = pd.to_datetime(tweets["time"],format = date_format)
tweets["hour"]   = pd.DatetimeIndex(tweets["time"]).hour
tweets["month"]  = pd.DatetimeIndex(tweets["time"]).month
tweets["day"]    = pd.DatetimeIndex(tweets["time"]).day
tweets["month_f"]  = tweets["month"].map({1:"JAN",2:"FEB",3:"MAR",
                                        4:"APR",5:"MAY",6:"JUN",
                                        7:"JUL",8:"AUG",9:"SEP"})

#language
def label_language(df) :
    if df["lang"] == "en" :
        return "English"
    elif df["lang"] == "es" :
        return "Spanish"
    else :
        return "Other"
tweets["lang"] = tweets.apply(lambda tweets:label_language(tweets),axis = 1)


#create new tweets column
tweets["tweets"] = tweets["text"]

#text manipulation
import  re
from nltk.corpus import stopwords
stop_words = stopwords.words("english")

#function to remove special characters , punctions ,stop words ,
#digits ,hyperlinks and case conversion
def string_manipulation(df,column)  : 
    #extract hashtags
    df["hashtag"]  = df[column].str.findall(r'#.*?(?=\s|$)')
    #extract twitter account references
    df["accounts"] = df[column].str.findall(r'@.*?(?=\s|$)')
    
    #remove hashtags and accounts from tweets
    df[column] = df[column].str.replace(r'@.*?(?=\s|$)'," ")
    df[column] = df[column].str.replace(r'#.*?(?=\s|$)'," ")
    
    #convert to lower case
    df[column] = df[column].str.lower()
    #remove hyperlinks
    df[column] = df[column].apply(lambda x:re.split('https:\/\/.*',str(x))[0])
    #remove punctuations
    df[column] = df[column].str.replace('[^\w\s]'," ")
    #remove special characters
    df[column] = df[column].str.replace("\W"," ")
    #remove digits
    df[column] = df[column].str.replace("\d+"," ")
    #remove under scores
    df[column] = df[column].str.replace("_"," ")
    #remove stopwords
    df[column] = df[column].apply(lambda x: " ".join([i for i in x.split() 
                                                      if i not in (stop_words)]))
    return df

tweets = string_manipulation(tweets,"text")


#trump tweets without retweets
tweets_trump   = (tweets[(tweets["handle"] == "realDonaldTrump") &
                         (tweets["is_retweet"] == False)].reset_index()
                  .drop(columns = ["index"],axis = 1))

#trump tweets with retweets
tweets_trump_retweets   = (tweets[(tweets["handle"] == "realDonaldTrump") &
                                  (tweets["is_retweet"] == True)].reset_index()
                                  .drop(columns = ["index"],axis = 1))

#hillary tweets without retweets
tweets_hillary  = (tweets[(tweets["handle"] == "HillaryClinton") &
                            (tweets["is_retweet"] == False)].reset_index()
                              .drop(columns = ["index"],axis = 1))

#hillary tweets with retweets
tweets_hillary_retweets  = (tweets[(tweets["handle"] == "HillaryClinton") &
                            (tweets["is_retweet"] == True)].reset_index()
                              .drop(columns = ["index"],axis = 1))

display(tweets_trump.head(4).style.set_properties(**{}).set_caption("Trump tweets"))
display(tweets_hillary.head(4).style.set_properties(**{}).set_caption("Hillary tweets"))


# ## Percentage of retweets

# In[ ]:


plt.style.use('ggplot')

plt.figure(figsize = (13,6))
plt.subplot(121)
tweets[tweets["handle"] ==
       "realDonaldTrump"]["is_retweet"].value_counts().plot.pie(autopct = "%1.0f%%",
                                                                wedgeprops = {"linewidth" : 1,
                                                                              "edgecolor" : "k"},
                                                                shadow = True,fontsize = 13,
                                                                explode = [.1,0.09],
                                                                startangle = 20,
                                                                colors = ["#FF3300","w"]
                                                               )
plt.ylabel("")
plt.title("Percentage of retweets - Trump")

plt.subplot(122)
tweets[tweets["handle"] ==
       "HillaryClinton"]["is_retweet"].value_counts().plot.pie(autopct = "%1.0f%%",
                                                                wedgeprops = {"linewidth" : 1,
                                                                              "edgecolor" : "k"},
                                                                shadow = True,fontsize = 13,
                                                                explode = [.09,0],
                                                                startangle = 60,
                                                                colors = ["#6666FF","w"]
                                                               )
plt.ylabel("")
plt.title("Percentage of retweets - Hillary")
plt.show()


# ## Languages used in tweets

# In[ ]:


plt.figure(figsize = (12,7))

plt.subplot(121)
ax = sns.countplot(y = tweets[tweets["handle"] == "realDonaldTrump"]["lang"] ,
                   linewidth = 1,edgecolor = "k"*3,
                   palette = "Reds_r")

for i,j in enumerate(tweets[tweets["handle"] == 
                            "realDonaldTrump"]["lang"].value_counts().values) :
    ax.text(.7,i,j,fontsize = 15)

plt.grid(True)
plt.title("Languages used in tweets - trump")
    
plt.subplot(122)
ax1 = sns.countplot(y = tweets[tweets["handle"] == "HillaryClinton"]["lang"] ,
                   linewidth = 1,edgecolor = "k"*3,
                    palette = "Blues_r")

for i,j in enumerate(tweets[tweets["handle"] == 
                            "HillaryClinton"]["lang"].value_counts().values) :
    ax1.text(.7,i,j,fontsize = 15)

plt.grid(True)
plt.ylabel("")
plt.title("Languages used in tweets - hillary")
plt.show()


# ## original authors of retweets

# In[ ]:


plt.figure(figsize = (10,14))

plt.subplot(211)
authors = tweets_trump_retweets["original_author"].value_counts().reset_index()
sns.barplot(y = authors["index"][:15] , 
            x = authors["original_author"][:15] ,
            linewidth = 1,edgecolor = "k",color = "#FF3300")
plt.grid(True)
plt.xlabel("count")
plt.ylabel("original author")
plt.title("original authors of retweets - Trump")

plt.subplot(212)
authors1 = tweets_hillary_retweets["original_author"].value_counts().reset_index()
sns.barplot(y = authors1["index"][:15] , 
            x = authors1["original_author"][:15] ,
            linewidth = 1,edgecolor = "k",color ="#6666FF")
plt.grid(True)
plt.xlabel("count")
plt.ylabel("original author")
plt.title("original authors of retweets - Hillary")
plt.show()


# ## tweets by month

# In[ ]:


plt.figure(figsize = (12,8))
sns.countplot(x = "month_f",hue = "handle",palette = ["#FF3300","#6666FF"],
              data = tweets.sort_values(by = "month",ascending = True),
             linewidth = 1,edgecolor = "k"*tweets_trump["month"].nunique())
plt.grid(True)
plt.title("tweets by month (2016)")
plt.show()


# ## import positive and negative words dictionaries

# In[ ]:


#positive words
positive_words = pd.read_csv(r"../input/positive-words/positive-words.txt",
                             header=None)
#negative words
negative_words = pd.read_csv(r"../input/negative-words/negative-words.txt",
                             header=None,encoding='latin-1')

#convert words to lists
def convert_words_list(df) : 
    words = string_manipulation(df,0)
    words_list = words[words[0] != ""][0].tolist()
    return words_list

positive_words_list = convert_words_list(positive_words)

#remove word trump from positive word list
positive_words_list = [i for i in positive_words_list if i not in "trump"]
negative_words_list = convert_words_list(negative_words)

print ( "positive words : " )
print (positive_words_list[:50])
print ( "negative words : " )
print (negative_words_list[:50])


# ## scoring tweets
# * scoring tweets based on positive and negative words count.
# * score = positive_count - negative_count
#  

# In[ ]:


# function to score tweets based on positive and negative words present
def scoring_tweets(data_frame,text_column) :
    #identifying +ve and -ve words in tweets
    data_frame["positive"] = data_frame[text_column].apply(lambda x:" ".join([i for i in x.split() 
                                                                              if i in (positive_words_list)]))
    data_frame["negative"] = data_frame[text_column].apply(lambda x:" ".join([i for i in x.split()
                                                                              if i in (negative_words_list)]))
    #scoring
    data_frame["positive_count"] = data_frame["positive"].str.split().str.len()
    data_frame["negative_count"] = data_frame["negative"].str.split().str.len()
    data_frame["score"]          = (data_frame["positive_count"] -
                                    data_frame["negative_count"])
    
    #create new feature sentiment :
    #+ve if score is +ve , #-ve if score is -ve , # neutral if score is 0
    def labeling(data_frame) :
        if data_frame["score"]   > 0  :
            return "positive"
        elif data_frame["score"] < 0  :
            return "negative"
        elif data_frame["score"] == 0 :
            return "neutral"
    data_frame["sentiment"] = data_frame.apply(lambda data_frame:labeling(data_frame),
                                               axis = 1)
        
    return data_frame

tweets         = scoring_tweets(tweets,"text")
tweets_trump   = scoring_tweets(tweets_trump,"text")
tweets_hillary = scoring_tweets(tweets_hillary,"text")

tweets[["text","positive","negative","positive_count",
              "negative_count","score","sentiment"]].head()


# ## Scores distribution

# In[ ]:


score_dist = tweets[tweets["is_retweet"] ==
                    False].groupby("handle")["score"].value_counts().to_frame()
score_dist.columns = ["count"]
score_dist = score_dist.reset_index().sort_values(by = "score",ascending = False)

trace = go.Bar(x = score_dist[score_dist["handle"] == "realDonaldTrump"]["score"],
               y = score_dist[score_dist["handle"] == "realDonaldTrump"]["count"],
               marker = dict(line = dict(width = 1,color = "black"),
                             color = "red"),name = "Donald Trump"
              )

trace1 = go.Bar(x = score_dist[score_dist["handle"] == "HillaryClinton"]["score"],
                y = score_dist[score_dist["handle"] == "HillaryClinton"]["count"],
                marker = dict(line = dict(width = 1,color = "black"),
                             color = "blue"),name = "Hillary Clinton"
              )
layout = go.Layout(dict(title = "Scores distribution",
                        plot_bgcolor  = "rgb(243,243,243)",
                        paper_bgcolor = "rgb(243,243,243)",
                        xaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                     gridwidth = 2),
                        yaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                     gridwidth = 2),
                        )
                  )
fig = go.Figure(data = [trace,trace1],layout = layout)
py.iplot(fig)


# ## Sentiment distribution of tweets

# In[ ]:


sent_dist = pd.crosstab(tweets[tweets["is_retweet"] == 
                               False]["sentiment"],
                        tweets[tweets["is_retweet"] ==
                               False]["handle"]).apply(lambda r:
                                                       r/r.sum()*100,axis = 0)

sent_dist = sent_dist.reset_index()
t1 = go.Bar(x = sent_dist["sentiment"],y = sent_dist["HillaryClinton"],
            name = "Hillary Clinton",
            marker = dict(line = dict(width = 1,color = "#000000"),color = "#6666FF"))

t2 = go.Bar(x = sent_dist["sentiment"],y = sent_dist["realDonaldTrump"],
           name = "Donald Trump",
           marker = dict(line = dict(width = 1,color = "#000000"),color = "#FF3300"))

layout = go.Layout(dict(title = "Sentiment distribution",
                        plot_bgcolor  = "rgb(243,243,243)",
                        paper_bgcolor = "rgb(243,243,243)",
                        xaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                     gridwidth = 2,title = "sentiment"),
                        yaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                     gridwidth = 2,title = "percentage"),
                        )
                  )
fig = go.Figure(data = [t1,t2],layout = layout)
py.iplot(fig)


# ## Correlation matrix

# In[ ]:


df_corr = tweets[tweets["is_retweet"] == False][[ 'retweet_count', 'favorite_count' ,
                        'score', "sentiment","handle" ]]

df_corr["neutral"]   = np.where(df_corr["score"] == 0,1,0)
df_corr["negative"]  = np.where(df_corr["score"] <  0,1,0) 
df_corr["positive"]  = np.where(df_corr["score"] >  0,1,0)

cols = ['retweet_count','favorite_count','neutral','negative', 'positive']

correlation_hillary  = df_corr[df_corr["handle"] == "HillaryClinton"][cols].corr()
correlation_trump    = df_corr[df_corr["handle"] == "realDonaldTrump"][cols].corr()

plt.figure(figsize = (12,4.5))
plt.subplot(121)
sns.heatmap(correlation_hillary,annot = True,cmap = "hot_r",
            linecolor = "grey",linewidths = 1)
plt.title("Correlation matrix - Hillary")

plt.subplot(122)
sns.heatmap(correlation_trump,annot = True,cmap = "hot_r",
            linecolor = "grey",linewidths = 1)
plt.title("Correlation matrix - Trump")
plt.show()


# ## Popular hashtags 

# In[ ]:


hashs_t = tweets_trump["tweets"].str.extractall(r'(\#\w+)')[0].value_counts().reset_index()
hashs_t.columns = ["hash","count"]

hashs_h = tweets_hillary["tweets"].str.extractall(r'(\#\w+)')[0].value_counts().reset_index()
hashs_h.columns = ["hash","count"]


plt.figure(figsize = (10,20))
plt.subplot(211)
ax = sns.barplot(x = "count" , y = "hash" ,
                 data = hashs_t[:25] , palette = "seismic",
                 linewidth = 1 , edgecolor = "k"* 25)
plt.grid(True)
for i,j in enumerate(hashs_t["count"][:25].values) :
    ax.text(3,i,j,fontsize = 10,color = "white")
plt.title("Popular hashtags used by trump")

plt.subplot(212)
ax1 = sns.barplot(x = "count" , y = "hash" ,
                 data = hashs_h[:25] , palette = "seismic",
                 linewidth = 1 , edgecolor = "k"* 25)
plt.grid(True)
for i,j in enumerate(hashs_h["count"][:25].values) :
    ax1.text(.3,i,j,fontsize = 10,color = "white")
plt.title("Popular hashtags used by hillary")
plt.show()


# ## wordcloud - hashtags

# In[ ]:


from wordcloud import WordCloud

hsh_wrds_t = tweets_trump["tweets"].str.extractall(r'(\#\w+)')[0]
hsh_wrds_h = tweets_hillary["tweets"].str.extractall(r'(\#\w+)')[0]

def build_word_cloud(words,back_color,palette,title) :
    word_cloud = WordCloud(scale = 7,max_words = 1000,
                           max_font_size = 100,background_color = back_color,
                           random_state = 0,colormap = palette
                          ).generate(" ".join(words))
    plt.figure(figsize = (13,8))
    plt.imshow(word_cloud,interpolation = "bilinear")
    plt.axis("off")
    plt.title(title)
    plt.show()



build_word_cloud(hsh_wrds_t,"black","rainbow","Hashtags - Trump")
build_word_cloud(hsh_wrds_h,"black","rainbow","Hashtags - Hillary")


# ## Popular twitter  account references 

# In[ ]:


accounts_t = tweets_trump["tweets"].str.extractall(r'(\@\w+)')[0].value_counts().reset_index()
accounts_t.columns = ["accounts","count"]

accounts_h = tweets_hillary["tweets"].str.extractall(r'(\@\w+)')[0].value_counts().reset_index()
accounts_h.columns = ["accounts","count"]

plt.figure(figsize = (10,20))

plt.subplot(211)
ax = sns.barplot(x = "count" , y = "accounts" ,
                 data = accounts_t[:25] , palette = "seismic",
                 linewidth = 1 , edgecolor = "k"* 25)
plt.grid(True)
for i,j in enumerate(accounts_t["count"][:25].values) :
    ax.text(3,i,j,fontsize = 10,color = "white")
plt.title("Popular twitter  account references by trump")

plt.subplot(212)
ax1 = sns.barplot(x = "count" , y = "accounts" ,
                 data = accounts_h[:25] , palette = "seismic",
                 linewidth = 1 , edgecolor = "k"* 25)
plt.grid(True)
for i,j in enumerate(accounts_h["count"][:25].values) :
    ax1.text(3,i,j,fontsize = 10,color = "white")
plt.title("Popular twitter  account references by trump")
plt.show()


# ## wordcloud - accounts

# In[ ]:


acc_wrds_t = tweets_trump["tweets"].str.extractall(r'(\@\w+)')[0]    
acc_wrds_h = tweets_hillary["tweets"].str.extractall(r'(\@\w+)')[0]    

build_word_cloud(acc_wrds_t,"black","rainbow","twitter account references - Trump")
build_word_cloud(acc_wrds_h,"black","rainbow","twitter account references - Hillary")


# ## "popular words in tweets"

# In[ ]:


pop_wrds_t = (tweets_trump["text"].apply(lambda x : 
                                         pd.value_counts(x.split(" ")))
            .sum(axis = 0).reset_index().sort_values(by = [0],ascending = False))
pop_wrds_t.columns = ["word","count"]
pop_wrds_t["word"] = pop_wrds_t["word"].str.upper()

pop_wrds_d = (tweets_hillary["text"].apply(lambda x :
                                           pd.value_counts(x.split(" ")))
            .sum(axis = 0).reset_index().sort_values(by = [0],ascending = False))
pop_wrds_d.columns = ["word","count"]
pop_wrds_d["word"] = pop_wrds_d["word"].str.upper()

plt.figure(figsize = (12,25))
plt.subplot(211)
ax = sns.barplot(x = "count",y = "word",data = pop_wrds_t[:30],
                linewidth = 1 ,edgecolor = "k"*30,palette = "Reds")
plt.title("popular words in tweets - Trump")
plt.grid(True)
for i,j in enumerate(pop_wrds_t["count"][:30].astype(int)) :
    ax.text(.8,i,j,fontsize = 9)
    
plt.subplot(212)
ax1 = sns.barplot(x = "count",y = "word",data = pop_wrds_d[:30],
                linewidth = 1 ,edgecolor = "k"*30,palette = "Blues")
plt.title("popular words in tweets - Hillary")
plt.grid(True)
for i,j in enumerate(pop_wrds_d["count"][:30].astype(int)) :
    ax1.text(.8,i,j,fontsize = 9)


# ## word cloud - popular words

# In[ ]:


build_word_cloud(pop_wrds_t["word"],"black","Set1","popular words in tweets - Trump")
build_word_cloud(pop_wrds_d["word"],"black","Set1","popular words in tweets - Hillary")


# 
# ## Popular positive and negative words used by trump

# In[ ]:


def word_count(data_frame,column) :
    words = data_frame[column].str.split(expand = True)
    words = words.stack().reset_index()[0].value_counts().reset_index()
    words.columns = ["words","count"]
    words = words.sort_values(by = "count",ascending = False)
    words["words"] = words["words"].str.upper()
    return words
    
pop_pos_words_t = word_count(tweets_trump,"positive")
pop_neg_words_t = word_count(tweets_trump,"negative")

pop_pos_words_d = word_count(tweets_hillary,"positive")
pop_neg_words_d = word_count(tweets_hillary,"negative")


plt.figure(figsize = (12,22))
plt.subplot(221)
ax1 = sns.barplot(x = "count" , y = "words" ,
                 data = pop_pos_words_t[:20] , 
                 linewidth = 1 , edgecolor = "k"* 20)
plt.grid(True)
for i,j in enumerate(pop_pos_words_t["count"][:20].values) :
    ax1.text(8,i,j,fontsize = 10)
plt.title("Popular positive words used by trump")

plt.subplot(222)
ax2 = sns.barplot(x = "count" , y = "words" ,
                 data = pop_neg_words_t[:20] , 
                 linewidth = 1 , edgecolor = "k"* 20)
plt.grid(True)
for i,j in enumerate(pop_neg_words_t["count"][:20].values) :
    ax2.text(8,i,j,fontsize = 10)
plt.ylabel("")
plt.title("Popular negative words used by trump")


######3
plt.subplot(223)
ax3 = sns.barplot(x = "count" , y = "words" ,
                 data = pop_pos_words_d[:20] , 
                 linewidth = 1 , edgecolor = "k"* 20)
plt.grid(True)
for i,j in enumerate(pop_pos_words_d["count"][:20].values) :
    ax3.text(8,i,j,fontsize = 10)
plt.title("Popular positive words used by hillary")

plt.subplot(224)
ax4 = sns.barplot(x = "count" , y = "words" ,
                 data = pop_neg_words_d[:20] , 
                 linewidth = 1 , edgecolor = "k"* 20)
plt.grid(True)
for i,j in enumerate(pop_neg_words_d["count"][:20].values) :
    ax4.text(8,i,j,fontsize = 10)
plt.ylabel("")
plt.title("Popular negative words used by hillary")

plt.subplots_adjust(wspace = .3)
plt.show()


# ## Hashtag references by twitter accounts

# In[ ]:


accounts = tweets["tweets"].str.extractall(r'(\@\w+)')[0].reset_index()[["level_0",0]]
hash_tag = tweets["tweets"].str.extractall(r'(\#\w+)')[0].reset_index()[["level_0",0]]
lf = hash_tag.merge(accounts,left_on = "level_0",right_on = "level_0",how = "left")[["0_x","0_y"]]
rt = accounts.merge(hash_tag,left_on = "level_0",right_on = "level_0",how = "left")[["0_x","0_y"]]
lf = lf.rename(columns = {"0_y" : "accs","0_x" : "hashs"})[["hashs","accs"]]
rt = rt.rename(columns = {"0_x" : "accs","0_y" : "hashs"})[["hashs","accs"]]
newdat = pd.concat([lf,rt],axis = 0)
newdat

import networkx as nx

def connect_hash_acc(word,connect_type) :
    
    if connect_type == "hashtag_to_account" : 
        df = newdat[newdat["hashs"] == word]
        df = df[df["accs"].notnull()]
    elif connect_type == "account_to_hashtag" : 
        df = newdat[newdat["accs"] == word] 
        df = df[df["hashs"].notnull()]
        
    G  = nx.from_pandas_edgelist(df,"hashs","accs")
    plt.figure(figsize = (13,10))
    nx.draw_networkx(G,with_labels = True,font_size = 10,
                     font_color = "k",
                     font_family  = "DejaVu Sans",
                     node_shape  = "h",node_color = "b",
                     node_size = 1000,linewidths = 10,
                     edge_color = "grey",alpha = .6)
    

connect_hash_acc("@realDonaldTrump","account_to_hashtag")


# In[ ]:


connect_hash_acc("@FoxNews","account_to_hashtag")


# ## Account references by hashtag

# In[ ]:


connect_hash_acc("#MakeAmericaGreatAgain","hashtag_to_account")


# In[ ]:


connect_hash_acc("#RNCinCLE","hashtag_to_account")


# 
# ## positive word references by trump

# In[ ]:


pw_t =  tweets_trump["positive"].str.split(expand = True).stack().reset_index()[0].str.upper()
pw_d =  tweets_hillary["positive"].str.split(expand = True).stack().reset_index()[0].str.upper()
build_word_cloud(pw_t,"black","cool","positive word references by trump")
build_word_cloud(pw_d,"black","cool","positive word references by hillary")


# # negative word references by trump

# In[ ]:


nw_t =  tweets_trump["negative"].str.split(expand = True).stack().reset_index()[0].str.upper()
nw_d =  tweets_hillary["negative"].str.split(expand = True).stack().reset_index()[0].str.upper()

build_word_cloud(nw_t,"black","cool","negative word references by trump")
build_word_cloud(nw_d,"black","cool","negative word references by hillary")


# ## Sentiment of tweets by hour of day

# In[ ]:


st_hr_t = pd.crosstab(tweets_trump["hour"],tweets_trump["sentiment"])
st_hr_t = st_hr_t.apply(lambda r:r/r.sum()*100,axis = 1)

st_hr_d = pd.crosstab(tweets_hillary["hour"],tweets_hillary["sentiment"])
st_hr_d = st_hr_d.apply(lambda r:r/r.sum()*100,axis = 1)

st_hr_t.plot(kind = "bar",figsize = (14,7),color = ["r","b","g"],
              linewidth = 1,edgecolor = "w",stacked = True)
plt.legend(loc = "best",prop = {"size" : 13})
plt.title("Sentiment of tweets by hour of day - Trump")
plt.xticks(rotation = 0)
plt.ylabel("percentage")

st_hr_d.plot(kind = "bar",figsize = (14,7),color = ["r","b","g"],
              linewidth = 1,edgecolor = "w",stacked = True)
plt.legend(loc = "best",prop = {"size" : 13})
plt.title("Sentiment of tweets by hour of day - hillary")
plt.xticks(rotation = 0)
plt.ylabel("percentage")


plt.show()


# ## favorite and retweets by sentiment

# In[ ]:


import itertools
lst =  ['negative', 'positive' ,'neutral']
cs  =  ["r","g","b"]

plt.figure(figsize = (13,13))

for i,j,k in itertools.zip_longest(lst,range(len(lst)),cs) :
    plt.subplot(2,2,j+1)
    plt.scatter(x = tweets_trump[tweets_trump["sentiment"] == i]["favorite_count"],
                y = tweets_trump[tweets_trump["sentiment"] == i]["retweet_count"],
                label = "Trump",linewidth = .7,edgecolor = "w",s = 60,alpha = .7)
    plt.scatter(x = tweets_hillary[tweets_hillary["sentiment"] == i]["favorite_count"],
                y = tweets_hillary[tweets_hillary["sentiment"] == i]["retweet_count"],
                label = "Hillary",linewidth = .7,edgecolor = "w",s = 60,alpha = .7)
    plt.title(i + " - tweets")
    plt.legend(loc = "best",prop = {"size":12})
    plt.xlabel("favorite count")
    plt.ylabel("retweet count")


# ## Average retweets and favorites by sentiment

# In[ ]:


avg_fv_rts_t = tweets_trump.groupby("sentiment")[["retweet_count",
                                                "favorite_count"]].mean()
avg_fv_rts_h = tweets_hillary.groupby("sentiment")[["retweet_count",
                                                "favorite_count"]].mean()

avg_fv_rts_t.plot(kind = "bar",figsize = (12,6),linewidth = 1,edgecolor = "k")
plt.xticks(rotation = 0)
plt.ylabel("average")
plt.title("Average retweets and favorites by sentiment - Trump")

avg_fv_rts_h.plot(kind = "bar",figsize = (12,6),linewidth = 1,edgecolor = "k")
plt.xticks(rotation = 0)
plt.ylabel("average")
plt.title("Average retweets and favorites by sentiment - Hillary")

plt.show()


# ## Document term matrix

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer

def return_dtm(df,column) :
    
    documents  = df[column].tolist()
    vectorizer = CountVectorizer()
    vec        = vectorizer.fit_transform(documents)
    dtm  = pd.DataFrame(vec.toarray(),columns = vectorizer.get_feature_names())
    dtm  = df[[column,"sentiment"]].merge(dtm,left_index = True,
                                          right_index = True,how = "left")
    dtm["sentiment"]  = dtm["sentiment"].map({"neutral" : 1,"positive" : 2,
                                          "negative" : 3})  
    
    return dtm

dtm_trump    = return_dtm(tweets_trump,"text")
dtm_hillary  = return_dtm(tweets_hillary,"text")
dtm_hillary  = dtm_hillary.rename(columns = {"text_x" : "text"})



display(dtm_trump.head(3).style.set_properties(**{}).set_caption("DTM - Trump"))
display(dtm_hillary.head(3).style.set_properties(**{}).set_caption("DTM - Hillary"))


# ## preprocessing

# In[ ]:


from sklearn.model_selection import train_test_split


def split_data(dtm_df) :
    
    #dependent and independent variables
    predictors = [i for i in dtm_df.columns if i not in ["text"] + ["sentiment"]]
    target     = "sentiment"
    
    #split
    train,test = train_test_split(dtm_df,test_size = .25,
                                  stratify = dtm_df[["sentiment"]],
                                  random_state  = 123)
    
    train_X = train[predictors]
    train_Y = train[target]
    test_X  = test[predictors]
    test_Y  = test[target]
    
    return train_X,train_Y,test_X,test_Y

train_X_trp,train_Y_trp,test_X_trp,test_Y_trp = split_data(dtm_trump)
train_X_hil,train_Y_hil,test_X_hil,test_Y_hil = split_data(dtm_hillary)

#plot 
x      = [train_Y_trp,test_Y_trp,train_Y_hil,test_Y_hil]
titles = ["train_data - trump","test_data - trump",
          "train_data - hillary","test_data - hillary"]

plt.figure(figsize = (12,12))
for i,j,k in itertools.zip_longest(x,range(len(x)),titles) :
    plt.subplot(2,2,j+1)
    counts = i.value_counts().reset_index()
    counts.columns = ["sentiment","count"]
    counts["sentiment"] = counts["sentiment"].map({1 : "neutral",2 : "positive" ,
                                                    3 : "negative" }) 
    plt.pie(x = counts["count"] ,labels = counts["sentiment"],autopct  = "%1.0f%%",
            wedgeprops = {"linewidth" : 1,"edgecolor" : "black"},
            colors = sns.color_palette("prism",4))
    plt.title(k)
    


# ## classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.metrics import recall_score,precision_score,f1_score

def classifier(train_X,train_Y,test_X,test_Y) :
    rfc = RandomForestClassifier(max_depth = 1000,max_features = 2000,
                                 n_estimators = 10,random_state = 123)
    rfc.fit(train_X,train_Y)
    predictions = rfc.predict(test_X)
    
    print ("accuracy_score  : ",accuracy_score(predictions,test_Y))
    print ("recall_score    : ",recall_score(predictions,test_Y,average = "macro"))
    print ("precision_score : ",precision_score(predictions,test_Y,average = "macro"))
    print ("f1_score        : ",f1_score(predictions,test_Y,average = "macro"))

    plt.figure(figsize = (8,6))
    sns.heatmap(confusion_matrix(predictions,test_Y),annot = True,
                xticklabels= [ "neutral","positive","negative"],
                yticklabels= [ "neutral","positive","negative"],
                fmt = "d",linecolor = "w",linewidths = 2)
    plt.title("confusion matrix")
    plt.show()

#classify trump tweets
classifier(train_X_trp,train_Y_trp,test_X_trp,test_Y_trp)


# In[ ]:


#classify hillary tweets
classifier(train_X_hil,train_Y_hil,test_X_hil,test_Y_hil)


# ## Network analysis of tweets

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer

def network_tweets(df,frequency,color,title) :
    #documents
    documents  = df[df["lang"] == "English"]["text"].tolist()
    vectorizer = CountVectorizer()
    vec        = vectorizer.fit_transform(documents)
    vec_t      = vectorizer.fit_transform(documents).transpose()
    
    #adjecency matrix for words
    adj_mat    = pd.DataFrame((vec_t * vec).toarray(),
                              columns = vectorizer.get_feature_names(),
                              index    = vectorizer.get_feature_names()
                             )
    # #stacking combinations
    adj_mat_stack   = adj_mat.stack().reset_index()
    adj_mat_stack.columns = ["link_1","link_2","count"]
    
    #drop same word combinations
    adj_mat_stack   = adj_mat_stack[adj_mat_stack["link_1"] !=
                                    adj_mat_stack["link_2"]] 
    
    #subset dataframe with combination count greater than 25 times
    network_sub = adj_mat_stack[adj_mat_stack["count"] > frequency]
    
    #plot network
    H = nx.from_pandas_edgelist(network_sub,"link_1","link_2",["count"],
                                create_using = nx.DiGraph())

    ax = plt.figure(figsize = (11,11))
    nx.draw(H,with_labels = True,alpha = .7,node_shape = "H",
            width = 1,node_color = color,
            font_weight = "bold",style = "solid", arrowsize = 15 ,
            font_color = "white",linewidths = 10,edge_color = "grey",
            node_size = 1300,pos = nx.kamada_kawai_layout(H))
    plt.title(title,color = "white")
    ax.set_facecolor("k")
    
network_tweets(tweets_trump,25,"#FF3300","Network analysis of tweet words - Trump")


# In[ ]:


network_tweets(tweets_hillary,25,"#6666FF","Network analysis of tweet words - Hillary")

