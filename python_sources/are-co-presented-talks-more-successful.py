#!/usr/bin/env python
# coding: utf-8

# # Are Co-presented talks more successful?

# If you've been to a conference lately, chances are you've seen a presenation given by two sepakers. Co-presenting has become very popular especially in the tech community. Here, one person would be delivering the content of the talk while the second person would showcase a service/software. Other examples of this trend would be the "fireside chat", where one person would interview the second person with the audience being the fly on the wall.
# 
# The assumption is that having two people present a talk will break up monologues and hence keep the audience attention for longer without appearing to be longwinded. It is also expected to be more fun as the presenters essentially conduct a little roleplaying exercise. Finally, having two or more brains on stage should help delivering an clear and informative talk as the audience hears different parts from the appropriate experts. 
# 
# In this notebook I will investigate the intuitive hypotheses that **co-presented talks are more fun, more informative and less longwinded**. I therefore will investigate these attributes between talks with one versus two or more speakers in recent TED talks, specifically focusing on tech-talks.  
# 

# In[ ]:


import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pandas.io.json import json_normalize


# ## The TED Dataset
# The data is shared under the Creative Commons License and hosted on Kaggle: https://www.kaggle.com/rounakbanik/ted-talks
# 
# An indept overview of this data was conducted by Rounak Banik https://www.kaggle.com/rounakbanik/ted-data-analysis, I therefore focus straight on the featurs for my hypothesis.

# In[ ]:


df = pd.read_csv('../input/ted_main.csv')
# Filtering out a talk that is mostly a video
df=df[df["event"]!="TED-Ed"]
df.head()


# ## Used Features 
# 
# * **name**: The official name of the TED Talk. Includes the title and the speaker.
# * **speaker_occupation:** The occupation of the main speaker.
# * **num_speaker:** The number of speakers in the talk.
# * **ratings:** A stringified dictionary of the various ratings given to the talk (inspiring, fascinating, jaw dropping, etc.)
# * **views:** The number of views on the talk.
# 

# ## Definition of co-presented talks

# In[ ]:


df['num_speaker_cap'] = "1"
df.loc[df.num_speaker>1,'num_speaker_cap']=">1"

df.groupby('num_speaker_cap').size()


# In[ ]:


58/2492*100


# ### Observations
# * Co-presented talks are fairly rare, making only about 2% of all talks

# # Tech talks have a higher ratio of co-presenters

# In[ ]:


df['type']="non-tech"
df['speaker_occupation'].fillna("",inplace=True)
a=["tech","research","computer","science","Computer scientist", "Researcher", "inventor", "Engineer", "Inventor", "Researcher", "futurist", "Data", "Science", "Biologist", "biologist", "Technologist", "Neuroscientist", "psychologist", "Neuroscientist", "Ecologist", "health", "psychologist", "Health"]
#a=["tech","research","computer","science","Computer scientist", "Researcher", "Engineer", "Researcher", "futurist", "Data", "Science", "Biologist", "biologist", "Technologist", "Neuroscientist", "psychologist", "Neuroscientist", "Ecologist", "health", "psychologist", "Health"]
df.loc[df.speaker_occupation.str.contains("|".join(a)),'type']="tech"

df.groupby(['type','num_speaker_cap']).size()


# In[ ]:


m=df.groupby(['type','num_speaker_cap']).size().unstack()
(m.div(m.sum(axis=1),axis=0)*100).round()


# **Observations**
# * Co-presenting is more popular amongst tech talks (5% vs 2%).

# # Co-presented talks attrackt fewer views
# 
# Let's fist find out if co-presented talks get more or less views than talks with single presenters. This will help assess what people's expectation is for the differnt presentation styles, as we can treat views as a direct measurement of 'conversion rate', i.e. how may people choose to give the talk a go based on the title and teaser image. 

# In[ ]:


g = sns.boxplot(x="num_speaker_cap", y="views", data=df, palette="PRGn")
g.set(ylim=(0, 4000000))


# In[ ]:


viewdiff=df.groupby('num_speaker_cap').agg({'views':np.median})
viewdiff


# Is the difference statistically significant?

# In[ ]:


stats.mannwhitneyu(df[df.num_speaker>1].views,df[df.num_speaker==1].views)


# In[ ]:


#(1.707530e+06-1.301605e+06)/1.707530e+06*100
(1131452.5-873904.0)/1131452.5*100


# This is a 23% reduction in views for co-presented talks.
# 
# Let's now do the same analysis for the tech talks.

# ## Co-presented tech talks attrackt similar numbers of views

# In[ ]:


g = sns.boxplot(x="num_speaker_cap", y="views", data=df[df['type']=="tech"], palette="PRGn")
g.set(ylim=(0, 4000000))


# In[ ]:


df_t=df[df['type']=="tech"]
stats.mannwhitneyu(df_t[df_t.num_speaker>1].views,df_t[df_t.num_speaker==1].views)


# ### Observations
# 
# * From this it would seem that people may shy away from co-presented talks (23% fewer views for co-presented talks, *p*-value=0.006)
# * While co-presented tech talks show a similar trend the difference is not statistically significant anymore.
# * The dataset is quite imbalanced, so this may be driven by the fact that mongst 2000+ talks there are more familiar names that result in more views because of their populatiry (think Hans Rosling, Bill Gates,...).
# 
# So let's have a look at the actual content, i.e. how much people liked co-resented talk once they actually watched them.

# # Ratings
# 
# Each talk was rated by people along 14 dimensions (funny, courageous,...). This element is a json object.  

# In[ ]:


df['ratings']=df['ratings'].str.replace("'",'"')
pd.read_json(df['ratings'].iloc[1])[['name','count']]


# Let's convert the JSON object into pandas columns for all talks. We need to normalize by their views and convert them into z-scores so that these dimensions are comparable. 

# In[ ]:


df=df.merge(df.ratings.apply(lambda x: pd.Series(pd.read_json(x)['count'].values,index=pd.read_json(x)['name'])), 
    left_index=True, right_index=True)
#normalize by Views
for i in range(19,33):
    df.iloc[:,i]=df.iloc[:,i]/df["views"]


# In[ ]:


# get a copy for the tech talks
df_a=df.copy()
df_t=df[df['type']=="tech"].copy()
for i in range(19,33):
    df_a.iloc[:,i]=(df_a.iloc[:,i] - df_a.iloc[:,i].mean())/df_a.iloc[:,i].std()
    df_t.iloc[:,i]=(df_t.iloc[:,i] - df_t.iloc[:,i].mean())/df_t.iloc[:,i].std()


#  ## Most longwinded talks of all time
# Let's get a feel for the data. Our hypothesis is that co-presentation makes talks less longwinded. So what are the most longwinded talks of all times?

# In[ ]:


l_a=df_a[['title', 'num_speaker_cap', 'type', 'main_speaker', 'views', 'Longwinded']].sort_values('Longwinded', ascending=False)
l_a[:10]


# ### Most longwinded co-presented talks of all times

# In[ ]:


l_a_c=df_a[df_a.num_speaker>1][['title', 'num_speaker_cap', 'type', 'main_speaker',"speaker_occupation", 'views', 'Longwinded']].sort_values('Longwinded', ascending=False)
l_a_c[:10]


# ### Most longwinded tech talk of all times

# In[ ]:


l_t=df_t[['title', 'num_speaker_cap', 'type', 'main_speaker',"speaker_occupation", 'views', 'Longwinded']].sort_values('Longwinded', ascending=False)
l_t[:10]


# ### Most longwinded co-presented tech talk of all times

# In[ ]:


l_t_c=df_t[df_t.num_speaker>1][['title', 'num_speaker_cap', 'type', 'main_speaker',"speaker_occupation", 'views', 'Longwinded']].sort_values('Longwinded', ascending=False)
l_t_c[:10]


# ### Observations
# * The most longwinded co-presented talks are tech talk.  

# ## Plotting the data
# Let's plot the data. We therefore need to first melt it for seaborn to be able to plot it as facet grid.

# In[ ]:


#df_plot=df.iloc[:,[6,12,14,16,17,18]+list(range(19,33))]
df_a_plot=df_a.iloc[:,[17,18]+list(range(19,33))].melt(id_vars=['num_speaker_cap','type'])
df_t_plot=df_t.iloc[:,[17,18]+list(range(19,33))].melt(id_vars=['num_speaker_cap','type'])
df_a_plot.iloc[list(range(1,10))]


# Significance scores for each boxplot.

# In[ ]:


def annotate(g,sign):
    for i in range(0,len(g.axes)):
        col=sns.xkcd_rgb["pale red"]
        if sign[i]>0.05:
            col=sns.xkcd_rgb["denim blue"]
        # significance
        g.axes[i].text(0.1,-2,'p-value='+"{:.2e}".format(sign[i]), color=col)
        #delta
        #g.axes[i].text(0.1,600,'delta='+"{:.2e}".format(delta[i]), color=col)


# ## Co-presented talks are significantly more Beautiful
# The box plot below holds all rating dimensions and compares single speaker talks with co-presented talks. Also listed is the p-value with red indicating statistical significance (<0.05), bonferroni corrected for multiple hypothesis testing. 

# In[ ]:


sign_a=[]
for i in range(19,33):
    sign_a.append(stats.mannwhitneyu(df_a[df_a.num_speaker>1].iloc[:,i],df_a[df_a.num_speaker==1].iloc[:,i]).pvalue*(33-19))
sign_a


# In[ ]:


g = sns.FacetGrid(df_a_plot, col="variable",  col_wrap=5)
g = g.map(sns.boxplot, "num_speaker_cap", "value")
g.set(ylim=(-2.5, 2))
g.set(xlabel='speaker number', ylabel='normalized votes')
annotate(g,sign_a)
plt.subplots_adjust(top=0.9)
g.fig.suptitle('All talks')


# ### Observations
# * Co-presented talks are significantly more Beautiful
# * However, they are also significantly less Fascinating, Informative, Inspiring and Persuasive

# ## Co-presented tech talks are trending towards more Jaw-dropping
# The box plot below holds all rating dimensions and compares single speaker talks with co-presented talks. Also listed is the p-value with red indicating statistical significance (<0.05), bonferroni corrected for multiple hypothesis testing. 

# In[ ]:


sign_t=[]
for i in range(19,33):
    sign_t.append(stats.mannwhitneyu(df_t[df_t.num_speaker>1].iloc[:,i],df_t[df_t.num_speaker==1].iloc[:,i]).pvalue*(33-19))
sign_t


# In[ ]:


g = sns.FacetGrid(df_t_plot[df_t_plot.type=="tech"], col="variable",  col_wrap=5)
g = g.map(sns.boxplot, "num_speaker_cap", "value")
g.set(ylim=(-2.5, 2))
g.set(xlabel='speaker number', ylabel='normalized votes')
annotate(g,sign_t)
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Tech talks')


# ### Observations
# * Co-presented talks are more Jaw-dropping, however this is not significant anymore after bonferroni correction. 
# * However, they are also significantly more Obnoxious.

# # Finding an overall score to rate talks
# We've seen that some ratings were better in co-presented talks, while others were worse. So let's generate a score combining all dimensions. 

# In[ ]:


def posneg (df):
    df["pos"]=(df["Funny"]+df["Beautiful"]+df["Ingenious"]+df["Courageous"]+df["Informative"]+df["Fascinating"]+df["Persuasive"]+df["Jaw-dropping"]+df["Inspiring"])/9
    df["neg"]=(df["Longwinded"]+df["Confusing"]+df["Unconvincing"]+df["Obnoxious"]/4)
    #zscore
    #df["pos"]=(df["pos"] - df["pos"].mean())/df["pos"].std()
    #df["neg"]=(df["neg"] - df["neg"].mean())/df["neg"].std()
    df["pos_vs_neg"]=df["pos"]-df["neg"]
    #df["pos_vs_neg"]=(df["pos_vs_neg"] - df["pos_vs_neg"].mean())/df["pos_vs_neg"].std()


# ## Co-presented tech talks are not significantly better over all than single speaker talks
# Let's compare this summary score, "Goodness", between single speaker and co-presented talks.

# In[ ]:


#Significance
posneg(df_a)
p_a=stats.mannwhitneyu(df_a[df_a.num_speaker>1].pos_vs_neg,df_a[df_a.num_speaker==1].pos_vs_neg).pvalue
print("Mann Whitney ransum p-value: {:.2f}".format(p_a))


# In[ ]:


g = sns.boxplot(x="num_speaker_cap", y="pos_vs_neg", data=df_a, palette="PRGn")
g.set_title("All talks")
g.set(ylim=(-4, +4))
g.axes.text(0.35,-3,'p-value='+"{:.2f}".format(p_a))
g.set(xlabel='speaker number', ylabel='"Goodness" score')


# ## Co-presented tech talks are rated significantly worse over all

# In[ ]:


posneg(df_t)
p_t=stats.mannwhitneyu(df_t[df_t.num_speaker>1].pos_vs_neg,df_t[df_t.num_speaker==1].pos_vs_neg).pvalue
p_t


# In[ ]:


g = sns.boxplot(x="num_speaker_cap", y="pos_vs_neg", data=df_t, palette="PRGn")
g.set_title("Tech talks")
g.set(ylim=(-4, +4))
g.axes.text(0.35,-3,'p-value='+"{:.2f}".format(p_t))
g.set(xlabel='speaker number', ylabel='"Goodness" score')


# ### Observations
# * Co-presented tech talks are significantly worse than single presenter talks.
# * While they can be more Jaw-dropping and there is a trend for them being more funny, they are also more Confusing, Obnoxious, Unconvincing and less Informative, Persuasive.

# ## Overall best co-presented talks of all times

# In[ ]:


j_a=df_a[df_a.num_speaker>1][['title', 'num_speaker', 'type', 'main_speaker', 'views', 'pos_vs_neg']].sort_values('pos_vs_neg', ascending=False)
j_a[:10]


# ## Magic happens at really good co-presented talks
# 

# In[ ]:


g=sns.lmplot(x="Jaw-dropping", y="Beautiful",data=df_a, hue="num_speaker_cap", fit_reg=False)
g.set(xscale="symlog", yscale="symlog")
g.set(ylim=(-1, 10))


# ### Great co-presented talks nailing two dimensions is even more pronounced in tech talks

# In[ ]:


g=sns.lmplot(x="Jaw-dropping", y="Beautiful",data=df_t, hue="num_speaker_cap", fit_reg=False)
g.set(xscale="symlog", yscale="symlog")
g.set(ylim=(-1, 10));


# ### Focusing on the top 10 for Jaw-dropping and Beautiful tech talks

# In[ ]:


top=10
top_j=df_t[['event','title','main_speaker','num_speaker_cap', 'Jaw-dropping','Beautiful']].sort_values('Jaw-dropping', ascending=False)[:top]
top_f=df_t[['event','title','main_speaker','num_speaker_cap', 'Jaw-dropping','Beautiful']].sort_values('Beautiful', ascending=False)[:top]

top_jf=pd.concat([top_j,top_f])
top_jf.drop_duplicates(inplace=True)
top_jf


# In[ ]:


g=sns.lmplot(x="Jaw-dropping", y="Beautiful",data=top_jf, hue="num_speaker_cap", fit_reg=False)
#g.set(xscale="symlog", yscale="symlog")


# # Conclusion
# * Presenting talks by two or more speakers are rare (2%) but more common amongst the tech community (5%).
# * The audience seems to not be drawn to co-presented talks (23% fewer number of views for co-presented talks).
# * This may be because co-presented talks seem to be significantly less Fascinating, Informative, Inspiring and Persuasive for general talks and significantly more Obnoxious for tech talks.
# * However, they are also significantly more Beautiful for general talks and more Jaw-dropping for tech talks. 
# * While the positives and negatives cancel each other out overall, the co-presented tech talks have a worse score across all dimensions than single speaker talks.
# * When co-presented talks get it right, however, they excel at both being more Beautiful and Jaw-dropping than single speaker talks.
# 
# So overall, I'd say co-presenting is a skill that needs twice the mastering than giving a talk alone. I therefore would like to see more organizers to encourage and people to pratice because when you do get it right it is Beautiful and Jaw-dropping.
