#!/usr/bin/env python
# coding: utf-8

# ## Why  Feature Engineering ?
# It might be obvious that model learns the patterns and interactions while it trains. So, how does the engineering comes into the place. Also, there is a line where we should leave some interactions for models to find out and not hand label features. The need for feature engineering;
# 
# ![Image](https://www.analyticsindiamag.com/wp-content/uploads/2018/01/data-cleaning.png)
# 
# <sup>Source:  analyticsindiamag </sup>
# 
# 1.  Helps the model to catch those exceptions when you engineer a significant interaction
# 2. The model converges faster if you happen to find good set of features
# 3. When you have new source of information. A chance to make better model. You engineer features!
# 
# Let's start with basic imports and loading the dataset...
# 

# In[ ]:


import os
import json
import string
import numpy as np
import pandas as pd
import plotly.offline as py
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go

from pandas.io.json import json_normalize
from plotly import tools
py.init_notebook_mode(connected=True)
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
color = sns.color_palette()
np.random.seed(13)
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Pause And Think! 
# 
# This is one of the stategies I adopt. I avoid looking into the head frame directly and instead focus on the competition problem. One of the reasons doing so is keeping my mind in a calm state where I could imagine what features I could think of or what insicere comments would read like?
# 
# ![Ask](http://www.gregorypouy.com/wp-content/uploads/2015/12/what-how-why.jpg)
# 
# 
# 
# Few pointers I have is;
# 1. Are they toxic with targetted words on race/region/religion? 
# 2. Do they contain obscene words ? Are these questions long or short?
# 3. And, dividing them into clusters will help my model predict - what cluster of insincerity does an insincere question lies in .... etc
# 
# So, let's check the head frame now..
# 

# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print("Train shape : ", train_df.shape)
print("Test shape : ", test_df.shape)
train_df.head(8)


# Let's print  insincere comments and have a look at few examples to get ideas for feature engineering

# In[ ]:


sincere_questions = train_df[train_df['target'] == 0]
insincere_questions = train_df[train_df['target'] == 1]
insincere_questions.tail(5)


# Now, that we have had a look. Let's try to create features for our question. 

# ## Average Length of Sincere vs. Insincere

# In[ ]:


sincere_questions['length'] = sincere_questions['question_text'].apply(lambda x : len(x))
insincere_questions['length'] = insincere_questions['question_text'].apply(lambda x : len(x))
train_df['length'] = train_df['question_text'].apply(lambda x : len(x))


# In[ ]:


sincere = go.Box(y=sincere_questions['length'].values, name = 'Sincere Questions', boxmean=True)
insincere = go.Box(y=insincere_questions['length'].values, name = 'Insincere Questions', boxmean=True)
data = [sincere, insincere]
layout = go.Layout(title = "Average Length of Sincere vs. Insincere")
fig = go.Figure(data=data,layout=layout)
py.iplot(fig)


# Seems like length doesn't explain insincerity but certainly has some information to add to the context. I am curious about the highest lenght question lets check it!

# In[ ]:


print(train_df.iloc[443216]['question_text'])
train_df[train_df['length'] == train_df['length'].max()]


# Oh, looks like someone was in dire need of completing their math homework. But, its surprising to see it marked as insincere. However, its important to keep in mind that there is significant noise in our data
# 
# From [data page](http://https://www.kaggle.com/c/quora-insincere-questions-classification/data);
# 
# > Note that the distribution of questions in the dataset should not be taken to be representative of the distribution of questions asked on Quora. This is, in part, because of the combination of sampling procedures and sanitization measures that have been applied to the final dataset.
# 
# 

# Now, let us create bulk of these common features; 
# 

# ## The Meta Features Based On Word/Character

# In[ ]:


from tqdm import tqdm # I love this handy tool! 
print(">> Generating Count Based And Demographical Features")
for df in ([train_df]):
    df['length'] = df['question_text'].apply(lambda x : len(x))
    df['capitals'] = df['question_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
    df['caps_vs_length'] = df.apply(lambda row: float(row['capitals'])/float(row['length']),axis=1)
    df['num_exclamation_marks'] = df['question_text'].apply(lambda comment: comment.count('!'))
    df['num_question_marks'] = df['question_text'].apply(lambda comment: comment.count('?'))
    df['num_punctuation'] = df['question_text'].apply(lambda comment: sum(comment.count(w) for w in '.,;:'))
    df['num_symbols'] = df['question_text'].apply(lambda comment: sum(comment.count(w) for w in '*&$%'))
    df['num_words'] = df['question_text'].apply(lambda comment: len(comment.split()))
    df['num_unique_words'] = df['question_text'].apply(lambda comment: len(set(w for w in comment.split())))
    df['words_vs_unique'] = df['num_unique_words'] / df['num_words']
    df['num_smilies'] = df['question_text'].apply(lambda comment: sum(comment.count(w) for w in (':-)', ':)', ';-)', ';)')))
    df['num_sad'] = df['question_text'].apply(lambda comment: sum(comment.count(w) for w in (':-<', ':()', ';-()', ';(')))


# Let's have a glance at new features ... 

# In[ ]:


train_df[train_df.columns[2:]].head(8)


#  ## Oh my! A Bad Word
# 
# We found a significant bad words in the insincere comments. Here I am using few profanity words list Let's device and see count for each and ratio features for the same. 

# In[ ]:


# List Of Bad Words by Google-Profanity Words 
bad_words = ['cockknocker', 'n1gger', 'ing', 'fukker', 'nympho', 'fcuking', 'gook', 'freex', 'arschloch', 'fistfucked', 'chinc', 'raunch', 'fellatio', 'splooge', 'nutsack', 'lmfao', 'wigger', 'bastard', 'asses', 'fistfuckings', 'blue', 'waffle', 'beeyotch', 'pissin', 'dominatrix', 'fisting', 'vullva', 'paki', 'cyberfucker', 'chuj', 'penuus', 'masturbate', 'b00b*', 'fuks', 'sucked', 'fuckingshitmotherfucker', 'feces', 'panty', 'coital', 'wh00r.', 'whore', 'condom', 'hells', 'foreskin', 'wanker', 'hoer', 'sh1tz', 'shittings', 'wtf', 'recktum', 'dick*', 'pr0n', 'pasty', 'spik', 'phukked', 'assfuck', 'xxx', 'nigger*', 'ugly', 's_h_i_t', 'mamhoon', 'pornos', 'masterbates', 'mothafucks', 'Mother', 'Fukkah', 'chink', 'pussy', 'palace', 'azazel', 'fistfucking', 'ass-fucker', 'shag', 'chincs', 'duche', 'orgies', 'vag1na', 'molest', 'bollock', 'a-hole', 'seduce', 'Cock*', 'dog-fucker', 'shitz', 'Mother', 'Fucker', 'penial', 'biatch', 'junky', 'orifice', '5hit', 'kunilingus', 'cuntbag', 'hump', 'butt', 'fuck', 'titwank', 'schaffer', 'cracker', 'f.u.c.k', 'breasts', 'd1ld0', 'polac', 'boobs', 'ritard', 'fuckup', 'rape', 'hard', 'on', 'skanks', 'coksucka', 'cl1t', 'herpy', 's.o.b.', 'Motha', 'Fucker', 'penus', 'Fukker', 'p.u.s.s.y.', 'faggitt', 'b!tch', 'doosh', 'titty', 'pr1k', 'r-tard', 'gigolo', 'perse', 'lezzies', 'bollock*', 'pedophiliac', 'Ass', 'Monkey', 'mothafucker', 'amcik', 'b*tch', 'beaner', 'masterbat*', 'fucka', 'phuk', 'menses', 'pedophile', 'climax', 'cocksucking', 'fingerfucked', 'asswhole', 'basterdz', 'cahone', 'ahole', 'dickflipper', 'diligaf', 'Lesbian', 'sperm', 'pisser', 'dykes', 'Skanky', 'puuker', 'gtfo', 'orgasim', 'd0ng', 'testicle*', 'pen1s', 'piss-off', '@$$', 'fuck', 'trophy', 'arse*', 'fag', 'organ', 'potty', 'queerz', 'fannybandit', 'muthafuckaz', 'booger', 'pussypounder', 'titt', 'fuckoff', 'bootee', 'schlong', 'spunk', 'rumprammer', 'weed', 'bi7ch', 'pusse', 'blow', 'job', 'kusi*', 'assbanged', 'dumbass', 'kunts', 'chraa', 'cock', 'sucker', 'l3i+ch', 'cabron', 'arrse', 'cnut', 'how', 'to', 'murdep', 'fcuk', 'phuked', 'gang-bang', 'kuksuger', 'mothafuckers', 'ghey', 'clit', 'licker', 'feg', 'ma5terbate', 'd0uche', 'pcp', 'ejaculate', 'nigur', 'clits', 'd0uch3', 'b00bs', 'fucked', 'assbang', 'mutha', 'goddamned', 'cazzo', 'lmao', 'godamn', 'kill', 'coon', 'penis-breath', 'kyke', 'heshe', 'homo', 'tawdry', 'pissing', 'cumshot', 'motherfucker', 'menstruation', 'n1gr', 'rectus', 'oral', 'twats', 'scrot', 'God', 'damn', 'jerk', 'nigga', 'motherfuckin', 'kawk', 'homey', 'hooters', 'rump', 'dickheads', 'scrud', 'fist', 'fuck', 'carpet', 'muncher', 'cipa', 'cocaine', 'fanyy', 'frigga', 'massa', '5h1t', 'brassiere', 'inbred', 'spooge', 'shitface', 'tush', 'Fuken', 'boiolas', 'fuckass', 'wop*', 'cuntlick', 'fucker', 'bodily', 'bullshits', 'hom0', 'sumofabiatch', 'jackass', 'dilld0', 'puuke', 'cums', 'pakie', 'cock-sucker', 'pubic', 'pron', 'puta', 'penas', 'weiner', 'vaj1na', 'mthrfucker', 'souse', 'loin', 'clitoris', 'f.ck', 'dickface', 'rectal', 'whored', 'bookie', 'chota', 'bags', 'sh!t', 'pornography', 'spick', 'seamen', 'Phukker', 'beef', 'curtain', 'eat', 'hair', 'pie', 'mother', 'fucker', 'faigt', 'yeasty', 'Clit', 'kraut', 'CockSucker', 'Ekrem*', 'screwing', 'scrote', 'fubar', 'knob', 'end', 'sleazy', 'dickwhipper', 'ass', 'fuck', 'fellate', 'lesbos', 'nobjokey', 'dogging', 'fuck', 'hole', 'hymen', 'damn', 'dego', 'sphencter', 'queef*', 'gaylord', 'va1jina', 'a55', 'fuck', 'douchebag', 'blowjob', 'mibun', 'fucking', 'dago', 'heroin', 'tw4t', 'raper', 'muff', 'fitt*', 'wetback*', 'mo-fo', 'fuk*', 'klootzak', 'sux', 'damnit', 'pimmel', 'assh0lez', 'cntz', 'fux', 'gonads', 'bullshit', 'nigg3r', 'fack', 'weewee', 'shi+', 'shithead', 'pecker', 'Shytty', 'wh0re', 'a2m', 'kkk', 'penetration', 'kike', 'naked', 'kooch', 'ejaculation', 'bang', 'hoare', 'jap', 'foad', 'queef', 'buttwipe', 'Shity', 'dildo', 'dickripper', 'crackwhore', 'beaver', 'kum', 'sh!+', 'qweers', 'cocksuka', 'sexy', 'masterbating', 'peeenus', 'gays', 'cocksucks', 'b17ch', 'nad', 'j3rk0ff', 'fannyflaps', 'God-damned', 'masterbate', 'erotic', 'sadism', 'turd', 'flipping', 'the', 'bird', 'schizo', 'whiz', 'fagg1t', 'cop', 'some', 'wood', 'banger', 'Shyty', 'f', 'you', 'scag', 'soused', 'scank', 'clitorus', 'kumming', 'quim', 'penis', 'bestial', 'bimbo', 'gfy', 'spiks', 'shitings', 'phuking', 'paddy', 'mulkku', 'anal', 'leakage', 'bestiality', 'smegma', 'bull', 'shit', 'pillu*', 'schmuck', 'cuntsicle', 'fistfucker', 'shitdick', 'dirsa', 'm0f0']
print(">> Words in bad_word list:", len(bad_words))


# In[ ]:


print(">> Generating Features on Bad Words")
for df in ([train_df]):
    df["badwordcount"] = df['question_text'].apply(lambda comment: sum(comment.count(w) for w in bad_words))
    df['num_chars'] =    df['question_text'].apply(len)
    df["normchar_badwords"] = df["badwordcount"]/df['num_chars']
    df["normword_badwords"] = df["badwordcount"]/df['num_words']


# Checking the bad_word features...

# In[ ]:


train_df[['badwordcount','num_chars','normchar_badwords','normword_badwords']].head(8)


# ## Tagging Parts Of Speech And More Feature Engineering..
# 
# I suspect that the insincere questions have significant adverbs/adjective that makes them toxic. I am hopeful that these features might model understand various POS structures in the question_text
# 
# 
# ![POS](https://cdn-images-1.medium.com/max/1600/1*fRjvBbgzo90x0MZdXZT82A.png)

# In[ ]:


import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

def tag_part_of_speech(text):
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    pos_list = pos_tag(text_splited)
    noun_count = len([w for w in pos_list if w[1] in ('NN','NNP','NNPS','NNS')])
    adjective_count = len([w for w in pos_list if w[1] in ('JJ','JJR','JJS')])
    verb_count = len([w for w in pos_list if w[1] in ('VB','VBD','VBG','VBN','VBP','VBZ')])
    return[noun_count, adjective_count, verb_count]


# In[ ]:


print(">> Generating POS Features")
for df in ([train_df]):
    df['nouns'], df['adjectives'], df['verbs'] = zip(*df['question_text'].apply(
        lambda comment: tag_part_of_speech(comment)))
    df['nouns_vs_length'] = df['nouns'] / df['length']
    df['adjectives_vs_length'] = df['adjectives'] / df['length']
    df['verbs_vs_length'] = df['verbs'] /df['length']
    df['nouns_vs_words'] = df['nouns'] / df['num_words']
    df['adjectives_vs_words'] = df['adjectives'] / df['num_words']
    df['verbs_vs_words'] = df['verbs'] / df['num_words']
    # More Handy Features
    df["count_words_title"] = df["question_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
    df["mean_word_len"] = df["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    df['punct_percent']= df['num_punctuation']*100/df['num_words']


# In[ ]:


train_df[['nouns','nouns_vs_length','adjectives_vs_length','verbs_vs_length','nouns_vs_words','adjectives_vs_words','verbs_vs_words']].head(8)


# ## Correlation Matrix

# In[ ]:


f, ax = plt.subplots(figsize= [20,15])
sns.heatmap(train_df.drop(['qid','question_text'], axis=1).corr(), annot=True, fmt=".2f", ax=ax, 
            cbar_kws={'label': 'Correlation Coefficient'}, cmap='viridis')
ax.set_title("Correlation Matrix for Insincerity and New Features", fontsize=18)
plt.show()


# ## Feedback And More...
# * Highly appreciate your feedback. 
# * Do share your feature engineering ideas and I shall see them implement in upcoming versions of this kernel
# 
# 
# ### Todo 
# * Show tutorial to use meta-features in XGBoost And LSTM Models
# * More Features And Engineering...
