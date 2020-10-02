#!/usr/bin/env python
# coding: utf-8

# ## STAR WARS - scripts analysis:
# 
# #First part is realised based on: https://www.kaggle.com/gulsahdemiryurek/star-wars-script-analysis 
# #Great notebook!

# ![](https://i.ibb.co/rvK9q0D/Darth-Revan.jpg)

# # Darth Revan

# ### Imports

# In[ ]:


import numpy as np
import pandas as pd
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
from PIL import Image
import plotly.plotly as py
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
from plotly import tools
import re
import nltk
from nltk.corpus import stopwords
import nltk as nlp


# In[ ]:


episodeIV = pd.read_csv('../input/star-wars-movie-scripts/SW_EpisodeIV.txt', delim_whitespace=True, names=["index","character","dialogue"] ,header = None)
episodeV = pd.read_csv('../input/star-wars-movie-scripts/SW_EpisodeV.txt', delim_whitespace=True, names=["index","character","dialogue"] ,header = None)
episodeVI = pd.read_csv('../input/star-wars-movie-scripts/SW_EpisodeVI.txt', delim_whitespace=True, names=["index","character","dialogue"] ,header = None)


# In[ ]:


episodeIV.drop(0,inplace=True)
episodeV.drop(0,inplace=True)
episodeVI.drop(0,inplace=True)
episodeIV.drop(["index"],axis=1,inplace=True)
episodeV.drop(["index"],axis=1,inplace=True)
episodeVI.drop(["index"],axis=1,inplace=True)


# In[ ]:


script_numIV=pd.DataFrame(episodeIV.character.value_counts()).iloc[:20]
script_numV=pd.DataFrame(episodeV.character.value_counts()).iloc[:20]
script_numVI=pd.DataFrame(episodeVI.character.value_counts()).iloc[:20]


# ![](https://cnet2.cbsistatic.com/img/3XDggbjQBCMg5Cvc-j5f7HxFsJ0=/1200x675/2019/11/12/d8253cd2-87f8-414e-81b7-a174196c040a/greedo-1.jpg)

# In[ ]:


greedo = episodeIV.loc[episodeIV['character']=='GREEDO']
greedo


# In[ ]:


trace = go.Bar(y=script_numIV.character, x=script_numIV.index,  marker=dict(color="crimson",line=dict(color='black', width=2)),opacity=0.75)
trace1 = go.Bar(y=script_numV.character,x=script_numV.index,marker=dict(color="blue",line=dict(color='black', width=2)),opacity=0.75)
trace2 = go.Bar(y=script_numVI.character, x=script_numV.index,marker=dict(color="green",line=dict(color='black', width=2)),opacity=0.75)


fig = tools.make_subplots(rows=3, cols=1,horizontal_spacing=1, subplot_titles=("A New Hope","The Empire Strikes Back","Return of The Jedi"))
 
fig.append_trace(trace, 1, 1)
fig.append_trace(trace1, 2, 1)
fig.append_trace(trace2, 3, 1)

fig['layout'].update(showlegend=False ,height=800,title="Number of Dialogues According to Character",paper_bgcolor='rgb(248, 248, 255)',
    plot_bgcolor='rgb(248, 248, 255)')


iplot(fig)


# In[ ]:


episodeIV["episode"]="A New Hope"
episodeV["episode"]="The Empire Strikes Back"
episodeVI["episode"]="Return of The Jedi"
data=pd.concat([episodeIV,episodeV,episodeVI],axis=0,ignore_index=True)


# In[ ]:


description_list=[]
for description in data.dialogue:
    description=re.sub("[^a-zA-Z]", " ", description)
    description=description.lower()
    description=nltk.word_tokenize(description)
    description=[word for word in description if not word in set(stopwords.words("english"))]
    lemma=nlp.WordNetLemmatizer()
    description=[lemma.lemmatize(word) for word in description]
    description=" ".join(description)
    description_list.append(description)


# In[ ]:


data["new_script"]=description_list
data


# In[ ]:


luke=data[data.character=="LUKE"]
yoda=data[data.character=="YODA"]
han=data[data.character=="HAN"]
vader=data[data.character=="VADER"]


# In[ ]:


wave_mask_yoda = np.array(Image.open("../input/star-wars-movie-scripts/wordcloud_masks/yoda.png"))
wave_mask_vader= np.array(Image.open("../input/star-wars-movie-scripts/wordcloud_masks/yoda.png"))
wave_mask_rebel= np.array(Image.open("../input/star-wars-movie-scripts/wordcloud_masks/yoda.png"))


# In[ ]:


plt.subplots(figsize=(15,15))
stopwords= set(STOPWORDS)
wordcloud = WordCloud(mask=wave_mask_vader,background_color="black",colormap="Reds" ,contour_width=2, contour_color="gray",
                      width=950,
                          height=950
                         ).generate(" ".join(vader.new_script))

plt.imshow(wordcloud ,interpolation='bilinear')
plt.axis('off')
plt.savefig('graph.png')

plt.show()


# In[ ]:


plt.subplots(figsize=(15,15))
stopwords= set(STOPWORDS)
wordcloud = WordCloud(mask=wave_mask_yoda,background_color="black",contour_width=3, contour_color="olivedrab",colormap="Greens",
                      stopwords=stopwords,   
                      width=950,
                          height=950
                         ).generate(" ".join(yoda.new_script))

plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('graph.png')

plt.show()


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
max_features = 400

count_vectorizer = CountVectorizer(max_features=max_features,stop_words = "english")

sparce_matrix = count_vectorizer.fit_transform(description_list).toarray()  # x

print("mosltly used {} words: {}".format(max_features,count_vectorizer.get_feature_names()))


# In[ ]:


plt.subplots(figsize=(15,15))
wordcloud = WordCloud(mask=wave_mask_rebel,background_color="black",contour_width=3, contour_color="tan",colormap="rainbow",  
                      width=950,
                          height=950
                         ).generate(" ".join(count_vectorizer.get_feature_names()))

plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('graph.png')

plt.show()


# ## Embeddings?! 

# In[ ]:


all_episodes = episodeIV.append(episodeV).append(episodeVI)
all_episodes.tail(15)


# In[ ]:


dialogue_all_episodes = pd.DataFrame(all_episodes['dialogue'])
dialogue_all_episodes.head()


# In[ ]:


base = [row.split(',') for row in dialogue_all_episodes['dialogue']]
base_word = [row.split(' ') for row in dialogue_all_episodes['dialogue']]


# In[ ]:


base


# In[ ]:


from gensim.models import Word2Vec
model = Word2Vec(base, min_count=1, size=50, sg=1) 

## sg: The training algorithm, either CBOW(0) or skip gram(1). The default training algorithm is CBOW.


# In[ ]:


model.similarity('Lord Vader', 'Hey')


# In[ ]:


model.most_similar('Lord Vader')[:7]


# ## Embeddings with additional tuning

# In[ ]:


char_dialogue_all_episodes = pd.DataFrame(all_episodes[['character','dialogue']])
char_dialogue_all_episodes.head()


# In[ ]:


char_dial_join = char_dialogue_all_episodes.apply(lambda x: ','.join(x.astype(str)), axis=1)
char_dial_join


# In[ ]:


char_base = [row.split(',') for row in char_dial_join]
char_base


# In[ ]:


model2 = Word2Vec(char_base, min_count=1,size= 50, window=3, sg = 1)


# In[ ]:


model2.similarity("YODA", "THREEPIO")


# In[ ]:


model2.most_similar('Artoo')[:10]


# ![](https://media3.giphy.com/media/3ohuAxV0DfcLTxVh6w/giphy.gif)

# In[ ]:




