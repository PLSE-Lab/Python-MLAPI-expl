#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import warnings
warnings.filterwarnings('ignore')
from ast import literal_eval
import plotly.express as px
import nltk
import re


# In[ ]:


meta_data = pd.read_csv("/kaggle/input/the-movies-dataset/movies_metadata.csv")
link_small = pd.read_csv("/kaggle/input/the-movies-dataset/links_small.csv")
keywords = pd.read_csv("/kaggle/input/the-movies-dataset/keywords.csv")
credits = pd.read_csv("/kaggle/input/the-movies-dataset/credits.csv")
ratings_small = pd.read_csv("/kaggle/input/the-movies-dataset/ratings_small.csv")
meta_data.head()


# In[ ]:


meta_data['genres'] = meta_data['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


# In[ ]:


languages = meta_data["original_language"].value_counts()
langues_df = pd.DataFrame({'languages':languages.index, 'frequency':languages.values}).head(10)

fig = px.bar(langues_df, x="frequency", y="languages",color='languages', orientation='h',
             hover_data=["languages", "frequency"],
             height=1000,
             title='Language which has more Movies')
fig.show()


# In[ ]:


top_movies = meta_data[["title","vote_count"]]
top_movies = top_movies.sort_values(by="vote_count",ascending=False)


# In[ ]:


fig = px.bar(data_frame=top_movies[:20],x="title",y="vote_count",color="title",title="Most Voted Movies")
fig.show()


# In[ ]:


s = meta_data.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genre'
gen_md = meta_data.drop('genres', axis=1).join(s)


# In[ ]:


genre_counts = gen_md.genre.value_counts()
genre_df = pd.DataFrame({'genre':genre_counts.index,"count":genre_counts.values})
fig = px.bar(data_frame=genre_df[:20],x="genre",y="count",color="genre")
fig.show()


# In[ ]:


adults_count = meta_data['adult'].value_counts()
adults_df = pd.DataFrame({"adults":adults_count.index,"count":adults_count.values})
fig = px.bar(data_frame=adults_df[:2],x="adults",y="count",color="adults")
fig.show()


# In[ ]:


budget_analysis = meta_data.sort_values(by="budget",ascending=False)
budget_analysis[["budget",'title']].head(10)


# In[ ]:


credits = pd.read_csv("/kaggle/input/the-movies-dataset/credits.csv")
credits.head()


# In[ ]:


links_small = pd.read_csv('/kaggle/input/the-movies-dataset/links_small.csv')
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')


# In[ ]:


ratings = pd.read_csv("/kaggle/input/the-movies-dataset/ratings.csv")
ratings.head()


# In[ ]:


link_small = link_small[link_small['tmdbId'].notnull()]['tmdbId'].astype('int')
meta_data = meta_data.drop([19730, 29503, 35587])


# In[ ]:


meta_data['id'] = meta_data['id'].astype('int')


# In[ ]:


smd = meta_data[meta_data['id'].isin(links_small)]
smd.shape


# In[ ]:


smd['tagline'] = smd['tagline'].fillna('')
smd['description'] = smd['overview']
smd['description'] = smd['description'].fillna('')


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
# from surprise import Reader, Dataset, SVD, evaluate


# In[ ]:


smd.overview.head()


# In[ ]:


tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(smd['description'])
print(tfidf_matrix.shape)


# Since we have used the TF-IDF vectorizer, calculating the dot product will directly give us the cosine similarity score. Therefore, we will use sklearn's linear_kernel() instead of cosine_similarities() since it is faster.

# In[ ]:


cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[ ]:


smd = smd.reset_index()
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])


# In[ ]:


indices.head()


# In[ ]:


def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]


# In[ ]:


get_recommendations("The Godfather").head(10)


# While our system has done a decent job of finding movies with similar plot descriptions, the quality of recommendations is not that great. "The Dark Knight Rises" returns all Batman movies while it is more likely that the people who liked that movie are more inclined to enjoy other Christopher Nolan movies. This is something that cannot be captured by the present system

# In[ ]:


meta_data.head()


# # Genre Classification - MultiLabel Classification

# In[ ]:


title_df = meta_data[["overview","genres","title"]]


# In[ ]:


title_df.head()


# In[ ]:


punctuation = """!()-[]{};:'"\, <>./?@#$%^&*_~"""
from nltk import word_tokenize
from nltk.corpus import stopwords
words = stopwords.words("english")
lemma = nltk.stem.WordNetLemmatizer()
def pre_process(text):
    text = str(text)
    remove_hyperlink = re.sub('http://\S+|https://\S+', '', text)
    for elements in remove_hyperlink:
        if elements in punctuation:
            remove_hyperlink = remove_hyperlink.replace(elements, " ")
    tokens = word_tokenize(remove_hyperlink)
    remove_words = [word for word in tokens if not word in words]
    text = [lemma.lemmatize(word) for word in remove_words]
    joined_words = " ".join(text)
    return joined_words


# In[ ]:


title_df['title'] = title_df['title'].apply(pre_process)


# In[ ]:


title_df.head()


# In[ ]:


from sklearn.preprocessing import MultiLabelBinarizer

multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(title_df['genres'])

# transform target variable
y = multilabel_binarizer.transform(title_df['genres'])


# In[ ]:


tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=5000)


# In[ ]:


from sklearn import model_selection
xtrain, xval, ytrain, yval = model_selection.train_test_split(title_df['title'], y, test_size=0.2, random_state=9)


# In[ ]:


xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
xval_tfidf = tfidf_vectorizer.transform(xval)


# In[ ]:


from sklearn.linear_model import LogisticRegression

# Binary Relevance
from sklearn.multiclass import OneVsRestClassifier

# Performance metric
from sklearn.metrics import f1_score


# In[ ]:


lr = LogisticRegression()
clf = OneVsRestClassifier(lr)


# In[ ]:


clf.fit(xtrain_tfidf, ytrain)


# In[ ]:


y_pred = clf.predict(xval_tfidf)


# In[ ]:


y_pred[3]


# In[ ]:


multilabel_binarizer.inverse_transform(y_pred)[3]


# In[ ]:


f1_score(yval, y_pred, average="micro")


# In[ ]:


y_pred_prob = clf.predict_proba(xval_tfidf)


# In[ ]:


t = 0.3 # threshold value
y_pred_new = (y_pred_prob >= t).astype(int)


# In[ ]:


f1_score(yval, y_pred_new, average="micro")


# In[ ]:


def inference(text):
    processed_text = pre_process(text)
    q_vec = tfidf_vectorizer.transform([processed_text])
    q_pred = clf.predict(q_vec)
    q_pred = clf.predict(q_vec)
    return multilabel_binarizer.inverse_transform(q_pred)


# In[ ]:


for i in range(5):
    k = xval.sample(1).index[0]
    print("Movie: ", title_df['title'][k], "\nPredicted genre: ", inference(xval[k])), print("Actual genre: ",title_df['genres'][k], "\n")


# # Text Generation using Pytorch

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import string
import unidecode
import random
import torch


# In[ ]:


# Check if GPU is available
train_on_gpu = torch.cuda.is_available()
if(train_on_gpu):
    print('Training on GPU!')
else: 
    print('No GPU available, training on CPU; consider making n_epochs very small')


# In[ ]:


meta_data.head()


# In[ ]:


content = meta_data["overview"]
content[:5]


# In[ ]:


content.shape


# In[ ]:


all_text = list(content)
def joinStrings(text):
    text = str(text)
    return ''.join(string for string in text)
text = joinStrings(all_text[:100])
# text = [item for sublist in author[:5].values for item in sublist]
len(text.split())


# In[ ]:


test_sentence = pre_process(text).split()


# In[ ]:


test_sentence


# In[ ]:


trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
            for i in range(len(test_sentence) - 2)]
chunk_len=len(trigrams)
print(trigrams[:3])


# In[ ]:




vocab = set(test_sentence)
voc_len=len(vocab)
word_to_ix = {word: i for i, word in enumerate(vocab)}


# In[ ]:


inp=[]
tar=[]
for context, target in trigrams:
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
        inp.append(context_idxs)
        targ = torch.tensor([word_to_ix[target]], dtype=torch.long)
        tar.append(targ)


# In[ ]:


import torch
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        
        self.encoder = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size*2, hidden_size, n_layers,batch_first=True,
                          bidirectional=False)
        self.decoder = nn.Linear(hidden_size, output_size)
    
    def forward(self, input, hidden):
        input = self.encoder(input.view(1, -1))
        output, hidden = self.gru(input.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))


# In[ ]:


def train(inp, target):
    hidden = decoder.init_hidden().cuda()
    decoder.zero_grad()
    loss = 0
    
    for c in range(chunk_len):
        output, hidden = decoder(inp[c].cuda(), hidden)
        loss += criterion(output, target[c].cuda())

    loss.backward()
    decoder_optimizer.step()

    return loss.data.item() / chunk_len


# In[ ]:




import time, math

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# In[ ]:


n_epochs = 300
print_every = 100
plot_every = 10
hidden_size = 100
n_layers = 1
lr = 0.015

decoder = RNN(voc_len, hidden_size, voc_len, n_layers)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

start = time.time()
all_losses = []
loss_avg = 0
if(train_on_gpu):
    decoder.cuda()
for epoch in range(1, n_epochs + 1):
    loss = train(inp,tar)       
    loss_avg += loss

    if epoch % print_every == 0:
        print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / n_epochs * 50, loss))
#         print(evaluate('ge', 200), '\n')

    if epoch % plot_every == 0:
        all_losses.append(loss_avg / plot_every)
        loss_avg = 0


# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure()
plt.plot(all_losses)


# In[ ]:




def evaluate(prime_str='this process', predict_len=100, temperature=0.8):
    hidden = decoder.init_hidden().cuda()

    for p in range(predict_len):
        
        prime_input = torch.tensor([word_to_ix[w] for w in prime_str.split()], dtype=torch.long).cuda()
        inp = prime_input[-2:] #last two words as input
        output, hidden = decoder(inp, hidden)
        
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        
        # Add predicted word to string and use as next input
        predicted_word = list(word_to_ix.keys())[list(word_to_ix.values()).index(top_i)]
        prime_str += " " + predicted_word
#         inp = torch.tensor(word_to_ix[predicted_word], dtype=torch.long)

    return prime_str


# In[ ]:


print(evaluate('this process', 40, temperature=1))


# In[ ]:




