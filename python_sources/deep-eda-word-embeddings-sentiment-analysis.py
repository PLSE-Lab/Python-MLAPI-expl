#!/usr/bin/env python
# coding: utf-8

# # Reuter News Exploratory Data Analysis (Word Embeddings and Sentiment Analysis)
# This kernel is created in attempt to imporve my current NLP, EDA and data visualization skills. The dataset provided is a collections of Reuter News headlines and timestamp for each news from year 2011 till year 2017. There are approximately 6M headlines in total, it's HUGE! Let's see what can we dig from all these headlines.
# 
# In this kernel, I will skip most of the feature engineering as it has been done. You can always refer to [sban's analysis](https://www.kaggle.com/shivamb/seconds-from-disaster-text-analysis-fe-updated) for more details.
# 
# ## Contents:
# 1. Data Preparation
# 2. Word Embeddings
#       - What is word embeddings?
#       - Clustering of words
#       - How to find key words of certain topic based on trained word embeddings?
#       - Keywords ( Country names ) & Comparison between year 2011 and 2017
#       - Contry names appearances VS Time Graph (2011 to 2017)
#       - Keywords ( Disaster ) 
#       - Disaster keywords appearance Time Graph (2011 to 2017)
#       - Keywords ( Company names ) 
#       - Company names appearances VS Time Graph (2011 to 2017)
#       - Keywords ( Politician names ) 
#       - Politician names appearances VS Time Graph (2011 to 2017)
#       - Keywords ( Sport stars names ) 
#       - Sport stars  names appearances VS Time Graph (2011 to 2017)
# 3. Sentiment Analysis
#       - Generate text features: sentiment polarity
#       - Normalized Dow Jones Index
#       - Group polarity based on days and find the average
#       - Match the sentiment polarity with the day Dow Jones Index opened
#       - Sum the normalized DJI closed value based on distinct polarity.
#       - Plot graph for Sentiment Polarity VS Average Closed Polarity

# ## 1. Data Preparation
# First of all, the required libraries will be loaded to this notebook. 

# In[1]:


#Some fundamental libraries
import numpy as np
import pandas as pd
import sys
import os
import re
import collections
import csv
import gc

#Visualisation libraries
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
import colorlover as cl

#NLP libraries
from six.moves import xrange 
import tensorflow as tf
from nltk.corpus import stopwords


# All of the headlines from year 2011 to 2017 will be used. Let's see hows the dataset looks like.

# In[2]:


list_data = ['reuters-newswire-2011.csv','reuters-newswire-2012.csv','reuters-newswire-2013.csv','reuters-newswire-2014.csv','reuters-newswire-2015.csv','reuters-newswire-2016.csv','reuters-newswire-2017.csv']
dfs=list()
for csv in list_data:
    df = pd.read_csv('../input/reuters-news-wire-archive/'+csv)
    dfs.append(df)
data = pd.concat(dfs).reset_index()
print(len(data))
data.head()


# In[3]:


data.tail()


# ## Word Embeddings
# 
# ### What is word embeddings? 
# I copy this introduction from tensorflow tutotial, you can refer here([Tensorflow tutorial](https://www.tensorflow.org/tutorials/word2vec)) for more details. 
# 
# Image and audio processing systems work with rich, high-dimensional datasets encoded as vectors of the individual raw pixel-intensities for image data, or e.g. power spectral density coefficients for audio data. For tasks like object or speech recognition we know that all the information required to successfully perform the task is encoded in the data (because humans can perform these tasks from the raw data). However, natural language processing systems traditionally treat words as discrete atomic symbols, and therefore 'cat' may be represented as Id537 and 'dog' as Id143. These encodings are arbitrary, and provide no useful information to the system regarding the relationships that may exist between the individual symbols. This means that the model can leverage very little of what it has learned about 'cats' when it is processing data about 'dogs' (such that they are both animals, four-legged, pets, etc.). Representing words as unique, discrete ids furthermore leads to data sparsity, and usually means that we may need more data in order to successfully train statistical models. Using vector representations can overcome some of these obstacles.
# 
# ![](https://www.tensorflow.org/images/audio-image-text.png)
# 
# Word embeddings is a particularly computationally-efficient predictive model for learning word embeddings from raw text. It comes in two flavors, the Continuous Bag-of-Words model (CBOW) and the Skip-Gram model (Mikolov et al.). Algorithmically, these models are similar, except that CBOW predicts target words (e.g. 'mat') from source context words ('the cat sits on the'), while the skip-gram does the inverse and predicts source context-words from the target words. This inversion might seem like an arbitrary choice, but statistically it has the effect that CBOW smoothes over a lot of the distributional information (by treating an entire context as one observation). For the most part, this turns out to be a useful thing for smaller datasets. However, skip-gram treats each context-target pair as a new observation, and this tends to do better when we have larger datasets. 
# 
# In this kernel, we will focus on the skip-gram model.
# 
# ### Skip-gram Model
# As an example, let's consider the dataset
# 
#     the quick brown fox jumped over the lazy dog
# 
# We first form a dataset of words and the contexts in which they appear. We could define 'context' in any way that makes sense, and in fact people have looked at syntactic contexts (i.e. the syntactic dependents of the current target word, see e.g. Levy et al.), words-to-the-left of the target, words-to-the-right of the target, etc. For now, let's stick to the vanilla definition and define 'context' as the window of words to the left and to the right of a target word. Using a window size of 1, we then have the dataset
# 
#     ([the, brown], quick), ([quick, fox], brown), ([brown, jumped], fox), ...
# 
# of (context, target) pairs. Recall that skip-gram inverts contexts and targets, and tries to predict each context word from its target word, so the task becomes to predict 'the' and 'brown' from 'quick', 'quick' and 'fox' from 'brown', etc. Therefore our dataset becomes
# 
#     (quick, the), (quick, brown), (brown, quick), (brown, fox), ...
#     
# We can visualize the learned vectors by projecting them down to 2 dimensions using for instance something like the t-SNE dimensionality reduction technique. When we inspect these visualizations it becomes apparent that the vectors capture some general, and in fact quite useful, semantic information about words and their relationships to one another. It was very interesting when we first discovered that certain directions in the induced vector space specialize towards certain semantic relationships, e.g. male-female, verb tense and even country-capital relationships between words, as illustrated in the figure below (see also for example Mikolov et al., 2013).
# 
# ![](https://www.tensorflow.org/images/linear-relationships.png)
# 
# This explains why these vectors are also useful as features for many canonical NLP prediction tasks, such as part-of-speech tagging or named entity recognition (see for example the original work by Collobert et al., 2011 (pdf), or follow-up work by Turian et al., 2010).

# ### Clustering of Words
# The reason of using word embedding model in this kernel is to cluster words based on their meaning, semantic and relationship to other words.  By using this, we can find group of specific words such as name of country by just dumping in a few country names that we know. 
# 
# However, before we can do that, we need to train our own word embeddings. As mentioned in introduction above, we will have to gather every single words in those headlines. 

# In[ ]:


texts=[]
for index, row in data.iterrows():
    if not pd.isnull(row['headline_text']):
        text = re.sub('\W',' ',row['headline_text'])
        text = text.split()
        for word in text:
            texts.append(word.lower())
            
print('Data size', len(texts))
del(texts)
gc.collect()


# Suprisingly that we got around 60M words from those headlines! If the window of skipgram model set to be 2, we will have total 120M data to be trained. Such a huge data will need more than 48 hours to train, so I will run it in my workstation instead of notebook. 
# 
# Anyway, this is the code that I've run.  The result of this model will create a csv that have the nearest 50 words to each words (only top 50,000 most freq word). Also, I have separated the process into 12 parts where each part will train 10M data. By doing this we can see how the relationships of words change from year 2011 to 2017.

# In[ ]:


'''
def read_data():
    stopWords = set(stopwords.words('english'))
    list_data = ['reuters-newswire-2011.csv','reuters-newswire-2012.csv','reuters-newswire-2013.csv','reuters-newswire-2014.csv','reuters-newswire-2015.csv','reuters-newswire-2016.csv','reuters-newswire-2017.csv']
    dfs=list()
    texts = []
    for csv in list_data:
        df = pd.read_csv('../input/reuters-news-wire-archive/'+csv)
        dfs.append(df)
    data = pd.concat(dfs)
    length = len(data)
    for index, row in data.iterrows():
        sys.stdout.write('\rProcessing----%d/%d'%(index+1,length))
        sys.stdout.flush()
        if not pd.isnull(row['headline_text']):
            text = re.sub('\W',' ',row['headline_text'])
            text = text.split()
            for word in text:
                #if word not in stopWords:
                texts.append(word.lower())
        #gc.collect()
    return texts

vocabulary = read_data()
print('Data size', len(vocabulary))
#print(vocabulary)

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 50000

def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    index = dictionary.get(word, 0)
    if index == 0:  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary

# Filling 4 global variables:
# data - list of codes (integers from 0 to vocabulary_size-1).
#   This is the original text but words are replaced by their codes
# count - map of words(strings) to count of occurrences
# dictionary - map of words(strings) to their codes(integers)
# reverse_dictionary - maps codes(integers) to words(strings)
data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                            vocabulary_size)
del vocabulary  # Hint to reduce memory.
#print(reverse_dictionary)
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0

# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  if data_index + span > len(data):
    data_index = 0
  buffer.extend(data[data_index:data_index + span])
  #print(buffer)
  data_index += span
  for i in range(batch_size // num_skips):
    context_words = [w for w in range(span) if w != skip_window]
    words_to_use = random.sample(context_words, num_skips)
    for j, context_word in enumerate(words_to_use):
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[context_word]
    if data_index == len(data):
      for word in data[:span]:
        buffer.append(word)
      data_index = span
    else:
      buffer.append(data[data_index])
      #print(buffer)
      data_index += 1
  # Backtrack a little bit to avoid skipping words in the end of a batch
  data_index = (data_index + len(data) - span) % len(data)
  return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
  print(batch[i], reverse_dictionary[batch[i]],
        '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

# Step 4: Build and train a skip-gram model.

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.
num_sampled = 64      # Number of negative examples to sample.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. These 3 variables are used only for
# displaying model accuracy, they don't affect calculation.
valid_size = 50000     # Random set of words to evaluate similarity on.
valid_window = 50000  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)


graph = tf.Graph()

with graph.as_default():

  # Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  # Explanation of the meaning of NCE loss:
  #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
  loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=vocabulary_size))

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  show_emb = tf.Print(valid_embeddings, [valid_embeddings])
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)

  # Add variable initializer.
  init = tf.global_variables_initializer()
num_steps = 50000001

with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  init.run()
  print('Initialized')

  average_loss = 0
  for step in xrange(num_steps):
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, skip_window)
    #print(len(batch_inputs))
    #print(len(batch_labels))
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val
    vocab_words = []
    vocab_code = []
    nearests = []
    if step % 100000 == 0:
      if step > 0:
        average_loss /= 10000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print('Average loss at step ', step, ': ', average_loss)
      average_loss = 0

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000000 == 0:
      sim = similarity.eval()
      show = show_emb.eval()
      p_embeddings = []
      for s in show:
        p_embeddings.append(s)
      #for s in show:
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[i]
        vocab_words.append(valid_word)
      
      rows = zip(vocab_words,p_embeddings)
      with open('Full_Embedding/5_epochs_v1/product_emb_epoch%d.csv'%(step/10000000), "w") as f:
        writer = csv.writer(f)
        for row in rows:
          writer.writerow(row)
      #PE = pd.DataFrame({"emb":p_embeddings})
      #PE.to_csv('product_emb_300d%d.csv'%(step/2300000), sep='\t', encoding='utf-8')
      vocab_words = []
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 50  # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = 'Nearest to %s:' % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = '%s %s,' % (log_str, close_word)
        #print(log_str)
        vocab_words.append(valid_word)
        nearests.append(log_str)
      w = pd.DataFrame({'stockcode':vocab_words})
      n = pd.DataFrame({'nearest':nearests})
      data_merged = pd.concat([w, n], axis=1)
      data_merged.to_csv('Nearest/v1/product_nearest_epoch%d.csv'%(step/10000001), sep='\t', encoding='utf-8')
      gc.collect()
  final_embeddings = normalized_embeddings.eval()

# Step 6: Visualize the embeddings.


# pylint: disable=missing-docstring
# Function to draw visualization of distance between embeddings.
def plot_with_labels(low_dim_embs, labels, filename):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)

try:
  # pylint: disable=g-import-not-at-top
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
  plot_only = 500
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
  labels = [reverse_dictionary[i] for i in xrange(plot_only)]
  plot_with_labels(low_dim_embs, labels, 'tsne.png') #os.path.join(gettempdir(), 'tsne.png'))

except ImportError as ex:
  print('Please install sklearn, matplotlib, and scipy to show embeddings.')
  print(ex)
'''


# In[4]:


def clean(text):
    text = text.split(': ')
    return text[1]

def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

def get_freq(keys,csv):
    stopWords = set(stopwords.words('english'))
    dfs=list()
    texts = []
    data = pd.read_csv(csv)
    #data = data[:10000]
    length = len(data)
    for index, row in data.iterrows():
        if index%100000 ==  0:
            sys.stdout.write('\rProcessing----%d/%d'%(index,length))
            sys.stdout.flush()
        if not pd.isnull(row['headline_text']):
            text = re.sub(r"\.", "", row['headline_text']) 
            text = re.sub('\W',' ',text)
            text = text.split()
            for word in text:
                word = word.lower()
                if word in keys:
                    texts.append(word)
    gc.collect()
    return texts

def get_y_data(key,NUM_KEY):
    y_data = []
    key = key[:NUM_KEY]
    print(key)
    for c, csv in enumerate(list_data):
        if c == 0:
            freq_key = get_freq(key,'../input/reuters-news-wire-archive/'+csv)
            data_freq, count_freq, dictionary_freq, reverse_dictionary_freq = build_dataset(freq_key,NUM_KEY+1)
            count_freq = dict(count_freq)
            for k in key:
                try: 
                    count_freq[k]
                except:
                    y_data.append([0])
                else:
                    y_data.append([count_freq[k]])

        else:
            freq_key = get_freq(key,'../input/reuters-news-wire-archive/'+csv)
            data_freq, count_freq, dictionary_freq, reverse_dictionary_freq = build_dataset(freq_key,NUM_KEY+1)
            count_freq = dict(count_freq)
            for index, k in enumerate(key):
                try: 
                    count_freq[k]
                except:
                    y_data[index].append(0)
                else:
                    y_data[index].append(count_freq[k])
    return(y_data)


# ### How to find key words of certain topic based on trained word embeddings?
# 1. We'll list out few words that related to our topic (e.g. country -> us, china, russia)
# 2. We'll find the 50 nearest words to each word in upper layer (us, china ,russia)
# 3. Then another 50 nearest words to each word in upper layer (50 from us, 50 from china, 50 from russia)
# 4. Now we'll have 3x50x50 (7500) words in dictionary.
# 5. Last we will count appearance of each word and sort them to a fixed length (e.g. Top 50 words)
# 
# Let's see what words can we find out with us, china and russia.
# 
# ## **Country**
#      
# ![](https://openclipart.org/image/2400px/svg_to_png/286828/countries_flags.png)

# In[5]:


def plot_nearest(_list,csv,title):
    word_output = []
    near = pd.read_csv(csv)
    near['nearest'] = near['nearest'].apply(clean)

    for layer1 in _list:
        layer2 = near.loc[near['stockcode']==layer1]['nearest'].values
        if layer2:
            layer2 = layer2[0].split(', ')
            for word_layer2 in layer2:
                word_layer2 = re.sub('\W','',word_layer2)
                word_output.append(word_layer2)
                layer3 = near.loc[near['stockcode']==word_layer2]['nearest'].values
                layer3 = layer3[0].split(', ')
                for word_layer3 in layer3:
                    word_layer3 = re.sub('\W','',word_layer3)
                    word_output.append(word_layer3)
                    
    data_list, count_list, dictionary_list, reverse_dictionary_list = build_dataset(word_output,50)
    count_list = dict(count_list)
    keys = list(dictionary_list.keys())
    value = list(dictionary_list.values())
    key = [ keys[i] for i in sorted(range(len(value)), key=lambda k: value[k])]
    value = [ count_list[k] for k in key ]
    key = key[1:]
    value = value[1:]

    n_phase = len(key)
    plot_width = 400

    # height of a section and difference between sections 
    section_h = 40
    section_d = 10

    # multiplication factor to calculate the width of other sections
    unit_width = plot_width / max(value)

    # width of each funnel section relative to the plot width
    phase_w = [int(v * unit_width) for v in value]

    # plot height based on the number of sections and the gap in between them
    height = section_h * n_phase + section_d * (n_phase - 1)

    # list containing all the plot shapes
    shapes = []

    # list containing the Y-axis location for each section's name and value text
    label_y = []

    for i in range(n_phase):
        if (i == n_phase-1):
                points = [phase_w[i] / 2, height, phase_w[i] / 2, height - section_h]
        else:
                points = [phase_w[i] / 2, height, phase_w[i+1] / 2, height - section_h]

        path = 'M {0} {1} L {2} {3} L -{2} {3} L -{0} {1} Z'.format(*points)

        shape = {
                'type': 'path',
                'path': path,
                'fillcolor': 'rgb(32,155,160)',
                'line': {
                    'width': 1
                }
        }
        shapes.append(shape)
        
        # Y-axis location for this section's details (text)
        label_y.append(height - (section_h / 2))

        height = height - (section_h + section_d)

    # For phase names
    label_trace = go.Scatter(
        x=[-350]*n_phase,
        y=label_y,
        mode='text',
        text=key,
        textfont=dict(
            color='rgb(200,200,200)',
            size=15
        )
    )
    
    # For phase values\
    value_trace = go.Scatter(
        x=[350]*n_phase,
        y=label_y,
        mode='text',
        text=value,
        textfont=dict(
            color='rgb(200,200,200)',
            size=15
        )
    )

    data = [label_trace, value_trace]

    layout = go.Layout(
        title="<b>"+title+"</b>",
        titlefont=dict(
            size=20,
            color='rgb(203,203,203)'
        ),
        shapes=shapes,
        height=1000,
        width=800,
        showlegend=False,
        paper_bgcolor='rgba(44,58,71,1)',
        plot_bgcolor='rgba(44,58,71,1)',
        xaxis=dict(
            showticklabels=False,
            zeroline=False,
        ),
        yaxis=dict(
            showticklabels=False,
            zeroline=False
        )
    )

    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig)
    return key
    


# In[6]:


list_country = ['us','china','russia']
key = plot_nearest(list_country,"../input/embeddings/product_nearest_epoch1.csv","Country Names From 10M Data")


# Its amazing right? Now we got a bunch of country names based on the 1st 10M words from headlines. However, the appearance of these words might not truly reflect their appearance in headlines as we can see us and china is not even in top 50 names. This might differ as the way headlines was written is different, country name such as us and china used too often in different sentiment which makes them not purely country name in point of view of word embeddings. 
# 
# ### Compare the list of country names between 10M and 100M training data.

# In[7]:


list_country = ['us','china','russia']
key_country = plot_nearest(list_country,"../input/embeddings/product_nearest_epoch10.csv","Country Names From 100M Data")


# ### Country names appearances Time Graph (2011 to 2017)

# In[12]:


def plot_freq_years(key,NUM_KEY,y_data,title):
    title = title
    labels = key
    colors = cl.scales['12']['qual']['Paired']

    for i in range(int(NUM_KEY/12)+1):
        colors += colors

    colors = colors[:NUM_KEY]

    years = [2011, 2012, 2013, 2014, 2015, 2016, 2017]
    x_data=[]
    for n in range(NUM_KEY):
        x_data.append(years)

    traces = []

    for i in range(0, NUM_KEY):
        traces.append(go.Scatter(
            x=x_data[i],
            y=y_data[i],
            mode='lines',
            name ='',
            text=key[i],
            line=dict(color=colors[i], width=2),
            connectgaps=True,
        ))

        traces.append(go.Scatter(
            x=[x_data[i][0], x_data[i][len(years)-1]],
            y=[y_data[i][0], y_data[i][len(years)-1]],
            mode='markers',
            marker=dict(color=colors[i], size=12)
        ))

    layout = go.Layout(
        xaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            autotick=False,
            ticks='outside',
            tickcolor='rgb(204, 204, 204)',
            tickwidth=2,
            ticklen=5,
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            ),
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            showticklabels=False,
        ),
        autosize=False,
        margin=dict(
            autoexpand=False,
            l=100,
            r=20,
            t=110,
        ),
        showlegend=False,
    )

    annotations = []

# Adding labels
    for y_trace, label, color in zip(y_data, labels, colors):
        # labeling the left_side of the plot
        annotations.append(dict(xref='paper', x=0.05, y=y_trace[0],
                                      xanchor='right', yanchor='middle',
                                      text=label + ' {}'.format(y_trace[0]),
                                      font=dict(family='Arial',
                                                size=10,
                                                color=colors,),
                                      showarrow=False))
        # labeling the right_side of the plot
        annotations.append(dict(xref='paper', x=0.95, y=y_trace[len(years)-1],
                                      xanchor='left', yanchor='middle',
                                      text='{}'.format(y_trace[len(years)-1]),
                                      font=dict(family='Arial',
                                                size=10,
                                                color=colors,),
                                      showarrow=False))
    # Title
    annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                                  xanchor='left', yanchor='bottom',
                                  text=title,
                                  font=dict(family='Arial',
                                            size=30,
                                            color='rgb(37,37,37)'),
                                  showarrow=False))
    # Source
    annotations.append(dict(xref='paper', yref='paper', x=0.5, y=-0.1,
                                  xanchor='center', yanchor='top',
                                  text='EDA Analysis',
                                  font=dict(family='Arial',
                                            size=12,
                                            color='rgb(150,150,150)'),
                                  showarrow=False))

    layout['annotations'] = annotations

    fig = go.Figure(data=traces, layout=layout)
    py.iplot(fig, filename='Reuter-Disasters')


# In[10]:


extra_country = ['us','china']
key_country = extra_country + key_country
y_data = get_y_data(key_country,50)
plot_freq_years(key_country,50,y_data,'Keyword Appearance for countries')


# ## Disaster
# Let see the clustered keywords for disaster by inserting 'famine','tornado','tsunami','landslide','flood','influenza'.
# #    
# ![](https://www.finglobal.com/wp-content/uploads/2016/08/natural-disasters-of-earths-past-and-future.jpg)

# ### Keywords for disaster

# In[8]:


list_disaster = ['famine','tornado','tsunami','landslide','flood','influenza']
key_disaster = plot_nearest(list_disaster,"../input/embeddings/product_nearest_epoch10.csv","Disaster From 100M Data")


# ### Disaster appearances Time Graph (2011 to 2017)

# In[ ]:


y_data = get_y_data(key_disaster,25)
plot_freq_years(key_disaster,25,y_data,'Keyword Appearance for disaster')


# ## Company
# Let see the clustered keywords for company by inserting 'apple', 'google', 'facebook', 'alibaba'.
# #   
# ![](http://1000logos.net/wp-content/uploads/2016/09/brand-logos.jpg)

# ### Keywords for companies

# In[9]:


list_company = ['apple', 'google', 'facebook', 'alibaba']
key_company = plot_nearest(list_company,"../input/embeddings/product_nearest_epoch10.csv","Companies From 100M Data")


# ### Companies appearances Time Graph (2011 to 2017)

# In[15]:


y_data = get_y_data(key_company,40)
plot_freq_years(key_company,40,y_data,'Keyword Appearance for companies')


# ## Politicians
# Let see the clustered keywords for politicians by inserting 'obama', 'trump', 'putin'.
# #    
# ![](http://twiplomacy.com/wp-content/uploads/2017/04/World-leaders-on-Insta_cover-ordered.png)

# ### Keywords for politicians

# In[13]:


list_politicians = ['obama', 'trump', 'putin']
key_politicians = plot_nearest(list_politicians,"../input/embeddings/product_nearest_epoch10.csv","Politicians From 100M Data")


# ### Politicians appearances Time Graph (2011 to 2017)

# In[14]:


y_data = get_y_data(key_politicians,20)
plot_freq_years(key_politicians,20,y_data,'Keyword Appearance for politicians')


# ## Sport stars
# Let see the clustered keywords for Sport stars by inserting 'ronaldo','messi','nadal','sharapova','phelps','lebron'.
# #     
# ![](https://static.independent.co.uk/s3fs-public/styles/article_small/public/thumbnails/image/2017/03/24/17/sharapova-2.jpg)

# ### Keywords for Sport stars

# In[20]:


list_sportstar = ['ronaldo','messi','nadal','sharapova','phelps','lebron']
key_sportstar = plot_nearest(list_sportstar,"../input/embeddings/product_nearest_epoch10.csv","Sport stars From 100M Data")


# ### Sport stars appearances Time Graph (2011 to 2017)

# In[21]:


y_data = get_y_data(key_sportstar,50)
plot_freq_years(key_sportstar,20,y_data,'Keyword Appearance for Sport Stars')


# # Leave your comment below if you have any interesting keywords to be analysed ! :D

# # Sentiment Analysis
# In this section we will focus on the relationship between sentiment polarity and Dow Jones Index. Before we start, the hypothesis of this analytic is to find out whether the positive sentiment of overall headlines will contibute to increment in Dow Jones Index. 
# 
# ### Generate text features: sentiment polarity
# Below is the code to generate sentiment polarity, but I have uploaded a processed data to save time. Let's look at the processed data and our  Dow Jones Index.
# 

# In[ ]:


'''
def get_polarity(text):
    global gloindex
    sys.stdout.write('\rProcessing----%d'%gloindex)
    sys.stdout.flush()
    try:
        textblob = TextBlob(text)
        pol = textblob.sentiment.polarity
    except:
        pol = 0.0
    gloindex+=1
    return pol

def get_weekday(date):
    year = int(str(date)[0:4])
    month = int(str(date)[4:6])
    day = int(str(date)[4:6])
    return datetime.datetime(year,month,day).weekday()

list_data = ['reuters-newswire-2011.csv','reuters-newswire-2012.csv','reuters-newswire-2013.csv','reuters-newswire-2014.csv','reuters-newswire-2015.csv','reuters-newswire-2016.csv','reuters-newswire-2017.csv']
dfs=list()
texts = []
for csv in list_data:
    df = pd.read_csv(csv)
    dfs.append(df)
data = pd.concat(dfs)
print(data.head())
print(len(data))
data['word_count'] = data['headline_text'].apply(lambda x : len(str(x).split()))
data['year'] = data['publish_time'].apply(lambda x : str(x)[0:4])
data['month'] = data['publish_time'].apply(lambda x : str(x)[4:6])
data['date'] = data['publish_time'].apply(lambda x : str(x)[6:8])
data['hour'] = data['publish_time'].apply(lambda x : str(x)[8:10])
data['minute'] = data['publish_time'].apply(lambda x : str(x)[10:])
data['weekday'] = data['publish_time'].apply(get_weekday)
data['polarity'] = data['headline_text'].apply(get_polarity)
data.to_csv('reuter_processed.csv', sep='\t')
'''


# In[ ]:


data_processed = pd.read_csv('../input/private-data/reuter_processed.csv')
data_processed.head()


# In[ ]:


dji = pd.read_csv('../input/private-data/DJIA_122017.csv')
dji.head()


# In[ ]:


dji['Date'] = pd.to_datetime(dji['Date'],errors='coerce', format='%m/%d/%Y')
data = [go.Scatter(x=dji.Date, y=dji.Close)]

py.iplot(data)


# ### Normalized Dow Jones Index

# In[ ]:


def Normalize(value,previous):
    return(((value/previous)-1)*100)

for index, row in dji.iterrows():
    #print(row)
    close = row['Close']
    if index > 0:
        value_nor = Normalize(close,pre)
        dji['Close'][index] = value_nor
    pre = close
    
dji = dji[1:]
dji.head()


# In[ ]:


dji['Date'] = pd.to_datetime(dji['Date'],errors='coerce', format='%m/%d/%Y')
data = [go.Scatter(x=dji.Date, y=dji.Close)]

py.iplot(data)


# ### Group polarity based on days and find the average
# Before we start grouping, there are few conditions and assumptions that we need to follow:
# 1. Reference time for Reuter news is UTC
# 2. Reference time for Dow Jones Index is EST
# 3. EST = UTC - 5h
# 4. Dow Jones Index open at 9.30am EST and close at 4.00 EST from Monday to Friday
# 5. Assume that news after closing time will not affect the index

# In[ ]:


data_processed = data_processed[data_processed['hour']<9]
pol_data = data_processed.groupby(['year','month','date'])['polarity'].mean().reset_index()
pol_data.tail()


# ### Match the sentiment polarity with the day Dow Jones Index opened

# In[ ]:


Date = []
Polarity = []
Close = []
for index, row in dji.iterrows():
    Year = int(str(row['Date'].year))
    Month = int(str(row['Date'].month))
    Day = int(str(row['Date'].day))
    pol = pol_data.loc[(pol_data['year'] == Year) & (pol_data['month']== Month) & (pol_data['date']== Day)]['polarity']
    if pol.values:
        Date.append(str(row['Date']))
        Polarity.append(pol.values[0])
        Close.append(row['Close'])

Date = pd.DataFrame({'Date':Date})
Polarity = pd.DataFrame({'Polarity':Polarity})
Close = pd.DataFrame({'Close':Close})

result = pd.concat([Date,Polarity,Close], axis=1)
result.head()


# ### Sum the normalized DJI closed value based on distinct polarity.

# In[ ]:


result['Polarity'] = result['Polarity'].apply(lambda x: round(x,3))
res = result.groupby('Polarity')['Close'].agg('sum').reset_index()
res.head()


# ### Plot graph for Sentiment Polarity VS Average Closed Polarity

# In[ ]:


datas = [go.Bar(
            x=res['Polarity'],
            y=res['Close'],
            textposition = 'auto',
            marker=dict(
                color='rgb(158,202,225)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=1.5),
            ),
            opacity=0.6
        )]
layout =  go.Layout(
    xaxis=dict(
        title='Sentiment Polarity',
        titlefont=dict(
            family='Arial, sans-serif',
            size=18,
            color='lightgrey'
        ),
        showticklabels=True,
        tickangle=45,
        tickfont=dict(
            family='Old Standard TT, serif',
            size=14,
            color='black'
        ),
        exponentformat='e',
        showexponent='All'
    ),
    yaxis=dict(
        title='Index Polarity',
        titlefont=dict(
            family='Arial, sans-serif',
            size=18,
            color='lightgrey'
        ),
        showticklabels=True,
        tickangle=45,
        tickfont=dict(
            family='Old Standard TT, serif',
            size=14,
            color='black'
        ),
        exponentformat='e',
        showexponent='All'
    )
)
fig = dict(data=datas, layout=layout)
py.iplot(fig, filename='bar-direct-labels')


# The line plot deny the hypothesis that we made earlier. We see no clear relationship between polarity of index behaviour and sentiment from general Reuter news. 
# 
# # To be continued...
# 
# ## Thanks for exploring, please upvote if you liked.
