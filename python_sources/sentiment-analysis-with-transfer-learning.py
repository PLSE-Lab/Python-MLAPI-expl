#!/usr/bin/env python
# coding: utf-8

# **In this kernel, we'll use transfer learning in Natural Language Processing with the FastAI library. We will apply the [Universal Language Fine-tuning for Text Classification](https://arxiv.org/pdf/1801.06146.pdf) paper to classify the sentiment of hotel reviews. As for transfer learning in Computer Vision, we can obtain relatively good results with not a huge sample of training data.**

# In[1]:


import numpy as np
import pandas as pd
from fastai import *
from fastai.text import *
from fastai.vision import *
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


# ## Load Data

# In[2]:


data = pd.read_csv('../input/Hotel_Reviews.csv')
data.head(3)


# The format of the given reviews is particular as the review is separated in positive and negative points. 
# 
# We will transform this problem in a binary classification problem.

# In[3]:


df_pos = pd.DataFrame(dict(text=data[data.Review_Total_Positive_Word_Counts>2].Positive_Review))
df_pos['score'] = np.ones(len(df_pos), dtype=int)
df_neg = pd.DataFrame(dict(text=data[data.Review_Total_Negative_Word_Counts>2].Negative_Review))
df_neg['score'] = np.zeros(len(df_neg), dtype=int)
df = pd.concat([df_pos, df_neg], axis=0)


# In[4]:


print('Number of reviews:', len(df))


# In[5]:


sns.countplot(data=df, x='score')
None


# The data seems quite balanced, we can use accuracy as the metric.
# 
# To see how powerful transfer learning is, we'll only work with 10000 samples. If you want to work with the full dataset, you can comment the next line, and you'll see that the result will be even better!

# In[6]:


df = df.sample(10000)


# In[7]:


# Train & test split
df_train, df_test = train_test_split(df[['score', 'text']], test_size=0.2)
df_train = df_train.dropna()
df_test = df_test.dropna()
df_test.head()


# ## Preprocess and fine-tune language model

# In fastai library, preprocessing text holds in a single line! It does behind the scene the different steps of preprocessing: cleaning, tokenizing, indexing, building vocabulary, etc.

# In[8]:


data_lm = TextLMDataBunch.from_df('./', df_train, df_test)


# In[9]:


data_lm.vocab.itos[:20]


# Then, rather than training a sentiment analysis model directly from scratch, we will fine-tune a pretrainde language model whose weights are available in fastai library. It uses the `AWD_LSTM` architecture, a LSTM network without attention but regularized with adaptive dropout (see [here](https://arxiv.org/pdf/1708.02182.pdf)).

# In[29]:


learn = language_model_learner(data_lm, AWD_LSTM, pretrained=URLs.WT103, drop_mult=0.5)


# To see how it is pretrained, we can try to generate sentences with it:

# In[30]:


# xxbos token stands for the beggining of a sentence
learn.predict('What', 100)


# Even if they don't make any sense, the generated sentences seem grammatically correct!

# In[31]:


learn.lr_find()
learn.recorder.plot()


# In[32]:


# First, fit only the last softmax layer
learn.freeze_to(-1)
learn.fit_one_cycle(1, 1e-2)


# In[33]:


# Then unfreeze the model and fit it again
learn.unfreeze()
learn.fit_one_cycle(5, 1e-3)


# We see here that the accuracy is about 30%, which means that our language model predict correctly the next word with a 0.3 probability. This is quite good for 1 min of training but especially only with 10000 training examples ! ^^
# 
# We can also generate text with ou language model:

# In[34]:


print(learn.predict('xxbos', n_words=100))


# The language model now generates sentences adapted to the context of the data we just trained it on!

# In[35]:


# Save the encoder
learn.save_encoder('fine_enc')


# ## Train a sentiment analysis model

# In[36]:


# Preprocess data
data_clas = TextClasDataBunch.from_df('./', df_train, df_test, vocab=data_lm.vocab, bs=32)


# In[37]:


# Build a classifier with the same architechure and weights as the language model we've just trained 
classifier = text_classifier_learner(data_clas, drop_mult=0.5, arch=AWD_LSTM)
classifier.load_encoder('fine_enc')


# In[39]:


classifier.lr_find()
classifier.recorder.plot()


# In[40]:


classifier.fit_one_cycle(1, 1e-2, moms=(0.8, 0.7))


# In[41]:


classifier.recorder.plot_losses()


# In only 1 epoch with 10K samples, we reached more than 90% accuracy !!
# 
# Let's see if we can improve this score with some Hyperparameter tuning techniques presented in the [ULMfit paper](https://arxiv.org/pdf/1801.06146.pdf).

# ## Hyperparameters tuning techniques

# ### Discriminative learning rates
# Discriminative learning rates is a technique which consists on applying different learning rates to each layer of the network. As you go from layers to layers, we need to decrease the learning rate as lowest levels represent the most general knowledge ([Yosinki et al. 2014](https://papers.nips.cc/paper/5347-how-transferable-are-features-in-deep-neural-networks.pdf)).

# In[42]:


classifier = text_classifier_learner(data_clas, drop_mult=0.5, arch=AWD_LSTM)
classifier.load_encoder('fine_enc')
classifier.lr_find()
classifier.recorder.plot()


# In[43]:


classifier.fit_one_cycle(1, slice(1e-4, 1e-2), moms=(0.8, 0.7))


# ### One cycle learning
# One cycle learning is a technique thatis commoly used in fastai library. It consists on a cycle of learning rate, which starts low, increases to the maximum value passed in the `fit_one_cycle` function, then decreases. It prevents our network from overfitting, you can find more information about it in [this paper](https://arxiv.org/pdf/1803.09820.pdf).

# In[44]:


classifier.recorder.plot_lr(show_moms=True)


# ### Gradual unfreezing

# Rather than fine-tuning all layers at once, [ULMfit paper](https://arxiv.org/pdf/1801.06146.pdf) experiments a gradual unfreezing from the last layer to the lowest ones, each time fitting one single epoch.

# In[45]:


classifier = text_classifier_learner(data_clas, drop_mult=0.5, arch=AWD_LSTM)
classifier.load_encoder('fine_enc')
classifier.lr_find()
classifier.recorder.plot()


# In[46]:


classifier.freeze_to(-1)
classifier.fit_one_cycle(1, 1e-2, moms=(0.8, 0.7))


# In[47]:


classifier.unfreeze()
classifier.freeze_to(-2)
classifier.fit_one_cycle(1, slice(1e-4, 5e-3), moms=(0.8, 0.7))


# In[48]:


classifier.unfreeze()
classifier.freeze_to(-3)
classifier.fit_one_cycle(1, slice(1e-5, 1e-3), moms=(0.8, 0.7))


# In[51]:


classifier.unfreeze()
classifier.fit_one_cycle(1, slice(1e-4/100., 1e-4), moms=(0.8, 0.7))


# ## Results

# In[52]:


preds, y, losses = classifier.get_preds(with_loss=True)
interp = ClassificationInterpretation(losses=losses, y_true=y, probs=preds, learn=classifier)
interp.plot_confusion_matrix()


# In[59]:


rev = [
    'The pool was dirty',
    'Loved our stay in this hotel The rooms were amazingly confortable',
    'water was cold and the room not isolated at all', 
    'could have been better',
    'Staff was amazing'
]

for s in rev:
    print(s, '\n=== Predicted:', classifier.predict(s))
    print()


# We've seen that just with a few minutes of training, we can obtain a sentiment analysis model that performs very well!

# In[ ]:




