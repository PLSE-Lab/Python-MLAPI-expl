#!/usr/bin/env python
# coding: utf-8

# # Introduction to NLP using Fastai
# > Implementing and decoding the revolutionary [ULMFiT](https://arxiv.org/abs/1801.06146) approach to train a language model on any downstream NLP task.

# ---
# This post was originally posted on my blog [here](https://harish3110.github.io/through-tinted-lenses/natural%20language%20processing/sentiment%20analysis/2020/06/27/Introduction-to-NLP-using-Fastai.html)
# 
# ---

# In continuation to my previous posts [1](https://harish3110.github.io/through-tinted-lenses/fastai/image%20classification/2020/03/29/Building-an-image-classifier-using-Fastai-V2.html), [2](https://harish3110.github.io/through-tinted-lenses/fastai/image%20classification/model%20fine-tuning/2020/04/10/Improving-baseline-model.html), which delved into the domain of computer vision by building and fine-tuning an image classification model using Fastai, I would like to venture into the fascinating domain of Natural Language Processing using Fastai.
# 
# For this post we'll be working on the [Real or Not? NLP with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started/overview) competition dataset on Kaggle to build a text classifier to distinguish between normal tweets and tweets sent out during a natural disaster using the [ULMFiT](https://arxiv.org/abs/1801.06146) approach and decoding this revolutionary paper that changed the NLP schenario for the better in the recent years.

# In[ ]:


# Installing and importing the necessary libraries 
get_ipython().system('pip install fastai2 --quiet')
get_ipython().system('pip install kaggle --quiet')

from fastai2.text.all import *

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


path = Path('../input/nlp-getting-started/')
Path.BASE_PATH = path
path.ls()


# In[ ]:


train = pd.read_csv(path/'train.csv')
test = pd.read_csv(path/'test.csv')


# In[ ]:


train.head() 


# In[ ]:


train['target'].value_counts()


# In[ ]:


test.head()


# In[ ]:


print(f'The training set has {len(train)} records.')
print(f'The test set has {len(test)} records.')


# ## The ULMFiT approach

# The Universal Language Model Fine-tuning (ULMFiT) is an inductive transfer learning approach developed by Jeremy Howard and Sebastian Ruder to all the tasks in the domain of natural language processing which sparked the usage of transfer learning in NLP tasks. 
# 
# **The ULMFiT approach to training NLP models is heralded as the ImageNet moment in the domain of Natural Language Processing** 
# 
# The model architecture used in the entire process of the ULMFiT approach is ubiquitous and is the well-known **AWD-LSTM** architecture. 
# 
# The ULMFiT approach can be braodly explained in the 3 major steps as shown below:

# ![](https://miro.medium.com/max/2000/1*9n9yv4EalUn76yP1Yffhfw.png 'The ULMFiT Process')

# ### Step 1: Training a general corpus language model

# A language model is first trained on a corpus of Wikipedia articles known as Wikitext-103 using a **self-supervised approach**, i.e. using the training labels in itself to train models, in this case training a LM to learn to predict the next word in a sequence. This resulting LM learns the semantics of the english language and captures general features in the different layers. 
# 
# This pretrained language model is trained on 28,595 Wikipedia articles and training process is very expensive and time consuming and is luckily open-sourced in the Fastai library for us to use. 

# ### Side Note: Text Pre-processing
# >Transforming and normalizing texts such that it can be trained on a neural network for language modeling

# In my previous post, [Building an image classifier using Fastai V2](https://harish3110.github.io/through-tinted-lenses/fastai/image%20classification/2020/03/29/Building-an-image-classifier-using-Fastai-V2.html#1.-Create-a-DataBlock:) we look at the datablock API of Fastai and where we apply the `resize` transform that ensures that all images used for training the image classifier model are resized to the same dimensions in order to be able to collate them in the GPU.
# 
# The same type of pre-processing needs to be done for texts in order to train a language model. Whether it's the articles in the Wikipedia 103 dataset or tweets in disaster dataset are of different lengths and can be very long. Thus the tweets corpus i.e. the dataset needs to pre-processed correctly in order to train a neural network on text data. 

# There are many ways the pre-processing for textual data can be done and Fastai approach is to apply the following 2 main transforms to texts:

# ---
# ***Note:*** A transform in Fastai is basically an **almost** reversible function that transforms data into another form(encoding) and also has the capability of getting back the original data(decoding) if needed. 
# 
# ---

# ##### 1. Tokenization
# 
# The first step is to gather all the unique `tokens` in the corpus being used. 
# 
# A `token` can be defined in numerous ways depedning on the person creating the language model based on the granularity level i.e. the smallest part of the text they would like to consider. In the simplest scenario, a word can be considered as the token.
# 
# So the idea is to get a list of all the unique words used in the general domain corpus(Wikipedia 103 dataset) and our added downstream dataset(Disaster tweets dataset) to build a vocabulary for training our language model.

# In[ ]:


# Let's take an example text from our training set to show a tokenization example

txt = train['text'].iloc[0]
txt


# In[ ]:


# Initializing the default tokenizer used in Fastai which is that of Spacy called `WordTokenizer`
spacy = WordTokenizer() 

# Wrapping the Spacy tokenizer with a custom Fastai function to make some custom changes to the tokenizer
tkn = Tokenizer(spacy) 

tkn(txt)


# In[ ]:


txts = L([i for i in train['text']])


# In[ ]:


# Setting up a tokenizer on the entire dataframe 'train'
tok = Tokenizer.from_df(train)
tok.setup(train)

toks = txts.map(tok)
toks[0]


# ---
# **Note:** The special tokens you can see above starting with 'xx' are special fastai tokens added on top of the spacy tokenizer used to indicate certain extra meanings in the text data as follows:
# 
# - `xxbos`:: Indicates the beginning of a text (here, a review)
# - `xxmaj`:: Indicates the next word begins with a capital (since we lowercased everything)
# - `xxunk`:: Indicates the next word is unknown
# 
# ---

# As mentioned above `Tokenizer` is a Fastai transform, which is basically a function with and `encodes` and `decodes` method available to tokenize a text and return it back to **almost** the same initial state.

# In[ ]:


tok.encodes(toks[0])


# In[ ]:


tok.decode(toks[0])


# The reason we don't get the original string back when applying `decode` is because the default tokenizer used in this case isn't `reversible`. 

# ##### 2. Numericalization
# 
# The next step in the pre-processing step is to index the tokens created earlier so that they can easily accessed. 

# In[ ]:


num = Numericalize()
num.setup(toks)
nums = toks.map(num)
nums[0][:10]


# In[ ]:


num.encodes(toks[0])


# In[ ]:


num.decode(nums[0][:10])


# ### Step 2: Fine-tuning pretrained LM to downstream dataset

# Despite having a vast language model pre-trained, it's always likely that the specific downstream task we would like to build our NLP model is a part of a slightly different distribution and thus  need to fine-tune this Wikitext 103 LM.
# 
# This step is much faster and it converges much faster as there will be an overlap to the general domain dataset. It only needs to adapt to the idiosyncrasies of the language used and not learn the language per say. 

# Since NLP models are more shallow in comparison to a computer vision model, the fine-tuning approaches need to be different and thus the paper provides novel fine-tuning techniques to do so:

# #### Discriminative Fine-tuning
# 
# Since different layers of the model capture different types of information and thus they should be fine-tuned to different extents. 
# 
# This idea is similar as the use of discriminative learning rates used in CV applications which I explained in detail in my previous [post](https://harish3110.github.io/through-tinted-lenses/image%20classification/2020/04/10/Improving-baseline-model.html#Discriminative-learning-rates).

# #### Slanted Triangular Learning Rates
# 
# The idea behind slanted learning rates is that for a pretrained language model to adpat/fine-tune itself to the downstream dataset, the fine-tuning process should ideally converge faster to asuitable region in the parameter space and thern refine its parameters there. 
# 
# So the slanted learning rates approach first linearly increases the learning rates for a short period and then linearly decays the learning rate slowly which is a modification of of Leslie Smith's traingular learning rate approache where the increase and decrease is almost the same. 

# ![](https://miro.medium.com/max/1096/1*QptmUluWXteT6oI5bD22rw.png 'Slanted Triangular Learning Rates')

# #### Creating a dataloader
# > Putting the pre-processed data in batches of text sequences for fine-tuning the language model

# In[ ]:


# dataset for fine-tuning language model which only needs the text data

df_lm = pd.concat([train, test], axis=0)[['text']]
df_lm.head()


# ---
# ***Note:*** An important trick used in creating a dataloader here is that we use all the data available to us i.e train and test data. In case we had a dataset with unlabeled reviews we could also use that to fine-tune the pre-trained model better since this step doesn't need labels and is self-supervised. 
# 
# ---

# Creating a dataloader for self-supervised learning task which tries to predict the next word in a sequence as represented by `text_` below. 
# 
# **Fastai handles text processing steps like tokenization and numericalization internally when `TextBlock` is passed to `DataBlock`.**

# In[ ]:


dls_lm = DataBlock(
    blocks=TextBlock.from_df('text', is_lm=True),
    get_x=ColReader('text'), 
    splitter=RandomSplitter(0.1) 
    # using only 10% of entire comments data for validation inorder to learn more
)


# In[ ]:


dls_lm = dls_lm.dataloaders(df_lm, bs=64, seq_len=72)


# ---
# ***Note:***
# 
# - Select the batch size `bs` based on how much your GPU can handle without running out of memory
# - The sequence length `seq_len` for the data split used here is the default sequence length used for training the Wikipedia 103 language model
# 
# ---

# In[ ]:


dls_lm.show_batch(max_n=3)


# In[ ]:


# Saving the dataloader for fast use in the future

# torch.save(dls_lm, path/'disaster_tweets_dls_lm.pkl')


# In[ ]:


# To load the Dataloaders in the future

# dls_lm = torch.load(path/'disaster_tweets_dls_lm.pkl')


# ---
# #### Fine-tuning the language model
# 
# Fine-tuning Wikitext 103 based LM to disaster tweets using ULMFiT fine-tuning methodologies. This fine-tuned LM can thus be used as the base to classify disaster texts in the next step.
# 
# The common metric used in CV models is accuracy but in sequence based models we use something called **perplexity** which is basically exponential of the loss as follows:
# 
# ```
# torch.exp(cross_entropy)
# ```

# In[ ]:


#fine-tuning wikitext LM to disaster tweets dataset

learn = language_model_learner(
    dls_lm, AWD_LSTM,
    metrics=[accuracy, Perplexity()]).to_fp16()


# In[ ]:


learn.model


# ---
# ##### Embedding Layer
# 
# We can see that the above `AWD LSTM` architecture used in ULMFiT has a bunch of layers called **embedding** layers as the input here.
# 
# The pre-processed text and the batching of data using dataloaders is followed by passing this data into an embedding layer which can be considered as a small neural network by itself which is used to calculate token i.e. word dependencies in the dataset. These layers are trained along with the main neural network model and learns relationships between words in the dataset along the way. 
# 
# An embedding layer is a computationally efficient method to represent tokens in a lesser dimension space, being less sparse and as a look-up table for all tokens in our dataset which captures relationships between the tokens. 
# 
# It's a much more computationally efficient approach to the traditional `one-hot encoding` appraoch which can make these types of task really expensive and inefficient.

# ![](https://www.fast.ai/images/kittenavalanche.png "In the above embediing layer learned, vectors for baby animal words are closer together, and an unrelated word like 'avalanche' is further away")

# If you would like to know more about word embedding check out this amazing [video](https://www.youtube.com/watch?v=25nC0n9ERq4) by Rachael Thomas, co-founder of Fastai.
# 
# ---

# In[ ]:


learn.lr_find()


# Let's train the last layer of the model using a learning rate of `1e-2` based on the above learning rate finder plot using Leslie Smith's [1 Cycle Training](https://arxiv.org/abs/1708.07120) approach.

# In[ ]:


learn.fine_tune(5, 1e-2)


# Once we have fine-tuned out LM to our downstream task, we save the `encoder` part of the model which portion of the model except the final layer that predicts the next word in the sequence. 
# 
# We can then use this`encoder` part, which is the portion that learns the language semantics, as our base to build a disaster tweets classification model. 

# In[ ]:


# Saving the encoder

learn.save_encoder('finetuned')


# ---

# ### Step 3: Training a classifier on the downstream NLP task

# Now that we have a language model fine-tuned to our downstream NLP dataset we can use the encoder portion of the fine-tuned language model which is the part that learns the features of the language used in the downstream dataset as the base to build a text classifier for tasks such as sentiment analysis, spam detection, fraud detection, document classifcation etc. 

# The encoder saved is then appended by a simple classifier consisting of two additional linear blocks consisting of the  standard batch normalization and dropout, with ReLU activations for the intermediate layer and a softmax activation at the last layer for the classification purpose. 

# Fine-tuning a classifier is a very critical task in a transfer learning method and is the main reason why transfer learning approaches failed until ULMFiT came along. 
# 
# Overly aggressive fine-tuning can result in **catastrophic forgetting** and too cautious fine-tuning can lead to extremely slow convergence. 
# 
# To tackle this problem, ULMFiT introduces a novel fine-tuning technique in **gradual unfreezing** besides also using **slanted triangular learning rates** and **discriminative fine-tuning** to successfully train a classifier using a pre-trained LM.

# #### Gradual Unfreezing

# The idea behind gradual unfreezing is that fine-tuning a classifier on all layers can result in catastrophic forgetting and thus each layer staring form the las layer is trained one after the other by freezing all the lower layers and only training the layer in question. 
# 
# The paper empirically found that after training the last layer of the model with a learning rate of `lr`, the subsequent layers can be trained one after another by reducing `lr` by a factor of `2.6`.

# #### Backpropagation Throught Time for Text Classification (BPT3C)

# Since the model architecture for training and fine-tuning the language is that of an LSTM, the paper implements the backpropagation through time(BPTT) approach to be able propagate gradients without them exploding or vanishing. 
# 
# In the ULMFiT approach, a modification to the traditional BPTT is made specifically in the fine-tuning classifier phase called **BPTT for Text Classification(BPT3C)** to make fine-tuning a classifier for large documents feasible. 

# Steps in BPT3C:
# - The document is divided into fixed length batches of size 'b'. 
# - At the beginning of each batch, the model is initiated with the final state of the previous batch by keeping track of the hidden states for mean and max-pooling. 
# - The gradients are back-propagated to the batches whose hidden states contributed to the final prediction. 
# - In practice, variable length back-propagation sequences are used. 
# 

# #### Concat Pooling
# 
# Since signals for classifying texts can exist anywhere and are not only limited to last word in the sequence, the ULMFiT approach also proposes to concatenate the last time step of the document by max-pooling and mean-pooling representations to provide more signal and better training

# #### Creating the classifier dataloader
# 
# Ensure that the sequence length and vocab passed to the `TextBlock` is same as that given while fine-tuning LM 

# In[ ]:


blocks = (TextBlock.from_df('text', seq_len=dls_lm.seq_len, vocab=dls_lm.vocab), CategoryBlock())
dls = DataBlock(blocks=blocks,
                get_x=ColReader('text'),
                get_y=ColReader('target'),
                splitter=RandomSplitter(0.2))


# In[ ]:


dls = dls.dataloaders(train, bs=64)


# In[ ]:


dls.show_batch(max_n=3)


# In[ ]:


len(dls.train_ds), len(dls.valid_ds)


# #### Defining the learner

# In[ ]:


learn = text_classifier_learner(dls, AWD_LSTM, metrics=[accuracy, FBeta(beta=1)]).to_fp16()
learn.load_encoder('finetuned')


# In[ ]:


learn.model


# ---
# #### Training the classifier
# > Fine-tuning a text classifier using gradual unfreezing, slanted learning rates and discriminating learning techniques.

# In[ ]:


learn.fit_one_cycle(1, 1e-2)


# In[ ]:


# Applying gradual unfreezing of one layer after another

learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(1e-3/(2.6**4),1e-2))


# In[ ]:


learn.freeze_to(-3)
learn.fit_one_cycle(1, slice(5e-3/(2.6**4),1e-2))


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(2, slice(1e-3/(2.6**4),3e-3))


# In[ ]:


learn.save('final_model')


# In[ ]:


learn.export()


# ---

# ## Creating a Kaggle submission file

# In[ ]:


sub = pd.read_csv(path/'sample_submission.csv')
sub.head()


# In[ ]:


dl = learn.dls.test_dl(test['text'])


# In[ ]:


preds = learn.get_preds(dl=dl)


# In[ ]:


# Let's view the output of a single row of data

preds[0][0].cpu().numpy()


# In[ ]:


# Since it's a multi-class problem and it uses softmax on the binary classes, 
# Need to calculate argmax of the output to get the best class as follows 

preds[0][0].cpu().argmax(dim=-1)


# In[ ]:


sub['target'] = preds[0].argmax(dim=-1)


# In[ ]:


sub.head()


# The above submission acheived a score of 0.80447 on the competition leaderboard. 

# ## Conclusion
# 
# In this post we have seen how to build a fine-tuned language model for any textual data corpus which captures the semantics of the dataset. The encoder part of this fine-tuned language model was then used to build a pretty-decent text classifier that can identify tweets describing a natural disaster. 
# 
# In the upcoming posts, I would try to delve deeper into building advanced NLP models like the Transformer architecture and researching other famous research papers in the domain of NLP. 

# ## References
# 
# - Fastai v2: [Documentation](www.dev.fasta.ai)
# - Fastbook Chapter 10: [NLP Deep Dive](https://github.com/fastai/fastbook/blob/master/10_nlp.ipynb)
# 
# ---
# 
# Happy learning, stay at home and stay safe! :)

# ---
