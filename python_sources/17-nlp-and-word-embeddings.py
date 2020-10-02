#!/usr/bin/env python
# coding: utf-8

# # Ch. 17 - NLP and Word Embeddings
# 
# Welcome to week 4! This week, we will take a look at natural language processing. From Wikipedia:
# > Natural language processing (NLP) is a field of computer science, artificial intelligence concerned with the interactions between computers and human (natural) languages, and, in particular, concerned with programming computers to fruitfully process large natural language data.
# 
# > Challenges in natural language processing frequently involve speech recognition, natural language understanding, and natural language generation.
# 
# While last week was about making computers able to see, this week is about making them able to read. This is useful in the financial industry where large amounts of information are usually presented in form of texts. Starting from ticker headlines, to news reports, to analyst reports all the way to off the record chit chat by industry figures on social media, text is in many ways at the very center of what the financial industry does. In this week, we will take a look at text classification problems and sentiment analysis.
# 
# ## Sentiment analysis with the IMDB dataset
# 
# Sentiment analysis is about judging how positive or negative the tone in a document is. The output of a sentiment analysis is a score between zero and one, where one means the tone is very positive and zero means it is very negative. Sentiment analysis is used for trading quite frequently. For example the sentiment of quarterly reports issued by firms is automatically analyzed to see how the firm judges its own position. Sentiment analysis is also applied to the tweets of traders to estimate an overall market mood. Today, there are many data providers that offer sentiment analysis as a service.
# 
# In principle, training a sentiment analysis model works just like training a binary text classifier. The text gets classified into positive (1) or not positive (0). This works exactly like other binary classification only that we need some new tools to handle text.
# 
# A common dataset for sentiment analysis is the corpus of [Internet Movie Database (IMDB)](http://www.imdb.com/) movie reviews. Since each review comes with a text and a numerical rating, the number of stars, it is easy to label the training data. In the IMDB dataset, movie reviews that gave less then five stars where labeled negative while movies that gave more than seven stars where labeled positive (IMDB works with a ten star scale). Let's give the data a look:

# In[ ]:


from tqdm import tqdm


# In[ ]:


get_ipython().system('ls ../input/glove-global-vectors-for-word-representation')


# In[ ]:


import os

imdb_dir = '../input/keras-imdb/aclImdb_v1/aclImdb' # Data directory
train_dir = os.path.join(imdb_dir, 'train') # Get the path of the train set

# Setup empty lists to fill
labels = []
texts = []

# First go through the negatives, then through the positives
for label_type in ['neg', 'pos']:
    # Get the sub path
    dir_name = os.path.join(train_dir, label_type)
    print('loading ',label_type)
    # Loop over all files in path
    for fname in tqdm(os.listdir(dir_name)):
        
        # Only consider text files
        if fname[-4:] == '.txt':
            # Read the text file and put it in the list
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()
            # Attach the corresponding label
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)


# We should have 25,000 texts and labels.

# In[ ]:


len(labels), len(texts)


# Half of the reviews are positive

# In[ ]:


import numpy as np
np.mean(labels)


# Let's look at a positive review:

# In[ ]:


print('Label',labels[24002])
print(texts[24002])


# And a negative review:

# In[ ]:


print('Label',labels[1])
print(texts[1])


# ## Tokenizing text
# 
# Computers can not work with words directly. To them, a word is just a meaningless row of characters. To work with words, we need to turn words into so called 'Tokens'. A token is a number that represents that word. Each word gets assigned a token. Tokens are usually assigned by word frequency. The most frequent words like 'a' or 'the' get tokens like 1 or 2 while less often used words like 'profusely' get assigned very high numbers.
# 
# We can tokenize text directly with Keras. When we tokenize text, we usually choose a maximum number of words we want to consider, our vocabulary so to speak. This prevents us from assigning tokens to words that are hardly ever used, mostly because of typos or because they are not actual words or because they are just very uncommon. This prevents us from over fitting to texts that contain strange words or wired spelling errors. Words that are beyond that cutoff point get assigned the token 0, unknown.

# In[ ]:


from keras.preprocessing.text import Tokenizer
import numpy as np

max_words = 10000 # We will only consider the 10K most used words in this dataset

tokenizer = Tokenizer(num_words=max_words) # Setup
tokenizer.fit_on_texts(texts) # Generate tokens by counting frequency
sequences = tokenizer.texts_to_sequences(texts) # Turn text into sequence of numbers


# The tokenizers word index is a dictionary that maps each word to a number. You can see that words that are frequently used in discussions about movies have a lower token number.

# In[ ]:


word_index = tokenizer.word_index
print('Token for "the"',word_index['the'])
print('Token for "Movie"',word_index['movie'])
print('Token for "generator"',word_index['generator'])


# Our positive review from earlier has now been converted into a sequence of numbers.

# In[ ]:


# Display the first 10 words of the sequence tokenized
sequences[24002][:10]


# To proceed, we now have to make sure that all text sequences we feed into the model have the same length. We can do this with Keras pad sequences tool. It cuts of sequences that are too long and adds zeros to sequences that are too short.

# In[ ]:


from keras.preprocessing.sequence import pad_sequences
maxlen = 100 # Make all sequences 100 words long
data = pad_sequences(sequences, maxlen=maxlen)
print(data.shape) # We have 25K, 100 word sequences now


# Now we can turn all data into proper training and validation data.

# In[ ]:


labels = np.asarray(labels)

# Shuffle data
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

training_samples = 20000  # We will be training on 10K samples
validation_samples = 5000  # We will be validating on 10000 samples

# Split data
x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]


# ## Embeddings
# 
# As the attuned reader might have already guessed, words and word tokens are categorical features. As such, we can not directly feed them into the neural net. Just because a word has a larger token value, it does not express a higher value in any way. It is just a different category. Previously, we have dealt with categorical data by turning it into one hot encoded vectors. But for words, this is impractical. Since our vocabulary is 10,000 words, each vector would contain 10,000 numbers which are all zeros except for one. This is highly inefficient. Instead we will use an embedding. 
# 
# Embeddings also turn categorical data into vectors. But instead of creating a one hot vector, we create a vector in which all elements are numbers.
# 
# ![Embedding](https://storage.googleapis.com/aibootcamp/Week%204/assets/embeddings.png)

# In practice, embeddings work like a look up table. For each token, they store a vector. When the token is given to the embedding layer, it returns the vector for that token and passes it through the neural network. As the network trains, the embeddings get optimized as well. Remember that neural networks work by calculating the derivative of the loss function with respect to the parameters (weights) of the model. Through backpropagation we can also calculate the derivative of the loss function with respect to the _input_ of the model. Thus we can optimize the embeddings to deliver ideal inputs that help our model. 
# 
# In practice it looks like this: We have to specify how large we want the word vectors to be. A 50 dimensional vector is able to capture good embeddings even for quite large vocabularies. We also have to specify for how many words we want embeddings and how long our sequences are.

# In[ ]:


from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

embedding_dim = 50

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()


# You can see that the embedding layer has 500,000 trainable parameters, that is 50 parameters for each of the 10K words.

# In[ ]:


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])


# In[ ]:


history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))


# Note that training your own embeddings is prone to over fitting. As you can see our model archives 100% accuracy on the training set but only 83% accuracy on the validation set. A clear sign of over fitting. In practice it is therefore quite rare to train new embeddings unless you have a massive dataset. Much more commonly, pre trained embeddings are used. A common pretrained embedding is [GloVe, Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/). It has been trained on billions of words from Wikipedia and the Gigaword 5 dataset, more than we could ever hope to train from our movie reviews. After downloading the GloVe embeddings from the [GloVe website](https://nlp.stanford.edu/projects/glove/) we can load them into our model:

# In[ ]:


glove_dir = '../input/glove-global-vectors-for-word-representation' # This is the folder with the dataset

print('Loading word vectors')
embeddings_index = {} # We create a dictionary of word -> embedding
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt')) # Open file

# In the dataset, each line represents a new word embedding
# The line starts with the word and the embedding values follow
for line in tqdm(f):
    values = line.split()
    word = values[0] # The first value is the word, the rest are the values of the embedding
    embedding = np.asarray(values[1:], dtype='float32') # Load embedding
    embeddings_index[word] = embedding # Add embedding to our embedding dictionary
f.close()

print('Found %s word vectors.' % len(embeddings_index))


# Not all words that are in our IMDB vocabulary might be in the GloVe embeddings though. For missing words it is wise to use random embeddings with the same mean and standard deviation as the GloVe embeddings

# In[ ]:


# Create a matrix of all embeddings
all_embs = np.stack(embeddings_index.values())
emb_mean = all_embs.mean() # Calculate mean
emb_std = all_embs.std() # Calculate standard deviation
emb_mean,emb_std


# We can now create an embedding matrix holding all word vectors.

# In[ ]:


embedding_dim = 100 # We now use larger embeddings

word_index = tokenizer.word_index
nb_words = min(max_words, len(word_index)) # How many words are there actually

# Create a random matrix with the same mean and std as the embeddings
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embedding_dim))

# The vectors need to be in the same position as their index. 
# Meaning a word with token 1 needs to be in the second row (rows start with zero) and so on

# Loop over all words in the word index
for word, i in word_index.items():
    # If we are above the amount of words we want to use we do nothing
    if i >= max_words: 
        continue
    # Get the embedding vector for the word
    embedding_vector = embeddings_index.get(word)
    # If there is an embedding vector, put it in the embedding matrix
    if embedding_vector is not None: 
        embedding_matrix[i] = embedding_vector


# This embedding matrix can be used as weights for the embedding layer. This way, the embedding layer uses the pre trained GloVe weights instead of random ones. We can also set the embedding layer to not trainable. This means, Keras won't change the weights of the embeddings while training which makes sense since our embeddings are already trained.

# In[ ]:


model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen, weights = [embedding_matrix], trainable = False))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()


# Notice that we now have far fewer trainable parameters.

# In[ ]:


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])


# In[ ]:


history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))


# Now our model over fits less but also does worse on the validation set.
# 
# # Using our model
# 
# To determine the sentiment of a text, we can now use our trained model.

# In[ ]:


# Demo on a positive text
my_text = 'I love dogs. Dogs are the best. They are lovely, cuddly animals that only want the best for humans.'

seq = tokenizer.texts_to_sequences([my_text])
print('raw seq:',seq)
seq = pad_sequences(seq, maxlen=maxlen)
print('padded seq:',seq)
prediction = model.predict(seq)
print('positivity:',prediction)


# In[ ]:


# Demo on a negative text
my_text = 'The bleak economic outlook will force many small businesses into bankruptcy.'

seq = tokenizer.texts_to_sequences([my_text])
print('raw seq:',seq)
seq = pad_sequences(seq, maxlen=maxlen)
print('padded seq:',seq)
prediction = model.predict(seq)
print('positivity:',prediction)


# ## Word embeddings as semantic geometry
# 
# One very interesting aspect of embeddings trained on large numbers of words is that they show patterns in which the geometric relationship between word vectors corresponds to the semantic relationship between these words.
# 
# ![Relations](https://storage.googleapis.com/aibootcamp/Week%204/assets/man_woman.jpg)
# 
# 
# In the picture above for instance you can see that the direction of feminine words to their male counterparts is roughly the same. In other words, if you where to substract the word vector for 'woman' from the word 'queen' and add the word vector for 'man' you would arrive at 'king'. This also works for other relationships like comparatives and superlatives. 
# 
# ![Rel Comp Sup](https://storage.googleapis.com/aibootcamp/Week%204/assets/comparative_superlative.jpg)
# 
# This highlights some interesting properties of language in which semantic meanings can be seen as directions which can be added or subtracted.
# 
# A sad side effect of training word vectors on human writing is that it captures human biases. For example it has been [shown](https://www.technologyreview.com/s/602025/how-vector-space-mathematics-reveals-the-hidden-sexism-in-language/) that for word vectors trained on news websites, 'Programmer' - 'Man' + 'Woman' equals 'Homemaker' reflecting the bias in language that assigns the role of homemaker more often to woman than men. Measuring these biases in embeddings and correcting them has become a field of [research on its own](https://www.technologyreview.com/s/602025/how-vector-space-mathematics-reveals-the-hidden-sexism-in-language/) which highlights how even professional writing from news outlets can be biased.

# ## Summary
# In this chapter you have taken the first steps into natural language processing. You have learned about tokenization and word embeddings. You have learned how to train your own embeddings and how to load pre trained embeddings into your model.
