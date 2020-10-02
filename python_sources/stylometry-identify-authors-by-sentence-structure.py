#!/usr/bin/env python
# coding: utf-8

# # Stylometry - or: Identify authors by sentence structure using "deep learning"

# ## Introduction & Features
# We live in changing times: With the way our social media activity increases, we now have a shift away from traditional communication towards a text-based one. While these type of messages may have significant advantages and allows us to persist in a globalized world, some question and problem arise we have to face. Undoubtfully, one crucial ability is the identification of an author through its texts; without direct contact, fakes are elsewhere hardly recognizable.
# The science of identifying authors by there writing is called Stylometry. With our ability to access big data and having sufficient computational power nowadays, the accuracy we may gain in such kinds of tasks is quite impressive.
# 
# 
# ![An analysis by Ali Arsalan Kazmi showing the two authors of the speeches of the Prime Minister of Pakistan ](http://aliarsalankazmi.github.io/blog_DA/assets/img/bcn_char4g_2.png)
# *A study by Ali Arsalan Kazmi identifying the two authors of the speeches of the Prime Minister of Pakistan*
# 
# 
# In this kernel, instead of classifying the texts we will use some Stylometry to identify the three authors of different horror stories. Every data mining algorithm is only as good as the features and data it is working on, so we might have a first look at the available features for such kind of analysis (according to "Writeprints: A Stylometric Approach to Identity-Level Identification and Similarity Detection in Cyberspace" by A. ABBASI)
# 
# * Lexical features: Lexical features describes the semantics of the words used by the author. Vocabulary richness and word-length distributions are classic examples.
# * Syntactic features: Such kind of features rely on an analysis of typical sentence structure considering elements like function words, punctuation, and part-of-speech tags.
# * Structural features: Structural features targets the actual composition of the text discussing its organization and layout. Due to the short extracts without any further information, this seems not applicable here.
# * Content-specific features: This type of features based on an analysis of essential keywords and phrases on specific topics like in scientific publication. Like the feature set before not applicable in our context.
# * Idiosyncratic features: Individual usage anomalies like misspellings and grammatical mistakes are taken into consideration. After the editing process of all the texts, it is unlikely that any valuable results remain.
# 
# In short, we seem to have the choice between a lexical and a syntactic approach. In a productive situation, one would probably focus on the first one; it is proven they usually performed far better than the pure sentence structure. So, if you are interested in such an analysis and gaining optimal performance, please take a look into all the other amazing kernels out there. In the following, we will focus on a more scientific question:
# 
# **Utilizing Deep learning technology, is it possible to identify an author just by the structure of the sentences he or she writes?**
# 
# ## Tools
# During our journey into data science, we will utilize a wide range of the epic tools freely available for Python. In the following, I just want to go through them briefly.
# 
# #### Data management: *Pandas*
# Pandas is a library for loading and modification of different data structures like series and tables. Especially its wide abilities for loading and saving into different formats will be utilized in the following.
# 
# #### Natural Language Processing: *SpaCy*
# When the terms "Python," "text intelligence," and "deep learning" are mentioned in one sentence, often the name "SpaCy" is used, too. The rather young project wants to be a performant alternative to NLTK and provided advanced features like a GloVe vectorization of words out of the box. All these features are not needed for our current experiment but probably more useful when targetting lexical features.  Personally, I just prefer SpaCy's style of handling problems, but NLTK would probably lead to the same results in that particular application as it used a variation of SpaCy's POS tagger. Feel free to use whatever tool you want.
# 
# #### Array handling: *Numpy*
# Numpy is one of THE tools in Python's machine learning universe. It allows the efficient handling of huge multidimensional arrays with minimal copy costs and memory footprint. Depending on the style of usage, the performance benefits in comparison with Python lists are in general quite impressive. 
# 
# #### Neuronal network layout: *Keras*
# Keras is a wonderful simple wrapper for designing and training neural networks using the performance of the far more complex frameworks Theano and TensorFlow. With its ability to add simply one layer after the other, it is an ideal starting point for own experiments with deep learning.
# 
# #### Swiss army knife for almost anything else: *scikit-learn*
# Last but not least one of the essential library for classical machine learning in Python. Independently if you want to use a PCA, a 10-fold cross validation, or a data pipeline: There is already a class for that. Its data-driven pipelines utilizing transformer and classifier will guide us through our tour.
# 
# First of all, we import all the necessary stuff we will need in the following. After the loading of training data, we convert the names of the authors into strings for an easier further processing and load the pre-computed models SpaCy relies on.

# In[ ]:


import pandas as pd
import spacy
import numpy as np

AUTHORS = { 'EAP' : 0, 'HPL' : 1, 'MWS' : 2 }

# Load SpaCy's models
SPACY = spacy.load('en')

# Load the training data
dataset = pd.read_csv("../input/train.csv")

# Convert the author strings into numbers
dataset['author'] = dataset['author'].apply(lambda x: AUTHORS[x])


# # Method

# What do we have at the beginning of our experiment? From a computational point of view just a massive amount of strings; containing different numbers of sentences and words. In short: A complete nightmare for a machine learning algorithm. To gain valuable knowledge out of it, we have to preprocess it: We need to extract the actual entities like words and punctuation and assign a syntactical meaning to those by detecting their role in the sentence (are they nouns? Or a type of punctuation?). We do not have to implement this "Tokenization" and "Part of Speech - tagging" ourselves, instead, we will use SpaCy for this task.
# After this step, we will have many different sized lists containing numbers representing the actual types of words. By default, machine learning algorithms are unable to deal with such kind of non-fixed length data. But what do we do now? It is trivial to find out that the smallest document contains just three entities (two words and a fullstop), while some have over 800!

# In[ ]:


sentence_lengths = np.fromiter((len(t.split()) for t in dataset['text']), count=len(dataset['text']), dtype='uint16')

print("Minimal sentence length {}: '{}'".format(
    np.min(sentence_lengths),
    dataset['text'][np.argmin(sentence_lengths)]
))

print("Maximal sentence length", np.max(sentence_lengths))


# We might think about of filling the lists just with some 0's at the end until they have all the size of the most extensive list. While no information would get lost in such a case, the speed we archive in training of algorithms depends highly on the input size. Should we get a massive amount of 0 just because of a small amount of outliers? Certainly not! As an alternative, we will use a combined approach: We will use a size where 95% of all elements are completely covered, fill shorter ones and truncate the small amount outliers for the sake of training efficiency.
# 
# Due to the fact that we are using scikit-learn, we will model this preprocessing as a so called Transformer. Keep in mind that you should in general not rely on global variables like I do with SpaCy in the following; unfortunately, the models are not serializable and will make our usage in further rather painful.

# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin

class PosPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, length_percentile = 95):
        self.length_percentile = length_percentile
        self._standartization_factor = 0

    def transform(self, X, *_):
        assert (self.sentence_size is not None), "Fitting required"
        
        # Create the output matrix
        result = np.zeros((len(X), self.sentence_size), dtype='uint8')
        
        # Tokenize and POS tag all the documents using multi-threading
        for i, x in enumerate(SPACY.pipe(X, batch_size=500, n_threads=-1)):
            # Store the POS-tags
            tags = np.fromiter((token.pos for token in x), dtype='uint8', count=len(x))
            
            # Pad and truncate data, if necessary, and store them in result
            last_index = len(tags) if len(tags) < self.sentence_size else self.sentence_size
            result[i, :last_index] = tags[:last_index]
        
        # Generate the factor one time to ensure applying the same factor at the next transformations
        if self._standartization_factor == 0:
            self._standartization_factor = np.min(result[result != 0]) - 1
    
        # Standartize all valid elements to count from 1
        result[result != 0] -= self._standartization_factor
        return result

    def fit(self, X, *_):
        # Define an optimal sentence size covering a specific percent of all sample
        self.sentence_size = int(np.percentile([len(t.split()) for t in X], self.length_percentile))
        return self
    
    def fit_transform(self, X, *_):
        self.fit(X)
        return self.transform(X)


# Now we have our set of features and our labels - let's get ready for rumble! Before we start to dive in into Deep Learning, it is reasonable to define a foundation for a further evaluation of the performance.
# For such cases, RandomForest is often a good candidate for a first prediction. This machine learning algorithm has only a small number of hyperparameter and frequently performs even better than neuronal networks while using only a fraction of the training time (i.e., on my experiments with the classification of spam you might find on GitHub). Due to the fact that we modeled our preprocessor as Transformer, we could now use the beautiful syntax for training and evaluation provided by scikit. I hope, I did not promise too much... ;)

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

syntax_pipeline = Pipeline([
        ('pre', PosPreprocessor()), 
        ('predictor', RandomForestClassifier(n_estimators=100))
])

# Get a testing split for further tests
test_split = int(0.1 * len(dataset))

# Train the model and evaluate it on former unseen testing data
syntax_pipeline.fit(dataset['text'][:test_split], dataset['author'][:test_split])
syntax_pipeline.score(dataset['text'][test_split:], dataset['author'][test_split:])


# Even if the extraction of knowledge is undoubtful a hard task, these results look not that promising. We have to remind ourselves that the probability of having a match just by rolling a dice is 33%. It seems, that the linear way RandomForest and most of the other machine learning algorithms work is unable to get the actual syntax of the sentence. What should we use instead?! 
# The probably most promising solution would be the usage of a specific group of Artificial Neural Networks: Recurrent Neural Networks.
# 
# Recurrent neural networks are, no surprise, a specific part of the family of artificial neural networks. Unlike the classical feed-forward neural networks, RNNs were conducted as a more potent tool especially for tasks involving the use of sequences of features rather than only features itself. While former type of network generates a fixed output vector out of a fixed input vector in a fixed number of computational steps, RNNs work on sequences of these vectors. Foundation of this is their ability to have an interior state which gets adapted between different samples and allows therefore further consideration of spatial frequency and common pattern. These capabilities were successfully applied in speech recognition, machine translation and even the generation of text. Instead of the classical, simple RNN neurons, we will evaluate LSTM and GRU neurons in this experiment. Both were designed to face the "long-term dependency problem": While RNNs can find dependencies between adjacent elements easily, further context commonly needed in language processing tasks are not possible. 
# 
# The most crucial part in using every artificial neuronal network is its actual design determining its final performance. While there are some heuristics for some of its hyperparameter, mostly we have to rely on extensive testing. We - as lazy data scientist - do not want to search for ourselves and therefore delegate the work to scikit-learn by creating the following classifier, which allows an automatization of the training process. No worries - the code might be long, but I explain it afterward.

# In[ ]:


from sklearn.base import BaseEstimator, ClassifierMixin
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, SimpleRNN, Activation, Dropout

class RnnClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 batch_size=32,
                 epochs=3,
                 dropout=0,
                 rnn_type='gru',
                 hidden_layer=[64, 32]):
        
        # How many samples are processed in one training step?
        self.batch_size = batch_size
        # How long should the artificial neural network train?
        self.epochs = epochs
        # How much dropout do we put into the model to avoid overfitting?
        self.dropout = dropout
        # Which type of RNN do we want?
        self.rnn_type = rnn_type
        # Do we have hidden layer? If yes, how many which how many neurons?
        self.hidden_layer = hidden_layer
        
        self._rnn = None
        self._num_classes = None
        self._num_words = None

    def fit(self, X, Y=None):
        assert (Y is not None), "Y is required"
        assert (self.rnn_type in ['gru', 'lstm', 'simple']), "Invalid RNN type"

        # How many different tags do we have?
        self._num_words = np.max(X) + 1
        
        # How many classes should we predict?
        self._num_classes = np.max(Y) + 1
        
        node_type = None
        if self.rnn_type is 'gru':
            node_type = GRU
        elif self.rnn_type is 'lstm':
            node_type = LSTM
        else:
            node_type = SimpleRNN
        
        # Transfer the data into a appropiated shape
        X = self._reshape_input(X)

        # Ready for rumble?! Here the actual neural network starts!
        self._rnn = Sequential()
        self._rnn.add(node_type(X.shape[1], 
                                input_shape=(X.shape[1], self._num_words), 
                                return_sequences=(len(self.hidden_layer) > 0)
                               ))
        
        # Add dropout, if needed        
        if self.dropout > 0:
            self._rnn.add(Dropout(self.dropout))

        # Add the hidden layers and their dropout
        for (i, hidden_neurons) in enumerate(self.hidden_layer):
            sequences = i != len(self.hidden_layer) - 1
            
            self._rnn.add(node_type(hidden_neurons, return_sequences=sequences))
            if self.dropout > 0:
                self._rnn.add(Dropout(self.dropout))
        
        # Add the output layer and compile the model
        self._rnn.add(Dense(3, activation='softmax'))
        self._rnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Convert the results in the right format and start the training process
        Y = to_categorical(Y, num_classes=self._num_classes)
        self._rnn.fit(X, Y, epochs=self.epochs,
                      batch_size=self.batch_size,
                      verbose=0)

        return self

    def predict(self, X, y=None):
        if self._rnn is None:
            raise RuntimeError("Fitting required before prediction!")
        
        # 'Softmax' returns a list of probabilities - just use the highest onw
        return np.argmax(
            self._rnn.predict(
                self._reshape_input(X), 
                batch_size=self.batch_size
        ))

    def score(self, X, y=None):
        assert (y is not None), "Y is required"

        # Evaluate the model on training data
        return self._rnn.evaluate(
            self._reshape_input(X), 
            to_categorical(y, num_classes=self._num_classes)
        )[1]
    
    def _reshape_input(self, X):
        result = np.resize(X, (X.shape[0], X.shape[1], self._num_words))
        for x in range(0, X.shape[0]):
            for y in range(0, X.shape[1]):
                 result[x, y] = to_categorical(X[x, y], num_classes=self._num_words)[0]
        return result


# This is a big chunk of code, isn't it? The actual design of the RNN it thereby rather trivial, most of the code is used to convert the samples from and into the needed format. Instead of dealing with numbers like we as humans would do it, Artifical Neural Networks prefer the so-called binary "One Hot Encoding." Therefore, instead of using  categories "1", "2," and "3" for the author, we have to use binary vectors of the form "(1, 0, 0)", "(0, 1, 0)," and "(0, 0, 1)" for the output. Following the same logic for the input, we have to add a third dimension, encoding the numbers at a specific sample in a specific position as such a vector.
# 
# The RNN is after all these transformations merely a sequential list of layers. We start with a layer with the size of the sentence we defined earlier; therefore every potential word matches precisely one neuron. Afterwards, a dynamic number of the so-called hidden layer between input and output is added. If specified in the parameters, we add so-called "Dropout" between them: Especial RNNs tends to overfit the data rather early and learn therefore more the training data than the actual model behind. Dropout adds a specified amount of random noise to the learning process to avoid this. In the end, we use a simple layer with three neurons and the softmax activation function to gain the final prediction. Each of its neurons will have a float between 0 and 1 describing the likelihood for that class to occur. By just getting the most likely index, we get the class of our entry according to this prediction. 
# 
# After we build this classifier, we are now able to run a GridSearch. This fancy term just describes a systematical evaluation of all the different permutations of a range of parameter we provided to find the one leading to an optimal model. Again, sci-kits pretty simple syntax leads to a straightforward structure.

# In[ ]:


from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.pipeline import Pipeline

# Create the pipeline
syntax_pipeline = Pipeline([
        ('pre', PosPreprocessor()), 
        ('rnn', RnnClassifier(batch_size=64))
])

# Create the grid search with specifying possible parameter
optimizer = GridSearchCV(syntax_pipeline, {
    'rnn__rnn_type' : ('lstm', 'gru'), 
    'rnn__hidden_layer' : ([], [64], [64, 32]),
    'rnn__epochs': (30, 60, 90),
    'rnn__dropout': (0, 0.15, 0.3)
}, cv=ShuffleSplit(test_size=0.10, n_splits=1, random_state=0))

# Start the search: That will take some time!
# optimizer.fit(dataset['text'], dataset['author'])


# After we train the model with 10% testing data, we might get the following results:
# 
# | Rank | Testing Score | Method | Hidden layer | Dropout | Epochs |
# |------|---------------|--------|--------------|---------|--------|
# | 1    | 63.78%        | GRU    | 64           | 30%     | 30     |
# | 2    | 62.33%        | GRU    | 64 + 32      | 15%     | 30     |
# | 3    | 62.17%        | GRU    | -            | 15%     | 30     |
# 
# As we can see, the more simple GRU neurons seems to be perform better than the more complex LSTM's on that kind of problem. Moreover, the number of hidden layers seems not to be crutial. With this evidencene, we are able to re-run the search due to our reduced scope we might have to consider. As we have to find the sweet spot when additional epochs do not longer result in improvement, we will check these parameter in detail:

# In[ ]:


from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.pipeline import Pipeline

# Create the pipeline
syntax_pipeline = Pipeline([
        ('pre', PosPreprocessor()), 
        ('rnn', RnnClassifier(batch_size=64, rnn_type='gru', hidden_layer=[64]))
])

# Create the grid search with specifying possible parabeter
optimizer = GridSearchCV(syntax_pipeline, {
    'rnn__epochs': (25, 30, 35, 40, 45, 50),
    'rnn__dropout': (0.25, 0.3, 0.35, 0.4)
}, cv=ShuffleSplit(test_size=0.10, n_splits=1, random_state=0))

# Start the second search: Again, that will take some time!
# optimizer.fit(dataset['text'], dataset['author'])


# In this case, I was able to gain an optimal performance of 65.3% with 35% dropout and a training over 40 epochs. Keep in mind: In roughly 2 out of 3 texts, the author was correctly identified only by the typical syntax he or she used! Just imagine the power of an analysis if we would now introduce the normally far more significant lexical features. Even if deep learning might be not suitable to be visualized in a straightforward way, its underlying power with the resulting opportunities is obvious.
# 
# Now, you might try to extend these results. Try to embed the semantic meaning and I am certain, the results will be quite impressive. It would be an honor for me to read your remarks or questions in the comments. I wish good luck with your further experiments!
