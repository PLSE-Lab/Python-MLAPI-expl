#!/usr/bin/env python
# coding: utf-8

# # Introduction: Book Recommendation System
# 
# In this notebook, we will build a book recommendation system based on a simple principle: books with Wikipedia pages that link to similar Wikipedia pages are similar to each other. In order to create this representation of similar books, we'll use the concept of neural network entity embeddings, mapping each book and each Wikipedia link (Wikilink) to a 50-number vector. 
# 
# The idea of entity embeddings is to map high-dimensional categorical variables to a low-dimensional _learned_ representation that _places similar entities closer together in the embedding space_. If we were to one-hot-encode the books (another representation of categorical data) we would have a 37,000 dimension vector for each book, with a single 1 indicating the book. In a one-hot encoding, similar books would not be "closer" to one another. By  training a neural network to learn entity embeddings, we not only get a reduced dimension representation of the books, we also get a representation that _keeps similar books closer to each other_. Therefore, the basic approach for a recommendation system is to create entity embeddings of all the books, and then for any book, find the closest other books in the embedding space. [Fortunately, thanks to a previous notebook](https://github.com/WillKoehrsen/wikipedia-data-science/blob/master/Downloading%20and%20Parsing%20Wikipedia%20Articles.ipynb), we have access to every single book article on Wikipedia, which will let us create an effective recommendation system.
# 
# ## Approach
# 
# To create entity embeddings, we need to build an embedding neural network and train it on a supervised machine learning task that will result in similar books (and similar links) having closer representations in embedding space. The parameters of the neural network - the weights - are the embeddings, and so during training, these numbers are adjusted to minimize the loss on the prediction problem. In other words, the network tries to accurately complete the task by changing the representations of the books and the links. 
# 
# Once we have the embeddings for the books and the links, we can find the most similar book to a given book by computing the distance between the embedded vector for that book and all the other book embeddings. We'll use the cosine distance which measures the angle between two vectors as a measure of similarity (another valid option is the Euclidean distance). We can also do the same with the links, finding the most similar page to a given page. (I use links and wikilinks interchangeably in this notebook). The steps we will follow are:
# 
# 1. Load in data and clean
# 2. Prepare data for supervised machine learning task
# 3. Build the entity embedding neural network
# 4. Train the neural network on prediction task
# 5. Extract embeddings and find most similar books and wikilinks
# 6. Visualize the embeddings using dimension reduction techniques 
# 
# ### Supervised Machine Learning Task: Map Books to Links
# 
# For our machine learning task, we'll set up the problem as identifying whether or not a particular link was present in a book article. The training examples will consist of (book, link) pairs, with some pairs true examples - actually  in the data - and others negative examples - do not occur in the data. It will be the network's job to adjust the entity embeddings of the books and the links in order to accurately make this classification. Although we are training for a supervised machine learning task, our end objective is not to make accurate predictions on new data, but learn the best entitiy embeddings, so we do not use a validation or testing set. We use the prediction problem as a means to an end rather than the final outcome. 
# 
# ## Neural Network Embeddings
# 
# Neural Network embeddings have proven to be very powerful concepts both for modeling language and for representing categorical variables. For example, the [Word2Vec word embeddings](https://www.tensorflow.org/tutorials/representation/word2vec) map a word to a vector based on training a neural network on millions of words. These embeddings can be used in any supervised model because they are just numerical representations of categorical variables. Much as we one-hot-encode categorical variables to use them in a random forest for a supervised task, we can also use entity embeddings to include categorical variables in a model. The embeddings are also useful because we can find entities that are close to one another in embedding space which might - as in a book recommendation system - allow us to find the most similar categories among tens of thousands of choices. 
# 
# We can also use the Entity Embeddings to visualize words or categorical variables, such as creating a map of all books on Wikipedia. The entity embeddings typically are still high-dimensional - we'll use 50 numbers for each entity - so we need to use a dimension reduction technique such as TSNE or UMAP to visualize the embeddings in lower dimensions. (These are both manifold embedding methods so in effect we will embed the embeddings for visualization!) We'll take a look at doing this at the end of the notebook and later will upload the embeddings into a application custom-built for this purpose ([projector.tensorflow.org](https://projector.tensorflow.org)). Entity embeddings are becoming more widespread thanks to the ease of development of neural networks in Keras and are a useful approach when we want to represent categorical variables with vectors that place similar categories close to one another. Other approaches for encoding categorical variables do not represent similar entities as being closer to one another, and entity embedding is a _learning-based method_ for this important task.
# 
# Overall, this project is a great look at the potential for neural networks to create meaningful embeddings of high dimensional data and a practical application of deep learning. The code itself is relatively simple, and the Keras library makes developing deep learning models enjoyable!
# 
# The code here is adapted from the excellent [Deep Learning Cookbook](http://shop.oreilly.com/product/0636920097471.do), the [notebooks for which can be found on GitHub](https://github.com/DOsinga/deep_learning_cookbook). Check out this book for practical applications of deep learning and great projects! 
# 
# __Note to Kaggle users__: Make sure GPU is enabled and Internet is connected on the Settings pane.

# ## Read in Data
# 
# The data is stored as json with line for every book. This data contains every single book article on Wikipedia which was parsed in the [Downloading and Parsing Wikipedia Data Notebook](https://github.com/WillKoehrsen/wikipedia-data-science/blob/master/Downloading%20and%20Parsing%20Wikipedia%20Articles.ipynb).

# In[ ]:


from IPython.core.interactiveshell import InteractiveShell

# Set shell to show all lines of output
InteractiveShell.ast_node_interactivity = 'all'


# In[ ]:


from keras.utils import get_file

x = get_file('found_books_filtered.ndjson', 'https://raw.githubusercontent.com/WillKoehrsen/wikipedia-data-science/master/data/found_books_filtered.ndjson')

import json

books = []

with open(x, 'r') as fin:
    # Append each line to the books
    books = [json.loads(l) for l in fin]

# Remove non-book articles
books_with_wikipedia = [book for book in books if 'Wikipedia:' in book[0]]
books = [book for book in books if 'Wikipedia:' not in book[0]]
print(f'Found {len(books)} books.')


# ## Data Cleaning
# 
# There are a few articles that were caught which are clearly not books (feel free to check out these articles yourself).

# In[ ]:


[book[0] for book in books_with_wikipedia][:5]


# Each legitimate book contains the __title, the information from the `Infobox book` template, the internal wikipedia links, the external links, the date of last edit, and the number of characters in the article__ (a rough estimate of the length of the article).

# In[ ]:


n = 21
books[n][0], books[n][1], books[n][2][:5], books[n][3][:5], books[n][3][:5], books[n][4], books[n][5]


# We will only use the wikilinks, which are saved as the third element (index 2) for each book.

# ## Map Books to Integers
# 
# First we want to create a mapping of book titles to integers. When we feed books into the embedding neural network, we will have to represent them as numbers, and this mapping will let us keep track of the books. We'll also create the reverse mapping, from integers back to the title.

# In[ ]:


book_index = {book[0]: idx for idx, book in enumerate(books)}
index_book = {idx: book for book, idx in book_index.items()}

book_index['Anna Karenina']
index_book[22494]


# ### Exploring Wikilinks
# 
# Although it's not our main focus, we can do a little exploration. Let's find the number of unique Wikilinks  and the most common ones. To create a single list from a list of lists, we can use the `itertools` chain method.

# In[ ]:


from itertools import chain

wikilinks = list(chain(*[book[2] for book in books]))
print(f"There are {len(set(wikilinks))} unique wikilinks.")


# How many of these are links to other books? 

# In[ ]:


wikilinks_other_books = [link for link in wikilinks if link in book_index.keys()]
print(f"There are {len(set(wikilinks_other_books))} unique wikilinks to other books.")


# ### Most Linked-to Articles
# 
# Let's take a look at which pages are most linked to by books on Wikipedia. 
# 
# We'll make a utility function that takes in a list and returns a sorted ordered dictionary of the counts of the items in the list. The `collections` module has a number of useful functions for dealing with groups of objects.

# In[ ]:


from collections import Counter, OrderedDict

def count_items(l):
    """Return ordered dictionary of counts of objects in `l`"""
    
    # Create a counter object
    counts = Counter(l)
    
    # Sort by highest count first and place in ordered dictionary
    counts = sorted(counts.items(), key = lambda x: x[1], reverse = True)
    counts = OrderedDict(counts)
    
    return counts


# We only want to count wikilinks from each book once, so we first find the set of links for each book, then we flatten the list of lists to a single list, and finally pass it to the `count_items` function.

# In[ ]:


# Find set of wikilinks for each book and convert to a flattened list
unique_wikilinks = list(chain(*[list(set(book[2])) for book in books]))

wikilink_counts = count_items(unique_wikilinks)
list(wikilink_counts.items())[:10]


# The most linked to pages are in fact not that surprising! One thing we should notice is that there are discrepancies in capitalization. We want to normalize across capitalization, so we'll lowercase all of the links and redo the counts.

# In[ ]:


wikilinks = [link.lower() for link in unique_wikilinks]
print(f"There are {len(set(wikilinks))} unique wikilinks.")

wikilink_counts = count_items(wikilinks)
list(wikilink_counts.items())[:10]


# That actually changes the rankings! This illustrates an important point: __make sure to take a look at your data before modeling!__ 
# 
# #### Remove Most Popular Wikilinks
# 
# I'm going to remove the __most popular__ wikilinks because these are not very informative. Knowing whether a book is hardcover or paperback is not that important to the content. We also don't need the two `Wikipedia:` links since these do not distinguish the books based on content. __I'd recommend playing around with the wikilinks that are removed because it might have a large effect on the recommendations.__
# 
# (This step is similar to the idea of [TF-IDF (Term Frequency Inverse Document Frequency](http://www.tfidf.com/). When dealing with words in documents, the words that appear most often across documents are usually not that helpful because they don't distinguish documents. TF-IDF is a way to weight a word higher for appearing more often within an article but decrease the weighting for a word appearing more often between articles.)

# In[ ]:


to_remove = ['hardcover', 'paperback', 'hardback', 'e-book', 'wikipedia:wikiproject books', 'wikipedia:wikiproject novels']
for t in to_remove:
    wikilinks.remove(t)
    _ = wikilink_counts.pop(t)


# Since there are so many unique wikilinks, I'm going to limit the list to wikilinks mentioned 4 or more times. Hopefully this reduces the noise that might come from wikilinks that only appear a few times. Keeping every single link will increas the training time significantly, but experiment with this parameter if you are interested.

# In[ ]:


# Limit to greater than 3 links
links = [t[0] for t in wikilink_counts.items() if t[1] >= 4]
print(len(links))


# #### Most Linked-to Books
# 
# As a final bit of exploration, let's look at the books that are mentioned the most by other books on Wikipedia. We'll take the set of links for each book so that we don't have multiple counts for books that are linked to by another book more than once. 

# In[ ]:


# Find set of book wikilinks for each book
unique_wikilinks_books = list(chain(*[list(set(link for link in book[2] if link in book_index.keys())) for book in books]))

# Count the number of books linked to by other books
wikilink_book_counts = count_items(unique_wikilinks_books)
list(wikilink_book_counts.items())[:10]


# It's not surprising that several of these are references. We also see that a few classics make it into the list! 
# 
# ### Potential Additional Cleaning Step
# 
# If you want to try more data cleaning, one option would be to clean the link entities. For example, both `the new york times` and `new york times` are in the links. These could clearly be combined into a single entry because they link to the same exact page. This might require manual inspection of the links, and I decided not to do this because of the time involved! The final embeddings turned out well even without this step, but it might make sense to do in the future. 
# 
# I'm not sure why the same link is represented as two different names (I extracted the title of the link to try and alleviate this issue), but it occurs many times, even for the same book! 

# In[ ]:


for book in books:
    if 'The New York Times' in book[2] and 'New York Times' in book[2]:
        print(book[0], book[2])
        break


# In[ ]:


wikilink_counts.get('the new york times')
wikilink_counts.get('new york times')


# ## Wikilinks to Index
# 
# As with the books, we need to map the Wikilinks to integers. We'll also create the reverse mapping.

# In[ ]:


link_index = {link: idx for idx, link in enumerate(links)}
index_link = {idx: link for link, idx in link_index.items()}

link_index['the economist']
index_link[300]
print(f'There are {len(link_index)} wikilinks that will be used.')


# # Supervised Machine Learning Task
# 
# Now that we have clean data, we'll move on to the second step: developing a supervised machine learning task to train an embedding neural network. As a reminder, we'll state the problem as: given a book title and a link, identify if the link is in the book's article.
# 
# ## Build a Training Set
# 
# In order for any machine learning model to learn, it needs a training set. We are going to treat this as a supervised learning problem: given a pair (book, link), we want the neural network to learn to predict whether this is a legitimate pair - present in the data - or not.
# 
# To create a training set, for each book, we'll iterate through the wikilinks on the book page and record the book title and each link as a tuple. The final `pairs` list will consist of tuples of every (book, link) pairing on all of Wikipedia.

# In[ ]:


pairs = []

# Iterate through each book
for book in books:
    # Iterate through the links in the book
    pairs.extend((book_index[book[0]], link_index[link.lower()]) for link in book[2] if link.lower() in links)
    
len(pairs), len(links), len(books)
pairs[5000]


# We now have over 770,000 positive examples on which to train! Each pair represents one Wikilink for one book. Let's look at a few examples.

# In[ ]:


index_book[pairs[5000][0]], index_link[pairs[5000][1]]


# In[ ]:


index_book[pairs[900][0]], index_link[pairs[900][1]]


# Later on we'll create the negative examples by randomly sampling from the links and the books and making sure the resulting pair is not in `pairs`. 

# In[ ]:


pairs_set = set(pairs)


# Just for fun, let's look at the (book, link) pairs that are represented most often in the data. 

# In[ ]:


x = Counter(pairs)
sorted(x.items(), key = lambda x: x[1], reverse = True)[:5]


# In[ ]:


index_book[13337], index_link[31111]
index_book[31899], index_link[65]
index_book[25899], index_link[30465]


# There's nothing wrong with books that link to the same page many times. They are just more likely to be trained on since there are more of them.

# ## Note about Training / Testing Set
# 
# To compute the embeddings, we are not going to create a separate validation or testing set. While this is a __must__ for a normal supervised machine learning task, in this case, our primary objective is not to make the most accurate model, but to generate the best embeddings. The prediction task is just the method through which we train our network to make the embeddings. At the end of training, we are not going to be testing our model on new data, so we don't need to evaluate the performance. Instead of testing on new data, we'll look at the embeddings themselves to see if books that we think are similar have embeddings that are close to each other. 
# 
# If we kept a separate validation / testing set, then we would be limiting the amount of data that our network can use to train. This would result in less accurate embeddings. Normally with any supervised model, we need to be concerned about overfitting, but again, because we do not need our model to generalize to new data and our goal is the embeddings, we will make our model as effective as possible by using all the data for training. In general, always have a separate validation and testing set (or use cross validation) and make sure to regularize your model to prevent overfitting. 

# ## Generator For Training Samples
# 
# We need to generate positive samples and negative samples to train the neural network. The positive samples are simple: pick a pair from `pairs` and assign it a 1. The negative samples are also fairly easy: pick one random link and one random book, make sure they are not in `pairs`, and assign them a -1 or a 0. (We'll use either a -1 or 0 for the negative labels depending on whether we want to make this a regression or a classification problem. Either approach is valid, and we'll try out both methods.)
# 
# The code below creates a generator that yields batches of samples each time it is called. Neural networks are trained incrementally - a batch at a time - which means that a generator is a useful function for returning examples on which to train. Using a generator alleviates the need to store all of the training data in memory which might be an issue if we were working with a larger dataset such as images. 

# In[ ]:


import numpy as np
import random
random.seed(100)

def generate_batch(pairs, n_positive = 50, negative_ratio = 1.0, classification = False):
    """Generate batches of samples for training"""
    batch_size = n_positive * (1 + negative_ratio)
    batch = np.zeros((batch_size, 3))
    
    # Adjust label based on task
    if classification:
        neg_label = 0
    else:
        neg_label = -1
    
    # This creates a generator
    while True:
        # randomly choose positive examples
        for idx, (book_id, link_id) in enumerate(random.sample(pairs, n_positive)):
            batch[idx, :] = (book_id, link_id, 1)

        # Increment idx by 1
        idx += 1
        
        # Add negative examples until reach batch size
        while idx < batch_size:
            
            # random selection
            random_book = random.randrange(len(books))
            random_link = random.randrange(len(links))
            
            # Check to make sure this is not a positive example
            if (random_book, random_link) not in pairs_set:
                
                # Add to batch and increment index
                batch[idx, :] = (random_book, random_link, neg_label)
                idx += 1
                
        # Make sure to shuffle order
        np.random.shuffle(batch)
        yield {'book': batch[:, 0], 'link': batch[:, 1]}, batch[:, 2]


# To get a new batch, call `next` on the generator. 

# In[ ]:


next(generate_batch(pairs, n_positive = 2, negative_ratio = 2))


# In[ ]:


x, y = next(generate_batch(pairs, n_positive = 2, negative_ratio = 2))

# Show a few example training pairs
for label, b_idx, l_idx in zip(y, x['book'], x['link']):
    print(f'Book: {index_book[b_idx]:30} Link: {index_link[l_idx]:40} Label: {label}') 


# The neural network will take in the book index and the link index and try to embed them in such a way that it can predict the label from the embeddings.

# # Neural Network Embedding Model
# 
# With our dataset and a supervised machine learning task, we're almost there. The next step is the most technically complicated but thankfully fairly simple with Keras. We are going to construct the neural network that learns the entity embeddings. The input to this network is the (book, link) (either positive or negative) as integers, and the output will be a prediction of whether or not the link was present in the book article. However, we're not actually interested in the prediction except as the device used to train the network by comparison to the label. What we are after is at the heart of the network: the embedding layers, one for the book and one for the link each of which maps the input entity to a 50 dimensional vector. The layers of our network are as follows:
# 
# 1. Input: parallel inputs for the book and link
# 2. Embedding: parallel embeddings for the book and link
# 3. Dot: computes the dot product between the embeddings to merge them together
# 4. Reshape: utility layer needed to correct the shape of the dot product
# 5. [Optional] Dense: fully connected layer with sigmoid activation to generate output for classification
# 
# After converting the inputs to an embedding, we need a way to combine the embeddings into a single number. For this we can use the dot product which does element-wise multiplication of numbers in the vectors and then sums the result to a single number. This raw number (after reshaping) is then the ouput of the model for the case of regression. In regression, our labels are either -1 or 1, and so the model loss function will be mean squared error in order to minimize the distance between the prediction and the output. Using the dot product with normalization means that the `Dot` layer is finding the cosine similarity between the embedding for the book and the link. Using this method for combining the embeddings means we are trying to make the network learn similar embeddings for books that link to similar pages. 
# 
# ### Classification vs Regression
# 
# For classification, we add an extra fully connected `Dense` layer with a `sigmoid` activation to squash the outputs between 0 and 1 because the labels are either 0 or 1. The loss function for classification is `binary_crossentropy` which measures the [error of the neural network predictions in a binary classification problem](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html), and is a measure of the similarity between two distributions. We can train with either classification or regression, and in practice, I found that both approaches produced similar embeddings. I'm not sure about the technical merits of these methods, and I'd be interested to hear if one is better than the other. 
# 
# The optimizer - the algorithm used to update the parameters (also called weights) of the neural network after calculating the gradients through backpropagation - is Adam in both cases ([Adam is a modification to Stochastic Gradient Descent](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)). We use the default parameters for this optimizer. The nice thing about modern neural network frameworks is we don't have to worry about backpropagation or updating the model parameters because that is done for us. It's nice to have an idea of what is occuring behind the scenes, but it's not entirely necessary to use a neural network effectively. 

# In[ ]:


from keras.layers import Input, Embedding, Dot, Reshape, Dense
from keras.models import Model


# In[ ]:


def book_embedding_model(embedding_size = 50, classification = False):
    """Model to embed books and wikilinks using the functional API.
       Trained to discern if a link is present in a article"""
    
    # Both inputs are 1-dimensional
    book = Input(name = 'book', shape = [1])
    link = Input(name = 'link', shape = [1])
    
    # Embedding the book (shape will be (None, 1, 50))
    book_embedding = Embedding(name = 'book_embedding',
                               input_dim = len(book_index),
                               output_dim = embedding_size)(book)
    
    # Embedding the link (shape will be (None, 1, 50))
    link_embedding = Embedding(name = 'link_embedding',
                               input_dim = len(link_index),
                               output_dim = embedding_size)(link)
    
    # Merge the layers with a dot product along the second axis (shape will be (None, 1, 1))
    merged = Dot(name = 'dot_product', normalize = True, axes = 2)([book_embedding, link_embedding])
    
    # Reshape to be a single number (shape will be (None, 1))
    merged = Reshape(target_shape = [1])(merged)
    
    # If classifcation, add extra layer and loss function is binary cross entropy
    if classification:
        merged = Dense(1, activation = 'sigmoid')(merged)
        model = Model(inputs = [book, link], outputs = merged)
        model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    # Otherwise loss function is mean squared error
    else:
        model = Model(inputs = [book, link], outputs = merged)
        model.compile(optimizer = 'Adam', loss = 'mse')
    
    return model

# Instantiate model and show parameters
model = book_embedding_model()
model.summary()


# There are nearly 4.0 million weights (parameters) that need to be learned by the neural network. Each of these represents one number in an embedding for one entity. During training, the neural network adjusts these parameters in order to minimize the loss function on the training data. 

# # Train Model
# 
# We have the training data - in a generator - and a model. The next step is to train the model to learn the entity embeddings. During this process, the model will update the embeddings (change the model parameters) to accomplish the task of predicting whether a certain link is on a book page or not. The resulting embeddings can then be used as a representation of books and links. 
# 
# There are a few parameters to adjust for training. The batch size should generally be as large as possible given the memory constraints of your machine. The negative ratio can be adjusted based on results. I tried 2 and it seemed to work well. The number of steps per epoch is chosen such that the model sees a number of examples equal to the number of pairs on each epoch. This is repeated for 15 epochs (which might be more than necessary).

# In[ ]:


n_positive = 1024

gen = generate_batch(pairs, n_positive, negative_ratio = 2)

# Train
h = model.fit_generator(gen, epochs = 15, 
                        steps_per_epoch = len(pairs) // n_positive,
                        verbose = 2)


# The loss decreases as training progresses which should give us confidence the model is learning something! 
# 
# 
# The entire trained model can be saved and later loaded in so you don't have to repeat the training. It's also possible to save certain layers.

# In[ ]:


model.save('first_attempt.h5')


# # Extract Embeddings and Analyze
# 
# The trained model has learned - hopefully - representations of books and wikilinks that place similar entities next to one another in the embedding space. To find out if this is the case, we extract the embeddings and use them to find similar books and links.

# In[ ]:


# Extract embeddings
book_layer = model.get_layer('book_embedding')
book_weights = book_layer.get_weights()[0]
book_weights.shape


# Each book is now represented as a 50-dimensional vector. 
# 
# We need to normalize the embeddings so that the dot product between two embeddings becomes the cosine similarity.

# In[ ]:


book_weights = book_weights / np.linalg.norm(book_weights, axis = 1).reshape((-1, 1))
book_weights[0][:10]
np.sum(np.square(book_weights[0]))


# Normalize just means divide each vector by the square root of the sum of squared components.

# # Finding Similar Books
# 
# We've trained the model and extracted the embeddings - great - but where is the book recommendation system? Now that we have the embeddings, we can use them to recommend books that our model has learned are most similar to a given book.
# 
# 
# ### Function to Find Most Similar Entities
# 
# The function below takes in either a book or a link, a set of embeddings, and returns the `n` most similar items to the query. It does this by computing the dot product between the query and embeddings. Because we normalized the embeddings, the dot product represents the [cosine similarity](http://blog.christianperone.com/2013/09/machine-learning-cosine-similarity-for-vector-space-models-part-iii/) between two vectors. This is a measure of similarity that does not depend on the magnitude of the vector in contrast to the Euclidean distance. (The Euclidean distance would be another valid metric of similary to use to compare the embeddings.)
# 
# Once we have the dot products, we can sort the results to find the closest entities in the embedding space. With cosine similarity, higher numbers indicate entities that are closer together, with -1 the furthest apart and +1 closest together.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 15

def find_similar(name, weights, index_name = 'book', n = 10, least = False, return_dist = False, plot = False):
    """Find n most similar items (or least) to name based on embeddings. Option to also plot the results"""
    
    # Select index and reverse index
    if index_name == 'book':
        index = book_index
        rindex = index_book
    elif index_name == 'page':
        index = link_index
        rindex = index_link
    
    # Check to make sure `name` is in index
    try:
        # Calculate dot product between book and all others
        dists = np.dot(weights, weights[index[name]])
    except KeyError:
        print(f'{name} Not Found.')
        return
    
    # Sort distance indexes from smallest to largest
    sorted_dists = np.argsort(dists)
    
    # Plot results if specified
    if plot:
        
        # Find furthest and closest items
        furthest = sorted_dists[:(n // 2)]
        closest = sorted_dists[-n-1: len(dists) - 1]
        items = [rindex[c] for c in furthest]
        items.extend(rindex[c] for c in closest)
        
        # Find furthest and closets distances
        distances = [dists[c] for c in furthest]
        distances.extend(dists[c] for c in closest)
        
        colors = ['r' for _ in range(n //2)]
        colors.extend('g' for _ in range(n))
        
        data = pd.DataFrame({'distance': distances}, index = items)
        
        # Horizontal bar chart
        data['distance'].plot.barh(color = colors, figsize = (10, 8),
                                   edgecolor = 'k', linewidth = 2)
        plt.xlabel('Cosine Similarity');
        plt.axvline(x = 0, color = 'k');
        
        # Formatting for italicized title
        name_str = f'{index_name.capitalize()}s Most and Least Similar to'
        for word in name.split():
            # Title uses latex for italize
            name_str += ' $\it{' + word + '}$'
        plt.title(name_str, x = 0.2, size = 28, y = 1.05)
        
        return None
    
    # If specified, find the least similar
    if least:
        # Take the first n from sorted distances
        closest = sorted_dists[:n]
         
        print(f'{index_name.capitalize()}s furthest from {name}.\n')
        
    # Otherwise find the most similar
    else:
        # Take the last n sorted distances
        closest = sorted_dists[-n:]
        
        # Need distances later on
        if return_dist:
            return dists, closest
        
        
        print(f'{index_name.capitalize()}s closest to {name}.\n')
        
    # Need distances later on
    if return_dist:
        return dists, closest
    
    
    # Print formatting
    max_width = max([len(rindex[c]) for c in closest])
    
    # Print the most similar and distances
    for c in reversed(closest):
        print(f'{index_name.capitalize()}: {rindex[c]:{max_width + 2}} Similarity: {dists[c]:.{2}}')
        
    


# (We know that this function works if the most similar book is the book itself. Because we multiply the item vector times all the other embeddings, the most similar should be the item itself with a similarity of 1.0.)

# In[ ]:


find_similar('War and Peace', book_weights)


# It works! The most similar books make sense at least for _War and Peace._
# 
# 

# In[ ]:


find_similar('War and Peace', book_weights, least = True, n = 5)


# The least similar books are quite a grab bag!

# In[ ]:


find_similar('War and Peace', book_weights, n = 5, plot = True)


# In[ ]:


find_similar('The Fellowship of the Ring', book_weights, n = 5)


# It doesn't work perfectly for every book as we can see.

# In[ ]:


find_similar('Artificial Intelligence: A Modern Approach', book_weights, n = 5)


# In[ ]:


find_similar('Weapons of Math Destruction', book_weights, n = 5)


# Not bad by the looks of the results! I encourage you to play around with the model and explore the resulting embeddings. __We have successfully built a book recommendation system using neural network embeddings.__
# 
# What should I read next? Currently I'm working through a fantastic collection of essays by Stephen Jay Gould.

# In[ ]:


find_similar('Bully for Brontosaurus', book_weights, n = 5)


# The algorithm has spoken! 

# In[ ]:


find_similar('Bully for Brontosaurus', book_weights, n = 5, plot = True)


# ## Wikilink Embeddings
# 
# We also have the embeddings of wikipedia links (which are themselves Wikipedia pages). We can take a similar approach to extract these and find the most similar to a query page. 
# 
# Let's write a quick function to extract weights from a model given the name of the layer.

# In[ ]:


def extract_weights(name, model):
    """Extract weights from a neural network model"""
    
    # Extract weights
    weight_layer = model.get_layer(name)
    weights = weight_layer.get_weights()[0]
    
    # Normalize
    weights = weights / np.linalg.norm(weights, axis = 1).reshape((-1, 1))
    return weights


# In[ ]:


link_weights = extract_weights('link_embedding', model)


# We can use the same `find_similar` function to find the most similar links to a given link.

# In[ ]:


find_similar('science fiction', link_weights, index_name = 'page')


# Based on my knowledge, it appears the neural network embeddings are also a good representation of the pages! Not only can we find the most similar books to a given book, but if we have a category of books that we enjoy, we can find another category that is similar.

# In[ ]:


find_similar('biography', link_weights, index_name = 'page')


# In[ ]:


find_similar('biography', link_weights, index_name = 'page', n = 5, plot = True)


# Now you not only know what books to look for, you know the books to avoid for a given category! 

# In[ ]:


find_similar('new york city', link_weights, index_name = 'page', n = 5)


# ## Classification Model
# 
# I was curious if training for the mean squared error as a regression problem was the ideal approach, so I also decided to experiment with a classification model. For this model, the negative examples receive a label of 0 and the loss function is binary cross entropy. The procedure for the neural network to learn the embeddings is exactly the same, only it will be optimizing for a slightly different measure.

# In[ ]:


model_class = book_embedding_model(50, classification = True)
gen = generate_batch(pairs, n_positive, negative_ratio=2, classification = True)


# In[ ]:


# Train the model to learn embeddings
h = model_class.fit_generator(gen, epochs = 15, steps_per_epoch= len(pairs) // n_positive,
                            verbose = 0)


# We'll go through the same process, extracting the weights and finding similar books based on the embedding space representation.

# In[ ]:


model_class.save('first_attempt_class.h5')


# In[ ]:


book_weights_class = extract_weights('book_embedding', model_class)
book_weights_class.shape


# In[ ]:


find_similar('War and Peace', book_weights_class, n = 5)


# In[ ]:


find_similar('The Fellowship of the Ring', book_weights_class, n = 5)


# In[ ]:


find_similar('The Better Angels of Our Nature', book_weights_class, n = 5)


# Things are looking pretty good with this model as well. It's hard to tell if this model is "better" than the regression model in the sense that the recommendations are improved. I would say that both are useful in surfacing similar books and pages.
# 
# Let's take a look at the link recommendations for this model.

# In[ ]:


link_weights_class = extract_weights('link_embedding', model_class)


# In[ ]:


find_similar('the washington post', link_weights_class, index_name = 'page', n = 5)


# In[ ]:


find_similar('category:almanacs', link_weights_class, index_name = 'page', n = 5)


# In[ ]:


find_similar('steven pinker', link_weights_class, index_name = 'page', n = 5)


# In[ ]:


find_similar('richard dawkins', link_weights_class, index_name = 'page', n = 5)


# Looks like I have some new topics to explore!

# # Visualizations
# 
# One of the most interesting parts about embeddings is that we can use them to visualize concepts such as _War and Peace_ or _biography_. First we have to take the embeddings from 50 dimensions down to either 3 or 2. We can do this using `pca`, `tsne`, or `umap`. We'll try both tsne and umap for comparison. TSNE takes much longer and is designed to retain local structure within the data. UMAP is generally quicker and is designed for a balance between local and global structure in the embedding.
# 
# ## Manifold Embeddings
# 
# [TSNE: t-Stochastic Distributed Neighbors Embedding](https://lvdmaaten.github.io/tsne/), and [UMAP: Uniform Manifold Approximation and Projection](https://github.com/lmcinnes/umap), are both methods that use the idea of a manifold to map vecotrs to a lower dimensional embedded space. Therefore, we are taking the 37,000 dimensions in the case of books, embedding them to 50 dimensions with the neural network, and then embedding them down to 2 dimensions with a manifold. The primary idea behind dimension reduction with a manifold is that there is a lower dimensional representation of the vectors that can still capture the variation between different groups. We want the embeddings to represent similar entities close to one another but in fewer dimensions that allow us to visualize the entities. 

# In[ ]:


from sklearn.manifold import TSNE
from umap import UMAP


# In[ ]:


def reduce_dim(weights, components = 3, method = 'tsne'):
    """Reduce dimensions of embeddings"""
    if method == 'tsne':
        return TSNE(components, metric = 'cosine').fit_transform(weights)
    elif method == 'umap':
        # Might want to try different parameters for UMAP
        return UMAP(n_components=components, metric = 'cosine', 
                    init = 'random', n_neighbors = 5).fit_transform(weights)


# In[ ]:


book_r = reduce_dim(book_weights_class, components = 2, method = 'tsne')
book_r.shape


# We've now taken the initial 37,000 dimension book vector and reduced it to just 2 dimensions.

# In[ ]:


InteractiveShell.ast_node_interactivity = 'last'

plt.figure(figsize = (10, 8))
plt.plot(book_r[:, 0], book_r[:, 1], 'r.')
plt.xlabel('TSNE 1'); plt.ylabel('TSNE 2'); plt.title('Book Embeddings Visualized with TSNE');


# There do appear to be a few noticeable clumps. However, it's difficult to derive any meaning from this plot since we aren't distinguishing books in any way.

# In[ ]:


book_ru = reduce_dim(book_weights_class, components = 2, method = 'umap')

plt.figure(figsize = (10, 8))
plt.plot(book_ru[:, 0], book_ru[:, 1], 'g.');
plt.xlabel('UMAP 1'); plt.ylabel('UMAP 2'); plt.title('Book Embeddings Visualized with UMAP');


# ### Book Embeddings by Genre
# 
# The above graphs are difficult to interpret. Let's plot the embeddings by the `genre` which is contained in the `Infobox` template data for each book. We'll limit it to the 10 most popular genres.

# In[ ]:


info = list(chain(*[set(book[1]) for book in books]))
info_counts = count_items(info)
list(info_counts.items())[:10]


# In[ ]:


genres = [book[1].get('genre', 'None').lower() for book in books]

# Remove genres not found
genre_counts = count_items(genres)
del genre_counts['none']
list(genre_counts.items())[:10]


# In[ ]:


# Include 10 most popular genres
genre_to_include = list(genre_counts.keys())[:10]


# In[ ]:


idx_include = []
genres = []

for i, book in enumerate(books):
    if 'genre' in book[1].keys():
        if book[1]['genre'].lower() in genre_to_include:
            idx_include.append(i)
            genres.append(book[1]['genre'].capitalize())
            
len(idx_include)


# Now we need to map the genres to integers to plot them as colors. 

# In[ ]:


ints, gen = pd.factorize(genres)
gen[:5]


# Finally, we can plot the embedding colored by the genre

# In[ ]:


plt.figure(figsize = (10, 8))

# Plot embedding
plt.scatter(book_r[idx_include, 0], book_r[idx_include, 1], 
            c = ints, cmap = plt.cm.tab10)

# Add colorbar and appropriate labels
cbar = plt.colorbar()
cbar.set_ticks([])
for j, lab in enumerate(gen):
    cbar.ax.text(1, (2 * j + 1) / ((10) * 2), lab, ha='left', va='center')
cbar.ax.set_title('Genre', loc = 'left')


plt.xlabel('TSNE 1'); plt.ylabel('TSNE 2'); plt.title('TSNE Visualization of Book Embeddings');


# The books do seem to be slightly separated based on the genre. The categories aren't exactly that helpful but we did what we could! We can also try UMAP. 

# In[ ]:


plt.figure(figsize = (10, 8))

# Plot embedding
plt.scatter(book_ru[idx_include, 0], book_ru[idx_include, 1], 
            c = ints, cmap = plt.cm.tab10)

# Add colorbar and appropriate labels
cbar = plt.colorbar()
cbar.set_ticks([])
for j, lab in enumerate(gen):
    cbar.ax.text(1, (2 * j + 1) / ((10) * 2), lab, ha='left', va='center')
cbar.ax.set_title('Genre', loc = 'left')


plt.xlabel('UMAP 1'); plt.ylabel('UMAP 2'); plt.title('UMAP Visualization of Book Embeddings');


# There doesn't appear to be much separation between the categories in the UMAP clustering. There are a lot of parameters to play around with in UMAP, and changing some of them might result in better clusters.

# ## Most Popular Books in Embedding
# 
# Let's see the embedding labeled with the 10 books most often mentioned by other books. 

# In[ ]:


plt.figure(figsize = (14, 12))

# Plot all books
plt.scatter(book_r[:, 0], book_r[:, 1], marker = '.', color = 'lightblue', alpha = 0.2)

# Plot genres
plt.scatter(book_r[idx_include, 0], book_r[idx_include, 1], 
            c = ints, cmap = plt.cm.tab10, alpha = 0.6)

# Add colorbar and appropriate labels
cbar = plt.colorbar()
cbar.set_ticks([])
for j, lab in enumerate(gen):
    cbar.ax.text(1, (2 * j + 1) / ((10) * 2), lab, ha='left', va='center')
    
cbar.ax.set_title('Genre', loc = 'left')
plt.xlabel('TSNE 1'); plt.ylabel('TSNE 2'); plt.title('TSNE Visualization of Book Embeddings');

for book in list(wikilink_book_counts.keys())[:10]:
    
    x, y = book_r[book_index[book], 0], book_r[book_index[book], 1];
    # Italize book title using latex
    s =  ''.join([' $\it{' + word + '}$' for word in book.split()])
    _ = plt.scatter(x, y, s = 250, color = 'r',
                    marker = '*', edgecolor = 'k')
    _ = plt.text(x - 10, y + 2, s, fontsize = 14);


# The `Encyclopedia`s of Science Fiction and Fantasy have nearly perfect overlap. 

# In[ ]:


book_r[book_index['The Encyclopedia of Science Fiction']]


# In[ ]:


book_r[book_index['The Encyclopedia of Fantasy']]


# Let's make a function that can color the plot by any attribute in the book infobox. We already saw this with the genre, but we can extend it to the author or any other information.

# In[ ]:


def plot_by_attribute(attribute):
    """Color book embedding by `attribute`"""
    # Find all the attribute values
    attrs = [book[1].get(attribute, 0) for book in books]

    # Remove attributes not found
    attr_counts = count_items(attrs)
    del attr_counts[0]
    
    # Include 10 most popular attributes
    attr_to_include, counts = list(attr_counts.keys())[:10], list(attr_counts.values())[:10]
    
    idx_include = []
    attributes = []

    # Iterate through books searching for the attribute
    for i, book in enumerate(books):
        # Limit to books with the attribute
        if attribute in book[1].keys():
            # Limit to attribute in the 10 most popular
            if book[1][attribute] in attr_to_include:
                idx_include.append(i)
                attributes.append(book[1][attribute])
                
    # Map to integers
    ints, attrs = pd.factorize(attributes)
    plt.figure(figsize = (12, 10))

    plt.scatter(book_r[:, 0], book_r[:, 1], marker = '.', color = 'lightblue', alpha = 0.2)
    
    # Plot embedding with only specific attribute highlighted
    plt.scatter(book_r[idx_include, 0], book_r[idx_include, 1], alpha = 0.6,
                c = ints, cmap = plt.cm.tab10, marker = 'o', s = 50)

    # Add colorbar and appropriate labels
    cbar = plt.colorbar()
    cbar.set_ticks([])
    tick_labels = [f'{attr}: {count}' for attr, count in zip(attr_to_include, counts)]
    # Labeling
    for j, lab in enumerate(tick_labels):
        cbar.ax.text(1, (2 * j + 1) / ((10) * 2), lab, ha='left', va='center')
    cbar.ax.set_title(f'{attribute.capitalize()}: Count', loc = 'left')


    plt.xlabel('TSNE 1'); plt.ylabel('TSNE 2'); plt.title(f'Book Embeddings with {attribute.capitalize()}');


# In[ ]:


plot_by_attribute('genre')


# In[ ]:


plot_by_attribute('author')


# In[ ]:


plot_by_attribute('country')


# In[ ]:


plot_by_attribute('language')


# The best clumpings are probably by genre although the countries also appear to be separated. For the language and country, it might be more accurate to do this will book articles on Wikipedias other than just the English version.

# Another option we have to try and make sense of this data is plot books that originally were closest together in the original embedding space. Reducing the dimensions to 2 may distort the distances between similar books in the 50 dimensional space.

# ## Plot Book Nearest Neighbors
# 
# To get a better sense of which books are located where, we can plot a book along with its nearest neighbors. These will be the nearest neighbors in the original embedding space, so they are not necessarily the closest in the reduced dimension representation.

# In[ ]:


def plot_closest(item, weights, index_name, n, plot_data):
    """Plot n most closest items to item"""
    
    # Find the closest items
    dist, closest = find_similar(item, weights, index_name, n, return_dist=True)
    
    # Choose mapping for look up
    if index_name == 'book':
        index = book_index
        rindex = index_book
    elif index_name == 'page':
        index = link_index
        rindex = index_link 
    
    plt.figure(figsize = (10, 9))
    plt.rcParams['font.size'] = 14
    
    # Limit distances
    dist = dist[closest]
    
    # Plot all of the data
    plt.scatter(plot_data[:, 0], plot_data[:, 1], alpha = 0.1, color = 'goldenrod')
    
    # Plot the item
    plt.scatter(plot_data[closest[-1], 0], plot_data[closest[-1], 1], s = 600, edgecolor = 'k', color = 'forestgreen')
    
    # Plot the closest items
    p = plt.scatter(plot_data[closest[:-1], 0], plot_data[closest[:-1], 1], 
                c = dist[:-1], cmap = plt.cm.RdBu_r, s = 200, alpha = 1, marker = '*')
    
    # Colorbar management
    cbar = plt.colorbar()
    cbar.set_ticks([])
    
    tick_labels = []
    # Tick labeling for colorbar
    for idx, distance in zip(closest[:-1], dist[:-1]):
        name_str = ''
        for word in rindex[idx].split():
            # Title uses latex for italize
            name_str += ' $\it{' + word + '}$'
        name_str += ': ' + str(round(distance, 2))
        tick_labels.append(name_str)
    
    for j, lab in enumerate(tick_labels):
        cbar.ax.text(1, (2 * j + 1) / ((n - 1) * 2), lab, ha='left', va='center', size = 12)
    cbar.ax.set_title(f'{index_name.capitalize()} and Cosine Distance', loc = 'left', size = 14)
    
    # Formatting for italicized title
    name_str = f'{index_name.capitalize()}s Most Similar to'
    for word in item.split():
        # Title uses latex for italize
        name_str += ' $\it{' + word + '}$'
    
    # Labeling
    plt.xlabel('TSNE 1'); plt.ylabel('TSNE 2'); 
    plt.title(name_str);


# In[ ]:


plot_closest('War and Peace', book_weights_class, 'book', 10, book_r)


# We can see that even though these are the closest books in the 50-dimensional embedding space, when we reduce it down to 2 dimensions, the same separations are not preserved. 

# In[ ]:


plot_closest('A Brief History of Time', book_weights_class, 'book', 10, book_r)


# This time, we get a much better grouping! I guess we have found the corner of Wikipedia books about the entirety of the universe.

# ## Pages Visualization
# 
# Let's look at reducing the dimension of the embedding for the wikilinks. We'll then visualize them in the same way as with the books.

# In[ ]:


link_r = reduce_dim(link_weights_class, components = 2, method = 'tsne')
link_r.shape


# We'll get the 10 most popular categories so we can plot them on the embedding.

# In[ ]:


idx_to_include = [idx for link, idx in link_index.items() if 'category:' in link]

categories = []

for book in books:
    for link in book[2]:
        if 'category:' in link.lower():
            categories.append(link)
            
c_counts = count_items(categories)
list(c_counts.items())[:5]


# In[ ]:


idx = []

# Find the index of the most popular links
for link in list(c_counts.keys())[:10]:
    link_idx = link_index[link.lower()]
    
    # Find index of category 
    index = int(np.where(np.array(idx_to_include) == link_idx)[0])
    idx.append(index)


# The next image shows all the link embedded with the 10 most popular categories labeled.

# In[ ]:


plt.figure(figsize = (12, 12))
plt.scatter(link_r[:, 0], link_r[:, 1], alpha = 0.6)

for i in idx:
    x, y = link_r[i, 0], link_r[i, 1]
    s = index_link[idx_to_include[i]].split(':')[-1]
    _ = plt.text(x, y, s, fontsize = 18);
    
plt.xlabel('TSNE 1'); plt.ylabel('TSNE 2'); plt.title('Wikilinks Category Embedding Visualized with TSNE');


# We do see some clumpings, but it's difficult to label them. If this was interactive, then we could get a lot more use from it. (This will be an upcoming topic for an article).

# We can also plot the closest pages to a given page. 

# In[ ]:


plot_closest('new york times', link_weights_class, 'page', 10, link_r)


# In[ ]:


plot_closest('james joyce', link_weights_class, 'page', 10, link_r)


# In[ ]:


plot_closest('margaret atwood', link_weights_class, 'page', 20, link_r)


# In[ ]:


plot_closest('leo tolstoy', link_weights_class, 'page', 20, link_r)


# ## Plot Multiple Books and Pages

# As a final visualization effort, we can try to identify clumpings of books or articles by plotting multiple books and those most similar. We'll show all of the embedding books in the background and then highlight the most similar books around a list of query books. 

# In[ ]:


def plot_groups(items, weights, embed, index_name = 'book', n = 5):
    closest = []
    dists = []
    
    # Select index and reverse index
    if index_name == 'book':
        index = book_index
        rindex = index_book
    elif index_name == 'page':
        index = link_index
        rindex = index_link
    
    for item in items:
        d, c = find_similar(item, weights, index_name, n, return_dist = True)
        d = d[c]
        closest.extend(c)
        dists.extend(d)
        
    ax = plt.figure(figsize = (10, 8))
    p = plt.scatter(embed[:, 0], embed[:, 1], alpha = 0.2, marker = 'o', color =  'lightblue')

    cmap = plt.cm.get_cmap('tab10_r', len(items))
    color_list = []
    
    for i, item in enumerate(items):
         # Plot the item
        plt.scatter(embed[index[item], 0], embed[index[item], 1], 
                    s = 200, alpha = 0.4, edgecolor = 'k', color = cmap(i))
        color_list.extend(i for _ in range(n))
        
    p = plt.scatter(embed[closest, 0], embed[closest, 1], c = color_list, cmap = cmap, s = 150,
                    marker = '*', alpha = 0.8)

    
    cbar = plt.colorbar(p)
    cbar.set_ticks([])
    
    tick_labels = []
    # Tick labeling for colorbar
    for item in items:
        name_str = ''
        for word in item.split():
            # Title uses latex for italize
            name_str += ' $\it{' + word + '}$'
        tick_labels.append(name_str)
        
    for j, lab in enumerate(tick_labels):
        cbar.ax.text(1, (2 * j + 1) / (len(items) * 2), lab, ha='left', va='center', size = 12)
    cbar.ax.set_title(f'Highlighted {index_name.capitalize()}s', loc = 'left', size = 14)
    plt.xlabel('TSNE 1'); plt.ylabel('TSNE 2'); plt.title(f'Embedded {index_name.capitalize()}s with Closest Neighbors');


# In[ ]:


plot_groups(['War and Peace', "The Once and Future King", 
             "Weapons of Math Destruction", "The Fellowship of the Ring",
             "A Brief History of Time", "Enlightenment Now"],
            book_weights_class, book_r, 'book')


# In[ ]:


plot_groups(['short stories', 'novel',
            'biography', 'historical novel', 
             'science fiction', 'non-fiction'], 
            link_weights_class, link_r, 'page')


# In[ ]:


random.seed(150)
plot_groups(list(random.sample(book_index.keys(), 6)),
            book_weights_class, book_r, 'book')


# In[ ]:


random.seed(150)
plot_groups(list(random.sample(link_index.keys(), 6)), link_weights_class, link_r, 'page')


# ### Save Embeddings for Visualization
# 
# We can save these embeddings for visualization in [projector.tensorflow.org](https://projector.tensorflow.org/). In order to work with this tool, the files must be tab separated. First, we'll save the names and book info to use as metadata.

# In[ ]:


get_ipython().run_cell_magic('capture', '', "with open('link_names.tsv', 'w' , encoding = 'utf-8') as fout:\n    for l in link_index.keys():\n        fout.write(str(l))\n        fout.write('\\n')")


# In[ ]:


import re
pattern = re.compile('[\\n]|<.*?>')
pattern.sub('', 'bill moushey \n < br >').strip()


# In[ ]:


import re
pattern = re.compile('[\\n]|<.*?>')

# Extract book metadata
authors = []
genres = []
pages = []
languages = []
countries = []
publishers = []


for book in books:
    info = book[1]
    for attr, l in zip(['author', 'genre', 'pages', 'langauge', 'country', 'publisher'],
                        [authors, genres, pages, languages, countries, publishers]):
        l.append(pattern.sub('', info.get(attr, 'None').lower()).strip())

book_info = pd.DataFrame({'author': authors, 'genre': genres, 'pages': pages,
                          'language': languages, 'country': countries,
                          'publisher': publishers}, index = list(book_index.keys()))

book_info = book_info.replace({'none': np.nan})
book_info.index.name = 'title'
book_info.head()


# In[ ]:


book_info.to_csv('book_info.tsv', sep = '\t')


# The book embeddings and link embeddings can be saved as tab separated files. We'll save all 50 dimensions of the embeddings because the tool will perform PCA (principal components analysis) or TSNE for us.

# In[ ]:


np.savetxt('book_embedding.tsv', book_weights_class, delimiter='\t')
np.savetxt('link_embedding.tsv', link_weights_class, delimiter='\t')


# In[ ]:


book_embedding = np.loadtxt('book_embedding.tsv', delimiter = '\t')
book_embedding.shape


# # Conclusions
# 
# In this notebook, we built an effective book recommendation system using the principle that books with similar outgoing links are similar and all of the book articles on Wikipedia. We embedded both the wikilinks and the books using a neural networ. To train the neural network, we developed a supervised machine learning problem of classifying if a given link was present on the page for a book.
# 
# More than just training the neural network, we saw how to thoroughly inspect the embeddings in order to find the closest books to a given book in embedding space. We also saw how to visualize the embeddings which sometimes can show us interesting clusterings. In the process, we took the original 37,000 dimensions of the books and reduced it first to 50 using the neural network and then to 2 using a manifold learning method. The neural network embedding was useful for making recommendations based on closest entities while the TSNE embedding is useful primarily for visualization.
# 
# We covered a number of topics that may be useful in a data science position:
# 
# 1. How to clean data for a machine learning project
# 2. How to set up a supervised learning task from a raw dataset
# 3. How to train a neural network to learn entity embeddings
# 4. How to inspect entity embeddings to find closest entities to a query
# 5. How to reduce the dimension of a dataset for visualization
# 6. How to format a plot to try and make sense of the data (we'll work a little more on this)
# 
# The next step that I'll work on is to visualize these embeddings in an interactive tool. Interactivity adds another element of insightto data analysis and I enjoy the exploratory nature of plots that let you move through the data. 
# 
# ## Potential Other Projects
# 
# There are several ways you can take this and build on it:
# 
# 1. Train the embeddings using not the internal links, but the external links
# 2. Train the embeddings using the infobox information. This would require framing a different machine learning problem
# 3. Make a recommendation system for other wikipedia articles such as people or landmarks.
# 4. Adjust the parameters of the models and data cleaning in this notebook to arrive at any other insights.
# 
# The most important part about learning data science is to practice data science! Therefore, I encourage anyone to alter this notebook, build on the code, share it with others, and best of all, teach it to others! The best way to learn is by doing and then teaching, so get started! 
# 
# Best,
# 
# Will

# In[ ]:




