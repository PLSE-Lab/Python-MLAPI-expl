#!/usr/bin/env python
# coding: utf-8

# # Comparing Project Gutenberg's 20 Most Popular Books

# In this analysis, we will use Doc2Vec to determine which of Project Gutenberg's 20 most popular books are most conceptually similar.

# In[ ]:


# Pull in some tools we'll need.
import codecs
import glob
import gensim
get_ipython().run_line_magic('pylab', 'inline')


# In[ ]:


# Create a list of all of our book files.
book_filenames = sorted(glob.glob("../input/*.rtf"))
print("Found books:")
book_filenames


# In[ ]:


# Read each book into the book_corpus, doing some cleanup along the way.
book_corpus = []
for book_filename in book_filenames:
    with codecs.open(book_filename, "r", "utf-8") as book_file:
        book_corpus.append(
            gensim.models.doc2vec.TaggedDocument(
                gensim.utils.simple_preprocess( # Clean the text with simple_preprocess
                    book_file.read()),
                    ["{}".format(book_filename)])) # Tag each book with its filename


# The next bit of code is where we build our model -- the fun part!
# 
# We set up our model with parameters that affect how it learns. In theory
# we can set those numbers to any value we like, although in practice some will produce more useful results than others. This is more of an art than a science, and experimentation is helpful here.
# 
# The first time you run this notebook, use the defaults so you can see how it works. But after that, go ahead and set these values to any number you like. (Afterward, rerun this code block and the ones after it to see the data update.)
# 
# **vector_size** is how many dimensions our idea space has. More dimensions lets you capture more possible concepts, but runs the risk that no books are is similar to any others (the space is too big, so everything is far apart).
# 
# **min_count** lets you ignore infrequent words (any word which is in the corpus less than min_count times). These are hard for the neural net to understand since it doesn't have much data about them. Setting min_count low will mean including rare words and ending up with garbage ideas about them, but setting it high will mean throwing out data you could have used.
# 
# **epochs** is how many times to run over the training data. Too few will be inaccurate but too many will be slow and have diminishing returns.
# 
# Want all the programmer details (including many other parameters you can set)? Check out the [gensim documentation](https://radimrehurek.com/gensim/models/doc2vec.html#gensim.models.doc2vec.Doc2Vec).
# 

# In[ ]:


# Set up the model.
model = gensim.models.Doc2Vec(vector_size = 300, 
                              min_count = 3, 
                              epochs = 100)


# In[ ]:


model.build_vocab(book_corpus)
print("model's vocabulary length:", len(model.wv.vocab))


# In[ ]:


model.train(book_corpus,
            total_examples=model.corpus_count,
            epochs=model.epochs)


# Below we will find the books that the neural net thinks are most similar to the given one. You can put any number you like (between 0 and 19) inside the parentheses. The numbers correspond to the order of the books inside the input folder (under "Data", in the right sidebar).
# 
# The output will show you pairs which contain the book's identifier and also a similarity score. (The list will be sorted by similarity.)

# In[ ]:


model.docvecs.most_similar(12) #The_Adventures_of_Tom_Sawyer_by_Mark_Twain


# In[ ]:


model.docvecs.most_similar(11) # The_Adventures_of_Sherlock_Holmes_by_Arthur_Conan_Doyle.rtf


# In[ ]:


model.docvecs.most_similar(16) # The_Prince_by_Nicolo_Machiavelli.rtf


# Incidentally, in deciding which *documents* are most similar, the neural net has also made inferences about which *words* are most similar. You can explore what it thinks by putting different words inside the quote marks below.
# 
# As with documents, it outputs both words and similarity scores. Can you identify a threshhold score above which the similarity seems reliable, and below which it's just grasping at straws?
# 
# Keep in mind that it won't know anything about words that don't appear in its corpus. If you enter a word it doesn't know about, it will throw an error. That's fine; you can just try a different word and rerun the cell.

# In[ ]:


model.wv.most_similar("monster")

