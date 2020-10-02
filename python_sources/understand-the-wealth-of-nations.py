#!/usr/bin/env python
# coding: utf-8

# # Introduction
# ![Wealth of Nations](https://upload.wikimedia.org/wikipedia/commons/1/1a/Wealth_of_Nations.jpg)
# 
# An Inquiry into the Nature and Causes of the Wealth of Nations, generally referred to by its shortened title The Wealth of Nations, is the magnum opus of the Scottish economist and moral philosopher Adam Smith. First published in 1776, the book offers one of the world's first collected descriptions of what builds nations' wealth, and is today a fundamental work in classical economics. By reflecting upon the economics at the beginning of the Industrial Revolution, the book touches upon such broad topics as the division of labour, productivity, and free markets.

# # What is this kernel for ?
# You may wonder why I am using Smith's book here. 
# For this project my goal is to apply **Natural Language Processing** techniques to analyse and extract insights in the book. 
# This book is very referred in the Economic Litterature but few of the people who cited it have actually read it. In this project I would like to study the core thoughts of Adam Smith

# # Text reading and cleaning
# From now I am going to read the text in memory and analyse it. 
# The text is freely available on the [Project Gutenberg website](https://www.gutenberg.org/ebooks/3300). I downloaded it in text format beforehand and uploaded on Kaggle.  
# Project Gutenberg was the first provider of free electronic books, or eBooks. Michael Hart, founder of Project Gutenberg, invented eBooks in 1971 and his memory continues to inspire the creation of eBooks and related technologies today.
# 
# ## Reading it with `open()`
# The built-in `open()` function helps us read the text. We are going to open it in read mode.

# In[ ]:


text_path = "../input/an-inquiry-into-the-nature-and-causes-of-wealth/An Inquiry.txt"
raw_text = open(text_path, "r").read()


# In[ ]:


type(raw_text), len(raw_text)


# We assign the content of the text to the raw_text variable. From here we can have a glimpse of the text. 
# Notice that Python reads the text as a large single string. 
# From now what we can do with the text is to slice some part of it to display. For example we can print the thousand first characters for sanity check.

# In[ ]:


print(raw_text[0:1000])


# As we see in this slice of the string, there are meta information provided by the Project Gutenberg in the beginning of the book. These information are not part of the original book, so we will remove them. To do so we are reading the text from the 719th index.  
# The last 18868 characters are also legal disclaimers added by the Project Gutenberg, so I will also remove them to have the text only.

# In[ ]:


text = raw_text[719:-18868]


# Since the text is read as a big string object we cannot really exploit it. To make it exploitable we need to tokenize the text.  
# There are two methods to tokenize the text : 
# - we can tokenize sentences or   
# - tokenize tokens (words, punctuation ...). 
# 
# # EXPLORATORY DATA ANALYSIS
# 
# In the first part of this project I want to explore the text in an statiscal fashion. So I will tokenize the text such that I can count the occurence of every token (word) used in the corpus.

# In[ ]:


from nltk.tokenize import word_tokenize


# In[ ]:


all_tokens = word_tokenize(text)


# The `word_tokenize()` function returns a list containing all the tokens that are in the corpus. Let's have a glance.

# In[ ]:


all_tokens[1:10]


# Now every token is represented as an element of the list. The advantage is that we can count the occurence of every token.  
# But it is likely that punctuations and stop words are the most frequent words in the corpus. 

# In[ ]:


from collections import Counter


# In[ ]:


token_counts = Counter(all_tokens)
token_counts.most_common(10)


# We see that `,` is the most frequent word in the corpus. In overall the 10 most common words do not give us any clue on what the text is about. We'll need to clean it before visualizing the text.

# ## Remove punctuations & stopwords
# 
# An easy way to remove punctuations in the list of the tokens is to use the `.isalpha()` method of the string class. Knowing that we can filter the list to keep only alphanumeric tokens.

# In[ ]:


# An example
"work".isalpha(), ",".isalpha()


# In[ ]:


alpha_tokens = [word for word in all_tokens if word.isalpha()]
len(all_tokens), len(alpha_tokens)


# In[ ]:


token_counts = Counter(alpha_tokens)
token_counts.most_common(10)


# Now that we have successfully removed the punctuations, we need to remove the stop words. Stop words are words that do not have a particular meaning. They are used to connect ideas, and making the language smooth. 
# The Natural Language Toolkit (nltk) package comes with a list of stopwords in English.   
# 
# Before doing so it's important to lower all the tokens. It's a strategy to normalize the text so there's no difference between "The", "the", or "THE".

# In[ ]:


# An example of stopwords in English
from nltk.corpus import stopwords
stopwords.words("english")[10:20]


# To remove the stopwords in the text we can write a list comprehension. 

# In[ ]:


# May take a little time to run
alpha_tokens = [word.lower() for word in alpha_tokens]
cleaned_tokens = [word for word in alpha_tokens if not word in stopwords.words("english")]

len(cleaned_tokens), len(alpha_tokens)


# In[ ]:


token_counts = Counter(cleaned_tokens)
token_counts.most_common(10)


# Now we are ready to go since we've cleaned the tokens.

# # VISUALIZATION

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

# Configure the way the plots will look
plt.style.use([{
    "figure.dpi": 300,
    "figure.figsize":(12,9),
    "xtick.labelsize": "large",
    "ytick.labelsize": "large",
    "legend.fontsize": "x-large",
    "axes.labelsize": "x-large",
    "axes.titlesize": "xx-large",
    "axes.spines.top": False,
    "axes.spines.right": False,
},'seaborn-poster', "fivethirtyeight"])


# In[ ]:


from wordcloud import WordCloud
str_cleaned_tokens = " ".join(cleaned_tokens) # the word cloud needs raw text as argument not list
wc = WordCloud(background_color="white", width= 800, height= 400).generate(str_cleaned_tokens)
plt.imshow(wc)
plt.axis("off");


# # MODELLING
# 
# So far what we've done is to split the text into tokens that we have studied individually. Even if this technique have given us some insights about the text, it is not enough because we took each token individually as if the book only contains 1262 occurences of price, 1232 occurences of country ignoring all the surroundings of these occurences.  
# 
# I want to come up with a technique that captures the surroundings of each token so that I can have a better understanding of the corpus.  
# I am going to use a family of word embedding algorithms called word2vec.
# 
# These models are shallow, two-layer neural networks that are trained to reconstruct linguistic contexts of words. Word2vec takes as its input a large corpus of text and produces a vector space, typically of several hundred dimensions, with each unique word in the corpus being assigned a corresponding vector in the space. **Word vectors are positioned in the vector space such that words that share common contexts in the corpus are located close to one another in the space**. (Wikipedia)
# 
# I will then use the properties of this techniques to visualize the data more accurately.

# ## Word2vec
# 
# The word2vec algorithm is implemented in the `gensim` library. It takes as input a sequence of sequences like this 
# ```python
# [[learning, data, science, good], [data, science, hot, job]]
# ```
# 
# To make our corpus look like a sequence of sequence we'll write some functions to automate the process.  
# The functions will :
# - lower the text
# - remove punctuations
# - remove stop words
# 

# Remember we say, we can tokenize our text into tokens or into sentences. For the exploratory part we were interested in counting occurences of each token, so we tokenized the text int individual tokens. Here we will tokenize it into sentences.

# In[ ]:


from nltk.tokenize import sent_tokenize, RegexpTokenizer


# In[ ]:


sentences = sent_tokenize(text)


# Here's how a sentence looks like.

# In[ ]:


print(sentences[100])


# As we can read, each sentence contains punctuations, stop words

# In[ ]:


tokenizer = RegexpTokenizer(r'\w+')

def remove_stopwords(text, stopw = stopwords.words("english")):
    list_of_sentences = []
    
    for sentences in text:
        list_of_words = []
        for word in sentences:
            if not word in stopw:
                list_of_words.append(word)
        list_of_sentences.append(list_of_words)
    return list_of_sentences

def clean_sent(sentences):
    """Sentence must be a list containing string"""
    stopw = stopwords.words("english")
    # Lower each word in each sentence        
    sentences = [tokenizer.tokenize(sent.lower()) for sent in sentences]
    sentences = remove_stopwords(sentences)
    return sentences


# In[ ]:


cleaned_sentences = clean_sent(sentences)


# Here's how the sentence will look after we apply the function on it.

# In[ ]:


print(cleaned_sentences[100])


# ## Fitting the word2vec model. 
# 
# Now that the data is prepared we can feed it into the word2vec algorithm.

# In[ ]:


from gensim.models import Word2Vec


# In[ ]:


model = Word2Vec(
    min_count= 10,# minimum word occurence 
    size = 300, # number of dimensions
    alpha = 0.01, #The initial learning rate
)


# In[ ]:


model.build_vocab(cleaned_sentences)
model.train(cleaned_sentences, total_examples = model.corpus_count, epochs = 60)


# In[ ]:


model.wv.most_similar("wealth")


# In[ ]:


model.wv.most_similar("france")


# In[ ]:


model.wv.most_similar("africa")


# In[ ]:


import pandas as pd
similar = pd.DataFrame(model.wv.most_similar("africa", topn= 10), columns = ["name", "height"])


# In[ ]:


similar.plot.barh(x = "name", y = "height");


# In[ ]:


model.wv.most_similar("king")


# Now it would be interesting to visualize all the corpus. 
# But first, we can store all the word vectors in a Pandas data frame.

# In[ ]:


all_words = model.wv.vectors


# In[ ]:


def wv_to_df(model):
    all_wv = model.wv.vectors
    
    df = pd.DataFrame(
        all_wv,
        index = model.wv.vocab.keys(),
        columns = ["dim" + str(i+1) for i in range(all_wv.shape[1])]
    )
    return df


# In[ ]:


df = wv_to_df(model)


# In[ ]:


df.head()


# In[ ]:


df["idx"] = df.index
df.head()


# In[ ]:


ax = df.plot.scatter("dim1", "dim2")
for i, point in df.iterrows():
    ax.text(point.dim1 + 0.005, point.dim2 + 0.008, point.idx)


# In[ ]:


ax = df.plot.scatter("dim10", "dim20")
for i, point in df.iterrows():
    ax.text(point.dim10 + 0.005, point.dim20 + 0.008, point.idx)


# It would be interesting if we could choose to zoom in a certain region of the scatterplot instead of having that big black hole. We can define a function that does that for us.

# In[ ]:


def plot_region(df, x, y,label, x_bounds, y_bounds, s=35, ftsize = None):
    slices = df[
        (x_bounds[0] <= df[x]) &
        (df[x] <= x_bounds[1]) & 
        (y_bounds[0] <= df[y]) &
        (df[y] <= y_bounds[1])
    ]
    print(slices.shape)
    ax = slices.plot.scatter(x, y, s=s)
    for i, point in slices.iterrows():
        ax.text(point[x] + 0.005, point[y] + 0.005, point[label], fontsize = ftsize)


# Now let's try to zoom in (-0.5, 0) on the x-axis and (-1, -0.5) on the y-axis.

# In[ ]:


plot_region(df, "dim1", "dim2", "idx", (-0.5, 0), (-1, -0.3))


# # DIMENSIONALITY REDUCTION
# 
# The word2vec model we build yields a 300 hundred dimensions dataset. This is quite huge. In order to represent the variability of the data in a 2D or 3D graph, it is important to reduce the dimensionality of the dataset. We can do this using t-SNE and PCA.

# ## Dimensionality reduction with PCA
# 
# Principal component analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components. This transformation is defined in such a way that the first principal component has the largest possible variance (that is, accounts for as much of the variability in the data as possible), and each succeeding component in turn has the highest variance possible under the constraint that it is orthogonal to the preceding components. The resulting vectors (each being a linear combination of the variables and containing n observations) are an uncorrelated orthogonal basis set. (Wikipedia)
# 
# The reason why I decide to use PCA here is not because the dimensions are correlated among them but because the PCA procedure will transform the data in a way that most of the variability of the dataset is found in the first n components.  
# The procedure is implemented in the scikit-learn library.

# In[ ]:


from sklearn.decomposition import PCA


# In[ ]:


pca_dimension_reduction = PCA(n_components= 2)
res =  pca_dimension_reduction.fit_transform(df.drop(columns = "idx"))


# In[ ]:


pca_coords = pd.DataFrame(res, columns = ["x", "y"])
pca_coords["words"] = df.index


# In[ ]:


pca_coords.head()


# In[ ]:


plot_region(pca_coords, "x", "y", "words", (0, 1), (1, 2))


# # t-SNE
# It is a nonlinear dimensionality reduction technique well-suited for embedding high-dimensional data for visualization in a low-dimensional space of two or three dimensions. Specifically, it models each high-dimensional object by a two- or three-dimensional point in such a way that similar objects are modeled by nearby points and dissimilar objects are modeled by distant points with high probability.

# In[ ]:


from sklearn.manifold import TSNE


# In[ ]:


tsne_dimension_reduction = TSNE(n_components=2)
res = tsne_dimension_reduction.fit_transform(df.drop(columns = "idx"))


# In[ ]:


res.shape


# In[ ]:


tsne_coords = pd.DataFrame(res, columns = ["x", "y"])
tsne_coords["words"] = df.index


# In[ ]:


tsne_coords.plot.scatter(x = "x", y = "y")
plt.title("Book corpus in 2D");


# In[ ]:


plot_region(tsne_coords, "x", "y", "words", (5, 20), (20, 40))


# In[ ]:




