#!/usr/bin/env python
# coding: utf-8

# # Introduction
# The Amazon Fine Foods Reviews Data Set contains the reviews of fine foods from Amazon. Some important statistics of this dataset are:
# 
# Number of reviews: 568,454
# Number of users: 256,059
# Number of products: 74,258
# Timespan: Oct 1999 - Oct 2012
# Number of Attributed in Data: 10
# 
# ## What are the Attributes?
#     1. Id - Review ID
#     2. ProductId - ID of the Product reviewed
#     3. UserId - ID of the User who reviewed
#     4. ProfileName - Name of User (in its Profile)
#     5. HelpfulnessNumerator - Number of People who found the review helpful
#     6. HelpfulnessDenominator - Number of People who indicated whether the review was helpful or not
#     7. Score - Rating (between 1 to 5)
#     8. Time - Timestamp of the review (in UNIX Time Stamp format)
#     9. Summary - Brief Summary (or basically the title) of the Review
#     10 . Text - Text of the Review
#     
# ## Goal
# Our main goal is to predict whether the review is positive or not. Positive Reviews can be considered as reviews having rating 4 or 5. Negative Reviews can be considered having rating 1 or 2. Rating of 3 is considered neutral (and as you will see eventually, will be ignored). Thus we are trying to determine the polarity (or sentiment) of the review.
# 
# ## Dataset Properties
# The data is available in both `.csv` and `.sqlite` formats. We will use the `sqlite` version to perform some simple SQL Queries on the data. 
# 
# So let's begin, a small journey on the Amazon Fine Foods Review Data and try to predict whether a given review is positive or not.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Data Cleaning for the Amazon Fine Foods Reviews Data Set
# Since we already have a `Score` column containing the ratings of the dataset, it's easy by assuming rating > 3 (4 or 5) as positive and rating below 3 (1 or 2) as negative while rating 3 as neutral. But since we are trying to predict whether the review is positive or negative, we will modify the dataset. We will remove the `Score` column and change it to `Positive` and `Negative` to better suit our task as done below.  

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3


# In[ ]:


conn = sqlite3.connect('/kaggle/input/amazon-fine-food-reviews/database.sqlite')
# Filtering the rows to ignore rating = 3 (Neutral Rating)
filtered_data = pd.read_sql_query("""SELECT * FROM Reviews WHERE Score != 3""",conn)
filtered_data


# Now we will give categorize the reviews with rating > 3 as positive and rating < 3 as negative. 

# In[ ]:


def partition(x):
    if x < 3:
        return "Negative"
    return "Positive"
actual_score = filtered_data["Score"]
positive_negative = actual_score.apply(partition)
filtered_data["Score"] = positive_negative
filtered_data


# Thus we now have 5,25,814 rows with reviews and their score as "Positive" and "Negative" for the and stored in the `filtered_data` variable.
# We will perform some cleaning on this data as well

# ## Data Cleaning : Deduplication
# The data tables has many duplicate entries (as will be shown below) and thus we need to remove these duplicate entries to get unbiased results for our analysis

# In[ ]:


display = pd.read_sql_query("""SELECT * FROM Reviews WHERE Score != 3 AND UserId='AR5J8UI46CURR' ORDER BY ProductId""",conn)
display


# As evident above, we have the same user having the same review on different product ids and different review ids. Usually we dont know this by default and we need to perform many queries on data to see if such problems exist.
# Here we have the same ProfileName, HelpfulnessNumeration, HelpfulnessDenominator, Score, and even same timestamp. We can look at the Amazon's Product Pages by following this URL ```https://amazon.com/dp/<ProductId>```. If you check for the above products, they are different variations of the same product. Hence the Time Stamp is same but ProductId's are different. Thus we will remove the duplicate entries by performing **deduplication** of data

# Thus we will use pandas `drop_duplicates()` to drop the duplicates in the above dataset. For this we first sort the data according to the ProductId and then just keep the first similar product review and drop all the others. This way we will have only one representative for each product

# In[ ]:


# Sorting the data according to the product id
# Parameters for sort_values are self-explanatory. Or can look at the documentation in pandas as well
sorted_data = filtered_data.sort_values("ProductId",axis=0,ascending=True,inplace=False,kind="quicksort",na_position="last")
# Deduplication of the Entries
final = sorted_data.drop_duplicates(subset=["UserId","ProfileName","Time","Text"],keep="first",inplace=False)
final


# Now we have 3,64,173 rows remaining. By simple calculation, we can find out that 3,64,173 is about 69.25% of the original 5,25,814 rows of the data

# ## Data Range Constraints
# After quick analysis, we also figure out that in certain rows, the value of `HelpfulnessDenominator` is less than the `HelfulnessNumerator` which is not practically possible (since the number of positive reviews cant exceed the total reviews of a review) and thus we remove those rows as well

# In[ ]:


# Example of the Range Constraints Issue
display = pd.read_sql_query("""SELECT * FROM Reviews WHERE Score !=3 AND Id = 44737 OR Id = 64422 ORDER BY ProductId""",conn)
display


# In[ ]:


final = final[final.HelpfulnessNumerator <= final.HelpfulnessDenominator]
final


# After performing this cleaning, we see that 3,64,171 rows of data remain. Lets look at how many Positive and Negative Reviews we have in this data.

# In[ ]:


final['Score'].value_counts()


# Remember, there is no proper way of this data cleaning and it's found using properly querying and analyzing the dataset and using your gut feeling. Another interesting thing here is that when we used the range constraints, we didnt lose much data but when we did deduplication, we lost about 31% of data. This is something we should take care about when performing cleaning, that we dont lose essential information as part of cleaning.
# We can also see that the number of positive reviews is more (about 2x) than negative reviews and this is an *imbalanced dataset*
# Also, the data cleaning done here is not exhaustive, and there might be more issues as well so you can query and analyze the data for the same

# # Text Pre-Processing
# Here we are trying to predict whether the review is positive or negative using the features of the dataset (about 8 features). Notably, we know here that the most important feature here is the `Text` of the review. How do we convert simple English Text to Vectors? This way we can represent them in some `d` dimension and then get a seperating *(hyper)plane* which separates the negative and the positive reviews.
# Our goal of converting to vectors is that semantically similar text must be closer geometrically as vectors. For example if we have 3 texts $r_1,r_2,r_3$, and if $sim(r_1,r_2) > sim(r_1,r_3)$, then $dist(v_1,v_2) < dist(v_1,v_3)$ where **sim** represents the (semantic) similarity between the texts, **dist** represents the distance between vectors and $v_1, v_2, v_3$ are the d-dimensional vector representations of $r_1, r_2, r_3$ respectively

# ## Bag of Words (BoW)
# This is a sparse vector representation of text where we form a dictionary of words and then put the count of words in the respective word position. 
# Example your dictionary contains `[a,an,are,cat,is,this]` and the sentence is *this is a cat* becomes `[1,0,0,1,1,1]`. Usually BoW Vectors are sparse since we might have a lot of unique words in the dictionary. 
# ### The Problem with Bag of Words
# Say we have 2 reviews:
# * $r_1$ : "This Pasta is very tasty and affordable"
# * $r_2$ : "This Pasta is not tasty and is affordable"
# 
# Our dictionary is `[This,Pasta,is,very,not,tasty,and,affordable]`. 
# Then the representations of $r_1$ and $r_2$ becomes:
# * $v_1$ : `[1,1,1,1,0,1,1,1]`
# * $v_2$ : `[1,1,2,0,1,1,1,1]`
# 
# If we calculate the distance between them, it comes out to be $\sqrt{3}$ which is quite low while the sentences semantically mean completely different things.
# Bag of Words is counting of common words and doesn't work very well when there are small changes in the terminology.
# We can even have a *Binary (or) Boolean Bag of Words* where instead of putting counts, we can put $1$ if the word occurs or $0$ otherwise. In such a case, the distance between the two vectors will be roughly equal to the number of differing words.
# 
# We will use some small techniques to improve our Bag of Words
# ## Text Pre-Processing Techniques
# ### Stop Words Removal
# We can remove stop words such as `is, and` etc and this way our Bag Of Words becomes smaller and meaningful. Interestingly `not` is a stop word and if we remove this word (say for the example we saw above), then the meaning might change for both ther reviews. So removing stop words might make you lose information too. But it does give you a smaller and meaningful vector since removing stop words reduces the size of the vocabulary.
# ### Case Folding
# Fancy words aside, this is basically making everything as lower case. Since say `Pasta` and `pasta` are same but they might represent different words in the vocabulary if we dont convert them to lowercase.
# ### Stemming
# Say we have two words in different reviews `tasty` and `tasteful` which come from the base word `taste` and these words speak of the same thing. So instead of calling them with 3 different words, we can change them by their root form. Thus Stemming is basically converting a word to its root form or the common form or them *stem form*. There are many Stemming Algorithms like `Porter Stemmer` or `Snowball Stemmer`.
# ### Lemmatization
# This is basically breaking the sentence into its constituent words. For example "This Pasta is very tasty" to `[This,Pasta,is,very,tasty]`. (Look up **Tokenization** as well). This is simple in English since words are separated by spaces. But say we have a word like "New York". This is a single word and cannot be broken into two seperate words New and York. How do you break sentence into words? Lemmatization is very language and context dependent. Usually done using Lemmatizers.
# 
# One more problem occurs. Say you have 2 reviews. "This pasta is tasty" and "This pasta is delicious". Here `tasty` and delicious are essentially synonyms and mean the same thing or are very similar in meaning. But in Bag of Words, we are considering them as words that are not related. In Bag of Words, we dont take the semantic meaning of the words into consideration. Techniques like `Word2Vec` takes the semantic meaning of words into context. (Explained later).
# 

# ## Unigrams, Bigrams and n-grams
# As we saw in the previous example of $r_1$ and $r_2$, after removing the stop words, they essentially convey the same meaning (they are completely opposite). One way to get around this problem is using bigrams.
# 
# Normal Lemmatization is basically unigrams. In unigrams, we create one dimension for every word. Instead if we use a pair of words, we use **bigrams**. In bigrams, pair of consecutive words are considered as a dimension!
# Example "The pasta is very tasty and affordable", we get the following bigrams `[(The,pasta),(pasta,is),(is,very),(very,tasty),(tasty,and),(and,affordable)]` (Forgive me if I missed something).
# 
# Similarly, we can have **Trigrams** where we take 3 consecutive words or **n-grams** where $n$ could be any integer. 
# Why do we have n-grams? Bag of Words in Unigrams completely discards the sequence information (for example `(very,tasty)` and `(not,tasty)`) case and using bigrams or n-grams, we can try to retain **partial sequence information** (some sequence information, but not the whole information).
# Bigrams or n-grams can be easily integrated into Bag of Words.
# 
# The obvious catch here is that the number of bigrams in any text is greater than the number of unigrams in that text and similarly as n increases, the number of n-grams increases. Thus as n increases, the dimensionality of the word vector increases.
# Eventhough not ideal, Bag of Words is very useful with bigrams (or n-grams)

# ## TF-IDF (Term Frequency and Inverse Document Frequency)
# This is an interesting variation of Bag of Words. 
# Say we have $N$ documents $r_1, r_2,..., r_N$ and each have a subset of words from a vocabulary of 6 words $w_1,w_2,...,w_6$. Bag of Words stores the occurence of each words for all the documents. Say $r_1 = w_1 w_2 w_3 w_2 w_5$, then the corresponding Bag of Words representation becomes `[1,2,1,0,1,0]`. 
# ### TF (Term Frequency)
# The **Term Frequency (TF)** $TF(w_i,r_j)$ can be defined as number of times $w_i$ occurs in $r_j$ divided by the total number of words in $r_j$ or basically $ TF(w_i,r_j) = \frac{\# w_i in r_j}{\# words in r_j} $.
# In the previos example, we can find $TF(w_2,r_1)$ as $\frac{1}{5}$.
# 
# One interesting thing here is as $0 \leq TF(w_i,r_j) \leq 1$, we can interpret it as a probability. So Term Frequency basically says what the is the probability of finding a given word in the document. Thus intuitively term frequency intuitively refers to how many times a word occurs in the document or probability of finding word $w_i$ in document $r_j$
# 
# ### IDF (Inverse Document Frequency)
# Say we have $N$ documents. Inverse Document Frequency is for a word in a corpus. Here our corpus is documents $r_1,...,r_N$ and lets call it D. The IDF of a word $w_i$ on a corpus $D$ is defined as $IDF(w_i,D) = \log{\frac{N}{n_i}}$ where $N$ is the total number of documents and $n_i$ is the number of documents containing the word $w_i$.
# 
# Notably, $n_i \leq N$ or basically $\frac{N}{n_i} \geq 1$ or $\log{\frac{N}{n_i}} \geq 0$. Thus IDF is always $\geq 0$
# 
# Here if $n_i$ increases, the $IDF$ decreases and vice versa (since IDF is inversely proportional to $n_i$). This essentially means that if a word is more frequent in the corpus, the value of IDF of that word will be lower. Thus rarer words have higher IDF and common words have lesser IDF.
# 
# ### Combining TF and IDF
# Say we have corpus as $r_1,...,r_N$ with words $w_1,...,w_n$. To get the vector representation of $r_1$, and for the cell $w_1$, we put the value $TF(w_1,r_1) \times IDF(w_1,D)$ or basically:
# 
# When we try to convert a document $r_i$ to vector $v_i$, then the cell $v[i,j]$ will be $TF(w_j,r_i) \times IDF(w_j,D)$. Interestingly, the value of $TF$ is higher if $w_j$ is frequent in the document and $IDF$ is higher if the word is rare in the corpus. Thus this way we are giving weights to words that occur frequently but also to words that are rare. This way we give more importance to rarer words in the corpus and also more importance to frequent words in a document.
# 
# ### Limitations of TF-IDF
# Eventhough after all this, TF-IDF cant take the semantic meaning of words into account and thus cases like "cheap & affordable" or "tasty & delicious" will still have an issue. 
# 
# To solve the semantic meaning issue, we will use `Word2Vec`
# 
# 

# ## Word2Vec
# We are in the home-stretch now. As we mentioned earlier, this takes the semantic meaning of the word into picture unlike Bag of Words and TF-IDF techniques. Right now we are going to take it as a black box which takes in a word and gives a corresponding vector for that word. Might include a later, more mathematical explaination of word2vec later. For now you can refer to this [link](https://www.tensorflow.org/tutorials/text/word_embeddings). In a nutshell, word2vec represents a word as a dense d-dimensional vector. But this is not a sparse vector (unlike Bag of Words and TF-IDF who represent text/sentences in sparse vectors). Say word2vec represents words $w_1,w_2,w_3$ as vectors $v_1,v_2,v_3$ respectively. If $w_1, w_2$ are (semantically) similar, then $v_1, v_2$ are geometrically closer in the d dimensional space.
# 
# Word2Vec can also understand relationships between the words. For example $v_{man} - v_{woman}$ is parallel to $v_{king} - v_{queen}$ or basically expressions like `king - man + woman ~ queen`.
# 
# Usually, more the dimension of the word vectors, richer information is stored in the vectors (and can learn more complex relationships). But to learn such complex relationships, we also need a larger data corpus.
# 
# At its core, word2vec looks at the sequence information in text. Intuitively, it checks the words that occur in the neighborhood of a given word whose vector we are finding and if neighborhoods of the word is similar, the vectors for those words will be similar as well.

# In[ ]:





# In[ ]:




