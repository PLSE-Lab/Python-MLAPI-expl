#!/usr/bin/env python
# coding: utf-8

# **NLP(Natural Language processing with Pyhton**

# Requirements:  need to have NLTK installed, along with downloading the corpus for stopwords

# In[47]:


import nltk


# # !conda install nltk #This installs nltk
# # import nltk # Imports the library
# # nltk.download() #Download the necessary datasets

# Get the Data
# We'll be using a dataset from the UCI datasets(https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)! This dataset is already located in the folder for this section.

#  use rstrip() plus a list comprehension to get a list of all the lines of text messages:

# In[48]:


message =[line.rstrip() for line in open('../input/SMSSpamCollection')]


# In[49]:


print(len(message))


# A collection of texts is also sometimes called "corpus". Let's print the first ten messages and number them using enumerate:

# In[50]:


for message_no, message in enumerate(message[:10]):
    print(message_no, message)
    print('\n')


# Due to the spacing we can tell that this is a TSV ("tab separated values") file, where the first column is a label saying whether the given message is a normal message (commonly known as "ham") or "spam". The second column is the message itself. (Note our numbers aren't part of the file, they are just from the enumerate call).
# 
# Using these labeled ham and spam examples, we'll train a machine learning model to learn to discriminate between ham/spam automatically. Then, with a trained model, we'll be able to classify arbitrary unlabeled messages as ham or spam.
# 
# 

# Instead of parsing TSV manually using Python, we can just take advantage of pandas! Let's go ahead and import it!

# In[51]:


import pandas as pd


# We'll use read_csv and make note of the sep argument, we can also specify the desired column names by passing in a list of names.

# In[52]:


message = pd.read_csv('../input/SMSSpamCollection', sep='\t',
                           names=["label", "message"])
message.head()


# In[53]:


message.describe()


# Let's use groupby to use describe by label, this way we can begin to think about the features that separate ham and spam!

# In[54]:


message.groupby('label').describe()


# Let's make a new column to detect how long the text messages are:

# In[55]:


message['length'] = message['message'].apply(len)
message.head()


# In[56]:


import matplotlib.pyplot as plt 
import seaborn as sns


# **Data Visulation part for Spam or ham**

# In[57]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[58]:


message['length'].plot(bins=50, kind='hist') 


# In[59]:


message.length.describe()


# In[60]:


message[message['length'] == 910]['message'].iloc[0]


# In[61]:


message.hist(column='length', by='label', bins=50,figsize=(12,4))


# Text Preprocessing phase
# 
# Our main issue with our data is that it is all in text format (strings). The classification algorithms that we've learned about so far will need some sort of numerical feature vector in order to perform the classification task. There are actually many methods to convert a corpus to a vector format. The simplest is the the bag-of-words approach, where each unique word in a text will be represented by one number.
# 
# In this section we'll convert the raw messages (sequence of characters) into vectors (sequences of numbers).
# 
# As a first step, let's write a function that will split a message into its individual words and return a list. We'll also remove very common words, ('the', 'a', etc..). To do this we will take advantage of the NLTK library. It's pretty much the standard library in Python for processing text and has a lot of useful features. We'll only use some of the basic ones here.
# 
# Let's create a function that will process the string in the message column, then we can just use apply() in pandas do process all the text in the DataFrame.
# 
# First removing punctuation. We can just take advantage of Python's built-in string library to get a quick list of all the possible punctuation:
# 
# 

# In[62]:


import string

mess = 'Sample message! Notice: it has punctuation.'

# Check characters to see if they are in punctuation
nopunc = [char for char in mess if char not in string.punctuation]

# Join the characters again to form the string.
nopunc = ''.join(nopunc)


# Now let's see how to remove stopwords. We can impot a list of english stopwords from NLTK 

# In[63]:


from nltk.corpus import stopwords
stopwords.words('english')[0:10] # Show some stop words


# In[64]:


nopunc.split()


# In[65]:


clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[66]:


clean_mess


# Now let's put both of these together in a function to apply it to our DataFrame later on:

# In[67]:


def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[68]:


message.head()


# Now let's "tokenize" these messages. Tokenization is just the term used to describe the process of converting the normal text strings in to a list of tokens (words that we actually want).
# 
# Let's see an example output on on column:

# In[69]:


# Check to make sure its working
message['message'].head(5).apply(text_process)


# **Vectorization**
# Currently, we have the messages as lists of tokens (also known as lemmas) and now we need to convert each of those messages into a vector the SciKit Learn's algorithm models can work with.
# 
# Now we'll convert each message, represented as a list of tokens (lemmas) above, into a vector that machine learning models can understand.
# 
# We'll do that in three steps using the bag-of-words model:
# 
#  1. Count how many times does a word occur in each message (Known as term frequency)
# 
# 2. Weigh the counts, so that frequent tokens get lower weight (inverse document frequency)
# 
# 3. Normalize the vectors to unit length, to abstract from the original text length (L2 norm)
# 
# Let's begin the first step:
# Each vector will have as many dimensions as there are unique words in the SMS corpus. We will first use SciKit Learn's CountVectorizer. This model will convert a collection of text documents to a matrix of token counts.
# 
# We can imagine this as a 2-Dimensional matrix. Where the 1-dimension is the entire vocabulary (1 row per word) and the other dimension are the actual documents, in this case a column per text message.
# 
#                                                                
# 

# In[70]:


from sklearn.feature_extraction.text import CountVectorizer


# There are a lot of arguments and parameters that can be passed to the CountVectorizer. In this case we will just specify the analyzer to be our own previously defined function:

# In[71]:


# Might take awhile...
bow_transformer = CountVectorizer(analyzer=text_process).fit(message['message'])

# Print total number of vocab words
print(len(bow_transformer.vocabulary_))


# Let's take one text message and get its bag-of-words counts as a vector, putting to use our new bow_transformer:

# In[72]:


message4 = message['message'][3]
print(message4)


# Now let's see its vector representation:

# In[73]:


bow4 = bow_transformer.transform([message4])
print(bow4)
print(bow4.shape)


# This means that there are seven unique words in message number 4 (after removing common stop words). Two of them appear twice, the rest only once. Let's go ahead and check and confirm which ones appear twice:

# In[74]:


print(bow_transformer.get_feature_names()[4073])
print(bow_transformer.get_feature_names()[9570])


# Now we can use .transform on our Bag-of-Words (bow) transformed object and transform the entire DataFrame of messages. Let's go ahead and check out how the bag-of-words counts for the entire SMS corpus is a large, sparse matrix:

# In[75]:


messages_bow = bow_transformer.transform(message['message'])


# In[76]:


print('Shape of Sparse Matrix: ', messages_bow.shape)
print('Amount of Non-Zero occurences: ', messages_bow.nnz)


# In[77]:


sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
print('sparsity: {}'.format(round(sparsity)))


# After the counting, the term weighting and normalization can be done with TF-IDF, using scikit-learn's TfidfTransformer.
# 
# 

# ### So what is TF-IDF?
# TF-IDF stands for *term frequency-inverse document frequency*, and the tf-idf weight is a weight often used in information retrieval and text mining. This weight is a statistical measure used to evaluate how important a word is to a document in a collection or corpus. The importance increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the corpus. Variations of the tf-idf weighting scheme are often used by search engines as a central tool in scoring and ranking a document's relevance given a user query.
# 
# One of the simplest ranking functions is computed by summing the tf-idf for each query term; many more sophisticated ranking functions are variants of this simple model.
# 
# Typically, the tf-idf weight is composed by two terms: the first computes the normalized Term Frequency (TF), aka. the number of times a word appears in a document, divided by the total number of words in that document; the second term is the Inverse Document Frequency (IDF), computed as the logarithm of the number of the documents in the corpus divided by the number of documents where the specific term appears.
# 
# **TF: Term Frequency**, which measures how frequently a term occurs in a document. Since every document is different in length, it is possible that a term would appear much more times in long documents than shorter ones. Thus, the term frequency is often divided by the document length (aka. the total number of terms in the document) as a way of normalization: 
# 
# *TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document).*
# 
# **IDF: Inverse Document Frequency**, which measures how important a term is. While computing TF, all terms are considered equally important. However it is known that certain terms, such as "is", "of", and "that", may appear a lot of times but have little importance. Thus we need to weigh down the frequent terms while scale up the rare ones, by computing the following: 
# 
# *IDF(t) = log_e(Total number of documents / Number of documents with term t in it).*
# 
# See below for a simple example.
# 
# **Example:**
# 
# Consider a document containing 100 words wherein the word cat appears 3 times. 
# 
# The term frequency (i.e., tf) for cat is then (3 / 100) = 0.03. Now, assume we have 10 million documents and the word cat appears in one thousand of these. Then, the inverse document frequency (i.e., idf) is calculated as log(10,000,000 / 1,000) = 4. Thus, the Tf-idf weight is the product of these quantities: 0.03 * 4 = 0.12.
# ____
# 
# Let's go ahead and see how we can do this in SciKit Learn:

# In[78]:


from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(messages_bow)
tfidf4 = tfidf_transformer.transform(bow4)
print(tfidf4)


# We'll go ahead and check what is the IDF (inverse document frequency) of the word "u" and of word "university"?

# In[79]:


print(tfidf_transformer.idf_[bow_transformer.vocabulary_['u']])
print(tfidf_transformer.idf_[bow_transformer.vocabulary_['university']])


# In[80]:


messages_tfidf = tfidf_transformer.transform(messages_bow)
print(messages_tfidf.shape)


# We'll be using scikit-learn here, choosing the Naive Bayes classifier to start with:

# In[81]:


from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(messages_tfidf, message['label'])


# Let's try classifying our single random message and checking how we do:

# In[82]:


print('predicted:', spam_detect_model.predict(tfidf4)[0])
print('expected:', message.label[3])


# Model Evaluation

# In[83]:


all_predictions = spam_detect_model.predict(messages_tfidf)
print(all_predictions)


# We can use SciKit Learn's built-in classification report, which returns precision, recall, f1-score, and a column for support (meaning how many cases supported that classification).

# In[84]:


from sklearn.metrics import classification_report
print (classification_report(message['label'], all_predictions))


# Train Test Split

# In[85]:


from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = train_test_split(message['message'], message['label'], test_size=0.2)

print(len(msg_train), len(msg_test), len(msg_train) + len(msg_test))


# The test size is 20% of the entire dataset (1115 messages out of total 5572), and the training is the rest (4457 out of 5572). Note the default split would have been 30/70.

# Creating a Data Pipeline
# Let's run our model again and then predict off the test set. We will use SciKit Learn's pipeline capabilities to store a pipeline of workflow. This will allow us to set up all the transformations that we will do to the data for future use. Let's see an example of how it works:

# In[86]:


from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])


#  directly pass message text data and the pipeline will do our pre-processing for us! We can treat it as a model/estimator API:
# 
# 
# 

# In[87]:


pipeline.fit(msg_train,label_train)


# In[88]:


predictions = pipeline.predict(msg_test)


# In[89]:


print(classification_report(predictions,label_test))


# In[104]:


from nltk.corpus import stopwords

#doing the stamming that is taking only root of the word that is loved can be taken as love
from nltk.stem.porter import PorterStemmer


# In[109]:


#corpus is collection of text
import re
corpus =[]
for i in range(0 ,5572):
    review = re.sub('[^a-zA-Z]', ' ' ,message['message'][i])
    review =review.lower()
    review =review.split()
    review =[word for  word in review if not word in set(stopwords.words('english'))]
    portstemmer =PorterStemmer()
    review =[portstemmer.stem(word) for  word in review if not word in set(stopwords.words('english'))]
    review =' '.join(review)
    corpus.append(review)


# In[110]:


corpus


# **Creating the Bag of word model**

# In[112]:


from sklearn.feature_extraction.text import CountVectorizer


# In[114]:


cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = message.iloc[:, 1].values


# In[115]:


X


# In[116]:


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# **# Using the elbow method to find the optimal number of clusters**

# In[119]:



from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[120]:


# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)


# In[124]:


# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
#plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of meaasge')

plt.legend()
plt.show()


# In[ ]:




