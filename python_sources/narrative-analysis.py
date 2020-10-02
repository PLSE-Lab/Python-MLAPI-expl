#!/usr/bin/env python
# coding: utf-8

# # Clustering, Topic Modeling Consumer Mortgage Complaints
# 
# ## Background
# 
# The Consumer Financial Protection Bureau (CFPB) was created as part of the Dodd-Frank Financial Protection Act, and it's mission is to empower and educate consumers about financial products. CFPB also is responsible for enforcing financial regulations. Therefore consumer complaints hold useful information both for CFPB and the customers they serve. In the following sections we connect to the complaints data and do an exploratory analysis of the narratives provided by customers. 
# 
# Along the way we'll highlight areas that may be useful to CFPB, both in educating & empowering consumers within its Division of Consumer Education and Engagement and Division of Research Markets and Regulation. 
# 
# 
# ## Getting the Data
# 
# After importing each package, we'll query the consumer compaints database provided by CFPB by
# 
# 1. Connecting to the database file
# 2. Creating a cursor for the connection
# 3. Creating and execute a SQL query, saving the results to the 'complaints' variable
# 4. Closing the connection.

# In[ ]:


import sqlite3
import os 
import nltk
import numpy as np
import scipy

con = sqlite3.connect("../input/database.sqlite")
cur = con.cursor()
sqlString = """ 
            SELECT complaint_id, date_received, consumer_complaint_narrative, company, timely_response
            FROM consumer_complaints
            WHERE product = "Mortgage" AND 
                            consumer_complaint_narrative != ""
            """
cur.execute(sqlString)
complaints = cur.fetchall()
con.close()


# ### Peeking at the Data
# 
# The rows selected -- complaint_id, consumer_complaint_narrative, company -- are stored as a list of tuples. There is one tuple for each row. As an example, let's randomly (well, sort of...) select a complaint and print its date, narrative, and company

# In[ ]:


import random
random.seed(7040)
rand_complaint = random.randint(0, len(complaints))
print(rand_complaint)
print(len(complaints))
print(complaints[rand_complaint])


# ## Some Questions
# 
# Our random complaint is voyeristically interesting (if a little disheartening), but reading it with CFPB in mind questions come to mind, like "Are there other, similar narratives?".

# To find similar complaints we need a way to compute similarity, and to do that we need to represent each narrative's text as vectors in a matrix, a so-called 'bag-of-words'.

# ## Document-Term Matrix
# 
# The first step in answering the second question is taking our raw text and process it. For each narrative we want two pieces of information
# 
# 1. What words appear, and
# 2. How many times each of the words appears
# 
# An efficient way to do this is with a Document Term Matrix (DTM) and Vocabulary. The DTM has a row for each mortgage complaint narrative and a column for each word in the vocabulary, resulting in an MxN matrix, where M is the number of complaints and N is the number of words in the vocabulary. The *ijth* entry corresponds to the count of the *jth* vocabulary word in *the ith* narrative. 
# 
# That may seem a bit abstract if you're unfamiliar with text analysis and/or linear algebra, but the basic concept -- counting word occurrences -- is in fact quite simple. To create the DTM, we'll first extract each complaint so we have a list of complaints.

# In[ ]:


complaint_list = []
for i in range(len(complaints)):
    complaint_list.append(complaints[i][2])


# ### Stop Words
# 
# In extracting a vocabulary for the text, we want balance: Including all words used is more than we need, but too few and we won't extract any meaningful information. Once we have a vocabulary we'll count up how many times each narrative uses each word in the vocabulary. Those counts will make up the DTM.
# 
# The big idea behind creating a DTM is that each document -- in our case mortgage complaint narratives -- can be represented as a vector. Using vectors we can compute things like distance and similarity between narratives. 
# 
# But some words -- like 'the', 'a', 'it' -- occur so frequently in English text they'll be in nearly 100% of the narratives and therefore don't add much value to our DTM. Think of it like this -- if I tell you two narratives use the word 'the' and 'it' five times you likely haven't learned anything about their content, but if I tell you two narratives contain the words 'refinance' and 'foreclosure' 5 times you can begin to make some inferences about what other words they include. 
# 
# In text analysis, these frequently occuring terms are known as 'stop words'. There's a dictionary of them in the nltk package but we'll also include some words from the text reading the example narrative above we don't want in the vocabulary. 

# In[ ]:


stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(['wells', 'fargo, bank', 'america','chase', 'x','xx','xxx','xxxx','xxxxx',
                'mortgage', 'x/xx/xxxx', 'mortgage', '00'])
print(stopwords[0:11])


# ### Words to Vectors
# Now that we've got our stop words, we can create a CountVectorizer object with our stop word list and feed it our complaint list. Then we'll coerce the matrix and vocabulary to numpy arrays because they have more methods that we'll use in later computations. 

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words = stopwords)
dtm = vectorizer.fit_transform(complaint_list)
vocab = vectorizer.get_feature_names()
dtm = dtm.toarray()
vocab = np.array(vocab)


# ## Using the DTM 
# 
# Now that we've processed and vectorized our text some doors have opened. For example, for any two narratives we could compute the
# 
# - **Cosine Similarity**: A measure of how similar two narratives are. It's the measure of the cosine of the angle between two vectors. A value of 1 corresponds to an angle of 0 degrees, or equivalent vectors and 90 degrees, or orthogonal vectors. 
# - **Euclidean Distance**:  The square root of the squares of each component part. Less formally, this is the length of a line between two points.
# 
# We could use the DTM, for example, to find the narratives which are most similar to each other. As an example, we'll loop through the DTM, measure the similarity of each narrative to the Rushmore complaint we started with (the randomly selected one from the beginning), and return the complaint that's most similar. 

# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity
 
sim = 0
for i in range(len(dtm)):
    if i == 2:
        pass
    else:
        s = cosine_similarity(dtm[2].reshape(1,-1),dtm[i].reshape(1,-1))[0][0]
        if s > sim:
            sim = s
            closest = i
print(sim, closest)


# So we have the complaint we started with and another which is the most similar according to the similarity measurement we defined. But what about a human reader? Would a person reading the texts notice any similarities? Let's print them out side-by-side and see

# In[ ]:


print("COMPLAINT 2: ",complaints[2])
print("COMPLAINT ", closest,": ", complaints[closest])


# The narratives are definitely unique, but it is reasuring -- at least in validating our similarity measurement -- that both coplaints stem from the same issue. That is lenders failing to cease reporting to credit bureaus after a customer's bankruptcy. So how could CFPB use this information? 
# 
# 1. When complaints are filed the most similar complaint -- or even the 10 most similar complaints -- could be used for outreach and connecting consumers. When filing a complaint you're probably interested in how similar complaints were resolved. 
# 
# 2. Clustering topics to learn more about categories of consumer complaints. Knowing about clusters would allow CFPB to identify systemic issues. 
# 
# 3. Using the vector representations CFPB could identify words distinctive to particular categories. 

# ### Finding Nearest-Neighbors
# 
# Now that we've defined a measure of 'similarity', we can use it to sort the narratives into buckets based on their proximity to each other. Again, the DTM is an array of arrays, and the arrays tally up the number of times each vocab word occurs. To use the Rushmore complaint, 

# In[ ]:


for i in range(len(vocab)):
    if dtm[2][i] != 0:
        print(vocab[i], dtm[2][i])


# With that in mind, let's take a look at the 5 nearest neighbors to the Rushmore complaint and see what, if anything, they have in common. 

# In[ ]:


# Given item's value and list of items with values, 
# return an ordered list of 5 items from list closest to given item
def addItem(itemValue, itemIndex, lst):
    newList = lst + [(itemValue, itemIndex)]
    newList = sorted(newList)
    while len(newList) > 5:
        newList.pop(0)
    return newList


# In[ ]:


nearestNeighbors=[(0,0)]
for i in range(len(dtm)):
    if i == 2:
        continue
    value = cosine_similarity(dtm[2].reshape(1,-1),dtm[i].reshape(1,-1))[0][0]
    if value > nearestNeighbors[0][0]:
        nearestNeighbors = addItem(value, i, nearestNeighbors)


# In[ ]:


nearestNeighbors


# In[ ]:


for tpl in nearestNeighbors:
    print(complaints[tpl[1]])


# ## Clustering Companies
# Other than individual complaint similarities CFPB may be interested in similarity of companies. Rather than randomly selecting a company and retrieving its neighbors, this time we'll compare each company to all other companies. That way we can apply machine learning clustering algorithms which we can visually inspect.
# 
# First we'll get all the companies into a list.

# In[ ]:


companies = np.array([complaints[i][3] for i in range(len(complaints))])
companies_unique = sorted(set(companies))
print(len(companies_unique))


# Now we'll create an empty array the size of our vocabulary for each of the 504 companies. The, for each company, we'll fill up the empty array with the sum of the company's individual complaint vectors from the DTM we created earlier. 

# In[ ]:


# Start with an empty array for each company
dtm_companies = np.zeros((len(companies_unique), len(vocab)))
# Now, for each company we'll store the sum of the frequency of each vocab
# word in the dtm_companies array
for i, company in enumerate(companies_unique):
    dtm_companies[i, :] = np.sum(dtm[companies == company, :], axis=0) 


# Now we'll use the cosine similarity measure and the companies DTM to create a distance matrix which includes dissimilarity  between each company's narratives and all other companies.

# In[ ]:


dist = 1 - cosine_similarity(dtm_companies)


# In[ ]:


from scipy.cluster.hierarchy import ward, dendrogram
linkage_matrix = ward(dist)


# In[ ]:


from scipy.cluster.hierarchy import ward, dendrogram
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
linkage_matrix = ward(dist)
dendrogram(linkage_matrix)
plt.show()


# ## Now What?
# So we've clustered the companies, but what have we learned? From the dendogram (above), we can se there are two distinct clusters, so perhaps the mortgage complaint narratives fit within one of two categories. 
