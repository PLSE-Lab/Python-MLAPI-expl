#!/usr/bin/env python
# coding: utf-8

# #  **Outlier Detection with e-commerce data**

# In e-commerce, it is common for firms to maintain large amounts of data. These data can range from transactional data, browsing history data or the entire catalogue of products. Usually, each product will be assigned a unique product ID. In this kernel, the products in the dataset are organised into categories. Each category has a unique ID. It is known that each category may contain outlier products. Outliers include but are not limited to:
# * Product name is not available
# * Product name is conjoined with another product name
# * Product name looks clean, but is intuitively different from other names in the group, for example:
# 
# **Product:** SEAGULL NAPTH 25g WRNA / PCS **ID:** 8886012805206
# 
# **Product:** SEA GULL WARNA RENTENG **ID:** 8886012805206
# 
# **Product:** MANGKOK SAMBAL ALL VAR **ID:** 8886012805206
# 
# **Product:** SEAQULL NAPT WARNA 25GR **ID:** 8886012805206
# 
# Also, notice that there is noise in the product names i.e. "SEA GULL NAPHT 25GR SG-519W 1PCSX 1.500,00:" but this does not qualify as a wrongly matched product name.
# Our task is to **propose an outlier detection model** that identifies products within each category that are likely to be incorrect. 

# Importing libraries and dataset

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data = pd.read_csv("../input/data.csv")


# At first glance, this seems like a problem for supervised learning. However, we do not have any training sets and in fact we are only given some guidelines on what the classes should look like. The given dataset has no labelled classes and thus this seems to be a problem for unsupervised learning.
# 
# Let's look at the first 10 records in our dataset.

# In[3]:


data.head(10)


# The product names seem to contain unnecessary information such as the product's weight and packaging. These information are not really useful in telling us what exactly the product is. Also, we can see that record 7 is in lowercase letters and it even has an apostrophe to make things worse.
# 
# We should simplify the product names before moving forward.

# ## **Pre-processing**

# In[ ]:


def validateString(s): # Tells us if a string has both letters and numbers
    letter_flag = False
    number_flag = False
    for i in s:
        if i.isalpha():
            letter_flag = True
        if i.isdigit():
            number_flag = True
        # Short circuit
        if number_flag and letter_flag:
            return True
    return False

# Preprocessing
data2 = data.copy()
data2['product_name'] = data2['product_name'].apply(lambda x: " ".join([x.lower() for x in x.split()]))
data2['product_name'] = data2['product_name'].str.replace('[^\w\s]','')
data2['product_name'] = data2['product_name'].apply(lambda x: " ".join([x for x in x.split() if x.isnumeric()==False]))
data2['product_name'] = data2['product_name'].apply(lambda x: " ".join([x for x in x.split() if validateString(x)==False]))


# In[ ]:


data2.head(10)


# Now it looks way better for us to proceed. Pre-processing steps done:
# 1. Change all letters to lowercase since words in upper and lower case are essentially the same product
# 2. Remove unwanted characters such as apostrophes
# 3. Remove numeric strings since most product names in general do not use numbers
# 4. Remove alphanumeric strings since most of them are the product's weight and packaging
# 
# ## **Data Transformation**
# **Idea of approach:** Use a text transformer to transform the documents and words into a matrix. Compute the cosine similarity and run the results through a clustering algorithm.
# 
# Most unsupervised learning algorithms need a similarity or distance matrix of some kind. In order to do that, we must first convert the words into a numeric form of some sort. A common way is to transform the words into a **TF-IDF** matrix. 
# 
# In this context we define "document" as one record in our dataset. A "collection of documents" refer to all records under one unique ID. For each document, the frequency of every word is recorded and normalised by the number of words in that document. This is the term-frequency for that word in that document. Every word in a collection of documents will have a inverse-document-frequency which is defined as $\log$(Total number of documents / Number of documents with term t in it). This is how it looks like:

# ![title](https://cdn-images-1.medium.com/max/1200/1*nq59g5QnmB_NCJ_1n0NgYQ.png)
# This helps to give some numeric value to each word. Words that appear often will have a lower TF-IDF score since they do not have much meaning. On the other hand, words that appear rarely will have a higher score since they will be helpful in idenfying documents. With these scores, we can then compute some notion of distance or similarity between documents such as cosine-similarity which can then be entered into a clustering algorithm.

# We will now demonstrate the approach on products with ID 8886012805206 since we know that the product "MANGKOK SAMBAL ALL VAR" is considered an outlier. This approach can then be repeated for all other IDs. Selecting the records that we need:

# In[ ]:


data_processed = data2.loc[data['barcode']==8886012805206,:]

# Keeping another copy consisting of unprocessed test
data_org = data.loc[data['barcode']==8886012805206,:]
data_processed.head()


# We need to import the text transformer TF-IDF and transform the product names.

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer

#define vectorizer parameters
vectorizer = TfidfVectorizer()

# We only want the product names
tfidf = vectorizer.fit_transform(data_processed.iloc[:,0])


# ## **Clustering**
# Next, import our clustering algorithm. I decided to use hierarchical agglomerative clustering since it is not restricted to certain distance measures such as Euclidean distance in K-means. Also, `tfidf` is stored as a sparse matrix by default in order to save space and we need to use `.toarray()` to convert it to a normal matrix. `n_clusters` has been set to 2 since we want our products to be either classified as "outlier" or "not outlier". I chose average linkage since we are not using Euclidean metrics.

# In[ ]:


from sklearn.cluster import AgglomerativeClustering
tfidf_dense = tfidf.toarray()
hc = AgglomerativeClustering(n_clusters=2, linkage='average', affinity='cosine').fit(tfidf_dense)
hc_result = hc.labels_


# Now that we have our results, we need to do some manipulation first in order for us to merge the results with our data.

# In[ ]:


# Reset index because we need to perform a join() on indexes
data_org = data_org.reset_index()

# Create another copy for later use
data_org2 = data_org.copy()

# Convert our result into a DataFrame
hc_result = pd.DataFrame(hc_result)

# Perform the join on index
data_org = data_org.join(hc_result)

# Rename columns
data_org = data_org.rename(index=str, columns={'index':'original_index', 0:'result'})


# Let's see which products have been classified as outliers.

# In[ ]:


print(data_org.loc[data_org['result']==0,:])


# Looks about right, since we know from the information given to us that "MANGKOK SAMBAL ALL VAR" is an actual outlier and the other products are all missing a name. We can also see the first 10 results to ensure that other products have been clustered correctly.

# In[ ]:


data_org.head(10)


# I believe it is mostly correct since the remaining products do contain similar words. I also tried K-means for comparison.

# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity
# Similarity
cos_sim = cosine_similarity(tfidf)

# Convert similarity to distance
dist = 1 - cos_sim


# Similarly, I tried to obtain cosine-similarity and since K-means requires distance, I subtracted the cosine-similarity from 1. I believe this makes some sense since cosine-similarity ranges from 0 to 1. If two vectors have a high cosine-similarity, there should be a small distance between both of them which is reflected by taking 1 - `cos-sim`. Following the code format from hierarchical clustering, we have:

# In[ ]:


from sklearn.cluster import KMeans
km = KMeans(n_clusters=2).fit(dist)
km_result = km.labels_
km_result = pd.DataFrame(km_result)
data_org2 = data_org2.join(km_result)
data_org2 = data_org2.rename(index=str, columns={'index':'original_index', 0:'result'})


# Looking at the results:

# In[ ]:


data_org2.head(10)


# This doesn't seem right. It failed to identify "MANGKOK SAMBAL ALL VAR" as an outlier. There could be many possible reasons as to why a clustering algorithm performs poorly. Currently, I am suspecting that my approach for the distance matrix is wrong. Additionally, it could also be that the data is just not good for K-means since K-means has certain characteristics such as splitting the data space into approximately equal sizes which is definitely not the case in our problem since the number of outliers for each category is likely to be less than the number of correct products. I think in general, different clustering algorithms have their own use cases.

# In conclusion, we now have a method that helps us detect outliers in product names by converting the product names into a numeric form and then performing clustering on the result. Although this is only demonstrated on one particular product group, I believe it can also be used on other product groups with similar results. With that said, this is probably not the most efficient method too. It would probably be better if we could somehow generalise this to include **all** category IDs at one go. This means the number of clusters would be the number of category IDs + 1 to account for the outliers.

# In[ ]:




