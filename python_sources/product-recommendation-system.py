#!/usr/bin/env python
# coding: utf-8

# In this notebook, I will attempt at implementing **Content Based Recommendation System**.

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import string
import re
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


pre_df=pd.read_csv("/kaggle/input/flipkart-products/flipkart_com-ecommerce_sample.csv", na_values=["No rating available"])


# In[ ]:


pre_df.head()


# In[ ]:


pre_df.info()


# In this dataset user information is not provided, so we can not build user based recommendation system. Also only 1849 product_rating have non missing value in, so product rating is also not going to help much with building our recommendation system. So, I will be implementing **Content Based(Description + Metadata) Recommendation System. **

# ## ***Data Preprocessing***

# In[ ]:


pre_df['product_category_tree']=pre_df['product_category_tree'].map(lambda x:x.strip('[]'))
pre_df['product_category_tree']=pre_df['product_category_tree'].map(lambda x:x.strip('"'))
pre_df['product_category_tree']=pre_df['product_category_tree'].map(lambda x:x.split('>>'))


# In[ ]:


#delete unwanted columns
del_list=['crawl_timestamp','product_url','image',"retail_price","discounted_price","is_FK_Advantage_product","product_rating","overall_rating","product_specifications"]
pre_df=pre_df.drop(del_list,axis=1)


# In[ ]:


from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer 
lem = WordNetLemmatizer()
stop_words = set(stopwords.words('english')) 
exclude = set(string.punctuation)
import string


# In[ ]:


pre_df.head()


# In[ ]:


pre_df.shape


# I am going to drop datapoints with duplicate product.

# In[ ]:


smd=pre_df.copy()
# drop duplicate produts
smd.drop_duplicates(subset ="product_name", 
                     keep = "first", inplace = True)
smd.shape


# ### Data Cleaning

# In[ ]:


def filter_keywords(doc):
    doc=doc.lower()
    stop_free = " ".join([i for i in doc.split() if i not in stop_words])
    punc_free = "".join(ch for ch in stop_free if ch not in exclude)
    word_tokens = word_tokenize(punc_free)
    filtered_sentence = [(lem.lemmatize(w, "v")) for w in word_tokens]
    return filtered_sentence


# In[ ]:


smd['product'] = smd['product_name'].apply(filter_keywords)
smd['description'] = smd['description'].astype("str").apply(filter_keywords)
smd['brand'] = smd['brand'].astype("str").apply(filter_keywords)


# In[ ]:


smd["all_meta"]=smd['product']+smd['brand']+ pre_df['product_category_tree']+smd['description']
smd["all_meta"] = smd["all_meta"].apply(lambda x: ' '.join(x))


# In[ ]:


smd["all_meta"].head()


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
# count_matrix = count.fit_transform(smd['all_meta'])
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(smd['all_meta'])


# ### Cosine Similarity
# I will be using the Cosine Similarity to calculate a numeric quantity that denotes the similarity between two products.
# Since we have used the TF-IDF Vectorizer, calculating the Dot Product will directly give us the Cosine Similarity Score.

# In[ ]:


from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


# We now have a pairwise cosine similarity matrix for all the products in our dataset. The next step is to write a function that returns the most similar products based on the cosine similarity score.

# In[ ]:


def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    product_indices = [i[0] for i in sim_scores]
    return titles.iloc[product_indices]


# In[ ]:


smd = smd.reset_index()
titles = smd['product_name']
indices = pd.Series(smd.index, index=smd['product_name'])


# Let us now try and get the top recommendations for a few products.

# 12219    Comfort Couch Engineered Wood 3 Seater Sofa
# 12199        @home Annulus Solid Wood Dressing Table
# 11866       Ethnic Handicrafts Solid Wood Single Bed
# 11857        Ethnic Handicrafts Solid Wood Queen Bed
# 5191                    HomeEdge Solid Wood King Bed

# In[ ]:


get_recommendations("FabHomeDecor Fabric Double Sofa Bed").head(5)


# In[ ]:


get_recommendations("Alisha Solid Women's Cycling Shorts").head(5)


# In[ ]:


get_recommendations("Alisha Solid Women's Cycling Shorts").head(5).to_csv("Alisha Solid Women's Cycling Shorts recommendations",index=False,header=True)


# In[ ]:




