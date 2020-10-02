#!/usr/bin/env python
# coding: utf-8

# # Which wine should I try next?
# ## -Creating a recommendation system using sklearn's CountVecterizer and TfidfVectorizer-
# 
# I like wine, but I am not a wine aficionado. Yet. In a wine shop, I mostly find myself overwhelmed by so many different types of wine. Even 20 minutes in, I still cannot pick a bottle to buy.
# 
# 
# One of the safest ways to venture out and expand a list of wines you like would be to find grape varieties that are similar to your favorite. In this kernel, we will use **CountVecterizer and TfidfVectorizer in sklearn.feature_extraction.text** to create a recommendation system that takes in your favorite grape variety as an input and generates a DataFrame displaying three similar wines and key words in their reviews.
# 
# You can find the dataset used in this kernel [here](https://www.kaggle.com/zynicide/wine-reviews). This kernel is inspired by a fellow Kaggler's [kernel](https://www.kaggle.com/ibtesama/getting-started-with-a-movie-recommendation-system) about different recommendation systems. I highly recommend reading [her work](https://www.kaggle.com/ibtesama/getting-started-with-a-movie-recommendation-system).
# 
# Let's explore the databset first!

# # Data exploration

# In[ ]:


import numpy as np 
import pandas as pd 

# Importe two csv files
original_data_1=pd.read_csv("/kaggle/input/wine-reviews/winemag-data_first150k.csv")
original_data_2=pd.read_csv("/kaggle/input/wine-reviews/winemag-data-130k-v2.csv")


# In[ ]:


original_data_1.head()


# In[ ]:


original_data_2.head()


# The recommendation system we will create needs two pieces of information only: variety and description. Let's create a DataFrame containing these two columns.

# In[ ]:


# Create a single DataFrame that contains variety and description only. Delete any rows that are duplicated or contain missing data.
variety_description= original_data_1[["variety", "description"]].append(original_data_2[["variety", "description"]])
variety_description=variety_description.drop_duplicates().dropna()
variety_description.head()


# > Looks good. Tinta de Toro? I've never heard of it!

# In[ ]:


# How many grape varieties are there in this DataFrame?
len(variety_description["variety"].unique().tolist())


# In[ ]:


variety_description.shape


# >There are **756 different varieties of grapes** represented in this dataset and **a total of 169,451 descriptions** of different wines. This indicate that we have multiple descriptions for a given wine.

# In[ ]:


# Create and display the chart showing the number of reviews per grape variety for the top 30 wines
variety_description["variety"].value_counts().iloc[:30].plot.bar()


# > Pinot Noir has the most reviews (16652 reviews or 10% of the total reviews), followed by Chardonnay and Cabernet Sauvignon.

# In[ ]:


# Count the number of reviews per grape variety. This returns a series.
variety_rev_number=variety_description["variety"].value_counts()

# Convert the Series to Dataframe
df_rev_number=pd.DataFrame({'variety':variety_rev_number.index, 'rev_number':variety_rev_number.values})
df_rev_number[(df_rev_number["rev_number"]>1)].shape


# > Out of 756 grape varieties, 603 have more than one reviews. These will be subject to the process in which we will grab top 100 common words from multiple reviews on a single grape variety.

# In[ ]:


# Create a ist of grape varieties that have more than one review
variety_multi_reviews=df_rev_number[(df_rev_number["rev_number"]>1)]["variety"].tolist()

# Create a ist of grape varieties that have only one review
variety_one_review=df_rev_number[(df_rev_number["rev_number"]==1)]["variety"].tolist()


# # Get top 100 common words in the review using CounterVectorizer and TfidfTransformer
# ## Why do we need to do this step?
# 
# * When people want a recommendation based on the wine they like, they are looking for a different grape variety, *not* the same wine from a differen region.  
# * A total of 603 varieties of grape has multiple reviews in this dataset.
# * Then, what is the best way to get a **representative review** per grape variety?  ---> Get the **top 100 words commonly used** in the reviews on the same grape variety. In other words:
# ![Screen%20Shot%202020-06-17%20at%203.35.57%20PM.png](attachment:Screen%20Shot%202020-06-17%20at%203.35.57%20PM.png)
# 
# * How can I extract top 100 common words? Use CountVectorizer!
# 
# 

# ## CountVectorizer and TfidfTransformer
# * CountVectorizer counts how many times each word in a given text appears.
#     * eg. "he": 2, "most":3, "the": 2.
# * Then, how can we extract the commonly used words in the text? We can use IDF (Inverse Document Frequency), which indicates which term/word is unique (or less frequently used). It's inverse, meaning that the lower IDF is, the more frequently the term is appearing in the text.
# * TfidfTransformer computes the IDF value for each term in the text.
# * Actually seeing the results of CountVectorizer and TfidfTransformer will help understand what they do. So, I ran them using a very simple text as an example below.
# 
# ### Demo of CountVectorizer with a simple example

# In[ ]:


# This demo is modified from https://kavita-ganesan.com/tfidftransformer-tfidfvectorizer-usage-differences/

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

docs=["there was a king named Matati", 
      "He had a lot of gold", 
      "He loved the gold most", 
      "The beginning of the Matati gold story"]

cv=CountVectorizer()

word_count_vect=cv.fit_transform(docs)

# Display the result of CountVectorizer output (Reference: https://gist.github.com/larsmans/3745866)

print("the result of CountVectorizer") 
print(pd.DataFrame(word_count_vect.A, columns=cv.get_feature_names()).to_string())


# Use TfidfTransformer to compute the IDF values
tfidf_trans=TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_trans.fit(word_count_vect)

# Display the IDF value for each term in the text
df_idf=pd.DataFrame(tfidf_trans.idf_, index=cv.get_feature_names(), columns=["idf_values"])
df_idf.sort_values(by=['idf_values'])


# ### Let's work on our dataset

# In[ ]:


variety_description=variety_description.set_index("variety")


# In[ ]:


from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

variety_description_2=pd.DataFrame(columns=["variety","description"])

# Define a CountVectorizer object
    # stop_words="english": Remove all the uninformative words such as 'and', 'the' from analysis
    # ngram=range(1,2): means unigrams and bigrams
cv=CountVectorizer(stop_words="english", ngram_range=(2,2))

# Define a TfidfTransformer object
tfidf_transformer=TfidfTransformer(smooth_idf=True, use_idf=True)

for grape in variety_multi_reviews:

    df=variety_description.loc[[grape]]

    # Generate word counts for the words used in the reviews of a specific grape variety
    word_count_vector=cv.fit_transform(df["description"])

    # Compute the IDF values
    tfidf_transformer.fit(word_count_vector)

    # Obtain top 100 common words (meaning low IDF values) used in the reviews. Put the IDF values in a DataFrame
    df_idf=pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(), columns=["idf_weights"])
    df_idf.sort_values(by=["idf_weights"], inplace=True)

    # Collect top 100 common words in a list
    common_words=df_idf.iloc[:100].index.tolist()
   
    # Convert the list to a string and create a dataframe
    common_words_str=", ".join(elem for elem in common_words)
    new_row= {"variety":grape, "description":common_words_str}

    # Add the variety and its common review words to a new dataframe
    variety_description_2=variety_description_2.append(new_row, ignore_index=True)


# > Here, the resulting DataFrame only has the grape varities with multipe reviews. We should add the grape varieties that got only one review.

# In[ ]:


variety_description_2=variety_description_2.set_index("variety")
variety_description_2=variety_description_2.append(variety_description.loc[variety_one_review])
variety_description_2


# In[ ]:


variety_description_2.shape


# > Great! We have the information about all 756 grape varieties. Now we are ready to do TF-IDF analysis!

# # TF-IDF analysis using TfidfVectorizer

# ## What is TF-IDF?
# * We already know that IDF stands for Inverse Document Frequency. TF-IDF is simply IDF multiplied by TF (Term frequency). TF-IDF will be high if 1) the term is unique (high IDF) in the whole document and 2) that term appeared frequently in a given text (e.g., a description of a specific wine).
# * Thus, the higher TF-IDF score of a term is, the more informative the term is.
# * We will do TF-IDF calculation using TfidfVectorizer. We can use CountVectorizer and then TfidfTransformer, just like above, but TfidfVectorizer would do all the steps required to get TF-IDF at once.

# In[ ]:


# Load a relevant library
from sklearn.feature_extraction.text import TfidfVectorizer

# Define a TfidVectorizer object. Remove all the uninformative words such as 'and,' 'the,' and 'him' from analysis. Bigrams only (ngram_range=(2,2)).
tfidf=TfidfVectorizer(stop_words="english", ngram_range=(2,2))

# Count the words in each description, calculate idf, and multiply idf by tf.
tfidf_matrix=tfidf.fit_transform(variety_description_2["description"])

# Resulting matrix should be # of descriptions (row) x # of bigrams (column)
tfidf_matrix.shape


# > Our 756 grape varieties are in rows and there are 65484 bigrams used in the reviews.

# ## Cosine Similarity
# ### How can we "measure" the similarity between the descriptions of two different grape varieties?
# 
# * There are several ways to quantify the similarity, but here we will be using **cosine similarity**.
# * Cosine similarity:
# ![Screen%20Shot%202020-06-17%20at%207.23.42%20PM.png](attachment:Screen%20Shot%202020-06-17%20at%207.23.42%20PM.png)
#     * Using TfidfVectorizer, we converted a text into a matrix of TF-IDF values. 
#     * Imagine plotting descriptions of two wines, such as Pinot Noir and Chardonnay, onto a space (although the graph above is only 2-dimensional). 
#     * The angle between them indicates how close or far they are.
#     * Thus, the more similar two descriptions are, the smaller angle is, and the higher cosine is.
#     * A detailed explanation of the cosine similarity could be found [here](https://www.machinelearningplus.com/nlp/cosine-similarity/)

# In[ ]:


# Since we used TfidfVectorizer to convert the text into a matrix, we can use linear_kernel to get cosine similarity, instead of sklearn's cosine_similarity
# Load linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# The cosine similarity matrix above does not contain the grape variety information anymore. So, we need a Series that we can use later to find the grape variety based on the index.

# In[ ]:


# Create a Series, where the index is the grape variety and the element is the index of the wine in the dataset.
variety_description_2=variety_description_2.reset_index()
indices = pd.Series(variety_description_2.index, index=variety_description_2['variety'])


# In[ ]:


# Make a function that takes in the grape variety as an input and produces a DataFrame of three similar varieties and key words of their reviews

def what_should_I_drink_next(grape, cosine_sim=cosine_sim):
    # Get the index of the input wine
    idx = indices[grape]

    # Get the pairwise similarity scores between the input wine and all the wines
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the wines based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Select the top three similarity scores
    sim_scores = sim_scores[1:4]

    # Get the grape variety indices
    wine_idx_list = [i[0] for i in sim_scores]
     
    # Create the output dataframe
    df=pd.DataFrame(columns=["similar wines", "Top 6 common words in wine reviews"])
     
    for wine_idx in wine_idx_list:
     
        g_variety=variety_description_2.iloc[wine_idx]["variety"]
    
        # Get top 6 common words in the review
        des=variety_description_2.iloc[wine_idx]["description"]
        
        if g_variety in variety_multi_reviews:     # If the wine has more than one reviews
            des_split=des.split(", ")
            key_words_list=des_split[:6]
            key_words_str=", ".join(key_words_list)
        
        else:
            key_words_str = des
            
        new_row={"similar wines": g_variety, "Top 6 common words in wine reviews": key_words_str}
        df=df.append(new_row, ignore_index=True)
    
    df.set_index("similar wines") 
    
    # Widen the column width so that all common words could be displayed
    pd.set_option('max_colwidth', 500)
   
    return df  


# In[ ]:


what_should_I_drink_next("Pinot Noir")


# In[ ]:


what_should_I_drink_next("Shiraz")


# # Thank you so much for reading this kernel!
# 
# I always learn a lot from other Kagglers' work and I hope mine was helpful to someone.
