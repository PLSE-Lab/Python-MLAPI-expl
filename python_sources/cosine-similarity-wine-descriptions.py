#!/usr/bin/env python
# coding: utf-8

# I developed a function to determine if any of the descriptions in the corpus are similar to an external description and then print the comparison values for each description. In order to determine the similarities, I used the cosine similarity method. The cosine similarity measures the similarity between two documents. If the value is closer to 1 then those two documents are similar. In my function, I loop through each description in the corpus, compare them and calculate a value for each of them. Once the values are printed, I find the maximum value and return the description text.
# 
# 
# $$ cos(x,y) = \frac{a \cdot b}{||a|| ||b||} $$

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import cross_val_score
from sklearn.metrics.pairwise import euclidean_distances
pd.set_option('display.max_colwidth', 1500)
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


descriptions=pd.read_csv("../input/winemag-data_first150k.csv",sep=",")


# In[ ]:



# unigram vector representation
vectorizer = TfidfVectorizer(stop_words='english',
                     binary=False,
                     max_df=0.95, 
                     min_df=0.15,
                     ngram_range = (1,2),use_idf = False, norm = None)
doc_vectors = vectorizer.fit_transform(descriptions['description'])
print(doc_vectors.shape)
print(vectorizer.get_feature_names())


# In[ ]:



def comp_description(query, results_number=20):
        results=[]
        q_vector = vectorizer.transform([query])
        print("Comparable Description: ", query)
        results.append(cosine_similarity(q_vector, doc_vectors.toarray()))
        f=0
        elem_list=[]
        for i in results[:10]:
            for elem in i[0]:
                    #print("Review",f, "Similarity: ", elem)
                    elem_list.append(elem)
                    f+=1
            print("The Review Most similar to the Comparable Description is Description #" ,elem_list.index(max(elem_list)))
            print("Similarity: ", max(elem_list))
            if sum(elem_list) / len(elem_list)==0.0:
                print("No similar descriptions")
            else:
                print(descriptions['description'].loc[elem_list.index(max(elem_list)):elem_list.index(max(elem_list))])
                
                
           


# In[ ]:


comp_description("Bright, fresh fruit aromas of cherry, raspberry, and blueberry. Youthfully with lots of sweet fruit on the palate with hints of spice and vanilla.")


# In[ ]:


comp_description('A semi-dry white wine with pear, citrus, and tropical fruit flavors; crisp and refreshing.')


# In[ ]:


comp_description('BLENDED CANADIAN WHISKY ARE EXPERTLY BLENDED AND PATIENTLY AGED')

