#!/usr/bin/env python
# coding: utf-8

# # Ingredient recommender system.
# _More like this on [kachkach.com](http://kachkach.com)_
# 
# In this notebook, we will use the dataset to build an ingredient recommender system. We will go from the most basic (counting ingredient co-occurrences) to the slightly more elaborate (matrix factorization), also looking at a useful information theory metric called "Pointwise Mutual Information" (PMI).

# ## Data loading, imports.

# In[ ]:


import numpy as np
import pandas as pd

import os
print(os.listdir("../input"))


# In[ ]:


df = pd.concat([pd.read_json('../input/train.json'), pd.read_json('../input/test.json')]).reset_index()
# Lower-casing all ingredients.
df.ingredients = df.ingredients.apply(lambda ings : [ing.lower() for ing in ings])


# In[ ]:


df.sample(5)


# ## Calculating ingredient co-occurrences.
# 
# We will start by calculating the number of recipes in which to ingredients occurred together. This intuitively gives us a measure of how common it is to see two ingredients mixed up, and is a good first attempt at building our ingredient recommender:

# In[ ]:


import itertools
# Example of what the itertools.combinations function does.
list(itertools.combinations(df.ingredients[0][:5], 2))


# In[ ]:


# Calculating ingredient counts and co-occurrences in recipes.
from collections import Counter
cooc_counts = Counter()
ing_count  = Counter()
for ingredients in df.ingredients:
    for ing in ingredients:
        ing_count[ing] += 1
    for (ing_a, ing_b) in itertools.combinations(set(ingredients), 2):
        # NOTE: just making sure we added pairs in a consistent order (a < b); you can also add both (a,b) and (b,a) if you want.
        if ing_a > ing_b:
            ing_a, ing_b = ing_b, ing_a
        cooc_counts[(ing_a, ing_b)] += 1


# In[ ]:


cooc_df = pd.DataFrame(((ing_a, ing_b, ing_count[ing_a], ing_count[ing_b], cooc) for (ing_a, ing_b), cooc in cooc_counts.items()), columns=['a', 'b', 'a_count', 'b_count', 'cooc'])
cooc_df.sample(10)


# In[ ]:


cooc_df[cooc_df.a == 'chillies'].sort_values('cooc', ascending=False).head(10)


# Notice how the elements with the highest co-occurrence count are not necessarily very similar, just overall very popular ingredients.

# ## Pointwise Mutual Information
# 
# We surfaced a number of issues with using the raw number of co-occurrences, mainly the fact that this measure is highly biased by the popularity of either items. One better metric when looking at correlation is [PMI](https://en.wikipedia.org/wiki/Pointwise_mutual_information), and its formula goes something like this:****
# 
# $$PMI(A, B) = log \frac{P(A, B)}{P(A) \times P(B)}$$

# In[ ]:


# We calculate P(A), P(B) and P(A, B) and PMI(A, B) from the previous df.
# P(A) is counts(A) / num_recipes
# P(A, B) is coocs(A, B) / sum(coocs)
p_a = cooc_df.a_count / sum(ing_count.values())
p_b = cooc_df.b_count / len(ing_count.values())
p_a_b = cooc_df.cooc / cooc_df.cooc.sum()
cooc_df['pmi'] = np.log(p_a_b / (p_a * p_b))


# Simply ordering by PMI values givers us an extra argument for removing rare ingredients: they have a noisy PMI value, since we didn't see them in enough contexts to give enough support to the "lift ratio" that PMI is. 
# This also shows that we really should be using unigrams/bigrams of the ingredient's textual description, instead of using the full ingredient name, as "vinegar" should be treated similarly to "red vinegar", but the current approach treats these two ingredients as being totally different.

# In[ ]:


cooc_df.sort_values('pmi', ascending=False).head(10)


# I would go all in when it comes to filtering low frequency ingredients.
# For low values of `min_count`, we get some very peculiar pairs which are likely due to the recipes being from the same website that has some advertising partnership with the brands mentionned.

# In[ ]:


min_count = 5
cooc_df[(cooc_df.a_count >= min_count) & (cooc_df.b_count >= min_count)].sort_values('pmi', ascending=False).head(20)


# With values around 40 or 50, we start to see real correlations appear:

# In[ ]:


min_count = 30
cooc_df[(cooc_df.a_count >= min_count) & (cooc_df.b_count >= min_count)].sort_values('pmi', ascending=False).head(20)


# We can also look at the pairs with the lowest PMI. We filter out pairs with only one co-occurrence, as those seem like outliers.
# Notice also that we cannot use negative values to acertain how unlikely a combination is: if a pair is so unlikely, we will have little to no support for it (small or null number of coocs). Because of this, we usually drop the negative PMI values.

# In[ ]:


min_count = 30
cooc_df[(cooc_df.a_count >= min_count) & (cooc_df.b_count >= min_count) & (cooc_df.cooc > 1)].sort_values('pmi', ascending=True).head(20)


# ## Matrix Factorization
# 
# There's an issue with the approach we used previously: we are only leveraging direct correlations between ingredients (say, the fact that there are 15 recipes with both `sushi rice` and `wasabi`) and not using all the knowledge that can be extracted from more subtle correlations, which is particularly useful for less popular items.
# Example: There might not be many recipes between some particular type of Mexican pepper and `corn tortillas`, but since that pepper appears with other ingredients similar to a `tortilla`, we would expect it to be similar to `corn tortillas`.
# 
# One solution to this *sparseness* problem (the fact that most pairs of ingredients have little to no co-occurrences) is to use Matrix Factorization. I won't go into the details of the linear algebra behind this technique (I would recommend checking the [matrix factorization wikipedia page](https://en.wikipedia.org/wiki/Matrix_decomposition)), but here goes a simple illustration of how we will use it:
# 
# - First, we create a matrix where rows and columns represent ingredients, and the values are the PMI of a pair of ingredients. (you might also use a binary co-occurrence signal, e.g 1 if there's any recipe with both ingredients, 0 otherwise; or use the raw number of co-occurrences, but PMI makes more sense in our case)
# - We factorize this matrix: You can think of it as "compressing" our matrix from a large but sparse NxN matrix, where N is the number of ingredients, to a smaller but dense NxK matrix, where K is a number that we choose (hereset to 120 as it gave decent results).
# 
# Matrix factorization is helpful because it _generalizes_ the knowledge we have about ingredients, and removes noise and redundancies in the data. The output of this step is a vector representing each ingredient, vectors that we can compare to each other using various similarity metrics. Given that in this case, the most popular ingredients will have larger vectors, we prefer [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) since it is not biased by a vector's norm.

# In[ ]:


from scipy.sparse import csr_matrix
data_df = cooc_df[cooc_df.pmi > 0].copy()
# Since the matrix is symetric, we add the same values for (b,a) as we have for (a,b)
data_df_t = data_df.copy()
data_df.a, data_df.b = data_df.b, data_df.a
data_df = pd.concat([data_df, data_df_t])

rows_idx, row_keys = pd.factorize(data_df.a)
cols_idx, col_keys = pd.factorize(data_df.b)
values = data_df.pmi

matrix = csr_matrix((values, (rows_idx, cols_idx)))
key_to_row = {key: idx for idx, key in enumerate(row_keys)}


# In[ ]:


from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(200)
factors = svd.fit_transform(matrix)


# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity
def most_similar(ingredient, topn=10):
    if ingredient not in key_to_row:
        print("Unknown ingredient.")
    factor = factors[key_to_row[ingredient]]
    cosines = cosine_similarity([factor], factors)[0]
    indices = cosines.argsort()[::-1][:topn + 1]
    keys = [row_keys[idx] for idx in indices if idx != key_to_row[ingredient]]
    return keys, cosines[indices]

def display_most_similar(ingredient, topn=10):
    print("- Most similar to '{}'".format(ingredient))
    for similar_ing, score in zip(*most_similar(ingredient, topn)):
        print("  . {} : {:.2f}".format(similar_ing, score))    


# And tada, we're done! Here are some examples of recommendations generated by our model:

# In[ ]:


display_most_similar('chile powder')


# In[ ]:


display_most_similar('harissa')


# In[ ]:


display_most_similar('rice noodles')


# In[ ]:


display_most_similar('pork')


# In[ ]:


display_most_similar('vanilla')


# In[ ]:


display_most_similar('whipped cream')


# In[ ]:


display_most_similar('buffalo mozarella')

