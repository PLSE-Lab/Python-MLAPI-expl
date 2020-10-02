#!/usr/bin/env python
# coding: utf-8

# # Word2Vec on Instacart products
# ### The goal of this kernel is to try a Word2Vec model on the data of product orders
# ### The orders can act as sentences and product ids can act as words, in this kernel we will see if the model will learn any useful information about the products from the order history of all users, maybe in the future this can be used as input to a classifier that recommends products.
# 
# * Original author's kernel's blog post: http://omarito.me/word2vec-product-recommendations/

# ### Load the needed libraries

# In[ ]:


# !pip install umap-learn
## requires internet connection

## UMAP is typically faster and can be better than PCA or even tsne for dim reduciton ;  https://umap-learn.readthedocs.io/en/latest/


# In[ ]:


import pandas as pd
import numpy as np
import gensim
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Load the Data

# In[ ]:


orders = pd.concat([pd.read_csv("../input/order_products__train.csv"),pd.read_csv("../input/order_products__prior.csv")])
print("orders",orders.shape)
products = pd.read_csv("../input/products.csv").set_index('product_id')
print("products",products.shape)


# ### Turn the product ID to a string
# #### This is necessary because Gensim's Word2Vec expects sentences, so we have to resort to this dirty workaround

# In[ ]:


orders["product_id"] = orders["product_id"].astype(str)
orders.head()


# ### Extract the ordered products in each order

# In[ ]:


# train_products = train_orders.groupby("order_id").apply(lambda order: order['product_id'].tolist())
# prior_products = prior_orders.groupby("order_id").apply(lambda order: order['product_id'].tolist())

# new 
sentences = orders.groupby("order_id").apply(lambda order: order['product_id'].tolist())
# print(sentences.shape)
# sentences.head()


# ### Create the final sentences

# In[ ]:


# sentences = prior_products.append(train_products)
longest = np.max(sentences.apply(len))
print("longest len",longest)
print("mean length",np.mean(sentences.apply(len)))
sentences = sentences.values


# ### Train Word2Vec model
# #### I have modified the window size to be equal to the longest order in our dataset. I've explained why in a blog post that is further explaining this kernel in details
# http://omarito.me/word2vec-product-recommendations/
# 
# * We could/should also consider setting sentenes not just aacording to ORDERS (i.e baskets), but also by USERS

# In[ ]:


model = gensim.models.Word2Vec(sentences, size=60, window=longest, min_count=4, workers=4)


# ### Organize data for visualization

# In[ ]:


vocab = list(model.wv.vocab.keys())


# ### PCA/lower dimensional embed transform the vectors into 2d
# 
# * Could use with multiple steps, e.g. pca and tsne (however, this is wasteful since tsne will refit anyway). 
# * Could try (also) with umap. 
# * another clustering pipeline example https://gist.github.com/stes/92db6023aa3dab5d13e49ece198102c7
# 
# * https://github.com/lmcinnes/umap

# In[ ]:


pca = PCA(n_components=2) # ORIG
# pca = PCA(n_components=50)

# pca = TSNE(n_components=2)
pca.fit(model.wv.syn0)


# ### Some helpers for visualization

# In[ ]:


def get_batch(vocab, model, n_batches=4):
    output = list()
    for i in range(0, n_batches):
        rand_int = np.random.randint(len(vocab), size=1)[0]
        suggestions = model.most_similar(positive=[vocab[rand_int]], topn=5)
        suggest = list()
        for i in suggestions:
            suggest.append(i[0])
        output += suggest
        output.append(vocab[rand_int])
    return output

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    """From Tensorflow's tutorial."""
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  #in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i,:]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
#     plt.savefig(filename)
    plt.show()


# ### Visualize a random sample

# In[ ]:


embeds = []
labels = []
for item in get_batch(vocab, model, n_batches=4):
    embeds.append(model[item])
    labels.append(products.loc[int(item)]['product_name'])
embeds = np.array(embeds)
embeds = pca.fit_transform(embeds)
plot_with_labels(embeds, labels)


# ### Save the model

# In[ ]:


model.save("product2vec.model")

