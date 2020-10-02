#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Here are several more example uses of gensim + glove2word2vec in Kaggle kernels


# In[ ]:


from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors


# In[ ]:


vectors = KeyedVectors.load_word2vec_format("../input/glove_w2v.txt") # import the data file


# In[ ]:


sim = vectors.similarity('dog', 'princess')
distance = vectors.distance("dog","princess")
print("distance ({:.4f}) + similarity ({:.4f}) = 1".format(distance,sim))
# in other words, distance == dissimilarity (hey let's try it)

sim = vectors.similarity('distance', 'dissimilarity')
print("{:.4f}".format(sim))
# rats. need a bigger model.


# In[ ]:


result = vectors.similar_by_word("patriots")
print("{}: {:.4f}".format(*result[0]))

result = vectors.similar_by_word("tacos")
print("{}: {:.4f}".format(*result[0]))


# In[ ]:


sim = vectors.n_similarity(['electronics', 'company'], ['electrical', 'engineer'])
print("{:.4f}".format(sim))

sim = vectors.n_similarity(['dog', 'park'], ['kitty', 'pool'])
print("{:.4f}".format(sim))


# In[ ]:


sent_one = 'I like turtles.'.lower().split()
sent_two = 'I really like turtles.'.lower().split()
distance = vectors.wmdistance(sent_one, sent_two)
print("distance 1:\t{}".format(distance))

sent_one = 'Yes, we have no bananas.'.lower().split()
sent_two = 'We have no bananas today.'.lower().split()
distance = vectors.wmdistance(sent_one, sent_two)
print("distance 2:\t{}".format(distance))

# compared to

sent_one = 'I love you, everything is fine.'.lower().split()
sent_two = 'I hate you, I want a divorce.'.lower().split()
distance = vectors.wmdistance(sent_one, sent_two)
print("distance 3:\t{}".format(distance)) # see? how was I supposed to get that?


# In[ ]:


result = vectors.most_similar(positive=['cowboy', 'truck'], negative=['boots'])
print("cowboy and truck minus boots equals {}".format(*result[0])) # xD

result = vectors.most_similar(positive=['human'], negative=['soul'])
print("humans without souls are {}".format(*result[0])) # :O :O :O

result = vectors.most_similar(positive=['hiking','adventure','wilderness'], negative=['peril','evisceration'])
print("a hiking adventure in the wilderness without peril and evisceration is just {}".format(*result[0]))


# In[ ]:


some_words = ["apples","watermelon","salt","pear"]
print(vectors.doesnt_match(some_words))
# see mom I told you salt doesn't go on those things

