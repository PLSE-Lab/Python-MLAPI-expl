#!/usr/bin/env python
# coding: utf-8

# # Background
# 
# I am not real happy with the sentence ranking in my previous [notebook](https://www.kaggle.com/dirktheeng/anserini-bert-squad-for-context-corpus-search).  It does pull up relevant answers, but it seems to me that the best ones don't always sit at the top of the list.  I want to try some semantic (meaning) based similarity matching with the thought that the most relevent answer will have the closest meaning to the question being asked.  I came across Google's Universal Sentance Encoder whigh is a derrivative of BERT.  This means that it does encoding comparison based on meaning.  I also came across a few good examples of visualizing the similarity.  The best one I've found can be viewed [here](https://towardsdatascience.com/use-cases-of-googles-universal-sentence-encoder-in-production-dd5aaab4fc15).
# 
# If this works I will build this into my previous work (assuming I can make it all fit in the VM).
# 
# # Set up Tensorflow

# In[ ]:


import tensorflow as tf
import tensorflow_hub as hub


# # Go get the model

# In[ ]:


#download the model to local so it can be used again and again
#!mkdir ../sentence_wise_email/module/module_useT
get_ipython().system('mkdir /kaggle/working/sentence_wise_email/')
get_ipython().system('mkdir /kaggle/working/sentence_wise_email/module/')
get_ipython().system('mkdir /kaggle/working/sentence_wise_email/module/module_useT')
# Download the module, and uncompress it to the destination folder. 
get_ipython().system('curl -L "https://tfhub.dev/google/universal-sentence-encoder-large/3?tf-hub-format=compressed" | tar -zxvC /kaggle/working//sentence_wise_email/module/module_useT')


# In[ ]:


tf.compat.v1.disable_eager_execution()
embed = hub.Module("/kaggle/working/sentence_wise_email/module/module_useT")


# In[ ]:


import numpy as np
word = "Elephant"
sentence = "I am a sentence for which I would like to get its embedding."
paragraph = (
    "Universal Sentence Encoder embeddings also support short paragraphs. "
    "There is no hard limit on how long the paragraph is. Roughly, the longer "
    "the more 'diluted' the embedding will be.")
messages = [word, sentence, paragraph]
# Reduce logging output.
#tf.compat.v1.logging.set_verbosity(tf.logging.ERROR)
with tf.compat.v1.Session() as session:
    session.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])
    message_embeddings = session.run(embed(messages))
for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
        print("Message: {}".format(messages[i]))
        print("Embedding size: {}".format(len(message_embedding)))
        message_embedding_snippet = ", ".join((str(x) for x in        message_embedding[:3]))
        print("Embedding[{},...]\n".
                   format(message_embedding_snippet))


# This is the money funcction.  This function converts a frase into a group of 512 semantic elements in a vector.

# In[ ]:


#Function so that one session can be called multiple times. 
#Useful while multiple calls need to be done for embedding. 
def embed_useT(module):
    with tf.Graph().as_default():
        sentences = tf.compat.v1.placeholder(tf.string)
        embed = hub.Module(module)
        embeddings = embed(sentences)
        session = tf.compat.v1.train.MonitoredSession()
    return lambda x: session.run(embeddings, {sentences: x})
embed_fn = embed_useT('/kaggle/working/sentence_wise_email/module/module_useT')
messages = [
    "we are sorry for the inconvenience",
    "we are sorry for the delay",
    "we regret for your inconvenience",
    "we don't deliver to baner region in pune",
    "we will get you the best possible rate"
]
embed_fn(messages)


# taking the dot product produces a similarity matrix

# In[ ]:


encoding_matrix = embed_fn(messages)
np.inner(encoding_matrix, encoding_matrix)


# I found this heat map plot code [here](https://www.learnopencv.com/universal-sentence-encoder/)

# In[ ]:


def plot_similarity(labels, features):
    corr = np.inner(features, features)
    sns.set(font_scale=1.2)
    g = sns.heatmap(corr,        xticklabels=labels,        yticklabels=labels,        vmin=0,        vmax=1,        cmap="YlOrRd")
    g.set_xticklabels(labels, rotation=90)
    g.set_title("Semantic Textual Similarity")
    plt.tight_layout()
    plt.savefig("Avenger-semantic-similarity.png")
    plt.show()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

plot_similarity(messages, encoding_matrix)


# In[ ]:


messages = [
    "How does weather, heat, and humidity affect the tramsmission of 2019-nCoV",
    "conclusions : a wide range of continuous warm and dry weather is conducive to the survival of 2019-ncov .",
    "formalin fixation and heating samples to 56oc, as used in routine tissue processing, were found to inactivate several coronaviruses and it is believed that 2019-ncov would be similarly affected .",
    "objective to investigate the impact of temperature and absolute humidity on the coronavirus disease 2019 (covid-19) outbreak .",
    "taken chinese cities as a discovery dataset, it was suggested that temperature, wind speed, and relative humidity combined together could best predict the epidemic situation .",
    "in the next years the engagement of the health sector would be working to develop prevention and adaptation programs in order to reduce the costs and burden of climate change."
]
encoding_matrix = embed_fn(messages)
plot_similarity(messages, encoding_matrix)


# That's positive!!! it's not too slow or too resource intence and it seems to do a better job of ranking the phrases relative to the question!!  I think this is a winner winner chicken dinner!!
