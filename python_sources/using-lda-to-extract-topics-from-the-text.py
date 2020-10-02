#!/usr/bin/env python
# coding: utf-8

# In this kernel we'll try some Latent Dirichlet Allocation to automatically extract the topics that characterize our text data :). This is my first one ever on Kaggle!

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ## Importing Our Data

# In[ ]:


train_variants = pd.read_csv("../input/training_variants")
test_variants = pd.read_csv("../input/test_variants")
train_text = pd.read_csv("../input/training_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
test_text = pd.read_csv("../input/test_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
print('Unique Classes: ', len(train_variants.Class.unique()))
train_text.head()


# Let's take a look into some of the shorter text entries vs. some of the larger text entries to see if we can gather some insights

# In[ ]:


train_text.loc[:, 'Text Count'] = train_text['Text'].apply(lambda x: len(x.split()))


# Now we can just clean up and eliminate the rows that have no text

# In[ ]:


train_text = train_text[train_text['Text Count'] != 1]
train_text_sorted = train_text.sort_values('Text Count', ascending=0)


# ## Exploring the Actual Text
# Let's take a look at the shortest and longest texts in our data

# In[ ]:


train_text_sorted.tail()


# In[ ]:


train_text_sorted['Text'][0]


# Very, very technical. Let's look at major patterns across classes in the text. Let's take a look at the most common words and gene variants in one of the classes.

# In[ ]:


from wordcloud import WordCloud
train_full = train_text.merge(train_variants, how="inner", left_on="ID", right_on="ID")
cloud = WordCloud(width=1440, height=1080).generate(" ".join(train_full[train_full.Class == 3]['Text']))
plt.figure(figsize=(20, 15))
plt.imshow(cloud)
plt.axis('off')
ax = plt.axes()
ax.set_title('Class 3 Text Word Cloud')


# In[ ]:


data = train_variants[train_variants['Class'] == 3].groupby('Gene')["ID"].count().reset_index()
sns.barplot(x="Gene", y="ID", data=data.sort_values('ID', ascending=False)[:7])
ax = plt.axes()
ax.set_title('Class 3 Top Gene Variants')


# ## Insights & Topic Modeling
# 
# 
# There are so many keywords that straight up tell us what the class is about. The most important insight of the word cloud, however, is the importance of **bigrams** in our text. "Amino Acid", "Homologous Recombination", "Breast Cancer" are only a few examples of many. My hypothesis is that we'll find terms like these extremely prevalent in technical literature such as our text.
# 
# 
# TF-IDF considering bigrams could give us a pretty good score without any other optimizations, but let's take a look at some more sophisticated topic modeling with **Latent Dirichlet Allocation** to see if we can algorithmically find these topics without having to do this inspection for every class.

# In[ ]:


from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components=9, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)


# Before we actually fit our topic model, let's clean up some of our text by getting the raw term counts from our text. Scikit's Count Vectorizer will also take into account stop words to remove some of the noise in our data.
# 
# 
# We plug in a max_features parameter of 50 given that 100 gave us way too many useless ones, but this can definitely be lowered.

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
n_features = 50
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=50,
                                stop_words='english')
tf = tf_vectorizer.fit_transform(train_full['Text'])


# In[ ]:


tf_feature_names = tf_vectorizer.get_feature_names()
tf_feature_names


# ## Fitting Our LDA Model

# In[ ]:


lda.fit(tf)


# In[ ]:


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


# ## Major Topics in Our Text!
# 
# We do get some fairly differentiated topics in our text data, where we can see **class 3** still related to **BRCA1** variants and breast cancer. 
# 
# 
# Some other classes are more concerned with **exons** & **kinases** while others are more focused on the p53 protein and **phosphorylation** pathways. Keep this in mind while analyzing each class and consider the general tone that the text in each class uses to explain a patient's gene variants. 
# 
# 
# Obviously, we can definitely improve this to achieve better separation between the topics, and would love to hear from you in the comments. This was my very first kernel and looking forward to hearing any feedback!

# In[ ]:


print_top_words(lda, tf_feature_names, 10)


# In[ ]:




