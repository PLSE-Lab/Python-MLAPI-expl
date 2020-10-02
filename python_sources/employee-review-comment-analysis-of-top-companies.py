#!/usr/bin/env python
# coding: utf-8

# # Import Libraries & Data

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv("../input/employee_reviews.csv")


# # EDA

# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.head()


# In[ ]:


df.groupby('company').size()


# # Plots of Company-wise Ratings on various parameters

# In[ ]:


df.groupby(['overall-ratings','company']).size().unstack().plot.bar(subplots=True, rot=0, figsize=(15,20), legend=True)


# In[ ]:


df.groupby(['work-balance-stars','company']).size().unstack().plot.bar(subplots=True, rot=0, figsize=(15,20), legend=True)


# In[ ]:


df.groupby(['culture-values-stars','company']).size().unstack().plot.bar(subplots=True, rot=0, figsize=(15,20), legend=True)


# In[ ]:


df.groupby(['carrer-opportunities-stars','company']).size().unstack().plot.bar(subplots=True, rot=0, figsize=(15,20), legend=True)


# In[ ]:


df.groupby(['comp-benefit-stars','company']).size().unstack().plot.bar(subplots=True, rot=0, figsize=(15,20), legend=True)


# In[ ]:


df.groupby(['senior-mangemnet-stars','company']).size().unstack().plot.bar(subplots=True, rot=0, figsize=(15,20), legend=True)


# # Heatmaps

# In[ ]:


def plot_heatmap(dataCol1, dataCol2, heading):
    grpby = df.groupby([dataCol1, dataCol2]).size()
    pct = grpby.groupby(level=1).apply(lambda x:100 * x / float(x.sum()))
    unstk_data = pct.unstack()
    fig, ax = plt.subplots()
    sns.heatmap(unstk_data, annot=True, linewidths=.5, ax=ax, cmap='YlGn')
    ax.set_title(heading)
    fig.tight_layout()
    plt.show()


# In[ ]:


plot_heatmap('overall-ratings','company', 'Overall-ratings in Companies in %' )


# ## Assuming 4 and above to be very good rating, and less than 4 to be not so good, some observations from the above plot are:
# ### 72% people in FB have rated it 5 and overall 87% people are very happy working in FB. This is followed by Gogle with 85%
# ### NetFlix has got a 55-45% which means people have neutral say about the company
# ### 60% people are happy with Amazon
# ### 72% people are happy with Apple
# ### 68% people are happy with Microsoft

# In[ ]:


plot_heatmap('work-balance-stars','company', 'Work-Life-Balance in Companies in %' )


# ## Assuming 4 and above to be very good rating, and less than 4 to be not so good, some observations from the above plot are for Work/Life Balance
# ### 46.2% people in amazon have a W/L balance
# ### 54% apple 
# ### 70.5% FB
# ### 76.3% Google
# ### 60.6% Microsoft
# ### 50.5% Netflix

# In[ ]:


plot_heatmap('culture-values-stars','company', 'Culture Values in Companies in %' )


# ## In cultural values,
# ### Facebook leads with 89%
# ### Amazon 48%
# ### Apple 60%
# ### GOogle 62%
# ### Microsoft 45%
# ### Netflix 36%

# In[ ]:


plot_heatmap('carrer-opportunities-stars','company', 'Career Opportunities in Companies in %' )


# In[ ]:


plot_heatmap('comp-benefit-stars','company', 'Compensation Benefits in Companies in %' )


# In[ ]:


plot_heatmap('senior-mangemnet-stars','company', 'Senior-Management Ratings in Companies in %' )


# # Summary, Pros, Cons Columns Analysis - Top 10 comments

# In[ ]:


#Define a function to get rid of stopwords present in the messages
from nltk.corpus import stopwords
import string

def message_text_process(mess):
    # Check characters to see if there are punctuations
    no_punctuation = [char for char in mess if char not in string.punctuation]
    # now form the sentence.
    no_punctuation = ''.join(no_punctuation)
    # Now eliminate any stopwords
    return [word for word in no_punctuation.split() if word.lower() not in stopwords.words('english')]    


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

vectorizer = TfidfVectorizer(analyzer=message_text_process)


# In[ ]:


n_top_words = 20
lda = LatentDirichletAllocation()

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()
    
def find_top_comments(corpus):
    tfidf_transformer = vectorizer.fit_transform(corpus)
    tf_feature_names = vectorizer.get_feature_names()    
    lda.fit_transform(tfidf_transformer)
    print_top_words(lda, tf_feature_names, n_top_words)


# In[ ]:


find_top_comments(df['pros'])


# In[ ]:


find_top_comments(df['cons'])


# In[ ]:


find_top_comments(df[df.summary.notnull()].summary)


# In[ ]:


df[df.summary.notnull()].summary.head(25)


# # WordClouds of Summary / Pros / Cons Columns

# In[ ]:


from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 

def generate_word_cloud(text):  
    comment_words = ' '
    stopwords = set(STOPWORDS) 

    # iterate through the csv file 
    for val in text: 

        # typecaste each val to string 
        val = str(val) 

        # split the value 
        tokens = val.split() 

        # Converts each token into lowercase 
        for i in range(len(tokens)): 
            tokens[i] = tokens[i].lower() 

        for words in tokens: 
            comment_words = comment_words + words + ' '
        
    wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
    
    # plot the WordCloud image                        
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 

    plt.show() 


# In[ ]:


grouped = df.groupby('company')

for name,group in grouped:
    print (name)
    generate_word_cloud(group['summary'])
    print('cons')
    generate_word_cloud(group['cons'])
    print('pros')
    generate_word_cloud(group['pros'])
    print('Advice to Management')
    generate_word_cloud(group['advice-to-mgmt'])
    

