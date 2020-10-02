#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Importing libraries

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
#To track function execution
from tqdm import tqdm
from bs4 import BeautifulSoup

#Libraries for Sentimental analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#Libraries for visualization
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

#Libraries for ML
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc


# In[ ]:


# Reading dataframe
df = pd.read_csv("../input/ds5230usml-project/Reviews.csv")


# In[ ]:


#Looking at top 5 rows
df.head(5)


# Not that we have 10 features and 568454 data points. There are some missing values in 'PROFILENAME' & 'SUMMARY' column. 

# In[ ]:


# Checking for null values in the dataframe.
df.isnull().sum()


# In[ ]:


# Inspecting entries with Null value in profileName column 
df[df["ProfileName"].isnull()]


# In[ ]:


# Dropping Null values
df.dropna(inplace=True)


# In[ ]:


# Checking if null value exist again
df.isnull().sum()


# In[ ]:


# Checking the columns of the reviews.
df.columns


# In[ ]:


#Checking the shape of the dataframe.
df.shape


# In[ ]:


# Checking for the info of the dataframe.
df.info()


# In[ ]:


# Statistical analysis of the dataframe.
df.describe()


# In[ ]:


# Checking number of reviews for each score.
df["Score"].value_counts()


# # **Exploratory Data Analysis**

# Note that more than 75% of our data is belonging to positive class(Score=4,5), i.e. we have imbalanced dataset.

# In[ ]:


total = df["Score"].count()
print(total)


# In[ ]:


percent_plot = pd.DataFrame({"Total":df["Score"].value_counts()})
percent_plot.reset_index(inplace = True)
percent_plot.rename(columns={"index":"Rating"},inplace=True)


# In[ ]:


percent_plot


# Below is the plot of number of ratings each score has received.

# In[ ]:


sns.barplot(x="Rating",y="Total", data=percent_plot)


# In[ ]:


percent_plot["Percent"] = percent_plot["Total"].apply(lambda x: (x/total)*100)


# In[ ]:


#percent_plot.drop(['percent'],axis=1, inplace = True)


# We can see that 5-star reviews constitute a large proportion (63.88%) of all reviews. The next most prevalent rating is 4-stars(14.18%), followed by 1-star (9.19%), 3-star (7.50%), and finally 2-star reviews (5.23%).

# In[ ]:


percent_plot


# Below is the plot of Ratings and its percentage.

# In[ ]:


sns.barplot(x="Rating", y="Percent", data = percent_plot)


# # **Text Exploration**

# In[ ]:


df.columns


# In[ ]:


df["word_count"] = df["Text"].apply(lambda x: len(str(x).split(" ")))
df[["Text","word_count"]].head()


# In[ ]:


# Checking the statistics of word count to check for range and average number of the words in each article.
df["word_count"].describe()


# In[ ]:


#Checking for top 20 most repeated words - Gives insights on data specific stop words.

common_words = pd.Series(' '.join(df["Text"]).split()).value_counts()
common_words[:20]


# In[ ]:


# Checking 20 most uncommon words
common_words[-20:]


# # **Text Preprocessing**

# In[ ]:


# Removing Stopwords
stop_words = set(stopwords.words("english"))

# Adding common words from our document to stop_words

add_words = ["the","I","and","a","to","of","is","it","for","in","this","that","my","with",     
"have",     
"but",      
"are",      
"was",      
"not",      
"you"]

stop_words = stop_words.union(add_words)


# In[ ]:


#Below Function is to clean the text and prepare it for the next phase.

from tqdm import tqdm
corpus = []

def clean_content(df):
    cleaned_content = []
    
    for sent in tqdm(df["Text"]):
        
        #Removing HTML comtent
        review_content = BeautifulSoup(sent).get_text()
        
        #Removing non-alphabetic charecters
        review_content = re.sub("[^a-zA-Z]"," ", review_content)
        
        #Tokenize the sentences
        words = word_tokenize(review_content.lower())
        
        #Removing the stop words
        sto_words_removed = [word for word in words if not word in stop_words]
        sto_words_removed = " ".join(sto_words_removed)
        corpus.append(sto_words_removed)
        cleaned_content.append(sto_words_removed)
        
    return (cleaned_content)


# In[ ]:


df["cleaned_text"] = clean_content(df)


# In[ ]:


df.head()


# # **Data Exploration**

# Building a wordcloud to visualize most frequently used words after Text pre-processing stage  

# In[ ]:


wordcloud = WordCloud(
                    background_color = "white",
                    stopwords = stop_words,
                    max_words = 100,
                    max_font_size = 50).generate(str(corpus))


# In[ ]:


# Displaying the word cloud
print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
#fig.savefig("word1.png", dpi=900)


# # **Sentimental Analysis**

# Performed Sentiment Analysis to classify the Reviews into Positive or negative reviews.
# 
# Input:
# 
# Text column of the dataFrame. 
# 
# Output:
# 
# Sentimental score report card with percentage of negative, positive, neutral and compound sentiment.
# Using this score report card, classified the sentence into possitive or negative sentence.
# 0 - Negative Sentence
# 1 - Positive Sentence

# In[ ]:


# Initializing the sentimental Intenity Analyzer
sid = SentimentIntensityAnalyzer()


# In[ ]:



# checking the polarity scores for first 5 articles
for i in range(0,5):
    print(sid.polarity_scores(df.loc[i]["Text"]))


# In[ ]:


#df["Text"][2]


# In[ ]:


df["sentimental_scores"] = df["Text"].apply(lambda x: sid.polarity_scores(x))


# In[ ]:


df["compound_sentiment"] = df["sentimental_scores"].apply(lambda score_dict: score_dict["compound"]) 


# In[ ]:


df.head()


# In[ ]:


df["sentiment"] = df["compound_sentiment"].apply(lambda x: 1 if x >= 0 else 0)
df.head()


# We can observe that the dataset mostly consists of positive sentiments which is shown in the below graph.

# In[ ]:


sns.countplot(x="sentiment", order = [1,0], data=df, palette='RdBu')
plt.xlabel("Sentiment")
plt.show()

Try plotting word cloud for positive sentiment and negative sentiment 
# # Working on only 4000 entries to get visualization. 

# In[ ]:


pos = df[df["sentiment"] == 1].sample(n = 2000)
neg = df[df["sentiment"] == 0].sample(n = 2000)
df_4000 = pd.concat([pos,neg])


# In[ ]:


senti_4000 = df_4000["sentiment"]


# In[ ]:


print(df_4000.shape)
print(senti_4000.shape)


# # Tf-Idf

# In[ ]:


tf_idf_vect = TfidfVectorizer(ngram_range = (1,2))
tf_idf = tf_idf_vect.fit_transform(df_4000['cleaned_text'].values)


# In[ ]:


# Performing standard scaling
from sklearn.preprocessing import StandardScaler
std = StandardScaler(with_mean = False)
scaled_data = std.fit_transform(tf_idf) 


# In[ ]:


# scaled_data = scaled_data.todense()


# In[ ]:


scaled_data.shape


# # T-SNE plot for the above Dataset

# In[ ]:


#from sklearn.mainfold import TSNE
from sklearn.manifold import TSNE
model = TSNE(n_components = 2, perplexity = 50)
tsne_data = model.fit_transform(scaled_data)


# In[ ]:


tsne_data = np.vstack((tsne_data.T, senti_4000)).T
tsne_df = pd.DataFrame(data = tsne_data, columns = ("dim1", "dim2", "score"))


# In[ ]:


sns.FacetGrid(tsne_df, hue = "score", size = 6).map(plt.scatter, "dim1","dim2").add_legend()
plt.title("TSNE for Pos and Nev reviews")
plt.show()

