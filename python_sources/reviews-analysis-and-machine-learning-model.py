#!/usr/bin/env python
# coding: utf-8

# # Alexa Review's Analysis and Machine Learning Model.

#  ### Introduction
#  The dataset used in this notebook contains Alex reviews. This dataset conatins 3150 reviews about the product, rating and feedback. Rating range from 1 to 5. Feedback column contains binary number 0/1, 1 for positive review and 0 for negative review. 

# ___

# ### Analysis Section

# ___

# The first part is importing all the required libraries for this section.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = 13, 10


# In[ ]:


dataframe = pd.read_csv('../input/amazon_alexa.tsv', sep = '\t')


# In[ ]:


sns.countplot(dataframe['feedback'] )


# In[ ]:


print("Percentage of negative reviews: ", (len(dataframe[dataframe['feedback'] == 0]) * 100)/len(dataframe))
print("Percentage of Positive reviews: ", (len(dataframe[dataframe['feedback'] == 1]) * 100)/len(dataframe))


# As from the plot above and percentage of positive and negative review, it is evident that Alex is very popular, with very less percentage of bad reviews.

# ___

# In[ ]:


sns.countplot(dataframe['rating'])


# The plot above shoes that most people rated Alex 5 stars. 4 star rating is pretty significant, Hence, it performed great in the market.

# ___

# In[ ]:


ax = sns.countplot(x = 'variation', data = dataframe,palette="Blues_d", order = dataframe['variation'].value_counts().index)
ax.set_xticklabels(ax.get_xticklabels(),rotation=30)


# The above plot shows that, "Black Dot" variant is high in demand and lowest being the "Walnut Finish". The graph is arranged in the descending order, and is pretty self-explanatory.

# In[ ]:


ax = sns.barplot(x = 'variation', y = 'rating', data = dataframe)
ax.set_xticklabels(ax.get_xticklabels(),rotation=30)


# The graph shows "walnut finish" variant with highest average rating, this is because the number of walnut finish variant is very less as compared with "Black dot" variant. But the overall rating ranges from 4.2 to 4.9. 

# In[ ]:


dataframe.groupby('variation').mean().reset_index()


# This table shoes the exact rating of each variant.

# ___

# In[ ]:


def wordclouds(x, label):
    from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
    text = []
    dataframe_pos = x['feedback'] == label
    for i in range(0, len(x)):
        review = dataframe['verified_reviews'][i]
        text.append(review)
    text = " ".join(text for text in text)

    stopwords = set(STOPWORDS)
    stopwords.remove('not')
    wordcloud = WordCloud(stopwords=stopwords, background_color="black", max_font_size=100).generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    


# In[ ]:


wordclouds(dataframe, 1)


# Above plot is a wordcloud of most frequent words used in the positive reviews.
# Plot for negative reviews can be made using the function, "wordclouds()". Pass dataframe and label as the argument.

# ___

# ### Machine Learning Section.

# Here, I have used NLTK library.

# In[ ]:


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# In[ ]:


corpus = []
stop_words = set(stopwords.words('english'))
stop_words.remove('not')
for i in range(0, len(dataframe)):
    review = re.sub('[^a-zA-Z]', ' ', dataframe['verified_reviews'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in stop_words]
    review = ' '.join(review)
    corpus.append(review)


# The above code block cleans the text and append it to corpus list. This code block converts the reviews to lower case, only keeps alphabets and remove special characters or any other characters. Then the porter stemmer, converts each word to its root word, example, fighting is converted to fight.

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2500)
X = cv.fit_transform(corpus).toarray()
y = dataframe.iloc[:, 4].values


# For this model I chose the bag of word model.

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn import tree
classifier = tree.DecisionTreeClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


# The machine learning models used is decision tree classifier as it gives the highest accuracy.

# In[ ]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
print("The F1 score is: ", f1_score(y_test, y_pred, average="macro")*100)
print("The precision score is: ", precision_score(y_test, y_pred, average="macro")*(100))
print("The recall score is: ", recall_score(y_test, y_pred, average="macro")*100) 
print("The accuracy score is: ", accuracy_score(y_test, y_pred)*100)


# This model has the highest accuracy, f1 score, precision score and recall score.

# ___

# Hence, Alexa is well liked and made its own place in the market. It would be fun to compare the reviews with google home.

# I hope you like it. If you have any question or suggestion, just drop them in the the comment section. Please like the notebook. Thank You.
