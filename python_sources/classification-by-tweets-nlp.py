#!/usr/bin/env python
# coding: utf-8

# ## Hello, in this Notebook I will work on Natural Language Processing.
# ## I will try classify Male Tweets from Female Tweets

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# ### EDA

# In[ ]:


df = pd.read_csv("../input/gender-classifier-DFE-791531.csv",encoding="latin1")


# In[ ]:


df.head()     # There might be some very useful columns but I will only work on text. So I am going to create simpler dataset


# ### Data PreProcessing

# In[ ]:


work_data = pd.DataFrame()
work_data["tweet"] = df.description
work_data["gender"] = df.gender


# In[ ]:


work_data.head()            # this is one is much easy to read and work with.


# In[ ]:


get_female = work_data["gender"] == "female"
get_male = work_data["gender"] == "male"
get_brand = work_data["gender"] == "brand"


# In[ ]:


female_rows = work_data[get_female]
male_rows = work_data[get_male]
brand_rows = work_data[get_brand]


# In[ ]:


print("total female tweets: ",female_rows.tweet.count())
print("total male tweets:   ",male_rows.tweet.count())
print("total brand tweets:  ",brand_rows.tweet.count())           # they are evenly distributed. Which is good


# ### Labelling

# In[ ]:


female_rows.gender = 0     # female
male_rows.gender = 1       # male
brand_rows.gender = 2      # brand


# ### Concating the brand and person rows

# In[ ]:


frames = [female_rows, male_rows, brand_rows]
data = pd.concat(frames,ignore_index=True)


# In[ ]:


data.tail()


# In[ ]:


data.info()   # I dont want the null values to became most used words in my bag of words. I will drop them.


# In[ ]:


data.dropna(inplace=True)


# In[ ]:


data.info()   # now we have much more clean and useful dataset


# ## Natural Language Processing

# In[ ]:


import re
import nltk as nlp


# In[ ]:


from nltk.corpus import stopwords
lemma = nlp.WordNetLemmatizer()  


# In[ ]:


tweets_list = []            # empty list
for each in data.tweet:
    each = re.sub("[^a-zA-Z]"," ", str(each))                                        # regex to clean unnecesarry chars
    each = each.lower()                                                              # lowercase all
    each = nlp.word_tokenize(each)                                                   # split all by tokenizing
    each = [word for word in each if not word in set(stopwords.words("english"))]    # delete stop words from your array
    each = [lemma.lemmatize(word) for word in each]                                  # lemmatize "memories" -> "memory"
    each = " ".join(each)                                                            # make them one string again
    tweets_list.append(each)                                                         # put them into big array


# In[ ]:


print("Original version: ", data.tweet.iloc[2174])
print("New version:      ", tweets_list[2174])    # no unnecesary words or symbols


# ## Bag of Words

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer

max_features = 600

count_vectorizer = CountVectorizer(max_features=max_features,stop_words = "english")

sparce_matrix = count_vectorizer.fit_transform(tweets_list).toarray()
words = count_vectorizer.get_feature_names()


# In[ ]:


print("Most used 600 words on all tweets (alphabetically first 100) :", words[:100])


# ## Train & Test Split

# In[ ]:


y = data.gender.values
x = sparce_matrix


# In[ ]:


from sklearn.model_selection import train_test_split 

train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.2)


# ## Machine Learning Model

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators = 100)

rfc.fit(train_x,train_y)


# In[ ]:


rfc.score(test_x,test_y)


# In[ ]:


y_head_ml = rfc.predict(test_x)


# In[ ]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_y,y_head_ml)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(cm,cbar=False,annot=True,cmap="Blues",fmt="d")
plt.show()


# ## ** Conclusion **
# 
# ### Brand is easy to seperate but genders are relatively more complex

# In[ ]:




