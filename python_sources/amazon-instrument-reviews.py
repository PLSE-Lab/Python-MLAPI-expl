#!/usr/bin/env python
# coding: utf-8

# <hr>
# <font size="+3" color="blue"><b><center>Amazon Instrument Reviews</center></b></font><br>
# 
# <hr>
# <img src='https://images.ctfassets.net/3s5io6mnxfqz/2EJtrZGZAfgaHRt9NKY91j/7c930d26d0d0c6b9a0ffcb79a96fefa2/AdobeStock_96567909.jpeg?w=900&fm=jpg&fl=progressive' height='width: 450px'>
# <hr>

# <font size="+2" color="blue"><b>Introduction</b></font><br>
# > This is my first ever public notebook not using competition datasets. Here I create a notebook aimed to predict the **sentiment of reviews** for **Amazon Musical Instruments** using the given **[dataset](https://www.kaggle.com/eswarchandt/amazon-music-reviews).** In this notebook, I will perform a **Exploratory Data Analysis** where I *preprocess, visualize, and test the data.* Lots of **visualizations** are used! I have achieved a **96% accuracy** in predicting the sentiment of reviews for **Amazon Musical Instruments.**
# 
# <font size="+2" color="blue"><b>Table of Contents</b></font><br>
# * [Libraries](#1)
# * [Formatting Dataset](#2)
# * [Data Visualization](#3)
# * [Data Preprocessing for Modelling](#4)
# * [Modelling](#5)
#     
# <hr>

# <font size="+3" color="blue"><b>Libraries</b></font><br><a id='1'></a>
# <hr>

# In[ ]:


import numpy as np 
import pandas as pd 

# Data Visualization
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
init_notebook_mode(connected = True)
import plotly.figure_factory as ff
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# NLP
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.stem import LancasterStemmer,WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import NearMiss
import re, string, unicodedata


plt.style.use('fivethirtyeight')
sns.set_style("darkgrid")

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

import warnings
warnings.filterwarnings('ignore')


# <font size="+3" color="blue"><b>Formatting Dataset</b></font><br><a id='2'></a>
# <hr>

# <font size='+0.95' color="black"><b>Reading Data</b></font><br>

# In[ ]:


data = pd.read_csv('../input/amazon-music-reviews/Musical_instruments_reviews.csv')
data.head()


# <font size='+0.95' color="black"><b>Dropping columns, rows, and renaming</b></font><br>

# In[ ]:


# Drop apparent useless columns
data.drop(['reviewerID', 'reviewerName', 'helpful'], axis=1, inplace=True)

# Rename
data = data.rename(columns={"asin": "product_id", 'overall':'rating', 'unixReviewTime':'unix_time'})

# Drop rows with missing values
data.isnull().sum().sort_values(ascending=False)
data = data.dropna()

data.head()


# <font size='+0.95' color="black"><b>Creating Features: review/year/sentiment/length</b></font><br>

# In[ ]:


# Sentiment function for creating "sentiment" feature
def sentiment(rating):
    if(int(rating) == 1 or int(rating) == 2 or int(rating) == 3):
        return 0
    else: 
        return 1
    
# Length function for creating "length" feature
def length(text):
    return len(text)

#########################################################

# Creating feature "review"
data['review'] = data['reviewText'] + ' ' + data['summary']

# Creating feature "year"
data['reviewTime'] = pd.to_datetime(data ['reviewTime'])
data['year'] = data['reviewTime'].dt.year

# Creating feature "sentiment"
data['sentiment'] = data['rating'].apply(sentiment)

# Creating feature "length"
data['length'] = data['review'].apply(length)

data.drop(['reviewText', 'summary', 'reviewTime'], axis=1, inplace=True)

data.head()


# In[ ]:


desc = ff.create_table(data.describe())
py.iplot(desc)


# <font size="+2" color="black"><b>Description of the Data</b></font><br>
# - **product_id:** The ID of the product sold
# - **rating:** The rating of the product sold
# - **unix_time:** Different format of describing time
# - **review:** The text content of the review
# - **year:** The year of the review made
# - **sentiment:** The sentiment of the review (1 for positive, 0 for negative)
# - **length:** The length of the review

# <font size="+3" color="blue"><b>Data Visualization</b></font><br><a id='3'></a>
# <hr>

# In[ ]:


sentiment = data['sentiment'].value_counts()

labels = sentiment.index
size = sentiment.values

colors = ['green', 'red']

sentiments = go.Pie(labels = labels,
                         values = size,
                         marker = dict(colors = colors),
                         name = 'Sentiments Piechart', hole = 0.3)

df = [sentiments]

layout = go.Layout(
           title = 'Sentiments for Amazon Musical Instruments')

fig = go.Figure(data = df,
                 layout = layout)

py.iplot(fig)


# > This piechart describes the sentiment of the reviews:
# - **87.9%** of people had a **positive sentiment**
# - **12.1%** of people had a **negative sentiment**
# - We can assume from this visualization that the reviews for the instruments are **pretty good**
# - We will be **predicting this feature** during **modelling**

# In[ ]:


plt.rcParams['figure.figsize'] = (15, 6)
plt.style.use('fivethirtyeight')
sns.countplot(data['sentiment'], data=data, palette='spring')
plt.show()


# > There are clearly more **positive sentiment values** than **negative sentiment values** 

# In[ ]:


ratings = data['rating'].value_counts()

labels = ratings.index
size = ratings.values

colors = ['green', 'lightgreen', 'gold', 'crimson', 'red']

rating = go.Pie(labels = labels,
                         values = size,
                         marker = dict(colors = colors),
                         name = 'Ratings Piechart', hole = 0.3)

df = [rating]

layout = go.Layout(
           title = 'Percentage Ratings for Amazon Musical Instruments')

fig = go.Figure(data = df,
                 layout = layout)

py.iplot(fig)


# > The ratings are pretty good. From this piechart we can state:
# - **67.6%** of people gave a rating of **5**
# - **20.3%** of people gave a rating of **4**
# - This means **87.9%** of people gave a **POSITIVE RATING**
# - **7.53%** of people gave a rating of **3**, which is an **AVERAGE RATING**
# - **2.44%** of people gave a rating of **2**
# - **2.12%** of people gave a rating of **1**
# - This means **4.56%** gave a **NEGATIVE RATING**

# In[ ]:


plt.rcParams['figure.figsize'] = (15, 6)
plt.style.use('fivethirtyeight')
sns.countplot(data['rating'], data=data, palette='viridis')
plt.title('Rating Frequency for Amazon Musical Instruments')
plt.show()


# > There are clearly much more **ratings of 5 and 4** than **ratings of 3, 2, and 1**. This states that this **product is probably good.**

# In[ ]:


color = plt.cm.plasma(np.linspace(0, 1, 15))
data['product_id'].value_counts()[:20].plot.bar(color = color, figsize = (15, 9))
plt.title('Most Common Product Ids - Top 20', fontsize = 20)
plt.xlabel('Amazon Instruments (Product Id)')
plt.ylabel('Count')
plt.show()


# > We just plotted the **first 20th most common product ids:**
# - Product_id named **'B003VWJ2K8'** has over **160** sold

# In[ ]:


color = plt.cm.hot(np.linspace(0, 1, 15))
data['product_id'].value_counts()[-20:].plot.bar(color = color, figsize = (15, 9))
plt.title('Least Common Product Ids - Bottom 20', fontsize = 20)
plt.xlabel('Amazon Instruments (Product Id)')
plt.ylabel('Count')
plt.show()


# > We just plotted the **last 20th least common product ids:**
# - The **last 20th least common product ids** each have **5 sold**

# In[ ]:


date = data['year'].value_counts()

labels = date.index
values = date.values

colors = ['yellow', 'gold', 'lightskyblue', 'lightcoral', 'pink', 'cyan']

years = go.Pie(labels = labels,
                         values = values,
                         marker = dict(colors = colors),
                         name = 'Date Piechart', hole = 0.3)

df = [years]

layout = go.Layout(
           title = 'Year Percentage Distribution for Amazon Musical Instruments Reviews')

fig = go.Figure(data = df,
                 layout = layout)

py.iplot(fig)


# > The ratings are pretty good. From this piechart we can state:
# - Reviews range from years **2004 - 2014**
# - **26.1%** of reviews are from **2014**
# - **39.5%** of reviews are from **2013**
# - **18.9%** of reviews are from **2012**
# - **9.81%** of reviews are from **2011**
# - **3.41%** of reviews are from **2010**
# - **1.24%** of reviews are from **2009**
# - **0.614%** of reviews are from **2008**
# - **0.215%** of reviews are from **2007**
# - **0.0975%** of reviews are from **2006**
# - **0.039%** of reviews are from **2005**
# - **0.0683%** of reviews are from **2004**
# - A clear positive trend in terms of **popularity**

# In[ ]:


fig = px.scatter(data, x="unix_time", y="rating", color='sentiment', marginal_x="histogram")
fig


# > The ratings are pretty good. From this piechart we can state:
# - Reviews range from around **1.09B to 1.41B in unix time**
# - Reviews are **much more occurrent later in the years than the reviews in the earlier years**
# - **Yellow dots** represent values that have **positive sentiments**
# - **Blue dots** represent values that have **negative sentiments**

# In[ ]:


# Credit to https://www.kaggle.com/roshansharma/amazon-alexa-reviews

cv = CountVectorizer(stop_words = 'english')
words = cv.fit_transform(data.review)
sum_words = words.sum(axis=0)


words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])

data.head()

color = plt.cm.spring(np.linspace(0, 1, 20))
frequency.head(20).plot(x='word', y='freq', kind='bar', figsize=(15, 6), color = color)
plt.title("Most Occuring Words - Top 20")
plt.show()


# > Bar chart for the **top 20 most occuring words**:
# - **'Guitar'** is the **most common word**, this makes sense because a **guitar is a musical instrument**
# - **Words such as 'great', 'good', 'like' and any word that is positive** makes sense in being in the most frequently used words in reviews
# - Words that imply a **negative connotation** such as 'bad' **don't exist in the most commonly used words** because most reviews have a **positive sentiment**
# - These words **fit in the context** of what a review of a musical instrument should be

# In[ ]:


color = plt.cm.spring(np.linspace(0, 1, 20))
frequency.tail(20).plot(x='word', y='freq', kind='bar', figsize=(15, 6), color = color)
plt.title("Least Occuring Words - Bottom 20")
plt.show()


# > Bar chart for the **least 20 most occuring words**:
# - Words are **complex** and used **infrequently**
# - Some words are completely **random** and **unrelated**
# - They each have an occurrence of **one**

# In[ ]:


words_freq

wordcloud = WordCloud(background_color = 'black', width = 1000, height = 1000).generate_from_frequencies(dict(words_freq))

plt.figure(figsize=(10, 10))
plt.axis('off')
plt.imshow(wordcloud)
plt.title("Most Common words", fontsize = 20)
plt.show()


# > Just a Simple **"For Fun"** WordCloud for the **most common words!**

# In[ ]:


plt.rcParams['figure.figsize'] = (12, 7)

sns.stripplot(data['rating'], data['length'], palette = 'Reds')
plt.title("Rating vs Length - Plot 1")
plt.show()


# In[ ]:


plt.rcParams['figure.figsize'] = (15, 7)
sns.violinplot(data['rating'], data['length'], palette = 'Reds')
plt.title('Rating vs Length - Plot 2', fontsize = 20)
plt.show()


# In[ ]:


plt.rcParams['figure.figsize'] = (15, 9)
plt.style.use('fivethirtyeight')

sns.boxenplot(data['rating'], data['length'], palette = 'deep')
plt.title("Rating vs Length - Plot 3")
plt.xticks(rotation = 90)
plt.show()


# In[ ]:


plt.rcParams['figure.figsize'] = (12, 7)

sns.stripplot(data['sentiment'], data['length'], palette = 'Reds')
plt.title("Sentiment vs Length - Plot 1")
plt.show()


# In[ ]:


plt.rcParams['figure.figsize'] = (15, 9)
plt.style.use('fivethirtyeight')

sns.boxenplot(data['sentiment'], data['length'], palette = 'deep')
plt.title("Sentiment vs Length - Plot 2")
plt.xticks(rotation = 90)
plt.show()


# > According to these **plots** on **Rating/Sentiment vs Length**:
# - The **higher** the rating, the **longer** the review
# - The **higher** the rating, the **greater** the outliers
# - The outliers are far from the **median**
# 
# Our conclusion is that people tend to have **more** to say when they are more **satisfied.**

# In[ ]:


plt.rcParams['figure.figsize'] = (15, 9)
plt.style.use('fivethirtyeight')

x = pd.crosstab(data['year'], data['rating'])
color = plt.cm.cool(np.linspace(0, 1, 8))
x.div(x.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, color = color)
plt.title("Year vs Rating", fontweight = 30, fontsize = 20)
plt.show()


# > Comparing **year and rating:**
# - **Negative sentiments** were **non-existent** in the first 2 years (2004, 2005)
# - **Rating of 1** is easily the **most frequent** in year **2006**
# - From **year 2007** to the **year 2014**, negative sentiments **decreased** despite **popularity increasing**

# In[ ]:


plt.rcParams['figure.figsize'] = (15, 7)
sns.violinplot(data['sentiment'], data['rating'], palette = 'spring')
plt.title('Sentiment vs Rating', fontsize = 20)
plt.show()


# > Reiterating the fact that **ratings of 1, 2, and 3** have a sentiment value of **0** while **ratings of 4 and 5** have a sentiment value of **1**

# In[ ]:


trace = go.Scatter3d(
    x = data['length'],
    y = data['sentiment'],
    z = data['rating'],
    name = 'Amazon Alexa',
    mode='markers',
    marker=dict(
        size=10,
        color = data['rating'],
        colorscale = 'Plasma',
    )
)
df = [trace]

layout = go.Layout(
    title = 'Length vs Sentiment vs Ratings',
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0  
    )
    
)
fig = go.Figure(data = df, layout = 
                layout)
iplot(fig)


# > We categorized the values based off its **Length, Sentiment, and Ratings**
# - Values are along the scale of **length** (x-axis)
# - Values with **different sentiments** are on **opposite sides** (y-axis)
# - The values are then **differentiated** by their **ratings** (z-axis)

# In[ ]:


plt.rcParams['figure.figsize'] = (12, 12)
plt.style.use('fivethirtyeight')
sns.heatmap(data.corr(), square=True, annot=True, cmap='RdBu')
plt.title('Numerical Correlations')
plt.show()


# > We are trying to predict **sentiment**:
# - **Unix_time and year** have **very little correlation** with **sentiment**
# - **Length** actually has **some correlation** with sentiment which makes **sense** based off of the **visualizations** above
# - **Rating** is based off of **sentiment** and was used for visualization, but **shouldn't** be used for training because it is too **similar** to sentiment (0.85)

# <font size="+3" color="blue"><b>Data Preprocessing for Modelling</b></font><br><a id='4'></a>
# <hr>

# > **Drop:**
# - **product_id** has **too many unique values** and **no correlation**
# - **unix_time** has **too many unique values** and **no correlation**
# - **year** is similar to **unix_time** because they are both based off of **time**, **no correlation**
# - **length** had some sort of **correlation** but isn't **strong** enough to be used
# - **rating** is **extremely correlated** to sentiment and is a **redundant feature**

# In[ ]:


# Drop for reasons above
data.drop(['unix_time', 'year', 'length', 'rating', 'product_id'], axis=1, inplace=True)
data.head()


# **Stop Words:** A stop word is a commonly used word ("the", "a", "an") that holds no importance and is pretty much useless.
# 
# <hr>
# 
# **Objective:** We do not want these words as they take up valuable processing time and space, so we remove them. Here we set up all of the stopwords we will be removing.

# In[ ]:


# Setting up stopwords
stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)

# Including punctuation as a stopword
stop.update(punctuation)


# **Lemmatization:** The process of reducing a word to its lemma.
# **Lemma:** The root of which a word is formed.
# 
# <hr>
# 
# **Example:**
# 
# Went -> Go
# 
# Better -> Good
# 
# **Another Example:**
# <img src='https://cdn-images-1.medium.com/max/1600/1*z4f7My5peI28lNpZdHk_Iw.png' style='height:300px'>
# 
# 
# 
# **Note: In this case we lemmatize the words in the reviews based off its type (Noun, verb, adverb, etc.)**

# In[ ]:


# Function for returning the type of word
def get_simple_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
    
# Lemmatizing words that are not stopwords (Stopwords we ignore)
lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            pos = pos_tag([i.strip()])
            word = lemmatizer.lemmatize(i.strip(),get_simple_pos(pos[0][1]))
            final_text.append(word.lower())
    return " ".join(final_text)

data['review'] = data['review'].apply(lemmatize_words)

data.head()


# <font size="+3" color="blue"><b>Modelling</b></font><br><a id='5'></a>
# <hr>

# > **Splitting the data:**
# - Our X variable only contains **feature 'review'** as it is the only **useful feature**
# - Our y variable contains given **sentiment values**

# In[ ]:


X = data['review']
y = data['sentiment']


# > **Count Vectorizer:**
# - Our data must be vectorized in order for it to be used by the model
# 
# >> **How does it work?:**
# - Create a vector the **length of every unique word** in the entire 'review' feature 
# - Every text that has a word in the dictionary is represented by **1**, otherwise **0**
# - This creates very large vectors **mostly full of 0's**
# 
# ![](https://miro.medium.com/proxy/1*YEJf9BQQh0ma1ECs6x_7yQ.png)
# 
# <font size='+1' color='red'><b>We choose not to use this method as we have a better method, however understanding this is important</b></font>

# In[ ]:


# We don't use this method but this is the code
cv = CountVectorizer(min_df=0,max_df=1,binary=False,ngram_range=(1,3))

cv_train_reviews=cv.fit_transform(X)


# > **TFIDF Vectorizer (Term Frequency - Inverse Document Frequency):**
# - Tokenizes just like **Count Vectorizer**
# - **Term Frequency:** How many times a word appears in a document
# - **Inverse Document Frequency:** Downscales words that appear a lot ('is', 'the') and scales up the words that appear **rarely**
# - **Term Frequency** is divided by the **document length**
# - **TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document)**
# - **IDF(t) = log_e(Total number of documents / Number of documents with term t in it)**
# 
# 
# >> **Example:**
# Consider a document containing 100 words wherein the word cat appears 3 times. The term frequency (i.e., tf) for cat is then (3 / 100) = 0.03. Now, assume we have 10 million documents and the word cat appears in one thousand of these. Then, the inverse document frequency (i.e., idf) is calculated as log(10,000,000 / 1,000) = 4. Thus, the Tf-idf weight is the product of these quantities: 0.03 * 4 = 0.12.
# 
# <img src='https://miro.medium.com/max/1200/1*V9ac4hLVyms79jl65Ym_Bw.jpeg' style='height:300px'>
# 
# **Source:** *http://www.tfidf.com/*
# 
# **We will be using this approach for vectorizing our tokens**

# In[ ]:


tf=TfidfVectorizer(min_df=0,max_df=1,use_idf=True,ngram_range=(1,2))
X=tf.fit_transform(X)


# > **SMOTETomek** is a class to perform **over- and under-sampling.**<br>
# **Details:** [SMOTETomek](http://glemaitre.github.io/imbalanced-learn/generated/imblearn.combine.SMOTETomek.html)
# 
# **Example:**
# <img src='https://glemaitre.github.io/imbalanced-learn/_images/sphx_glr_plot_smote_tomek_001.png' style='height:350px'>

# In[ ]:


# Class to perform over- and under-sampling

smk = SMOTETomek(random_state=42 , sampling_strategy = 0.8)
X,y=smk.fit_sample(X,y)


# > **Splitting, Training, Predictions, and Score:**
# - Splitting data into **train** and **test** data
# - Using Multinomial Naive Bayes with pre-set parameters
# - Fitting model with **X_train** and **y_train**
# - Predicting model
# - Getting score

# In[ ]:


X_train, X_test, y_train ,y_test = train_test_split(X, y, test_size = 0.2 , random_state = 0)

mnb_model=MultinomialNB()

mnb_model = mnb_model.fit(X_train, y_train)

predictions = mnb_model.predict(X_test)

score=accuracy_score(y_test, predictions)

print("Score: ", score)


# > **Inspecting results**
# - Classification report
# - Confusion Matrix

# In[ ]:


# CLASSIFICATION REPORT

mnb_bow_report = classification_report(y_test, predictions)
print(mnb_bow_report)


# In[ ]:


# CONFUSION MATRIX

confusion_matrix = pd.crosstab(y_test, predictions, rownames=['Actual'], colnames=['Predicted'])

sns.heatmap(confusion_matrix, annot=True)


# 
# <hr>
# 
# <img src='https://images.saymedia-content.com/.image/MTcxNDk4OTk3NjEyMTYwNDgz/giphy.gif' style='height: 400px'>
# 
# <hr>
