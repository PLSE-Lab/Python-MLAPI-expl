#!/usr/bin/env python
# coding: utf-8

# [Hacker News](https://news.ycombinator.com) is a community where users can submit articles, and other users can upvote those articles. The articles with the most upvotes make it to the front page, where they're more visible to the community.
# 
# Our data set consists of submissions users made to Hacker News. A few developers have gathered this data. I will use the dataset uploaded by [Anthony](https://www.kaggle.com/antgoldbloom). This data set is Hacker News posts from the last 12 months (up to September 26 2016). Our data has the following columns:
# * `title`: title of the post (self explanatory)
# * `url`: the url of the item being linked to
# * `num_points`: the number of upvotes the post received
# * `num_comments`: the number of comments the post received
# * `author`: the name of the account that made the post
# * `created_at`: the date and time the post was made (the time zone is Eastern Time in the US)
# ---
# **Goal:** to train a linear regression model that predicts the number of upvotes a headline would receive. To do this, we'll need to convert each headline to a numerical representation. We'll use the **bag of words** appraoch where:
# >a text (such as a sentence or a document) is represented as the bag (multiset) of its words, disregarding grammar and even word order but keeping multiplicity ~Wiki
# 
# The first step in creating a bag of words model is tokenization. In tokenization, we break a sentence up into disconnected words.

# ## <center> Data Exploration <center>

# In[ ]:


import pandas as pd
import numpy as np
import warnings
def fxn():
    warnings.warn("deprecated", DeprecationWarning)
    
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
    
submissions = pd.read_csv('../input/HN_posts_year_to_Sep_26_2016.csv')
print('Shape of dataset is:',submissions.shape)
submissions.head()


# In[ ]:


#percent of missing values for each column
pd.DataFrame(submissions.isnull().sum()/submissions.shape[0]*100,columns=['% Missing Values']).round(2)


# In[ ]:


#who has posted the most? and how many?
print('Highest number of posts is {1} made by {0}'.format(submissions['author'].value_counts().index.tolist()[0],submissions['author'].value_counts().tolist()[0]))


# Oh wow! such an active user! let's see how much points he has recieved?

# In[ ]:


jonbaer=submissions[submissions['author']=='jonbaer']
print('jonbaer recieved {0:.2f} average points, while average points for all posts is {1:.2f}'.format(jonbaer['num_points'].mean(),submissions['num_points'].mean()))


# Now let's look at who has gotten the most number of votes on average.

# In[ ]:


ave_votes_byauthor=submissions.groupby('author').mean()
ave_votes_byauthor['num_points'].sort_values(ascending=False).head(5)


# In[ ]:


import numpy as np
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)

def histly(df,target):
    title_text='Histogram of log of average {0} by user'.format(target)
    
    data = [go.Histogram(x=np.log1p(df[target]))]
    
    shapes_list=[{
        'type': 'line',
        'xref': 'x',
        'yref': 'paper',
        'x0': np.log1p(df[target].mean()),
        'y0':0,
        'x1': np.log1p(df[target].mean()),
        'y1':1,
        'line': {
            'color': 'b',
            'width': 5,
            'dash': 'dashdot'
        }}]
        
    annotations_list=[{
            'x':np.log1p(df[target].mean()),
            'y': 50,
            'xref':'x',
            'yref':'y',
            'text':'Average across all data',
            'showarrow':True,
            'arrowhead':7,
            'ax':100,
            'ay':-100
            }]
        
    layout = go.Layout(
        title=title_text,
        font=dict(size=14, color='b'),
        xaxis={
        'title':'Log of average',
        'titlefont':{
            'size':18,
            'color':'b'
        }
        },
        yaxis={
        'title':'Count',
        'titlefont':{
            'size':18,
            'color':'b'
        }
        },
        autosize=True,
        shapes=shapes_list,
        annotations=annotations_list
        )
    
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig)


# In[ ]:


histly(ave_votes_byauthor,'num_points')


# ## <center> Date Preparation <center>

# I dont plan to utilize url_hostname for now. The only columns I will be using for regression analysis are headlines and points. Let's only gather `title` and `num_points`, and then drop na `title` rows.
# 
# **Note** due to the large size of the data, when I tried to create the `counts` dataframe, I got a Memory Error. Therefore, for the representation purposes, I only use a small portion of the data. However, I will switch to the full dataset when fitting the regression model.

# In[ ]:


train=submissions.loc[:,['title','num_points']]

#sampling 5% of the daset for the representation purposes of the next two steps.
train=train.sample(frac=0.05,axis=0).reset_index()

train=train.dropna()
train.shape


# There are four ways to remove punctuations:
# * **sets**
#     - exclude = set(string.punctuation) \n s = ''.join(ch for ch in s if ch not in exclude)
# * **regex**
#     - s = re.sub(r'[^\w\s]','',s) OR re.compile('[%s]' % re.escape(string.punctuation)).sub('',s)
# * **translate**
#     - s = s.translate(str.maketrans('','',string.punctuation))
# * **replace**
#     - for c in string.punctuation: \n s=s.replace(c,"")
# 
# Among all these approaches, `translate()` method beats the others in terms of speed. please refer to **[this post](https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string-in-python)** on StackOverflow. But please note that the syntax mentioned for `translate()` in that post is applicable in Python 2. For Python 3, please refer to **[this post](https://stackoverflow.com/questions/23175809/str-translate-gives-typeerror-translate-takes-one-argument-2-given-worked-i)**.

# In[ ]:


#removing the punctuations.
import string
train['title_nopuncs']=train['title'].apply(lambda x: x.translate(str.maketrans('','',string.punctuation)))


# Also, we should lower case the titles. Apple, apple, and APPLE are all the same!

# In[ ]:


#lower casing titles
train['title_nopuncs']=train['title_nopuncs'].apply(lambda x: x.lower())


# Now, we'd like to to tokenize the titles. I use `split()` function. One could use `nltk.tokenize` as well. Based on **[this post](https://stackoverflow.com/questions/9602856/most-efficient-way-to-split-strings-in-python)**, `split()` works fairly good on not too long strings.

# In[ ]:


# tokenizing the headlines
train['tokenz'] = train['title_nopuncs'].apply(lambda x: x.split())
train['tokenz'].head()


# Now, we should use find unique tokens. I can think of two approaches:
# * creates a master list of all the tokenz, and call unique() function on it.
# * create an emppty list, and append the unique tokenz to it. **Don't do this! It takes forever! Obviously.**
# 
# 
# **OR** use **[this](https://stackoverflow.com/questions/1720421/how-to-concatenate-two-lists-in-python)** awesome post on StackOverflow and find the following approach!

# In[ ]:


import itertools

#this will create a list of all words
words=list(itertools.chain.from_iterable(train['tokenz']))

#this will create a list of unique words
unique_words=list(set(words))

print('Number of unique words:',len(set(unique_words)))


# Next, we should create the **bag of words** matrix. It is a way of representing text data while performing machine learning. The three steps in this approach are:
# * tokenizing : we have already taken care of this!
# * counting: This is what we are about to do. Basically we count how many times those unique words occured in each headline, and format this information in a dataframe.
# * normalizing: we don't want too frequent, and once-in-a-lifetime words exist in our data!

# In[ ]:


#forming a dataframe of 0 values
counts = pd.DataFrame(0,index=np.arange(train.shape[0]), columns=unique_words)
#counts.shape


# In[ ]:


#now counting the number of words in each headline and adding it to our dataframe
for index, row in train.iterrows():
    for token in row['tokenz']:
        counts.iloc[index][token]+=1


# Interestingly, we could use the `sklearn.feature_extraction` that does all the steps that we have just implemented!

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X=vectorizer.fit_transform(list(train['title']))
counts=pd.DataFrame(X.toarray(),columns=vectorizer.get_feature_names())
#print( vectorizer.vocabulary_)


# Too many columns. There are two types of features that will reduce regression accuracy:
# * The ones that occur only a few times. These will cause over fitting.
# * The ones that occur too many times, such as `a` and `and`. These are often called `stopwords`, and do not indicate any relationship with the upvotes.  
# Let's remove any word that occur fewer than 5 and more than 100 times.

# In[ ]:


count_sum=counts.sum()
counts=counts.drop(count_sum[(count_sum>100) | (count_sum<5)].index,axis=1)


# ## <center> Model Fitting <center>

# In[ ]:


# spliting data into train and validation sets
from sklearn.cross_validation import train_test_split

X_train,X_test,y_train,y_test=train_test_split(counts,train['num_points'],train_size=0.8,random_state=1)


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lr = LinearRegression()
lr.fit(X_train,y_train)
pred=lr.predict(X_test)
rmse=(mean_squared_error(pred,y_test))**0.5
print('RMSE is: {0:.2f}'.format(rmse))


# Which is pretty high! But please remember we are only using 5% of the data set. A larger data would drastically enhance model accuracy.
