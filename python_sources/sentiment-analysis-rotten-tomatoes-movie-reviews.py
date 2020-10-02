#!/usr/bin/env python
# coding: utf-8

# ****

# <center><h1>Sentiment Analysis</h1></center>
# The dataset is comprised of tab-separated files with phrases from the Rotten Tomatoes dataset. The train/test split has been preserved for the purposes of benchmarking, but the sentences have been shuffled from their original order. Each Sentence has been parsed into many phrases by the Stanford parser. Each phrase has a PhraseId. Each sentence has a SentenceId. Phrases that are repeated (such as short/common words) are only included once in the data.<br><br>
# 
# train.tsv contains the phrases and their associated sentiment labels. We have additionally provided a SentenceId so that you can track which phrases belong to a single sentence.<br>
# test.tsv contains just phrases. You must assign a sentiment label to each phrase.<br>
# The sentiment labels are:<br>
# 
# 0 - negative<br>
# 1 - somewhat negative<br>
# 2 - neutral<br>
# 3 - somewhat positive<br>
# 4 - positive

# In[ ]:


#-------For DataFrame and Series data manipulation
import pandas as pd
import numpy as np
#-------Data visualisation imports
import seaborn as sns
import matplotlib.pyplot as plt
#-------Interactive data visualisation imports
from plotly import __version__
#print(__version__)
import cufflinks as cf
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)
cf.go_offline()
#-------To make data visualisations display in Jupyter Notebooks
get_ipython().run_line_magic('matplotlib', 'inline')
#-------To split data into Training and Test Data
from sklearn.model_selection import train_test_split
#-------To make pipelines
from sklearn.pipeline import Pipeline
#-------For Natural Language Processing data cleaning
from sklearn.feature_extraction.text import TfidfTransformer
#CountVectorizer converts collection of text docs to a matrix of token counts.
from sklearn.feature_extraction.text import CountVectorizer
import string
import nltk
from nltk.corpus import stopwords
#-------For model scoring
from sklearn.metrics import classification_report


# <h1>Importing Data</h1>
# Let's import the training data first to see what we are working with

# In[ ]:


trainmessages = pd.read_csv('../input/train.tsv', sep='\t')


# In[ ]:


trainmessages.info()


# In[ ]:


trainmessages['Phrase'].describe()


# In[ ]:


trainmessages.head()


# In[ ]:


#we have to import the test data to fit to our model
testmessages = pd.read_csv('../input/test.tsv', sep='\t')
testmessages.head()


# <h1>Exploring the data</h1>
# Let's see what kind of information we can get from the data.

# In[ ]:


sns.countplot(data=trainmessages,x='Sentiment')


# In[ ]:


#to get the numerical values of the above countplot
trainmessages['Sentiment'].iplot(kind='hist')


# Let's see if we have any null values in our training data.

# In[ ]:


trainmessages.isnull().sum()


# In[ ]:


trainmessages.isna().sum()


# We will try to get the lengths of each Phrase to see if it has an effect on Sentiment rating.

# In[ ]:


trainmessages['Length'] = trainmessages['Phrase'].apply(lambda x: len(str(x).split(' ')))


# In[ ]:


trainmessages['Length'].unique()


# In[ ]:


data = [dict(
  type = 'box',
  x = trainmessages['Sentiment'],
  y = trainmessages['Length'],
  transforms = [dict(
    type = 'groupby',
    groups = trainmessages['Sentiment'],
  )]
)]
iplot({'data': data}, validate=False)


# There might be a correlation between Phrase and Sentence ID and/or Phrase Length since they want us to output PhraseID and Sentiment only...
# Maybe if we do a pairplot we might spot some patters....

# In[ ]:


sns.pairplot(trainmessages,hue='Sentiment',vars=['PhraseId','SentenceId','Length'])


# In[ ]:


#double-check for any empty Phrases
trainmessages = trainmessages[trainmessages['Phrase'].str.len() >0]


# In[ ]:


#are there any empty Phrases? if so let's remove them
trainmessages[trainmessages['Phrase'].str.len() == 0].head()


# In[ ]:


trainmessages = trainmessages[trainmessages['Phrase'].str.len() != 0]


# In[ ]:


trainmessages[trainmessages['Phrase'].str.len() == 0].head()


# In[ ]:


#create function to clean data
def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all capitalized words
    2. Remove all punctuation
    3. Remove all stopwords
    4. Returns a list of the cleaned text
    """
    #Remove capitalized words (movie names, actor names, etc.)
    nocaps = [name for name in mess if name.islower()]
    
    #Join the characters again to form the string.
    nocaps = ' '.join(nocaps)
    
    # Check characters to see if they are in punctuation
    nopunc = [char for char in nocaps if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    nostopwords = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    
    # Join the characters again to form the string.
    nostopwords = ' '.join(nostopwords)
    
    return nostopwords


# <h1>Fitting and Predicting</h1>
# 
# So I have decided to focus on two simple methods:<br>
# 
# 1. We can simply tokenize the words without merging SentenceId's.
# 2. We can merge words/Phrases from a similar SentenceId or simply keep the longest phrase. 
# 
# I will only show the model/method I used to get the highest accuracy.<br>
# 
# <h2>Splitting Data into Training and Test Data</h2>
# To know the true predictive power of our model, we have to split the training .tsv file into training and test data.

# In[ ]:


#because of our imbalanced classes (categorical labels) we can try over-sampling (making copies of the under-represented classes)
sent_2 = trainmessages[trainmessages['Sentiment']==2]
#we will copy class 0 11 times
sent_0 = trainmessages[trainmessages['Sentiment']==0]
#we will copy class 1 2 times
sent_1 = trainmessages[trainmessages['Sentiment']==1]
#we will copy class 3 2 times
sent_3 = trainmessages[trainmessages['Sentiment']==3]
#we will copy class 4 8 times
sent_4 = trainmessages[trainmessages['Sentiment']==4]

#-----------------------------------------------------
trainmessages = trainmessages.append([sent_0,sent_0,sent_0,sent_0,sent_0,sent_0,sent_0,sent_0,sent_0,sent_0])
trainmessages = trainmessages.append([sent_1,sent_1])
trainmessages = trainmessages.append([sent_3])
trainmessages = trainmessages.append([sent_4,sent_4,sent_4,sent_4,sent_4,sent_4,sent_4])


# In[ ]:


#to check the amounts of each class
sns.countplot(data=trainmessages,x='Sentiment')


# In[ ]:


#we split our train.tsv into training and test data to test model performance
X = trainmessages['Phrase']
y = trainmessages['Sentiment']
msg_train,msg_test,label_train,label_test = train_test_split(X,y)


# <h1>Predicting Sentiment based on Words</h1>
# We can try to predict now by ***keeping the most words and their respective Sentiment scores***

# In[ ]:


#let's try using the RandomForestClassifier model to predict
from sklearn.ensemble import RandomForestClassifier
pipelineRFC = Pipeline([
    ('bow',CountVectorizer(analyzer=text_process)),
    ('tfidf',TfidfTransformer()),
    ('classifier',RandomForestClassifier())
])


# In[ ]:


pipelineRFC.fit(msg_train,label_train)
preds = pipelineRFC.predict(msg_test)
print(classification_report(label_test,preds))


# <h1>Predictions Submission</h1>
# Let's prepare the file we want to submit

# In[ ]:


#we choose the pipeline with the BEST most ACCURATE model and store the predictions in a variable
preds = pipelineRFC.predict(testmessages['Phrase'])


# In[ ]:


sub = pd.DataFrame(columns=['PhraseId','Sentiment'])
sub['PhraseId'] = testmessages['PhraseId']
sub['Sentiment'] = pd.Series(preds)


# In[ ]:


sub.head()


# In[ ]:


sns.countplot(data=sub,x='Sentiment')


# In[ ]:


#Convert DataFrame to a csv file that can be uploaded
subfile = 'RT Movie Review Predictions.csv'
sub.to_csv(subfile,index=False)
print('Saved file: ' + subfile)


# In[ ]:




