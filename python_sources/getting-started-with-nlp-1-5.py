#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import re
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd 

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 5000)


# In[ ]:


text1 = "For whomsoever shall hold this hammer, if he shall be worthy shall possess the Power of Thor" 
text2 = "Did you do it ? yeS"
text3 = "What dId it coSt ? EveryThing"
text4 = text2 +' '+ text3


# In[ ]:


print(text1, text2, text3, text4, sep="\n")


# In[ ]:


print(len(text1), len(text2), len(text3), len(text4), sep="\n")


# ### Split the sentence in words - How ? 
# ### Split them by a delimiter. CSV(comma separated value) files are read by discarding the *commas* in between data, same way these words are separated by whitespace.
# 
# > https://nlp.stanford.edu/IR-book/html/htmledition/tokenization-1.html

# In[ ]:


text1a = text1.split(' ')
text4a = text4.split(' ')
print(len(text1a), len(text4a))
print(text1a, text4a, sep='\n')


# ### Now the words are tokenized(chopping sentence into pieces/tokens). This O/P is then fed as I/P for Parsing or Text Mining
# 
# # String Operations

# ## Remember List Comprehension from C1 - Iterate through collections in one line

# In[ ]:


# Long words: Words that have more than 3 characters
[w for w in text1a if len(w)>3] 
# Long words: Words that have less than 3 characters
[w for w in text1a if len(w)<3] 


# In[ ]:


print([w for w in text1a if w.istitle()])              # Words that have 1st character in Capital
print([w for w in text1a if w.endswith('r')])          # Words that end with character 'r'


# ### unique words in a list of words using set() function

# In[ ]:


print("Words that are unique without preprocessing (1 '?', 1 'it', 2 'Did & dId'")
print(set(text4a))
print("\nWords that are unique with preprocessing (1 '?', 1 'it', 1 'did'")
print(set([w.lower() for w in text4a]))


# ### Convert all words to lowercase || uppercase || title

# In[ ]:


print('Uppercase : ', [w.upper() for w in text4a], '\n') # Convert all words to lowercase
print('Lowercase : ', [w.lower() for w in text4a], '\n') # Convert all words to uppercase
print('Title     : ', [w.title() for w in text4a], '\n') # Convert all words to title(1st letter caps)


# ### Splitting a sentence based a word(stopper word/connectors)

# In[ ]:


print(text1)
print(text1.split('shall'), '\n')


# ### Merging 3 sentences with a Connector (and/will/hence etc) using .join() function

# In[ ]:


print('shall'.join(text1.split('shall')))
print('will'.join(text1.split('shall')))
print('should'.join(text1.split('shall')))


# In[ ]:


temp = "ouagadougou"
print("split a string - ouagadougou(eg:sentence/word) into substring - ou(eg:words):")
print("ouagadougou".split('ou')) 


# In[ ]:


print("\nsplit a substring - ouagadougou(eg:word) into separate characters:")
print("Method 1", list("ouagadougou"))       
print("Method 2", [c for c in "ouagadougou"])


# In[ ]:


"ouagadougou".split('ou')# Error         --> split a word into character


# In[ ]:


textnick = "      See, it's things like this that give me trust issues.  "
print(textnick.split(' '))


# ### Note Below, the disappearance of the whitespaces at the beginning and end of the string

# In[ ]:


textnick = "      See, it's things like this that give me trust issues.  "
print(textnick.strip().split(' '))


# ## Find first occurance of a substring/character in a String/Sentence
# #### To find all occurances, we can use *Regular Expressions* re.find_iter function 
# > https://docs.python.org/2/library/re.html#module-contents
# 

# In[ ]:


Movies = "Marvel - Hulk, Marvel - Thor, Marvel - Ironman, Marvel - Captain America, Marvel - Avengers"
print("INDEX of First occurance of the substring Marvel in the sentence Movies from the start", Movies.find('Marvel'))
print("\nINDEX of First occurance of the substring Marvel in the sentence Movies from the end", Movies.rfind('Marvel'))
print("\nAll occurances of Marvel in the sentence is replaced by DC:\n", Movies.replace('Marvel', 'DC'))


# # Handling larger Texts
# ## Reading files by line

# In[ ]:


f = open('../input/UNHDR.txt', 'r')
EOL_space = f.readline()
EOL_space


# In[ ]:


EOL_space.rstrip()  # To remove the \n character from the end of the line use strip() and its add on versions 


# ## Reading the full file

# In[ ]:


f = open('../input/UNHDR.txt', 'r')
f.seek(0) # Reset the reading pointer to the start
HDRUN = f.read()
print("There are {} sentences in this file".format(len(HDRUN.splitlines())))
HDRUN.splitlines()


# ## File Operations

# # Most Annoying part of Text preprocessing - Regular Expressions [Regex]
# ## But also the most important part

# ###  Python Regular Expressions Made Easy (2017) -->  https://www.youtube.com/playlist?list=PLGKQkV4guDKH1TpfM-FvPGLUyjsPGdXOg

# In[ ]:


text5 = '"Ethics are built right into the ideals and objectives of the United Nations" #UNSG @ NY Society for Ethical Culture bit.ly/2guVelr @UN @UN_Women'
text5a = text5.split(' ')
print(text5a)


# ### Extract elements with special symbols (meanings) @ and # are used in tweets

# In[ ]:


print([w for w in text5a if w.startswith('#')])
print([w for w in text5a if w.startswith('@')])


# In[ ]:


import re
[w for w in text5a if re.search('@[A-Za-z0-9_]+', w)]


# ### Meta Characters in Regular Expressions
# 
# ```
# .       --> Wilcard Character that matches a single character
# ^       --> Start of the String
# $       --> End of the String
# []      --> Matches one of the charaters, that are within the square brackets 
# [a-z]   --> Matches one of the range of characters a,b,c,d,e,f,g......x,y,z
# [^abc]  --> Matches any character except a,b,c - Exclude a,b,c
# [a|b]   --> Matches either a or b, where a & b are strings
# ()      --> Scoping for Operators - just normal use
# \       --> Escape character for special characters (\n,\t,\b,\d, \D)
# 
# ```
# 
# ### Meta Characters: Character Symbols
# ```
# \b      --> Matches Word boundary
# \d      --> Any digit, equivalent to [0-9]
# \D      --> Any non-digit, equivalent to [^0-9]
# \s      --> Any whitespace, equivalent to [\t\n\r\f\v]
# \S      --> Any non-whitespace, equivalent to [^\t\n\r\f\v]
# \w      --> Any alphanumeric Character, equivalent to [a-zA-Z0-9_]
# \W      --> Any non-alphanumeric Character, equivalent to [^a-zA-Z0-9_]
# ```
# 
# ### Meta Characters : Repetitions
# ```
# *       --> Matches zero or more occurences
# +       --> Matches one or more occurences
# ?       --> Matches zero or one occurences
# {n}     --> Matches exactly in repetitions, n>=0
# {n,0}   --> Matches Atleast n repetition    
# {0,n}   --> Matches Atmost n repetition
# {m,n}   --> Atleast Atleast m and Atmost n repetitions  
# ```

# In[ ]:


[w for w in text5a if re.search('@\w+', w)]


# ### Find (Ctr + F) all vowels

# In[ ]:


temp = "ouagadougou"
re.findall(r'[aeiou]', temp)


# ### Find (Ctr + F) all consonants 

# In[ ]:


re.findall(r'[^aeiou]', temp)


# ## Date Variations 
# > 01-11-2018<br/>
# > 01/11/2018<br/>
# > 01/11/18<br/>
# > 11/01/2018<br/>
# > 11 Nov 2018<br/>
# > 11 November 2018<br/>
# > Nov 11, 2018<br/>
# > November 11, 2018<br/>

# In[ ]:


# Regular Expression for Dates
dateStr = '01-11-2018\n01/11/2018\n01/11/18\n11/01/2018\n11 Nov 2018\n11 November 2018\nNov 11, 2018\nNovember 11, 2018\n'
print(re.findall(r'\d{2}[/-]\d{2}[/-]\d{4}', dateStr))
print(re.findall(r'\d{2}[/-]\d{2}[/-]\d{2,4}', dateStr))
print(re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', dateStr))


# In[ ]:


print(re.findall(r'\d{2} (Jan|Feb|Mar|Apr|May|Jun|July|Aug|Sep|Oct|Nov|Dec) \d{4}', dateStr))
print(re.findall(r'\d{2} (?:Jan|Feb|Mar|Apr|May|Jun|July|Aug|Sep|Oct|Nov|Dec) \d{4}', dateStr))
print(re.findall(r'\d{2} (?:Jan|Feb|Mar|Apr|May|Jun|July|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}', dateStr))
print(re.findall(r'(?:\d{2}? )?(?:Jan|Feb|Mar|Apr|May|Jun|July|Aug|Sep|Oct|Nov|Dec)[a-z]* (?:\d{2}, )?\d{4}', dateStr))
print(re.findall(r'(?:\d{1,2}? )?(?:Jan|Feb|Mar|Apr|May|Jun|July|Aug|Sep|Oct|Nov|Dec)[a-z]* (?:\d{1,2}, )?\d{2,4}', dateStr))


# In[ ]:


import pandas as pd
time_sentences = ["Monday: The doctor's appointment is at 2:45pm.", 
                  "Tuesday: The dentist's appointment is at 11:30 am.",
                  "Wednesday: At 7:00pm, there is a basketball game!",
                  "Thursday: Be back home by 11:15 pm at the latest.",
                  "Friday: Take the train at 08:10 am, arrive at 09:00am."]

df = pd.DataFrame(time_sentences, columns=['text'])
df


# # To Perform String Operations on Series/Dataframe - .str is used

# In[ ]:


df['text'].str.len()


# In[ ]:


df['text'].str.split().str.len()


# In[ ]:


print("\nColumns with Word Appointment : \n",df['text'].str.contains('appointment'))
print("\nColumns with Word day : \n",df['text'].str.contains('day'))
print("\nColumns with timestamp at night - pm : \n",df['text'].str.contains('pm'))


# #### Find count of digits in the data 

# In[ ]:


# How many times a digit occurs in every sentence
df['text'].str.count(r'\d')


#  #### Find all unique digits in the data 

# In[ ]:


df['text'].str.findall(r'\d')


# #### Find time values in the form of digits in the data 

# In[ ]:


df['text'].str.findall(r'(\d?\d):(\d\d)')


# In[ ]:


df['text'].str.replace(r'\w+day\b', ':-)   ')


# In[ ]:


df['text'].str.replace(r'(\w+day\b)', lambda x: x.group()[0][:3])


# In[ ]:


df['text'].str.extract(r'(\d?\d):(\d\d)')


# In[ ]:


df['text'].str.extractall(r'((\d?\d):(\d\d) ?([ap]m))')


#  ## Name groups in str

# In[ ]:


df['text'].str.extractall(r'(?P<Time>(?P<Hour>\d?\d):(?P<Minute>\d\d) ?(?P<Period>[ap]m))')


# # NLTK - Natural Language Processing Toolkit

# In[ ]:


import nltk


# In[ ]:


print(dir(nltk))


# In[ ]:


from nltk.book import *


# #### Counting the vocabulary of words

# In[ ]:


text7


# In[ ]:


sents()


# In[ ]:


print(len(sent7))
print(sent7)


# In[ ]:


len(text7)


# In[ ]:


len(set(text7))


# In[ ]:


len(set([w.lower() for w in text7]))


# In[ ]:


print(list(set(text7))[:10])


# ### Frequency of Words

# In[ ]:


dist = FreqDist(text7)
print(type(dist))
print(len(dist))
print(dist)


# In[ ]:


print([w for w in dist.items()][1:500])


# In[ ]:


type(dist.items())


# In[ ]:


vocab = dist.keys()
list(vocab)[:10]


# In[ ]:


dist['four']


# #### How many times a large word occurs in the text corpus

# In[ ]:


freqwords = [w for w in vocab if len(w)>5 and dist[w]>100]
freqwords


# 
# ## Stemming and Lemmatization 
# #### https://www.youtube.com/watch?v=p1ccbR2P_xA
# 
# ### Normalization and Stemming
# 

# In[ ]:


input1 = "List listed lists listing listings"
words1 = input1.lower().split(' ')
words1


# In[ ]:


porter = nltk.PorterStemmer()
[porter.stem(t) for t in words1]


# In[ ]:


input2 = "Trouble troubling troubled troubler"
words2 = input2.lower().split(' ')
words2


# In[ ]:


porter = nltk.PorterStemmer()
[porter.stem(t) for t in words2]


# In[ ]:


lancast = nltk.LancasterStemmer()
[lancast.stem(t) for t in words2]


# ### Lemmatization

# In[ ]:


udhr = nltk.corpus.udhr.words('English-Latin1')
print(udhr[:20])


# In[ ]:


WNlemma = nltk.WordNetLemmatizer()
print([WNlemma.lemmatize(t) for t in udhr[:20]])


# ### Tokenization

# In[ ]:


text11 = "Children shouldn't drink a sugary drink before bed."
text11.split(' ')


# #### Note that not & . are separate tokens  (inorder to account for negations in the text)

# In[ ]:


print(nltk.word_tokenize(text11))


# #### Note that not all . are end of sentence like in "U.S." and "2.99." 

# In[ ]:


text12 = "This is the first sentence. A gallon of milk in the U.S. costs $2.99. Is this the third sentence? Yes, it is!"
sentences = nltk.sent_tokenize(text12)
print("There are {} sentences in the above document|text corpus".format(len(sentences)))
sentences


# ## Advanced NLP taks with NLTK
# ### POS (Parts-of-Speech)Tagging 

# In[ ]:


from nltk.help import upenn_tagset
dir(upenn_tagset)


# In[ ]:


nltk.help.upenn_tagset(tagpattern='MD')


# In[ ]:


nltk.help.upenn_tagset(tagpattern='V*')


# In[ ]:


text11 = "Children shouldn't drink a sugary drink before bed."
nltk.pos_tag(nltk.word_tokenize(text11))


# In[ ]:


nltk.help.upenn_tagset(tagpattern='NNP')
nltk.help.upenn_tagset(tagpattern='RB')
nltk.help.upenn_tagset(tagpattern='VB')
nltk.help.upenn_tagset(tagpattern='DT')
nltk.help.upenn_tagset(tagpattern='JJ')
nltk.help.upenn_tagset(tagpattern='NN')
nltk.help.upenn_tagset(tagpattern='IN')


# In[ ]:


nltk.pos_tag(nltk.word_tokenize("Visiting aunts can be a nuisance"))


# In[ ]:


nltk.pos_tag(nltk.word_tokenize("I never said you stole my money"))


# ### Parsing sentence structure

# In[ ]:


text15 = nltk.word_tokenize("Alice loves Bob")
grammar = nltk.CFG.fromstring("""
S -> NP VP
VP -> V NP
NP -> 'Alice' | 'Bob'
V -> 'loves'
""")

parser = nltk.ChartParser(grammar=grammar)
trees = parser.parse_all(text15)
for tree in trees:
    print(tree)


# 
# #### Ambiguity in Parsing sentences "I saw the man with a telescope" 
# #### 1. Did you see with the telescope ?
# #### 2. Did you see the man who was holding a telescope ?
# 

# In[ ]:


text16 = nltk.word_tokenize("I saw the man with a telescope")
grammar1 = nltk.data.load('../input/mygrammar.cfg')
grammar1


# In[ ]:


parser = nltk.ChartParser(grammar1)
trees = parser.parse_all(text16)
for tree in trees:
    print(tree)


# In[ ]:


from nltk.corpus import treebank
text17 = treebank.parsed_sents('wsj_0001.mrg')[0]
print(text17)


# ### POS tagging and parsing ambiguity

# In[ ]:


text18 = nltk.word_tokenize("The old man the boat")
nltk.pos_tag(text18)


# In[ ]:


text19 = nltk.word_tokenize("Colorless green ideas sleep furiously")
nltk.pos_tag(text19)


# ## Case Study : Sentiment Analysis

# In[ ]:


import time
start = time.time()
df = pd.read_csv('../input//Amazon_Unlocked_Mobile.csv')
end = time.time()


# In[ ]:


df['Reviews']


# ### Drop missing values and remove neutral ratings (3)
# 
# 

# In[ ]:


df.dropna(inplace=True)
df = df[df['Rating']!=3]
df.head()


# ### Encode Ratings 4,5 as 1 (Positive Sentiment)
# ### Encode Ratings 1,2 as 0 (Negative Sentiment)
# 
# 

# In[ ]:


df['Positively_Rated'] = np.where(df['Rating'] >3,1,0)
df.head()


# ### Looking at the mean - we have imbalanced classes

# In[ ]:


df['Positively_Rated'].mean()


# #### Most ratings are positive

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['Reviews'], df['Positively_Rated'], random_state=0)

print('X_train first entry:\n\n', X_train.iloc[0])
print('\n\nX_train shape: ', X_train.shape)


# ## Count Vectorizor
# 

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer().fit(X_train)


# In[ ]:


vect.get_feature_names()[::2000]


# In[ ]:


len(vect.get_feature_names())


# ### transform the documents in the training data to a document-term matrix ==> Bad of Words Representation

# In[ ]:


X_train_vectorized = vect.transform(X_train)
X_train_vectorized


# In[ ]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train_vectorized, y_train)


# In[ ]:


from sklearn.metrics import roc_auc_score

y_predictions = model.predict(vect.transform(X_test))
print('AUC : ',roc_auc_score(y_test, y_predictions))


# In[ ]:


# get the feature names as numpy array
feature_names = np.array(vect.get_feature_names())

# Sort the coefficients from the model
sorted_coef_index = model.coef_[0].argsort()


# #### Find the 10 smallest and 10 largest coefficients

# In[ ]:


# The 10 largest coefficients are being indexed using [:-11:-1] 
# so the list returned is in order of largest to smallest
print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))


# ## Tf-idf

# In[ ]:


from sklearn.feature_extraction import text
print([w for w in dir(text) if not w.startswith('_')])


from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(min_df=5).fit(X_train)
len(vect.get_feature_names())


# In[ ]:


X_train_vectorized = vect.transform(X_train)

model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

y_predictions = model.predict(vect.transform(X_test))
print('AUC : ',roc_auc_score(y_test, y_predictions))


# In[ ]:


feature_names = np.array(vect.get_feature_names())

sorted_tfidf_index = X_train_vectorized.max(0).toarray()[0].argsort()

print('Smallest tfidf:\n{}\n'.format(feature_names[sorted_tfidf_index[:10]]))
print('Largest tfidf: \n{}'.format(feature_names[sorted_tfidf_index[:-11:-1]]))


# In[ ]:


sorted_coef_index = model.coef_[0].argsort()

print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))


# ### these reviews are treated the same by our current model but they are not (two different)

# In[ ]:


print(model.predict(vect.transform(['not an issue, phone is working',
                                    'an issue, phone is not working'])))


# In[ ]:


print(model.predict(vect.transform(['it is not that i am so smart it is just that i stay with problems longer'])))


# ## n-grams
# #### Fit the CountVectorizer to the training data specifiying a minimum document frequency of 5 and extracting 1-grams and 2-grams

# In[ ]:


vect = CountVectorizer(min_df=5, ngram_range=(1,2)).fit(X_train)
X_train_vectorized = vect.transform(X_train)
len(vect.get_feature_names())


# In[ ]:


model = LogisticRegression()
model.fit(X_train_vectorized, y_train)
y_predictions = model.predict(vect.transform(X_test))
print('AUC: ', roc_auc_score(y_test, y_predictions))


# In[ ]:


feature_names = np.array(vect.get_feature_names())
sorted_coef_index = model.coef_[0].argsort()

print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))


# ### these reviews are now correctly identified

# In[ ]:


print(model.predict(vect.transform(['not an issue, phone is working',
                                    'an issue, phone is not working'])))


# ### Models in NLTK for Text classification

# In[ ]:


print([w for w in dir(nltk) if ('classifier' in w.lower())|('tree' in w.lower())])


# In[ ]:


from nltk.classify import NaiveBayesClassifier

vect = CountVectorizer(min_df=5, ngram_range=(1,2)).fit(X_train)
X_train_vectorized = vect.transform(X_train)
print(X_train_vectorized.shape, y_train.shape)

classifier = NaiveBayesClassifier.train(X_train_vectorized, y_train)
classifier.classify(X_test)

nltk.classify.util.accuracy(classifier, y_test)
classifier.label()


# ## WordNet

# In[ ]:


from nltk.corpus import wordnet as wn


# In[ ]:


deer = wn.synset('deer.n.01')
elk = wn.synset('elk.n.01')
horse = wn.synset('horse.n.01')


# In[ ]:


deer.path_similarity(elk)


# In[ ]:


deer.path_similarity(horse)


# ### Information Criteria to find Lin Similarity

# In[ ]:


from nltk.corpus import wordnet_ic


# In[ ]:


brown_ic = wordnet_ic.ic('ic-brown.dat')

deer.lin_similarity(elk, brown_ic)


# In[ ]:


deer.lin_similarity(horse, brown_ic)


# ## LDA Model

# In[ ]:


import gensim
from gensim import corpora, models


# In[ ]:





# ### This is just the first draft version. Will include some more of my own code with lots of updates in parts 2,3,4,5

# In[ ]:




