#!/usr/bin/env python
# coding: utf-8

#  **                                                                     TOXIC Comments : Sentiment Analysis                                            **

# **Brief Introduction**

# It is very easy to hurt someone's sentiment on social media or forums, specially when it is anonymous. 
# 
# The aim of this project, launched by Jigsaw, is to protect the harrassment of the people with the use of the best algorithm to detect and classify toxic comments from the datasets provided by them.
# 
# The goal of this Notebook is :
# 
# * to draw insight from the dataset provided by Jigsaw
# * and then benchmarking the most commonly used algorithm for Natural Language Processing

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.metrics import log_loss

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack


# In[ ]:


#NLP tools
import re
import string
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
stopwords = nltk.corpus.stopwords.words('english')


# In[ ]:


#Plot and image tools
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import gridspec
import seaborn as sns
sns.set_style("dark")


# In[ ]:


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


#Loading the Data
train = pd.read_csv('../input/train.csv', error_bad_lines=False).fillna(' ')
test = pd.read_csv('../input/test.csv', error_bad_lines=False).fillna(' ')
subm = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


#A quick look at our training dataset
train.head()


# In[ ]:


# the size of our training dataset
train.shape


# So, from the above dataset we can see that there are 6 categories of undesirable comments:
# 
# 1. Toxic
# 2. Severe Toxic
# 3. Obscene
# 4. Threat
# 5. Insult 
# 6. Identity hate

# **Finding more information on the data**

# Total number of toxic comments:

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(ngram_range=(1, 2),
                             max_df=0.5,
                             min_df=4,
                             max_features=1000)
vector_space_model = vectorizer.fit_transform(train['comment_text'].values.astype('U').tolist()) # converting the dtype object to unicode string 
n_comments = vector_space_model.shape[0]
print('%d Total Comments' % n_comments)


# In[ ]:


training_set_size = int(n_comments * 0.33)
X = vector_space_model[:training_set_size,:]
Z = vector_space_model[training_set_size:vector_space_model.shape[0]-1,:]
print('%d comments for the estimation of the parameters and %d for the evaluation' % 
      (X.shape[0], Z.shape[0]))


# In[ ]:


from sklearn import linear_model

X = X.toarray()
Y = train['toxic'][:training_set_size]
model = linear_model.BayesianRidge(verbose=True)
model.fit(X, Y)


# In[ ]:


from sklearn.preprocessing import binarize
from sklearn.metrics import confusion_matrix

ground_truth = train['toxic'][training_set_size:vector_space_model.shape[0]-1]
prediction = model.predict(Z)
prediction = binarize(prediction.reshape(-1, 1), 0.5)


# In[ ]:


toxic_ids = [i for i, c in enumerate(prediction) if c == 1]
toxic_ids


# **Checking for sample toxic comments**

# In[ ]:


comment_id = toxic_ids[0]
print('Content of the comment: \n%s\n' % train['comment_text'][training_set_size+comment_id])
print('Is this comment "toxic" according to the model?\n%s' % str(model.predict(Z[comment_id,:]) >0.5))


# In[ ]:


comment_id = toxic_ids[1]
print('Content of the comment: \n%s\n' % train['comment_text'][training_set_size+comment_id])
print('Is this comment "toxic" according to the model?\n%s' % str(model.predict(Z[comment_id,:]) >0.5))


# In[ ]:


comment_id = toxic_ids[2]
print('Content of the comment: \n%s\n' % train['comment_text'][training_set_size+comment_id])
print('Is this comment "toxic" according to the model?\n%s' % str(model.predict(Z[comment_id,:]) >0.5))


# **Working on Clean Comments**

# Now, we will create a new column for "clean" comments. These comments  correspond to none of the 6 categories as stated above.

# In[ ]:


rowsums=train.iloc[:,2:].sum(axis=1)
train['clean']=(rowsums==0)
train['clean'].sum()


# **Bar Charts : Showcasing the categories of toxic comments**

# In[ ]:


colors_list = ["brownish green", "pine green", "ugly purple",
               "blood", "deep blue", "brown", "azure"]

palette= sns.xkcd_palette(colors_list)

x=train.iloc[:,2:].sum()

plt.figure(figsize=(9,6))
ax= sns.barplot(x.index, x.values,palette=palette)
plt.title("Number per Class")
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Type ')
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 10, label, 
            ha='center', va='bottom')

plt.show()


# We have more than 1.4 lakhs clean comments. And, we can see that we have a very unbalanced dataset. Even non-clean comments are not equally reparted. This might be an issue whilel training the learning algorithms. This also means that if we predict 'clean' for each comment, our result will not be so bad in term of accuracy.
# 
# 

# In[ ]:


# A list that contains all the text data
comment_text_list = train.apply(lambda row : nltk.word_tokenize( row['comment_text']),axis=1)


# In[ ]:


comment_text_list.head()


# **Odd Comments : Containing a high rate of punctuation symbols or capital letters**

# In[ ]:


#An odd comment contains a high rate of punctuation symbols or capital letters
rate_punctuation=0.7
rate_capital=0.7
def odd_comment(comment):
    punctuation_count=0
    capital_letter_count=0
    total_letter_count=0
    for token in comment:
        if token in list(string.punctuation):
            punctuation_count+=1
        capital_letter_count+=sum(1 for c in token if c.isupper())
        total_letter_count+=len(token)
    return((punctuation_count/len(comment))>=rate_punctuation or 
           (capital_letter_count/total_letter_count)>rate_capital)

odd=comment_text_list.apply(odd_comment)


# In[ ]:


odd_ones=odd[odd==True]
odd_comments=train.loc[list(odd_ones.index)]
odd_comments[odd_comments.clean==False].count()/len(odd_comments)


# Hence, we could see that more than 65% of the so-called odd comments are not clean. It seems to be an interesting feature to add to the dataset. So, we will have to train a model for these specific odd comments that cannot be treated the same way as the normal or the clean ones. 

# **Bar Charts : Showcasing the categories of toxic comments based on Odd Comments**

# In[ ]:


colors_list = ["brownish green", "pine green", "ugly purple",
               "blood", "deep blue", "brown", "azure"]

palette= sns.xkcd_palette(colors_list)

x=odd_comments.iloc[:,2:].sum()


plt.figure(figsize=(9,6))
ax= sns.barplot(x.index, x.values, alpha=0.8, palette=palette)
plt.title("Number per category")
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Type ', fontsize=12)

rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

plt.show()


# In[ ]:


# A quick check for empty comments
empty_comments=train[train.comment_text==""]
empty_comments


# In[ ]:


# A quick check for duplicated comments
duplicate=train.comment_text.duplicated()
duplicate[duplicate==True]


# In[ ]:


# Storing each categories of non clean comments in specific arrays
toxic=train[train.toxic==1]['comment_text'].values
severe_toxic=train[train.severe_toxic==1]['comment_text'].values
obscene=train[train.obscene==1]['comment_text'].values
threat=train[train.threat==1]['comment_text'].values
insult=train[train.insult==1]['comment_text'].values
identity_hate=train[train.identity_hate==1]['comment_text'].values


# **Exploring toxic data from WordCloud Patterns**

# Wordclouds are a quick way to see which words are dominant in a text. Now, we will see that which words are the most dominated in toxic labeled comments.

# In[ ]:


from wordcloud import WordCloud, STOPWORDS
plt.figure(figsize=(16,13))
wc = WordCloud(background_color="black", max_words=500, stopwords=stopwords, max_font_size= 60)
wc.generate(" ".join(toxic))
plt.title("Wordlcloud Toxic Comments", fontsize=30)
plt.imshow(wc.recolor( colormap= 'Set1' , random_state=1), alpha=0.98)
plt.axis('off')
plt.savefig('Toxic_wc.png')


# It's time for some text processing now.

# **Lemmatization**

# <b>Lemmatization </b>usually refers to doing things properly with the use of a vocabulary and morphological analysis of words, normally aiming to remove inflectional endings only and to return the base or dictionary form of a word, which is known as the lemma .
# 
# For example : Lemmatization(drank)=drink
# 
# We would also be replacing apostrophe words like "won't" ==> will not, "can't"==> cannot., etc.
# 
# So, it's time to start with the <b>toxic category</b> :

# In[ ]:


replacement_patterns = [
 (r'won\'t', 'will not'),
 (r'can\'t', 'cannot'),
 (r'i\'m', 'i am'),
 (r'ain\'t', 'is not'),
 (r'(\w+)\'ll', '\g<1> will'),
 (r'(\w+)n\'t', '\g<1> not'),
 (r'(\w+)\'ve', '\g<1> have'),
 (r'(\w+)\'s', '\g<1> is'),
 (r'(\w+)\'re', '\g<1> are'),
 (r'(\w+)\'d', '\g<1> would')
]
class RegexpReplacer(object):
    def __init__(self, patterns=replacement_patterns):
         self.patterns = [(re.compile(regex), repl) for (regex, repl) in
         patterns]
     
    def replace(self, text):
        s = text
        for (pattern, repl) in self.patterns:
             s = re.sub(pattern, repl, s)
        return s


# In[ ]:


from nltk.stem import WordNetLemmatizer
lemmer = WordNetLemmatizer()
stopwords = nltk.corpus.stopwords.words('english')
from nltk.tokenize import TweetTokenizer
#from replacers import RegexpReplacer
replacer = RegexpReplacer()
tokenizer=TweetTokenizer()

def comment_process(category):
    category_processed=[]
    for i in range(category.shape[0]):
        comment_list=tokenizer.tokenize(replacer.replace(category[i]))
        comment_list_cleaned= [word for word in comment_list if ( word.lower() not in stopwords 
                              and word.lower() not in list(string.punctuation) )]
        comment_list_lemmed=[lemmer.lemmatize(word, 'v') for word in comment_list_cleaned]
        category_processed.extend(list(comment_list_lemmed))
    return category_processed


# In[ ]:


toxic1=comment_process(toxic)


# In[ ]:


fd=nltk.FreqDist(word for word in toxic1)

x=[fd.most_common(150)[i][0] for i in range(99)]
y=[fd.most_common(150)[i][1] for i in range(99)]

palette= sns.light_palette("crimson",100,reverse=True)
plt.figure(figsize=(45,15))
ax= sns.barplot(x, y, alpha=0.8,palette=palette)

plt.title("Occurences per word in Toxic comments 1", fontsize=40)
plt.ylabel('Occurrences', fontsize=30)
plt.xlabel(' Word ', fontsize=30)

# Adding the text labels
rects = ax.patches
labels = y
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
    plt.xticks(rotation=60, fontsize=18)
plt.show()


# The bar chart/plot can be opened in a new tab/window for a bigger and better view.
# 
# From the above plot, we could see that some of the toxic words have lots of occurences which do not have any existance  in the first wordcloud. 
# Hence, using the word cloud library on unprocessed data in the first place might be misleading. 
# 
# So, it seems that we have some more work to do on the data processing part.

# In[ ]:


def wordcloud_plot(category, name) : 
    plt.figure(figsize=(20,15))
    wc = WordCloud(background_color="black", max_words=500, min_font_size=6 
                 , stopwords=stopwords, max_font_size= 60)
    wc.generate(" ".join(category))
    plt.title("Twitter Wordlcloud " + name +  " Comments", fontsize=30)
    # plt.imshow(wc.recolor( colormap= 'Pastel1_r' , random_state=17), alpha=0.98)
    plt.imshow(wc.recolor( colormap= 'Set1' , random_state=21), alpha=0.98)
    plt.axis('off')
    plt.savefig(name+'_wc.png')
    return(True)

wordcloud_plot(toxic1,'Toxic')


# In[ ]:


severe_toxic1=comment_process(severe_toxic)
obscene1=comment_process(obscene)
threat1=comment_process(threat)
insult1=comment_process(insult)
identity_hate1=comment_process(identity_hate)


# In[ ]:


wordcloud_plot(severe_toxic1,'Severe_toxic')


# In[ ]:


wordcloud_plot(obscene1,'Obscene')


# In[ ]:


wordcloud_plot(threat1,'Threat')


# In[ ]:


wordcloud_plot(insult1,'Insult')


# In[ ]:


wordcloud_plot(identity_hate1,'Identity_Hate')


# **Conclusion from the above wordcloud patterns**

# Some categories(Wordcloud patterns) share the same vocabulary in terms of richness and word frequency, like Insult and Toxic categories. And some others, like Identity and Hate category, use specific words more frequently. 

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 1),
    max_features=20000)
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    ngram_range=(1, 4),
    max_features=20000)
char_vectorizer.fit(all_text)
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)

train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])

scores = []
submission = pd.DataFrame.from_dict({'id': test['id']})
for class_name in class_names:
    train_target = train[class_name]
    classifier = LogisticRegression(solver='sag')

    cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc'))
    scores.append(cv_score)
    print('CV score for class {} is {}'.format(class_name, cv_score))

    classifier.fit(train_features, train_target)
    submission[class_name] = classifier.predict_proba(test_features)[:, 1]

print('Total CV score is {}'.format(np.mean(scores)))

submission.to_csv('submission.csv', index=False)


# In[ ]:




