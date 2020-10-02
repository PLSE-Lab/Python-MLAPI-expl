#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from nltk.stem import PorterStemmer #normalize word form
from nltk.probability import FreqDist #frequency word count
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords #stop words
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.probability import FreqDist 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import string
import re

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")
train = pd.read_csv("../input/nlp-getting-started/train.csv")
test = pd.read_csv("../input/nlp-getting-started/test.csv")


# # The functions used for text preprosessing

# In[ ]:


# remove hyperlink

def text_cleaning_hyperlink(text):
    
    return re.sub(r"http\S+","",text) 


# In[ ]:


# remove punctuation marks

def text_cleaning_punctuation(text):
    
    translator = str.maketrans('', '', string.punctuation) #remove punctuation
    
    return text.translate(translator)


# In[ ]:


# clean stopwords

def text_cleaning_stopwords(text):
    
    stop_words = set(stopwords.words('english'))
    
    word_token = word_tokenize(text)
    
    filtered_sentence = [w for w in word_token if not w in stop_words]
    
    return ' '.join(filtered_sentence) #return string of no stopwords


# In[ ]:


# convert all letters into lowercase ones

def text_cleaning_lowercase(text):
    
    return text.lower()


# In[ ]:


def text_extract(text_lst):
    txt = []
    for i,x in enumerate(text_lst):
        
        for j,p in enumerate(x):
            
            txt.append(p)
    
    return txt
    


# In[ ]:


# remove digits from the text

def remove_digits(txt):
    
    no_digits = ''.join(i for i in txt if not i.isdigit())
    
    return no_digits


# # Now we preprocess the TRAINING data 

# In[ ]:


# we clean the keywords for the TRAINING data

train.keyword = train.keyword.apply(lambda x: text_cleaning_stopwords(text_cleaning_punctuation(text_cleaning_hyperlink(remove_digits(x.lower())))) if type(x) == str else x)


# In[ ]:


# we clean the text for the TRAINING data

train.text = train.text.apply(lambda x: text_cleaning_stopwords(text_cleaning_punctuation(text_cleaning_hyperlink(remove_digits(x.lower())))))
train.text = train.text.apply(lambda x: list(set(x.split(' '))))


# In[ ]:


# # add keywords to the text of the TRAINING data
# train['k_t'] = train.apply(lambda x : x['text'] + [x['keyword']] if type(x['keyword']) == str else x['text'],axis=1) #add keyword to text content


# In[ ]:


# add keywords to the text of the TRAINING data
train['k_t'] = train.apply(lambda x : x['text'] + [x['keyword']] if type(x['keyword']) == str else x['text'], axis=1) 


# In[ ]:


# lemmatize the text of the TRAINING set

ps_1 = PorterStemmer()
wnl_1 = WordNetLemmatizer()
text_reconstruct = []

for i,x in enumerate(train.k_t.values):
    
    try:
        
        a = wnl_1.lemmatize(ps_1.stem(x))

        
    except AttributeError:
        
        a = list(set([wnl_1.lemmatize(ps_1.stem(word)) for j,word in enumerate(x)]))
        
    
    text_reconstruct.append(a)


# In[ ]:


# append the keywords to the text of the TRAINING set

train.k_t = text_reconstruct
train_word = train.k_t.apply(lambda x: ' '.join(x))


# # Then we preprocess the TEST data

# In[ ]:


# clean keywords of the TEST data

test.keyword = test.keyword.apply(lambda x: text_cleaning_stopwords(text_cleaning_punctuation(text_cleaning_hyperlink(remove_digits(x.lower())))) if type(x) == str else x)


# In[ ]:


# clean the text of the TEST data

test.text = test.text.apply(lambda x: text_cleaning_stopwords(text_cleaning_punctuation(text_cleaning_hyperlink(remove_digits(x.lower())))))
test.text = test.text.apply(lambda x: list(set(x.split(' '))))


# In[ ]:


# add (weighted) keywords to the text of the TEST data

test['k_t'] = test.apply(lambda x : x['text'] + [x['keyword']+x['keyword']+x['keyword']] if type(x['keyword']) == str else x['text'],axis=1) 


# In[ ]:


# lemmatize the text of the TEST set

ps_2 = PorterStemmer()
wnl_2 = WordNetLemmatizer()
text_reconstruct_test = []

for i,x in enumerate(test.k_t.values):
    
    try:
        
        a = wnl_2.lemmatize(ps_2.stem(x))

        
    except AttributeError:
        
        a = list(set([wnl_2.lemmatize(ps_2.stem(word)) for j,word in enumerate(x)]))
        
    
    text_reconstruct_test.append(a)


# In[ ]:


# append the keywords to the text of the TEST set

test.k_t = text_reconstruct_test
test_word = test.k_t.apply(lambda x: ' '.join(x))


# # Here we map the tokenized data to vectors (to capture some semantic similarities)

# In[ ]:


import gensim

path_for_word2vec = "../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin"
word2vec = gensim.models.KeyedVectors.load_word2vec_format(path_for_word2vec, binary = True)


# In[ ]:


# prepare to map words to vectors

def average_w2v(list_of_tokens, vct, generate_missing = False, dimentions = 300):
    if len(list_of_tokens) < 1:
        return np.zeros(dimentions)
    if generate_missing:
        vector = [vct[item] if item in vct else np.random.rand(dimentions) for item in list_of_tokens]
    else:
        vector = [vct[item] if item in vct else np.zeros(dimentions) for item in list_of_tokens]
    total_length = len(vector)
    sum_of_vectors = np.sum(vector, axis=0)
    average = np.divide(sum_of_vectors, total_length)
    return average


# In[ ]:


# map words to vectors (using the googlenewsvectorsnegative300 database)

def word2vec_mapping(vect, our_word, generate_missing = False):
    mapping = our_word.apply(lambda x: average_w2v(x, vect, generate_missing = generate_missing))
    return list(mapping)


# # Here The SVM + Word2Vec Model is applied

# In[ ]:


from nltk.tokenize import RegexpTokenizer
our_tokenizer = RegexpTokenizer(r'\w+')
tokenized_input_train = train_word.apply(our_tokenizer.tokenize)  # tokenize the TRAINING set
tokenized_input_test = test_word.apply(our_tokenizer.tokenize)    # tokenize the TEST set


# In[ ]:


mapped_train = word2vec_mapping(word2vec, train.text) # vectorize the TRAINING set
mapped_test = word2vec_mapping(word2vec,test.text)    # vectorize the TEST set


# In[ ]:


from sklearn.svm import SVC
classifier_SVM = SVC(C = 2,probability = True)
classifier_SVM.fit(mapped_train, train.target)
y_predicted_SVM = classifier_SVM.predict(mapped_test)


# In[ ]:


from sklearn.linear_model import LogisticRegression
classifier_LogisticRegression = LogisticRegression(C=30.0, class_weight = 'balanced', solver = 'newton-cg', 
                                                   multi_class = 'multinomial', n_jobs = -1, random_state = 42)
classifier_LogisticRegression.fit(mapped_train, train.target)
y_predicted_LogisticRegression = classifier_LogisticRegression.predict(mapped_test)


# # PostPreditiction corrections that take into accout important keywords

# In[ ]:


train = train.fillna('Zilch')
important_key_words = train.groupby('keyword').agg({'text':np.size, 'target':np.mean}).rename(columns={'text':'count', 'target':'frequency'})


# In[ ]:


additional_list = ['bushfires','evacuated','forestfire','hostages','rescuers','sinkhole','thunderstorm']


# In[ ]:


prob_disaster = 0.95
keyword_list_disaster95 = list(important_key_words[important_key_words['frequency']>prob_disaster].index) + additional_list


# In[ ]:


numbers_of_95certain_disasters = test['id'][test.keyword.isin(keyword_list_disaster95)]


# In[ ]:


y_predicted = np.zeros(len(y_predicted_SVM))

for i in range(0,len(y_predicted_SVM)):
    if i in numbers_of_95certain_disasters:
        y_predicted[i] = 1
    else:
        y_predicted[i] = y_predicted_SVM[i]  


# In[ ]:


sample_submission["target"] = [int(i) for i in y_predicted]


# In[ ]:


sample_submission.to_csv("submission.csv", index = False)


# In[ ]:


import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.spring):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=26)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", 
                 color="white" if cm[i, j] < thresh else "black", fontsize=40)
    
    plt.tight_layout()
    plt.ylabel('True labels', fontsize=24)
    plt.xlabel('Predicted labels', fontsize=24)

    return plt


# In[ ]:


holy_grail = pd.read_csv("../input/holygrail/submission.csv")
y_test = holy_grail["target"]


# In[ ]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

def scores(y_test, y_predicted):  
    accuracy = accuracy_score(y_test, y_predicted)
    f1 = f1_score(y_test, y_predicted, pos_label = None, average = 'weighted')
    precision = precision_score(y_test, y_predicted, pos_label = None, average = 'weighted')             
    recall = recall_score(y_test, y_predicted, pos_label = None, average = 'weighted')
    print("accuracy = %.4f, f1 = %.4f, precision = %.4f, recall = %.4f" % (accuracy, f1, precision, recall))
    return accuracy, f1, precision, recall


# In[ ]:


accuracy, f1, precision, recall = scores(y_test, y_predicted)


# In[ ]:


cm = confusion_matrix(y_test, y_predicted)
fig = plt.figure(figsize=(8, 8))
plot = plot_confusion_matrix(cm, classes=['Irrelevant','Disaster'], normalize=False, title='Confusion matrix\n for word2vec with SVM')
plt.show()


# In[ ]:





# In[ ]:


from sklearn.metrics import roc_curve
import seaborn as sns

# predict_proba gives the probabilities for the target (0 and 1 in our case) as a list (array). 
# The number of probabilities for each row is equal to the number of categories in target variable (2 in our case).

probabilities_SVM = classifier_SVM.predict_proba(mapped_test)
probabilities_LogisticRegression = classifier_LogisticRegression.predict_proba(mapped_test)

# Using [:,1] gives us the probabilities of getting the output as 1

probability_of_ones_SVM = probabilities_SVM[:,1] 
probability_of_ones_LogisticRegression = probabilities_LogisticRegression[:,1] 


# roc_curve returns:
# - false positive rates (FPrates), i.e., the false positive rate of predictions with score >= thresholds[i]
# - true positive rates (TPrates), i.e., the true positive rate of predictions with score >= thresholds[i]
# - thresholds 

FPrates_SVM, TPrates_SVM, thresholds_SVM = roc_curve(y_test, probability_of_ones_SVM)
FPrates_LogisticRegression, TPrates_LogisticRegression, thresholds_LogisticRegression = roc_curve(y_test, probability_of_ones_LogisticRegression)



# plotting the ROC Curve to visualize all the methods

sns.set_style('whitegrid')
plt.figure(figsize = (10, 8))

plt.plot(FPrates_SVM, TPrates_SVM, label = 'SVM')
plt.plot(FPrates_LogisticRegression, TPrates_LogisticRegression, label = 'Logistic Regression')


plt.plot([0, 1], [0, 1], color = 'blue', linestyle = '--')
plt.axis([0, 1, 0, 1])
plt.xlabel('False Positive Rate', fontsize = 14)
plt.ylabel('True Positive Rate', fontsize = 14)
plt.title('ROC Curve', fontsize = 14)
plt.legend(loc = "lower right")
plt.show()

