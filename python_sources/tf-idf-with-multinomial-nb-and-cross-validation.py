#!/usr/bin/env python
# coding: utf-8

# #### Part 1: Words preprocessing 
# #### The following code (using the entire data set as an example) shows how stop words and punctuations removal and word stemming work by illustrating them using a few text samples. The steps by which vocabulary count (tf), idf, and the final tf-idf expression are calculated (using "CountVectorizer" and "TfidfTransformer") are also demonstrated using a text sample.
# #### Part 2: Modeling 
# #### The data is split into a 80% training set and a 20% test set. "GridSearchCV" with 10-folds cross-validation with a "Pipeline" as its estimator consisting of "CountVectorizer", "TfidfTransformer", and the classifier "MultinomialNB" is used to tune the hyperparameter alpha value in "MultinomialNB". When alpha = 0.08, the score (accuracy) of the model "MultinomialNB" on the test data set (split using a random number of 42) is about 98.3 %

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
from time import time
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords


# In[ ]:


# import nltk
# nltk.download()
# (The above is the code that may be needed the first time nltk libraries are downloaded.)


# In[ ]:


data = pd.read_csv("../input/sms-spam-collection-dataset/spam.csv", encoding='latin-1')
data.head()


# In[ ]:


data.info()


# #### Drop Unnamed columns.

# In[ ]:


data = data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
data.head()


# #### Change column names.

# In[ ]:


data.columns = ['class', 'text']
data.head()


# In[ ]:


count_class = pd.value_counts(data["class"])
count_class


# #### (The data has 4825 ham samples and 747 spam samples.)
# ### Part 1: Words preprocessing (using the entire data set as an example)
# #### Now look at a few samples of "text".

# In[ ]:


for i in range(5): 
    print(data['text'][i])


# #### Use "stopwords" from "nltk.corpus" to remove stop words. Use one text sample (the 5th sample) to see how it works.

# In[ ]:


stopwords_list = stopwords.words('english')
stopwords_list


# In[ ]:


data['text'][4]


# In[ ]:


words_new = [i for i in data['text'][4].split() if i.lower() not in stopwords.words('english')]
' '.join(words_new)


# #### It can be seen that the stop words "I", "don't", "to", "here", and two "he" are removed. Next use the characters in "string.punctuation" to remove punctuations in a text sample. Use another text sample (the 3rd text sample) to see how it works.

# In[ ]:


punc_string = string.punctuation
print(punc_string)
print(len(punc_string))


# In[ ]:


data['text'][2]


# In[ ]:


data['text'][2].translate(str.maketrans(' ', ' ', punc_string))  


# #### The above code seems to delete punctuations but concatenate the two words surrounding a punctuation into one word. It seems beneficial to insert a space to keep the words apart.

# In[ ]:


data['text'][2].translate(str.maketrans(punc_string, len(punc_string)*' '))


# #### It can be seen that the two words surrounding a punctuation are not concatenated anymore. Next use "SnowballStemmer" from "nltk" to perform word stemming. Use two text samples to show how it works.

# In[ ]:


print(data['text'][3])
print(data['text'][4])


# In[ ]:


stemmer = SnowballStemmer('english')
print(' '.join([stemmer.stem(word) for word in data['text'][3].split()]))
print(' '.join([stemmer.stem(word) for word in data['text'][4].split()]))


# #### The stemming changes "early" and "already" to "earli" and "alreadi", respectively, and "goes" to "goe" and "lives" to "live". It also changes capital U to lower case u, "Nah" to "nah". 
# #### Next create a function to combine the preprocessing steps of first removing stop words, then removing punctuations, and finally performing word stemming.

# In[ ]:


def preprocess(text):
    # Remove stopwords
    words = [i for i in text.split() if i.lower() not in stopwords_list]
    text_1 = ' '.join(words)
    # Remove puctuation and replace with a space.
    text_2 = text_1.translate(str.maketrans(punc_string, len(punc_string)*' '))
    words_1 = text_2.split()
    # Perform word stemming 
    words_2 = [stemmer.stem(word) for word in words_1]
    text_3 = ' '.join(words_2)
    return text_3


# #### Check how the function "preprocess" works in one text sample.

# In[ ]:


print(data['text'][4])
preprocess(data['text'][4])


# #### The results look good. Now apply the function to all text samples.

# In[ ]:


data['text'] = data['text'].apply(preprocess)


# #### Now use "CountVectorizer" to learn the vocabulary dictionary and to return the term-document matrix.

# In[ ]:


data.head(10)


# In[ ]:


count_v = CountVectorizer()
word_count_matrix = count_v.fit_transform(data['text'])
word_count_matrix


# In[ ]:


print(word_count_matrix)


# #### The above shows the sparse representation of the counts. Now use ".toarray" to change the sparse representation to an array and use "get_feature_names()" to match the counts and the corresponding words. Show the 30 most frequently occurring vocabulary.

# In[ ]:


count_list = word_count_matrix.toarray().sum(axis=0)
word_list = count_v.get_feature_names()


# In[ ]:


word_freq = pd.DataFrame(count_list, index=word_list, columns=['Freq'])
word_freq.sort_values(by='Freq', ascending=False).head(30)


# ####  Now use the 24th text sample as an example to show how the words counts of this text sample can be shown from the results of "CountVectorizer".

# In[ ]:


data['text'][23]


# In[ ]:


text_freq = pd.DataFrame(word_count_matrix.toarray()[23], index=count_v.get_feature_names(), columns=['word freq'])
# remove the rows with 0 frequency count
text_freq = text_freq[text_freq['word freq']!=0]
text_freq


# #### Note that "CountVectorizer" removes the one-letter word "u" and the number "3" from the text. 
# #### Use "TfidfTransformer" to learn the idf vector (by "fit") from the count matrix.

# In[ ]:


tf_idf = TfidfTransformer()
tf_idf.fit(word_count_matrix)
tf_idf.idf_


# In[ ]:


tf_idf.idf_.shape


# In[ ]:


idf = pd.DataFrame(tf_idf.idf_, index=count_v.get_feature_names(), columns=['idf_weight'])
idf.sort_values(by='idf_weight')


# In[ ]:


print('max = ' + str(tf_idf.idf_.max()))
print('min = ' + str(tf_idf.idf_.min()))
print('mean = ' + str(tf_idf.idf_.mean()))


# #### From the closeness of the mean idf (8.3) and the max idf (8.9) it can be inferred that most words only sparsely occurring in text samples with similar frequencies. (idf would be close to 1 if a word occurs in almost all samples). 
# #### Previously the term frequency (as the DataFrame "text_freq") of the 24th text sample have been found. Now let's find the itf values of the vocabulary in the 24th text sample.

# In[ ]:


word_list_to_drop =[i for i in idf.index if i not in text_freq.index]
# (the list of words that need to be dropped)
idf_1 = idf.drop(word_list_to_drop)
idf_1


# #### Use "TfidfTransformer" to transform (use "transform") the count matrix to a tf-idf representation and output the tf-idf representation of the 24th text sample.

# In[ ]:


tf_idf_vector = tf_idf.transform(word_count_matrix)
tf_idf_vector


# In[ ]:


tf_idf_1 = pd.DataFrame(tf_idf_vector.toarray()[23], index=count_v.get_feature_names(), columns=['tf_idf'])
# remove the rows with tf-idf = 0
tf_idf_1 = tf_idf_1[tf_idf_1['tf_idf']!=0.0]
tf_idf_1


# #### Cancatenate the tf, idf, and tf-idf DataFrames of the 24th text sample together into one DataFrame.

# In[ ]:


df = pd.concat([text_freq, idf_1, tf_idf_1], axis=1)
df


# #### Add a new column that is the multiplication of "word freq" and "idf_weight".

# In[ ]:


df['(word freq)x(idf_weight)'] = df['word freq'] * df['idf_weight']
df = df.reindex(['word freq', 'idf_weight', '(word freq)x(idf_weight)', 'tf_idf'], axis=1)
df


# #### The following code illustrates that tf-idf value is calculated by first multiplying "word freq" (tf) and "idf_weight" (idf) and then perform l2 normalization on the resulting vector. 

# In[ ]:


df_1 = pd.DataFrame(df['(word freq)x(idf_weight)']).T
# (Perform a transpose of the "(word freq)x(idf_weight)" column)
df_1


# In[ ]:


# Perform l2 normalization on the "(word freq)x(idf_weight)" vector using the "normalize" function.
normalize(df_1)


# In[ ]:


np.sum(np.square(normalize(df_1)))


# #### It can be seen that the "normalize(df_1)" values are the same as the "tf_idf" column of the "df" DataFrame and the sum of squares of its vector elements is 1.0, which indicates l2 normalization. 
# ### Part 2: Modeling
# #### Split the original preprocessed data into training and test sets.

# In[ ]:


data.head()


# In[ ]:


X = data['text']
y = data['class']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ####  Create a Pipeline consisting of "CountVectorizer", "TfidfTransformer", and the classifier "MultinomialNB". Fit the Pipeline using the training data set, where 
# #### 1. The text (X_train) is fit and transformed first using CountVectorizer(), then,
# #### 2. The results are fit and transformed using TfidfTransformer(), and, 
# #### 3. Finally MultinomialNB() is fit using the transformed results and the labels (y_train) using the default hyperparameter alpha value of 1.0.

# In[ ]:


pipe = Pipeline([
    ('vector', CountVectorizer()), 
    ('tfidf', TfidfTransformer()), 
    ('mulNB', MultinomialNB())
])


# In[ ]:


pipe.fit(X_train, y_train)


# #### Call the "score" method of pipeline, where,
# #### 1. the test data (X_test) is first transformed (not fit) using CountVectorizer(), which was fit earlier using X_train, and then transformed (not fit) using TfidfTransformer(), which was fit earlier using the output of CountVectorizer() trained using X_train), and then,
# #### 2. the "score" methed of  MultinomialNB() is called to calculate the model accuracy on the transformed test data and labels (y_test).
# 
# #### (Step 1. above ensures that dimensions are matched when the "score" and "predict" metheds of  MultinomialNB() are called using X_test.)

# In[ ]:


pipe.score(X_test, y_test)


# #### Tune the hyperparameter "alpha" in MultinomialNB() using "GridSearchCV" with Pipeline as its estimator,
# #### 1. GridSearchCV() is fit using X_train and y_train, where,
# #### 2. The Pipeline is fit using the training folds of a split, then,
# #### 3. The "score" methed of pipeline is called where the validation fold of the same split is transformed (not fit) using CountVectorizer() and TfidfTransformer() sequentially, and the "score" methed of MultinomialNB() is called to calculate the model accuracy on the transformed validation fold test data and labels.
# #### 4. The scores of all splits are compared and the hyperparameter in the split with the highest score is chosen.
# #### Finally the "score" method of GridSearchCV() is called to calculate the model accuracy on the given test data (X_test) and labels (y_test).  

# In[ ]:


parameters = {
    'mulNB__alpha': [1, 0.7, 0.4, 0.2, 0.1, 0.09, 0.08, 0.07, 0.06, 0.03, 0.01] 
}

grid = GridSearchCV(pipe, param_grid=parameters, cv=10, refit=True)
t0 = time()
grid.fit(X_train, y_train)
print("done in %0.3fs" % (time() - t0))


# In[ ]:


print(grid.best_score_)
print(grid.best_params_)


# In[ ]:


grid.score(X_test, y_test)


# #### When alpha = 0.08, the score (accuracy) of the model "MultinomialNB" on the test data set (split using a random number of 42) is about 98.3 %, which improves from the score of about 96.7% when alpha = 1.0. 

# In[ ]:




