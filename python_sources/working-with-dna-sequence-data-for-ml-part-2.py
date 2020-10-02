#!/usr/bin/env python
# coding: utf-8

# # Working with DNA sequence data for machine learning part 2 - clasification of gene function

# In the [previous notebook](https://www.kaggle.com/thomasnelson/working-with-dna-sequence-data-for-ml), I showed some simple ways to encode DNA sequences are ordinal vectors, one-hot encoded vectors or as counts of k-mers (using scikit learn's NLP tools).  In this notebook, I will take it to the next step and apply what we learned to build a classification model that can predict a gene's function based on the DNA sequence of the coding sequence alone.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Let's open the data for human and see what we have.

# In[ ]:


human = pd.read_table('../input/humandata/human_data.txt')
human.head()


# ### We have some data for human DNA sequence coding regions and a class label.  We also have data for Chimpanzee and a more divergent species, the dog.  Let's get that.

# In[ ]:


chimp = pd.read_table('../input/chimp-data/chimp_data.txt')
dog = pd.read_table('../input/dog-data/dog_data.txt')
chimp.head()
dog.head()


# ### Here are the definitions for each of the 7 classes and how many there are in the human training data.  They are gene sequence function groups.

# In[ ]:


from IPython.display import Image
Image("../input/image2/Capture1.PNG")


# ### Let's define a function to collect all possible overlapping k-mers of a specified length from any sequence string.

# In[ ]:


# function to convert sequence strings into k-mer words, default size = 6 (hexamer words)
def getKmers(sequence, size=6):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]


# ### Now we can convert our training data sequences into short overlapping  k-mers of legth 6.  Lets do that for each species of data we have using our getKmers function.

# In[ ]:


human['words'] = human.apply(lambda x: getKmers(x['sequence']), axis=1)
human = human.drop('sequence', axis=1)
chimp['words'] = chimp.apply(lambda x: getKmers(x['sequence']), axis=1)
chimp = chimp.drop('sequence', axis=1)
dog['words'] = dog.apply(lambda x: getKmers(x['sequence']), axis=1)
dog = dog.drop('sequence', axis=1)


# ### Now, our coding sequence data is changed to lowercase, split up into all possible k-mer words of length 6 and ready for the next step.  Let's take a look.

# In[ ]:


human.head()


# ### Since we are going to use scikit-learn natural language processing tools to do the k-mer counting, we need to now convert the lists of k-mers for each gene into string sentences of words that the count vectorizer can use.  We can also make a y variable to hold the class labels.  Let's do that now.

# In[ ]:


human_texts = list(human['words'])
for item in range(len(human_texts)):
    human_texts[item] = ' '.join(human_texts[item])
y_h = human.iloc[:, 0].values                         #y_h for human


# In[ ]:


human_texts[0]


# In[ ]:


y_h


# ### Now let's do the same for chimp and dog.

# In[ ]:


chimp_texts = list(chimp['words'])
for item in range(len(chimp_texts)):
    chimp_texts[item] = ' '.join(chimp_texts[item])
y_c = chimp.iloc[:, 0].values                       # y_c for chimp

dog_texts = list(dog['words'])
for item in range(len(dog_texts)):
    dog_texts[item] = ' '.join(dog_texts[item])
y_d = dog.iloc[:, 0].values                         # y_d for dog


# ## Now let's review how to use sklearn's "Natural Language" Processing tools to convert our k-mer words into uniform length numerical vectors that represent counts for every k-mer in the vocabulary.

# In[ ]:


# Creating the Bag of Words model using CountVectorizer()
# This is equivalent to k-mer counting
# The n-gram size of 4 was previously determined by testing
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range=(4,4))
X = cv.fit_transform(human_texts)
X_chimp = cv.transform(chimp_texts)
X_dog = cv.transform(dog_texts)


# ### Let's see what we have...  for human we have 4380 genes converted into uniform length feature vectors of 4-gram k-mer (length 6) counts.  For chimp and dog we have the expected same number of features with 1682 and 820 genes respectively.

# In[ ]:


print(X.shape)
print(X_chimp.shape)
print(X_dog.shape)


# ### If we have a look at class balance we can see we have relatively balanced dataset.

# In[ ]:


human['class'].value_counts().sort_index().plot.bar()


# In[ ]:


chimp['class'].value_counts().sort_index().plot.bar()


# In[ ]:


dog['class'].value_counts().sort_index().plot.bar()


# ### So now that we know how to transform our DNA sequences into uniform length numerical vectors in the form of k-mer counts and ngrams, we can now go ahead and build a classification model that can predict the DNA sequence function based only on the sequence itself.
# 
# ### Here I will use the human data to train the model, holding out 20% of the human data to test the model.  Then we can really challenge the model's generalizability by trying to predict sequence function in other species (the chimpanzee and dog).
# 
# ### So below we will - 1: train/test spit.  2: Build simple multinomial naive Bayes classifier and 3: test the model performance.

# In[ ]:


# Splitting the human dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y_h, 
                                                    test_size = 0.20, 
                                                    random_state=42)


# In[ ]:


print(X_train.shape)
print(X_test.shape)


# ### A multinomial naive Bayes classifier will be created.  I previously did some parameter tuning and found the ngram size of 4 (reflected in the Countvectorizer() instance) and a model alpha of 0.1 did the best.  Just to keep it simple I won't show that code here.

# In[ ]:


### Multinomial Naive Bayes Classifier ###
# The alpha parameter was determined by grid search previously
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB(alpha=0.1)
classifier.fit(X_train, y_train)


# ### Now let's make predictions on the human hold out test set and see how it performes on unseen data.

# In[ ]:


y_pred = classifier.predict(X_test)


# ### Okay, so let's look at some model performce metrics like the confusion matrix, accuracy, precision, recall and f1 score.  We are getting really good results on our unseen data, so it looks like our model did not overfit to the training data.  In a real project I would go back and sample many more train test splits since we have a relatively small data set.

# In[ ]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
print("Confusion matrix\n")
print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))
def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))


# ### Now for the real test.  Let's see how our model perfoms on the DNA sequences from other species.  First we'll try the Chimpanzee, which we would expect to be very similar to human.  Then we will try man's (and woman's) best friend, the Dog DNA sequences. 

# ### Make predictions for the Chimp and dog sequences

# In[ ]:


# Predicting the chimp, dog and worm sequences
y_pred_chimp = classifier.predict(X_chimp)
y_pred_dog = classifier.predict(X_dog)


# ### Now, let's examine the typical performace metrics for each species.

# In[ ]:


# performance on chimp genes
print("Confusion matrix\n")
print(pd.crosstab(pd.Series(y_c, name='Actual'), pd.Series(y_pred_chimp, name='Predicted')))
accuracy, precision, recall, f1 = get_metrics(y_c, y_pred_chimp)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))


# In[ ]:


# performance on dog genes
print("Confusion matrix\n")
print(pd.crosstab(pd.Series(y_d, name='Actual'), pd.Series(y_pred_dog, name='Predicted')))
accuracy, precision, recall, f1 = get_metrics(y_d, y_pred_dog)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))


# ### The model seems to perform well on human data.  It also does on Chimpanzee.  That might not be a surprize since the chimp and human are so similar genetically.  The performance on dog is not quite as good.  We would expect this since the dog is more divergent from human than the chimpanze.
