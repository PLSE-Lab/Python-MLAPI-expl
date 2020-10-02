#!/usr/bin/env python
# coding: utf-8

# # E Commerce Reviews 

# ### Inspiration:
# [Tutorials on Bag of words](https://www.kaggle.com/rochachan/bag-of-words-meets-bags-of-popcorn)
# 
# [Abhishek's Kernel on NLP](https://www.kaggle.com/abhishek/approaching-almost-any-nlp-problem-on-kaggle)
# 
# [Jeremy Howard's kernel on Naive Bayes](https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline)
# 
# [sban's advanced kernel on LSTM](https://www.kaggle.com/shivamb/beginners-guide-to-text-generation-using-lstms)
# 
# Many more kernels dealing with NLP.

# ### Objective:

# This is my first extensive kernel dealing with an text classification problem.Therefore I have tried my hands on whatever i have learnt so far on NLP.I try to do a data analysis and visualisation before creating a simple naive bayes model to predict review scores using the reviews.Thanks for reading through.Also check out my other kernel on [whiskey classification using reviews](https://www.kaggle.com/gsdeepakkumar/classy-whisky-approach-through-nlp)

# In[ ]:




# Loading the required libraries 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
import itertools
warnings.filterwarnings('ignore')
review =pd.read_csv("../input/Womens Clothing E-Commerce Reviews.csv")


# In[ ]:


review.head()


# In[ ]:


text = review[['Review Text','Rating']]
text.shape


# In[ ]:


text['Review Text'][0]
text[text['Review Text']==""]=np.NaN
text['Review Text'].fillna("No Review",inplace=True)


# In[ ]:


# Split into train and test data:
split = np.random.randn(len(text)) <0.8
train = text[split]
test = text[~split]
print("Total rows in train:",len(train),"and test:",len(test))
ytrain=train['Rating']
ytest=test['Rating']


# ### Examine the length of the comments:
# 

# In[ ]:


lens=train['Review Text'].str.len()
print("Mean Length:",lens.mean(),"Standard Deviation",lens.std(),"Maximum Length",lens.max())


# In[ ]:


lens.hist()


# We find that the length of the text varies.Let us see how the length is distributed for every rating .

# In[ ]:


plt.figure(figsize=(8,8))
text['Length']=lens
fx=sns.boxplot(x='Rating',y='Length',data=text)
plt.title("Distribution of length with respect to rating")
plt.xlabel("Rating")
plt.ylabel("Length")


# There seems to be a slight difference in the length of the reviews for different rating.

# We will now convert the text files into numerical vectors through **Bag of words** model.For this , first we will clean the reviews - remove stopwords as a baseline model.We can also look at removing punctuations,numbers.

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import log_loss,confusion_matrix,classification_report, accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import re


# In[ ]:


count_vect = CountVectorizer(analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), stop_words = 'english',max_features=5000)
count_vect.fit(list(train['Review Text'].values.astype('U'))+list(test['Review Text'].values.astype('U')))
xtrain=count_vect.transform(train['Review Text'].values.astype('U'))
xtest=count_vect.transform(test['Review Text'].values.astype('U'))


# Now we train naive bayes model on the data and look at the log loss value.

# In[ ]:


## Applying naive bayes:

model = MultinomialNB()
model.fit(xtrain, ytrain)
predictions = model.predict(xtest)


# ### Metrics:

# In[ ]:


### Lets check the accuracy score.
print(accuracy_score(ytest, predictions))


# In[ ]:


conf_matrix=confusion_matrix(ytest,predictions)


# The model has an accuracy of 62 %.

# In[ ]:


### Print confusion matrix:
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plt.figure(figsize=(8,8))
plot_confusion_matrix(conf_matrix, classes=['1', '2','3','4','5'],
                      title='Confusion matrix')
plt.show()


# **Thanks for reading my kernel.**
