#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # 1. Load Libraries

# In[ ]:


import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

# sklearn for feature extraction & modeling
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
# Iteratively read files
import glob
import os

# For displaying images in ipython
import seaborn as sns
sns.set(color_codes = True)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


get_ipython().system(' pip install joblib')


# In[ ]:


df = pd.read_csv('/kaggle/input/amazon-fine-food-reviews/Reviews.csv')
df.head()


# In[ ]:


from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# In[ ]:


heading_1 = df[df["Score"]==1]["Summary"] # Extract only sumary of reviws
collapsed_heading_1 = heading_1.str.cat(sep=' ')

heading_2 = df[df["Score"]==2]["Summary"] # Extract only sumary of reviws
collapsed_heading_2 = heading_2.str.cat(sep=' ')

heading_3 = df[df["Score"]==3]["Summary"] # Extract only sumary of reviws
collapsed_heading_3 = heading_3.str.cat(sep=' ')

heading_4 = df[df["Score"]==4]["Summary"] # Extract only sumary of reviws
collapsed_heading_4 = heading_4.str.cat(sep=' ')

heading_5 = df[df["Score"]==5]["Summary"] # Extract only sumary of reviws
collapsed_heading_5 = heading_4.str.cat(sep=' ')


# In[ ]:


# Create stopword list:
stopwords = set(STOPWORDS)
#stopwords.update(["Subject","re","fw","fwd"])

print("Word Cloud for Sports")

# Generate a word cloud image
wordcloud = WordCloud(stopwords=stopwords, background_color="white",max_words=50).generate(collapsed_heading_1)

# Display the generated image:
# the matplotlib way:1
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

print("\nWord Cloud for Business")

# Generate a word cloud image
wordcloud = WordCloud(stopwords=stopwords, background_color="white",max_words=50).generate(collapsed_heading_2)

# Display the generated image:
# the matplotlib way:1
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

print("\nWord Cloud for Politics")
# Generate a word cloud image
wordcloud = WordCloud(stopwords=stopwords, background_color="white",max_words=50).generate(collapsed_heading_3)

# Display the generated image:
# the matplotlib way:1
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

print("\nWord Cloud for Technology")

# Generate a word cloud image
wordcloud = WordCloud(stopwords=stopwords, background_color="white",max_words=50).generate(collapsed_heading_4)

# Display the generated image:
# the matplotlib way:1
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
     
print("\nWord Cloud for Entertainment")
# Generate a word cloud image
wordcloud = WordCloud(stopwords=stopwords, background_color="white",max_words=50).generate(collapsed_heading_5)

# Display the generated image:
# the matplotlib way:1
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# # Building Pipeline

# In[ ]:


# Building Pipeline for raw text transformation
clf = Pipeline([
    ('vect', CountVectorizer(stop_words= "english")),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB()),
    ])


# In[ ]:


df_1 = df.dropna()
df_1.shape, df.shape


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df_1["Summary"], df_1["Score"],random_state = 42,
                                                   test_size = 0.20)
X_train.shape,X_test.shape,y_train.shape


# # Train the mode

# In[ ]:


model = clf.fit(X_train,y_train)


# In[ ]:


print("Accuracy of Naive Bayes Classifier is {}".format(model.score(X_test,y_test)))


# In[ ]:


# Predict on Test data
y_predicted = model.predict(X_test)
y_predicted[0:10]


# In[ ]:


#Confusion Matrix
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_predicted)
np.set_printoptions(precision=2)
cnf_matrix


# In[ ]:


import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# In[ ]:


#With Normalization
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['sport','business','politics','tech','entertainment'],
                      title='Confusion matrix, without normalization')
# With normalization
plt.figure()
plot_confusion_matrix(cnf_matrix, classes= ['sport','business','politics','tech','entertainment'], normalize=True,title='Normalized confusion matrix')

plt.show()


# In[ ]:


df_1.Score.value_counts()


# In[ ]:


# Building Pipeline for raw text transformation
clf = Pipeline([
    ('vect', CountVectorizer(stop_words= "english")),
    ('tfidf', TfidfTransformer()),
    ('classifier', RandomForestClassifier()),
    ])


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df_1["Summary"], df_1["Score"],random_state = 42,
                                                   test_size = 0.20)
X_train.shape,X_test.shape,y_train.shape


# In[ ]:


model = clf.fit(X_train,y_train)


# In[ ]:


print("Accuracy of Naive Bayes Classifier is {}".format(model.score(X_test,y_test)))


# In[ ]:


#With Normalization
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['sport','business','politics','tech','entertainment'],
                      title='Confusion matrix, without normalization')
# With normalization
plt.figure()
plot_confusion_matrix(cnf_matrix, classes= ['sport','business','politics','tech','entertainment'], normalize=True,title='Normalized confusion matrix')

plt.show()


# In[ ]:


# Predict on Test data
y_predicted = model.predict(X_test)
y_predicted[0:10]

#Confusion Matrix
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_predicted)
np.set_printoptions(precision=2)
cnf_matrix

#With Normalization
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['sport','business','politics','tech','entertainment'],
                      title='Confusion matrix, without normalization')
# With normalization
plt.figure()
plot_confusion_matrix(cnf_matrix, classes= ['sport','business','politics','tech','entertainment'], normalize=True,title='Normalized confusion matrix')

plt.show()


# In[ ]:




