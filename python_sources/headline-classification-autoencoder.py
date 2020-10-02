#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# In this kernel we will build an autoencoder to discriminate between real and fake news headlines. The dataset we will use is [A Million News Headlines](https://www.kaggle.com/therohk/million-headlines), which contains real news headlines. For evaluation, we will use [Fake News Net](https://github.com/KaiDMML/FakeNewsNet) and [Fake News Dataset](http://web.eecs.umich.edu/~mihalcea/downloads.html#FakeNews). These datasets contain both real and fake headlines. After performing some basic pre-processing, we will train an autoencoder to represent real headlines. To perform classification, we will reconstruct inputs and calculate the overall reconstruction error. If the error is above a certain threshold, the item will be classified as fake, otherwise as real. To evaluate the network's performance, we will measure percentile and F1-score accuracy metrics.

# ## Reading Data
# 
# First we are going to read the evaluation data for real and fake headlines, and then we are going to concatenate the two dataframes.

# In[ ]:


import pandas as pd

d_fake = pd.read_csv('../input/fake-news-data/fnn_politics_fake.csv')
headlines_fake = d_fake.drop(['id', 'news_url', 'tweet_ids'], axis=1).rename(columns={'title': 'headline'})
headlines_fake['fake'] = 1

d_real = pd.read_csv('../input/fake-news-data/fnn_politics_real.csv')
headlines_real = d_real.drop(['id', 'news_url', 'tweet_ids'], axis=1).rename(columns={'title': 'headline'})
headlines_real['fake'] = 0

eval_data = pd.concat([headlines_fake, headlines_real])


# Next we will read fake and real headlines from the Fake News Dataset.

# In[ ]:


import os

def read_data(d):
    """Each file has a headline as the first line, followed by some white space and then the article content.
    We need to exract the headline and the content of each file and store them in lists."""
    files = os.listdir(d)
    headlines, contents = [], []
    for fname in files:
        if fname[:5] != 'polit':
            continue
        
        f = open(d + '/' + fname)
        text = f.readlines()
        f.close()

        if len(text) == 2:
            # One of the lines is missing
            if len(text[1]) <= 1:
                # There is no article content or headline
                continue
        elif len(text) >= 3:
            # More than one empty line encountered
            text[1] = text[-1]
        else:
            # Only one or zero lines is file
            continue
        
        headline, content = text[0][:-1].strip().rstrip(), text[1][:-1]
        headlines.append(headline)
        contents.append(content)
    
    return headlines, contents


fake_dir = '../input/fake-news-data/fnd_news_fake'
fake_headlines, fake_content = read_data(fake_dir)
fake_headlines = pd.DataFrame(fake_headlines, columns=['headline'])
fake_headlines['fake'] = 1

real_dir = '../input/fake-news-data/fnd_news_real'
real_headlines, real_content = read_data(real_dir)
real_headlines = pd.DataFrame(real_headlines, columns=['headline'])
real_headlines['fake'] = 0


# We will now concatenate these two new datasets into an evaluation dataset.

# In[ ]:


eval_data = pd.concat([eval_data, fake_headlines, real_headlines])
eval_data['fake'].value_counts()
eval_data.head()


# Now let's read our training data:

# In[ ]:


all_news = pd.read_csv('../input/all-the-news/articles3.csv', nrows=300000)
all_news = all_news.rename(columns={'title': 'headline'})
all_news['fake'] = 0
data = all_news[['headline', 'fake']]

# data = pd.concat([data, all_news])
data.head()


# ## Data Processing
# 
# We also need to format the data. We will split the dataset into features `X` and target `Y`. For `Y`, we simply store the label at the target column. For `X`, we are first going to tokenise and pad our text input before storing it.

# In[ ]:


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def format_data(data, max_features, maxlen, tokenizer=None, shuffle=False):
    if shuffle:
        data = data.sample(frac=1).reset_index(drop=True)
    
    data['headline'] = data['headline'].apply(lambda x: str(x).lower())

    X = data['headline']
    Y = data['fake'].values # 0: Real; 1: Fake

    if not tokenizer:
        filters = "\"#$%&()*+./<=>@[\\]^_`{|}~\t\n"
        tokenizer = Tokenizer(num_words=max_features, filters=filters)
        tokenizer.fit_on_texts(list(X))

    X = tokenizer.texts_to_sequences(X)
    X = pad_sequences(X, maxlen=maxlen)

    return X, Y, tokenizer


# The `max_features` and `max_len` variables denote the length of each vector and the vocabulary length.

# In[ ]:


max_features, max_len = 5000, 25
X, Y, tokenizer = format_data(data, max_features, max_len, shuffle=True)
X_eval, Y_eval, tokenizer = format_data(eval_data, max_features, max_len, tokenizer=tokenizer)


# In[ ]:


import pickle
pickle.dump(tokenizer, open('tokenizer.pkl', 'wb'))


# ## Model
# 
# The model we will use is based around a bi-directional RNN (either GRU or LSTM), with max pooling. The encoder is comprised of two RNN layers, while the decoder uses an RNN with a dense layer on top of it. The reconstruction of the original input occurs on a final Dense layer.

# In[ ]:


from keras.layers import Input, Dense, Bidirectional, GRU, Embedding, Dropout, LSTM
from keras.layers import concatenate, SpatialDropout1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.models import Model
from keras import regularizers

epochs=20

# Input shape
inp = Input(shape=(max_len,))

encoder = Embedding(max_features, 50)(inp)
encoder = Bidirectional(LSTM(75, return_sequences=True))(encoder)
encoder = Bidirectional(LSTM(25, return_sequences=True,
                        activity_regularizer=regularizers.l1(10e-5)))(encoder)

decoder = Bidirectional(LSTM(75, return_sequences=True))(encoder)
decoder = GlobalMaxPooling1D()(decoder)
decoder = Dense(50, activation='relu')(decoder)
decoder = Dense(max_len)(decoder)

model = Model(inputs=inp, outputs=decoder)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(X, X, epochs=epochs, batch_size=64, verbose=1)

model.save_weights('model{}.h5'.format(epochs))


# In[ ]:


model.evaluate(X, X)


# Time to compute our results!

# In[ ]:


results = model.predict(X_eval, batch_size=1, verbose=1)


# ## Classification
# 
# *(code here is modified from [this blog](https://medium.com/@curiousily/credit-card-fraud-detection-using-autoencoders-in-keras-tensorflow-for-hackers-part-vii-20e0c85301bd))*
# 
# Now we need to calculate the reconstruction error of the test set.

# In[ ]:


mse = np.mean(np.power(X_eval - results, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': mse,
                         'true_class': Y_eval})
error_df.describe()


# In[ ]:


from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)


# We now need to compute the optimal threshold to make our predictions. We will split the process into two:
# 
# 1. Find the general range where the threshold lies.
# 2. In that range, find a more specific threshold value.

# In[ ]:


LABELS = ['REAL', 'FAKE']
best, threshold = -1, -1

# General Search
for t in range(0, 3500000, 10000):
    y_pred = [1 if e > t else 0 for e in error_df.reconstruction_error.values]
    score = f1_score(y_pred, error_df.true_class, average='micro', labels=[0, 1])
    if score > best:
        best, threshold = score, t

# Specialized Search around general best
for t in range(threshold-10000, threshold+10000):
    y_pred = [1 if e > t else 0 for e in error_df.reconstruction_error.values]
    score = f1_score(y_pred, error_df.true_class, average='micro', labels=[0, 1])
    if score > best:
        best, threshold = score, t

print(threshold, best)


# We are going to visualize the data points against the threshold line.

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

groups = error_df.groupby('true_class')
fig, ax = plt.subplots()

for name, group in groups:
    ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='',
            label="Fake" if name == 1 else "Real")

ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.show();


# In[ ]:


LABELS = ['FAKE', 'REAL']
errors = error_df.reconstruction_error.values
y_pred = [1 if e > threshold else 0 for e in errors] # final predictions
conf_matrix = confusion_matrix(error_df.true_class, y_pred)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()


# Next, we are going to compute the F1 score as well.

# In[ ]:


from sklearn.metrics import f1_score

def accuracy_f1(preds, correct):
    """Returns F1-Score for predictions"""
    return f1_score(preds, correct, average='micro', labels=[0, 1])

accuracy_f1(y_pred, error_df.true_class)


# ## Scaling Error
# 
# Right now the errors lie in $[0, \infty)$. It is useful in some cases (for example, using these predictions in ensembling) to scale the error in $[0, 1]$. We cannot though simply min-max all of the values together, since then the final output wouldn't be a representative probability. Instead, we are going to do the following: Values below the threshold will be scaled to $[0, 0.5]$ and values above the threshold will be scaled to $[0.5, 1]$. The threshold, after scaling, is set to `0.5`.

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
minmax_0_05 = MinMaxScaler(feature_range=(0, 0.5))
minmax_05_1 = MinMaxScaler(feature_range=(0.5, 1))


# With the scalers initialized, we need to fit them. We are going to fit `minmax_0_05` to items below the threshold, and `minmax_05_1` to items above the threshold.

# In[ ]:


errors_below = np.array([i for i, e in enumerate(errors) if e <= threshold])
errors_above = np.array([i for i, e in enumerate(errors) if e > threshold])

minmax_0_05.fit(errors[errors_below].reshape(-1, 1))
minmax_05_1.fit(errors[errors_above].reshape(-1, 1))


# Finally, we are going to convert our errors array to the scaled outputs.

# In[ ]:


errors_mm = np.array([minmax_0_05.transform(e.reshape(1, -1)) if i in errors_below
                      else minmax_05_1.transform(e.reshape(1, -1))
                      for i, e in enumerate(errors)]).flatten()

y_pred2 = [1 if e > 0.5 else 0 for e in errors_mm]


# We are now going to calculate the percentile accuracy of the scaled predictions, alongside the F1-score (this score may differ slightly for values right around the threshold, but the difference is negligible).

# In[ ]:


def accuracy_percentile(preds, Y_validate):
    """Return the percentage of correct predictions for each class and in total"""
    real_correct, fake_correct, total_correct = 0, 0, 0
    _, (fake_count, real_count) = np.unique(Y_validate, return_counts=True)

    for i, r in enumerate(preds):
        if r == Y_validate[i]:
            total_correct += 1
            if r == 0:
                fake_correct += 1
            else:
                real_correct += 1

    print('Real Accuracy:', real_correct/real_count * 100, '%')
    print('Fake Accuracy:', fake_correct/fake_count * 100, '%')
    print('Total Accuracy:', total_correct/(real_count + fake_count) * 100, '%')


accuracy_percentile(y_pred2, error_df.true_class)


# In[ ]:


from sklearn.metrics import f1_score

def accuracy_f1(preds, correct):
    """Returns F1-Score for predictions"""
    return f1_score(preds, correct, average='micro', labels=[0, 1])

accuracy_f1(y_pred2, error_df.true_class)


# Finally, we'll store the predictions.

# In[ ]:


pd.Series(errors_mm).to_csv('autoencoder.csv', index=False)

