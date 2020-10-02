#!/usr/bin/env python
# coding: utf-8

# In this competition, we have a multi-label scenario, because a sample can have any number of labels (or none at all). To solve it, we'll use an ensemble of models that includes:
# - Bi-directionnal LSTM + visu (present notebook)
# - ConvLSTM ?
# - Random Forest (works well with imbalanced datasets)
# - XGBoost

# ## Loading libraries

# In[ ]:


import numpy as np, pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))


# ## Define paths

# In[ ]:


path = '../input/'
comp = 'jigsaw-toxic-comment-classification-challenge/'

EMBEDDING_FILE = f'{path}glove6b50d/glove.6B.50d.txt'
#EMBEDDING_FILE = f'{path}glove6b100dtxt/glove.6B.100d.txt'

TRAIN_DATA_FILE = f'{path}{comp}train.csv'
TEST_DATA_FILE = f'{path}{comp}test.csv'


# ## Load train and test sets

# In[ ]:


train = pd.read_csv(TRAIN_DATA_FILE)
test = pd.read_csv(TEST_DATA_FILE)

print('train shape:', train.shape,
      '\ntest shape:', test.shape)


# In[ ]:


# Check the data
train.isnull().any(), test.isnull().any()


# No missing values, we're good to go.

# In[ ]:


#visualize word distribution:
#first, we add a new column where we put the number of words of the corresponding comment_text
train["document_length"] = train["comment_text"].apply(lambda words: len(words.split(" ")))
max_seq_len = np.round(train["document_length"].mean() + train["document_length"].std()).astype(int)

plt.figure(figsize=(15,8))
sns.set(font_scale = 1.5)
sns.distplot(train["document_length"], hist=True, kde=True, color='b', label='Document length')
label = 'Max length = {}'.format(max_seq_len)
plt.axvline(x=max_seq_len, color='k', linestyle='--', label=label)
plt.legend()
plt.show()

#free space
del train["document_length"]


# We'll also look at how are distributed the different classes:
# 

# In[ ]:


list_classes = list(train.columns[2:].values)
num_classes = len(list_classes)
y_train = train[list_classes].to_numpy()
distrib_classes = train[list_classes].sum(axis=0)

#visualize classes distribution
plt.figure(figsize=(15,8))
sns.set(font_scale = 1.5)
ax= sns.barplot(list_classes, distrib_classes)

plt.title("Comments in each category", fontsize=24)
plt.ylabel('Number of comments', fontsize=18)
plt.xlabel('Comment Type ', fontsize=18)

#add the count above:
rects = ax.patches
for rect, distrib_classe in zip(rects, distrib_classes):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, distrib_classe, ha='center', va='bottom', fontsize=18)

plt.show()


# We have an unbalanced dataset: a lot of "good" comments compared to "bad" ones. In particular, we have very few "threat" comments (only 0.3%)... This might pose a problem if we want to split the training set to create an evaluation set, as the "stratify" argument in train_test_split might fail.

# Let's see how many comments have multiple labels:

# In[ ]:


#Sum on each row: results go from 0 (good comment) to 6 (the "winners" that have all tags)
rowSums = train[list_classes].sum(axis=1)
multiLabel_counts = rowSums.value_counts()
multiLabel_counts = multiLabel_counts.iloc[1:]

sns.set(font_scale = 1.5)
plt.figure(figsize=(15,8))
ax = sns.barplot(multiLabel_counts.index, multiLabel_counts.values)

plt.title("Comments having multiple labels ")
plt.ylabel('Number of comments', fontsize=18)
plt.xlabel('Number of labels', fontsize=18)

#adding the text labels
rects = ax.patches
labels = multiLabel_counts.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
    
plt.show()


# ## Wordcloud representations

# In[ ]:


from wordcloud import WordCloud

plt.figure(figsize=(18,7))

for i in range (1,7):
    plt.subplot(2, 3, i)
    subset = train[train[train.keys()[i+1]] == 1]
    text = subset.comment_text.values
    cloud_i = WordCloud(background_color='black',
                        collocations=False,
                        max_words = 100
                       ).generate(" ".join(text))
    
    plt.axis('off')
    title = list_classes[i-1]
    plt.title(title,fontsize=15)
    plt.imshow(cloud_i)

plt.show()


# ## Load and parse the GloVe word-embeddings file

# In[ ]:


#load embeddings
print('loading word embeddings...')
embeddings_index = {}
f = open(EMBEDDING_FILE)

#more readable:
for line in f:
    values = line.strip().split(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('found %s word vectors.' % len(embeddings_index))

#alternative, shorter but less readable:
#def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
#embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))


# ## Preprocessing the text

# First, we'll define stopwords for future removing because we don't want these words taking up space in our database, or taking up processing time.
# 
# (Note to self: not sure if it's a good idea, we might loose information by removing them... Have to check it)

# In[ ]:


#Import the librairies
import re   # module re for regular expression
import nltk #Natural Language Toolkit
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer #Tokenizer for preprocessing

preprop_tokenizer = RegexpTokenizer(r'\w+')


# Let's set the list of stopwords:

# In[ ]:


stop_words = set(stopwords.words('english'))
#stop_words


# We'll add some punctuation marks to the list. We will keep '?' and '!' as they may help classify violent comments

# In[ ]:


stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}', '_'])
stop_words.update(['zero','one','two','three','four','five','six','seven','eight','nine','ten','may','also','across','among','beside','however','yet','within'])


# We are ready to pre-process our data:

# In[ ]:


raw_comments_train = train['comment_text'].tolist()
raw_comments_test = test['comment_text'].tolist() 

print("pre-processing training data...")
processed_comments_train = []
for comment in raw_comments_train:
    tokens = preprop_tokenizer.tokenize(comment)
    filtered = [word for word in tokens if word not in stop_words]
    processed_comments_train.append(" ".join(filtered))

print("pre-processing test data...")
processed_comments_test = []
for comment in raw_comments_test:
    tokens = preprop_tokenizer.tokenize(comment)
    filtered = [word for word in tokens if word not in stop_words]
    processed_comments_test.append(" ".join(filtered))
    
print("Done.")


# ## Tokenizing the text

# In[ ]:


#Config parameters:
embed_size = 50 # how big is each word vector
max_features = 20000#20000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = max_seq_len # max number of words in a comment to use. Here, mean+std as defined earlier


# In[ ]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

print("tokenizing input data...")
tokenizer = Tokenizer(num_words=max_features)

####creates the vocabulary index (i.e. word -> index dictionary) based on word frequency. (0 is reserved for padding, and lower integer means more frequent word.)
tokenizer.fit_on_texts(processed_comments_train)

#Transforms each text in texts to a sequence of integers taken from the word_index dictionary:
list_tokenized_train = tokenizer.texts_to_sequences(processed_comments_train)
list_tokenized_test = tokenizer.texts_to_sequences(processed_comments_test)

#pad or trunc the sequences
X_train = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_test = pad_sequences(list_tokenized_test, maxlen=maxlen)

word_index = tokenizer.word_index
print('Found %s unique tokens.' %len(word_index))


# ## Preparing the GloVe word-embeddings matrix

# We will now create our embedding matrix, with random initialization for words that aren't in GloVe. We'll use the same mean and standard deviation of embeddings the GloVe has when generating the random init.

# In[ ]:


all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
emb_mean,emb_std


# In[ ]:


#embedding matrix
print('preparing embedding matrix...')
words_not_found = []
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: 
        embedding_matrix[i] = embedding_vector
    else:
        words_not_found.append(word)
print('Done.')


# Let's look at some words not found in the embeddings:

# In[ ]:


print("sample words not found: ", np.random.choice(words_not_found, 10))


# ## Callbacks

# Accuracy is not helpful with imbalanced dataset. Thus, we'll use Area under ROC (ROC-AUC) as our performance metric. We'll also look at the precision and recall

# In[ ]:


from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))


# ## Split into a train, val, and eval sets

# In[ ]:


from sklearn.model_selection import train_test_split

#Create validation split
X_train_splitted_tmp, X_val, y_train_splitted_tmp, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, shuffle=True)
#Create training and evaluation split
X_train_splitted, X_eval, y_train_splitted, y_eval = train_test_split(X_train_splitted_tmp, y_train_splitted_tmp, test_size=0.1, random_state=42)


# In[ ]:


print(np.shape(X_train_splitted), np.shape(y_train_splitted))
print(np.shape(X_val) , np.shape(y_val))
print(np.shape(X_eval) , np.shape(y_eval))


# Check the distibutions in the new sets:

# In[ ]:


#New train set:
distrib_classes_y_train_splitted = y_train_splitted.sum(axis=0)
distrib_classes_y_train_splitted/len(y_train_splitted)*100


# In[ ]:


#Validation set:
distrib_classes_y_val = y_val.sum(axis=0)
distrib_classes_y_val/len(y_val)*100


# In[ ]:


#Evaluation set:
distrib_classes_y_eval = y_eval.sum(axis=0)
distrib_classes_y_eval/len(y_eval)*100


# Using train_test_split without the stratifying option still gives us distributions that are somehow similar. That will be good enough for a first try with a first model.

# ## Tackle imbalance with class_weight

# In[ ]:


from keras import backend as K

def calculating_class_weights(y_true):
    from sklearn.utils.class_weight import compute_class_weight
    number_dim = np.shape(y_true)[1]
    weights = np.empty([number_dim, 2])
    for i in range(number_dim):
        weights[i] = compute_class_weight('balanced', [0.,1.], y_true[:, i])
    return weights
class_weights = calculating_class_weights(y_train_splitted)
class_weights


# In[ ]:


def get_weighted_loss(weights):
    def weighted_loss(y_true, y_pred):
        return K.mean((weights[:,0]**(1-y_true))*(weights[:,1]**(y_true))*K.binary_crossentropy(y_true, y_pred), axis=-1)
    return weighted_loss


# In[ ]:


#How to load this model and the custom weighted loss (for model ensembling)
#model = load_model("path/to/model.hd5f", custom_objects={"weighted_loss": get_weighted_loss(weights)}


# ## Model : GloVe-Pretrained Bidirectional LSTM + dropout

# Simple bidirectional LSTM with two fully connected layers and dropout.

# In[ ]:


from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Conv1D, BatchNormalization
from keras.layers import Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate, SpatialDropout1D
from keras.models import Model
from keras.optimizers import adam


# In[ ]:


inputs = Input(shape=(maxlen,))
#x = BatchNormalization()(inputs)
x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable = False)(inputs)
#x = SpatialDropout1D(0.2)(x)
x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1, kernel_initializer='he_normal'))(x)
#x = Conv1D(32, kernel_size = 3, padding = "valid", kernel_initializer = "he_normal")(x)
x = GlobalMaxPooling1D()(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
outputs = Dense(6, activation="sigmoid")(x) #Sigmoid gives independent probabilities for each class.

model_LSTM = Model(inputs=inputs, outputs=outputs)
model_LSTM.name = 'model_LSTM'

model_LSTM.compile(loss=get_weighted_loss(class_weights), optimizer=adam(lr=1e-3), metrics=['acc']) #binary_crossentropy independently optimises each class.

model_LSTM.summary()


# In[ ]:


batch_size = 128
epochs = 6


# In[ ]:


from keras.callbacks import EarlyStopping,ModelCheckpoint

filepath="best_weights.hdf5"
mcp = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,  save_weights_only=True, mode='min')
earlystop = EarlyStopping(monitor="val_acc", mode="max", patience=4)
RocAuc_val = RocAucEvaluation(validation_data=(X_val, y_val), interval = 1)

callbacks_list = [RocAuc_val, mcp, earlystop]


# In[ ]:


history_model_LSTM = model_LSTM.fit(X_train_splitted, 
                                    y_train_splitted, 
                                    batch_size=batch_size, 
                                    epochs=epochs, 
                                    validation_data=(X_val, y_val),
                                    callbacks = callbacks_list, 
                                    verbose=1)


# ## Visualization

# In[ ]:


#Define a smooth function to display the training and validation curves
def plot_learning_curves(history):
    val_loss = history.history['val_loss']
    loss = history.history['loss']
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    
    epochs = range(1, len(loss)+1 )
    
    # Plot the loss and accuracy curves for training and validation 
    fig, ax = plt.subplots(2,1, figsize=(12, 12))
    ax[0].plot(epochs, loss, 'bo', label="Training loss")
    ax[0].plot(epochs, val_loss, 'b', label="Validation loss",axes =ax[0])
    legend = ax[0].legend(loc='best', shadow=True)
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')#

    ax[1].plot(epochs, acc, 'bo', label="Training accuracy")
    ax[1].plot(epochs, val_acc, 'b',label="Validation accuracy")
    legend = ax[1].legend(loc='best', shadow=True)
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    return


# In[ ]:


# Visualisation:
plot_learning_curves(history_model_LSTM)


# ## Evaluation of the model

# In[ ]:


#Loading model weights
model_LSTM.load_weights(filepath)
#Get the prediction:
print('Predicting....')
y_pred = model_LSTM.predict(X_eval,batch_size=1024,verbose=1)
print('Done.')


# ## Accuracy classification score

# In multilabel classification, the [accuracy classification score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html) computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_eval.

# In[ ]:


from sklearn.metrics import accuracy_score

accuracy_score(y_eval, np.where(y_pred > 0.9, 1, 0)) #np.where(a > 0.5, 1, 0)


# ## Confusion matrices

# In[ ]:


from sklearn.metrics import multilabel_confusion_matrix

#you can fine tune the threshold for increasing recall or precision
y_pred_col0 = np.where(y_pred[:,0] > 0.4, 1, 0)
y_pred_col1 = np.where(y_pred[:,1] > 0.5, 1, 0)
y_pred_col2 = np.where(y_pred[:,3] > 0.3, 1, 0)
y_pred_col3 = np.where(y_pred[:,3] > 0.6, 1, 0)
y_pred_col4 = np.where(y_pred[:,4] > 0.5, 1, 0)
y_pred_col5 = np.where(y_pred[:,5] > 0.5, 1, 0)

y_pred_col0 = np.expand_dims(y_pred_col0, axis=1)
y_pred_col1 = np.expand_dims(y_pred_col1, axis=1)
y_pred_col2 = np.expand_dims(y_pred_col2, axis=1)
y_pred_col3 = np.expand_dims(y_pred_col3, axis=1)
y_pred_col4 = np.expand_dims(y_pred_col4, axis=1)
y_pred_col5 = np.expand_dims(y_pred_col5, axis=1)

y_pred_colTot = np.concatenate((y_pred_col0, y_pred_col1, y_pred_col2, y_pred_col3, y_pred_col4, y_pred_col5), axis=1)

mcm = multilabel_confusion_matrix(y_eval, y_pred_colTot, sample_weight=None, samplewise=False)

fig = plt.figure(figsize = (12,10))
for i in range(1,7):
    plt.subplot(2,3,i)
    if i%2==0:
        cmap = "Reds"
    else:
        cmap = "Blues"
    sns.set(font_scale=0.8)
    title = '{}'.format(list_classes[i-1])
    plt.title(title, fontsize = 15)
    sns.heatmap(mcm[i-1], cmap=cmap, square=True, fmt='.0f', cbar=False, annot=True)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# ## Classification report

# In[ ]:


from sklearn.metrics import classification_report

cr = classification_report(y_eval, np.round(y_pred), target_names = list_classes)
print(cr)


# ## Submission

# In[ ]:


#Production: 
history_model_LSTM = model_LSTM.fit(X_train, 
                                    y_train, 
                                    batch_size=batch_size, 
                                    epochs=epochs,
                                    callbacks = callbacks_list, 
                                    verbose=1)


# In[ ]:


print('Predicting....')
y_test = model_LSTM.predict(X_test,batch_size=1024,verbose=1)
sample_submission = pd.read_csv(f'{path}{comp}sample_submission.csv')
sample_submission[list_classes] = y_test
sample_submission.to_csv('submission.csv', index=False)


# ## To do list:
# - Test labelPowerset

# In[ ]:


###---  Test with the LabelPowerset + RandomOverSampler   ---####
###-------------------Not used-------------------------------####

#from skmultilearn.problem_transform import LabelPowerset
#from imblearn.over_sampling import RandomOverSampler
#
#lp = LabelPowerset()
#ros = RandomOverSampler(random_state=42)
## Applies the above stated multi-label (ML) to multi-class (MC) transformation.
#yt = lp.transform(y_train_splitted)
#X_resampled, y_resampled = ros.fit_sample(X_train_splitted, yt)
## Inverts the ML-MC transformation to recreate the ML set
#y_train_splitted = lp.inverse_transform(y_resampled)

