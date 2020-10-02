#!/usr/bin/env python
# coding: utf-8

# # 1. Import

# In[ ]:


# System
import os

# Time
import time
import datetime

# Numerical
import numpy as np
import pandas as pd

# Tools
import itertools
from collections import Counter

# NLP
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import SnowballStemmer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
# from pywsd.utils import lemmatize_sentence

# Preprocessing
from sklearn import preprocessing
from sklearn.utils import class_weight as cw
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Model Selection
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

# Evaluation Metrics
from sklearn import metrics 
from sklearn.metrics import f1_score, accuracy_score,confusion_matrix,classification_report

# Deep Learing Preprocessing - Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical

# Deep Learning Model - Keras
from keras.models import Model
from keras.models import Sequential

# Deep Learning Model - Keras - CNN
from keras.layers import Conv1D, Conv2D, Convolution1D, MaxPooling1D, SeparableConv1D, SpatialDropout1D,     GlobalAvgPool1D, GlobalMaxPool1D, GlobalMaxPooling1D 
from keras.layers.pooling import _GlobalPooling1D
from keras.layers import MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D

# Deep Learning Model - Keras - RNN
from keras.layers import Embedding, LSTM, Bidirectional

# Deep Learning Model - Keras - General
from keras.layers import Input, Add, concatenate, Dense, Activation, BatchNormalization, Dropout, Flatten
from keras.layers import LeakyReLU, PReLU, Lambda, Multiply



# Deep Learning Parameters - Keras
from keras.optimizers import RMSprop, Adam

# Deep Learning Callbacs - Keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

print(os.listdir("../input"))


# # 2. Functions

# In[ ]:


# print date and time for given type of representation
def date_time(x):
    if x==1:
        return 'Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
    if x==2:    
        return 'Timestamp: {:%Y-%b-%d %H:%M:%S}'.format(datetime.datetime.now())
    if x==3:  
        return 'Date now: %s' % datetime.datetime.now()
    if x==4:  
        return 'Date today: %s' % datetime.date.today() 


# # 3. Read Data

# In[ ]:


input_directory = r"../input/"
output_directory = r"../output/"

if not os.path.exists(output_directory):
    os.mkdir(output_directory)
    
figure_directory = "../output/figures"
if not os.path.exists(figure_directory):
    os.mkdir(figure_directory)
    
    
file_name_pred_batch = figure_directory+r"/result"
file_name_pred_sample = figure_directory+r"/sample"


# In[ ]:


df = pd.read_csv(input_directory + "spam.csv", encoding='latin-1')
df.head()


# In[ ]:


df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1, inplace=True)
df = df.rename(columns={"v1":"label", "v2":"text"})
df.head()


# In[ ]:


df_new = df.copy()
df_stat = df.copy()


# In[ ]:


lmm = WordNetLemmatizer()
porter_stemmer = PorterStemmer()
snowball_stemmer = SnowballStemmer('english')

stop_words = set(stopwords.words('english'))


# In[ ]:


# df_new['parsed'] = df_new['text'].apply(lambda x: x.lower())
# df_new['parsed'] = df_new['parsed'].apply(lambda x: word_tokenize(x))

# df_new['no_stop'] = df_new['parsed'].apply(lambda x: [word for word in str(x).split() if word not in stop_words])

# df_new['stem'] = df_new['no_stop'].apply(lambda x: [snowball_stemmer.stem(word) for word in x])
# df_new['stem'] =  df_new['stem'].apply(lambda x: " ".join(x))

# df_new['lemi'] =  df_new['no_stop'].apply(lambda x: " ".join(x))
# df_new['lemi'] =  df_new['lemi'].apply(lambda x: lmm.lemmatize(x))

# df_new['parsed'] = df_new['parsed'].apply(lambda x: ' '.join(x))
# df_new['no_stop'] = df_new['no_stop'].apply(lambda x: ' '.join(x))
# df_new['stem'] = df_new['stem'].apply(lambda x: ' '.join(x))
# df_new['lemi'] = df_new['lemi'].apply(lambda x: ' '.join(x))

# df_new.head()


# In[ ]:


df_stat["text_clean"] = df_stat["text"].apply(lambda x: re.sub("[^a-zA-Z]", " ", x.lower()))

df_stat["length"] = df_stat["text"].apply(lambda x: len(x))
df_stat["token_count"] = df_stat["text"].apply(lambda x: len(x.split(" ")))
df_stat["unique_token_count"] = df_stat["text"].apply(lambda x: len(set(x.lower().split(" "))))
df_stat["unique_token_count_percent"] = df_stat["unique_token_count"]/df_stat["token_count"]

df_stat["length_clean"] = df_stat["text_clean"].apply(lambda x: len(x))
df_stat["token_count_clean"] = df_stat["text_clean"].apply(lambda x: len(x.split(" ")))

df_stat.head()


# # 4 . Visualization

# In[ ]:


sns.set_style("ticks")
figsize=(20, 5)

ticksize = 18
titlesize = ticksize + 8
labelsize = ticksize + 5

xlabel = "Label"
ylabel = "Count"

title = "Number of ham and spam messages"


params = {'figure.figsize' : figsize,
          'axes.labelsize' : labelsize,
          'axes.titlesize' : titlesize,
          'xtick.labelsize': ticksize,
          'ytick.labelsize': ticksize}

plt.rcParams.update(params)

col1 = "label"
col2 = "label"
sns.countplot(x=df[col1])
plt.title(title.title())
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.xticks(rotation=90)
plt.plot()

df.label.value_counts()


# In[ ]:


s1 = df_stat[df_stat['label'] == 'ham']['text'].str.len()
s2 = df_stat[df_stat['label'] == 'spam']['text'].str.len()
s3 = df_stat[df_stat['label'] == 'ham']['text_clean'].str.len()
s4 = df_stat[df_stat['label'] == 'spam']['text_clean'].str.len()
s5 = df_stat[df_stat['label'] == 'ham']['text'].str.split().str.len()
s6 = df_stat[df_stat['label'] == 'spam']['text'].str.split().str.len()
s7 = df_stat[df_stat['label'] == 'ham']['text_clean'].str.split().str.len()
s8 = df_stat[df_stat['label'] == 'spam']['text_clean'].str.split().str.len()

sns.set()
sns.set_style("ticks")

figsize=(20, 15)

ticksize = 14
titlesize = ticksize + 8
labelsize = ticksize + 5

xlabel = "Length"
ylabel = "Count"

title1 = "Length Distribution"
title2 = "Length Distribution (Clean)"
title3 = "Word Count Distribution"
title4 = "Word Count Distribution (Clean)"



params = {'figure.figsize' : figsize,
          'axes.labelsize' : labelsize,
          'axes.titlesize' : titlesize,
          'xtick.labelsize': ticksize,
          'ytick.labelsize': ticksize}

plt.rcParams.update(params)
# fig.subplots_adjust(hspace=0.5, wspace=0.5)

col1 = "len"
col2 = "label"
plt.subplot(221)
sns.distplot(s1, label='Ham')
sns.distplot(s2, label='Spam')
plt.title(title1.title())
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.legend()

plt.subplot(222)
sns.distplot(s3, label='Ham (Clean)')
sns.distplot(s4, label='Spam (Clean)')
plt.title(title2.title())
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.legend()

plt.subplot(223)
sns.distplot(s5, label='Ham Word')
sns.distplot(s6, label='Spam Word')
plt.title(title3.title())
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.legend()

plt.subplot(224)
sns.distplot(s7, label='Ham Word')
sns.distplot(s8, label='Spam Word')
plt.title(title4.title())
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.legend()
plt.show()


# # 5. Preprocessing

# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(df["text"],df["label"], test_size = 0.2, random_state = 10)


# In[ ]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# # 6. Feature Extraction

# In[ ]:


vect = CountVectorizer()
X_train_df = vect.fit_transform(X_train)
X_test_df = vect.transform(X_test)


# In[ ]:


print(vect.get_feature_names()[0:20])
print(vect.get_feature_names()[-20:])


# # 7. Model Trainning

# In[ ]:


models = {
    "SVC": svm.SVC(kernel="linear"),
    "MultinomialNB": MultinomialNB(),
    "LogisticRegression": LogisticRegression(),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "RandomForestClassifier": RandomForestClassifier(),
    "AdaBoostClassifier": AdaBoostClassifier(),
    "BaggingClassifier": BaggingClassifier(),
    "ExtraTreesClassifier": ExtraTreesClassifier()
}
prediction = dict()
score_map = {}

for model_name in models:
    model = models[model_name]
    model.fit(X_train_df,y_train)
    prediction[model_name] = model.predict(X_test_df)
    score = accuracy_score(y_test, prediction[model_name])
    score_map[model_name] = score
#     print("{}{}{}".format(model_name, ": ", score))


# In[ ]:


result = pd.DataFrame()
result["model"] = score_map.keys()
result["score"] = score_map.values()
result["score"] = result["score"].apply(lambda x: x*100)


# In[ ]:


def plot_model_performace(result):
    sns.set_style("ticks")
    figsize=(22, 6)

    ticksize = 12
    titlesize = ticksize + 8
    labelsize = ticksize + 5

    xlabel = "Model"
    ylabel = "Score"

    title = "Model Performance"

    params = {'figure.figsize' : figsize,
              'axes.labelsize' : labelsize,
              'axes.titlesize' : titlesize,
              'xtick.labelsize': ticksize,
              'ytick.labelsize': ticksize}

    plt.rcParams.update(params)

    col1 = "model"
    col2 = "score"
    sns.barplot(x=col1, y=col2, data=result)
    plt.title(title.title())
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=90)
    plt.grid()
    plt.plot()
    plt.show()
    print(result)


# In[ ]:


plot_model_performace(result)


# # 8. Hyper Parameter Search

# In[ ]:


param_grid = {
    "C": np.concatenate(
        [
            np.arange(0.0001, 0.001, 0.0001),
            np.arange(0.001, 0.01, 0.001),
            np.arange(0.01, 0.1, 0.01),
            np.arange(0.1, 1, 0.1),
            np.arange(1, 10, 1),
            np.arange(10, 100, 5)
        ],
        axis=None),
    
    "kernel": ("linear", "rbf", "poly", "sigmoid"),
#     "kernel": ("linear", "poly"),
#     "degree": list(np.arange(1,25, 1)),
#     "gamma": np.concatenate(
#         [
#             np.arange(0.0001, 0.001, 0.0001),
#             np.arange(0.001, 0.01, 0.001),
#             np.arange(0.01, 0.1, 0.01),
#             np.arange(0.1, 1, 0.1),
#             np.arange(1, 10, 1),
#             np.arange(10, 100, 5)
#         ],
#         axis=None)
}
# print(param_grid)
# model = svm.SVC(class_weight="balanced")
# grid = GridSearchCV(model, param_grid, n_jobs=-1, verbose=1, cv=3)
# grid.fit(X_train_df,y_train)
# print("{}{}".format("Best Estimator: ", grid.best_estimator_))
# print("{}{}".format("Best Params: ", grid.best_params_))
# print("{}{}".format("Best Scores: ", grid.best_score_))


# In[ ]:


param_grid = {
    "alpha": np.concatenate(
        [
            np.arange(0.0001, 0.001, 0.0001),
            np.arange(0.001, 0.01, 0.001),
            np.arange(0.01, 0.1, 0.01),
            np.arange(0.1, 1, 0.1),
            np.arange(1, 10, 1),
            np.arange(10, 100, 5)
        ]) 
}

model = MultinomialNB()
grid_cv_model = GridSearchCV(model, param_grid, n_jobs=-1, verbose=3, cv=3)
grid_cv_model.fit(X_train_df, y_train)


# # 9. Evaluation Metrics

# In[ ]:


print("{}{}".format("Best Estimator: ", grid_cv_model.best_estimator_))
print("{}{}".format("Best Params:    ", grid_cv_model.best_params_))
print("{}{}".format("Best Scores:    ", grid_cv_model.best_score_))


# In[ ]:


print(classification_report(y_test, prediction['MultinomialNB'], target_names = ["Ham", "Spam"]))


# In[ ]:


def plot_confusion_matrix(y_test, y_pred, title=""):
    conf_mat = confusion_matrix(y_test, y_pred)
    conf_mat_normalized = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]

#     sns.set_style("ticks")
    figsize=(22, 5)

    ticksize = 18
    titlesize = ticksize + 8
    labelsize = ticksize + 5

    xlabel = "Predicted label"
    ylabel = "True label"


    params = {'figure.figsize' : figsize,
              'axes.labelsize' : labelsize,
              'axes.titlesize' : titlesize,
              'xtick.labelsize': ticksize,
              'ytick.labelsize': ticksize}

    plt.rcParams.update(params)

    plt.subplot(121)
    sns.heatmap(conf_mat, annot=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.subplot(122)
    sns.heatmap(conf_mat_normalized, annot=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


    print("Confusion Matrix:\n")
    print(conf_mat)
    print("\n\nConfusion Matrix Normalized:\n")
    print(conf_mat_normalized)


# In[ ]:


plot_confusion_matrix(y_test, prediction['MultinomialNB'], title="MultinomialNB")


# In[ ]:


X_test[y_test < prediction["MultinomialNB"] ]


# In[ ]:


X_test[y_test > prediction["MultinomialNB"] ]


# # 10. Deep Learning

# ## Output Configuration

# In[ ]:


main_model_dir = output_directory + r"models/"
main_log_dir = output_directory + r"logs/"

try:
    os.mkdir(main_model_dir)
except:
    print("Could not create main model directory")
    
try:
    os.mkdir(main_log_dir)
except:
    print("Could not create main log directory")



model_dir = main_model_dir + time.strftime('%Y-%m-%d %H-%M-%S') + "/"
log_dir = main_log_dir + time.strftime('%Y-%m-%d %H-%M-%S')


try:
    os.mkdir(model_dir)
except:
    print("Could not create model directory")
    
try:
    os.mkdir(log_dir)
except:
    print("Could not create log directory")
    
model_file = model_dir + "{epoch:02d}-val_acc-{val_acc:.2f}-val_loss-{val_loss:.2f}.hdf5"


# In[ ]:


print("Settting Callbacks")

checkpoint = ModelCheckpoint(
    model_file, 
    monitor='val_acc', 
    save_best_only=True)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=2,
    verbose=1,
    restore_best_weights=True)


reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.6,
    patience=1,
    verbose=1)


callbacks = [checkpoint, reduce_lr, early_stopping]

# callbacks = [early_stopping]

print("Set Callbacks at ", date_time(1))


# ## 10.1. Preprocessing

# In[ ]:


X = df.text
Y = df.label

label_encoder = LabelEncoder()

Y = label_encoder.fit_transform(Y)

Y = Y.reshape(-1, 1)


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15)

max_words = len(set(" ".join(X_train).split()))
max_len = X_train.apply(lambda x: len(x)).max()

# max_words = 1000
# max_len = 150


# In[ ]:


tokenizer = Tokenizer(num_words=max_words)

tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_train_seq = sequence.pad_sequences(X_train_seq, maxlen=max_len)


# In[ ]:


# Calculate Class Weights
def get_weight(y):
    class_weight_current =  cw.compute_class_weight('balanced', np.unique(y), y)
    return class_weight_current


# In[ ]:


class_weight = get_weight(Y_train.flatten())


# ## 10.2 Model

# In[ ]:


def get_rnn_model():
    model = Sequential()
    
    model.add(Embedding(max_words, 50, input_length=max_len))
    model.add(LSTM(64))
    
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(256, activation='relu'))
    
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(1, activation='sigmoid'))
    
    model.summary()
    
    return model


def get_cnn_model():   
    model = Sequential()
    
    model.add(Embedding(max_words, 50, input_length=max_len))
    
    model.add(Conv1D(64, 3, padding='valid', activation='relu', strides=1))
    model.add(GlobalMaxPooling1D())
    
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(256, activation='relu'))
    
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(1, activation='sigmoid'))
    
    model.summary()
    return model


# In[ ]:


def plot_performance(history=None, figure_directory=None, ylim_pad=[0, 0]):
    xlabel = 'Epoch'
    legends = ['Training', 'Validation']

    plt.figure(figsize=(20, 5))

    y1 = history.history['acc']
    y2 = history.history['val_acc']

    min_y = min(min(y1), min(y2))-ylim_pad[0]
    max_y = max(max(y1), max(y2))+ylim_pad[0]


    plt.subplot(121)

    plt.plot(y1)
    plt.plot(y2)

    plt.title('Model Accuracy\n'+date_time(1), fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.ylim(min_y, max_y)
    plt.legend(legends, loc='upper left')
    plt.grid()

    y1 = history.history['loss']
    y2 = history.history['val_loss']

    min_y = min(min(y1), min(y2))-ylim_pad[1]
    max_y = max(max(y1), max(y2))+ylim_pad[1]


    plt.subplot(122)

    plt.plot(y1)
    plt.plot(y2)

    plt.title('Model Loss\n'+date_time(1), fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.ylim(min_y, max_y)
    plt.legend(legends, loc='upper left')
    plt.grid()
    if figure_directory:
        plt.savefig(figure_directory+"/history")

    plt.show()


# In[ ]:


model1 = get_rnn_model()


# In[ ]:


# loss = 'categorical_crossentropy'
loss = 'binary_crossentropy'
metrics = ['accuracy']


# ## 10.3. Model Trainning

# ### 10.3.1. RNN

# In[ ]:


print("Starting...\n")

start_time = time.time()
print(date_time(1))

print("\n\nCompliling Model ...\n")
learning_rate = 0.001
optimizer = Adam(learning_rate)
# optimizer = Adam()

model1.compile(optimizer=optimizer, loss=loss, metrics=metrics)

verbose = 1
epochs = 100
batch_size = 128
validation_split = 0.2

print("Trainning Model ...\n")

history1 = model1.fit(
    X_train_seq,
    Y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=verbose,
    callbacks=callbacks,
    validation_split=validation_split,
    class_weight =class_weight
    )

elapsed_time = time.time() - start_time
elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

print("\nElapsed Time: " + elapsed_time)
print("Completed Model Trainning", date_time(1))


# #### 10.3.1.2  Visualization

# In[ ]:


plot_performance(history=history1)


# ### 10.3.1. RNN

# In[ ]:


model2 = get_cnn_model()


# In[ ]:


print("Starting...\n")

start_time = time.time()
print(date_time(1))

print("\n\nCompliling Model ...\n")
learning_rate = 0.001
optimizer = Adam(learning_rate)
# optimizer = Adam()

model2.compile(optimizer=optimizer, loss=loss, metrics=metrics)

verbose = 1
epochs = 100
batch_size = 128
validation_split = 0.2

print("Trainning Model ...\n")

history2 = model2.fit(
    X_train_seq,
    Y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=verbose,
    callbacks=callbacks,
    validation_split=validation_split,
    class_weight =class_weight
    )

elapsed_time = time.time() - start_time
elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

print("\nElapsed Time: " + elapsed_time)
print("Completed Model Trainning", date_time(1))


# #### 10.3.1.2 Visualization

# In[ ]:


plot_performance(history=history2)


# ## 10.5 Inference/ Prediction

# In[ ]:


test_X_seq = tokenizer.texts_to_sequences(X_test)
test_X_seq = sequence.pad_sequences(test_X_seq, maxlen=max_len)
accuracy1 = model1.evaluate(test_X_seq, Y_test)
accuracy2 = model2.evaluate(test_X_seq, Y_test)


# ### 10.5.1 Evaluation

# In[ ]:


print("Model Performance of RNN (Test Accuracy):")
print('Accuracy: {:0.2f}%\nLoss: {:0.3f}\n'.format(accuracy1[1]*100, accuracy1[0]))

print("\nModel Performance of RNN (Test Accuracy):")
print('v: {:0.2f}%\nLoss: {:0.3f}\n'.format(accuracy2[1]*100, accuracy2[0]))


# In[ ]:


ypreds1 = model1.predict_classes(test_X_seq, verbose=1)
ypreds2 = model2.predict_classes(test_X_seq, verbose=1)


# In[ ]:


print(classification_report(Y_test, ypreds1, target_names = ["Ham", "Spam"]))


# In[ ]:


plot_confusion_matrix(Y_test, ypreds1, title="RNN")


# In[ ]:


print(classification_report(Y_test, ypreds2, target_names = ["Ham", "Spam"]))


# #### 10.5.1.2 Visualization

# In[ ]:


plot_confusion_matrix(Y_test, ypreds2, title="CNN")


# In[ ]:


row1 = pd.DataFrame({'model': 'RNN', 'score': accuracy1[1]*100}, index=[-1])
result = pd.concat([row1, result.ix[:]]).reset_index(drop=True)
row2 = pd.DataFrame({'model': 'CNN', 'score': accuracy2[1]*100}, index=[-1])
result = pd.concat([row2, result.ix[:]]).reset_index(drop=True)


# In[ ]:


plot_model_performace(result)


# # Reference:
# 1. [Text Preprocessing and Machine Learning Modeling](https://www.kaggle.com/futurist/text-preprocessing-and-machine-learning-modeling)
# 2. [keras mlp cnn test for text classification](https://www.kaggle.com/jacklinggu/keras-mlp-cnn-test-for-text-classification)

# In[ ]:




