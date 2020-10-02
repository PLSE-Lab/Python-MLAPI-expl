#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk # wonderful tutorial can be found here https://pythonprogramming.net/tokenizing-words-sentences-nltk-tutorial/

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/amazon-alexa-reviews/amazon_alexa.tsv', sep='\t')


# # Initial data exploration

# In[ ]:


df.head(10)


# In[ ]:


df.describe()


# It seems unreasonable to use any data except for the reviews and their rating. We could have product variation categorized and then create chart which one is the best, but it is outside the scope of the problem. Feedback and date are also unnecessary.

# Second, the distribution of ratings in dataset is way off. 25-th percentile already has value of 4, and 50-th - of 5. If there is enough data for different ratings, it would seem reasonable to construct a subset with better distribution.

# In[ ]:


#omitting unnecessary columns
cdf = df[['rating', 'verified_reviews']]

print(cdf['rating'].value_counts())


# In[ ]:


cdf = pd.concat([cdf[cdf['rating'] < 5], cdf[cdf['rating'] == 5].sample(frac=1).iloc[:300]])
cdf['rating'].describe()


# Perfect dataset supposed to have equal amount of items with every rating, and their mean should be 3.0. Constructing such dataset would shrink dataset to 480 entries. Instead we can have dataset with 1164 entries with a mean of 3.5. That could create a shift in models predictions, but more data will help prevent overfitting and help model generalize data better.

# In[ ]:


cdf['rating'].hist(bins=5)


# # Assessing word vectors data

# One of the most precise method when analysing natural language is to use word2vec approach. Let's check how many different words there is and create a full set of all used words in review. That would be useful later when chosing pre-trained word vectors.

# In[ ]:


text_body = ''
for row in cdf.iterrows():
    text_body += row[1]['verified_reviews'] + ' '
    
cleaned_text_body = re.sub('[^a-zA-Z]', ' ', text_body)
word_list = nltk.tokenize.word_tokenize(cleaned_text_body.lower())
word_set = set(word_list)


# In[ ]:


len(word_set)


# Now that we got complete set of words used in dataset we can estimate how good or bad a certain vectorization of words could serve.

# In[ ]:


embeddings = {}
f = open('../input/glove6b100dtxt/glove.6B.100d.txt', 'r', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    vector = np.asarray(values[1:], dtype='float32')
    embeddings[word] = vector
f.close()


# In[ ]:


def assess_embeddings(set_of_words):
    c = 0
    missing_embeddings = []
    for word in set_of_words:
        if word in embeddings:
            c+=1
        else:
            missing_embeddings.append(word)

    print(c/len(set_of_words)*100, 'percents of words in reviews are covered in embeddings')
    return missing_embeddings

missing_embeddings = assess_embeddings(word_set)    


# In[ ]:


print(sorted(missing_embeddings))


# Embeddings successfully cover almost all word set, except for some typos. Since typos are only ~3% of words, we can ignore the problem unless we would want to improve accuracy.

# We will use the Keras library to solve the problem

# In[ ]:


import keras


# # Preparing data

# In[ ]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# In[ ]:


tokenizer = Tokenizer(num_words=len(word_set))
tokenizer.fit_on_texts([cleaned_text_body])
print('Words in word_index:', len(tokenizer.word_index))


# As we could expect, keras' tokenizer uses an algorithm that is somewhat different from the one we chose earlier. However, the amount of words in created word_index is bigger only by 1. Generally, it is better to use already made solutions than to create something from scratch. Let's quickly assess this word_index and make a decision whether it is reasonable to use it.

# In[ ]:


_ = assess_embeddings(set([kvp for kvp in tokenizer.word_index]))


# The results are just as good, so the tokenizer's word_index stays.

# In[ ]:


cdf['cleaned_text'] = cdf['verified_reviews'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))
cdf['cleaned_text'] = cdf['cleaned_text'].apply(lambda x: re.sub(' +',' ', x)) #remove consecutive spacing


# In[ ]:


cdf['sequences'] = cdf['cleaned_text'].apply(lambda x: tokenizer.texts_to_sequences([x])[0])
cdf['sequences'].head(10)


# In[ ]:


# Need to know max_sequence_length to pad other sequences
max_sequence_length = cdf['sequences'].apply(lambda x: len(x)).max()
cdf['padded_sequences'] = cdf['sequences'].apply(lambda x: pad_sequences([x], max_sequence_length)[0])


# In[ ]:


print(cdf['padded_sequences'][2])


# Now split into train, validation and test subsets

# In[ ]:


train = cdf.sample(frac=0.8)
test_and_validation = cdf.loc[~cdf.index.isin(train.index)]
validation = test_and_validation.sample(frac=0.5)
test = test_and_validation.loc[~test_and_validation.index.isin(validation.index)]

print(train.shape, validation.shape, test.shape)


# In[ ]:


def get_arrayed_data(df_set):
    setX = np.stack(df_set['padded_sequences'].values, axis=0)
    setY = pd.get_dummies(df_set['rating']).values #using one-hot encoding
    
    return (setX, setY)

trainX, trainY = get_arrayed_data(train)
validationX, validationY = get_arrayed_data(validation)
testX, testY = get_arrayed_data(test)


# # Building the model

# In[ ]:


from keras.layers import Embedding


# In[ ]:


embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, 100))
for word, i in tokenizer.word_index.items():
    # words that are not in pretrained embedding will be zero vectors.
    if word in embeddings:
        embedding_matrix[i] = embeddings[word]


# In[ ]:


embedding_layer = Embedding(len(tokenizer.word_index) + 1, 100,
                            weights=[embedding_matrix],
                            input_length=max_sequence_length,
                            trainable=False)


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Input, LSTM, Flatten, Dropout


# In[ ]:


def simple_reccurent_model(input_shape, output_shape):
    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(64, dropout=0.2))
    #model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(output_shape, activation='softmax'))
    return model


# In[ ]:


model = simple_reccurent_model(trainX.shape[1], trainY.shape[1])
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


model.fit(trainX, trainY, batch_size=64, epochs=100)


# Let's evaluate this simple model on validation set:

# In[ ]:


score, accuracy = model.evaluate(validationX, validationY, batch_size=64)
print(accuracy)


# ### 64% of accuracy
# It is much better than random guessing, though. Let's check the failed predictions manually.

# In[ ]:


reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))


# In[ ]:


for i, y in enumerate(validationY):
    #p = model.predict(np.array([validationX[i]])).round()[0].astype('int32')
    prediction = model.predict(np.array([validationX[i]])).argmax()
    actual = y.argmax()
    if prediction != actual:
        print("Validation", i)
        print("Predicted review:", prediction + 1, ", actual review:", actual + 1)
        #print(validationX[i])
        text = []
        for word_i in validationX[i]:
            if word_i in reverse_word_map:
                text.append(reverse_word_map[word_i])
        print(' '.join(text))
        print()


# ## First try conclusion
# While in some cases the model cannot be really guilty for misunderstanding (validation example 1, validation example 13), in many cases there could be some improvement. Time to tune model.

# In[ ]:


def tunable_reccurent_model(input_shape, output_shape, hyperparams):
    model = Sequential()
    model.add(embedding_layer)
    
    for i, lstm_size in enumerate(hyperparams['lstm_sizes']):
        model.add(LSTM(lstm_size, dropout=hyperparams['dp']))
    
    for i, dense_size in enumerate(hyperparams['dense_sizes']):
        model.add(Dense(dense_size, activation=hyperparams['dense_activation']))
        model.add(Dropout(hyperparams['dp']))
    
    model.add(Dense(output_shape, activation='softmax'))
    return model


# In[ ]:


def evaluate_model(input_shape, output_shape,
                   hyperparams, train_set, validation_set,
                   train_epochs=100):
    model = simple_reccurent_model(trainX.shape[1], trainY.shape[1])
    model.compile(loss='categorical_crossentropy',
                  optimizer=hyperparams['optimizer'],
                  metrics=['accuracy'])
    
    model.fit(train_set[0], train_set[1], batch_size=hyperparams['batch_size'], epochs=train_epochs, verbose=0)
    _, train_accuracy = model.evaluate(train_set[0], train_set[1])
    _, validation_accuracy = model.evaluate(validation_set[0], validation_set[1])
    print("Train accuaracy:", train_accuracy, "Validation Accuracy:", validation_accuracy)
    return validation_accuracy


# In[ ]:


lstm_sizes = [[32], [64], [128], [64, 32]]
dense_sizes = [[32], [64], [128], [64, 32]]
dense_activations = ['relu', 'tanh', 'sigmoid']
dps = [0.1, 0.2, 0.3]
optimizers = ['Adam', 'SGD', 'RMSprop']
epochs = [100, 125, 150]
batch_sizes = [32, 64, 128]

results = []
counter=1
# all hyperparameters here are enumerated not in random order - the least important are closer to outer cycle
for ep in epochs:
    for optimizer in optimizers:
        for dense_activation in dense_activations:
            for batch_size in batch_sizes:
                for dp in dps:
                    for dense_size in dense_sizes:
                        for lstm_size in lstm_sizes:
                            hyperparams = {
                                'lstm_sizes': lstm_size,
                                'dp': dp,
                                'dense_sizes': dense_size,
                                'dense_activation': dense_activation,
                                'optimizer': optimizer,
                                'batch_size': batch_size
                            }
                            #print("Interation", counter)
                            #acc = evaluate_model(trainX.shape[1], trainY.shape[1],
                            #                    hyperparams, (trainX, trainY), (validationX, validationY),
                            #                    ep)
                            #results.append((acc, hyperparams, {'ep': ep, 'batch_size': batch_size}))
                            #counter+=1
                            #print()
                            


# This method would find the best model among all the combinations, but it will take a lot of time. If every training would take similar amount of time that the original simple model took (about 10 minutes on 100 epochs), trying only the combinations of lstm_sizes and dense_sizes would take 6 * 6 * 10 = 360 minutes = 6 hours, and we got combinations of other hyperparameters. The required time would be enourmous.
# 
# Alternative is to try every hyperparameter with default model, measure the impact on accuracy and then, judging on how good model performed with certain parameter, try to manually find the required combination.
# 
# Pros of the second tactics are that it is much more computationally efficient. Cons are that some hyperparameters might improve performance only with combination with others, and we are likely to miss it.
# 
# Nevertheless, we have to choose the computationally efficient path.

# Additional time optimization is to cut the amount of epochs here. On 10 epochs the results probably would not be very much impressive, but it could be enough for us to determine if those hyperparameters are comparatively work or not.
# 
# We will not try here total grid search of hyperparameters. Instead we focus on how each hyperparameter changes performance individually, and then try to finish tuning manually.

# In[ ]:


import copy


# In[ ]:


lstm_sizes = [[32], [64], [64, 32]]
dense_sizes = [[32], [64], [64, 32]]
dense_activations = ['relu', 'tanh', 'sigmoid']
dps = [0.1, 0.2, 0.3]
optimizers = ['Adam', 'SGD', 'RMSprop']
epochs = 10
batch_sizes = [32, 64, 128]

hyperparams = {
    'lstm_size': lstm_sizes,
    'dense_size': dense_sizes,
    'dense_activation': dense_activations,
    'dp': dps,
    'optimizer': optimizers,
    'batch_size': batch_sizes
}

default_hyperparams = {
    'lstm_size': [64],
    'dp': 0.2,
    'dense_size': [64],
    'dense_activation': 'relu',
    'optimizer': 'adam',
    'batch_size': 64
}

counter = 1
validation_results = []
for hp_name, hp_list in hyperparams.items():
    accs = []
    for hp_val in hp_list:
        hp = copy.deepcopy(default_hyperparams)
        hp[hp_name] = hp_val
        print("Interation", counter)
        acc = evaluate_model(trainX.shape[1], trainY.shape[1],
                            hp, (trainX, trainY), (validationX, validationY), epochs)
        counter+=1
        accs.append(acc)
        print()
    validation_results.append((hp_name, accs))


# In[ ]:


fig = plt.figure(figsize=(6, 18))

for i, result in enumerate(validation_results):
    ax = fig.add_subplot(len(validation_results), 1, i+1)
    hp_name = result[0]
    hp_errors = result[1]
    
    ax.set_title(hp_name)
    ax.plot(range(0, len(hp_errors)), hp_errors)
    plt.sca(ax)
    x_labels = hyperparams[hp_name]
    plt.xticks(range(0, len(hp_errors)), x_labels)
    
fig.tight_layout()
plt.show()


# In[ ]:


# edited function to return model
def evaluate_model(input_shape, output_shape,
                   hyperparams, train_set, validation_set,
                   train_epochs=100, verbose=0):
    model = simple_reccurent_model(trainX.shape[1], trainY.shape[1])
    model.compile(loss='categorical_crossentropy',
                  optimizer=hyperparams['optimizer'],
                  metrics=['accuracy'])
    
    model.fit(train_set[0], train_set[1], batch_size=hyperparams['batch_size'], epochs=train_epochs, verbose=verbose)
    _, train_accuracy = model.evaluate(train_set[0], train_set[1])
    _, validation_accuracy = model.evaluate(validation_set[0], validation_set[1])
    print("Train accuaracy:", train_accuracy, "Validation Accuracy:", validation_accuracy)
    return validation_accuracy, model


# In[ ]:


tuned_hyperparams = {
    'lstm_size': [32],
    'dp': 0.2,
    'dense_size': [64],
    'dense_activation': 'relu',
    'optimizer': 'adam',
    'batch_size': 32
}

acc, model = evaluate_model(trainX.shape[1], trainY.shape[1],
    tuned_hyperparams, (trainX, trainY), (validationX, validationY), 100, 0)


# In[ ]:


tuned_hyperparams = {
    'lstm_size': [32],
    'dp': 0.5,
    'dense_size': [64],
    'dense_activation': 'relu',
    'optimizer': 'adam',
    'batch_size': 32
}

acc, model2 = evaluate_model(trainX.shape[1], trainY.shape[1],
    tuned_hyperparams, (trainX, trainY), (validationX, validationY), 50, 0)


# Third model and it didn't improve accuracy on validation set. We have a huge overfitting. Small dataset is the main suspect in that.
# 
# We have to reconsider the problem. Choosing the rating on 1-5 scale is quite arbitrary proccess. As we saw earlier when we inspected errors, there are misleading samples.
# 
# This time let's assume that rating of 1-3 is bad and of 4-5 is good. What is percentage of error in that case?

# In[ ]:


error_count = 0
for i, y in enumerate(validationY):
    #p = model.predict(np.array([validationX[i]])).round()[0].astype('int32')
    prediction = model2.predict(np.array([validationX[i]])).argmax() > 3
    actual = y.argmax() > 3
    if prediction != actual:
        print("Validation", i)
        print("Predicted review is good:", prediction, ", actual review is good:", actual)
        #print(validationX[i])
        text = []
        for word_i in validationX[i]:
            if word_i in reverse_word_map:
                text.append(reverse_word_map[word_i])
        print(' '.join(text))
        print()
        error_count+=1


# In[ ]:


print("Accuracy of prediction whether review was good:",(validationY.shape[0] - error_count)/validationY.shape[0] * 100)


# That's more like it! 82% accuracy in prediction whether the review was good or bad.
# 
# The other helpful thing to use when analizing results is confusion matrix.

# In[ ]:


predicted = [x > 3 for x in model2.predict(validationX).argmax(axis=1)]
actual = [x > 3 for x in validationY.argmax(axis=1)]


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


cnf_matrix = confusion_matrix(actual, predicted)
sns.heatmap(cnf_matrix)
tn, fp, fn, tp = cnf_matrix.ravel()
print("True positive:", tp, ", True negative:", tn,
      ", False positive:", fp, ", False negative:", fn)


# It seems like the validation set is severely unbalanced - it contains far more negative reviews than positive.
# 
# Regardless the confusion matrix results, in our case, for Amazon it would be worse to have False positive review than False negative, so our model's errors are rather acceptable.
# 
# Now let's check our model against the test data. If it is not as unbalanced as validation set, there are good chances that model will perform better.

# In[ ]:


_, test_accuracy = model2.evaluate(testX, testY)
print(test_accuracy)


# ## 48% of accuracy in precise predictions.
# Now let's check how model generally predicts whether reviews are good or not.

# In[ ]:


def check_general_accuracy(model, setX, setY):
    predicted = [x > 3 for x in model.predict(setX).argmax(axis=1)]
    actual = [x > 3 for x in setY.argmax(axis=1)]

    cnf_matrix = confusion_matrix(actual, predicted)
    sns.heatmap(cnf_matrix)
    tn, fp, fn, tp = cnf_matrix.ravel()
    print("True positive:", tp, ", True negative:", tn,
          ", False positive:", fp, ", False negative:", fn)

    print("Total accuracy:", (tp+tn)/testX.shape[0]*100)


# In[ ]:


check_general_accuracy(model2, testX, testY)


#  **80%** of accuracy in predicting general moods of reviews.

# Possible ways to improve model:
# * get bigger dataset
# * balance dataset better
# * perform more diligent search for optimal hyperparameters
# * try other models

# # Convolutional model

# Before wrapping up, let's try to solve this problem using another architecture of model.
# 
# We had a choice to treat each word in sentence as just another value and feed it into plain neural network, or treat it as a sequence in RNN, and we used it as such.
# 
# Now, we can try a somewhat middleground at this. Convolutional neural network, due to it's nature will have an effect of a sequence model with rolling window of chosen width.

# In[ ]:


from keras.layers import Conv1D, MaxPooling1D


# In[ ]:


def simple_conv_model(input_shape, output_shape):
    model = Sequential()
    model.add(embedding_layer)
    
    model.add(Conv1D(32, 5, activation='relu'))
    model.add(Conv1D(64, 5, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Dropout(0.2))
        
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    
    model.add(Dense(output_shape, activation='softmax'))
    return model


# In[ ]:


model = simple_conv_model(trainX.shape[1], trainY.shape[1])
model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint("weights-improvements.hdf5",
     monitor='val_acc', verbose=1, save_best_only=True, mode='max')


# In[ ]:


model.fit(trainX, trainY, validation_data=(validationX, validationY),
          batch_size=64, callbacks=[checkpoint], epochs=100)


# In[ ]:


model.load_weights("weights-improvements.hdf5")


# In[ ]:


_, test_accuracy = model.evaluate(testX, testY)
print(test_accuracy)


# Not too great in precise rating predictions.

# In[ ]:


check_general_accuracy(model, testX, testY)


# # 77% of general accuracy for convolutional network
# Convolutional neural network predictions was slightly less accurate comparing to LSTM-based network, but no major tuning was performed.
# 
# Possible improvements stay same as for LSTM-based network:
# * get bigger dataset
# * balance dataset better
# * perform more diligent search for optimal hyperparameters

# # Conclusion
# The task of precise prediction how many stars a person gave to a product based on their review is especially hard because people aren't obliged to be explain in reviews all their thoughts, and even then there are no strict rules for machine to learn. Much easier task, however, is to determine whether the review was good (>3 stars) or bad. The improvements could be made if it was the task from the beginning, and model could be trained specifically to determine that.

# In[ ]:




