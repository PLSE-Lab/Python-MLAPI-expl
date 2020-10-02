import pandas as pd
from   keras                        import initializers, regularizers, constraints, callbacks, optimizers
from   keras.layers                 import Conv1D, Embedding, GlobalMaxPooling1D, concatenate, Input, Dense
from   keras.models                 import Sequential, Model
from   keras.preprocessing.sequence import pad_sequences
from   keras.preprocessing.text     import Tokenizer
from   sklearn.model_selection      import train_test_split
from   gensim.models.keyedvectors   import KeyedVectors
import numpy as np

print "Start Loading data"
train_data = pd.read_csv('../input/train.csv')
test_data  = pd.read_csv('../input/test.csv')
print "End Loading data"

#first part inspired by https://www.kaggle.com/eikedehling/feature-engineering.
#Extract some features that our intuition or knowledge on the topic tells that might be important for the model.
#The purpose is to introduce them in the neural network after the convolution layers on the word2vec embedding of the
#sentences.

print "Start getting extra features"
total = train_data.loc[:, 'id' : 'comment_text']
total = pd.concat([total, test_data], sort = 'False')

#the following features were found correlated to the predicted value
total['total_length']          = total['comment_text'].apply(len)
total['capitals']              = total['comment_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
total['caps_vs_length']        = total.apply(lambda row: float(row['capitals'])/float(row['total_length']),axis = 1)
total['num_exclamation_marks'] = total['comment_text'].apply(lambda comment: comment.count('!'))
total['num_punctuation']       = total['comment_text'].apply(lambda comment: sum(comment.count(w) for w in '.,;:'))
total['num_unique_words']      = total['comment_text'].apply(lambda comment: len(set(w for w in comment.split())))

total = total.drop('total_length', axis = 1)
total = total.drop('capitals', axis = 1)

train = total[0 : train_data.shape[0]]
test  = total[train_data.shape[0] : (train_data.shape[0] + test_data.shape[0])]
print "End getting extra features"

#now let's tokenize the sentences
print "Start tokenizing sentences"
NUM_WORDS = 20000
tokenizer = Tokenizer(num_words = NUM_WORDS)

tokenizer.fit_on_texts(total['comment_text'])
sequences_train  = tokenizer.texts_to_sequences(train_data['comment_text'])
sequences_test   = tokenizer.texts_to_sequences(test_data['comment_text'])
#word index maps every word in the training and set sentences to a token index
word_index       = tokenizer.word_index

#padding to unify length
max_length       = 200
x_train          = pad_sequences(sequences_train, maxlen = max_length)
x_test           = pad_sequences(sequences_test, maxlen = max_length)
x_features_train = train[["caps_vs_length", "num_exclamation_marks", "num_punctuation", "num_unique_words"]]
x_features_test  = test[["caps_vs_length", "num_exclamation_marks", "num_punctuation", "num_unique_words"]]
y_train          = train_data[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
print "End tokenizing sentences"

# word_vectors is obtained from word2vec which maps almost every English word to a 300 size
# vectors which has a short cos distance with close related words
print "Start loading word2vec vectors"
word_vectors = KeyedVectors.load_word2vec_format('../input/GoogleNews-vectors-negative300.bin', binary = True)
print "End loading word2vec vectors"
# now we embed every word in the word_index to have a matrix where the row i will contain the corresponding vector
# obtained from word2vec. This way we can embed and map directly from a word in any seen sentence to its embedded vector
# which will feed the first conv1D layer. From word_index we take the index given a word, and then from embedding_matrix
# we take the word2vec embedding given the index
print "Start embedding matrix creation"
EMBEDDING_DIM = 300
vocabulary_size = min(len(word_index) + 1, NUM_WORDS)
embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))
for word, i in word_index.items():
    if i < NUM_WORDS:
        try:
            embedding_vector = word_vectors[word]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25), EMBEDDING_DIM)

del(word_vectors)
print "End embedding matrix creation"

print "Start creating model"
#in the input layer the vectors that will come in will be of the max_length size
input_embedding = Input(shape = (max_length,))

# after there will the embedding layer that will take the tokenized and padded words and will turn them into
# dense words through the word2vec pretrained model
embedding = Embedding(vocabulary_size,
                      EMBEDDING_DIM,
                      weights = [embedding_matrix],
                      trainable = False)(input_embedding)

#3 parallel convolution layers that will try to extract features from 2-grams, 3-grams, and 4-grams
conv1 = Conv1D(filters = 100, kernel_size = 2, padding = 'same')(embedding)
conv2 = Conv1D(filters = 100, kernel_size = 3, padding = 'same')(embedding)
conv3 = Conv1D(filters = 100, kernel_size = 4, padding = 'same')(embedding)

pool1 = GlobalMaxPooling1D()(conv1)
pool2 = GlobalMaxPooling1D()(conv2)
pool2 = GlobalMaxPooling1D()(conv3)

#at this level after features have been extracted from the convolutional layer, we want to
# plug in the special features.
input_features = Input(shape = (4,))
merged_tensor = concatenate([pool1, pool2, pool2, input_features], axis = 1)
output = Dense(units = 6, activation = 'sigmoid')(merged_tensor)
model = Model(inputs = [input_embedding, input_features], outputs = output)
print "End creting model"

print "Start compiling model"
#binary since it is a multilabel and not a multiclass classification problem
model.compile(loss      = 'binary_crossentropy',
              optimizer = optimizers.Adam(),
              metrics   = ['accuracy'])
print "End compiling model"


print "Start spliting train and validation"
Xtrain, Xval, ytrain, yval = train_test_split(x_train, y_train,
                                              train_size = 0.95, random_state = 233)

Xfeaturestrain, Xfeaturesval, ytrain, yval = train_test_split(x_features_train, y_train,
                                                            train_size = 0.95, random_state = 233)
print "End spliting train and validation"

model.summary()

print "Start TRAINING"
model.fit(x = [Xtrain, Xfeaturestrain], y = ytrain,
          validation_data = ([Xval, Xfeaturesval], yval), verbose = 1)
print "End TRAINING"

print "Start computing prediction"
y_pred = model.predict([x_test, x_features_test])
print "End computing prediction"

print "Start submiting"
submission = pd.read_csv('../input/sample_submission.csv')
submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
submission.to_csv('../input/my_prediction.csv', index = False)
print "END"