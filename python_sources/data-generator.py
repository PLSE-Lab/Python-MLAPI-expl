from keras.backend import sparse_categorical_crossentropy
from keras.optimizers import Adam
from numpy import array, argmax
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, GRU, Dropout
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint
import sklearn
import tensorflow as tf
def load_doc(filename):
    file = open(filename, mode='rt', encoding='utf-8')
    text = file.read()
    file.close()
    return text

def to_sentences(doc):
    return doc.lower().strip().split('\n')

def to_pairs(text):
    listss = text.strip().split('\n')
    X = list()
    y = list()
    for lists in listss:
        a=lists.split('  ')
        X.append(a[0])
        y.append(a[1])
    return X,y

def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def max_length(lines):
    return max(len(line.split()) for line in lines)

def encode_sequence(tokenizer, lenth, lines):
    X = tokenizer.texts_to_sequences(lines)
    X = pad_sequences(X, maxlen=lenth, padding='post')
    return X

def encode_output(sequences, vocab_size):
    ylist = list()
    for sequence in sequences:
        encoded = to_categorical(sequence, num_classes=vocab_size)
        ylist.append(encoded)
    y = array(ylist)
    y.reshape(sequences.shape[0],sequences.shape[1],vocab_size)
    return y



# text = load_doc('dataset.txt')
# X,y = to_pairs(text)


X_file = '../input/europarl-v7.fr-en.en'
Y_file = '../input/europarl-v7.fr-en.fr'
NUM_EPOCHS = 100
BATCH_SIZE = 2
NUM_TRAIN = 0
NUM_TEST = 0



X = load_doc(X_file)
X = to_sentences(X)

y = load_doc(Y_file)
y = to_sentences(y)

eng_length = max_length(X)
print('English Max Length: %d' % (eng_length))
eng_toenizer = create_tokenizer(X)
eng_vocab_size = len(eng_toenizer.word_index) + 1
print('English Vocabulary Size: %d' % eng_vocab_size)

fr_tokenizer = create_tokenizer(y)
fr_vocab_size = len(fr_tokenizer.word_index) + 1
print('French Vocabulary Size: %d' % fr_vocab_size)
fr_length = max_length(y)
print('French Max Length: %d' % (fr_length))




def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
    model = Sequential()
    model.add(Embedding(src_vocab, n_units, input_length=src_timesteps,mask_zero=True))
    model.add(LSTM(n_units))
    model.add(RepeatVector(tar_timesteps))
    model.add(LSTM(n_units, return_sequences=True))
    model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
    return model


def define_model_2(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
    model = Sequential()
    # Embedding
    model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
    # Encoder
    model.add(Bidirectional(GRU(128)))
    model.add(RepeatVector(tar_timesteps))
    # Decoder
    model.add(Bidirectional(GRU(128, return_sequences=True)))
    model.add(TimeDistributed(Dense(512, activation='relu')))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])
    return model


# def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
#     model = Sequential()
#     model.add(Embedding(src_vocab, n_units, input_length=src_timesteps,input_shape=trainX.shape,mask_zero=True))
#     model.add(LSTM(n_units))
#     model.add(RepeatVector(tar_timesteps))
#     model.add(LSTM(n_units, return_sequences=True))
#     model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
#     return model

def a_data_generator(inputXPath,inputYPath, batch_size, lb , mode="train", aug=None):
    eng = open(inputXPath,"r")
    french = open(inputYPath,"r")
    while True:
        X = []
        y = []
        while len(X)< batch_size:
            lineX = eng.readline()
            lineY = french.readline()
            if lineX =="":
                lineX.seek(0)
                lineY.seek(0)
                lineX = eng.readline()
                lineY = french.readline()
                if mode == "eval":
                    break

            X.append(lineX.strip().lower())
            y.append(lineY.strip().lower())

        if mode =='train':
            trainX = encode_sequence(eng_toenizer, eng_length, X[:batch_size-5])
            trainY = encode_sequence(fr_tokenizer, fr_length, y[:batch_size-5])
            trainY = encode_output(trainY, fr_vocab_size)
        else:
             trainX = encode_sequence(eng_toenizer, eng_length, X[batch_size - 4:])
             trainY = encode_sequence(fr_tokenizer, fr_length, y[batch_size - 4:])
             trainY = encode_output(trainY, fr_vocab_size)
         # prepare validation data
         # testX = encode_sequence(eng_toenizer, eng_length, X[5:])
         # testY = encode_sequence(hin_tokenizer, hin_length, y[5:])
         # testY = encode_output(testY, hin_vocab_size)
        print('Train X shape',trainX.shape)
        print('Train y shape',trainY.shape)
        yield(trainX , trainY)



# define model
model = define_model(eng_vocab_size, fr_vocab_size, eng_length, fr_length, 256)
model.compile(optimizer='adam', loss='categorical_crossentropy')
model
# summarize defined model
print(model.summary())
# plot_model(model, to_file='model.png', show_shapes=True)

filename = "model.h5"
checkpoint = ModelCheckpoint(filename,monitor='val_loss',verbose=1,save_best_only=True, mode='min')
# history = model.fit(trainX, trainY, epochs=400, batch_size=1, validation_data=(testX, testY), callbacks=[checkpoint], verbose=2)
train_gen = a_data_generator(X_file,Y_file,32,0)
test_gen = a_data_generator(X_file,Y_file,32,0,"test")
history = model.fit_generator(train_gen,
                              steps_per_epoch=10,
                              validation_data=test_gen,
                              validation_steps=10,
                              epochs=NUM_EPOCHS,
                              use_multiprocessing=True)

model.save("model.h5")
model.save('translator.h5')
import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
#
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_sequence(model, tokenizer, source):
    prediction = model.predict(source, verbose=0)[0]
    integers = [argmax(vector) for vector in prediction]
    target = list()
    for i in integers:
        word = word_for_id(i, tokenizer)
        if word is None:
            break
        target.append(word)
    return ' '.join(target)

# def evaluate_model(model, tokenizer,sources,raw_dataset):
#     actual,predicted = list(), list()
#     for i, source in enumerate(sources):
#         source = source.reshape((1,source.shape[0]))
#         translation = predict_sequence(model, eng_toenizer , source)
#         raw_target, raw_src = raw_dataset[i]
#         if i<10 :
#             print('src=[%s], target=[%s], predicted=[%s]' %(raw_src, raw_target, translation))
#         actual.append([raw_target.split()])
#         predicted.append(translation.split())
#         # print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
#         # print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
#         # print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
#         # print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))