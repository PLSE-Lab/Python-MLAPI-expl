import pickle
import tensorflow as tf

from os import path
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense

def preprocess_data(filepath):
    data = pickle.load(open(filepath, "rb"))
    X_train, y_train, X_test, y_test = data["x_train"], data["y_train"], data["x_test"], data["y_test"]
    vocab_size, max_length = data["vocab_size"], data["max_length"]

    # Dataset to huge for 6 hours of compute
    partition = round(len(X_train)/2)

    padded_X_train = pad_sequences(sequences=X_train[:partition], maxlen=max_length)
    padded_X_test = pad_sequences(sequences=X_test[:partition], maxlen=max_length)

    padded_y_train = to_categorical(y_train[:partition], num_classes=2)
    padded_y_test = to_categorical(y_test[:partition], num_classes=2)

    return padded_X_train, padded_X_test, padded_y_train, padded_y_test, vocab_size


def classify(X_train, X_test, y_train, y_test, vocab_size):
    # initialize the Sequential model
    model = Sequential()

    # add layers to the model
    model.add(Embedding(input_dim=vocab_size, output_dim=256, input_length=X_train.shape[1]))
    model.add(LSTM(256))
    model.add(Dense(2, activation="sigmoid"))

    # compile the model
    # chose RMSProp as the optimizer, since it is usually a good choice for recurrent neural networks.
    model.compile(optimizer=RMSprop(), loss='binary_crossentropy', metrics=['accuracy'])

    # print model
    model.summary()

    # run on Tesla P100-PCIE-16GB
    with tf.device('/GPU:0'):
        # train the model
        model.fit(X_train, y_train, batch_size=256, epochs=10, verbose=1)

    # save the model
    model.save("lstm_model.h5")


    # evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=1)  # Should acquire at least 90 % accuracy from the LSTM
    print('Test loss:\t', loss,
          '\nTest accuracy:\t', acc)


def get_file_path(file):
    return path.abspath(file)


def main():
    dataset_sklearn = "../input/keras-data.pickle"
    X_train, X_test, y_train, y_test, vocab_size = preprocess_data(get_file_path(dataset_sklearn))
    classify(X_train, X_test, y_train, y_test, vocab_size)


if __name__ == '__main__':
    main()
    






