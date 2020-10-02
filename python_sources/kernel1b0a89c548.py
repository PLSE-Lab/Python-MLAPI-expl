# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# Input data files are available in the "../input/" directory.
# Any results you write to the current directory are saved as output.

import tensorflow as tf
import tensorflow_docs as tfdocs
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import string
import json


def read_corpus(data_path):
    data = np.array(pd.read_csv(data_path, names=column_names, skiprows=1, na_values="?", sep=",", skipinitialspace=True))
    labels = np.array(pd.read_csv(data_path, names=column_names, skiprows=1, na_values="?", sep=",", skipinitialspace=True)["revenue"].get_values())

    return data, labels


def remove_unnecessary_columns(data_list, column_list, delete_indexes):

    dropped_columns = []

    # remove dropping columns
    for row in data_list:
        new_row = []
        for i in range(len(row)):
            if i not in delete_indexes:
                new_row.append(row[i])
        dropped_columns.append(new_row)

    cleaned_data = dropped_columns
    cleaned_columns = []
    for i in range(column_list.__len__()):
        if i not in delete_indexes:
            cleaned_columns.append(column_list[i])

    return cleaned_data, cleaned_columns


def remove_punctuation(list):
    return ''.join([x for x in list if x not in string.punctuation])


def preprocessing_data(data, columns, map_columns):
    # IMDB_ID - check for duplicates in data
    result = []
    for row in data:
        for i in range(len(row)):
            if i == map_columns['imdb_id']:
                result.append(row[i])

    r = set(result)
    if r.__len__() == data.__len__():
        print("NO DUPLICATES")
    else:
        print("DUPLICATES")

    # BELONGS_TO_COLLECTION R = {0, 1} belongs or not
    for row in data:
        for i in range(len(row)):
            if i == map_columns['belongs_to_collection']:
                if not isinstance(row[i], float):
                    row[i] = 1
                else:
                    row[i] = 0

    # HOMEPAGE - mapping values to numbers 1 if it has otherwise 0
    for row in data:
        for i in range(len(row)):
            if i == map_columns['homepage']:
                if isinstance(row[i], str):
                    row[i] = 1
                else:
                    row[i] = 0

    # STATUS - mapping values to numbers
    for row in data:
        for i in range(len(row)):
            if i == map_columns['status']:
                if row[i] == "Released":
                    row[i] = 1
                else:
                    row[i] = 0

    # ORIGINAL_LANGUAGE - map to values 1 to n in binary
    original_languages = []
    for row in data:
        for i in range(len(row)):
            if i == map_columns['original_language']:
                original_languages.append(row[i])

    original_languages = list(set(original_languages))

    for row in data:
        for i in range(len(row)):
            if i == map_columns['original_language']:
                row[i] = bin(original_languages.index(row[i]))[2:]

    # PRODUCTION_COMPANIES - 4 intervals based  on company rank (range from 1 to 9996), 0 - NAN
    intervals = [[1, 50], [51, 100], [101, 1000], [1001, 10000], [10001, 100000]]
    binary_values = [0, 1, 10, 11, 100]
    for row in data:
        for i in range(len(row)):
            if i == map_columns['production_companies']:
                if not isinstance(row[i], float):
                    temp = row[i].replace("\\", "")
                    temp = temp.replace("s\' ", "s")
                    temp = temp.replace("O\'C", "oC")
                    temp = temp.replace("l\'", "l")
                    temp = temp.replace("l, \'", "l\", \"")
                    temp = temp.replace("d\'A", "dA")
                    temp = temp.replace("w\'s", "ws")
                    temp = temp.replace("y \"Tor\"", "y Tor")
                    temp = temp.replace("L\'i", "L i")
                    temp = temp.replace("L\'A", "L A")
                    temp = temp.replace("I\'m", "I m")
                    temp = temp.replace("d\'I", "d I")
                    temp = temp.replace("d\'O", "d O")
                    temp = temp.replace("\"DIA\"", "DIA")
                    temp = temp.replace("t\'s", "t s")
                    temp = temp.replace("n\'t", "n t")
                    temp = temp.replace("\"Tsar\"", "Tsar")
                    temp = temp.replace("D\'A", "D A")
                    temp = temp.replace("n\' ", "n ")
                    temp = temp.replace("r\'s", "r s")
                    temp = temp.replace("n\'s", "n s")
                    temp = temp.replace("g\'s", "g s")
                    temp = temp.replace("e\'s", "e s")
                    temp = temp.replace("e\'r", "e r")
                    temp = temp.replace("O\' S", "O S")
                    temp = temp.replace("o\' B", "o B")
                    temp = temp.replace("N\' C", "N C")
                    temp = temp.replace("y\'s", "y s")
                    temp = temp.replace("d\'E", "d E")
                    temp = temp.replace("L\'I", "L I")
                    temp = temp.replace("t \'9", "t 9")
                    temp = temp.replace("c\'s", "c s")
                    temp = temp.replace("k\'s", "k s")
                    temp = temp.replace("\"Kvadrat\"", "Kvadrat")
                    temp = temp.replace("\'", "\"")
                    json_format = json.loads(temp)
                    value = json_format[0]['id']
                    index = 0
                    for inter in intervals:
                        if value in range(inter[0], inter[1]):
                            row[i] = binary_values[index]
                        index += 1
                else:
                    row[i] = 0

    # PRODUCTION_COUNTRIES - one hot on range values
    range_list = []
    for row in data:
        for i in range(len(row)):
            if i == map_columns['production_countries']:
                if not isinstance(row[i], float):
                    temp = row[i].replace("D'I", "D I")
                    temp = temp.replace("\'", "\"")
                    json_format = json.loads(temp)
                    country = json_format[0]['iso_3166_1']
                    range_list.append(country)

    range_list = list(set(range_list))

    for row in data:
        for i in range(len(row)):
            if i == map_columns['production_countries']:
                if not isinstance(row[i], float):
                    temp = row[i].replace("D'I", "D I")
                    temp = temp.replace("\'", "\"")
                    json_format = json.loads(temp)
                    country = json_format[0]['iso_3166_1']
                    row[i] = bin(range_list.index(country))[2:]
                else:
                    row[i] = 0


    # RELEASE_DATE - divide in intervals based on year range-years =(1921, 2017)
    intervals = [[1920, 1940], [1941, 1960], [1961, 1980], [1981, 2000], [2000, 2020]]
    binary_map = [0, 1, 10, 11, 100]

    for row in data:
        for i in range(len(row)):
            if i == map_columns['release_date']:
                if not isinstance(row[i], float):
                    part_year = int(row[i].split("/").pop())
                    year = 0
                    if part_year > 17:
                        year = 1900 + part_year
                    else:
                        year = 2000 + part_year
                    index = 0
                    for inter in intervals:
                        if year in range(inter[0], inter[1]):
                            row[i] = binary_map[index]
                        index += 1

    # CREW - counting the number - the bigger the more money movie will make
    count = 0
    for row in data:
        count += 1
        for i in range(len(row)):
            if i == map_columns['crew']:
                if not isinstance(row[i], float):
                    res = len(row[i].split("}")) - 1
                    row[i] = res
                else:
                    row[i] = 0

    delete_indexes = [5, 7, 8, 10, 15, 17, 18, 19, 20]
    data, columns = remove_unnecessary_columns(data, columns, delete_indexes)

    # GENRES 1- 20
    genres = ['Comedy', 'Drama', 'Family', 'Romance', 'Thriller', 'Action', 'Animation', 'Adventure', 'Horror',
              'Documentary', 'Music', 'Crime', 'Science Fiction', 'Mystery', 'Foreign', 'Fantasy', 'War', 'Western',
              'History', 'TV Movie']
    values = []
    value = []
    for row in data:
        for i in range(len(row)):
            if i == map_columns['genres']:
                if not isinstance(row[i], float):
                    value = []
                    temp = row[i].replace("\'", "\"")
                    json_format = json.loads(temp)
                    for el in json_format:
                        value.append(el["name"])

                else:
                    row[i] = 0

        values.append(value)

    columns = np.concatenate((columns, genres))
    for i in range(len(data)):
        for j in range(0, 20):
            if genres[j] in values[i]:
                data[i].append(1)
            else:
                data[i].append(0)

    delete_indexes = [3]  # genres
    data, columns = remove_unnecessary_columns(data, columns, delete_indexes)

    # REVENUE if it nan convert to zero
    for row in data:
        for i in range(len(row)):
            if i == 12 and isinstance(row[i], float):
                row[i] = 0

    print(data[0])

    return data, columns


if __name__ == "__main__":
    print("Hello there!")
    column_names = ['id', 'belongs_to_collection', 'budget', 'genres', 'homepage',
                    'imdb_id', 'original_language', 'original_title', 'overview',
                    'popularity', 'poster_path', 'production_companies', 'production_countries',
                    'release_date', 'runtime', 'spoken_languages', 'status', 'tagline',
                    'title', 'Keywords', 'cast', 'crew', 'revenue']
    map_columns = {}
    for i in range(column_names.__len__()):
        map_columns[column_names[i]] = i

    train_path = "../input/train.csv"
    train_data, train_labels = read_corpus(train_path)
    train_data, column_names_train = preprocessing_data(train_data, column_names, map_columns)

    test_path = "../input/test.csv"
    test_data, test_labels = read_corpus(test_path)
    test_data, column_names_test = preprocessing_data(test_data, column_names, map_columns)

    # convert np-array to pd dataframe
    train_data = pd.DataFrame(data=train_data, columns=column_names_train, index=None)
    test_data = pd.DataFrame(data=test_data, columns=column_names_test, index=None)

    print(train_data)

    sns.pairplot(train_data[["popularity", "status"]], diag_kind="kde")

    train_stats = train_data.describe()
    train_stats.pop("revenue")
    train_stats = train_stats.transpose()

    test_stats = test_data.describe()
    test_stats.pop("revenue")
    test_stats = test_stats.transpose()

    def norm(x):
        return (x - train_stats['mean']) / train_stats['std']

    def norm_test(x):
        return (x - test_stats['mean']) / test_stats['std']
   # normed_train_data = norm(train_data)
   # normed_test_data = norm_test(test_data)


    # MODEL


    def build_model():
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=[len(train_data.keys())]),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])

        optimizer = tf.keras.optimizers.RMSprop(0.001)

        model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=['mae', 'mse'])
        return model


    model = build_model()
    model.summary()

    example_batch = train_data[:10]
    example_result = model.predict(example_batch)

    EPOCHS = 1000

    history = model.fit(
        train_data, train_labels,
        epochs=EPOCHS, validation_split=0.2, verbose=0,
        callbacks=[tfdocs.modeling.EpochDots()])
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()

    plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
    plotter.plot({'Basic': history}, metric="mae")
    plt.ylim([0, 10])
    plt.ylabel('MAE [revenue]')

    plotter.plot({'Basic': history}, metric="mse")
    plt.ylim([0, 20])
    plt.ylabel('MSE [revenue^2]')

    model = build_model()

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    early_history = model.fit(train_data, train_labels,
                              epochs=EPOCHS, validation_split=0.2, verbose=0,
                              callbacks=[early_stop, tfdocs.modeling.EpochDots()])

    plotter.plot({'Early Stopping': early_history}, metric="mae")
    plt.ylim([0, 10])
    plt.ylabel('MAE [revenue]')

    loss, mae, mse = model.evaluate(test_data, test_labels, verbose=2)

    print("Testing set Mean Abs Error: {:5.2f} revenue".format(mae))

    test_predictions = model.predict(test_data).flatten()

    a = plt.axes(aspect='equal')
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values [revenue]')
    plt.ylabel('Predictions [revenue]')
    lims = [0, 50]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)

    error = test_predictions - test_labels
    plt.hist(error, bins = 25)
    plt.xlabel("Prediction Error [revenue]")
    _ = plt.ylabel("Count")