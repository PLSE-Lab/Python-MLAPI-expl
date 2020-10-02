import librosa
import pandas as pd
import numpy as np
import os
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

train_model = False

if train_model == True:
    header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label'
    header = header.split()

    file = open('data.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)

    data = pd.read_csv('data.csv')
    already = data.loc[:,'filename']
    didnotwork = []

    count = 0
    genres = "Fake Real".split()
    for g in genres:
        audiopath = 'insert path to audio here'
        for filename in os.listdir(audiopath):
            print(count)
            count += 1
            if (filename in already) == False:
                try:
                    songname = audiopath + '/{filename}'
                    y, sr = librosa.load(songname, mono=True, duration=30)
                    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
                    rmse = librosa.feature.rmse(y=y)
                    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
                    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                    zcr = librosa.feature.zero_crossing_rate(y)
                    mfcc = librosa.feature.mfcc(y=y, sr=sr)
                    to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
                    for e in mfcc:
                        to_append += f' {np.mean(e)}'
                    to_append += f' {g}'
                    file = open('data.csv', 'a', newline='')
                    with file:
                        writer = csv.writer(file)
                        writer.writerow(to_append.split())
                except:
                    print('did not work')
                    didnotwork.append(filename)

    data = pd.read_csv('data.csv')


    genre_list = data.iloc[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(genre_list)
    print(y)

    scaler = StandardScaler()
    scaler.fit(np.array(data.iloc[:, 1:-1], dtype = float))
    X = scaler.transform(np.array(data.iloc[:, 1:-1]))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


    from keras import models
    from keras import layers
    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train,
                        y_train,
                        epochs=100,
                        batch_size=128)


    test_loss, test_acc = model.evaluate(X_test,y_test)
    print('test_acc: ',test_acc)


    predictions = model.predict(X_test)