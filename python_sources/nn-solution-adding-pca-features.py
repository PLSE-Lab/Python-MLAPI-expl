import pandas as pd
import  numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from keras import Sequential
from keras.optimizers import Adam
from keras.layers import Dense,Softmax,Dropout,BatchNormalization,Activation
import  keras as keras



from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder

# edit the data_root to reflect the location of your data
DATA_ROOT = r''



def load_train_data(path):
    '''
    loads and returns pandas dataframe with training data
    :param path: path to the location of the training data
    :return: DataFrame
    '''
    df = pd.read_csv(path, skipinitialspace=True)
    return df


def load_test_data(path):
    '''
    loads and returns pandas dataframe with testing data (submission)
    :param path: path to the location of the testing data
    :return: DataFrame
    '''
    df = pd.read_csv(path, skipinitialspace=True)

    df['id'] = df['id'].astype(int)
    return df


def random_submission(df):
    '''
    appends a target column to the pandas dataframe
    :param df: testing dataframe
    :return: DataFrame
    '''
    df['target'] = 0.5
    return df


def make_random_submission():
    '''
    creates and saves the random submission
    :return: None
    '''
    df = load_test_data(DATA_ROOT + 'adsse_test.csv')
    df = random_submission(df)
    # extracting id and target column for the submission
    df = pd.concat([df['id'], df['target']], axis=1, keys=['id', 'target'])
    # saves id and target column to a csv file
    df.to_csv(DATA_ROOT + 'submission.csv', columns=['id', 'target'], index=False)


def logistic_regression_submission():
    '''
    creates and saves the logistic regression submission
    :return: None
    '''
    # loads training and testing data
    train_data = load_train_data(DATA_ROOT + 'adsse_train.csv')
    test_data = load_test_data(DATA_ROOT + 'adsse_test.csv')

    # separates features from target
    X = train_data.ix[:, train_data.columns != 'target'].values
    X_test = test_data.ix[:, test_data.columns != 'id'].values
    y = train_data['target'].values

    pca = PCA()
    pca.fit(np.append(X,X_test,axis=0))
    X=np.append(X,pca.transform(X),axis=1)
    X_test = np.append(X_test,pca.transform(X_test),axis=1)

   # principleComponents = pca.fit_transform(X)
    #np.append(X,principleComponents)

    y = keras.utils.to_categorical(y,num_classes=2)

    def create_model():
        model = Sequential()

        model.add(Dense(128, input_dim=X.shape[1]))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.25))

        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.25))

        model.add(Dense(16))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.25))


        model.add(Dense(2, activation='softmax'))

        #compile
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.005, amsgrad=False), metrics=['accuracy'])
        return model

    clf = create_model()

  #  CallBack = keras.callbacks.TensorBoard(log_dir=r"", histogram_freq=1,
   #                                        batch_size=128, write_graph=True,
    #                                       write_grads=False, write_images=False, embeddings_freq=0,
     #                                      embeddings_layer_names=None, embeddings_metadata=None)


    hist = clf.fit(X,y,epochs=120, verbose=2, batch_size=128, shuffle=True, validation_split=0.15) #, callbacks=[CallBack]

    #visualize
    #from keras.utils import plot_model
   # plot_model(clf, to_file='model.png')


    # makes predictions and saves them to a csv file
    test_data['target'] = clf.predict_proba(X_test)[:, 1]
    test_data = pd.concat([test_data['id'], test_data['target']], axis=1, keys=['id', 'target'])
    test_data.to_csv(DATA_ROOT + 'submission.csv', columns=['id', 'target'], index=False)




if __name__ == '__main__':
    # create a submission from simple logistic regression
    logistic_regression_submission()
    # create a submission with p=0.5 confidence in the prediction
    # uncomment below to create a submission with only 0.5
    # make_random_submission()