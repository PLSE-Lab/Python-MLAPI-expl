# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

"""
This is just a naive bayesian classifier meant to classify each grocery item to the proper class.

As the underline event model of the classifier I used the multinomial distribution
"""

filepath = '../input/grocery-exprenses/spesa_dataset.csv'

class BayesianTypeClassifier:
    """
    A class representing a naive bayes type classifier,
    that classify an item based on its name
    """

    def __init__(self, filename: str):
        """
        The BayesanTypeClassifier constructor
        :param filename the csv file to read and train the classifier
        """
        self.filename = filename
        self.df = pd.read_csv(filename, delimiter=';', encoding=open(filename).encoding) # the dataframe
        self.model =  make_pipeline(TfidfVectorizer(), MultinomialNB()) # the model
        self.le = preprocessing.LabelEncoder() # the label encoder

    def generate_random_test_sample(self, n=100) -> tuple:
        """
        Generate a random sample from the classifier data frame
        :param n sample size
        :return tuple containing the sample names and the target class
        """
        df1 = self.df[['nome', 'tipo']]
        rand_names = df1.sample(100)

        # encoding in a proper way the types
        le_1 = preprocessing.LabelEncoder()
        le_1.fit(df1['tipo'].tolist())

        target = list(le_1.transform(rand_names['tipo'].tolist()))

        return rand_names['nome'], target

    def fit(self):
        """
        train the model on the given dataset
        """
        names = np.array(self.df['nome'].tolist())
        types = self.df['tipo'].tolist()

        # encoding in a proper way the types
        self.le.fit(types)

        # list of the corresponding type of the names
        corr_types = np.array(list(self.le.transform(types)))

        self.model.fit(names, corr_types)

    def predict_category(self, s):
        """
        Given a name the model predict the category
        :param s the name whose category should be predicted
        :return the predicted category
        """
        pred = self.model.predict([s])
        return self.le.inverse_transform([pred[0]])




model = BayesianTypeClassifier(filepath)
model.fit() # train the model

test_data, test_target = model.generate_random_test_sample(50) # generate a random sample of data

i = 0
c = 0

for data in test_data:
    print(f'{data} type: {model.predict_category(data)[0]}, expected type: {model.le.inverse_transform([test_target[i]])[0]}', end=' ')
    print(model.predict_category(data)[0] == model.le.inverse_transform([test_target[i]])[0])

    if model.predict_category(data)[0] == model.le.inverse_transform([test_target[i]])[0]:
        c += 1

    i += 1

print(f'accuracy {(c / len(test_data)) * 100}%') # todo: improve accuracy of the classifier

#TODO: analyze the classifier accuracy
