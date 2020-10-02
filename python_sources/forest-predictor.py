import pandas as pd
from sklearn import metrics
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier

__author__ = 'Achint Verma'
"""
Forest type prediction
Dataset: https://archive.ics.uci.edu/ml/datasets/Forest+type+mapping
"""
class Model:

    def __init__(self):
        testing_data = pd.read_csv('../input/testing.csv')
        training_data = pd.read_csv('../input/training.csv')
        x_train = training_data.iloc[:, 1:].values
        training_data['class'] = training_data['class'].str.strip()
        y_train = training_data['class'].values
        # y_train = self.encode(y_train)

        x_test = testing_data.iloc[:, 1:].values
        testing_data['class'] = testing_data['class'].str.strip()
        y_test = testing_data['class'].values
        # y_test = self.encode(y_test)

        self.xTrain, self.yTrain, = x_train, y_train
        self.xTest, self.yTest = x_test, y_test
        self.model = AdaBoostClassifier(svm.SVC(kernel='linear'), n_estimators=3, learning_rate=0.8,
                                        algorithm='SAMME')


    def train(self):
        self.model.fit(self.xTrain, self.yTrain)


    def predict(self):
        self.prediction = self.model.predict(self.xTest)
        score = metrics.accuracy_score(self.yTest, self.prediction)
        print('\nPrediction accuracy on test data:', score)


    def predict_class(self, input):
        predicted_class = self.model.predict([input])[0]
        if predicted_class == 's':
            class_name = 'Sugi forest'
        elif predicted_class == 'h':
            class_name = 'Hinoki forest'
        elif predicted_class == 'd':
            class_name = 'Mixed deciduous'
        else:
            class_name = 'Other'
        return class_name


    def encode(self, data):
        series = pd.Series(data)
        encoded_data = pd.get_dummies(series)
        return encoded_data


if __name__ == "__main__":
    model = Model()
    model.train()
    model.predict()
    # sample individual class prediction
    predicted_class = model.predict_class(model.xTest[1])
    print('Forest Type:', predicted_class)