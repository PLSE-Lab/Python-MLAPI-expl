import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, MultiLabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin

data = pd.read_csv("../input/winemag-data_first150k.csv")
data = data.drop_duplicates("description")
data = data.dropna(subset=['province', 'price'])

X_train = data[:80000]
X_test = data[80000:]
y_train = X_train.price.tolist()
y_test = X_test.price.tolist()

variety_encoder = OneHotEncoder()
variety_encoder.fit(np.array(data.variety).reshape(-1, 1))

province_encoder = MultiLabelBinarizer()

x_train_province = []
for i in data.province:
    x_train_province.append(i.lower().split())

province_encoder.fit(x_train_province)

class ItemSelector(BaseEstimator, TransformerMixin):

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        if self.key == "province":
            x = []
            for i in data_dict[self.key]:
                x.append(i.lower().split())

            fitted = province_encoder.transform(x)
            return fitted
        elif self.key == 'variety':
            fitted = variety_encoder.transform(np.array(data_dict[self.key]).reshape(-1, 1))
            return fitted.toarray()
        elif self.key == 'points':
            return np.transpose(np.matrix(data_dict[self.key]))
        else:
            return data_dict[self.key]

def score(a, b):
    score = 0
    for i in range(len(a)):
        score += np.abs(a[i] - b[i]) / b[i]

    return score / len(a)

def regression(regressor_model):

    regressor = Pipeline([
        ('features', FeatureUnion([
            ('description', Pipeline([
                ('selector', ItemSelector(key='description')),
                ('vectorizer', TfidfVectorizer(stop_words = 'english'))
            ])),
            ('province', Pipeline([
                ('selector', ItemSelector(key='province'))
            ])),
            ('variety', Pipeline([
                ('selector', ItemSelector(key='variety'))
            ])),
            ('points', Pipeline([
                ('selector', ItemSelector(key='points'))
            ]))
        ])),
        ('regressor', regressor_model)
    ])

    #regressor = regressor.fit(X_train, data["price"][:80000])
    regressor = regressor.fit(X_train, y_train)
    print("Fitted")
    # test_X = data.description[80000:]
    #test_y = data["price"][80000:].tolist()
    predictions = regressor.predict(X_test)
    print("predicted")

    return score(predictions, y_test)

# print("Ridge: " + str(regression(Ridge())))
# print("Gradient Boosting Regressor: " + str(regression(GradientBoostingRegressor())))
# arr = [8, 9, 10, 11]
# y = []
# for n in arr:
#     y.append(regression(KNeighborsRegressor(n_neighbors=n)))
#
# plt.plot(arr, y)
# plt.show()

# 11 and 1 give the lowest error
# 11 = 0.5165486593321018
# 5 = 0.5204515283809694
score = regression(KNeighborsRegressor(n_neighbors=10))

print("Mean Percent Error:" + str(score))
