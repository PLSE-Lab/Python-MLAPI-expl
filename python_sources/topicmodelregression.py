# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from datetime import date

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

stocknews = pd.read_csv("../input/Combined_News_DJIA.csv")
end_train = date(2014,12,31)
num_train = len(stocknews[pd.to_datetime(stocknews["Date"]) <= end_train])
data = stocknews.filter(regex=("Top.*")).apply(lambda x: "".join(str(x.values)), axis=1).values
vector = CountVectorizer(stop_words="english")
X = vector.fit_transform(data)
lda = LatentDirichletAllocation(learning_method="online")
doc_topic = lda.fit_transform(X)

Y = stocknews["Label"]

train_X = X[:num_train, :]
test_X = X[num_train:, :]
train_Y = Y[:num_train]
test_Y = Y[num_train:]
rf = RandomForestClassifier()
rf.fit(train_X, train_Y)
pred = rf.predict(test_X)

test_index = stocknews["Date"][num_train:]
output = pd.DataFrame(pred, index=test_index, columns=["Predict"])
output.to_csv("prediction.csv", index_label="Date")

print("Finished!")