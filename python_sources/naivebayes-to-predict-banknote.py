import pandas as pd
import math
from sklearn.model_selection import train_test_split
import naive_bayes as NB

df = pd.read_csv('../input/banknote-authentication-uci/BankNoteAuthentication.csv', index_col=False)
#print(df.describe())

train_data, test_data = train_test_split(df , test_size = 0.2 , shuffle=True)
#print(test_data.head())

GNB = NB.GaussianNaiveBayes()
GNB.train(train_data)
GNB.test(test_data)
print('Accuracy:' , GNB.accuracy)


