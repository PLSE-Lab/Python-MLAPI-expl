import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.cross_validation import train_test_split

credit = pd.read_csv('../input/creditcard.csv')
df = credit.drop('Time', axis=1)

missing_value_df = df.isnull().astype(int)
missing_value_stat = missing_value_df.apply(sum, axis=0)
print('There are %d parameters including missing value' \
      % missing_value_stat[missing_value_stat > 0].shape[0])

X = df.drop('Class', axis=1)
y = df['Class']

# split data
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, random_state=0)
print('Train set have %d row, %d columns' % (X_train.shape[0], X_train.shape[1]))
print('Test set have %d row, %d columns' % (X_test.shape[0], X_test.shape[1]))

from sklearn.linear_model.logistic import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print('Accuracy:', accuracy_score(y_test, predictions))

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
predictions2 = classifier.predict_proba(X_test)
false_positive_rate, recall, thresholds = \
    roc_curve(y_test, predictions2[:, 1])
roc_auc = auc(false_positive_rate, recall)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, recall, 'b',
         label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out')
plt.show()