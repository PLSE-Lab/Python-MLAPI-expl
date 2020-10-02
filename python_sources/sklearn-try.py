import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv('../input/creditcard.csv')

Y = df['Class'].values
X = df.iloc[:, 1:30]

# Number of data points in the minority class
number_records_fraud = len(df[df.Class == 1])
fraud_indices = np.array(df[df.Class == 1].index)

# Picking the indices of the normal classes
normal_indices = df[df.Class == 0].index

# Out of the indices we picked, randomly select "x" number (number_records_fraud)
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace = False)
random_normal_indices = np.array(random_normal_indices)

# Appending the 2 indices
under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])

# Under sample dataset
under_sample_data = df.iloc[under_sample_indices,:]

X_undersample = under_sample_data.ix[:, 1:30]
y_undersample = under_sample_data.ix[:, under_sample_data.columns == 'Class']

sds = StandardScaler()


X = sds.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X_undersample, y_undersample)

rfClass = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0, criterion='entropy')
rfClass.fit(X_train, Y_train)

print(classification_report(Y_test, rfClass.predict(X_test), digits=5))
