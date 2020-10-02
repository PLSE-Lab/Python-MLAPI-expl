import numpy as np
import pandas as pd

filename1 = "../input/train.csv"
filename2 = "../input/test.csv"
training_data = pd.read_csv(filename1, encoding = "UTF-8")
test_data = pd.read_csv(filename2, encoding = "UTF-8")

training_feature = training_data.iloc[:,1:].as_matrix()
training_labels = training_data.iloc[:,0].as_matrix() 
test_feature = test_data.iloc[:,:].as_matrix()

from sklearn.neural_network import MLPClassifier as MLPC
clf = MLPC(hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
clf.fit(training_feature, training_labels)
ans2 = clf.predict(test_feature)