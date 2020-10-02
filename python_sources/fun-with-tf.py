from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn import datasets, model_selection
import tensorflow as tf
import numpy as np

mnst = datasets.load_digits()
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(mnst.data, mnst.target, test_size=0.25)

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=64)]
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
    hidden_units=[100,100,100],
    n_classes=10,
    model_dir="/tmp/mnst_model")
classifier.fit(x=X_train,
    y=Y_train,
    steps=2000)
    
accuracy_score = classifier.evaluate(x=X_test,
    y=Y_test)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))
    
