# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

### DISCLAIMER : This is my first attempt at using Keras for an NLP task. The code I have here has been sourced from a Keras blog - 
###              https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html - with a few modifications of my own.
###
###              I would like to thank Francois Chollet for a simple and neat tutorial. 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np

import os
import pickle
import re
import fastText as ft
from collections import Counter


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print(os.listdir("../input"))

# Loading Train and Test Data
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

# Prepare training data
train_data['fastText_labels'] = train_data.target.apply(lambda x: ' __label__' + str(x))

# Train - fastText
fastText_train = train_data.question_text + train_data.fastText_labels
fastText_train = [x.strip() for x in fastText_train.values.tolist()]

# Write to a training file
training_file = open('fastText_train.txt', 'w')
training_file.write("\n".join(fastText_train))

# Train final classifiers
classifier1 = ft.FastText.train_supervised('fastText_train.txt', lr=0.01, wordNgrams=1, epoch=5)
classifier2 = ft.FastText.train_supervised('fastText_train.txt', lr=0.01, wordNgrams=2, epoch=5)
classifier3 = ft.FastText.train_supervised('fastText_train.txt', lr=0.01, wordNgrams=3, epoch=5)

# Predict test data
predictions1 = classifier1.predict([t.strip() for t in test_data.question_text.tolist()])
predictions2 = classifier2.predict([t.strip() for t in test_data.question_text.tolist()])
predictions3 = classifier3.predict([t.strip() for t in test_data.question_text.tolist()])

# Combine predictions
majority_vote = np.array([])
for i in range(len(predictions1[0])):
    majority_vote = np.append(majority_vote, Counter([predictions1[0][i][0],
                                                   predictions2[0][i][0],
                                                   predictions3[0][i][0]]).most_common(1)[0][0])

# Write submission file
submit = pd.DataFrame({'qid': test_data.qid, 
                       'prediction': pd.Series(majority_vote)})
submit.prediction = submit.prediction.apply(lambda x: re.sub('__label__', '', x))
submit.to_csv('submission.csv', index=False)

# Any results you write to the current directory are saved as output.