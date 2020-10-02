#!/usr/bin/env python
# coding: utf-8

# [Ludwig](http://https://uber.github.io/ludwig/) is a relatively new AutoML framework open sourced by Uber.
# Let's see how it does on the Titanic....
# TD;LR it is actually very very easy to work with. Handles automatically or almost automaticall all the missing values and dummies. Very easy to install and configure. This kernel gives about 0.75. It is without any feature engineering

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().system('pip install ludwig')


# In[ ]:


from ludwig import LudwigModel
import yaml
import logging

titanic_yaml = """
input_features:
    -
        name: Pclass
        type: category
    -
        name: Sex
        type: category
    -
        name: Age
        type: numerical
        missing_value_strategy: fill_with_mean
    -
        name: SibSp
        type: numerical
    -
        name: Parch
        type: numerical
    -
        name: Fare
        type: numerical
        missing_value_strategy: fill_with_mean
    -
        name: Embarked
        type: category

output_features:
    -
        name: Survived
        type: binary
"""

# train a model
model_definition = yaml.load(titanic_yaml)
print(yaml.dump(model_definition))
model = LudwigModel(model_definition)
training_dataframe = pd.read_csv('../input/train.csv')
print("training...")
train_stats = model.train(training_dataframe, logging_level=logging.INFO)
print("finished training.\n")



# In[ ]:


# obtain predictions
test_dataframe = pd.read_csv('../input/test.csv')
print("predicting...")
predictions = model.predict(test_dataframe, logging_level=logging.INFO)
print("finised predicting\n")
model.close()
print(predictions.head())


# In[ ]:





# In[ ]:


#make submission
submission_df = test_dataframe
predictions.Survived_predictions = predictions.Survived_predictions.astype(np.int8)
submission_df['Survived'] = predictions['Survived_predictions']


# In[ ]:


submission_df.head()
submission_df.to_csv('submission.csv', columns=['PassengerId', 'Survived'], index=False)


# This submission get ~0.75
# with a bit of feature engineering you can get it much much higher (see the tutorials)

# In[ ]:




