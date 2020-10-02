#!/usr/bin/env python
# coding: utf-8

# <img src="https://rhyme.com/assets/img/logo-dark.png" align="center" width=150 height=37.5></img>
# <h2 style="text-align:center;">Poker Hand Classification using Random Forests</h2>

# Standard Imports:

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings;warnings.simplefilter('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ### Task 1: About the Data
# ***

# ![Alt text](https://farm4.staticflickr.com/3585/3299226824_4637597b74_z_d.jpg "Cards by bl0ndeeo2, Creative Commons License (https://flic.kr/p/62xpc7) ")

# The [dataset](http://archive.ics.uci.edu/ml/datasets/Poker+Hand) we'll be exploring in this post is the Poker Hand data from the UCI Machine Learning Repository.
# 
# Each record in the dataset is an example of a hand consisting of five playing cards drawn from a standard deck of 52. Each card is described using two attributes (suit and rank), for a total of 10 predictive attributes. The target column describes the hand, with the possibilities being:    
# 
#     0: Nothing in hand; not a recognized poker hand     
#     1: One pair; one pair of equal ranks within five cards     
#     2: Two pairs; two pairs of equal ranks within five cards     
#     3: Three of a kind; three equal ranks within five cards     
#     4: Straight; five cards, sequentially ranked with no gaps     
#     5: Flush; five cards with the same suit     
#     6: Full house; pair + different rank three of a kind     
#     7: Four of a kind; four equal ranks within five cards     
#     8: Straight flush; straight + flush     
#     9: Royal flush; {Ace, King, Queen, Jack, Ten} + flush     
#     
# The order of cards is important, which is why there are 480 possible Royal Flush hands as compared to 4 (one for each suit).

# In[ ]:


#reading the data
poker_df = pd.read_csv('/kaggle/input/poker-hand-testing.data')


# ### Task 2: Separate the Data into Features and Targets

# In[ ]:


poker_df.head()


# In[ ]:


#defining the columns and labels
poker_df.columns = ['first_suit', 'first_rank', 'second_suit', 'second_rank', 'third_suit', 'third_rank', 
                    'fourth_suit', 'fourth_rank', 'fifth_suit', 'fifth_rank', 'hand']

labels = ['zilch', 'one_pair', 'two_pair', 'three_of_a_kind', 'straight',
          'flush', 'full_house', 'four_of_a_kind', 'straight_flush', 'royal_flush' ]

#segregating features and target variables
x = poker_df.iloc[:, 0:9]
y = poker_df.hand


# ### Task 3: Evaluating Class Balance

# In[ ]:


#using yellowbricks API
from yellowbrick.classifier import ClassBalance, ROCAUC, ClassificationReport, ClassPredictionError

poker_balance = ClassBalance(size = (1080,720), labels = labels)
poker_balance.fit(y)

poker_balance.show()


# ### Task 4: Upsampling from Minority Classes

# In[ ]:


poker_df.loc[poker_df['hand'] >= 5, 'hand'] = 5
y = poker_df.hand
labels = ['zilch', 'one_pair', 'two_pair', 'three_of_a_kind', 'straight', 'fluh_or_better'] 


# ### Task 5: Training the Random Forests Classifier

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

RFC = RandomForestClassifier(n_estimators = 150,
                             class_weight = 'balanced',
                             n_jobs = -1)

RFC.fit(x_train, y_train)


# ### Task 6: Classification Accuracy

# In[ ]:


y_pred = RFC.predict(x_test)

from sklearn.metrics import accuracy_score
print('Accuracy - ', accuracy_score(y_test, y_pred))


# ### Task 7: ROC Curve and AUC 

# In[ ]:


from yellowbrick.classifier import ROCAUC

rocauc = ROCAUC(RFC, size=(1080,720), classes = labels)

rocauc.score(x_test, y_test)
rocauc.show()


#  ### Task 8: Classification Report Heatmap

# In[ ]:


from yellowbrick.classifier import ClassificationReport

report = ClassificationReport(RFC, size=(720,540), classes = labels, cmap = 'PuBu')
report.score(x_test, y_test)
report.show()


# ### Task 9: Class Prediction Error

# In[ ]:


from yellowbrick.classifier import ClassPredictionError

error = ClassPredictionError(RFC, size=(1020,720), classes = labels)
error.score(x_test, y_test)
error.show()


# The above used metrics are helpful in testing the accuracy and various metrics in multi class problems. However even after upsampling, our model still sufferes from imbalance. Correcting this imbalance isn't in the scope of this project.
