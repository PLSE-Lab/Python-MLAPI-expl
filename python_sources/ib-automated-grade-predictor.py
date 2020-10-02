#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf

import keras
from keras.models import Model
from keras.utils import to_categorical
from keras.layers import Input, LSTM, Dense, Dropout, concatenate


# In[ ]:


# UTs, FOAs and Mid-Terms of students in Maths in chronological order of exams
train_marks = np.array([[6, 7, 6, 7, 7, 6, 7],   
                        [3, 4, 3, 5, 6, 6, 5],
                        [2, 1, 3, 1, 2, 3, 3],
                        [5, 4, 5, 5, 5, 6, 6],
                        [3, 3, 4, 4, 5, 5, 7],
                        [7, 7, 7, 7, 7, 7, 7],
                        [2, 3, 2, 2, 3, 3, 2],
                        [1, 2, 2, 2, 2, 1, 2],
                        [4, 3, 4, 4, 3, 5, 4],
                        [3, 3, 4, 3, 4, 3, 4],
                        [5, 4 ,3, 3, 5, 5, 6],
                        [3, 2, 3, 3, 3, 1, 2],
                        [7, 6, 5, 7, 5, 6, 7],
                        [4, 5, 5, 6, 4, 3, 6],
                        [2, 1, 2, 1, 1, 1, 1],
                        [4, 5, 5, 6, 6, 7, 7],
                        [2, 3, 3, 4, 4, 4, 5],
                        [6, 6, 6, 7, 6, 5, 7],
                        [7, 7, 7, 6, 6, 7, 7],
                        [3, 2, 4, 4, 3, 2, 1],
                        [5, 6, 5, 4, 5, 4, 5],
                        [7, 5, 5, 5, 4, 3, 7],
                        [1, 1, 1, 1, 2, 1, 3],
                        [5, 6, 5, 5, 6, 6, 6],
                        [4, 5, 6, 6, 6, 7, 7],
                        [2, 3, 4, 4, 5, 6, 6]]) 

train_marks = train_marks.reshape((26, 7, 1))

max_features = np.max(train_marks, axis=1)
min_features = np.min(train_marks, axis=1)
range_features = max_features - min_features
median_features = np.median(train_marks, axis=1)
mean_features = np.mean(train_marks, axis=1)

train_engineered_features = np.concatenate((max_features, min_features, range_features, median_features, mean_features), axis=1)

train_actual_predicted_grades = np.array([6, 5, 2, 5, 5, 6, 2, 1, 4, 3, 4, 2, 6, 4, 0, 5, 3, 5, 6, 3, 4, 5, 1, 5, 6, 5]) # Predicted Grade of student in Maths (On-Hot Vector from 1-7) (MINUS 1)
train_actual_predicted_grades = to_categorical(train_actual_predicted_grades)
# Eg. [1, 0, 0, 0, 0, 0, 0] represents a score of 1 and [0, 0, 0, 1, 0, 0, 0] a score of 4


#   ** TO-DO : Add real data of at least 50 students over 6 different subjects (mix and match)**

# **Input marks to network**

# In[ ]:


test_marks = Input((7,1,))
engineered_features_var = Input((5,))


# **Long Short Term Memory (LSTM) Network with recurrent dropout followed by Dense layers with dropout.
# A softmax layer to produce categorical output (predicted grade is a whole number from 1 to 7)**

# **We use LSTMs to understand the trend or pattern in the student's marks; to see how he is evolving (improving or otherwise)**

# In[ ]:


lstm = LSTM(50, recurrent_dropout=0.2, activation='relu')(test_marks)

dense1 = Dense(20, activation='relu')(lstm)
dropout = Dropout(0.2)(dense1)

combined_features = concatenate([dropout, engineered_features_var])
dense2 = Dense(10)(combined_features)

predicted_grades = Dense(7, activation='softmax')(dense2)


# **TO-DO : Add hand-made features like median, maximum, minimum, range etc of student marks at the first Dense layer and do not forget to add another dense layer before the softmax output layer**

# **Create and compile model (yet to add the max, min, median etc features to the network)**

# In[ ]:


model = Model(inputs=[test_marks, engineered_features_var], outputs=predicted_grades)
model.compile(loss='categorical_crossentropy', optimizer='adam')


# **Train model**

# In[ ]:


model.fit([train_marks, train_engineered_features], train_actual_predicted_grades, epochs=1000) # validation_data=(np.array(val_marks), np.array(val_actual_predicted_grades)))


# **TO-DO : Use this trained model to predict predicted grades of several students in all their subjects (complete report for colleges**

# In[ ]:


# UTs, FOAs and Mid-Terms of students in Maths in chronological order of exams
test_marks = np.array([[4, 5, 5, 5, 6, 6, 7],
                       [3, 2, 3, 3, 2, 2, 3],
                       [6, 7, 7, 6, 7, 6, 7],
                       [3, 4, 5, 5, 4, 5, 5]]) 

test_marks = test_marks.reshape((4, 7, 1))

max_features = np.max(train_marks, axis=1)
min_features = np.min(train_marks, axis=1)
range_features = max_features - min_features
median_features = np.median(train_marks, axis=1)
mean_features = np.mean(train_marks, axis=1)

test_engineered_features = np.concatenate((max_features, min_features, range_features, median_features, mean_features), axis=1)


# In[ ]:


test_predicted_grades = model.predict([test_marks, test_engineered_features])


# In[ ]:


final_predicted_grades = []
for i in range(0, len(test_predicted_grades)):
    final_predicted_grades.append(list.index(list(test_predicted_grades[i]), max(test_predicted_grades[i])) + 1)


# In[ ]:


final_predicted_grades


# In[ ]:




