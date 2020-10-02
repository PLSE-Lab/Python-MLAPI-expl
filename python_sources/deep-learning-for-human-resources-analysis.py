#!/usr/bin/env python
# coding: utf-8

# The Deep Learning course provided knowledge on Machine Learning techniques applicable to Artificial Intelligence. This project is inspired by https://www.kaggle.com/ludobenistant/hr-analytics and concerns the application of a deep learning model to a dataset of 14999 industry employees to determine if they will remain working at the structure or they left.
# The dataset contains the following parameters (features):
# 
#     satisfaction_level: Satisfaction level (numerical value between 0 and 1)
#     last_evaluation: Last evaluation (numerical value between 0 and 1)
#     number_project: Number of assigned projects
#     average_montly_hours: Average monthly hours worked
#     time_spend_company Time spent in company
#     Work_accident: If they have had an accident at work (binary value)
#     promotion_last_5years: If they have had a promotion in the last 5 years (binary value)
#     sales: Department (characters transformed into sequential numbers)
#     salary: Salary (band, high-medium-low transformed into numbers 3-2-1)
# 
# and the following label
# 
#     left: If the employee left or not (binary value)
# 
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


hr_data = pd.read_csv("../input/HR_comma_sep.csv")


# In[ ]:


hr_data.info()


# First of all, data analysis was performed, using data mining techniques, to display the distributions of the various parameters on the model taken into consideration in the form of graphs. The dataset consists of 14999 records. Each record consists of 10 values, associated with the 10 attributes that describe the various employees taken into consideration by the company to conduct the analysis path.
# 

# In[ ]:


axes_ind = 0
for col in ['satisfaction_level','last_evaluation','average_montly_hours']:
    
    if axes_ind > 7: break
    
    df1 = hr_data[hr_data['left']==0][col]
    df2 = hr_data[hr_data['left']==1][col]
    max_col = max(hr_data[col])
    min_col = min(hr_data[col])

    plt.hist([df1, df2], 
                 bins = 20,
                 edgecolor='black',
                 range=(min_col, max_col), 
                 stacked=True)

    plt.legend(('Stayed','Left'), loc='best')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('Count')
    
    axes_ind += 1
    plt.show()

for col in ['number_project','Work_accident','salary','promotion_last_5years','sales']:
    cat_xt = pd.crosstab(hr_data[col], hr_data['left'])
    cat_xt.plot(kind='bar', stacked=True, title= col)
    plt.xlabel(col)
    plt.legend(('Stayed','Left'), loc='best')
    plt.xticks(rotation=60)
    plt.ylabel('count')


# 

# Import keras library

# In[ ]:


import numpy
# fix random seed for reproducibility
numpy.random.seed(7)
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split, GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.optimizers import Adam

dataset =hr_data


# DEEP LEARNING
# Variables Transformation:
# I decided to change the "left" binary attribute with the corresponding categorical, replacing the value 0 with stayed and 1 with left, to facilitate the reading of the graphs. The remaining binary values have not changed.
# For the generation of boxplots, performed in the phase of elimination of the outliers, the numerical attributes were normalized with the z-score method. For the clustering phase I normalized the numerical data on a scale from 0 to 1 using min-max scaling.
# The values of the continuous attributes have been grouped into bins to limit their dispersion and facilitate the reading in the subsequent graphs that have been used in the analysis path.
# Finally, to be able to apply deep learning algorithms faster, I have transformed the sales and salary categorical values into sequential integer values.

# In[ ]:


# transform columns in sequantial
salary = dataset['salary'].unique()
salary_mapping =dict(zip(salary, range(0, len(salary) + 1)))
dataset['salary_int'] = dataset['salary'].map(salary_mapping).astype(int)
depart = dataset['sales'].unique()
depart_mapping =dict(zip(depart, range(0, len(depart) + 1)))
dataset['depart_int'] = dataset['sales'].map(depart_mapping).astype(int)
dataset= dataset.drop(["salary","sales"],axis=1)

# slpit dataset in train and test
train,test=train_test_split(dataset, test_size=0.2)

# split train and test into input (X) and output (Y) variables
X_TRAIN = train.drop(["left"],axis=1)
Y_TRAIN= train["left"]
X_TEST = test.drop(["left"],axis=1)
Y_TEST=  test["left"]

print (salary_mapping)
print (depart_mapping)

print (train.shape)


# To optimize the model, the Exhaustive Grid Search technique was used with a series of values  that I thought were the most appropriate to the problem. Although not an optimal method but in order to minimize the times that otherwise would have been exponential given the range of the dataset (14999 records), I divided the parameters of the model into two groups. With the first group:
#     batch_size = [10, 20, 40]
#     epochs = [50, 100, 200]
#     neurons = [16, 32, 64]
# I ran the model and got the following best result: Best: 0.759313 using {'batch_size': 20, 'epochs': 50, 'neurons': 16}
# Having fixed the aforementioned parameters, I have made a second gridsearch run again with the others:
#     dropout_rate = [0.0, 0.4, 0.8]
#     weight_constraint = [1, 3, 5]
#     activation = ['softmax', 'relu', 'sigmoid']
#     learn_rate = [0.01, 0.1]
#     momentum = [0.2, 0.8]
# 
# Finding the following best result:
# Best: 0.767231 using {'activation': 'relu', 'batch_size': 20, 'dropout_rate': 0.4, 'epochs': 50,' learn_rate ': 0.01,' momentum ': 0.8,' neurons': 16, 'weight_constraint': 3}
# 

# In[ ]:





def create_model(neurons=16, dropout_rate=0.4, weight_constraint=3,activation='relu',learn_rate=0.01, momentum=0.8):
    #create model
    model2 = Sequential()
    model2.add(Dense(neurons, input_dim=9, kernel_initializer='uniform',
                        activation=activation, kernel_constraint=maxnorm(weight_constraint)))
    model2.add(Dense(neurons, kernel_initializer='uniform', activation=activation, kernel_constraint=maxnorm(weight_constraint)))
    model2.add(Dense(neurons, kernel_initializer='uniform', activation=activation, kernel_constraint=maxnorm(weight_constraint)))
    model2.add(Dense(neurons, kernel_initializer='uniform', activation=activation, kernel_constraint=maxnorm(weight_constraint)))
    model2.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    # Compile model
    optimizer = SGD(lr=learn_rate, momentum=momentum)
    model2.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model2
best_model = create_model()
history_best = best_model.fit(X_TRAIN, Y_TRAIN, epochs=50, batch_size=20, validation_split=0.2,verbose=1)

                                                    


# But the result clearly wrong, since it does not converge, both in terms of accuracy and loss as shown by the following graphs

# In[ ]:


# list all data in history
print(history_best.history.keys())
# summarize history for accuracy
plt.plot(history_best.history['acc'])
plt.plot(history_best.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history_best.history['loss'])
plt.plot(history_best.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()


# So I changed the approach and, by studying similar kernels, I set the parameters very close to the defaults.
# Making a few attempts and analyzing the resulting graphs I have established an acceptable compromise in the following model:

# In[ ]:


def create_model_easy():
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=9, activation='sigmoid'))
    model.add(Dense(16, activation='sigmoid'))
    model.add(Dropout(0,7))
    model.add(Dense(16, activation='sigmoid'))
    model.add(Dropout(0,7))
    model.add(Dense(16, activation='sigmoid'))
    model.add(Dropout(0,7))
    model.add(Dense(16, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    
    optimizer = Adam(decay=0.001)
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Fit the model
my_model = create_model_easy()
history = my_model.fit(X_TRAIN, Y_TRAIN, epochs=500, batch_size=30, validation_split=0.2,verbose=1)

print("Model finished")


# In[ ]:


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()


# In[ ]:


scores = my_model.evaluate(X_TEST, Y_TEST)
print("\n%s: %.2f%%" % (my_model.metrics_names[0], scores[0]*100))
print("\n%s: %.2f%%" % (my_model.metrics_names[1], scores[1]*100))


# In[ ]:




