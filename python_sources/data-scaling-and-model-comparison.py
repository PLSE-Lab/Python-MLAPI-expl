#!/usr/bin/env python
# coding: utf-8

# # Goal
# ### I am the new learner in data science. In this time, I'm trying to work on the digit recognition project.
# ### I would like to compare different data scaling and model selection, and it can understand what is the difference accuracy among the methods

# # Importing Library

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
#print(os.listdir())

#Plot Graph
import matplotlib.pyplot as plt

#--------------For Machine Learning-----------------#
##Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
##Modeling
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#CNN
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
##Model Saving and Reading
from sklearn.externals import joblib
from keras.models import load_model


# # Reading Training Data

# In[ ]:


#train_data = pd.read_csv('../input/train.csv')
#test_data = pd.read_csv('../input/test.csv')
train_data = pd.read_csv('../input/digit-recognizer/train.csv')
test_data = pd.read_csv('../input/digit-recognizer/test.csv')


# ### Image Edge Extraction 

# In[ ]:


#Show the training sample digit
plt.figure(num='training_sample',figsize=(10,10)) 
for row in range(1,26):
    plt.subplot(5, 5, row) #row must be >0 for subplot function
    plt.title('Digit:' + str(train_data.iloc[row-1,0]))
    plt.axis('off')
    plt.imshow(train_data.iloc[row-1,1:].values.reshape(28,28))


# # ~~ Start ~~

# # Starting to Split data for Training Model
# 

# In[ ]:


#Split tarin-test sets of 80%-20% respectively, 
#and set the random_state for consistent even I rerun the project.
x_train, x_test, y_train, y_test = train_test_split(train_data.iloc[:, 1:], 
                                                      train_data.iloc[:, 0], test_size = 0.2, random_state = 1)


# In[ ]:


#Data Description
x_train.describe()


# In[ ]:


y_train.value_counts().sort_index()


# # Step 2. Scaling
# 
# #### Reason of Scaling --- https://medium.com/greyatom/why-how-and-when-to-scale-your-features-4b30ab09db5e
# 
# Most of the times, your dataset will contain features highly varying in magnitudes, units and range. But since, most of the machine learning algorithms use Eucledian distance between two data points in their computations, this is a problem.
# 
# If left alone, these algorithms only take in the magnitude of features neglecting the units. The results would vary greatly between different units, 5kg and 5000gms. The features with high magnitudes will weigh in a lot more in the distance calculations than features with low magnitudes.
# 
# To supress this effect, we need to bring all features to the same level of magnitudes. This can be acheived by scaling.
# 
# ## Scaling Method:
# I would like to try all scaling method to understand what is the difference of accuracy in one of the models.
# 
# It contains 4 scaling methods:
# 1. Standarisation
# 2. Mean Normalisation
# 3. Min-Max Scaling
# 4. Unit Vector

# Create the function to find the model score with different scaling methods.
# 
# Further Improvement: 
# 1. Add the hyperparameter tuning into the function for more efficiency and convenient.
# 2. Add the training fitting score(cross validation checking score)

# In[ ]:


#Function to find out all scaling result to compare:
def model_score(model_name , x_train, x_test, y_train, y_test):
    model_path = '../input/digit-recognition-model-backup/' #read back the trained model
    score_save_path = '../input/'
    history_path = '../input/'
    score_list = {}
    scaling_method = {}
    model_algo = {}
    stdsc = StandardScaler()    
    scaling_method = {'original':[x_train, x_test],
                      'Standarisation':[stdsc.fit_transform(x_train), stdsc.fit_transform(x_test)],
                      'Mean Normalisation':[x_train.apply(lambda x: (x - np.mean(x))/(255-0)), 
                                           x_test.apply(lambda x: (x - np.mean(x))/(255-0))],
                      'Unit':[x_train.apply(lambda x: (x - 0)/(255 - 0)),
                                     x_test.apply(lambda x: (x - 0)/(255 - 0))]}
    for method in scaling_method:
        _x_train, _x_test = scaling_method[method]
        
        ###SVM###-----------------------------------------------------------
        if model_name.upper() == 'SVM':
            try:
                model = joblib.load(model_path + 'svm_model' + '_' + method + '.pkl')
                print('Model Reading Success')
            except: #If no exist model, we train
                print('No existed Model, it is fitting...')
                model = svm.SVC(gamma = 0.0001) #Since 'Auto' for orginial data is expensive
                model.fit(_x_train, y_train)
                print('Model is fitted')
            #Model Saving
            #joblib.dump(model, model_path + 'svm_model' + '_' + method + '.pkl') # Kaggle Only provide read mode
        ###Logistics###-----------------------------------------------------    
        elif model_name.upper() == 'LOGISTIC':
            try:
                model = joblib.load(model_path + 'log_model' + '_' + method + '.pkl')
                print('Model Reading Success')
            except:
                print('No existed Model, it is fitting...')
                model = LogisticRegression(random_state = 1)
                model.fit(_x_train, y_train)
                print('Model is fitted')
            #Model Saving
            #joblib.dump(model, model_path + 'log_model' + '_' + method + '.pkl')
        
        ###Decision Tree###-------------------------------------------------
        elif model_name.upper() == 'DECISION TREE':
            try:
                model = joblib.load(model_path + 'tree_model' + '_' + method + '.pkl')
                print('Model Reading Success')
            except:
                model = DecisionTreeClassifier(random_state = 1)
                model.fit(_x_train, y_train)
                print('Model is fitted')
            #Model Saving
            #joblib.dump(model, model_path + 'tree_model' + '_' + method + '.pkl')
        
        ###Random Forecast###-----------------------------------------------
        elif model_name.upper() == 'RANDOM FOREST':
            try:
                model = joblib.load(model_path + 'RF_model' + '_' + method + '.pkl')
                print('Model Reading Success')
            except:
                model = RandomForestClassifier(random_state = 1)
                model.fit(_x_train, y_train)
                print('Model is fitted')
            #Model Saving
            #joblib.dump(model, model_path + 'RF_model' + '_' + method + '.pkl')


        ###Convolutional Neural Network (CNN) ###----------------------------
        elif model_name.upper() == 'CNN':
            num_class = 10 #Digit 0-9
            #Data Reshaping
            if isinstance(_x_train, pd.DataFrame):
                re_x_train = _x_train.values.reshape(_x_train.shape[0], 28,28,1).astype('float32')
                re_x_test = _x_test.values.reshape(_x_test.shape[0], 28,28,1).astype('float32')
            else:
                re_x_train = _x_train.reshape(_x_train.shape[0], 28,28,1).astype('float32')
                re_x_test = _x_test.reshape(_x_test.shape[0], 28,28,1).astype('float32')

            re_y_train = np_utils.to_categorical(y_train, num_class)
            re_y_test = np_utils.to_categorical(y_test, num_class)
            
            try:
                model = load_model(model_path + 'cnn_model' + '_' + method + '.h5')
                print('Model Reading Success')
            except:
                ##model building
                model = Sequential()
                #convolutional layer with rectified linear unit activation
                model.add(Conv2D(32, kernel_size=(3, 3),   #32 convolution filters used each of size 3x3
                                 activation='relu',
                                 input_shape=(28,28,1)))
                model.add(Conv2D(64, (3, 3), activation='relu')) #64 convolution filters used each of size 3x3
                model.add(MaxPooling2D(pool_size=(2, 2))) #choose the best features via pooling
                #randomly turn neurons on and off to improve convergence
                model.add(Dropout(0.25))
                #flatten since too many dimensions, we only want a classification output
                model.add(Flatten())
                #fully connected to get all relevant data
                model.add(Dense(128, activation='relu'))
                #one more dropout for convergence' sake :) 
                model.add(Dropout(0.5))
                #output a softmax to squash the matrix into output probabilities
                model.add(Dense(num_class, activation='softmax'))
                #Model Compile
                model.compile(loss=keras.losses.categorical_crossentropy,
                              optimizer=keras.optimizers.Adadelta(),
                              metrics=['accuracy'])
                #Model Fitting
                history = model.fit(re_x_train,
                          re_y_train,
                          batch_size = 128,
                          epochs = 12,
                          verbose = 1,
                          validation_data = (re_x_test,
                                             re_y_test)
                         )
                #Model Saving
                #model.save(model_path + 'cnn_model' + '_' + method + '.h5')  
                hist_df = pd.DataFrame(history.history) # convert the history.history dict to a pandas DataFrame
                # save to json:  
                hist_json_file = history_path + 'cnn_model' + '_' + method + '_history.json' 
                with open(hist_json_file, mode='w') as f:
                    hist_df.to_json(f)
                
                
        else:
            raise NameError('Model Name Error')
            
        ###Fitting into the test data to get the score###---------------------
        print('Model Saved')
        print('Fitting the score')
        if model_name.upper() in ['SVM', 'LOGISTIC', 'DECISION TREE', 'RANDOM FOREST']:
            score = model.score(_x_test, y_test)
            print(model_name.upper() + '-' + method + "_score: "+ str(score))
            score_list[model_name + '_' + method] = score
            model_algo[method] = model
        if model_name.upper() in ['CNN']:
            score = model.evaluate(re_x_test, re_y_test, verbose = 0)
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])
            score_list[model_name + '_' + method] = score[1]
            model_algo[method] = model
            
    ###To get the score DF -----------------------------------------------------
    score_df = pd.DataFrame.from_dict(score_list, orient='index').reset_index(drop = False)
    split = score_df["index"].str.split("_", n = 1, expand = True)
    score_df['model'] = split[0]
    score_df['scaling_method'] = split[1]
    score_df = score_df.drop(columns = ['index'])
    score_df.columns = ['score', 'model', 'scaling_method']
    score_df = score_df[['model', 'scaling_method', 'score']]
    score_df.to_excel(score_save_path +model_name + '_score.xlsx', index = False)
    print('Score is saved as Excel')
    return(score_df, model_algo)
    


# ## Model_1 - SVM

# In[ ]:


svm_score, svm_algo = model_score('SVM', x_train, x_test, y_train, y_test)


# ## Model_2 - Logistics Regression

# In[ ]:


log_score, log_algo = model_score('Logistic', x_train, x_test, y_train, y_test)


# ## Model_3 - Decision Tree

# In[ ]:


tree_score, tree_algo = model_score('Decision Tree', x_train, x_test, y_train, y_test)


# ## Model_4 - Random Forest

# In[ ]:


rf_score, rf_algo = model_score('Random Forest', x_train, x_test, y_train, y_test)


# ## Model_5 - CNN

# In[ ]:


cnn_score, cnn_algo = model_score('CNN', x_train, x_test, y_train, y_test)


# In[ ]:


#The below function is used to get back the CNN history training record,
#and even for plotting the training and testing trend
import json,codecs
def read_cnn_hist():
    data_scaling = ['original', 'Standarisation', 'Mean Normalisation', 'Unit']
    dict_hist = {}
    backup_path = '../input/digit-recognition-model-backup/'
    for scaling in data_scaling:
        try:
            path = backup_path + 'cnn_model_' + scaling + '_history.json'
            with codecs.open(path, 'r', encoding='utf-8') as f:
                    n = json.loads(f.read())
            dict_hist[scaling] = n
        except:
            print('No such ' + scaling + ' history file')
    return(dict_hist)
cnn_hist = read_cnn_hist()
cnn_hist


# In[ ]:


def show_train_history(history_dict):
    plt.subplots_adjust(wspace =0, hspace =0.7)
    plt.subplot(2,1,1)
    plt.title('Loss -- Train vs Test')
    plt.plot(history_dict['loss'].values())
    plt.plot(history_dict['val_loss'].values())
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper right')
    
    plt.subplot(2,1,2)
    plt.title('Acc -- Train vs Test')
    plt.plot(history_dict['acc'].values())
    plt.plot(history_dict['val_acc'].values())
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper right')

#Here I Just show the Unit Scaling data in training CNN history 
show_train_history(history_dict=cnn_hist['Unit'])


# # Step 3. Model and Scaling Methods Comparison

# In[ ]:


# Read all score excel
import glob
all_data = pd.DataFrame()
input_path = '../input/digit-recognition-model-backup/'
for f in glob.glob(input_path + "*.xlsx"):
    df = pd.read_excel(f)
    all_data = all_data.append(df,ignore_index=True)
all_data


# In[ ]:


# Show the comparison in table:
pivot_table = pd.pivot_table(all_data, index = ['scaling_method'], columns = ['model'], values = 'score').sort_index(ascending = False)
pivot_table.style.highlight_max()


# The above is shown the maximum value among the data scaling method among different models.

# # Step 4. Hyperparameters tuning
# If we seek more accuary model, we can try the hyperparameters tuning. It returns the best hyper-parameter value.(e.g. gamma, regulation_rule value etc.)
# However, the running cost is very expensive. For example, in SVM, I tried it and my local computer's CPU was overheat.
# Therefore, I just write down the code and method for reference.

# In[ ]:


#Hyperparameter Grid Search
#from sklearn.model_selection import GridSearchCV
#from sklearn.pipeline import make_pipeline
#pipe_svc = make_pipeline(svm.SVC(random_state = 1))
#param_range = [0.0001, 0.001, 0.1, 1.0, 100.0, 1000.0]
#param_grid = [{'svc__C' : param_range, 'svc__kernel' :['linear']},
#            {'svc__C' : param_range, 'svc__gamma' : param_range, 'svc__kernel' :['linear']}]
#gs = GridSearchCV(estimator = pipe_svc, param_grid= param_grid, scoring = 'accuracy', cv = 10 , n_jobs = 3)
#gs = gs.fit(X_train_std, x_label)


# The hyper-parameters tuning of Decision Tee example is shown below: 

# In[ ]:


#Hyperparameter Grid Search
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

pipe_tree = DecisionTreeClassifier(random_state = 1)
param_grid = {'max_depth':np.arange(3, 10)}
gs = GridSearchCV(estimator  = pipe_tree, param_grid = param_grid)
gs.fit(x_train, y_train)


# # Step 5. Cross Validation Checking
# Cross validation checking is used to prevent the overfitting problem.
# 
# Below link is shown the reason and the method of CV checking:
# https://towardsdatascience.com/5-reasons-why-you-should-use-cross-validation-in-your-data-science-project-8163311a1e79
# 
# http://localhost:8888/notebooks/Desktop/My_Python/Machine%20Learning/Digit%20Recognition/Digit%20Recognition%20-%20Self_learning.ipynb#4.-Unit-Vector
# 
# For my `model_score` function have shown the acceptable testing score in testing data. Therefore, the model is not overfitting. If we would like to further confirm the model , of course, we can use the cross validation checking. I would like to show the example of cross validation checking of decision tree in below:

# In[ ]:


from sklearn.model_selection import cross_val_score
cross_val_score(DecisionTreeClassifier(random_state = 1), x_train, y_train, scoring='accuracy', cv = 5)


# The above has shown the score of the decision tree model, and it verified the model is not overfitting by specific training data.

# At the end, I would like to use the CNN model to show the predicted digit, and try to verify it whether is correct.

# In[ ]:


unit_x_test = x_test.apply(lambda x: (x - 0)/(255 - 0))
unit_x_test = unit_x_test.values.reshape(unit_x_test.shape[0], 28,28,1).astype('float32')
prediction = cnn_algo['Unit']
prediction = prediction.predict_classes(unit_x_test)


# In[ ]:


prediction


# In[ ]:


#Plot y-train Data
plt.figure(num='cnn_test_fig',figsize=(10,10)) 
for row in range(1,26):
    plt.subplot(5, 5, row) #must be >0
    plt.title('Predicted Digit:' + str(prediction[row-1]))
    plt.axis('off')
    plt.imshow(x_test.reset_index(drop = True).loc[row-1].values.reshape(28,28))


# We can see the picture and predicted digit is almost the same. Therefore, I would like to use this model to predict the test-data for submission.

# # Prediction For Submission

# In[ ]:


# Predict the Test.csv for submission
# I would like to use "Unit" Scaling
x_submission = test_data.apply(lambda x: (x - 0)/(255 - 0)).values.reshape(test_data.shape[0], 28,28,1).astype('float32')
x_submission


# In[ ]:


prediction = cnn_algo['Unit']
predicted = prediction.predict_classes(x_submission)


# In[ ]:


predicted


# In[ ]:


#Plot Submission Test Data set
plt.figure(num='cnn_submission_fig',figsize=(10,10)) 
for row in range(1,26):
    plt.subplot(5, 5, row) #must be >0
    plt.title('Predicted Digit:' + str(predicted[row-1]))
    plt.axis('off')
    plt.imshow(test_data.loc[row-1].values.reshape(28,28))


# In[ ]:


#Submission
submission = pd.DataFrame.from_dict(dict(enumerate(predicted)), orient = 'index')
submission = submission.reset_index()
submission.columns = ['ImageId', 'Label']
submission['ImageId'] = submission['ImageId'] + 1
submission.to_csv('submission.csv', index = False)
submission


# # Summary
# For digit recognition, most of the model using 'Unit' Scaling method would perform better result. It is the same with most of the data scientist statement. 
# 
# However, it has observed that the SVM model using 'Standardisation' would perform better. The reason might be the SVM is focusing on the maximum of the margin, and the data scaling for maximum the margin might perform better after 'Standardisation'.
# 
# During working on this project, I have learned and noted some following points:
# 1. In Data Scaling, try it to use the self-formula, not only rely on sklearn library.
# 2. My model_score function can write to be better, and it can apply on different projects too.
# 3. CNN model is really very useful and powerful for digit recognition.
# 4. When Training model, it really should use cloud computing rather than the local computer, it really was spending lots of resource of my PC.
# 5. Learning coding from other data scientist is very important, and try to summarise them and create my function would learn a lot
# 6. Data is really interesting.
