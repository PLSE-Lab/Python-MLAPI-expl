#!/usr/bin/env python
# coding: utf-8

# * # make_fun_with_NN_V6
# 
# Functionally: the model is being trained to Summarize, substract, multiply integer numbers.
# 
# I am discussing: how fast the model learns (speed of conversion), volume of data, needed for learning, overfitting, ability 
# to generalize etc.
# 
# Alexei A: I am looking at keras model object instant as a "black box" and analysing its training/test behavior without any preconcieved
# notions on a very simple data set.
# 
# Summary: the simple, 5 layer model easily learns how to add and substract numbers (if the input numbers presented as features, with each decimal digit as separate source feature.
# It needs at least several thousand input samples. Train accuracy is reached at 100% (after about 200 epochs), test data accuracy is about 99%.
# 30 units in the last hidden layer is enough to memorize 10,000 results, evidently NN has some ability for features generalization.
# 
# Training for any reason is not monotonically decreasing loss function, see discussion.
# 
# Multiplication is "not so easy", with test data accuracy only of 70%. The reason is not very clear.
# This Kaggle does not have any input data (generates results itself, just wanted to learn numpy array and pandas dataframe).
# 

# In[ ]:


import os
if not os.path.exists('input'):
    print("dir did not exist(created:",'input')
    os.makedirs('input')
if not os.path.exists('output'):
    print("dir did not exist(created):",'output')
    os.makedirs('output')      


# In[ ]:


# NN learns to Calculate numbers  (making fun of NN)

import pandas as pd
import numpy as np
#import tensorflow as tf
import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Dense, Activation
#import matplotlib.pyplot as plt
import time
from datetime import datetime
import os
#print(os.listdir("./input"))
from sklearn.model_selection import train_test_split

import random
random.seed(1234) # seed, since we need reproducibility of the results


now = time.strftime("%c")
print("<make_fun_with_NN_V2>",now) # Mon May 28 11:03:16 2018
secs_start= time.time()

# INCREASE this value (to about 200 epochs) to reach good accuracy:
ITERATION_COUNT=10 # number of EPOCHS on FIT MODEL (20 epochs it just starting to learn, underfitting) (about 500 epochs overfitting)

# the larger the range, the slower model will learn, the more epochs will be needed:
# on other side,for smallerst range (e.g. 100 rows),the faster will learn, will overfit, also will not generalize well into test data  set:
LR = 1000; UR=1000+10000 # lower bound, upper bound  of the train range******************************************************************************
operat = '+' # training to SUM(+) subtract (-) mupliply (*)

if operat == '+':
    print ('TRAINING TO ADD..')
if operat == '-':
    print ('TRAINING TO SUBSTRACT..')
if operat == '*':
    print ('TRAINING TO MULTIPLY..')

# populate input values:
debug_print = 1 # 1=> printout all results 0=>suppress printout
change_second_operand = 1 # 0= simplified version (can sum number with itself for addition and can find square of number for multiplication)

# filling data (matrix X) with the result of the specified operation (+,-,*) for the specified lower and Upper range:
def fill_data(LR:int, UR:int, X:np.array=[], operat:str="+")->np.array:
    """ "fill_data " and return np.array"""
    random.seed(1234) # seed, since we need reproducibility of the results!
    a=np.array(range(LR,UR)) # first number
    b=np.array(range(LR,UR)) # second number

    #print('<change_second_operand>',change_second_operand)

    if change_second_operand == 1:
        for id,b_val in enumerate (b): # scrolling through input rows of the second operand
            #b_val = b_val +  random.randint(0,UR-LR) # shifting second operand with the random offset
            b_val = b_val +  random.randint(0,10) # shifting second operand with the random offset
            #print('<id>',id,'<b_val>',b_val )
            b[id] =  b_val # assing back into array for the second argument
    # end enumerate b


    #for n in a:
    #    print(n)

    #for i in (range(LR,UR)):
    #    print(i, a[i-1])
    if operat == '+':
        Y = map(lambda x,y:  x  +  y ,  a, b) # 2 4 6 ...18  20 22 24...196 198 # assign the sum

    if operat == '-':
        Y = map(lambda x,y: y - x, a,b) # 1 3 5 7 ... 15 17    19 21 23...195 197 assign the substraction

    if operat == '*':
        Y = map(lambda x,y:x*y, a,b) # 1 4 9 16 25 38 49 64 81 mult=10: 100 121 144 169 196 ...9409 9604 9801
    #Y=np.array(Y)
    #print(type(Y)) # <class 'numpy.ndarray'>
    #print(Y.shape)
    #Y = map(lambda x,y:x-1+y, a,b) # 1 3 5 7 ... 15 17    19 21 23...195 197
    #Y = map(lambda x,y:x*y, a,b) # 1 4 9 16 25 38 49 64 81 mult=10: 100 121 144 169 196 ...9409 9604 9801

    # note: I know that the code below is not "pythonian style",  in python it could be written without any loops, just do not know yet, how to do it in proper way..
    
    for id,y in enumerate (Y): # scrolling through input rows
        # before assignment, ***DISTORT**** the output for about 50% of the results
        #(keeping the output the SAME/correct for about 50%):
        
        rnd = random.randint(0,1)  # 1 to distort, 0=keep not affected, distored every second (on average) result if rand value = 0, result is not distotred, if !=0, distorted
        rnd1 = random.randint(1,10)  # shift the distorted result by this value (pretty far from the result) (the further the distorted number the harder for NN to learn for mult)
        y_distorted = y # init assignment (value will be distorted for some rows below):
        if (rnd!=0):
            y_distorted = y + rnd1 # 50% wrong, 50% of results correct! (distorted result is on average 5 numbers from the correct value )
        
        # ASSIGN LABEL HERE:
        if rnd== 0: X[id,0] = 1  # FLAG the regults: assign the label: 0=>wrong 1=> correct (correct was "not distorted") note: X matrix is initiated with zeroes

        # 0=> label+1..10 digits for result + 11..20 digits first number+ 21..30 digits second number
        # splitting the numbers into decimal values (each decimal position separately) for  first,second source number and for result:
        c=0
        while(a[id]>0): # scroll through digits of the first number
            #print('<id>',id,'<c>',c, '<a[id]>' , a[id],  '<a[id]%10>', a[id]%10)
            X[id,20-c] = a[id]%10 # assigning each next digit into prev pos 20 (20,19,18 etc)
            a[id]=a[id]//10
            c=c+1

        c=0
        while(b[id]>0): # scroll through digits of the second number
            #print('<id>',id,'<c>',c, '<b[id]>' , b[id],  '<b[id]%10>', b[id]%10)
            X[id,30-c] = b[id]%10 # assigning each next digit into prev pos 30 (30,29,28 etc)
            b[id]=b[id]//10
            c=c+1

        c=0
        while(y_distorted>0): # scroll through digits of the RESULT (distorted for part of the rows)
            #print('<id>',id,'<c>',c, '<y_distorted>' , y_distorted,  '<y_distorted%10>', y_distorted%10)
            X[id,10-c] = y_distorted%10 # assigning each next digit into prev pos 10 (10,9,8 etc)
            y_distorted=y_distorted//10
            c=c+1

        #print ('#%d %s' % (id, y ))  # aav use this syntax!
    if debug_print >=1 and '<n_X_0>'=='<>':
        for n in X:
            print('<n_X_0>',n)

    return X  # np.array filled
#  end fill_data *******************************************************************




def conv_to_numbers(X:np.array=[])->np.array:
     """ convert from array of digits (31 cols) to array of 3 numbers (2 source, 1 result, 1 label return np.array"""
# so to easily visualize source numbers and results    
     # example: array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6], ...
     # to [[1,12,6,6],...
     #print(X.shape) # (999, 31)
     
     rownum=0
     #result_array = np.array([0,0,0,0])
     result_array = np.empty((0,4),int)
     for row in X:
         #print(row) # [1 0 0 0 0 0 0 0 0 0 8 0 0 0 0 0 0 0 0 0 4 0 0 0 0 0 0 0 0 0 4]
        #print('<rownum>',rownum,'<i>',i, '<value>',row[ len(row)-1-i])
        #print(i)
         a=0;b=0; res=0; d=1; label=0; c=0
         while(c<10): # scroll through digits of the each number
            #a = x[id,20-c]%10 # account weight of each next digit into prev pos 20 (20,19,18 etc)
            a = a +  (row[20-c] * d) # add current digit to the FIRST number (with the weight), starting from the right
            b = b +  (row[30-c] * d) # add current digit to the SECOND number (with the weight), starting from the right
            res = res +  (row[10-c] * d) # add current digit to the RESULT (with the weight), starting from the right
            #print('<rownum>',rownum,'<c>',c,'<val>', row[20-c],'<a>',a,'<b>',b,'<res>',res)
            c=c+1; d=d*10 # d= decimal weight of the position
         #END WHILE c (scroll through 10 digits for each number)
         label = row[0]
         #print('<a>',a,'<b>',b,'<res>',res,'<label>',label) # <a> 5 <b> 5 <res> 10 <label> 1 ... <a> 7 <b> 7 <res> 18 <label> 0
         rownum =rownum+1
         res_row = np.array([label,res,a,b]) # 4 columns in this sequence, example:  [ 1 10  5  5]
         #print('<res_row>',res_row) # [ 1 10  5  5]
         result_array =(np.vstack((result_array,res_row)))
         #print(result_array)
        #end for row in X ************************
     #print(result_array)
    # example:
# =============================================================================
#  [[ 0  5  1  1] (result is incorrect, labelled incorrect)
#  [ 0  8  2  2]
#  [ 0  9  3  3]
#  [ 0 12  4  4]
#  [ 1 10  5  5]  (result is correct, labelled correct)
# =============================================================================
     return result_array  # np.array filled
 # end conv_to_numbers ***********************************************************************



X= np.zeros((UR-LR,31), dtype=int) # creating 2D matrix for results, 30 columns, 0=label 1..9=result,10..19,20..29=input numbers

# populate data  into X matrix (0  col=label, 1,2=numbers)
data = fill_data(LR, UR, X, operat=operat)  # start from LowerRange=LR, end with UpperRange=UR
# data
print('<0_input_data_shape>',data.shape) # <0_input_data_shape> (99, 31)
print('<0<input_data_columns_count>' , data.shape[1]) # <0<input_data_columns_count> 31



# CONVERT from NUMPY array into PANDAS DATAFRAME:

full_data_with_labels = pd.DataFrame(data)
print('<df.shape>',full_data_with_labels.shape) # <df.shape> (99, 31)

#print('<print_full_train_data>\n', conv_to_numbers(data)) #

print('<1<full_data_with_labels>(shape)',full_data_with_labels.shape )
#print('<2<full_data_with_labels>(head)',full_data_with_labels.head(50) )
# now1

print('<train_data_label_1>',sum(   data[:,0] == 1)     )  # how many labels  in train data point to 1 (correct)
print('<train_data_label_0>',sum(   data[:,0] == 0)     ) # how many labels  in train data point to 0 (incorrect)

train_labels =  data[:,0]  # assigning the zero column as LABELS
train_labels =  pd.DataFrame(train_labels)

#print(train_labels.head(10),'\n<train_labels_0>') # 0,0,1,0,0,1,0,0,0...

full_data=  full_data_with_labels.drop([0], axis=1)
#print(full_data.head(5),'\n<full_data_0>')

# Assign the output to four variables
X_training, X_testing, Y_training, Y_testing = train_test_split(full_data, train_labels, test_size=0.30, random_state=101)
print('Shape of Y_training', Y_training.shape)
print('Shape of Y_testing', Y_testing.shape)

#print('Y_training', Y_training) # [700 rows x 1 columns] 290  0 167  0 486  0 683  1 876  1
#print('Y_testing', Y_testing) # [300 rows x 1 columns] 545  0 298  1 109  0 837  1 194  0

train_data= X_training
train_labels=Y_training


print('<saving train data')
#full_data_with_labels.to_csv("./input/make_fun_with_NN_(FULL_DATA_With_Labels).csv",index=False) # 890 label=1
#train_labels.to_csv("./input/make_fun_with_NN_(train_labels).csv",index=False)

#train_data.to_csv("./input/make_fun_with_NN_(train_data).csv",index=False)
#X_testing.to_csv("./input/make_fun_with_NN_(X_testing).csv",index=False)
#Y_testing.to_csv("./input/make_fun_with_NN_(Y_testing).csv",index=False)

print('<END_CELL1_DATA_FILLED>')
# END CELL 1 *************************************************************************************



# In[ ]:


# START CELL 2 MODEL DEFINITION, TRAIN *************************************************************************************

# change the epochs number to 50 or more (to get 100% accuracy, but not to overfit):
#ITERATION_COUNT=20 # number of EPOCHS on FIT MODEL (20 epochs it just starting to learn, underfitting) (about 500 epochs overfitting)

#print(full_data.head(100),'\n<n_full_data_1>')
now = time.strftime("%c")
print(now) # Mon May 28 11:03:16 2018
print('<train_data>(shape)',train_data.shape)

model = Sequential()
model.add(Dense(500, input_shape=(30,), activation="relu"  )) # 31 cols input
model.add(Dense(250, input_shape=(500,), activation="relu"  )) #
model.add(Dense(100, input_shape=(250,), activation="relu"  )) #
model.add(Dense(30, input_shape=(100,), activation="relu"  )) #
model.add(Dense(1, input_shape=(30,), activation="sigmoid"  )) # 1 output,not 2! sigmoid=>binary_crossentropy


#print('<train_labels>(shape)',train_labels.shape) # <train_labels>(shape) (99, 1)


print('<1_model_instantiation>')
#print(type(model)) # <class 'tensorflow.python.keras.engine.sequential.Sequential'>

# Configures the model for training. (i.e. we assign final params requested for training)
model.compile(loss=keras.losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])
print("<1_call model compile>")

now = time.strftime("%c")

print('<train_started>',now) # Mon May 28 11:03:16 2018
model.fit(train_data, train_labels,   epochs=ITERATION_COUNT) # TRAINING HERE *************************************
now = time.strftime("%c")
print('<train_completed>',now) # Mon May 28 11:03:16 2018
print('<MODEL_TRAINED end cell2')


# In[ ]:




# ***************************************************************************************************************
# WORK WITH TEST  DATA STARTED ***********************************************************************************

# set to 1 ONLY if you need to see TEST data RANGE outside of TRAIN data range (i.e. a "dirty trick")
TEST_DATA_OUTSIDE_OF_RANGE = 0

if TEST_DATA_OUTSIDE_OF_RANGE == 1:
    # generate  TEST DATA, Produce PREDICTIONS against TEST DATA, save into a file:
    now = time.strftime("%c")
    print('<FILLING_TEST_DATA_1>',now) # Mon May 28 11:03:16 2018
    # uncomment next several lines (ONLY IF we want to see train data  to be in a different range vs  train data, i.e. whether NN can generalize to a range of data it never seen before):
    # Fill TEST DATA (for a different range vs train): (training was done for range, starting from 1000 upwards):
    lower_test=10000; upper_test=10000+10000;
    #lower_test=10000; upper_test=10100;
    zero_matrix= np.zeros((upper_test-lower_test,31), dtype=int) # creating 2D matrix for results, 4 columns, 0=label 1=result,2,3=input numbers
    # populate data  into X matrix (0  col=label, 1,2=numbers)
    test_data = fill_data(lower_test, upper_test, zero_matrix, operat=operat)  # start from LowerRange=LR, end with UpperRange=UR
    #pd.DataFrame(test_data).to_csv("./input/make_fun_with_NN(TEST_DATA).csv",index=False)
    print('<Saved_test_data_1>','<lower_test>',lower_test, '<upper_test>',upper_test)
    print('<test_data_1>',test_data)
    #print('<test_data_numbers>',conv_to_numbers(test_data))

    test_data_labels = test_data[:,0]
    #print(test_data_labels)
    test_data_no_labels = np.delete( test_data, 0,  axis=1  )
    #print('<test_data_no_labels_1>',test_data_no_labels)
    #pd.DataFrame(test_data_no_labels).to_csv("./input/make_fun_with_NN(TEST_DATA_NO_LABELS).csv",index=False)
# end TEST_DATA_OUTSIDE_OF_RANGE>0

test_data_no_labels= X_testing # populate from the split
test_data_labels= Y_testing # populate from the split


#predictions = model.predict_classes(train_data)
test_predictions = model.predict_classes(test_data_no_labels)[:,0] # (99, 1)  (spits out two dim array,we need one dim array)
# [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,.. 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
#print('<predictions_20>',pd.DataFrame(test_predictions).head(20))

now = time.strftime("%c")
print('<saving_test_predictions>',now) # Mon May 28 11:03:16 2018
#pd.DataFrame(test_predictions).to_csv("./output/make_fun_with_NN(test_predictions).csv",index=False)

test_predictions = np.array(pd.DataFrame(test_predictions))
test_data_labels = np.array(pd.DataFrame(test_data_labels))

#print('<print_test_predictions>\n',test_predictions)
#print('<print_test_data_labels>\n',test_data_labels)

#test_data_labels = test_data[:,0] # 0 0 0 1 0 1 0.. 0 1 0    (99,)
#print('<test_predictions.shape>',pd.DataFrame(test_predictions).shape,'<test_data_labels.shape>',pd.DataFrame(test_data_labels).shape) # (300, 1) (300, 1)
# COMPARE TEST_DATA Lables with PREDICTIONS: (assign True, if prediction is the same as the label):
pred_correctness_array = np.array(test_predictions)==np.array(test_data_labels) # True=Prediction same as label False=Prediction DIFFERENT than label
# True=>1 False=>0
pred_correctness_array = pred_correctness_array*1 # array([1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
#print('<print_pred_correctness_array>\n',pred_correctness_array) #  (300, 1)  [[1] [0] [1] [0] [1] [1]

pred_count_int = len(test_predictions)
correct_pred_count_int = int(sum((pred_correctness_array==1) *1)) # 97   (1=correct)
wrong_pred_count_int = int(sum((pred_correctness_array==0) *1)) # 2  (0= wrong)
#print('<pred_count>',pred_count_int,'<correct_pred_count>',correct_pred_count_int,'<wrong_pred_count>',wrong_pred_count_int )
correct_ratio_float = float(correct_pred_count_int / (pred_count_int)) # 0.9797979797979798
wrong_ratio_float = float(wrong_pred_count_int / (pred_count_int)) # 0.02

print('<correct_ratio>{0:,.3f}'.format(correct_ratio_float),'<wrong_ratio>{0:,.3f}'.format(wrong_ratio_float))

wrong_res_idx = (pred_correctness_array==0) # true ONLY if prediction is wrong:
#wrong_res_idx =wrong_res_idx [:] # 2 dim=>1 dim
#print('<print_wrong_res_idx>\n',wrong_res_idx) # [False False False False False  True False False False False False False
correct_res_idx = (pred_correctness_array==1) # indexes for correct predictions

#print('<correct_res_idx>',correct_res_idx) #  [[ True] [False] [False]

test_data = np.hstack((test_data_labels,test_data_no_labels) ) # adding test labels to a test data (for visualization of the test data correcness)
#print(pd.DataFrame(test_data).shape) # (30, 31)
#print('print_test_data<label><data>',test_data)

# now1
test_data_numbers = conv_to_numbers(test_data) # converting from a long string into a readable format (numbers)
#print(test_data_numbers.shape) #  (30, 4)
#print('print_test_data\n<test_label><result>(<num1><num2>\n', test_data_numbers[0:1000]) # [[  0 162  77  77] [  1  84  42  42] [  0  53  26  26] [  1   4   2   2]
#print('<print_test_predictions>\n',test_predictions)
#print('<test_predictions>\n',type(test_predictions)) #  <class 'numpy.ndarray'> shape=(6, 1)

test_results_with_predict = np.hstack((test_predictions,test_data_numbers))

#print('<wrong_res_idx>\n',wrong_res_idx)

#print('<print_test_result_with_Predict\n<Test_Predict><Test_label><test_data>\n',test_results_with_predict)


#print('<wrong_predicting_1_cnt',len(wrong_predict_only[wrong_predict_only[:,1] ==1]))
#print('<wrong_predicting_0_cnt',len(wrong_predict_only[wrong_predict_only[:,1] ==0]))

# multiply by vector with correct predictions (assigning all zeroes to the wrong predictions rows):
#correct_predict_only = np.where(pred_correctness_array ==1, test_results_with_predict*1, test_results_with_predict*(-1))
correct_predict_only = np.hstack((pred_correctness_array,test_results_with_predict))
# remove wrong predictions: (first row is 1=correct prediction), first rows=0 wrong preditiction:
correct_predict_only =  correct_predict_only[correct_predict_only[:, 0] != 0]

#print('<CORRECT_test_Predict_ONLY>\n[<correct=1><2_test_pred><3_test_label><4_5_6_data>] \n',correct_predict_only[0:100,:])

print('<correct_predict_count>',len(correct_predict_only[:,0])     )
#print('<correct_predicting_1_cnt',len(correct_predict_only[correct_predict_only[:,1] ==1]))
#print('<correct_predicting_0_cnt',len(correct_predict_only[correct_predict_only[:,1] ==0]))

wrong_predict_only = np.hstack((pred_correctness_array,test_results_with_predict))
# remove correct predictions, filter against wrong prediction rows only: (1=correct, 0=wrong)
wrong_predict_only =  wrong_predict_only[wrong_predict_only[:, 0] == 0]

#wrong_predict_only = np.delete(wrong_predict_only, wrong_predict_only[:,0], axis=1) # commented out no needed

#print('WRONG_test_Predict_ONLY>\n[<correct(0=wrong)>  <2_test_pred>, <3_test_label> ,4_5_6_data] \n',wrong_predict_only[0:100,:])

print('<wrong_predict_count>',len(wrong_predict_only[:,0])     )


# ## Discussion
# Alexei A: I am looking at keras models as "black box" and analysing their behavior without any preconcieved 
# notions on very simple data sets.
# Functionally: the model is trained to SUM, multiply,divide numbers.
# I am re-analysing the most evident things: how fast the model learns, what it can (and could NOT) learn etc.
# Each source number and result presented as 10 dimensions, shifted by the decimal weight (so input number can be up to 10 bln).
# About 50% of results are correct results (and label indicates correct result, label=1).
# About 50% of results are shifted (from the correct result value) and labels indicates "wrong result" (label = 0).
# Model attempts to predict the correct result.
# So total number of source features is: 10+10+10=> 30 and one label. 
# 
# 
# DISCUSSION - DETAILED TRAINING:
# 
# Lets teach the standart/default model to add numbers: 1 hidden layer, relu, for output: softmax activation:
# model = Sequential()
# model.add(Dense(500, input_shape=(30,), activation="relu"  )) # 30 cols input
# model.add(Dense(250, input_shape=(500,), activation="relu"  )) #
# model.add(Dense(100, input_shape=(250,), activation="relu"  )) #
# model.add(Dense(30, input_shape=(100,), activation="relu"  )) #
# model.add(Dense(1, input_shape=(30,), activation="sigmoid"  )) # 1 output,not 2! sigmoid=>binary_crossentropy
# 
# This is the structure of the input data: label=1 points to a CORRECT sum; about 50% of the samples are correct (and labeled as correct), for others result is "distorted" and labeled as incorrect.
# 
# Examples of data:  
# each digits is assigned into its place, starting from the right, below is example of 6+6=> 12
# First goes the label (1=correct, 0=wrong, then result (10 digits), then first operand (10 digits), then second  operand (10 digits):
#      # example: array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6], 
#      # see label = 1 (correct) [[1,12,6,6],...
# 
# Examples of data (data formatted into a numeric form by conv_to_numbers routine:
# Training to ADD:
# <print_full_train_data>
#  [[    1  9220  1000  8220]
#  [    1  3916  1001  2915]
#  [    0  2128  1002  1124]
#  ...
#  [    0 26210 10997 15205]
#  [    1 26064 10998 15066]
#  [    1 31398 10999 20399]]
# <1<full_data_with_labels>(shape) (10000, 31)
# 
# Training to SUBTRACT: (first operand substracted from the second):
#  [[    1  7220  1000  8220]
#  [    1  1914  1001  2915]
#  [    0   124  1002  1124]
#  ...
#  [    0  4216 10997 15205]
#  [    1  4068 10998 15066]
#  [    1  9400 10999 20399]]
# 
# 
# Training to MULTIPLY:
# operat = '*'
# LR = 1000; UR=1000+10000 # lower bound, upper bound  of the train range****************************************************
# 
# 
#  [[        1   8220000      1000      8220]
#  [        1   2917915      1001      2915]
#  [        0   1126250      1002      1124]
#  ...
#  [        0 167209393     10997     15205]
#  [        1 165695868     10998     15066]
#  [        1 224368601     10999     20399]]
# 
# 

# ## Results.
# 
# ### For addition, subtraction:
# 
# 
# If the model has ONLY about 100 rows to train/test, then it overfits (as  expected, training accuracy reaches 100% after several epochs, test accuracy goes down.
# 
# With a normal dataset size (about 10,000 rows and up):
# On training the loss function goes does to 0.0001 (after about 100  epochs), accuracy = 100 on train data.
# 
# ### PROBLEM with training: 
# loss fuction and accuracy are NOT always MONOTONICAL.
# See example:
# Epoch 175/200
# 7000/7000 [==============================] - 1s 196us/step - loss: 1.4667e-07 - acc: 1.0000
# Epoch 176/200
# 7000/7000 [==============================] - 1s 196us/step - loss: 0.2870 - acc: 0.9569
# 
# ### QUESTION:
# We see that after reaching 100% accuracy (on epoch 122 and retaining it for about 50 epochs till epoch 175, shown above), we suddenly lose precision (and loss function increases hundreds of times).  After that the NN for the next 30-50 epochs attempts to "Correct" the training and gradually (again) reaches 100% accuracy and low loss function.
# Can anybody explain WHY this is so (for SOME training error loss "Jumps" and for OTHER training the loss and precision is MONOTONICAL, i.e. loss descreases, accuracy keeps at 100% and never falls back)? Is it a hight Learning Rate (how to change learning rate, how to make it depend upon Epoch number)? Or something else needs to be changed in model configuration? What is the root reason of periodic "Falling back" in accuracy? 
# 
# 
# >TESTS data accuracy is close to *** 99% ***  (for 10,000 rows in data set, i.e. 7,000 in train and 3,000 in test data set).
# 
# Example: we trained on addition the NN:
# Epoch 200/200
# 7000/7000 [==============================] - 1s 157us/step - loss: 0.0052 - acc: 0.9986
# <train_completed> Fri May 31 16:10:58 2019
# <MODEL_TRAINED end cell2
# 
# 
# <saving_test_predictions> Fri May 31 16:14:38 2019
# <correct_ratio>0.994 <wrong_ratio>0.006
# <correct_predict_count> 2982
# <wrong_predict_count> 18
# 
# If I use a "dirty trick" (training in one range and test data in another range): the "transferability" of the results is pretty good into the smaller range (example: train: 1,000...10,000 and test range is 0..1000), accuracy is about 96%. See commented out block in FILLING_TEST_DATA. Trasferrability is NOT good for going to a larger values (e.g. train data is 1,000..10,000 and test data> 10,000) with about 70% of test accuracy. This is as expected, because the NN has never "observed" values in the corresponding higher digital positions.
# 
# 
# ### For Multiplication
# The training gives  100% accuracy, with low loss function (i.e. training converges). However, TEST subset accuracy is ONLY about 70% (?). Reason: probably, against each first/second operand pair, just not enough train data (i.e. only one label, which may be correct result/not correct result). In other words, not enough correct results for any operand combination ("rarified" data) - not enough data to generalize. 
# 
# 
# ### What other/similar stuff can be done?
# 
# ### Prime Numbers.
# 
# What if we use NN to analyze/predict prime numbers? First I want to generate sequences of primes (I know there are ready prime files pregenerated, I just want to to practice with Python Lambda, recursive functions and decorators). 
# Then assess training dataset accuracy: how many values can be memorized, if hiden cell has only about 30..100 nodes in the last layer?  Even to be able to memorize the train data set (with limited number of neurons in the last hidden layer), the NN must "understand" something about structure of the Prime number, correct?
# NN tries to learn some pattern, but what if there is NO pattern? 
# Note: yes, I know that simple NN will NOT be able to successfully generalize re. the prime numbers (otherwise,all modern cryptography would be crashed :-)   but still it is interesting, what will be the limit of "understanding" of primes by NN? For example, the frequency/ratio of occurence (of primes and prime twins) probably will be predicted with good precision.
# NN (probably) will understand that even numbers, numbers divided by 3, 5 etc can not be primes, but will it be able to "memorize" 10,000 values having just 100 nodes? Will be ever reach 100% accuracy on train data? Will we ever reach 99% accuracy on TEST data, what do you think? My gut feeling - NO, this is not possible (without large number of cells in the last hidden layer).
# Even more interesting is "twin primes" (pairs of sequential primes). Certainly, NN could memorize any number of such values by heart (having enough neurons). But what about TEST data- that it never seen:   will we reach in NN even more that 50% of the reproducible accuracy for test data (certainly, all evident things like even numbers have to be excluded from test data set). What is your  feeling?
# 
# 
# 
