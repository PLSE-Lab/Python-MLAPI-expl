#!/usr/bin/env python
# coding: utf-8

# This is a general purpose image classifier that can be used for most image classification problems. No knowledge of neural networks or Tensorflow is required to use it. An example of use on the Autism data set is shown below
# source_dir='c://Temp//autism'
# subject='autism'
# t_split=5
# v_split=5
# epochs=30
# batch_size=80
# lr_rate=.0025
# image_size=128
# rand_seed=256
# model_size='L'
# mode='sep'
# 
# TF2_classify(source_dir,mode,subject, t_split=t_split, v_split=v_split, epochs=epochs,batch_size=batch_size,
#          lr_rate=lr_rate,image_size=image_size,rand_seed=rand_seed, model_size=model_size)
#          
# the program operates in one of two modes, If mode-'all' the training, test and validation files are taken from the source_dir and split into train, test and validation files as defined by t_split and v_split integer percentages.
# If model='sep' images are read in from the train, test and valid directories within the source directory. T-split and
# v_split values are not used.
# epochs is the number of training epochs
# lr_rate is the learning rate 
# batch size is the number of images processed as a group during training. 
# The program has 3 models. model_size='L' is a large model based on the MobileNet architecture. It is accurate but propcessing time and memory requirements can be large. For this model set batch_size-80. If you get a resource exhaust error reduce its value. If model_size="M' the program uses a medium sixed model. Execution is fairly fast but it is less accurate. A batch size of 150 works well. If model_size="S" a small model is used. Execution is fast but accuracy is reduced.
# rand_seed sets the seed for the random generators. It's value is arbitrary  but when changed will give a different mix of training, test and validation files when mode='all'.
# image_size is the size that images are converted to for processing. If mode="L" image size is set internally at 224.
# subject is a string you can use to denote the name of files that are stored in your source directory at the conclusion of
# the program.
# At the conclusion of training test results are displayed and you can save the error list to a file in the source directory.
# You are also given the option to run for additional epochs starting from where you left off. At the conclusion of the 
# program two files are stored in the source_dir. One of them is the resulting trained model file. It is labelled as
# subject-image_size-accuracy.h5 , For example autism-224-95.35.h5 means the subject was autism, the image size was 224 X 224 and the accuracy on the test set was 95.35%. This model file can then be used with a prediction program.
# The other file saved is a text file that can be easily converted into a python dictionary. The key is the class number and the value is the associated class name. It is labelled as subject.txt. This file will also be needed by a prediction program to generate a list of classes. 
# If you elected to save the error list it is stored in the source directory as error-list-M-accuracy.txt where M is the model type.
# To use the program you need a python 3 environment in which you have loaded the modules
# tensorflow 2.0, numpy, matplot, sklearn, tqdm cv2,random and PIL

# In[ ]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Activation,Dropout,Conv2D, MaxPooling2D,BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# In[ ]:


# set ansi color values
Cblu ='\33[34m'
Cend='\33[0m'   # sets color back to default 
Cred='\033[91m'
Cblk='\33[39m'
Cgreen='\33[32m'
Cyellow='\33[33m'


# The function below defines the paths for the data. When mode=ALL it takes all the data from a single
# directory. When the mode is not ALL it assumes the data is seperated into training set, validation set and a test set. This function may need to be modified dependent on the names of the sub directories. It also generates an output text file listing the class names and labels. This is useful for use with a companion prediction program that uses the trained model to make predictions. For this data set there is no test set. I used the validation data as the test set and created a seperate validation set in the make_generators function

# In[ ]:


def get_paths(source_dir,output_dir,mode,subject):
    # NOTE if running on kaggle these directories will need to be amended
    # if all files are in a single kaggle input directory change the string 'consolidated'
    # to match the directory name used in the database.
    #if files are seperated in training, test and validation directories change the strings
    # 'train', 'test' and 'valid' to match the directory names used in the database
    if mode =='ALL':
        # all data is in a single directory must be split into train, test, valid data sets
        train_path=os.path.join(source_dir, 'train')    
        classes=os.listdir(train_path) 
        class_num=len(classes)
        test_path=None        
        valid_path=None
       
    else:
        # data is seperated in 3 directories train, test, valid
        test_path=os.path.join(source_dir,'test')
        classes=os.listdir(test_path)
        class_num=len(classes)  #determine number of class directories in order to set leave value intqdm    
        train_path=os.path.join(source_dir, 'train')
        valid_path=os.path.join(source_dir,'validation')
                  
    # save the class dictionary as a text file so it can be used by predictor.py in the future
    #saves file as subject.txt  structure is similar to a python dictionary
    msg=''
    for i in range(0, class_num):
        msg=msg + str(i) + ':' + classes[i] +','
    id=subject  + '.txt'   
    dict_path=os.path.join (output_dir, id)
    f=open(dict_path, 'w')
    f.write(msg)
    f.close()    
    return [train_path, test_path, valid_path,classes]
      
    
   


# the function below makes the model based on the selection of the model size. model_size='L' selects
# the large model which is based on transfer learning for MobileNet. MobileNet works best when the
# image size is selected as 224 X 224 if your image files are that size or above.
# Mobilenet only has pretrained weights for image sizes 224 X 224, 160 X 160, 128 X 128 and 96 X 96.
# Function get_dim takes the input height and width and selects a new height and width from the
# list above that is the closest to what was specified. Otherwise the network would be randomly
# initialized and the training time would be very large.
# 
# If model_size='M' a medium sized model is used. It is faster for computation but less accurate then the
# large model. If model_size='S' a small model is used. Again it is less accurate than the medium and large
# models but has significantly less computations.

# In[ ]:


def make_model(classes,lr_rate, height,width,model_size, rand_seed):
    size=len(classes)
    if model_size=='V1':
        weights='imagenet'
        if height==224:
            Top=True            
            cut=-2
        else:
            Top=False
            cut=-1            
        mobile = tf.keras.applications.mobilenet.MobileNet( include_top=Top,
                                                           input_shape=(height,width,3),
                                                           pooling='avg', weights=weights,
                                                           alpha=1, depth_multiplier=1)
        x=mobile.layers[cut].output
        x=Dense(128, kernel_regularizer = regularizers.l2(l = 0.015), activation='relu')(x)
        x=Dropout(rate=.5,name= 'drop2', seed=rand_seed)(x)
        predictions=Dense (size, activation='softmax')(x)
        model = Model(inputs=mobile.input, outputs=predictions)       
        for layer in model.layers:
            layer.trainable=True
        model.compile(Adam(lr=lr_rate), loss='categorical_crossentropy', metrics=['accuracy'])
        
    elif model_size=='V2':    
        weights='imagenet'
        if height==224:
            Top=True            
            cut=-2
        else:
            Top=False
            cut=-1 
             
        mobile =keras.applications.mobilenet_v2.MobileNetV2(input_shape=(height, width,3),  include_top=Top, weights='imagenet'  )
        x=mobile.layers[cut].output
        x=Dense(128, kernel_regularizer = regularizers.l2(l = 0.015), activation='relu')(x)
        x=Dropout(rate=.5,name= 'drop2', seed=rand_seed)(x)
        predictions=Dense (size, activation='softmax')(x)
        model = Model(inputs=mobile.input, outputs=predictions)
        model.summary()
        for layer in model.layers:
            layer.trainable=True
        #model.compile(Adam(lr=lr_rate), loss='categorical_crossentropy', metrics=['accuracy'])
        model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=lr_rate),loss='binary_crossentropy', metrics=['accuracy'])
        
    return model


# this data set does not have test images only training and validation images. For convinience I will split the training set into a training set and a validation set. Parameter v_split is used to split the training set. Note the validation generator does not used the validation path path[2] but uses the training path path[0]. Also note the use of subset='training for the train generator, and subset='validation in the validation generator. The validation data from the data set is used as the test set in the test generator path[1] is set to be the path to the validation data.

# In[ ]:


def make_generators( paths, mode, batch_size, v_split, classes, height, width, rand_seed, model_size):
    #paths[0]=train path,paths[1]=test path paths[2]= valid path paths[3]=classes
    v_split=v_split/100.0
    file_names=[]
    labels=[] 
    # determine batch_size for test images determine numberof test files
    test_batch_size, test_steps=get_batch_size(paths[1], model_size)
    valid_batch_size, valid_steps=get_batch_size(paths[2], model_size)
    train_batch_size, train_steps=get_batch_size(paths[0], model_size)
    trbatch= 80 if model_size=='V1' else 40
    
    if mode == 'SEP':
        train_gen=ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input,
                horizontal_flip=True,
                samplewise_center=True,
                width_shift_range=.2,
                height_shift_range=.2,
                validation_split=v_split,
                samplewise_std_normalization=True).flow_from_directory(paths[0], target_size=(height, width),
                batch_size=trbatch, seed=rand_seed, class_mode="categorical")
        
        valid_gen=ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input,
                samplewise_center=True,                
                samplewise_std_normalization=True).flow_from_directory(paths[2], 
                target_size=(height, width), batch_size=valid_batch_size,
                seed=rand_seed, shuffle=False, class_mode="categorical")
        
        test_gen=ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input,
                samplewise_center=True,                
                samplewise_std_normalization=True).flow_from_directory(paths[1],
                target_size=(height, width), batch_size=test_batch_size,
                seed=rand_seed, shuffle=False )
        for file in test_gen.filenames:
            file_names.append(file)            
        for label in test_gen.labels:
            labels.append(label)
        
        return [train_gen, test_gen, valid_gen, file_names, labels, [train_steps, test_steps, valid_steps]]
                  
    else:
        # all data is in a single directory there are no test images use validation images as test images
        train_gen=ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input,
                             horizontal_flip=True,
                             samplewise_center=True,
                             validation_split=v_split,
                             samplewise_std_normalization=True).flow_from_directory(paths[0],
                                                                                    target_size=(height, width),
                                                                                    batch_size=batch_size,
                                                                                    subset='training',seed=rand_seed)
        valid_gen=ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input,
                             horizontal_flip=False,
                             samplewise_center=True,
                             validation_split=v_split,
                             samplewise_std_normalization=True).flow_from_directory(paths[0],
                                                                                    target_size=(height, width),
                                                                                    batch_size=batch_size,
                                                                                    subset='validation',
                                                                                    seed=rand_seed, shuffle=False)
        
        
        for file in val_gen.filenames:
            file_names.append(file)
        for label in val_gen.labels:
            labels.append(label)
    return [train_gen, test_gen,valid_gen, file_names, labels ]


# This function initiates training

# In[ ]:


def train(model, callbacks, train_gen, val_gen,steps_list, epochs,start_epoch):
    # steps_list[0]=training steps, steps_list[2]=validations steps
    start=time.time()
    data = model.fit_generator(generator = train_gen,
                               validation_data= val_gen, epochs=epochs, initial_epoch=start_epoch,
                               validation_steps=steps_list[2],callbacks = callbacks, verbose=1)
    stop=time.time()
    duration = stop-start
    hrs=int(duration/3600)
    mins=int((duration-hrs*3600)/60)
    secs= duration-hrs*3600-mins*60
    msg='{0}Training took\n {1} hours {2} minutes and {3:6.2f} seconds {4}'
    print(msg.format(Cblu,hrs, mins,secs,Cend))
    return data
    


# This function produces two plots. The first is a plot of training loss and validation loss versus epochs. The second plot is training accuracy and validation accuracy vs epochs

# In[ ]:


def tr_plot(tacc,vacc,tloss,vloss):
    #Plot the training and validation data
    Epoch_count=len(tloss)
    Epochs=[]
    for i in range (0,Epoch_count):
        Epochs.append(i+1)
    index_loss=np.argmin(vloss)#  this is the epoch with the lowest validation loss
    val_lowest=vloss[index_loss]
    index_acc=np.argmax(vacc)
    val_highest=vacc[index_acc]
    plt.style.use('fivethirtyeight')
    sc_label='best epoch= '+ str(index_loss+1)
    vc_label='best epoch= '+ str(index_acc + 1)
    fig,axes=plt.subplots(nrows=1, ncols=2, figsize=(15,5))
    axes[0].plot(Epochs,tloss, 'r', label='Training loss')
    axes[0].plot(Epochs,vloss,'g',label='Validation loss' )
    axes[0].scatter(index_loss+1,val_lowest, s=150, c= 'blue', label=sc_label)
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[1].plot (Epochs,tacc,'r',label= 'Training Accuracy')
    axes[1].plot (Epochs,vacc,'g',label= 'Validation Accuracy')
    axes[1].scatter(index_acc+1,val_highest, s=150, c= 'blue', label=vc_label)
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    plt.tight_layout
    #plt.style.use('fivethirtyeight')
    plt.show()
    
 


# The function below processes the results of making predictions on the test set using the model weights that were saved 
# for the lowest validation loss. It determines the number of prediction errors and the accuracy. IT has two modes of 
# operation dependent on the setting of the input parameter kaggle. On Kaggle when you commit your kernel it does not allow
# user input. So I created the kaggle parameter to deal with that problem. With kaggle=True this function will print out a list of errors that includes the filename, true class, predicted class and probability of the prediction only if there are less than 35 errors. When kaggle=False, the user is given the option to print out the error. In either case an output file is created in the output directory that contains this error list data. The function also creates a horizontal bar chart of the number of errors by  class.

# In[ ]:


def display_pred(output_dir, pred, file_names, labels, subject, model_size,classes, kaggle):    
    trials=len(labels)
    errors=0
    e_list=[]
    prob_list=[]
    true_class=[]
    pred_class=[]
    x_list=[]
    index_list=[]
    pr_list=[]
    error_msg=''    
    for i in range (0,trials):
        p_class=pred[i].argmax()
        if p_class !=labels[i]: #if the predicted class is not the same as the test label it is an error
            errors=errors + 1
            fname=os.path.basename(file_names[i])
            e_list.append(fname)  # list of file names that are in error
            true_class.append(classes[labels[i]]) # list classes that have an eror
            pred_class.append(classes[p_class]) #class the prediction selected
            prob_list.append(100 *pred[i][p_class])# probability of the predicted class
            add_msg='{0:^24s}{1:5s}{2:^20s}\n'.format(classes[labels[i]], ' ', file_names[i])
            error_msg=error_msg + add_msg
            
    accuracy=100*(trials-errors)/trials
    print('{0}\n There were {1} errors in {2} trials for an accuracy of {3:7.3f}{4}'.format(Cblu,errors, trials,accuracy,Cend))
    if kaggle==True and errors<26:
        ans='Y'
    else:
        ans='N'
    if kaggle==False:
        ans=input('To see a listing of prediction errors enter Y to skip press Enter\n ')
    if ans== 'Y' or ans  =='y':
        msg='{0}{1}{2:^20s}{1:3s}{3:^20s}{1:3s}{4:^20s}{1:5s}{5}{6}'
        print(msg.format(Cblu, ' ', 'File Name', 'True Class', 'Predicted Class', 'Probability', Cend))
        for i in range(0,errors):
            msg='{0}{1:^20s}{0:3s}{2:^20s}{0:3s}{3:^20s}{0:5s}{4:^6.2f}'
            print (msg.format(' ',e_list[i], true_class[i], pred_class[i], prob_list[i]))
    if kaggle==True:
        ans='Y'
    else:
        ans=input('\nDo you want to save the list of error files?. Enter Y to save or press Enter to not save  ')
    if ans=='Y' or ans=='y':
        acc='{0:6.2f}'.format(accuracy)
        if model_size=='L':
            ms='Large'
        elif model_size=='M':
            ms= 'Medium'
        else:
            ms= 'Small'
        header='Classification subject: {0} There were {1} errors in {2} tests for an accuracy of {3} using a {4} model\n'.format(subject,errors,trials,acc,ms)
        header= header +'{0:^24s}{1:5s}{2:^20s}\n'.format('CLASS',' ', 'FILENAME') 
        error_msg=header + error_msg
        file_id='error list-' + model_size + acc +'.txt'
        file_path=os.path.join(output_dir,file_id)
        f=open(file_path, 'w')
        f.write(error_msg)
        f.close()
    for c in classes:
        count=true_class.count(c)
        x_list.append(count)
        pr_list.append(c)
    for i in range(0, len(x_list)):  # only plot classes that have errors
        if x_list[i]==0:
            index_list.append(i)
    for i in sorted(index_list, reverse=True):  # delete classes with no errors
        del x_list[i]
        del pr_list[i]      # use pr_list - can't change class_list must keep it fixed
    fig=plt.figure()
    fig.set_figheight(len(pr_list)/4)
    fig.set_figwidth(6)
    plt.style.use('fivethirtyeight')
    for i in range(0, len(pr_list)):
        c=pr_list[i]
        x=x_list[i]
        plt.barh(c, x, )
        plt.title( subject +' Classification Errors on Test Set')
    if errors>0:
        plt.show()
    if kaggle==False:
        ans=input('Press Enter to continue')
    return accuracy        


# this function is called at the end of the program. It saves the trained model to the output directory
# so it can be used with a companion prediction program. The model is saved with the title
# subject-height-width-model.txt where subject is the subject the user entered, height and width are those set by the user, model is the model_size, L, M or S.

# In[ ]:


def save_model(output_dir,subject, accuracy, height, width, model, weights):
    # save the model with the  subect-accuracy.h5
    acc=str(accuracy)[0:5]
    tempstr=subject + '-' +str(height) + '-' + str(width) + '-' + acc + '.h5'
    model.set_weights(weights)
    model_save_path=os.path.join(output_dir,tempstr)
    model.save(model_save_path)    


# This function receives the model weights that were saved during training for the lowest validation loss.
# It then makes predictions on the test set. Note when the data set has no test set it makes predictions on the validation set.

# In[ ]:


def make_predictions( model, weights, test_gen, lr):
    config = model.get_config()
    pmodel = Model.from_config(config)  # copy of the model
    pmodel.set_weights(weights) #load saved weights with lowest validation loss
    pmodel.compile(Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])    
    print('Training has completed. Now loading test set to see how accurate the model is')
    results=pmodel.evaluate(test_gen, verbose=0)
    print('Model accuracy on Test Set is {0:7.2f} %'.format(results[1]* 100))
    predictions=pmodel.predict_generator(test_gen, verbose=0)     
    return predictions


# This function is called when the specified number of epochs have completed training. It operates in two modes based on the setting of the parameter kaggle. If kaggle is false this function print out the test set accuracy then asks the user if they wish to enter a value for how many more epochs to run, or the user can enter H to end the program.
# If kaggle-true this function evaluates the accuracy.If the accuracy is >=95% the program halts. If the accuracy is between 85% to 95% it will automatically run the training for 6 more epochs. If the accuracy is less than 85% it will automatically run the training for 8 more epochs.
# This dependency on the kaggle parameter was necessitated because when committing a kernel Kaggle does
# not allow user input.

# In[ ]:


def wrapup (output_dir,subject, accuracy, height,width, model, weights,run_num, kaggle):
    if accuracy >= 95:
        msg='{0} With an accuracy of {1:5.2f} % the results appear satisfactory{2}'
        print(msg.format(Cgreen, accuracy, Cend))
        if kaggle:
            save_model(output_dir, subject, accuracy, height, width , model, weights)
            print ('*********************Process is completed******************************')            
            return [False, None]        
    elif accuracy >=85 and accuracy < 95:
        if kaggle:
            if run_num==2:
                save_model(output_dir, subject, accuracy, height, width , model, weights)
                print ('*********************Process is completed******************************')
                return [False, None]
            else:
                print('running for 6 more epochs to see if accuracy improves')
                return[True,6] # run for 8 more epochs
        else:
            msg='{0}With an accuracy of {1:5.2f} % the results are mediocure. Try running more epochs{2}'
            print (msg.format(Cblu, accuracy,Cend))
    else:
        if kaggle:
            if run_num==2:
                save_model(output_dir, subject, accuracy, height,width , model, weights)
                print ('*********************Process is completed******************************')
                return [False, None]
            else:
                print('Running for 8 more epochs to see if accuracy improves')
                return[True,8] # run for 8 more epochs
        else:
            msg='{0} With an accuracy  of {1:5.2f} % the results would appear to be unsatisfactory{2}'
            print (msg.format(Cblu, accuracy, Cend))
            msg='{0}You might try to run for more epochs or get more training data '
            msg=msg + 'or perhaps crop your images so the desired subject takes up most of the image{1}'
            print (msg.format(Cblu, Cend))
    
    tryagain=True
    if kaggle==False:
        while tryagain==True:
            ans=input('To continue training from where it left off enter the number of additional epochs or enter H to halt  ')
            if ans =='H' or ans == 'h':
                run=False
                tryagain=False
                save_model(output_dir, subject, accuracy, height,width , model, weights)                                      
                print ('*********************Process is completed******************************')
                return [run,None]
            else:
                try:
                    epochs=int(ans)
                    run=True
                    tryagain=False
                    return [run,epochs]
                except ValueError:
                    print('{0}\nyour entry {1} was neither H nor an integer- re-enter your response{2}'.format(Cred,ans,Cend))


# This function sets the width and height of the MobileNet model. Mobile Net models only
# have pre-trained weights for the dimensions in the list. This function takes the original
# width and height and determine the closest standard width and height.

# In[ ]:


def set_dim(w,h):
    wh_list=[224,160, 128, 96 ]
    if w  in wh_list and h in wh_list and w==h:
        return(w,w)
    else:
        x=h if h>w else w
        # find closest value to what is in the list
        delta_min=np.inf
        for s in wh_list:
            delta =abs( x-s)
            if delta< delta_min:
                delta_min=delta
                h=s    
    return (h,h)


# This function calculates the proper batch size  and steps so that the  data generator goes through the list of files exactly one time. It counts all the files within the directory
# including files in all sub directories(classes)

# In[ ]:


def get_batch_size(dir, model_size): 
    max_batch_size = 80 if model_size=='V1' else 40
    count=0    
    dir_list=os.listdir(dir) # lists content of  directory
    for d in dir_list:       # d is one of the class sub directories
        d_path=os.path.join(dir,d)
        if os.path.isdir(d_path):                        
            file_list=os.listdir(d_path)
            for f in file_list:      # f is a file in a class directory
                f_path=os.path.join(d_path, f)
                if os.path.isfile(f_path):
                    count=count + 1
    factors=[]
    # find out if number of good files is divisable
    for i in range (1, int(count/2) +2):        
        if count % i ==0:
            factors.append(i) 
    # find the largest factor that is less than or equal to 100    
    end=len(factors)-1 
    for i in range(end, -1, -1):
        if factors[i]<=max_batch_size:
            batch_size=int(factors[i])
            steps=int(count/batch_size)
            break    
    return (batch_size, steps)


# This is the main function. It calls the path function to get the test set paths. Then calls the make_generators function to create the generators. Then the make_model function is called to create the model. This function includes two classes tr and val used to subclass the keras callbacks class.
# The tr class has a function on_batch_end called end the end of every batch while training. The function monitors the training accuracy. If the training accuracy does not improves for 10 consecutive batches it
# lowers the learning rate by a factor of .95. It continues to monitor the training accuracy until the
# accuracy reaches above 90%. At that point this function stops operating and the on_epoch_end function
# defined in the val class takes over. IT monitors the validation loss at the end of each epoch. If the
# validation loss is the lowest to date it saves that loss value and also saves the model weights. If the loss at the end of the current epoch is above the stored lowest loss the function reduces the learning rate by a factor of .5.
# This main function inititates training by calling the train function. When training is complete the history object results contains a list of the training loss, validation loss, training accuracy and validation accuracy. These values are saved and appended to a list of these values stored at the end of each training cycle. The function then calls the make_predictions function to make predictions on the test set using the model weights saved for the lowest validation loss. It then calls the display_predictions function to display the results of the predictions. Finally it calls the wrapup function which processes the prediction results and determines if more epochs will be run.

# In[ ]:


def TF2_classify(source_dir, output_dir, mode, subject, v_split=5, epochs=20, batch_size=80,
                 lr_rate=.002, height=224, width=224, rand_seed=128, model_size='V1', kaggle=False):
    model_size=model_size.upper()
    width, height =set_dim(width, height)
    mode=mode.upper() 
    height, width = set_dim(height, width)
    paths=get_paths(source_dir,output_dir,mode,subject)
    #paths[0]=train path,paths[1]=test path paths[2]= valid path paths[3]=classes 
    gens=make_generators( paths, mode, batch_size, v_split, paths[3], height, width, rand_seed, model_size)
    #gens[0]=train generator gens[1]= test generator  gens[2]= validation generator
    #gens[3]=test file_names  gens[4]=test labels gens[5]=[train_steps, test_steps, valid_steps]
    model=make_model(paths[3],lr_rate, height, width, model_size, rand_seed) 
    class val(tf.keras.callbacks.Callback):
        # functions in this class adjust the learning rate 
        lowest_loss=np.inf
        best_weights=model.get_weights()
        lr=float(tf.keras.backend.get_value(model.optimizer.lr))
        epoch=0
        highest_acc=0
        
        def __init__(self):
            super(val, self).__init__()
            self.lowest_loss=np.inf
            self.best_weights=model.get_weights()
            self.lr=float(tf.keras.backend.get_value(model.optimizer.lr))
            self.epoch=0
            self.highest_acc=0
            
        def on_epoch_end(self, epoch, logs=None): 
            val.lr=float(tf.keras.backend.get_value(self.model.optimizer.lr))
            val.epoch=val.epoch +1            
            v_loss=logs.get('val_loss')
            v_acc=logs.get('accuracy')
            if val.highest_acc<v_acc:
                val.highest_acc=v_acc
                val.best_weights=model.get_weights()
            if v_acc<=.95 and v_acc<val.highest_acc:
                lr=float(tf.keras.backend.get_value(self.model.optimizer.lr))
                ratio=v_acc/val.highest_acc  # add a factor to lr reduction
                new_lr=lr * .7 * ratio
                tf.keras.backend.set_value(model.optimizer.lr, new_lr)
                msg='{0}\n current accuracy {1:7.4f} % is below 95 % and below the highest accuracy of {2:7.4f}, reducing lr to {3:11.9f}{4}'
                print(msg.format(Cyellow, v_acc* 100, val.highest_acc, new_lr,Cend))   
            if val.lowest_loss > v_loss:
                msg='{0}\n validation loss improved,saving weights with validation loss= {1:7.4f}\n{2}'
                print(msg.format(Cgreen, v_loss, Cend))
                val.lowest_loss=v_loss
                val.best_weights=model.get_weights()
                
            else:
                 if v_acc>.95 and val.lowest_loss<v_loss:
                        # reduce learning rate based on validation loss> val.best_loss
                        lr=float(tf.keras.backend.get_value(self.model.optimizer.lr))
                        ratio=val.lowest_loss/v_loss  # add a factor to lr reduction
                        new_lr=lr * .7 * ratio
                        tf.keras.backend.set_value(model.optimizer.lr, new_lr)
                        msg='{0}\n current loss {1:7.4f} exceeds lowest loss of {2:7.4f}, reducing lr to {3:11.9f}{4}'
                        print(msg.format(Cyellow, v_loss, val.lowest_loss, new_lr,Cend))
           
    callbacks=[val()]
    run_num=0
    run=True
    tacc=[]
    tloss=[]
    vacc=[]
    vloss=[]
    start_epoch=0
    while run:
        run_num=run_num +1
        if run_num==1:
            print(' Starting Training Cycle')
        else:
            print('Resuming training from epoch {0}'.format(start_epoch))
        results=train(model,callbacks, gens[0], gens[2],gens[5], epochs,start_epoch)
        # returns data from training the model - append the results for plotting
        tacc_new=results.history['accuracy']
        tloss_new=results.history['loss']
        vacc_new =results.history['val_accuracy']
        vloss_new=results.history['val_loss']
        for d in tacc_new:  # need to append new data from training to plot all epochs
            tacc.append(d)
        for d in tloss_new:
            tloss.append(d)
        for d in vacc_new:
            vacc.append(d)
        for d in vloss_new:
            vloss.append(d)       
        last_epoch=results.epoch[len(results.epoch)-1] # this is the last epoch run
        tr_plot(tacc,vacc,tloss,vloss) # plot the data on loss and accuracy
        bestw=val.best_weights  # these are the saved weights with the lowest validation loss
        lr_rate=val.lr 
        predictions=make_predictions(model, bestw, gens[1], lr_rate)
        accuracy=display_pred(output_dir, predictions, gens[3], gens[4], subject, model_size, paths[3], kaggle)        
        model_path=os.path.join(source_dir, 'autism-224-224-95.71.h5')        
        decide=wrapup(output_dir,subject, accuracy, height, width, model, bestw,run_num, kaggle)
        run=decide[0]        
        if run==True:
            epochs=last_epoch + decide[1]+1
            start_epoch=last_epoch +1
        
           


# The code below determines where the source of the data resides, and sets parameters for the TF_classify function. The source_dir is the main directory where the data is stored. mode if set to all indictes that all the data is in the source directory. If mode is not ALL the program assumes that the data is partitioned into train, test and validation subdirectories of the source dir. Subject is a string used to define the subject of the classification and is used to labeloutput file names. epochs is the number of initial epochs to train. output_dir is the directory where output files will be stored. batch size is the training batch size. For the large model it is best set to 80 otherwise a resource exhaust error may get thrown. model_size is a single string character either 'L', 'M' or 'S' for selection of the large, medium or small model.v-split is used when there is either no validation set or if you regard it as to small to be representative of the data set probability distribution. If you elect to create your own
# validation set v_split is the percentage of training samples that will be used for validation. Set the markup for make_generators to see how to handle that case. Otherwise v_split is not used. The kagle parameter is a boolean. It was added because Kaggle commits do not allow user inputs. When set to true
# it preempts any user input code and automatically makes a selection based on model accuracy on the test set.

# In[ ]:


d=r'/kaggle/input'
d_list=os.listdir(d)
p1=os.path.join(d, d_list[0])
source_dir=os.path.join(p1, 'DisasterModel')
output_dir=r'/kaggle/working'
subject='disasters'
v_split=8
epochs=15
batch_size=80
lr_rate=.002
height=224
width=224
rand_seed=100
model_size='V1'
mode='SEP'
kaggle=True  # added to deal with fact that kaggle 'commit' does not allow user entry
              # set to True if you are doing a kaggle commit
if kaggle:
    output_dir=r'/kaggle/working'
    

TF2_classify(source_dir, output_dir, mode,subject, v_split= v_split, epochs=epochs,batch_size= batch_size,
         lr_rate= lr_rate,height=height, width=width,rand_seed=rand_seed, model_size=model_size, kaggle=kaggle)


    


# In[ ]:





# In[ ]:




