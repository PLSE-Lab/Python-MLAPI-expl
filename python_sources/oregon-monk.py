#!/usr/bin/env python
# coding: utf-8

# # Using Monk Library for Image Classification
# 
# [Monk](https://github.com/Tessellate-Imaging/monk_v1) is one of the easiest to learn computer vision library! It is a unified wrapper built on top of popular frameworks like Tensorflow, PyTorch and MxNet. It is a great library for people who have to switch between different frameworks, and also for people who just want to get a model working without writing too much code. It has great features for visualisation and also for tracking the model's training and performance.
# 
# In this notebook, we will go through some of the elemental features of monk which makes it so easy to use!

# ### Cloning the Monk_v1 repository and installing required packges and dependencies

# In[ ]:


get_ipython().system('git clone https://github.com/Tessellate-Imaging/monk_v1.git')


# In[ ]:


get_ipython().system('pip install GPUtil')
get_ipython().system('pip install pylg')


# ### Initialisation :
# 
# prototype.Prototype( *project name*, *experiment name*)
#     
#    This creates files and directories in the following manner :
#    
#        - WORKSPACE -
#                  PROJECT NAME - 
#                               EXPERIMENT NAME -
#                                               1) experiment.json
#                                               2) output - 
#                                                         1) logs (all training graphs are stored here)
#                                                         2) models (all trained models are stored here)
#                                                          
#    We can have different names for the project and experiments. We can store multiple different experiments under the same project. This
#     is really helpful when we want to compare different models for the same data. Monk library has a great feature for comparing different
#     experiments too!
#     
#    We will using the [Oregon wildlife dataset](http://https://www.kaggle.com/virtualdvid/oregon-wildlife) to train a classifier.
#    Our project name will be 'oregon-wildlife' and experiment name 'oregon-pytorch-freezed' since we'll be using PyTorch backend.
#    
#    Monk often outputs the summary of the model after everytime we make a change. Going through it will help you understand what operations/preprocessing it is carrying out. We can also change it if we have some specific operations in mind, different from the default choices. To get started, the default operations usually work pretty well and we won't have to change it most of the times.  

# In[ ]:


import os
import sys
sys.path.append('monk_v1/monk/')

from pytorch_prototype import prototype
ptf = prototype(verbose=1)
ptf.Prototype('oregon-wildlife', 'oregon-pytorch-freezed')


# ### Default mode :
# 
# * We will be using Default mode to train our model. It is the easiest way to train a classification network!
# * All we have to do is send in the required parameters to the Default method and it takes care of everything else.
# * Arguments :
#      - dataset_path = the path of the training data
#      - model_name = specify which model you want to use for the training. You can check the available models using List_Models() method.
#      - freeze_base_network (*type : bool*) = You have the option to either use the pretrained weights or not. True would mean you want
#        to use pretrained weights and only train the final layer of the model. False would mean
#        you want to train the entire network from scratch. Now you know why the experiment name is 'oregon-pytorch-***freezed***'
#      - num_epochs = specify the number of epochs for which you want to train.

# In[ ]:


data_dir = '../input/oregon-wildlife/oregon_wildlife/oregon_wildlife'

ptf.Default(dataset_path = data_dir,
           model_name = 'vgg16',
           freeze_base_network = True,
          num_epochs=7)


# ### Hyperparamater Tuning!
# 
# * One of the most useful features of monk is hyperparameter analysis. Using this we can analyse which hyperparameters would work best for our
# model. Here we define a list of learning rates (lrs) and optimizers we want to test along with the percentage of training data and the number of epochs. The analyser works in the following way -
# 
#    Given a model, it keeps all the other hyperparameters/model constant and runs through the percentage of the training data we want for a given number of epochs. It iterates throught the list of the hyperparameter we wish to analyse, and changes it after it completes going through the train set for the specified number of epochs. 
# 
# This cell demonstrates how we can analyse the learning rate and optimizer using the monk analyser & then update our model according to the results of the analysis.

# In[ ]:


lrs = [0.01, 0.03, 0.06]
percent_data = 5 
epochs = 5

analysis1 = ptf.Analyse_Learning_Rates('lr-cycle', lrs, percent_data, num_epochs=epochs, state='keep_none')


# In[ ]:


optimizers = ['sgd', 'adam', 'momentum-rmsprop']

analysis2 = ptf.Analyse_Optimizers('optim-cycle', optimizers, percent_data, num_epochs=epochs, state='keep_none')


# ### Now we update the training parameters and optimizer according to the analyser results
# 
# By default monk saves the model weights after every epoch. We can change this by using the Training_Params method. This saves precious disk space.

# In[ ]:


ptf.Training_Params(save_intermediate_models = False,
                   num_epochs = 7)

# after reloading num_epochs changes to 10, after reloading some hyperparams change idk why

ptf.optimizer_momentum_rmsprop(0.01, weight_decay = 0.01)
ptf.Reload()


# ### Time to train!
# 

# In[ ]:


ptf.Train()


# ### Evaluating the model
# 
# Since we do not have an expilicit test set for this dataset, we treat our training set as the test set and evaluate our model on it.

# In[ ]:


ptf = prototype(verbose=1);
ptf.Prototype("oregon-wildlife", "oregon-pytorch-freezed", eval_infer=True);

ptf.Dataset_Params(dataset_path=data_dir);
ptf.Dataset();

accuracy, class_based_accuracy = ptf.Evaluate()


# Great! We have a classification model ready with just a few lines of code. We can also see how the model performs over different
# categories of data which gives nice insights on which categories our model finds difficult. We can use this information to improve upon our
# existing model.
# 
# You can check out the Monk library by visiting their [GitHub](https://github.com/Tessellate-Imaging/monk_v1) page. They have a detailed
# roadmap covering everything from how to train your first model using monk to how you can use all the powerful features to your advantage.
