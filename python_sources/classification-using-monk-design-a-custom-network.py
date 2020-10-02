#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# 
# ## Author - Tessellate Imaging - https://www.tessellateimaging.com/
# ### Monk Library - https://github.com/Tessellate-Imaging/monk_v1
# 
# ### Monk is an opensource low-code tool for computer vision and deep learning
# 
# ### Monk features
# 
#     low-code
#     unified wrapper over major deep learning framework - keras, pytorch, gluoncv
#     syntax invariant wrapper
# 
# ### Enables
# 
#     to create, manage and version control deep learning experiments
#     to compare experiments across training metrics
#     to quickly find best hyper-parameters
# 
# ### At present it supports transfer learning, and custom model creation for pytorch and gluon backend, but we are working each day to incorporate
# 
#     GUI based custom model creation
#     various object detection and segmentation algorithms
#     deployment pipelines to cloud and local platforms
#     acceleration libraries such as TensorRT
#     preprocessing and post processing libraries
# 
# ### To contribute to Monk AI or Monk Object Detection repository raise an issue in the git-repo or dm us on linkedin
# 
#     Abhishek - https://www.linkedin.com/in/abhishek-kumar-annamraju/
#     Akash - https://www.linkedin.com/in/akashdeepsingh01/
# 
# 

# In[ ]:


# Clone the library


# In[ ]:


get_ipython().system(' git clone https://github.com/Tessellate-Imaging/monk_v1.git')


# In[ ]:


# Install the pre-requisistes


# In[ ]:


get_ipython().system('pip install -r monk_v1/installation/requirements_kaggle.txt')


# In[ ]:


# Add to path


# In[ ]:


import sys
sys.path.append("/kaggle/working/monk_v1/monk/")


# In[ ]:


# Unzip datasets


# In[ ]:


get_ipython().system('unzip -q /kaggle/input/aerial-cactus-identification/train.zip')


# In[ ]:


get_ipython().system('unzip -q /kaggle/input/aerial-cactus-identification/test.zip')


# In[ ]:


# Import prototype


# In[ ]:


from pytorch_prototype import prototype


# In[ ]:


# Set Dataset


# In[ ]:


gtf = prototype(verbose=1);
gtf.Prototype("sample-project-1", "sample-experiment-1");

gtf.Dataset_Params(dataset_path="train/",
           path_to_csv="/kaggle/input/aerial-cactus-identification/train.csv",
        input_size=(32, 32), batch_size=16, shuffle_data=True, num_processors=3);

gtf.apply_random_horizontal_flip(train=True, val=True);
gtf.apply_normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], train=True, val=True, test=True);

gtf.Dataset();


# In[ ]:


# Create model and debug simultaneously


# In[ ]:


network = [];
network.append(gtf.convolution(output_channels=16));
network.append(gtf.batch_normalization());
network.append(gtf.relu());
network.append(gtf.convolution(output_channels=16));
network.append(gtf.batch_normalization());
network.append(gtf.relu());
network.append(gtf.max_pooling());
gtf.debug_custom_model_design(network);


# In[ ]:


subnetwork = [];
branch1 = [];
branch1.append(gtf.convolution(output_channels=16));
branch1.append(gtf.batch_normalization());
branch1.append(gtf.convolution(output_channels=16));
branch1.append(gtf.batch_normalization());

branch2 = [];
branch2.append(gtf.convolution(output_channels=16));
branch2.append(gtf.batch_normalization());

branch3 = [];
branch3.append(gtf.identity())

subnetwork.append(branch1);
subnetwork.append(branch2);
subnetwork.append(branch3);
subnetwork.append(gtf.concatenate());


network.append(subnetwork);
gtf.debug_custom_model_design(network);



# In[ ]:


network.append(gtf.convolution(output_channels=32));
network.append(gtf.batch_normalization());
network.append(gtf.relu());
network.append(gtf.max_pooling());
gtf.debug_custom_model_design(network);


# In[ ]:


subnetwork = [];
branch1 = [];
branch1.append(gtf.convolution(output_channels=32));
branch1.append(gtf.batch_normalization());
branch1.append(gtf.convolution(output_channels=32));
branch1.append(gtf.batch_normalization());

branch2 = [];
branch2.append(gtf.convolution(output_channels=32));
branch2.append(gtf.batch_normalization());

branch3 = [];
branch3.append(gtf.identity())

subnetwork.append(branch1);
subnetwork.append(branch2);
subnetwork.append(branch3);
subnetwork.append(gtf.add());


network.append(subnetwork);
gtf.debug_custom_model_design(network);



# In[ ]:


network.append(gtf.convolution(output_channels=32));
network.append(gtf.batch_normalization());
network.append(gtf.relu());
network.append(gtf.max_pooling());
gtf.debug_custom_model_design(network);




# In[ ]:


network.append(gtf.flatten());
network.append(gtf.fully_connected(units=1024));
network.append(gtf.dropout(drop_probability=0.2));
network.append(gtf.fully_connected(units=2));
gtf.Compile_Network(network, data_shape=(3, 32, 32));


# In[ ]:





# In[ ]:





# In[ ]:


# Set training parameters


# In[ ]:


gtf.Training_Params(num_epochs=50, display_progress=True, display_progress_realtime=True, 
        save_intermediate_models=False, save_training_logs=True);


gtf.optimizer_sgd(0.001);
gtf.lr_fixed();
gtf.loss_softmax_crossentropy();


# In[ ]:


gtf.Train();


# In[ ]:





# In[ ]:





# In[ ]:


# Step 0 - Using Pytorch
from pytorch_prototype import prototype

# Step 1 - Load experiment in evaluation mode
ptf = prototype(verbose=1);
ptf.Prototype("sample-project-1", "sample-experiment-1", eval_infer=True)


# Step 2 - Run inference on dataset
output = ptf.Infer(img_dir="test/");


# In[ ]:


num_0 = 0;
num_1 = 1;


# In[ ]:


# Create submission
import pandas as pd
sub = pd.read_csv("/kaggle/input/aerial-cactus-identification/sample_submission.csv");
for i in range(len(output)):
    index = int(sub[sub['id']==output[i]['img_name']].index[0])
    if(int(output[i]['predicted_class']) == 0):
        num_0 += 1;
    else:
        num_1 += 1;
    sub['has_cactus'][index] = int(output[i]['predicted_class'])
sub.to_csv("submission.csv", index=False);


# In[ ]:


num_0, num_1


# In[ ]:


get_ipython().system('rm -r /kaggle/working/monk_v1/')


# In[ ]:


get_ipython().system('rm -r test')


# In[ ]:


get_ipython().system('rm -r train')


# In[ ]:





# In[ ]:




