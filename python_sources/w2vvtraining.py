#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system(' cat /kaggle/input/dataset-preprocessing/version_notes.txt')


# ## Environment Setup

# In[ ]:


get_ipython().system(' apt-get install --assume-yes python-pip > /dev/null')
get_ipython().system(' python2 -m pip install --upgrade pip --user > /dev/null')
get_ipython().system(' python2 -m pip --version')


# In[ ]:


get_ipython().system(' apt-get install --assume-yes python-pydot python-pydot-ng graphviz > /dev/null')
get_ipython().system(' python2 -m pip install --user --no-warn-script-location numpy scipy matplotlib ipython jupyter pandas > /dev/null')
get_ipython().system(' python2 -m pip install --user --no-warn-script-location tensorflow==1.15 keras tensorboard_logger pydot graphviz > /dev/null')


# In[ ]:


import shutil
shutil.copytree("/kaggle/input/w2vv-scripts/", "/kaggle/working/w2vv")
shutil.copytree("/kaggle/input/w2vv-model-20-14/train_results/f8bc9b776008665361f95ebc2fc01be54f2436cf", "/kaggle/working/pretrained_model")
get_ipython().system(' mv /kaggle/working/w2vv/* /kaggle/working/')

_ = shutil.copytree("/kaggle/input/dataset-preprocessing/", "/kaggle/working/VisualSearch")


# In[ ]:


import os

os.symlink("/kaggle/input/word2vec-flickr30k/word2vec", "/kaggle/working/VisualSearch/word2vec")
os.symlink("/kaggle/working/VisualSearch/data_w2vvtrain/FeatureData", "/kaggle/working/VisualSearch/data_w2vvval/FeatureData")
os.symlink("/kaggle/working/VisualSearch/data_w2vvtrain/FeatureData", "/kaggle/working/VisualSearch/data_w2vvtest/FeatureData")

os.environ["HOME"] = "/kaggle/working/"


# In[ ]:


# ! (echo "import tensorflow as tf" ; echo "print tf.test.is_gpu_available()") | python2


# ## Training

# In[ ]:


train_set_name = "data_w2vvtrain"
val_set_name = "data_w2vvval"
test_set_name = "data_w2vvtest"
feature_name = "pyresnet152-pool5os"
overwrite = 1
n_caption = 2
max_epochs = 14
BoW_size = 5
vocab_name = 'word_vocab_{}.txt'.format(BoW_size)
pretrained_model_path = '/kaggle/working/pretrained_model/best_model.h5'


# In[ ]:


# Generate a dictionary on the training set
get_ipython().system(' sh ./do_gene_vocab.sh $train_set_name $BoW_size')


# In[ ]:


# training --init_model_from $pretrained_model_path  --sent_maxlen 100
get_ipython().system(' python2 w2vv_trainer.py $train_set_name $val_set_name  $test_set_name     --overwrite $overwrite --img_feature $feature_name --n_caption $n_caption     --max_epochs $max_epochs --bow_vocab $vocab_name --rnn_vocab $vocab_name --init_model_from $pretrained_model_path')


# In[ ]:


# evaluation (Notice that a script file do_test_${testCollection}.sh will be automatically generated when the training process ends.)
get_ipython().system(' sh ./do_test_data_w2vvtest.sh # shoud bedo_test_${testCollection}.sh')


# In[ ]:


# delete files from current directory
get_ipython().system(' find ./ -maxdepth 1 -type f | grep -v "__" | xargs rm')

# delete training directories
get_ipython().system(' rm -rf util')
get_ipython().system(' rm -rf w2vv')
get_ipython().system(' rm -rf simpleknn')
get_ipython().system(' rm -rf pretrained_model')
get_ipython().system(' rm -rf basic')

# move training results
get_ipython().system(' mv VisualSearch/data_w2vvtrain/train_results ./')
get_ipython().system(' mv VisualSearch/data_w2vvtest/results ./')

# remove all other redundant data
get_ipython().system(' rm -rf VisualSearch')

