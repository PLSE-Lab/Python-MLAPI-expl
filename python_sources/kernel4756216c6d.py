#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system(' rm -R /kaggle/working/tacotron')
get_ipython().run_line_magic('cd', '/kaggle/working')
get_ipython().system(' git clone https://github.com/van26101998/tacotron.git')
get_ipython().run_line_magic('cd', './tacotron')
get_ipython().system(' git submodule init; git submodule update')


# In[ ]:


get_ipython().system(' apt install tree')
get_ipython().run_line_magic('cd', '/kaggle')
# ! tree 
get_ipython().run_line_magic('cd', '/kaggle/working/tacotron')


# In[ ]:


get_ipython().system(' pip install inflect')
get_ipython().system(' pip uninstall torch torchvision -y')
get_ipython().system(' pip install torch==1.0.0')
get_ipython().system(' pip uninstall tensorboard tensorboardX -y')
get_ipython().system(' pip install tensorboardX==1.1')


# In[ ]:


# Training testing split
import pandas as pd
import os
# from text_processing import normalize_text
data_dir = '/kaggle/input/vnspeech'
csv_file = '/kaggle/input/vnspeech/filelist.csv'

data = pd.read_csv(csv_file, sep='|', header=None)
# Suffle
data = data[:int(len(data)/3)]
print(len(data))
data = data.sample(frac=1).reset_index(drop=True)

train_ratio = 0.8
train_index = int(train_ratio * len(data))

with open('./training.txt', 'w') as fd:
    for i, fname in enumerate(data[0][:train_index]):
        fd.write('{}|{}\n'.format(os.path.join(data_dir, fname) , data[1][i]))

with open('./testing.txt', 'w') as fd:
    for i, fname in enumerate(data[0][train_index:]):
        fd.write('{}|{}\n'.format(os.path.join(data_dir, fname) , data[1][i+train_index]))


# In[ ]:


get_ipython().system(' python train.py -o output -l logs -c /kaggle/input/vnspeechchkpnt500/checkpoint_1000 ')


# In[ ]:


get_ipython().system('tree')


# In[ ]:


get_ipython().system(' ls')


# In[ ]:


# import os
# os.chdir(r'/kaggle/working')
# os.rename('/kaggle/working/tacotron/output/checkpoint_1100', '/kaggle/working/checkpoint_1100')
# from IPython.display import FileLink
# FileLink('checkpoint_1100')

