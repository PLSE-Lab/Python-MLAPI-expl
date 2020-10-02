#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('cd', '/kaggle/input/wheatdetection/wheatdetection')
get_ipython().system(' python ./tools/test_net.py MODEL.PRETRAINED \'(False)\' DATASETS.ROOT_DIR \'("/kaggle/input/global-wheat-detection")\'TEST.WEIGHT \'("/kaggle/input/resnestwheat/giou-fold0.bin")\' OUTPUT_DIR \'("/kaggle/working/")\'')


# In[ ]:




