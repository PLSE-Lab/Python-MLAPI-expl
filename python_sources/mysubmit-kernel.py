#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('cp -r /kaggle/input/mmdwhl/mmd_whl/cocoapi cocoapi;')


# In[ ]:


cd /kaggle/working/cocoapi/PythonAPI


# In[ ]:


get_ipython().system('python setup.py build_ext install')


# In[ ]:


cd /kaggle/working/


# In[ ]:


get_ipython().system('pip install /kaggle/input/mmdwhl/mmd_whl/addict-2.2.1-py3-none-any.whl')
get_ipython().system('pip install /kaggle/input/mmdwhl/mmd_whl/mmcv-0.5.1-cp37-cp37m-linux_x86_64.whl')
get_ipython().system('pip install /kaggle/input/mmdwhl/mmd_whl/terminal-0.4.0-py3-none-any.whl')
get_ipython().system('pip install /kaggle/input/mmdwhl/mmd_whl/terminaltables-3.1.0-py3-none-any.whl')
get_ipython().system('cp -r /kaggle/input/mmdwhl/mmd_code/* /kaggle/working')
get_ipython().system('ln -s /kaggle/input/global-wheat-detection/test /kaggle/working/data')
get_ipython().system('mkdir data/annotations')
get_ipython().system('python data/get_test_json.py')
get_ipython().system('pip install -v -e .')
get_ipython().system('chmod +x tools/dist_test.sh')


# In[ ]:


import run
config = '/kaggle/input/my-pth/faster_rcnn_r101_fpn_1x.py'
model = '/kaggle/input/my-pth/epoch_12.pth'
num_gpu = 1
run.test(config, num_gpu,model)
run.json2submit('results.bbox.json','data/annotations/test.json','submission.csv',0)


# In[ ]:


get_ipython().system('ls . |grep -v submission.csv | xargs rm -rf')


# In[ ]:


get_ipython().system('cat /kaggle/input/mmdwhl/mmd_code/run.py')

