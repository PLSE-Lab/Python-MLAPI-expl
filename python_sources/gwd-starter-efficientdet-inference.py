#!/usr/bin/env python
# coding: utf-8

# Train kernel: https://www.kaggle.com/ateplyuk/gwd-starter-efficientdet-train

# In[ ]:


import numpy as np 
import pandas as pd 
import sys
import os
import itertools

sys.path.append('/kaggle/input/mod-efficientdet/')
sys.path.append('/kaggle/input/eff-torch/')


# In[ ]:


cp -r /kaggle/input/cocoapi/* /tmp


# In[ ]:


cd /tmp/PythonAPI


# In[ ]:


get_ipython().system('make')


# In[ ]:


from infer_detector import Infer


# In[ ]:


gtf = Infer()


# In[ ]:


gtf.Model(model_dir="/kaggle/input/gwd-starter-efficientdet-train/trained/")


# In[ ]:


img_path = "/kaggle/input/global-wheat-detection/test/cc3532ff6.jpg"
duration, scores, labels, boxes = gtf.Predict(img_path, ['wheat'], vis_threshold=0.7);


# In[ ]:


from IPython.display import Image
Image(filename='output.jpg') 


# In[ ]:


TEST_PATH = '/kaggle/input/global-wheat-detection/test/'
test_ids = os.listdir(TEST_PATH)


# In[ ]:


res = []
for idx, row in enumerate(test_ids):
    img_path = TEST_PATH + row
    duration, scores, labels, boxes = gtf.Predict(img_path, ['wheat'], vis_threshold=0.5)
    
    sc = len(scores[scores > 0.5])
    
    b = boxes[:sc].cpu().numpy()
    s = scores[:sc].cpu().numpy()
    
    bs = []
    for i, el in enumerate(b):
        el = list(map(int, el)) 
        el[2] = el[2] - el[0]
        el[3] = el[3] - el[1]
        el.insert(0, s[i])      
        bs.append(el)        

    
    rl = list(itertools.chain.from_iterable(bs))
    rs = ' '.join(str(e) for e in rl)
    
    
    dres = {
            'image_id': row.split('.')[0],
            'PredictionString': rs
            }   
        
    res.append(dres)     


# In[ ]:


test_df = pd.DataFrame(res, columns=['image_id', 'PredictionString'])
test_df.head()


# In[ ]:


cd /kaggle/working


# In[ ]:


test_df.to_csv('submission.csv', index=False)

