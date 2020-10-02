#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastai.vision import *
from fastai import *
import pandas as pd
import csv


# In[ ]:


get_ipython().system("cp -r '../input/train data/Train data' input2")


# In[ ]:


path = Path('input2')
for i in path.ls():
    print(i)


# In[ ]:


import os
import shutil
src=path/'leukocoria'
src_files = os.listdir(src)
for file_name in src_files:
    full_file_name = os.path.join(src, file_name)
    copy_file_name = os.path.join(src, 'copy'+file_name)
    if (os.path.isfile(full_file_name)):
        shutil.copy(full_file_name, copy_file_name)


# In[ ]:


src=path/'leukocoria'
src_files = os.listdir(src)
for file_name in src_files:
    full_file_name = os.path.join(src, file_name)
    copy_file_name = os.path.join(src, 'copy'+file_name)
    if (os.path.isfile(full_file_name)):
        shutil.copy(full_file_name, copy_file_name)


# In[ ]:


print(len([name for name in os.listdir(path/'leukocoria')]))


# In[ ]:


np.random.seed(42)


# In[ ]:


data = (ImageList.from_folder(path) 
        .split_by_rand_pct(.2)             
        .label_from_folder()
        .transform(get_transforms(max_zoom=1),size=224)
        .databunch(bs=128, num_workers=0)
        .normalize(imagenet_stats)) 


# In[ ]:


data.classes


# In[ ]:


# data.show_batch(rows=1,figsize=(7,8))
data.classes, data.c, len(data.train_ds), len(data.valid_ds)


# In[ ]:


learn = create_cnn(data, models.resnet50, metrics=accuracy)


# In[ ]:


learn.fit(9)


# In[ ]:


learn.save('stage-1')


# In[ ]:


learn.unfreeze()


# In[ ]:


# learn.lr_find()


# In[ ]:


# learn.recorder.plot()


# In[ ]:


# learn.fit_one_cycle(2,slice(1e-05,1e-04))
# learn.load('stage-1')
learn.fit(10, slice(4e-05,2e-04))
# learn.fit_one_cycle(6, max_lr=slice(1e-04))


# In[ ]:


path1 = Path('../input/evaluation data/Evaluation data')
with open('submission.csv', mode='w') as employee_file:
    employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    employee_writer.writerow(['Id','Category'])
#     path = Path('input1/evaluation data/Evaluation data')
    for i in path1.ls():
       img=open_image(i)
       head, tail = os.path.split(i)
       tail=tail[:-4]
       pred_class,pred_idx,outputs = learn.predict(img)
       if(pred_idx==0):
        employee_writer.writerow([tail,1])
       else:
        employee_writer.writerow([tail,0])
    


# In[ ]:


# learn.lr_find()
# learn.recorder.plot()


# In[ ]:


learn.fit(10, slice(3e-05,1e-04))


# In[ ]:


path1 = Path('../input/evaluation data/Evaluation data')
# path1.ls()


# In[ ]:


with open('submission1.csv', mode='w') as employee_file:
    employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    employee_writer.writerow(['Id','Category'])
#     path = Path('input1/evaluation data/Evaluation data')
    for i in path1.ls():
       img=open_image(i)
       head, tail = os.path.split(i)
       tail=tail[:-4]
       pred_class,pred_idx,outputs = learn.predict(img)
       if(pred_idx==0):
        employee_writer.writerow([tail,1])
       else:
        employee_writer.writerow([tail,0])
    


# In[ ]:


# df=pd.read_csv('submission.csv')
# df.sort_values(by=['Id'])
# df.head()
learn.export()
get_ipython().system('cp input2/export.pkl export1.pkl')


# In[ ]:


get_ipython().system('rm -rf input2')


# In[ ]:


# # import the modules we'll need
# from IPython.display import HTML
# import pandas as pd
# import numpy as np
# import base64

# # function that takes in a dataframe and creates a text link to  
# # download it (will only work for files < 2MB or so)
# def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
#     csv = df.to_csv()
#     b64 = base64.b64encode(csv.encode())
#     payload = b64.decode()
#     html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
#     html = html.format(payload=payload,title=title,filename=filename)
#     return HTML(html)
# create_download_link(df)


# In[ ]:


# interp = ClassificationInterpretation.from_learner(learn)

# losses,idxs = interp.top_losses()

# len(data.valid_ds)==len(losses)==len(idxs)
# interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:




