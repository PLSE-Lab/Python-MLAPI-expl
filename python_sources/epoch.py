#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.vision import *
from fastai.metrics import error_rate


# In[ ]:


bs = 64


# In[ ]:


get_ipython().system('mkdir ../files')


# In[ ]:


get_ipython().system('cp -r ../input ../files/')


# In[ ]:


# !cp -r ../input/train\ data/Train\ data/leukocoria/* ../files/input/train\ data/Train\ data/leukocoria/


# In[ ]:


path = Path('../files/input/train data/Train data')
test = Path('../files/input/evaluation data/Evaluation data/')


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


print(len([name for name in os.listdir(path/'leukocoria')]))


# In[ ]:


import os
import shutil
src=path/'leukocoria'
src_files = os.listdir(src)
for file_name in src_files:
    full_file_name = os.path.join(src, file_name)
    copy_file_name = os.path.join(src, 'copy1'+file_name)
    if (os.path.isfile(full_file_name)):
        shutil.copy(full_file_name, copy_file_name)


# In[ ]:


print(len([name for name in os.listdir(path/'leukocoria')]))


# In[ ]:


path.ls()


# In[ ]:


np.random.seed(42)
data = (ImageList.from_folder(path) 
        .split_by_rand_pct(.2)             
        .label_from_folder()            
        .add_test(ImageList.from_folder(test))
        .transform(get_transforms(),size=224)
        .databunch(bs=bs, num_workers=0)
        .normalize(imagenet_stats)) 


# In[ ]:


# data.test_ds.x.items


# In[ ]:


# data.classes


# In[ ]:


# data.show_batch(rows=3, figsize=(7,8))


# In[ ]:


learn = cnn_learner(data, models.resnet50, metrics=accuracy)


# In[ ]:


learn.fit_one_cycle(4)


# In[ ]:


learn.save('stage-1')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()


# In[ ]:


with open('submission0.csv', mode='w') as submit_file:
    submit_writer = csv.writer(submit_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    submit_writer.writerow(['Id','Category'])
    for i in test.ls():
        img=open_image(i)
        head, tail = os.path.split(i)
        tail=tail[:-4]
        pred_class,pred_idx,outputs = learn.predict(img)
        if(pred_idx==0):
            submit_writer.writerow([tail,1])
        else:
            submit_writer.writerow([tail,0])


# In[ ]:


learn.unfreeze()


# In[ ]:


# learn.lr_find()
# learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(4, max_lr=slice(5e-5,3e-4))


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()


# In[ ]:


with open('submission1.csv', mode='w') as submit_file:
    submit_writer = csv.writer(submit_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    submit_writer.writerow(['Id','Category'])
    for i in test.ls():
        img=open_image(i)
        head, tail = os.path.split(i)
        tail=tail[:-4]
        pred_class,pred_idx,outputs = learn.predict(img)
        if(pred_idx==0):
            submit_writer.writerow([tail,1])
        else:
            submit_writer.writerow([tail,0])


# In[ ]:


learn.save('stage-2')


# In[ ]:


interp.plot_confusion_matrix()


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


interp.most_confused(min_val=2)

