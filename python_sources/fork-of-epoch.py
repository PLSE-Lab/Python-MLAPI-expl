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


bs = 128


# In[ ]:


get_ipython().system('mkdir ../files')


# In[ ]:


get_ipython().system('cp -r ../input ../files/')


# In[ ]:


path = Path('../files/input/train data/Train data')
test = Path('../files/input/evaluation data/Evaluation data/')


# In[ ]:


path.ls()


# In[ ]:


np.random.seed(4)


# In[ ]:


data = (ImageList.from_folder(path) 
        .split_by_rand_pct(.2)             
        .label_from_folder()
        .transform(get_transforms(max_zoom=1),size=224)
        .databunch(bs=bs, num_workers=0)
        .normalize(imagenet_stats)) 


# In[ ]:


# data.test_ds.x.items


# In[ ]:


data.classes


# In[ ]:


data.show_batch(rows=3, figsize=(15,11))


# In[ ]:


learn = cnn_learner(data, models.resnet50, metrics=accuracy)


# In[ ]:


learn.fit_one_cycle(4)


# In[ ]:


learn.save('stage-1')


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


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()


# In[ ]:


# !nvidia-smi


# In[ ]:


# torch.cuda.empty_cache()


# In[ ]:


learn.unfreeze()


# In[ ]:


# learn.lr_find()
# learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(4, max_lr=slice(1e-4,1e-3))


# In[ ]:


with open('submission2.csv', mode='w') as submit_file:
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


# learn.lr_find()
# learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(4, max_lr=slice(6e-5,6e-4))


# In[ ]:


learn.save('stage-3')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()


# In[ ]:


with open('submission3.csv', mode='w') as submit_file:
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


# learn.lr_find()
# learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(4, max_lr=slice(3e-6,3e-5))


# In[ ]:


learn.save('stage-4')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()


# In[ ]:


with open('submission4.csv', mode='w') as submit_file:
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


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(4, max_lr=slice(3e-5,3e-4))


# In[ ]:


learn.save('stage-5')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()


# In[ ]:


with open('submission5.csv', mode='w') as submit_file:
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




