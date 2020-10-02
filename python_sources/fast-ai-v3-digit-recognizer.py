#!/usr/bin/env python
# coding: utf-8

# # DIGIT RECOGNIZER
# 
# This is my solution to the [digit recognizer](https://www.kaggle.com/c/digit-recognizer) competition using ResNet34.

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai import *
from fastai.vision import *
import csv
from PIL import Image


# In[ ]:


bs = 64


# In[ ]:


mkdir '/kaggle/working/train'


# In[ ]:


mkdir '/kaggle/working/test'


# In[ ]:


csv_path = Path('../input/digit-recognizer')
csv_path.ls()


# In[ ]:


img_path = Path('/kaggle/working/')
img_path.ls()


# In[ ]:


def convert_pixels_to_image(pixels):
    pixels = np.array(pixels, dtype='uint8')
    pixels = pixels.reshape((28, 28))
    image = Image.fromarray(pixels)
    
    return image


# In[ ]:


counter = dict()

with open(csv_path/'train.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)

    for row in csv_reader:
        pixels = row[1:]
        image = convert_pixels_to_image(pixels)

        label = row[0]

        if label not in counter:
            counter[label] = 0
        counter[label] += 1

        filename = '{}_{}.jpg'.format(label, counter[label])
        image.save(img_path/'train'/filename)


# In[ ]:


counter = 0

with open(csv_path/'test.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)

    for row in csv_reader:
        pixels = row
        image = convert_pixels_to_image(pixels)
        counter += 1

        filename = '{}.jpg'.format(counter)
        image.save(img_path/'test'/filename)


# In[ ]:


fnames = get_image_files(img_path/'train')


# In[ ]:


pat = re.compile(r'/([^/]+)_\d+.jpg$')


# In[ ]:


np.random.seed(2)
data = ImageDataBunch.from_name_re(img_path, fnames, pat, ds_tfms=get_transforms(), size=14, bs=bs, num_workers=0
                                  ).normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=3, figsize=(7,6))


# In[ ]:


data.classes, data.c, len(data.train_ds), len(data.valid_ds)


# In[ ]:


learn = cnn_learner(data, models.resnet50, metrics=error_rate)


# In[ ]:


lr = 0.01


# In[ ]:


learn.fit_one_cycle(5, slice(lr))


# In[ ]:


learn.save('stage-1')


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(5, max_lr=slice(1e-4,1e-3))


# In[ ]:


learn.save('stage-2')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(5, max_lr=slice(1e-4,1e-3))


# In[ ]:


learn.save('stage-3')


# In[ ]:


np.random.seed(2)
data = ImageDataBunch.from_name_re(img_path, fnames, pat, ds_tfms=get_transforms(), size=28, bs=bs, num_workers=0
                                  ).normalize(imagenet_stats)


# In[ ]:


learn.data = data


# In[ ]:


learn.freeze()


# In[ ]:


learn.fit_one_cycle(5, slice(lr))


# In[ ]:


learn.save('stage-4')


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(5, max_lr=slice(1e-4,1e-3))


# In[ ]:


learn.save('stage-5')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(5, max_lr=slice(1e-4,1e-3))


# In[ ]:


learn.save('stage-6')


# In[ ]:


learn.export()


# In[ ]:


learn = load_learner(img_path)


# In[ ]:


img = open_image(img_path/'test'/'1.jpg')
img


# In[ ]:


pred_class,pred_idx,outputs = learn.predict(img)
print(pred_class)


# In[ ]:


test = os.listdir(img_path/'test')
test.sort(key=lambda f: int(re.sub('\D', '', f)))

with open(img_path/'submission.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['ImageId', 'Label'])
    
    for image_file in test:
        image = os.path.join(img_path/'test', image_file) 
        image_id = Path(image).stem

        img = open_image(image)
        pred_class,pred_idx,outputs = learn.predict(img)
        label = str(pred_class)
        
        writer.writerow([image_id, label])

