#!/usr/bin/env python
# coding: utf-8

# This notebook is the groundwork for [this](https://medium.com/@oneironaut.oml/solving-captchas-with-deeplearning-part-3-one-model-to-solve-it-all-d1fbd51f7f9e) blogpost. It's part of a 3-part miniseries, the other posts are [here](https://medium.com/@oneironaut.oml/solving-captchas-with-deeplearning-part-1-multi-label-classification-b9f745c3a599) and [here](https://medium.com/@oneironaut.oml/solving-captchas-with-deeplearning-part-2-single-character-classification-ac0b2d102c96).

# In[ ]:


from fastai.vision import *
import os
path = Path('../input/samples/samples')
os.listdir(path)[:10]


# In[ ]:


files = os.listdir(path)

fig, ax = plt.subplots(nrows=4, figsize=(14,7))

for a, filename in zip(ax.flatten(), files[:4]):
    a.imshow(PIL.Image.open(path/filename))
    a.axis('off')

plt.show()


# In[ ]:


#convert label
labels = [[char for char in code.name[:-4]] for code in (path).glob('*.png')]
labels = set([letter for label in labels for letter in label])
print(len(labels), 'different labels were found')

encoding_dict = {l:e for e,l in enumerate(labels)}
decoding_dict = {e:l for l,e in encoding_dict.items()}

code_dimension = len(labels)
captcha_dimension = 5

def to_onehot(filename):
    code = filename.name[:-4]
    onehot = np.zeros((code_dimension, captcha_dimension))
    for column, letter in enumerate(code):
        onehot[encoding_dict[letter], column] = 1
    return onehot.reshape(-1)

def to_idx(filename):
    code = filename.name[:-4]
    return np.array([encoding_dict[c] for c in code])#, dtype=torch.long)

def decode(onehot):
    onehot = onehot.reshape(code_dimension, captcha_dimension)
    idx = np.argmax(onehot, axis=0)
    return [decoding_dict[i.item()] for i in idx]

def label_accuracy(preds, actuals):
    pred = torch.unbind(preds)
    act = torch.unbind(actuals)
    
    valid = 0
    total = 0
    
    for left,right in zip(pred,act):
        total+=1
        p = decode(left)
        a = decode(right)
        if p==a: valid += 1

    return torch.tensor(valid/total).cuda()

def char_accuracy(n):
    def c_acc(preds, actuals):
        pred = torch.unbind(preds)
        act = torch.unbind(actuals)

        valid = 0
        total = 0

        for left,right in zip(pred,act):
            total+=1
            p = decode(left)
            a = decode(right)
            if p[n]==a[n]: valid += 1

        return torch.tensor(valid/total).cuda()
    return c_acc


# In[ ]:


data = (ImageList.from_folder(path)
        .split_by_rand_pct(0.2)
        .label_from_func(to_onehot, label_cls = FloatList)
        .transform(get_transforms(do_flip=False))
        .databunch()
        .normalize()
       )


# In[ ]:


learn = cnn_learner(data, models.resnet50, model_dir='/tmp',
                    metrics=[label_accuracy, char_accuracy(0),char_accuracy(1),char_accuracy(2),char_accuracy(3),char_accuracy(4)],
                   ps=0.)


# In[ ]:


lr_find(learn)
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(5, 5e-2)


# In[ ]:


learn.unfreeze()
#lr_find(learn)
#learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(25, slice(1e-3, 1e-2))


# In[ ]:


def multi_cross_entropy(inp, target):
    re_inp = inp.view(-1, 5, 19)
    re_target = target.view(-1, 5, 19)
    soft = F.log_softmax(re_inp, dim=0)
    cross = soft*re_target
    return -cross.mean()


# In[ ]:


data = (ImageList.from_folder(path)
        .split_by_rand_pct(0.2)
        .label_from_func(to_onehot, label_cls = FloatList)
        .transform(get_transforms(do_flip=False))
        .databunch()
        .normalize()
       )


# In[ ]:


learn = cnn_learner(data, models.resnet50, model_dir='/tmp',
                    metrics=[label_accuracy, char_accuracy(0),char_accuracy(1),char_accuracy(2),char_accuracy(3),char_accuracy(4)],
                    loss_func=multi_cross_entropy,
                   ps=0.)


# In[ ]:


lr_find(learn)
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(20, 1e-1)


# In[ ]:


learn.unfreeze()
lr_find(learn)
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(25, slice(1e-4, 1e-2))

