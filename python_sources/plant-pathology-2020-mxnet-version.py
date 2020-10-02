#!/usr/bin/env python
# coding: utf-8

# ## Import everything ##

# In[ ]:


import mxnet as mx
import numpy as np
import pandas as pd
import os, random, sys, time, cv2
import matplotlib.pyplot as plt
from mxnet import gluon, nd, image, init, autograd
from mxnet.gluon import data as gdata, model_zoo, nn, loss as gloss, utils as gutils
import gluoncv

mx.random.seed(1024)
size = 320
batch_size = 16
epoch = 25
num_workers = 0 if sys.platform.startswith('win32') else 4


# ## Load data ##

# In[ ]:


train = pd.read_csv('../input/plant-pathology-2020-fgvc7/train.csv')
test = pd.read_csv('../input/plant-pathology-2020-fgvc7/test.csv')

images = list(train['image_id'].values)
labels = list(train[['healthy', 'multiple_diseases', 'rust', 'scab']].values)

test_images = list(test['image_id'].values)
train_dir = '../input/plant-pathology-2020-fgvc7/images'

assert len(labels)==len(images), 'train_label, train_ids lenth different'
sample_num = len(images)
ids = range(0, sample_num)

val_percent = 0.1
val_num = int(len(images)*val_percent)
val_ids = random.sample(range(0, sample_num), val_num)
train_ids = [id for id in ids if id not in val_ids]
assert len(train_ids)==sample_num-val_num, 'num wrong!'
train_num = len(train_ids)

val_images, val_labels = [None] * val_num, [None] * val_num
train_images , train_labels = [None] * train_num, [None] * train_num

for idx, value in enumerate(val_ids):
    val_images[idx] = images[value]
    val_labels[idx] = labels[value]
    
for idx, value in enumerate(train_ids):
    train_images[idx] = images[value]
    train_labels[idx] = labels[value]

print('Train data: {}, val data: {}, test data: {}.'.format(len(train_images), len(val_images), len(test_images)))


# ## Create dataset class ##

# In[ ]:


def read_images(root=train_dir, is_train=True):
    if is_train==True:
        data_images = train_images
        data_labels = train_labels
        data_num = len(data_images)
        data_type = 'Train dataset'
    elif is_train==False:
        data_images = val_images
        data_labels = val_labels
        data_num = len(data_images)
        data_type = 'Val dataset'
    elif is_train=='Test':
        data_images = test_images
        data_num = len(data_images)
        data_labels = [None] * data_num
        data_type = 'Test dataset'
    features, labels = nd.zeros(shape=(data_num, size, size, 3)), nd.zeros(shape=(data_num, 4))
    for i in range(data_num):
        features[i] = image.imresize(image.imread(os.path.join(root, data_images[i] + '.jpg')), size, size)
        labels[i] = data_labels[i]
    print('{} read finished'.format(data_type))
    return features, labels

class PlantDataset(gdata.Dataset):
    def __init__(self, is_train, train_dir):
#         self.rgb_mean = nd.array([0.485, 0.456, 0.406])
#         self.rgb_std = nd.array([0.229, 0.224, 0.225])
        features, labels = read_images(root=train_dir, is_train=is_train)
        self.features = [self.normalize_image(feature) for feature in features]
        self.labels = labels
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
#         return (img.astype('float32') / 255 - self.rgb_mean) / self.rgb_std
        return img.astype('float32') / 255 

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def __len__(self):
        return len(self.features)


# ## Augmention and create data iterator ##

# In[ ]:


plant_train = PlantDataset(True, train_dir)
plant_val = PlantDataset(False, train_dir)
plant_test = PlantDataset('Test', train_dir)

jitter_param = 0.4
lighting_param = 0.1
flip_aug = gdata.vision.transforms.Compose([
    gdata.vision.transforms.RandomFlipLeftRight(),
    gdata.vision.transforms.RandomFlipTopBottom(),
    gdata.vision.transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                                 saturation=jitter_param),
    gdata.vision.transforms.RandomLighting(lighting_param),
    gdata.vision.transforms.ToTensor(),
    gdata.vision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

no_aug = gdata.vision.transforms.Compose([
    gdata.vision.transforms.ToTensor(),
    gdata.vision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

train_iter = gdata.DataLoader(plant_train.transform_first(flip_aug), batch_size, shuffle=True,
                             num_workers=num_workers, last_batch='keep')
val_iter = gdata.DataLoader(plant_val.transform_first(no_aug), batch_size,
                             num_workers=num_workers, last_batch='keep')
test_iter = gdata.DataLoader(plant_test.transform_first(no_aug), batch_size,
                             num_workers=num_workers, last_batch='keep')


# ## Define more function ##

# In[ ]:


def get_net(netname, ctx):
    if netname == 'densenet':
        net = model_zoo.vision.densenet161(pretrained=True).features
    elif netname == 'mobilenet':
        net = model_zoo.vision.mobilenet_v2_1_0(pretrained=True).features
    elif netname == 'seresnext':
        net = gluoncv.model_zoo.get_model('SE_ResNext101_64x4d', pretrained=True).features
    elif netname == 'resnext':
        net = gluoncv.model_zoo.get_model('ResNext101_64x4d', pretrained=True).features[:-1]
        net.add(CBAM(2048, 64))
        net[-1].initialize(init.Xavier())
    net.add(nn.Dense(512), nn.Dropout(0.25), nn.Dense(4))
    net[-3:].initialize(init.Xavier())
    net.collect_params().reset_ctx(ctx)
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.00002, 'wd': 1e-3})
    net.hybridize()
    return net, trainer

def cross_entropy(y_hat, y):
    y_hat = y_hat.log()
    return -y * y_hat

def softmax_to_onehot(x, axis=1, ctx=mx.gpu(0)):
    a = nd.argmax(x, axis=1)
    b = nd.zeros(shape=(x.shape[0], x.shape[1]), ctx=ctx)
    b[nd.arange(len(a)), a] = 1.
    return b
    
def evaluate_accuracy(data_iter, net, ctx):
    """Evaluate accuracy of a model on the given data set."""
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc_sum, n = nd.array([0]), 0
    for batch in data_iter:
        features, labels = batch
        if labels.dtype != features.dtype:
            labels = labels.astype(features.dtype)
        Xs = gutils.split_and_load(features, ctx)
        ys = gutils.split_and_load(labels, ctx)
        for X, y in zip(Xs, ys):
            y = y.astype('float32')
            acc_sum += (softmax_to_onehot(net(X), axis=1, ctx=ctx[0]) == y).sum().copyto(mx.cpu())
            n += y.size
        acc_sum.wait_to_read()
    return acc_sum.asscalar() / n

def train(train_iter, val_iter, net, loss, trainer, ctx, num_epochs, verbose=0):
    """Train and evaluate a model."""
    print('training {} on {}'.format(net.name, ctx))
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, m, start = 0.0, 0.0, 0, 0, time.time()
        for i, batch in enumerate(train_iter):
            features, labels = batch
            batch_size = features.shape[0]
            if labels.dtype != features.dtype:
                labels = labels.astype(features.dtype)
            Xs = gutils.split_and_load(features, ctx)
            ys = gutils.split_and_load(labels, ctx)
            ls = []
            with autograd.record():
                y_hats = [nd.softmax(net(X)) for X in Xs]
                ls = [cross_entropy(y_hat, y) for y_hat, y in zip(y_hats, ys)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
            train_l_sum += sum([l.sum().asscalar() for l in ls])
            n += sum([l.size for l in ls])
            train_acc_sum += sum([(softmax_to_onehot(y_hat, axis=1, ctx=ctx[0]) == y).sum().asscalar()
                                 for y_hat, y in zip(y_hats, ys)])
            m += sum([y.size for y in ys])
        test_acc = evaluate_accuracy(val_iter, net, ctx[0])
        
        if verbose == 0:
            print('epoch %d/%d, loss %.4f, train acc %.3f, val acc %.3f, '
                  'time %.1f sec'
                  % (epoch + 1, num_epochs, train_l_sum / n, train_acc_sum / m, test_acc,
                     time.time() - start))
        else:
            if (epoch + 1) % verbose == 0 or epoch == 0:
                print('epoch %d/%d, loss %.4f, train acc %.3f, val acc %.3f, '
                      'time %.1f sec'
                      % (epoch + 1, num_epochs, train_l_sum / n, train_acc_sum / m, test_acc,
                         time.time() - start))
    print('Train finished')
    return net


# ## Define models

# In[ ]:


ctx = mx.gpu(0)
train_loss = gloss.SoftmaxCrossEntropyLoss(axis=1, from_logits=True)
sub_dir = "../input/plant-pathology-2020-fgvc7/sample_submission.csv"


# ### MobileNet

# In[ ]:


def use_mobile_net(epoch=30):
    preds_mobile = []
    net_mobile, trainer_mobile = get_net('mobilenet', ctx=mx.gpu(0))
    net_mobile = train(train_iter, val_iter, net_mobile, train_loss, trainer_mobile, ctx, epoch, verbose=10)
    for X, _ in test_iter:
        y_hat_mobile = nd.softmax(net_mobile(X.as_in_context(ctx))).asnumpy()
        preds_mobile.extend(y_hat_mobile)
    sub_mobile = pd.read_csv(sub_dir)
    sub_mobile.loc[:, 'healthy':] = preds_mobile
    sub_mobile.to_csv('submission_mobile.csv', index=False)
    sub_mobile.head(5)
    return y_hat_mobile


# ### SE_ResNext

# In[ ]:


def use_seresnext(epoch=30):
    preds_seresnext = []
    net_seresnext, trainer_seresnext = get_net('seresnext', ctx=mx.gpu(0))
    net_seresnext = train(train_iter, val_iter, net_seresnext, train_loss, trainer_seresnext, ctx, epoch, verbose=5)
    for X, _ in test_iter:
        y_hat_seresnext = nd.softmax(net_seresnext(X.as_in_context(ctx))).asnumpy()
        preds_seresnext.extend(y_hat_seresnext)
    sub_seresnext = pd.read_csv(sub_dir)
    sub_seresnext.loc[:, 'healthy':] = preds_seresnext
    sub_seresnext.to_csv('submission_seresnext.csv', index=False)
    print(sub_seresnext.head(5))
    return preds_seresnext


# ### DenseNet

# In[ ]:


def use_densenet(epoch=30):
    preds_dense = []
    net_dense, trainer_dense = get_net('densenet', ctx=mx.gpu(0))
    net_dense = train(train_iter, val_iter, net_dense, train_loss, trainer_dense, ctx, epoch, verbose=10)
    for X, _ in test_iter:
        y_hat_dense = nd.softmax(net_dense(X.as_in_context(ctx))).asnumpy()
        preds_dense.extend(y_hat_dense)
    sub_dense = pd.read_csv(sub_dir)
    sub_dense.loc[:, 'healthy':] = preds_dense
    sub_dense.to_csv('submission_dense.csv', index=False)
    print(sub_dense.head(5))
    return preds_dense


# ### ResNext_CBAM

# In[ ]:


class CAM(nn.HybridBlock):
  def __init__(self, num_channels, ratio, **kwargs):
    super(CAM, self).__init__(**kwargs)
    with self.name_scope():
      self.avg_pool = nn.GlobalAvgPool2D()
      self.max_pool = nn.GlobalMaxPool2D()
      self.conv1 = nn.Conv2D(num_channels // ratio, 1, use_bias=False)
      self.conv2 = nn.Conv2D(num_channels, 1, use_bias=False)

  def hybrid_forward(self, F, X):
    X_avg = self.avg_pool(X)
    X_avg = self.conv1(X_avg)
    X_avg = F.relu(X_avg)
    X_avg = self.conv2(X_avg)

    X_max = self.max_pool(X)
    X_max = self.conv1(X_max)
    X_max = F.relu(X_max)
    X_max = self.conv2(X_max)

    Y = X_avg + X_max
    Y = F.sigmoid(Y)
    return Y


class SAM(nn.HybridBlock):
  def __init__(self, kernel_size=7, **kwargs):
    super(SAM, self).__init__(**kwargs)
    with self.name_scope():
      self.kernel_size = kernel_size
      assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
      self.padding = 3 if self.kernel_size == 7 else 1

      self.conv = nn.Conv2D(1, kernel_size=self.kernel_size, padding=self.padding, use_bias=False)

  def hybrid_forward(self, F, X):
    X_avg = F.mean(X, axis=1, keepdims=True)
    X_max = F.max(X, axis=1, keepdims=True)
    Y = F.concat(X_avg, X_max, dim=1)
    Y = self.conv(Y)
    Y = F.sigmoid(Y)
    return Y


class CBAM(nn.HybridBlock):
  def __init__(self, num_channels, ratio, **kwargs):
    super(CBAM, self).__init__(**kwargs)
    with self.name_scope():
      self.num_channels = num_channels
      self.ratio = ratio
      self.cam = CAM(self.num_channels, self.ratio)
      self.sam = SAM()

  def hybrid_forward(self, F, X):
    residual = X
    Y = F.broadcast_mul(self.cam(X), X)
    Y = F.broadcast_mul(self.sam(Y), Y)
    Y = F.relu(Y)
    return Y + residual


# In[ ]:


def use_resnet(epoch=30):
    preds_res = []
    net_res, trainer_res = get_net('resnext', ctx=mx.gpu(0))
    net_res = train(train_iter, val_iter, net_res, train_loss, trainer_res, ctx, epoch, verbose=5)
    for X, _ in test_iter:
        y_hat_res = nd.softmax(net_res(X.as_in_context(ctx))).asnumpy()
        preds_res.extend(y_hat_res)
    sub_res = pd.read_csv(sub_dir)
    sub_res.loc[:, 'healthy':] = preds_res
    sub_res.to_csv('submission_res.csv', index=False)
    print(sub_res.head(5))
    return preds_res


# ## Train and predict

# In[ ]:


# preds_mobile = use_mobile_net(epoch)
# preds_res = use_resnet(epoch)
preds_seresnext = use_seresnext(epoch)
preds_dense = use_densenet(epoch)


# ## Result ensemble ##

# ### Method 1 ###

# In[ ]:


def weighted_ensemble(preds_a, preds_b):
    sub1, sub2, sub3 = pd.read_csv(sub_dir), pd.read_csv(sub_dir), pd.read_csv(sub_dir)
    pred_1, pred_2, pred_3 = [], [], []
    for a, b in zip(preds_a, preds_b):
        pred_1.append(a * 0.25 + b * 0.75)
        pred_2.append(a * 0.5 + b * 0.5)
        pred_3.append(a * 0.75 + b * 0.25)

    sub1.loc[:, 'healthy':] = pred_1
    sub2.loc[:, 'healthy':] = pred_2
    sub3.loc[:, 'healthy':] = pred_3
    sub1.to_csv('submission1.csv', index=False)
    sub2.to_csv('submission2.csv', index=False)
    sub3.to_csv('submission3.csv', index=False)

weighted_ensemble(preds_seresnext, preds_dense)


# ### Method 2 ###

# In[ ]:


def average_ensemble_three(preds_a, preds_b, preds_c):
    sub = pd.read_csv(sub_dir)
    pred = []
    for a, b, c in zip(preds_a, preds_b, preds_c):
        pred.append((a + b + c) / 3.)
    sub.loc[:, 'healthy':] = pred
    sub.to_csv('submission.csv', index=False)
    print(sub.head())

def average_ensemble_two(preds_a, preds_b):
    sub = pd.read_csv(sub_dir)
    pred = []
    for a, b in zip(preds_a, preds_b):
        pred.append((a + b) / 2.)
    sub.loc[:, 'healthy':] = pred
    sub.to_csv('submission.csv', index=False)
    print(sub.head(5))
    
# average_ensemble_three(preds_res, preds_seresnext, preds_dense)
# average_ensemble_two(preds_seresnext, preds_dense)

