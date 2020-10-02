#!/usr/bin/env python
# coding: utf-8

# # Fastai Darknet Model

# This kernel implements a Darknet model for the HUman Protein Atals comp.
# It uses the 4 separated images as input, concatenating them in a 4 channel RGBY image. To do so, I modified the first convolution in the Darknet implementation from 3-->4 channels.
# The model trained is a Darknet_small. Better results can be achieved with computing power and a deeper network (Darknet53 for instance, check yolo [website](https://pjreddie.com/darknet/yolo/) )
# 
# This kernel uses the fastai library version 0.7 and one cycle training schedule.  No transforms are used.
# 
# To be able to train in the limited computing power of kallge, I trained a 128 model, then a 256 and finally this one, but only one epoch.

# In[ ]:


from fastai.conv_learner import *
from fastai.dataset import *

from pathlib import Path
import json
torch.cuda.set_device(0)
from pathlib import Path
torch.cuda.set_device(0)


# In[ ]:


ls /kaggle/input/


# sample : can be used to train in smaller dataset. For debug set sample to 3000 for instance.

# In[ ]:


MASKS = 'train.csv'
SUB = 'sample_submission.csv'
TRAIN = Path('train/')
TEST = Path('test/')
PATH = Path('/kaggle/input/human-protein-atlas-image-classification/')
TMP = Path('/kaggle/working/tmp/')
MODEL = Path('/kaggle/working/model/')
PRETRAINED = '/kaggle/input/4-channel-darknet-sz-256/model/256'
seg = pd.read_csv(PATH/MASKS).set_index('Id')
sample_sub = pd.read_csv(PATH/SUB).set_index('Id')

sample= len(seg)
seg.head()


# In[ ]:


train_names_png = [TRAIN/f for f in os.listdir(PATH/TRAIN)]
train_names = list(seg.index.values)
train_names_sample = list(seg.index.values)[0:sample]
test_names_png = [TEST/f for f in os.listdir(PATH/TEST)]
test_names = list(sample_sub.index.values)
test_names_sample = list(sample_sub.index.values)[0:sample]
len(train_names_sample), len(test_names)


# In[ ]:


TMP.mkdir(exist_ok=True)
MODEL.mkdir(exist_ok=True)


# In[ ]:


def rgba_open(fname, path=PATH, sz=128):
    '''open RGBA image from 4 different 1-channel files.
    return: numpy array [4, sz, sz]'''
    flags = cv2.IMREAD_GRAYSCALE
    red = cv2.imread(str(path/(fname+ '_red.png')), flags)
    blue = cv2.imread(str(path/(fname+ '_blue.png')), flags)
    green = cv2.imread(str(path/(fname+ '_green.png')), flags)
    yellow = cv2.imread(str(path/(fname+ '_yellow.png')),flags)
    im = np.array([red, green, blue, yellow], dtype=np.float32)
    if sz==512:
        return im/255
    else:
        rgba = cv2.resize(np.rollaxis(im, 0,3), (sz, sz), interpolation = cv2.INTER_CUBIC)
        return np.rollaxis(rgba, 2,0)/255


# This is to be able to use a smaller data sample to debug, sample=31072 is the full dataset. 

# In[ ]:


seg2 = seg.iloc[0:sample]
val_idxs = get_cv_idxs(sample)


# In[ ]:


class CustomDataset(FilesDataset):
    def __init__(self, fnames, y, transform, path, sz):
        self.y=y
        self.fnames = fnames
        self.sz = sz
        assert(len(fnames)==len(y))
        super().__init__(fnames, transform, path)
        
    def get_x(self, i): 
        return rgba_open(self.fnames[i], self.path, self.sz)
        
    def get_y(self, i):
        return self.y[i]
    def get_sz(self): return self.sz
    def get_c(self): return 28
    @property
    def is_multi(self):
        return True


# Compute `y` vector of targets.

# In[ ]:


indexes = seg2.Target.apply(str.split)
y = np.zeros((sample, 28))
for i in range(sample):
    y[i,np.array(indexes[i], dtype=int)]=1


# In[ ]:


len(train_names_sample),  y.shape, y.dtype


# In[ ]:


((val_x,trn_x),(val_y,trn_y)) = split_by_idx(val_idxs, np.array(train_names_sample), y)


# In[ ]:


# tfms = tfms_from_model(resnet34, sz=sz, crop_type=CropType.NO, aug_tfms=[])
def get_data(sz=128, bs=32):
    datasets = ImageData.get_ds(CustomDataset, (trn_x,trn_y), (val_x,val_y), sz=sz, tfms=(None,None), path=PATH/TRAIN)
    datasets[4] = CustomDataset(test_names, test_names, None, PATH/TEST, sz)
    return ImageData(PATH, datasets, bs=bs, num_workers=4, classes=28)


# ## Darknet Model definition

# One improvement could be to train in `Darknet([1, 2, 4, 8, 8, 4])` for instance.

# In[ ]:


class ConvBN(nn.Module):
    "convolutional layer then batchnorm"

    def __init__(self, ch_in, ch_out, kernel_size = 3, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(ch_out, momentum=0.01)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x): return self.relu(self.bn(self.conv(x)))


# In[ ]:


class DarknetBlock(nn.Module):
    def __init__(self, ch_in):
        super().__init__()
        ch_hid = ch_in//2
        self.conv1 = ConvBN(ch_in, ch_hid, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBN(ch_hid, ch_in, kernel_size=3, stride=1, padding=1)

    def forward(self, x): return self.conv2(self.conv1(x)) + x


# In[ ]:


class Darknet(nn.Module):
    "Replicates the darknet classifier from the YOLOv3 paper (table 1)"

    def make_group_layer(self, ch_in, num_blocks, stride=1):
        layers = [ConvBN(ch_in,ch_in*2,stride=stride)]
        for i in range(num_blocks): layers.append(DarknetBlock(ch_in*2))
        return layers

    def __init__(self, num_blocks, num_classes=1000, start_nf=32):
        super().__init__()
        nf = start_nf
        layers = [ConvBN(4, nf, kernel_size=3, stride=1, padding=1)]
        for i,nb in enumerate(num_blocks):
            layers += self.make_group_layer(nf, nb, stride=(1 if i==1 else 2))
            nf *= 2
        layers += [nn.AdaptiveAvgPool2d(1), Flatten(), nn.Linear(nf, num_classes)]
#         layers += [nn.Sigmoid()]
        self.layers = nn.Sequential(*layers)

    def forward(self, x): return self.layers(x)


# In[ ]:


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val +             ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.sum(dim=1).mean()


# In[ ]:


from sklearn.metrics import fbeta_score
import warnings

def f1_(preds, targs, start=0.17, end=0.24, step=0.01):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return max([fbeta_score(targs, (preds>th), 1, average='samples')
                    for th in np.arange(start,end,step)])


# In[ ]:


m = Darknet([1, 2, 4, 4, 3], 28).cuda()


# In[ ]:


m


# In[ ]:


md = get_data(512,8)


# In[ ]:


learn = Learner.from_model_data(m, md, tmp_name=TMP, models_name=MODEL)
learn.crit = FocalLoss()
learn.opt_fn = optim.Adam
learn.metrics = [f1_]


# loading 256 model

# In[ ]:


learn.load(PRETRAINED)


# In[ ]:


lr = 1E-2


# Train with one cycle policy

# In[ ]:


learn.fit(lr/10,1,cycle_len=1,use_clr_beta=(10,10, 0.85, 0.9))


# In[ ]:


learn.save('512')


# In[ ]:


learn.load('512')


# ## Predictions and submission file

# We first get predictions on the validation set with corresponding targets. We will use this to compute an "optimised" set of thresholds

# In[ ]:


p_v, t_v = learn.predict_with_targs()


# In[ ]:


def sigmoid(a):
    return 1/(1+np.exp(-a))


# In[ ]:


sp_v = sigmoid(p_v) #compute the sigmoid of the network output


# Helper optimisation functions for F1 metric

# In[ ]:


def f1_np(y_pred, y_true, threshold=0.5):
    '''numpy f1 metric'''
    y_pred = (y_pred>threshold).astype(int)
    TP = (y_pred*y_true).sum(1)
    prec = TP/(y_pred.sum(1)+1e-7)
    rec = TP/(y_true.sum(1)+1e-7)
    res = 2*prec*rec/(prec+rec+1e-7)
    return res.mean()


def f1_n(y_pred, y_true, thresh, n, default=0.5):
    '''partial f1 function for index n'''
    threshold = default * np.ones(y_pred.shape[1])
    threshold[n]=thresh
    return f1_np(y_pred, y_true, threshold)

def find_thresh(y_pred, y_true):
    '''brute force thresh finder'''
    ths = []
    for i in range(y_pred.shape[1]):
        aux = []
        for th in np.linspace(0,1,100):
            aux += [f1_n(y_pred, y_true, th, i)]
        ths += [np.array(aux).argmax()/100]
    return np.array(ths)


# In[ ]:


ths = find_thresh(sp_v, t_v); ths


# Before optim: `f1 = 0.32`after `f1= 0.48`

# In[ ]:


f1_np(sp_v, t_v, 0.5), f1_np(sp_v, t_v, ths)


# In[ ]:


preds = learn.predict(is_test=True)


# A threshold optimisation can immrove this results a lot.

# In[ ]:


preds = sigmoid(preds)
threshold = ths
print(preds.shape)
classes = np.array([str(n) for n in range(28)])
res = np.array([" ".join(classes[(np.where(pp>threshold))])for pp in preds])


# In[ ]:


filenames = np.array([os.path.basename(fn).split('.')[0] for fn in test_names])


# In[ ]:


res.shape, filenames.shape


# In[ ]:


frame = pd.DataFrame(np.array([filenames, res]).T, columns = ['Id','Predicted'])


# In[ ]:


frame.head()


# In[ ]:


frame.to_csv('submission.csv', index=False)

