"""
Script with classes and functions extended from fastai library specifically for 
Deep Fake Detection Challenge
"""

import pandas as pd
import numpy as np
import torch
import cv2
import glob
from fastai import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.basics import *
from fastai.vision import learner
EPS=1e-5
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def debug(*args):
    if VERBOSE == True:
        print(*args)

def basename_and_ext(filename):
    return os.path.splitext(filename)


def get_dfdc_transforms():
    """
    Gets default transforms for DFDC
    - Crop pad with reflection
    - Flip  horizantal
    - Rotation upto 5 degrees
    - Zoom upto 1.3
    - Cutouts 1 to 3 holes of size 20 to 80 px 
    - Probability 0.75
    """
    xtfms=[cutout(n_holes=(1,3), length=(20, 80), p=.75)]
    tfms = get_transforms(flip_vert=True, max_rotate=5, max_zoom=1.3, max_lighting=None, max_warp=None, xtra_tfms=xtfms)
    return tfms

def jpg_to_mp4name(jpg):
    "abcdefg_1_0.jpg to abcdefg.mp4"
    return jpg.split('_')[0] + '.mp4'

def mp4_to_glob(video, faces_path='faces'):
    "abcdefg.mp4 to abcdefg*.jpg"
    basename,_ = basename_and_ext(video)
    return f'{faces_path}/{basename}*.jpg'

def concat(im1, im2, im3):
    "concatenates 3 RGB images horizontally"
    dst = PIL.Image.new('RGB', (im1.width + im2.width + im3.width , im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    dst.paste(im3, (2*im1.width, 0))
    return dst

def crop_pad(pil_img, size=224, border=cv2.BORDER_REFLECT_101):
    """
    crops along dimension(s) that is bigger than desired size
    pads along dimension(s) that is smaller than desired size
    original image is not resized and kept centered
    """
    X = np.array(pil_img)
    height, width = X.shape[0:2]
    top = 0
    bottom = 0
    left = 0
    right = 0
    if height < size:
        top = (size - height) // 2
        bottom = size - top - height
    if width < size:
        left = (size - width) // 2
        right = size - left - width
    if height > size:
        topcrop =  (height - size) // 2
        bottomcrop = topcrop + size
        X = X[topcrop:bottomcrop,:,:]
    if width > size:
        leftcrop =  (width - size) // 2
        rightcrop = leftcrop + size
        X = X[:, leftcrop:rightcrop,:]
    Y = cv2.copyMakeBorder(X, top, bottom, left, right, border)
    return PIL.Image.fromarray(Y)

def size_to_fit(pil_img, size=224, border=cv2.BORDER_REFLECT_101):
    """
    resizes to set bigger dimension to desired size
    scales the smaller dimension while preserving aspect ratio
    centers and reflects the border
    """

    X = np.array(pil_img)
    h, w = X.shape[0:2]
    if w > h:
        h = h * size // w
        w = size
    else:
        w = w * size // h
        h = size

    X = cv2.resize(X, (w, h), interpolation=cv2.INTER_AREA)

    height = h
    width = w
    top = 0
    bottom = 0
    left = 0
    right = 0
    if height < size:
        top = (size - height) // 2
        bottom = size - top - height
    if width < size:
        left = (size - width) // 2
        right = size - left - width
    if height > size:
        topcrop =  (height - size) // 2
        bottomcrop = topcrop + size
        X = X[topcrop:bottomcrop,:,:]
    if width > size:
        leftcrop =  (width - size) // 2
        rightcrop = leftcrop + size
        X = X[:, leftcrop:rightcrop,:]
    Y = cv2.copyMakeBorder(X, top, bottom, left, right, border)
    return PIL.Image.fromarray(Y)

def to_fastai(pil_img):
    "converts PIL image to tensor and divides by 255"
    x = pil2tensor(pil_img,np.float32)
    x.div_(255)
    return Image(x)

class DeepFakeImageList_v1(ImageList):
    "fastai ImageList extension that applies custom transformations unique to DFDC"
    resize_option = 2
       # 0 - No custom resize, resizing to be done with fastai transform 
       # 1 - keep original size, center and crop if too big or pad and reflect if too small
       # 2 - center and size to fit with same aspect ratio and reflect the border
    
    @classmethod
    def from_df(cls, df, **kwargs):
        return cls(items=range(len(df)),  inner_df=df, **kwargs)    

    @classmethod
    def from_pils(cls, pil_series,**kwargs):
        return cls(items=range(len(pil_series)),inner_df=pil_series, **kwargs)

    def label_from_array(self, array, label_cls=None, **kwargs):
        return self._label_from_list(array[self.items.astype(np.int)],label_cls=label_cls,**kwargs)

    def get_image_from_path(self, pth):
        im = PIL.Image.open(pth)
        return im
       
    def get_image_from_pil(self, i):
        im = PIL.Image.open(pth)
        return im

    def get(self, i):
        if(type(self.inner_df) == pd.core.series.Series):
            im = self.inner_df[i]
        else:
            row = self.inner_df.iloc[i]
            pth=row['name']
            # Randomly selects one face per video
            files = glob.glob(pth)
            file = random.choice(files)
            im = self.get_image_from_path(file)
        if self.resize_option == 1:
            im = crop_pad(im)
        elif self.resize_option == 2:
            im = size_to_fit(im)
        im = to_fastai(im)
        return im

class DeepFakeImageList(ImageList):
    "fastai ImageList extension that applies custom transformations unique to DFDC"
    resize_option = 2
       # 0 - No custom resize, resizing to be done with fastai transform 
       # 1 - keep original size, center and crop if too big or pad and reflect if too small
       # 2 - center and size to fit with same aspect ratio and reflect the border
    
    @classmethod
    def from_df(cls, df, **kwargs):
        return cls(items=range(len(df)),  inner_df=df, **kwargs)    

    @classmethod
    def from_pils(cls, pil_series,**kwargs):
        return cls(items=range(len(pil_series)),inner_df=pil_series, **kwargs)

    def label_from_array(self, array, label_cls=None, **kwargs):
        return self._label_from_list(array[self.items.astype(np.int)],label_cls=label_cls,**kwargs)

    def get_image_from_path(self, pth):
        im = PIL.Image.open(pth)
        return im
       
    def get_image_from_pil(self, i):
        im = PIL.Image.open(pth)
        return im

    def get(self, i):
        if(type(self.inner_df) == pd.core.series.Series):
            im = self.inner_df[i]
        else:
            row = self.inner_df.iloc[i]
            pthstr=row['name']
            # Randomly selects one face per video
            #files = glob.glob(pth)
            files = [f.strip() for f in pthstr.split(',')]
            file = random.choice(files)
            im = self.get_image_from_path(file)
        if self.resize_option == 1:
            im = crop_pad(im)
        elif self.resize_option == 2:
            im = size_to_fit(im)
        im = to_fastai(im)
        return im

# Real data gathering was a lot more messy - involved use of batch folders for validation split and ssim scores for filtering out bad fakes
def get_deepfakeimagelist_data(bs=4, faces_path='faces', tfms=[[],[]], **resizeargs):
    'Gathers real and fake faces and returns custom imagedata bunch, that randomly selects one face from a video on each get'
    pair_df = pd.read_csv(f'../input/fakereal-pairs-in-dfdc-test-videos/dfdc_test_video_pairs.csv')
    
    files = os.listdir(faces_path)
    fake_files = [f for f in files if jpg_to_mp4name(f) in pair_df['filename'].unique()]
    real_files = [f for f in files if jpg_to_mp4name(f) in pair_df['original'].unique()]
    fake_df = pd.DataFrame(columns=['name', 'label', 'is_valid'])
    fake_df['name'] = pd.Series(fake_files)
    fake_df['name'] = fake_df.apply(lambda x: mp4_to_glob(x['name']),axis=1)
    fake_df['label'] = 1.
    fake_df['is_valid'] = False # In reality validation set is determined by video batch folder 00, 01 etc
    fake_df['is_valid'].loc[15:] = True

    real_df = pd.DataFrame(columns=['name', 'label', 'is_valid'])
    real_df['name'] = pd.Series(real_files)
    real_df['name'] = real_df.apply(lambda x: mp4_to_glob(x['name']),axis=1)
    real_df['label'] = 0.
    real_df['is_valid'] = False # In reality validation set is determined by video batch folder 00, 01 etc
    real_df['is_valid'].loc[8:] = True

    fake_train_df = fake_df[fake_df['is_valid'] == False].copy()
    nf = len(fake_train_df)
    real_train_df = real_df[real_df['is_valid'] == False].copy().sample(n=nf, replace=True)

    fake_val_df = fake_df[fake_df['is_valid'] == True].copy()
    nfv = len(fake_val_df)
    real_val_df = real_df[real_df['is_valid'] == True].copy().sample(n=nfv, replace=True)
    
    print ("Training:")
    print ('Fake ', len(fake_train_df) )
    print ('Real ', len(real_train_df))
    print ("Validation:")
    print ('Fake ', len(fake_val_df))
    print ('Real ', len(real_val_df))
    df = pd.concat((real_train_df,\
                    real_val_df,\
                    fake_train_df,\
                    fake_val_df\
                   ))
    df =  df.sample(frac=1)
    df.reset_index(inplace=True)
  
    databunch =(DeepFakeImageList.from_df(df).split_from_df(col='is_valid')\
                .label_from_df(cols='label', label_cls=FloatList)\
                .transform(tfms, **resizeargs)\
                .databunch(bs=bs)).normalize(imagenet_stats)
    return databunch

###################################################################################
# Custom metrics
class DFDCAUROC(AUROC):
    "Tailors fastai auc_roc_score for binary classification"
    def on_epoch_begin(self, **kwargs):
        self.targs, self.preds = Tensor([]), Tensor([])
        
    def on_batch_end(self, last_output:Tensor, last_target:Tensor, **kwargs):
        last_output = torch.sigmoid(last_output)[:,-1]
        last_output=torch.clamp(last_output,.01,0.99)
        # last_output = F.softmax(last_output, dim=1)[:,-1]
        self.preds = torch.cat((self.preds, last_output.cpu()))
        self.targs = torch.cat((self.targs, last_target.cpu()))
    
    def on_epoch_end(self, last_metrics, **kwargs):
        return add_metrics(last_metrics, auc_roc_score(self.preds, self.targs))

class RealLoss(Callback):
    "Reports separate loss on real faces"
    def on_epoch_begin(self, **kwargs):
        self.targs, self.preds = Tensor([]), Tensor([])
        
    def on_batch_end(self, last_output:Tensor, last_target:Tensor, **kwargs):
        if not torch.isnan(last_output).any():
            last_output = torch.sigmoid(last_output)[:,-1]
            last_output=torch.clamp(last_output,.01,0.99)
            op = last_output.cpu()
            tg = last_target.cpu()
            realtargs = (tg <= 0.5)
            tg = tg[realtargs]
            op = op[realtargs]
            if(len(op) > 0):
                self.preds = torch.cat((self.preds, op))
                self.targs = torch.cat((self.targs, tg))
    
    def on_epoch_end(self, last_metrics, **kwargs):
        return add_metrics(last_metrics, nn.BCELoss()(self.preds, self.targs.type(torch.float32)))    

class FakeLoss(Callback):
    "Reports separate loss on fake faces"
    def on_epoch_begin(self, **kwargs):
        self.targs, self.preds = Tensor([]), Tensor([])
        
    def on_batch_end(self, last_output:Tensor, last_target:Tensor, **kwargs):
        if not torch.isnan(last_output).any():
            last_output = torch.sigmoid(last_output)[:,-1]
            last_output=torch.clamp(last_output,.01,0.99)
            op = last_output.cpu()
            tg = last_target.cpu()
            realtargs = (tg > 0.5)
            tg = tg[realtargs]
            op = op[realtargs]
            if(len(op) > 0):
                self.preds = torch.cat((self.preds, op))
                self.targs = torch.cat((self.targs, tg))
    
    def on_epoch_end(self, last_metrics, **kwargs):
        return add_metrics(last_metrics, nn.BCELoss()(self.preds, self.targs))
    
class DFDCBCELoss(Callback):
    "Reports BCE loss as metric. Useful when using custom loss function"

    def on_epoch_begin(self, **kwargs):
        self.targs, self.preds = Tensor([]), Tensor([])
        
    def on_batch_end(self, last_output:Tensor, last_target:Tensor, **kwargs):
        if not torch.isnan(last_output).any():
            if not torch.isnan(last_target).any():
                last_output = torch.sigmoid(last_output)[:,-1]
                if not torch.isnan(last_output).any():
                    last_output=torch.clamp(last_output,0.01,0.99)
                    op = last_output.cpu()
                    tg = last_target.cpu()
                    self.preds = torch.cat((self.preds, op))
                    self.targs = torch.cat((self.targs, tg))
                else:
                    print("sigmoid nan! ", last_output)
            else:
                print("target nan! ", last_target)
        else:
                print("output nan! ", last_output)
            
    def on_epoch_end(self, last_metrics, **kwargs):
        return add_metrics(last_metrics, nn.BCELoss()(self.preds, self.targs))
    
###################################################################################
# Custom losses

class WeightedBCELoss(nn.Module):
    "Increases weight for negative class. Useful when real faces are fery few in comparison to fake faces."
    def __init__(self, neg_weight=1.0 ):
        super(WeightedBCELoss, self).__init__()
        self.neg_weight=neg_weight


    def forward(self, inputs, targets):
        targets= targets.unsqueeze(-1)
        weights = torch.ones_like(targets)
        weights.requires_grad=False
        realtargs = (targets <= 0.5)
        if(sum(realtargs) > 0):
            weights[realtargs] = self.neg_weight
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        weighted_bce_loss = bce_loss * weights

        return torch.mean(weighted_bce_loss)

class FocalLoss(nn.Module):
    "Increases weight for negative class. Useful as real faces are fery few in comparison to fake faces."
    "Lifted from https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938"
    def __init__(self, alpha=4, gamma=1, logits=True, reduction='elementwise_mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        targets= targets.unsqueeze(-1)
        if self.logits:
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        bce_loss=torch.clamp(bce_loss,min=EPS)
        pt = torch.exp(-bce_loss)
        f_loss = self.alpha * (1-pt)**self.gamma * bce_loss

        if self.reduction is None:
            return f_loss
        else:
            return torch.mean(f_loss)
        
        
###################################################################################
# Custom callbacks
class ReloadData(LearnerCallback):
    "Reloads / resamples data after given epochs"
    every = 2
    def on_epoch_begin(self, epoch, **kwargs):
        if (epoch % self.every == 0):
            print("switching data at epoch:", epoch)
            self.learn.to_fp32()
            self.learn.data = get_all_data()
            self.learn.to_fp16()
            
    
def adaptiveThreshold(rgbimg):
    image = cv2.cvtColor(rgbimg, cv2.COLOR_RGB2GRAY)
    return cv2.adaptiveThreshold(image, 255, 
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 3, 2)

def otsuThreshold(rgbimg):
    image = cv2.cvtColor(rgbimg, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(image,(5,5),0)
    _, img = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return img


def blur(rgbimg):
    image = cv2.cvtColor(rgbimg, cv2.COLOR_RGB2HSV) # convert to HSV
    image = cv2.blur(image, (9,9))
    return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

def gaussian(rgbimg):
    image = cv2.cvtColor(rgbimg, cv2.COLOR_RGB2HSV) # convert to HSV
    image = cv2.GaussianBlur(image, (9,9),0)
    return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

def apply_filter(rgbimg, func=None):
    image = cv2.cvtColor(rgbimg, cv2.COLOR_RGB2HSV) # convert to HSV
    image = func(image)
    return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    
def median(rgbimg):
    image = cv2.cvtColor(rgbimg, cv2.COLOR_RGB2HSV) # convert to HSV
    image = cv2.medianBlur(image, 9)
    return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

def adaptiveThreshold(rgbimg):
    image = cv2.cvtColor(rgbimg, cv2.COLOR_RGB2GRAY)
    return cv2.adaptiveThreshold(image, 255, 
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 3, 2)
    

def sobelxy(rgbimg) :
    image = cv2.cvtColor(rgbimg, cv2.COLOR_RGB2GRAY) 
    sobel_x = cv2.Sobel(image.astype('float32'), cv2.CV_32F, dx = 1, dy = 0, ksize = 1)
    sobel_y = cv2.Sobel(image.astype('float32'), cv2.CV_32F, dx = 0, dy = 1, ksize = 1)
    blended = cv2.addWeighted(src1=sobel_x, alpha=0.5, src2=sobel_y,
                          beta=0.5, gamma=0)
    return blended.astype('uint8')

def laplacian(rgbimg):
    #image = cv2.cvtColor(rgbimg, cv2.COLOR_RGB2GRAY) 
    image = cv2.Laplacian(rgbimg.astype('float32'),cv2.CV_32F)
    return image.astype('uint8')
    
def dft1(c1):
    dft = cv2.dft(np.float32(c1),flags = cv2.DFT_COMPLEX_OUTPUT)
    # shift the zero-frequncy component to the center of the spectrum
    dft_shift = np.fft.fftshift(dft)
    # save image of the image in the fourier domain.
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
    return magnitude_spectrum

def dft(rgbimg):
    r = rgbimg[:,:,0]
    g = rgbimg[:,:,1]
    b = rgbimg[:,:,1]
    r,g,b=cv2.split(rgbimg)
    r=dft1(r)
    g=dft1(g)
    b=dft1(b)
    merged= cv2.merge((r,g,b))
    return merged.astype('uint8')    

def fit_sgd_warm(learn, n_cycles, lr, mom, cycle_len, cycle_mult):
    n = len(learn.data.train_dl)
    phases = [(TrainingPhase(n * (cycle_len * cycle_mult**i))
                 .schedule_hp('lr', lr, anneal=annealing_cos)
                 .schedule_hp('mom', mom, anneal=annealing_cos)) for i in range(n_cycles)]
    sched = GeneralScheduler(learn, phases)
    learn.callbacks.append(sched)
    if cycle_mult != 1:
        total_epochs = int(cycle_len * (1 - (cycle_mult)**n_cycles)/(1-cycle_mult)) 
    else: total_epochs = n_cycles * cycle_len
    learn.fit(total_epochs)


    
# Inference
def logits(p):
    return log( p / (1-p))


def gsigmoid( t):
    'Gentle sigmoid function that spreads out the predictions'
    return (t / (1 + abs(t)) + 1.) / 2.

def get_fake_preds(lrnr, testfaces, sigmoid=False, fatten=False):
    count = 0
    fakeness = 0.5
    preds = torch.tensor(())
    preds.new_empty((0,2),dtype=torch.float32)
    tfms = [[],[]]
    nf = len(testfaces)

    if(nf > 0):
        faces = pd.Series(testfaces)
        with warnings.catch_warnings():
          warnings.simplefilter("ignore", UserWarning)
          test_data = (DeepFakeImageList.from_pils(faces)
                 .split_none()
                 .label_from_array(np.arange(nf))
                 .transform(tfms)
                 .databunch(bs=nf))\
                 .normalize(imagenet_stats)
        return get_preds_from_learner(lrnr, test_data, sigmoid=sigmoid, fatten=fatten)

def get_preds_from_learner(lrnr, test_data, sigmoid=False, fatten=True):
    if (DEVICE != 'cpu'):
        lrnr.to_fp32()
    lrnr.data = test_data
    lrnr.data.valid_dl = lrnr.data.train_dl
    if (DEVICE != 'cpu'):
        lrnr.to_fp16()
    preds,y = lrnr.get_preds(ds_type=DatasetType.Valid)
    preds = preds[np.argsort(y)]
    preds = preds.detach().float().cpu()
    preds = preds.squeeze()
    # Depending on loss function used get_preds returns probability or logits
    if sigmoid == True:
        preds = torch.sigmoid(preds) if fatten == False else gsigmoid(preds)
    else:
        preds = gsigmoid(logits(preds)) if fatten == True else preds
    preds = torch.clamp(preds, EPS, 1-EPS)
    return preds.numpy()

def aggregate(preds, framenums):
    fp = pd.DataFrame(columns=['n','label'])
    fp['n'] = framenums
    fp['label'] = preds

    # get fakest face from each frame
    frame_preds = fp.groupby(['n']).agg({'label': ['max']})
    fp_df = pd.DataFrame(frame_preds)
    fp_df.reset_index(inplace=True)
    fp_df.columns = ['n','label']
    fakeness = fp_df['label'].mean()
    
    # spread out the probabilities to eliminate over-confident predictions
    fakeness = gsigmoid(logits(fakeness))
    return fakeness


def logloss(modeltype, eps=1e-5):
    def logloss_inner(x):
        predicted = x[modeltype]
        truth = x['truth']
        p = np.clip(predicted, eps, 1 - eps)
        if truth == 1:
            return -log(p)
        else:
            return -log(1 - p)
    return logloss_inner

