#!/usr/bin/env python
# coding: utf-8

# # Train on Crops - Predict on Full - CV 0.60+
# In this kernel we use a trick that we learned in Kaggle's Steel Competition. Since segmentation neural networks are all convolutions, you can train with one input size and predict with a different input size (explained in detail [here][1]). First we will resize all training and test images into size `700x1050` from their original `1400x2100`. Next we will train with random `352x512` crops and then feed full test images (`700x1050` images) into our network and get full segmentation masks!
# 
# [1]: https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/114321
# 

# # Load Data
# Kaggle has recently upgraded to TensorFlow 2.0. This is causing memory issues, so we will install TensorFlow 1.14 here. Next we will load and restructure the `train.csv` dataframe.

# In[ ]:


get_ipython().system('pip install tensorflow-gpu==1.14.0')
get_ipython().system('pip install keras==2.2.4')


# In[ ]:


import pandas as pd, numpy as np, os
from PIL import Image 
import cv2, keras, gc
import keras.backend as K
from keras import layers
from keras.models import Model
from keras.models import load_model
from keras.callbacks import LearningRateScheduler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt, time
from sklearn.metrics import roc_auc_score, accuracy_score
#os.listdir('../input/understanding_cloud_organization/')


# In[ ]:


sub = pd.read_csv('../input/understanding_cloud_organization/sample_submission.csv')
sub['Image'] = sub['Image_Label'].map(lambda x: x.split('.')[0])

PATH = '../input/understanding_cloud_organization/train_images/'
train = pd.read_csv('../input/understanding_cloud_organization/train.csv')
train['Image'] = train['Image_Label'].map(lambda x: x.split('.')[0])
train['Label'] = train['Image_Label'].map(lambda x: x.split('_')[1])
train2 = pd.DataFrame({'Image':train['Image'][::4]})
train2['e1'] = train['EncodedPixels'][::4].values
train2['e2'] = train['EncodedPixels'][1::4].values
train2['e3'] = train['EncodedPixels'][2::4].values
train2['e4'] = train['EncodedPixels'][3::4].values
train2.set_index('Image',inplace=True,drop=True)
train2.fillna('',inplace=True); train2.head()
train2[['d1','d2','d3','d4']] = (train2[['e1','e2','e3','e4']]!='').astype('int8')
train2.head()


# # Helper Functions
# Below are functions to manipulate `rle` masks. Click "code" to the right to see the code.

# In[ ]:


def mask2rleX(img0, shape=(1050,700), shrink=2):
    # USAGE: embeds into size shape, then shrinks, then outputs rle
    # EXAMPLE: img0 can be 600x1000. It will center load into
    # a mask of 700x1050 then the mask is downsampled to 350x525
    # finally the rle is outputted. 
    a = (shape[1]-img0.shape[0])//2
    b = (shape[0]-img0.shape[1])//2
    img = np.zeros((shape[1],shape[0]))
    img[a:a+img0.shape[0],b:b+img0.shape[1]] = img0
    img = img[::shrink,::shrink]
    
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle2maskX(mask_rle, shape=(2100,1400), shrink=1):
    # Converts rle to mask size shape then downsamples by shrink
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T[::shrink,::shrink]

def mask2contour(mask, width=5):
    w = mask.shape[1]
    h = mask.shape[0]
    mask2 = np.concatenate([mask[:,width:],np.zeros((h,width))],axis=1)
    mask2 = np.logical_xor(mask,mask2)
    mask3 = np.concatenate([mask[width:,:],np.zeros((width,w))],axis=0)
    mask3 = np.logical_xor(mask,mask3)
    return np.logical_or(mask2,mask3) 

def clean(rle,sz=20000):
    if rle=='': return ''
    mask = rle2maskX(rle,shape=(525,350))
    num_component, component = cv2.connectedComponents(np.uint8(mask))
    mask2 = np.zeros((350,525))
    for i in range(1,num_component):
        y = (component==i)
        if np.sum(y)>=sz: mask2 += y
    return mask2rleX(mask2,shape=(525,350), shrink=1)


# # Data Generator
# This data generator outputs random crops of size `352x512`. These crops are taken from the original `1400x2100` training images after they are resized to `700x1050` pixels. The masks are cropped to match the image crops. Also we have horizontal and vertical flip augmentation.
# 
# Below we display examples. The image on the left is the original image. The yellow rectangle is an original mask. The black rectangle is a random crop. The image on the right is the random crop outputted by the data generator. Notice how the original mask is cropped too.

# In[ ]:


class DataGenerator(keras.utils.Sequence):
    # USES GLOBAL VARIABLE TRAIN2 COLUMNS E1, E2, E3, E4
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=8, shuffle=False, width=512, height=352, scale=1/128., sub=1., mode='train_seg',
                 path='../input/understanding_cloud_organization/train_images/', ext='.jpg', flips=False, shrink=2):
        'Initialization'
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.path = path
        self.scale = scale
        self.sub = sub
        self.path = path
        self.ext = ext
        self.width = width
        self.height = height
        self.mode = mode
        self.flips = flips
        self.shrink = shrink
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        ct = int(np.floor( len(self.list_IDs) / self.batch_size))
        if len(self.list_IDs)>ct*self.batch_size: ct += 1
        return int(ct)

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X, y, msk, crp = self.__data_generation(indexes)
        if (self.mode=='display'): return X, msk, crp
        elif (self.mode=='train_seg')|(self.mode=='validate_seg'): return X, msk
        elif (self.mode=='train')|(self.mode=='validate'): return X, y
        else: return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(int( len(self.list_IDs) ))
        if self.shuffle: np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' 
        # Initialization
        lnn = len(indexes)
        X = np.empty((lnn,self.height,self.width,3),dtype=np.float32)
        msk = np.empty((lnn,self.height,self.width,4),dtype=np.int8)
        crp = np.zeros((lnn,2),dtype=np.int16)
        y = np.zeros((lnn,4),dtype=np.int8)
        
        # Generate data
        for k in range(lnn):
            img = cv2.imread(self.path + self.list_IDs[indexes[k]] + self.ext)
            img = cv2.resize(img,(2100//self.shrink,1400//self.shrink),interpolation = cv2.INTER_AREA)
            # AUGMENTATION FLIPS
            hflip = False; vflip = False
            if (self.flips):
                if np.random.uniform(0,1)>0.5: hflip=True
                if np.random.uniform(0,1)>0.5: vflip=True
            if vflip: img = cv2.flip(img,0) # vertical
            if hflip: img = cv2.flip(img,1) # horizontal
            # RANDOM CROP
            a = np.random.randint(0,2100//self.shrink-self.width+1)
            b = np.random.randint(0,1400//self.shrink-self.height+1)
            if (self.mode=='predict'):
                a = (2100//self.shrink-self.width)//2
                b = (1400//self.shrink-self.height)//2
            img = img[b:self.height+b,a:self.width+a]
            # NORMALIZE IMAGES
            X[k,] = img*self.scale - self.sub      
            # LABELS
            if (self.mode!='predict'):
                for j in range(1,5):
                    rle = train2.loc[self.list_IDs[indexes[k]],'e'+str(j)]
                    mask = rle2maskX(rle,shrink=self.shrink)
                    if vflip: mask = np.flip(mask,axis=0)
                    if hflip: mask = np.flip(mask,axis=1)
                    msk[k,:,:,j-1] = mask[b:self.height+b,a:self.width+a]
                    if (self.mode=='train')|(self.mode=='validate'):
                        if np.sum( msk[k,:,:,j-1] )>0: y[k,j-1]=1
            if (self.mode=='display'):
                crp[k,0] = a; crp[k,1] = b

        return X, y, msk, crp


# In[ ]:


types = ['Fish','Flower','Gravel','Sugar']
train_batch = DataGenerator(train2.index[:8], mode='display',batch_size=1,scale=1,sub=0)
for k,image in enumerate(train_batch):
    plt.figure(figsize=(15,8))
    
    # RANDOMLY PICK CLOUD TYPE TO DISPLAY FROM NON-EMPTY MASKS
    idx = np.argwhere( train2.loc[train2.index[k],['d1','d2','d3','d4']].values==1 ).flatten()
    d = np.random.choice(idx)+1
    
    # DISPLAY ORIGINAL
    img = Image.open(PATH+train2.index[k]+'.jpg'); img=np.array(img)
    mask = rle2maskX( train2.loc[train2.index[k],'e'+str(d)] )
    contour = mask2contour( mask,10 )
    img[contour==1,:2]=255
    diff = np.ones((1400,2100,3),dtype=np.int)*255-img.astype(int)
    img=img.astype(int); img[mask==1,:] += diff[mask==1,:]//6
    mask = np.zeros((1400,2100))
    a = image[2][0,1]*2
    b = image[2][0,0]*2
    mask[a:a+2*352,b:b+2*512]=1
    mask = mask2contour(mask,20)
    img[mask==1,:]=0
    plt.subplot(1,2,1); 
    plt.title('Original - '+train2.index[k]+'.jpg - '+types[d-1])
    plt.imshow(img);
    
    # DISPLAY RANDOM CROP
    img = image[0][0,]
    mask = image[1][0,:,:,d-1]
    contour = mask2contour( mask )
    img[contour==1,:2]=255
    diff = np.ones((352,512,3),dtype=np.int)*255-img.astype(int)
    img=img.astype(int); img[mask==1,:] += diff[mask==1,:]//6
    plt.subplot(1,2,2)
    plt.title('Training Crop - '+train2.index[k]+'.jpg - '+types[d-1])
    plt.imshow( img.astype(int) ); 
    plt.show()


# # Build Segmentation Model
# We will build a segmentation model using Qubvel's Keras Segmentation models [here][1]. Our architecture will be FPN (feature pyramid network) and our backbone will be Efficientnetb2. We will use Jaccard loss and Adam optimizer with learning rate 1e-4. Our metric will be Dice coef.
# 
# [1]: https://github.com/qubvel/segmentation_models

# In[ ]:


get_ipython().system(' pip install segmentation-models')

from segmentation_models import Unet,FPN
from segmentation_models.losses import bce_jaccard_loss, jaccard_loss
from keras.optimizers import Adam

def build_model():
    #model = Unet('resnet34', input_shape=(None,None,3), classes=4, activation='sigmoid')
    model = FPN('efficientnetb2', input_shape=(None, None, 3), classes=4, activation='sigmoid')
    #model = FPN('inceptionv3', input_shape=(None, None, 3), classes=4, activation='sigmoid')

    #model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=[dice_coef])
    #model.compile(optimizer=Adam(lr=0.0001), loss=bce_jaccard_loss, metrics=[dice_coef])
    model.compile(optimizer=Adam(lr=0.0001), loss=jaccard_loss, metrics=[dice_coef])
    #model.compile(optimizer=Adam(lr=0.0001), loss=dice_coef_loss, metrics=[dice_coef])
    return model


# In[ ]:


from keras import backend as K

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


# # Train Segmentation Model on Crops
# We will train with `352x512` crops, `batch_size=8`, and use 3-Fold validation. During training we predict OOF. We predict `test.csv` afterwards by saving the 3 models from the 3 folds. We will train each fold for 4 epochs. In our predicted oof masks, all pixel probabilities over 0.4 will be converted to 1 and less than 0.4 to 0. Any mask with fewer than `4*20000` pixels (predicted on `700x1050` image) will be regarded as no mask.

# In[ ]:


oof = np.empty_like(train2[['e1','e2','e3','e4']].values)

# K-FOLD MODELS
skf = KFold(n_splits=3, shuffle=True, random_state=42)
for k, (idxT, idxV) in enumerate( skf.split(train2) ):
        
    # TRAIN MODEL
    print(); print('#'*10,'FOLD',k,'#'*10)
    print('Train on',len(idxT),'Validate on',len(idxV))
    model = build_model()        
    train_gen = DataGenerator(train2.index[idxT],flips=True, shuffle=True)
    val_gen = DataGenerator(train2.index[idxV])
    h = model.fit_generator(train_gen, epochs = 4, verbose=2, validation_data = val_gen)
        
    # PREDICT OOF
    print('Predict OOF: ',end='')
    oof_gen = DataGenerator(train2.index[idxV], width=1024, height=672, batch_size=2, mode='predict')
    for b,batch in enumerate(oof_gen):
        btc = model.predict_on_batch(batch)
        for j in range(btc.shape[0]):
            for i in range(btc.shape[-1]):
                mask = (btc[j,:,:,i]>0.4).astype(int); rle =''
                if np.sum(mask)>4*20000: rle = mask2rleX( mask )
                oof[idxV[2*b+j],i] = rle
        if b%50==0: print(2*b,', ',end='')
        
    # SAVE MODEL AND FREE GPU MEMORY 
    model.save('Seg_'+str(k)+'.h5', overwrite=True)
    del train_gen, val_gen, oof_gen, model, h, idxT, idxV, btc, batch, b
    K.clear_session(); x=gc.collect()


# # Evaluate Segmentation Model using OOF
# We will evaluate OOF using Kaggle's Dice metric. For post processing, we will remove any contiguous piece of predicted mask with fewer than 20000 pixels (predicted on `350x525` image). And we will remove any mask with less than 0.5 probability as determined by our cloud classifer from our previous notebook [here][1]
# 
# [1]: https://www.kaggle.com/cdeotte/cloud-bounding-boxes-cv-0-58

# In[ ]:


def dice_coef6(y_true_rle, y_pred_prob, y_pred_rle, th):
    if y_pred_prob<th:
        if y_true_rle=='': return 1
        else: return 0
    else:
        y_true_f = rle2maskX(y_true_rle,shrink=4)
        y_pred_f = rle2maskX(y_pred_rle,shape=(525,350))
        union = np.sum(y_true_f) + np.sum(y_pred_f)
        if union==0: return 1
        intersection = np.sum(y_true_f * y_pred_f)
        return 2. * intersection / union


# In[ ]:


# LOAD CLASSIFICATION PREDICTIONS FROM PREVIOUS KERNEL
# https://www.kaggle.com/cdeotte/cloud-bounding-boxes-cv-0-58
for k in range(1,5): train2['o'+str(k)] = 0
train2[['o1','o2','o3','o4']] = np.load('../input/cloudpred1/oof.npy')[:len(train2),]

# LOAD OOF SEGMENTATION PREDICTIONS FROM 3-FOLD ABOVE
for k in range(1,5): train2['ee'+str(k)] = ''
train2[['ee1','ee2','ee3','ee4']] = oof
for k in range(1,5): train2['ee'+str(k)] = train2['ee'+str(k)].map(clean)

# COMPUTE KAGGLE DICE
th = [0.5,0.5,0.5,0.5]
for k in range(1,5):
    train2['ss'+str(k)] = train2.apply(lambda x:dice_coef6(x['e'+str(k)],x['o'+str(k)],x['ee'+str(k)],th[k-1]),axis=1)
    dice = np.round( train2['ss'+str(k)].mean(),3 )
    print(types[k-1],': Kaggle Dice =',dice)
dice = np.round( np.mean( train2[['ss1','ss2','ss3','ss4']].values ),3 )
print('Overall : Kaggle Dice =',dice)


# # View OOF Examples
# Below yellow outlines are true masks and blue outlines (with shaded insides) are predicted masks. The Dice score for each predicted mask is displayed above each image.

# In[ ]:


for d in range(1,5):
    print('#'*27); print('#'*5,types[d-1].upper(),'CLOUDS','#'*7); print('#'*27)
    plt.figure(figsize=(20,15)); k=0
    for kk in range(9):
        plt.subplot(3,3,kk+1)
        while (train2.loc[train2.index[k],'e'+str(d)]==''): k += 1
        f = train2.index[k]+'.jpg'
        img = Image.open(PATH+f); img = img.resize((525,350)); img = np.array(img)
        rle1 = train2.loc[train2.index[k],'e'+str(d)]; mask = rle2maskX(rle1,shrink=4)
        contour = mask2contour(mask,5); img[contour==1,:2] = 255
        rle2 = train2.loc[train2.index[k],'ee'+str(d)]; mask = rle2maskX(rle2,shape=(525,350))
        contour = mask2contour(mask,5); img[contour==1,2] = 255
        diff = np.ones((350,525,3),dtype=np.int)*255-img
        img=img.astype(int); img[mask==1,:] += diff[mask==1,:]//4
        dice = np.round( dice_coef6(rle1,1,rle2,0),3 )
        plt.title(f+'  Dice = '+str(dice)+'   Yellow true, Blue predicted')
        plt.imshow(img); k += 1
    plt.show()


# # Predict Test Images
# We load the models from our 3 folds above and use them to predict `test.csv`. Note that we must choose an input size that is divisble by 32. Therefore we choose `672x1024` instead of `700x1050`. We segment the middle of each test image ignoring the 14 pixel wide border around the edge.

# In[ ]:


from keras.models import load_model
model1 = load_model('Seg_0.h5',custom_objects={'dice_coef':dice_coef,'jaccard_loss':jaccard_loss})
model2 = load_model('Seg_1.h5',custom_objects={'dice_coef':dice_coef,'jaccard_loss':jaccard_loss})
model3 = load_model('Seg_2.h5',custom_objects={'dice_coef':dice_coef,'jaccard_loss':jaccard_loss})


# In[ ]:


print('Computing masks for',len(sub)//4,'test images with 3 models'); sub.EncodedPixels = ''
PTH = '../input/understanding_cloud_organization/test_images/'
test_gen = DataGenerator(sub.Image[::4].values, width=1024, height=672, batch_size=2, mode='predict',path=PTH)

for b,batch in enumerate(test_gen):
    btc = model1.predict_on_batch(batch)
    btc += model2.predict_on_batch(batch)
    btc += model3.predict_on_batch(batch)
    btc /= 3.0
    for j in range(btc.shape[0]):
        for i in range(btc.shape[-1]):
            mask = (btc[j,:,:,i]>0.4).astype(int); rle = ''
            if np.sum(mask)>4*20000: rle = mask2rleX( mask )
            sub.iloc[4*(2*b+j)+i,1] = rle
    if b%50==0: print(b*2,', ',end='')


# # Submit to Kaggle
# Using our classification predictions from our previous notebook [here][1], we will remove any segmentation mask with probability less than 0.5
# 
# [1]: https://www.kaggle.com/cdeotte/cloud-bounding-boxes-cv-0-58

# In[ ]:


# LOAD CLASSIFICATION PREDICTIONS FROM PREVIOUS KERNEL
# https://www.kaggle.com/cdeotte/cloud-bounding-boxes-cv-0-58
sub['p'] = np.load('../input/cloudpred1/preds.npy').reshape((-1))[:len(sub)]
sub.loc[sub.p<0.5,'EncodedPixels'] = ''

sub.EncodedPixels = sub.EncodedPixels.map(clean)
sub[['Image_Label','EncodedPixels']].to_csv('submission.csv',index=False)
sub.head(25)

