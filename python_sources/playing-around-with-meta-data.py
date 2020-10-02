#!/usr/bin/env python
# coding: utf-8

# In this simple kernel, I'm going to guess the degree a patient has Pneumothorax, and fill the submission with the most frequent masked pixels.
# 
# I know it is somewhat silly, but let's play around with the dataset.
# 
# Thanks to [@seesee](https://www.kaggle.com/seesee), the dataset is now available on Kaggle as well:
# https://www.kaggle.com/seesee/siim-train-test

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pylab as pl
import seaborn as sns
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pydicom as dcm
from glob import glob


Width, Height = 1024, 1024

# modified from the provided .py files
def mask2rle(img, width=Width, height=Height):
    if img is None or pl.isnan(img).all():
        return '-1'
    rle = []
    lastColor = 0;
    currentPixel = 0;
    runStart = -1;
    runLength = 0;

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel;
                    runLength = 1;
                else:
                    rle.append(str(runStart));
                    rle.append(str(runLength));
                    runStart = -1;
                    runLength = 0;
                    currentPixel = 0;
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor;
            currentPixel+=1;

    return " ".join(rle)

def rle2mask(rle, width=Width, height=Height):
    mask= np.zeros(width* height)
    if rle == '-1' or rle == ' -1':
        return mask.reshape(width, height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height)


# In[ ]:


trFolders = glob('../input/siim-train-test/siim/dicom-images-train/*')
tsFolders = glob('../input/siim-train-test/siim/dicom-images-test/*')
trainDF = pd.read_csv('../input/siim-train-test/siim/train-rle.csv', index_col='ImageId')


# DCM files have many attributes, that in our case only **ViewPosition**, **PatientAge**, **PatientSex** had meaningful values. View Position has two values of PA and AP, which are most probably posterior-anteror and anterior-posterior. They show the direction at which the scan was taken. i.e. from front or from back. (Correct me if I'm wrong)

# In[ ]:


Attrs = ['ViewPosition', 'PatientAge', 'PatientSex',
#         'StudyDate', 'Modality', 'ConversionType',
#          'ReferringPhysicianName','BodyPartExamined',
#          'SeriesNumber', 'InstanceNumber','PatientOrientation',
#          'SamplesPerPixel','PhotometricInterpretation','Rows', 'Columns', 
#         'PixelSpacing','BitsAllocated','BitsStored','HighBit', 'PixelRepresentation', 
#         'LossyImageCompression', 'LossyImageCompressionMethod',
        ]


def readMetaData(Folders, DF=pd.DataFrame()):
    missing = []
    for Fol in Folders:
        Sub = glob(Fol+'/*')
        File = glob(Sub[0]+'/*.dcm')[0]
        ID = File.split('/')[-1][:-4]
        try:
            DCM = dcm.dcmread(File)
            for atr in Attrs:
                DF.loc[ID, atr] = getattr(DCM, atr)
            Im = DCM.pixel_array
            for i, Part in enumerate([Im, 
                            Im[:512, :512], 
                            Im[:512, 512:], 
                            Im[512:, :512], 
                            Im[512:, 512:]]):
                DF.loc[ID, 'ImageMean_%d'%i] = Part.mean()
                DF.loc[ID, 'ImageStd_%d'%i] = Part.std()
        except Exception:
            missing.append(File)
    return DF, missing


# In[ ]:


trainDF, missingTR = readMetaData(trFolders, trainDF)
testDF, missingTS = readMetaData(tsFolders)
print(f'{len(missingTR)} files in train set are not downloaded perperly')
print(f'{len(missingTS)} files in test set are not downloaded perperly')


# From the provided masks I creat a new target column which shows the greatness (size) of the *Pneumothorax* and its log values. Later I'm going to predict this value based on only simple features provided in the metadata of the files.

# In[ ]:


trainDF.rename(columns={' EncodedPixels':'EncodedPixels'}, inplace=True)
trainDF['PneumMask'] = trainDF.EncodedPixels.apply(rle2mask)
trainDF['PneumSize'] = trainDF.PneumMask.apply(lambda mask: len(np.where(mask>0)[0]))
trainDF['PneumSizeLog'] = trainDF.PneumSize.apply(np.log1p)
trainDF.dropna(inplace=True)


# The masks are located on the lungs and if we aggregate them on top of each other:

# In[ ]:


MaskFreq = np.mean(trainDF[trainDF.PneumSize > 0].PneumMask.values, 0)
MaskFreq /= 255  # when the pixel is present in the mask, its value is 255
# It is the probability of being in the mask if the patient is diagnosed with Pneumothorax.

pl.figure(figsize=(10,10))
pl.imshow(MaskFreq.T);
pl.title('Most Frequent Spots')
pl.axis('off');


# We can see there are locations in the lungs more vulnerable to Pneumothorax. We later use this Map to guess Pneumothorax spots in the test set.
# 
# Let's take a look at frequency of Pneumothorax spots for different meta-data conditions:

# In[ ]:


pl.figure(figsize=(16,5))
for i, (Cond, Label)  in enumerate(zip([trainDF.PatientSex=='F', 
                                     trainDF.PatientSex=='M', 
                                     trainDF.ViewPosition=='PA', 
                                     trainDF.ViewPosition=='AP'], 
                                    ['Female','Male','PA View', 'AP View'])):
    pl.subplot(1,4,i+1)
    pl.imshow(np.sum(trainDF[Cond].PneumMask.values, 0).T);
    pl.title(Label)
    pl.axis('off');
pl.subplots_adjust(wspace=0.01, left=0.01, right=0.99)


# Lets continue with preparing datasets:
# 
# Some data cleaning is always necessary, here we should decide what to do with ages greater than 100. I guess they are typos, so I divide them by 10.

# In[ ]:


for DF in [trainDF, testDF]:
    DF['PatientAge'] = DF.PatientAge.astype(int)
    DF['PatientAge'] = DF.PatientAge.apply(lambda x:x if x<100 else int(x/10))


# In[ ]:


trainDF.head()


# In[ ]:


testDF.head()


# Distribution of Patient Ages seems to be similar in train and test sets:

# In[ ]:


sns.distplot(trainDF.PatientAge);
sns.distplot(testDF.PatientAge);


# In[ ]:


print('Train Set:\n', trainDF.PatientSex.value_counts())
print('Test Set:\n', testDF.PatientSex.value_counts())


# In[ ]:


print('Train Set:\n', trainDF.ViewPosition.value_counts())
print('Test Set:\n', testDF.ViewPosition.value_counts())


# There are more Male subjects, and more PA scans in both of the train and test sets.

# In[ ]:


pl.figure(figsize=(15,7))
pl.scatter(trainDF.PatientAge, trainDF.PneumSizeLog, 
           c=list(map(lambda k: 5 if k == 'F' else  10, trainDF.PatientSex)),
           s=list(map(lambda k: 5 if k == 'PA' else 10, trainDF.ViewPosition)), 
          )
pl.xlabel('Age')
pl.ylabel('Pneum Size (Log)')
pl.legend(['F:Blue \nM:Yellow']);


# In[ ]:


pl.figure(figsize=(15,10))
sns.lineplot(x='PatientAge',hue='PatientSex', y='PneumSizeLog', data=trainDF);


# The relation of detected Pneum Size to the Patients age, sex, and imaging direction is not strong.

# In[ ]:


pl.figure(figsize=(15,7))
pl.scatter(trainDF.ImageMean_0, trainDF.ImageStd_0, c=trainDF.PneumSizeLog, s=trainDF.PneumSizeLog*2+1);
pl.xlabel('Pixel Average')
pl.ylabel('Pixel STD')
cb=pl.colorbar()
cb.set_label('Pneum Size')


# Simple Pixel statistics also don't seem to be predictive of Pneum Size. 
# 
# But Let's build our model anyways:
# 
# I make a simple CatBoostRegressor model with no fancy hyper-parameters:

# In[ ]:


from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as mse

Vars = ['ViewPosition', 'PatientAge', 'PatientSex'] +         sum([['ImageMean_%d'%i, 'ImageStd_%d'%i] for i in range(5)], []) 
nFolds = 10
trOutput = pl.zeros(len(trainDF))
tsOutput = pl.zeros(len(testDF))
spliter = KFold(nFolds)
for trInx, vlInx in spliter.split(trainDF[Vars], trainDF.PneumSizeLog):
    Xtr, Xvl = trainDF[Vars].iloc[trInx], trainDF[Vars].iloc[vlInx]
    Ytr, Yvl = trainDF.PneumSizeLog.iloc[trInx], trainDF.PneumSizeLog.iloc[vlInx]
    Reg = CatBoostRegressor(objective='RMSE', n_estimators=10000)
    Reg.fit(Xtr, Ytr, cat_features=['ViewPosition', 'PatientSex'], silent=True,
            eval_set=[(Xvl, Yvl)], early_stopping_rounds=500)
    trOutput[vlInx] = Reg.predict(Xvl)
    tsOutput += Reg.predict(testDF[Vars]) / nFolds

pl.plot(trainDF.PneumSizeLog, trOutput, '.')
pl.title('rmse: %.5g' % mse(trainDF.PneumSizeLog, trOutput)**0.5)
pl.xlabel('train log Pneumothorax Size')
pl.ylabel('train Output');


# In[ ]:


tsOutput[tsOutput<0] = 0
testDF['PneumSize'] = np.expm1(tsOutput)


# Based on the Frequency of Pneumothorax spots and predicted size of the colapsed area we can extract masks that are most probable, with the desired size:

# In[ ]:


def getMask(numPos, MaskFreq=MaskFreq):
    prob = MaskFreq / MaskFreq.sum() * numPos
    return np.where(pl.rand(prob.shape[0], prob.shape[1])<=prob, 255, 0)


testDF['Mask'] = testDF.PneumSize.apply(getMask)


# Let's make a submission based on that:

# In[ ]:


sub = pd.read_csv('../input/siim-acr-pneumothorax-segmentation/sample_submission.csv', index_col='ImageId')
sub = sub.join(testDF[['Mask']])
sub['EncodedPixels'] = sub.Mask.apply(mask2rle)
sub[['EncodedPixels']].to_csv('submission.csv', index_label='ImageId')


# In[ ]:


sub.sample(10)

