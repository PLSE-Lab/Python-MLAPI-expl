#!/usr/bin/env python
# coding: utf-8

# This kernel shows a simple experiment with confusing an image classifying NN by applying gradient descent to the input image. We can apparently thoroughly confuse one NN (resnet50 in the example below) without much visible change to the image, and without confusing another NN classifier (inceptionV3 in the example below).

# ## Load libraries

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import os
from keras import backend as K
from keras.applications import resnet50,inception_v3
from keras.preprocessing import image
get_ipython().system('cp ../input/testinp/imagenetCIDS.py .')
import imagenetCIDS
print(len(imagenetCIDS.idList),'imagenet classes')


# ## Load pretrained NN models

# In[ ]:


def copyInputFile(fPathL,outDir) :
    inFPath = os.path.join(*(['..','input']+fPathL))
    outFPath = os.path.join(outDir,fPathL[-1])
    if os.path.exists(outFPath) :
        print(outFPath,'already exists')
    else :
        print('copying',inFPath)
        print('->',outFPath)
        with open(inFPath,'rb') as inF :
            with open(outFPath,'wb') as outF :
                outF.write(inF.read())
def loadKerasModel(mName,mClass,include_top=True) :
    modelDir = os.path.expanduser(os.path.join('~', '.keras', 'models'))
    if not os.path.exists(modelDir):
        print('creating',modelDir)
        os.makedirs(modelDir)
    copyInputFile(['keras-pretrained-models','imagenet_class_index.json'],modelDir)
    mFName = mName + '_weights_tf_dim_ordering_tf_kernels'
    if not include_top :
        mFName += '_notop'
    copyInputFile(['keras-pretrained-models',mFName+'.h5'],
                  modelDir)
    return mClass(weights='imagenet',include_top=include_top)


# In[ ]:


resnet = loadKerasModel('resnet50',resnet50.ResNet50)
incept = loadKerasModel('inception_v3',inception_v3.InceptionV3)


# ## Load St. Bernard image and resize for input to NN models

# In[ ]:


def displayImgArray(img,figsize=(12,10),axis='off') :
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(img / 255.)
    ax.axis(axis)
    plt.show()
def checkOrigImage(fPath,disp=True,targSquareSize=224, distortToSquare=False) :
    img = image.img_to_array(image.load_img(fPath))
    imHeight,imWidth,_ = img.shape
    print('original shape',img.shape)
    if distortToSquare :
        targWidth = targHeight = targSquareSize
    else :
        if imHeight <= imWidth :
            targWidth = targSquareSize
            targHeight = (targWidth*imHeight)//imWidth
        else :
            targHeight = targSquareSize
            targWidth = (targHeight*imWidth)//imHeight
    if disp :
        displayImgArray(img)
    return fPath,(targHeight,targWidth)
def loadSquareImage(origImageInfo,disp=True) :
    fPath,targSize = origImageInfo
    img = image.img_to_array(image.load_img(fPath, target_size=targSize))
    if img.shape[0] != img.shape[1] :
        squareImSize = max(img.shape[0],img.shape[1])
        squareImg = np.zeros((squareImSize,squareImSize,img.shape[2]),img.dtype)
        squareImg[:img.shape[0],:img.shape[1],:] = img
        img = squareImg
    if disp :
        displayImgArray(img,(6,6))
    return img


# In[ ]:


origImInfo = checkOrigImage('../input/testinp/stbernard.jpg',distortToSquare=True)
print(origImInfo)


# In[ ]:


sqIm = loadSquareImage(origImInfo)


# ## Calculate the spoofed image using gradient descent on image pixels

# In[ ]:


meanVals = np.array([103.939, 116.779, 123.68]) # mean pixel values for resnet
def unprocessIm(img) :
    """undo the image preprocessing - works only with resnet!"""
    img = img.copy()
    for i,meanVal in enumerate(meanVals) :
        img[..., i] += meanVal
    img = img[...,::-1]
    return img
def clipX(x) :
    """clip the processed image to valid bounds - works only with resnet!"""
    np.minimum(x,255.0-meanVals,x)
    np.maximum(x,-meanVals,x)
def classifyImg(img,mModule,model,doPrep=True,addLabel=None,thresh=0.05) :
    if doPrep :
        x = mModule.preprocess_input(np.expand_dims(img.copy(), axis=0))
    else :
        x = img
    preds = model.predict(x)[0]
    predInds = [(i,v) for i,v in enumerate(preds) if v>=thresh]
    predInds.sort(key = lambda x : x[1], reverse=True)
    if addLabel is not None and addLabel not in [i for i,v in predInds] :
        predInds.append((addLabel,preds[addLabel]))
    print([(imagenetCIDS.idList[i],v) for i,v in predInds])
    return preds,predInds
def makeSpoof(img,mModule,model,spoofLabel,spoofThresh=0.1) :
    model.compile('adam','mse')
    print('trying to spoof',repr(imagenetCIDS.idList[spoofLabel]))
    grads = model.optimizer.get_gradients(model.output[0][spoofLabel],model.input)
    gradsFunc = K.function(inputs = [model.input], outputs = grads)
    inp = np.expand_dims(img.copy(), axis=0)
    print(inp.shape)
    x = mModule.preprocess_input(inp)
    for i in range(100) :
        g = gradsFunc([x])
        #print(np.max(g))
        x += (1.0/np.max(g[0]))*g[0]
        clipX(x)
        print(str(i)+':',end=' ')
        preds,predInds = classifyImg(x,mModule,model,doPrep=False,addLabel=spoofLabel)
        if spoofLabel==predInds[0][0] and preds[spoofLabel]>=spoofThresh :
            break
    return unprocessIm(x[0])


# In[ ]:


spoofIm = makeSpoof(sqIm,resnet50,resnet,spoofLabel=100) # 100 = Black Swan!


# ## Original image is classified as St. Bernard by both models

# In[ ]:


displayImgArray(sqIm,(6,6))
_ = classifyImg(sqIm,resnet50,resnet)
_ = classifyImg(sqIm,inception_v3,incept)


# ## Spoofed image doesn't appear much different

# In[ ]:


displayImgArray(spoofIm,(6,6))


# ## But is classified as a Black Swan by the spoofed NN model

# In[ ]:


_ = classifyImg(spoofIm,resnet50,resnet)


# ## Though the second NN model still says St. Bernard

# In[ ]:


_ = classifyImg(spoofIm,inception_v3,incept)


# ## Stats on differences between original and spoofed images

# In[ ]:


diffIm = spoofIm-sqIm
print('min/max pixel difference:',np.min(diffIm),np.max(diffIm))
print('average pixel difference:',np.average(np.abs(diffIm)))


# ## Directly visualizing the differences doesn't show much

# In[ ]:


displayImgArray(np.abs(diffIm),(6,6))


# ## Unless we bump up the contrast a lot

# In[ ]:


displayImgArray((5.0*(diffIm-np.min(diffIm)))*(np.abs(diffIm)>=1.0),(6,6))


# In[ ]:




