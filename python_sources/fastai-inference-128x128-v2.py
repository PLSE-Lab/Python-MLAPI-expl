#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_cell_magic('time', '', '%reload_ext autoreload\n%autoreload 2\n%matplotlib inline\nimport matplotlib.pyplot as  py\nimport cv2\nimport pandas as pd\nfrom fastai.vision import *\nimport os\nimport glob\nimport imageio\nimport warnings\nwarnings.filterwarnings("ignore")')


#  <h1>Preprocessing to obtain 128x128 images</h1>

# In[ ]:


get_ipython().run_cell_magic('time', '', "HEIGHT = 137\nWIDTH = 236\nSIZE = 128\nstats = (0.0692, 0.2051)\n#check https://www.kaggle.com/iafoss/image-preprocessing-128x128\nTEST = ['/kaggle/input/bengaliai-cv19/train_image_data_0.parquet',\n        '/kaggle/input/bengaliai-cv19/train_image_data_1.parquet',\n        '/kaggle/input/bengaliai-cv19/train_image_data_2.parquet',\n        '/kaggle/input/bengaliai-cv19/train_image_data_3.parquet']\ndef bbox(img):\n    rows = np.any(img, axis=1)\n    cols = np.any(img, axis=0)\n    rmin, rmax = np.where(rows)[0][[0, -1]]\n    cmin, cmax = np.where(cols)[0][[0, -1]]\n    return rmin, rmax, cmin, cmax\n\ndef crop_resize(img0, size=SIZE, pad=16):\n    #crop a box around pixels large than the threshold \n    #some images contain line at the sides\n    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 80)\n    #cropping may cut too much, so we need to add it back\n    xmin = xmin - 13 if (xmin > 13) else 0\n    ymin = ymin - 10 if (ymin > 10) else 0\n    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH\n    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT\n    img = img0[ymin:ymax,xmin:xmax]\n    #remove lo intensity pixels as noise\n    img[img < 28] = 0\n    lx, ly = xmax-xmin,ymax-ymin\n    l = max(lx,ly) + pad\n    #make sure that the aspect ratio is kept in rescaling\n    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')\n    return cv2.resize(img,(size,size))\nima=[]\nfor fname in TEST:\n    df = pd.read_parquet(fname)\n        #the input is inverted\n    data = 255 - df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)\n    for idx in range(len(df)):\n        #name = df.iloc[idx,0]\n        #normalize each image by its max val\n        img = (data[idx]*(255.0/data[idx].max())).astype(np.uint8)\n        img = crop_resize(img)\n        ima.append(img)")


# In[ ]:


del TEST
del HEIGHT
del WIDTH
del SIZE
del img
del data
del df
       


# # Saving the images in a directory
# 

# In[ ]:


get_ipython().run_cell_magic('time', '', "im128=np.array(ima)\ndef save_imgs(path:Path, data):\n    path.mkdir(parents=True,exist_ok=True)\n    for i in range(len(data)):\n        imageio.imsave(path/'{}.png'.format(i),data[i])\n        \nsave_imgs(Path('/data/test'),im128)")


# In[ ]:


#!cp /kaggle/input/grapheme-imgs-128x128 -r /data/train
del ima
del im128


# # Databunch creation for test images

# In[ ]:


get_ipython().run_line_magic('time', '')
ptrain = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')
ptrain['Image_path'] = ptrain.apply(lambda row: '/kaggle/input/grapheme-imgs-128x128/' + row.image_id + '.png', axis = 1)
ptrain['grapheme_root'] = ptrain.apply(lambda row: str(row.grapheme_root), axis = 1)
ptrain['vowel_diacritic'] = ptrain.apply(lambda row: str(row.vowel_diacritic), axis = 1)
ptrain['consonant_diacritic'] = ptrain.apply(lambda row: str(row.consonant_diacritic), axis = 1)


# In[ ]:


get_ipython().run_line_magic('time', '')
ptra=glob.glob('/kaggle/input/grapheme-imgs-128x128/*')
p1=pd.DataFrame(ptra,columns=['Image_path'])
def process(s):
    return str(s).split('/')[4]
p1['Image_path']=p1['Image_path'].apply(process)
ptrain['Image_path']=ptrain['Image_path'].apply(process)
p3=p1.merge(ptrain,on='Image_path',)


# In[ ]:


del ptrain
del p1
del ptra


# In[ ]:


get_ipython().run_line_magic('time', '')
tfms = get_transforms(do_flip=False,)
data = ImageDataBunch.from_folder('../input', 
                                  train="grapheme-imgs-128x128",
                                  size=128,bs=128).normalize(stats)
test=ImageList.from_folder('/data/test')


# In[ ]:


data.add_test(test,tfm_y=False)


# # Databunch creation for training images

# In[ ]:


get_ipython().run_cell_magic('time', '', "data_cd = ImageDataBunch.from_df(path='/kaggle/input/',folder='grapheme-imgs-128x128',df=p3,bs=128,size=128,label_col='consonant_diacritic',tfm_y=False).normalize(imagenet_stats)\ndata_gr = ImageDataBunch.from_df(path='/kaggle/input/',folder='grapheme-imgs-128x128',df=p3,bs=128,size=128,label_col='grapheme_root',tfm_y=False).normalize(imagenet_stats)\ndata_vd = ImageDataBunch.from_df(path='/kaggle/input/',folder='grapheme-imgs-128x128',df=p3,bs=128,size=128,label_col='vowel_diacritic',tfm_y=False).normalize(imagenet_stats)")


# # Model loading 

# In[ ]:


get_ipython().run_cell_magic('time', '', "if not os.path.exists('/root/.cache/torch/checkpoints'):\n        os.makedirs('/root/.cache/torch/checkpoints')\n!cp /kaggle/input/fastai-pretrained-models/densenet121-a639ec97.pth /root/.cache/torch/checkpoints/densenet121-a639ec97.pth\n\nlearn_cd = cnn_learner(data_cd, models.densenet121, metrics=[error_rate, accuracy],model_dir = Path('../kaggle/working'),).to_fp16()\nlearn_vd = cnn_learner(data_vd, models.densenet121, metrics=[error_rate, accuracy],model_dir = Path('../kaggle/working'),).to_fp16()\nlearn_gr = cnn_learner(data_gr, models.densenet121, metrics=[error_rate, accuracy], model_dir = Path('../kaggle/working'),).to_fp16()")


# In[ ]:


del data_cd
del data_vd
del data_gr


# In[ ]:


get_ipython().run_cell_magic('capture', '', "learn_gr.load('/kaggle/input/modelgr/best_gr_model')\nlearn_cd.load('/kaggle/input/models/best_cd_model',)\nlearn_vd.load('/kaggle/input/models/best_vd_model',)")


# In[ ]:


get_ipython().run_cell_magic('capture', '', 'm1_pred1=[]\nm2_pred2=[]\nm3_pred3=[]\nfor i in data.test_ds:\n    y1=learn_cd.predict(i[0])\n    y2=learn_vd.predict(i[0])\n    y3=learn_gr.predict(i[0])\n    m2_pred2.append(y1[1].item())\n    m3_pred3.append(y2[1].item())\n    m1_pred1.append(y3[1].item())\n    del y1\n    del y2\n    del y3')


# In[ ]:


del learn_gr
del learn_vd
del learn_cd


# # Prediction

# In[ ]:


# Converting data to submission format

# m1 CD
# m2 VD
sample_sub = pd.read_csv('/kaggle/input/bengaliai-cv19/sample_submission.csv')
cd = 0 
cd_itr = 0
gr = 1
gr_itr = 0
vd = 2 
vd_itr = 0
length = sample_sub['target'].shape[0]
for i in range(length):
    if(i==gr):
        sample_sub.at[i,'target'] = m1_pred1[gr_itr]
        gr_itr+=1
        gr+=3
    if(i==cd):
        sample_sub.at[i,'target'] = m2_pred2[cd_itr]
        cd_itr+=1
        cd+=3
    elif(i==vd):
        sample_sub.at[i,'target'] = m3_pred3[vd_itr]
        vd_itr+=1
        vd+=3
#print(sample_sub.head())
del cd
del cd_itr
del gr
del gr_itr
del vd
del vd_itr
del length


# In[ ]:


# Writing to submission csv file
sample_sub.to_csv('submission.csv', index=False)


# In[ ]:


sample_sub.head()


# In[ ]:


del sample_sub


# In[ ]:


get_ipython().system('ls')

