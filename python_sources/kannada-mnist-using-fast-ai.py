#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.vision import *
from fastai.metrics import *
import imageio


# In[ ]:


path = Path('/kaggle/input/Kannada-MNIST')
path.ls()


# In[ ]:


train_data = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')
predict_data = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')


# In[ ]:


train_data.describe()


# In[ ]:


train_data.head()


# In[ ]:


train_data.shape


# In[ ]:


predict_data.describe()


# In[ ]:


predict_data.head()


# In[ ]:


predict_data.shape


# In[ ]:


print(f'train_data shape : {train_data.shape}')
print(f'predict_data shape  : {predict_data.shape}')


# In[ ]:


def to_img_shape(data_X, data_y=[]):
    data_X = np.array(data_X).reshape(-1,28,28)
    data_X = np.stack((data_X,)*3, axis=-1)
    data_y = np.array(data_y)
    return data_X,data_y


# In[ ]:


train_data_X, train_data_y = train_data.loc[:,'pixel0':'pixel783'], train_data['label']


# In[ ]:


print(f'train_data shape : {train_data_X.shape}')
print(f'train_data_y shape : {train_data_y.shape}')


# In[ ]:


from sklearn.model_selection import train_test_split

train_X, validation_X, train_y, validation_y = train_test_split(train_data_X, train_data_y, test_size=0.20,random_state=7,stratify=train_data_y)


# In[ ]:


print(f'train_X shape : {train_X.shape}')
print(f'train_y shape : {train_y.shape}')
print(f'validation_X shape : {validation_X.shape}')
print(f'validation_y shape : {validation_y.shape}')


# In[ ]:


train_X,train_y = to_img_shape(train_X,train_y)
validation_X,validation_y = to_img_shape(validation_X,validation_y)


# In[ ]:


print(f'train_X shape : {train_X.shape}')
print(f'train_y shape : {train_y.shape}')
print(f'validation_X shape : {validation_X.shape}')
print(f'validation_y shape : {validation_y.shape}')


# In[ ]:


def save_imgs(path:Path, data, labels):
    path.mkdir(parents=True,exist_ok=True)
    for label in np.unique(labels):
        (path/str(label)).mkdir(parents=True,exist_ok=True)
    for i in range(len(data)):
        if(len(labels)!=0):
            imageio.imsave( str( path/str(labels[i])/(str(i)+'.jpg') ), data[i] )
        else:
            imageio.imsave( str( path/(str(i)+'.jpg') ), data[i] )

save_imgs(Path('/data/train'),train_X,train_y)
save_imgs(Path('/data/valid'),validation_X,validation_y)


# In[ ]:


path = Path('/data')
path.ls()


# In[ ]:


tfms = get_transforms(do_flip=False )


# In[ ]:


data = (ImageList.from_folder('/data/') 
        .split_by_folder()          
        .label_from_folder()        
        .add_test_folder()          
        .transform(tfms, size=64)   
        .databunch())


# In[ ]:


#Another way to create data bunch
#data = ImageDataBunch.from_folder(path, ds_tfms=tfms, size=64)


# In[ ]:


data


# In[ ]:


data.show_batch(5,figsize=(6,6))


# In[ ]:


data.classes, data.c, len(data.train_ds), len(data.valid_ds)


# In[ ]:


get_ipython().system('mkdir -p /root/.cache/torch/checkpoints/')
get_ipython().system('cp /kaggle/input/fast-ai-models/resnet50-19c8e357.pth /root/.cache/torch/checkpoints/resnet50-19c8e357.pth')


# In[ ]:


learn = cnn_learner(data, models.resnet50, metrics=[error_rate, accuracy], model_dir=Path('/kaggle/input/fast-ai-models'))


# In[ ]:


learn.fit_one_cycle(5)


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_top_losses(20, figsize=(20,20))


# In[ ]:


interp.plot_confusion_matrix(figsize=(20,20), dpi=100)


# In[ ]:


interp.most_confused(min_val=2)


# In[ ]:


learn.model_dir = '/kaggle/output/fast-ai-models/'


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


lr = slice(1e-04)


# In[ ]:


learn.save('stage-1')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(3,lr)


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_top_losses(20, figsize=(20,20))


# In[ ]:


interp.plot_confusion_matrix(figsize=(20,20), dpi=100)


# In[ ]:


interp.most_confused(min_val=2)


# In[ ]:


learn.save('stage-2')


# In[ ]:


predict_data.drop('id',axis = 'columns',inplace = True)
sub_df = pd.DataFrame(columns=['id','label'])


# In[ ]:


my_predict_data = np.array(predict_data)


# In[ ]:


# Handy function to get the image from the tensor data
def get_img(data):
    t1 = data.reshape(28,28)/255
    t1 = np.stack([t1]*3,axis=0)
    img = Image(FloatTensor(t1))
    return img


# In[ ]:


from fastprogress import progress_bar
mb=progress_bar(range(my_predict_data.shape[0]))


# In[ ]:


for i in mb:
    timg=my_predict_data[i]
    img = get_img(timg)
    sub_df.loc[i]=[i+1,int(learn.predict(img)[1])]


# In[ ]:


def decr(ido):
    return ido-1

sub_df['id'] = sub_df['id'].map(decr)
sub_df.to_csv('submission.csv',index=False)


# In[ ]:


# Displaying the submission file
sub_df.head()

