#!/usr/bin/env python
# coding: utf-8

# # Model Ensembling and Transfer Learning with the Kannada MNIST datasets (and fastai)

# ## Brief background: Kannada MNIST
# This Playground Code Competition is to detect Kannada digits. Kannada is a language spoken predominantly by people of Karnataka in southwestern India. The digits range from omdu (1) to hattu (10).

# ## Playground Competiton: Let's play!
# The goal of this notebook is to implement two techniques: model ensembling and transfer learning. 
# 
# In [model ensembling](https://towardsdatascience.com/simple-guide-for-ensemble-learning-methods-d87cc68705a2), the decisions of many models is combined to make a final prediction. All of the kernels I have submitted so far for this competition have been a single model that predicts one of 10 categories for the digits. In this kernel, instead of creating one model for 10 digits, I created ten models, each predicting a single digit. For example, the first model would predict whether the digit was of category "0" or "not 0," the second model would predict category "1" or "not one," etc. 
# 
# However, I don't want to fully train 10 models. This is what brings me to the second part of my goal: [transfer learning](https://machinelearningmastery.com/how-to-improve-performance-with-transfer-learning-for-deep-learning-neural-networks/). In transfer learning, a model that has learned in one setting is used to improve generalization in another setting. For example, a model might be trained on the images in [ImageNet](http://www.image-net.org/), and then used to perform another task, like identifying the subset of [dog breeds](https://www.kaggle.com/c/dog-breed-identification). Here, I want to create a 10 Kannada digit model, and transfer the trained convolutional portion to my single digit models.
# 
# This kernel follows the following steps:
# 1. Define commmonly used methods for reading and creating data structures
# 1. Train the model I will use for transfer learning (aka, the 'backbone' model)
# 1. Create a new model for single digit detection that incorporates the pretrained 'backbone' model
# 1. Train the single digit models and retain their predictions
# 1. Ensemble the results of all models to obtain the submission
# 
# I've used fastai in this kernel, but my hope is that you can take the ideas presented here, and apply it to whatever framework you like best.

# In[ ]:


from fastai import *
from fastai.vision import *
import gc; 
DATAPATH = Path('/kaggle/input/Kannada-MNIST/')
#PATH = Path.cwd()
#DATAPATH = PATH/'data'

import os
for dirname, _, filenames in os.walk(DATAPATH):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


csv_train = DATAPATH/'train.csv'
csv_test = DATAPATH/'test.csv'
csv_extra = DATAPATH/'Dig-MNIST.csv'


# # 1. Commonly used methods

# The data is provided in a comma separated value file, and I need it to be a Pandas `DataFrame`. When creating the DataFrame, I add extra columns:
# - fn: is a duplicate of the index, which is the 'image number'
# - is_valid: defines a 10% validity set
# - binary: this column will be used in the single digit models

# In[ ]:


def panda_from_csv(csv):
    panda = pd.read_csv(csv)
    panda['fn'] = panda.index
    panda['is_valid'] = np.random.random(len(panda))
    panda['is_valid'] = panda['is_valid'] > 0.9
    panda['binary'] = 1
    return panda


# In this notebook, I'm going to be using the Fastai APIs. This custom ImageList for pixels [was inspired by this very helpful kernel](https://www.kaggle.com/heye0507/fastai-1-0-with-customized-itemlist). To open an image, the function is passed a 'filename' (fn), which is a string containing the index number with '../' in front. This code finds the correct row, and selects the needed colums of that row. The resulting array is reshaped into a 28x28 matrix, and returned as an image. 
# 
# Previously, I used [this super interesting kernel](https://www.kaggle.com/joatom/kannada-mnist-speed-up-fastai-image-processing), but I was getting random performance after ensembling. After debugging I'm still stumped as to why, so I reverted back to the slower version.

# In[ ]:


class PixelImageItemList(ImageList):
    
    def open(self,fn):
        img_pixel = self.inner_df.loc[self.inner_df['fn'] == int(fn[2:])].values[0,1:785]
        img_pixel = img_pixel.reshape(28,28)
        return vision.Image(pil2tensor(img_pixel,np.float32).div_(255))


# `bs` is the batch size, and `tfms` gets the standard fastai transforms, minus the image flipping. I did not flip the images because that would create noise instead of helping the model generalize (think about a '3').

# In[ ]:


bs = 512
tfms=get_transforms(do_flip=False)


# In fastai, the data and dataloaders are stored as a `DataBunch`. This codde will get the images, create the training/validation sets and add the labels. Next, I specify whether I want test time augmentation or not. Finally, I create the databunch, and normalize.

# In[ ]:


def get_databunch(df, col_label, tta=True):
    
    src = (PixelImageItemList.from_df(df,'.',cols='fn')
          .split_from_df('is_valid')
          .label_from_df(cols=col_label))
    with_tta = (tfms[0],tfms[0])
    if not tta:
        with_tta = (tfms[0],[])
    data = (src.transform(tfms=with_tta)
           .databunch(num_workers=2,bs=bs)
           .normalize())
    return data


# # 2. Train the 'backbone' model

# The model I am using is an implementation of the 'best' original MNIST architecture found [here](https://www.kaggle.com/cdeotte/how-to-choose-cnn-architecture-mnist). This great tutorial was created using Keras, and here I've re-implemented it using a combination of Pytorch and Fastai. The Fastai `conv_layer` function returns a sequence of convolutional, ReLU and batchnorm layers. 
# 
# Note that this model is divided into two sections. The first is the backbone, that contains the convolutional layers, and the second is the 'head', that contains the linear layers. Since I will use the backbone of this model as the pretrained portion of my single digit models, I am calling this the 'backbone model'.

# In[ ]:


best_architecture = nn.Sequential(
    # backbone
    nn.Sequential(
        conv_layer(1,32,stride=1,ks=3),
        conv_layer(32,32,stride=1,ks=3),
        conv_layer(32,32,stride=2,ks=5),
        nn.Dropout(0.4),
    
        conv_layer(32,64,stride=1,ks=3),
        conv_layer(64,64,stride=1,ks=3),
        conv_layer(64,64,stride=2,ks=5),
        nn.Dropout(0.4),),
    
    # head
    nn.Sequential(
        Flatten(),
        nn.Linear(3136, 128),
        relu(inplace=True),
        nn.BatchNorm1d(128),
        nn.Dropout(0.4),
        nn.Linear(128,10),
        nn.Softmax(dim=-1))
)


# This methods returns a `Learner`. In fastai the `Learner` is a container that holds the data, the model, and other associated details. I have chosen to use weighted cross entropy as my loss function across all models. The second line of this function specifies where the 'backbone' and 'head' of my model are.

# In[ ]:


def get_learner(dbch, arch, weights):
    learn = Learner(dbch, arch, loss_func = nn.CrossEntropyLoss(weight=class_weights), 
                    metrics=[accuracy])
    learn.split([learn.model[0], learn.model[1]]);
    return learn


# Since I am using weighted cross entropy, I want to pass in the weights of my classes. In the 10 digit datasets, the classes are even, so I have a tensor of 10 1's. This is probably not needed at this point, but the weights become especially important in the single digit models.

# In[ ]:


weights = [1.,1.,1.,1.,1.,1.,1.,1.,1.,1.]
class_weights=torch.FloatTensor(weights).cuda()


# ## Train the out of domain Kannada digits

# The first training set is the out of domain Kannada digits. From the [website](https://www.kaggle.com/c/Kannada-MNIST):
# > ...the author also disseminated an additional real world handwritten dataset (with 10k images), termed as the 'Dig-MNIST dataset' that can serve as an out-of-domain test dataset. It was created with the help of volunteers that were non-native users of the language, authored on a smaller sheet and scanned with different scanner settings compared to the main dataset. This 'dig-MNIST' dataset serves as a more difficult test-set.
# 
# Here, I'm using it as a noisy, small, initial training set.

# In[ ]:


dbch_extra = get_databunch(panda_from_csv(csv_extra), 'label', False)


# Show what numbers are in a single batch to make sure everything seems ok. A characteristic of this dataset is noisy vertical bars that appears to be artifacts.

# In[ ]:


dbch_extra.show_batch(rows=3,figsize=(4,4),cmap='bone')


# In[ ]:


learn = get_learner(dbch_extra, best_architecture, class_weights)


# The backbone model is completely untrained so far. Therefore, I want to train all of the layers from the beginning. Since I told the learner which part of the model was my 'backbone' and which is my 'head,' the default behavior is to freeze (don't train) the backbone. To preven this, I will 'unfreeze' to train the entire model.

# Before training, I want to find an acceptable learning rate. The following graph helps to find one in the right ballpark.
# - commented out to reduce time for competition, 1e-2 was a good learning rate

# In[ ]:


#learn.lr_find()
#learn.recorder.plot()


# In[ ]:


def train_kannada_mnist(u_epochs,lr):
    learn.unfreeze()
    learn.fit_one_cycle(u_epochs,lr)


# This is a small dataset, so I've made the arbitrary choice to train for 10 epochs. When I ran this on my computer, a learning rate of 1e-2 was acceptable.

# In[ ]:


train_kannada_mnist(10,1e-2)


# In[ ]:


learn.save('after_extra_kannada')


# ## Train on the Kannada MNIST training set

# Now the model will be pretrained on the Kannada MNIST training set. It is very important that I keep the training dataframe because it specifies my training/validation split. If I created a new training/testing split for my single digit models, some of the digits from the training set could then be in the validation set. I'm not entirely sure this would matter because I am using a different model head, but I'm not going to risk it.

# In[ ]:


train_df = panda_from_csv(csv_train)


# In[ ]:


dbch_train = get_databunch(train_df, 'label', False)


# In[ ]:


dbch_train.show_batch(rows=3,figsize=(4,4),cmap='bone')


# In[ ]:


learn = get_learner(dbch_train, best_architecture, class_weights)
learn.load('after_extra_kannada');


# Find an appropriate learning rate for this pretrained model.
# - commented out due to time constraints in the competition, 1e-2 was fine for the lr

# In[ ]:


#learn.lr_find()
#learn.recorder.plot()


# This model has quite a different loss curve. However, 1e-2 still seems to be a good choice.
# 
# Now the backbone model is already partially trained. Therefore, we will first train the 'head' on the training digits by using `freeze()` to not train the 'backbone' layers. Then, the entire model is 'unfrozen' and fine tuned for a few epochs. The number of frozen and unfrozen epochs is arbitrary chosen. Since I'm using this as my pretrained model, I'm not concerned about training for lots of epochs. 99% accuracy tells me that the model is good at recognizing the digits. 

# In[ ]:


def train_kannada_mnist(f_epochs,u_epochs,lr):
    learn.freeze()
    learn.fit_one_cycle(f_epochs,lr)
    learn.unfreeze()
    learn.fit_one_cycle(u_epochs,lr/2)


# In[ ]:


train_kannada_mnist(1,3,1e-2)


# In[ ]:


learn.save('after_train_kannada')


# # 3. Create the single digit model

# To create a single digit classifier, I create a new 'head' with two output categories. I reduced the amount of dropout from 0.4 to 0.1 because of the reduction in categories. In addition, I removed the number of nodes in the linear layer from 128 to 64.

# In[ ]:


binary_head = nn.Sequential(
    Flatten(),
    nn.Linear(3136, 64),
    relu(inplace=True),
    nn.BatchNorm1d(64),
    nn.Dropout(0.1),
    nn.Linear(64,2),
    nn.Softmax(dim=-1)
)


# Then, my new architecture would be the 'backbone' of the first model, combined with the new two category 'head'. 

# In[ ]:


empty_binary = nn.Sequential(best_architecture[0],binary_head)


# However, the model I just created is empty. My pretrained model contains the previously trained backbone, with the new 'head'. 

# In[ ]:


pretrained_model = nn.Sequential(learn.model[0],binary_head)


# Before training the single digit classifiers, I want to ensure that I have an appropriate learning rate. Here, I am creating a the labels for the 0th category, and creating a learner with the new pretrained single digit model.

# In[ ]:


is_0 = train_df['label'] == 0
train_df['binary'] = np.multiply(is_0,1)
dbch_bin = get_databunch(train_df, 'binary', True)


# Here we can see that now the labels are either 1 or 0.

# In[ ]:


dbch_bin.show_batch(rows=3,figsize=(4,4),cmap='bone')


# Here it's important to note that the categories of the digits are mostly (if not all) 0. This makes sense because now instead of having 10 equally sized categories, we have one positive category as 1 and the nine other categories combined to make up the 0 category. This creates an imbalanced data set. To account for the data imbalance, I will adjust the weights of the categories for the cross entropy loss.

# In[ ]:


weights = [1.,9.]
class_weights=torch.FloatTensor(weights).cuda()


# In[ ]:


binary_model = get_learner(dbch_bin, pretrained_model, class_weights)


# Find an acceptable lr for this new model.
# - This is commented out due to time constraints in the competition.

# In[ ]:


#binary_model.lr_find()
#binary_model.recorder.plot()


# It seems like 1e-2 will work for these models, too. Save the model so it can be used for single digit recognition later.

# In[ ]:


binary_model.save('pretrained_binary')


# I'm a bit worried about running out of RAM on the Kaggle kernel, although I'm not sure how founded that fear is. Here's two variables I know I won't need below.

# In[ ]:


del(binary_model)
del(dbch_bin)
gc.collect();


# # 4. Train the single digit models

# First the test set is created as a PixelImageList. We will add the test set to the learner after it is trained.

# In[ ]:


df_test = panda_from_csv(csv_test)
df_test.rename(columns={'id':'label'}, inplace=True)
test_set = PixelImageItemList.from_df(df_test,path='.',cols='fn')


# The predictions from the models will be ensembled after all 10 models have been trained. I make a numpy array that will hold all of the model predictions. Here, there are 5,000 test images, 10 single digit models, n_tta_aug test time augmentations, and 2 outcomes of each model. For the test time augmentation, augmentation are applied to the test set, allowing the same model to make multiple predictions about a digit.

# In[ ]:


n_tta_aug = 13
preds = np.zeros([len(df_test),10,n_tta_aug,2])


# This functions trains all of the single digit models and records the output.The following process is repeated for each digit:
# 1. New labels are created from the trainin_df. These labels are stored in the 'binary' column.
# 1. A databunch is created with the 'binary' column as the label, and test time augmentation.
# 1. The model 'head' is trained
#     - Note that I chose not to unfreeze the model and train the whole thing. I have pretrained my 'backbone' to decent accuracy, and I note that unfreezing causes a decrease in accuracy.
# 1. The test set is added, and the test time augmentation predictions are recorded

# In[ ]:


def train_binary_models(f_epochs, lr):

    for model_num in range(0,10):
        print('training model for',model_num)
        # create the new binary labels
        is_num = train_df['label'] == model_num
        train_df['binary'] = np.multiply(is_num,1)
        
        # get a new data bunch
        dbch_bin = get_databunch(train_df, 'binary', True)
        
        # create a learner and load the empty binary model
        learn = get_learner(dbch_bin, empty_binary, class_weights)
        learn.load('pretrained_binary')
        
        # train the model
        learn.freeze()
        learn.fit_one_cycle(f_epochs,lr)
        
        # add the test set
        learn.data.add_test(test_set)
        for npred in range(n_tta_aug):
            new_preds, _ = learn.get_preds(DatasetType.Test)
            preds[:,model_num,npred,:] = new_preds


# Train and test each model for an arbitrary number of epochs.

# In[ ]:


train_binary_models(3, 1e-2)


# In[ ]:


np.save('preds.npy',preds)


# # 5. Ensemble the results of the model

# In[ ]:


preds = np.load('preds.npy')


# In[ ]:


preds = torch.tensor(preds)


# In[ ]:


preds.shape


# With one line, we can get the ensembled predictions. Broken down from the inside out:
# - preds[:,:,:,1] = the probability of True digit
# - torch.mean(preds[:,:,:,1],dim=2 = the mean probability of True across aurmentations
# - torch.argmax(torch.mean(preds[:,:,:,1],dim=2),dim=1) = the digit classifier with highest (argmax) mean True across augmentations 

# In[ ]:


winner = torch.argmax(torch.mean(preds[:,:,:,1],dim=2),dim=1)


# In[ ]:


winner.size(),winner[:10]


# Another way to find the 'winner' of the models

# In[ ]:


# Find digit classifier (=argmax) with highest fraction of augmentation runs where the true class shows higher response
#torch.argmax(torch.mean(torch.argmax(preds,dim=-1).double(),dim=-1),dim=-1)


# Create a dataframe of the submission, and turn it into a csv file.

# In[ ]:


submission = pd.DataFrame({ 'id': np.arange(0,len(winner)),'label': winner })


# In[ ]:


submission.to_csv("submission.csv", index=False)


# fin.
