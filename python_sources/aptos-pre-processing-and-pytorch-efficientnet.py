#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# ## EfficientNet
# 
# Using the excellent pytorch-implementation from Luke Melas-Kyriazi:  
# https://github.com/lukemelas/EfficientNet-PyTorch
# 
# EfficientNet-paper:  
# https://arxiv.org/abs/1905.11946
# 
# ## Pre-processing and data augmentation
# 
# I use a pre-processing scheme similar to the one used by the [winner of the 2015 competition](http://blog.kaggle.com/2015/09/09/diabetic-retinopathy-winners-interview-1st-place-ben-graham).
# 
# However, I made two changes:
# * added a center crop, such that the shape of the eye is not altered by resizing (during pre-processing)
# * increased the zoom-level of training images to match the zoom-level of test images (during data-augmentation)
# 
# In hindsight, that way probably not necessary but it's still interesting to think it through.
# 
# **Motivation of avoiding shape-distorting resizing:**
# 
# The discrepancy between the local CV and the LB made me wary of a correlation between amount of black space around the eye and the diagnosis. (For instance, it would be conceivable that for more severe diagnoses, doctors take a closer look leading to a more zoomed-in image than for healthy eyes). If such a correlation existed in the training data but not in the public LB-data, that could explained the difference between validation- and LB-score.
# 
# Since the black space around the eyes is mostly to the left and right of rectangular images, resizing those to square images alters the shape of the eyeball. If that correlates with the diagnosis, the model could learn features based on the shape as nicely explained in [this kernel](https://www.kaggle.com/taindow/be-careful-what-you-train-on).
# 
# For this reason, before resizing, I crop the largest possible square from the center of the image. This square image is then resized. This way, the shape of the eye is preserved exactly as in the original image.
# 
# 
# **Zoom difference between training and test-images:**
#  
# A quick look at training- and test-images reveals that test-images tend to have a higher zoom which could also be a reason for the CV / LB - discrepancy. To remove this difference, I added a random-center-crop to the augmentation of the training-data but (but not the test-data). The random center-crop works just like the regular center-crop implemented in torchvision only randomly chooses the size of the cropped image within a user-specified range.
# 
# ## Runtime
# 
# About half an hour.

# In[ ]:


get_ipython().run_cell_magic('capture', '', '\nimport warnings \nwarnings.filterwarnings("ignore")\nimport os\nfrom os.path import join\nimport time\nfrom tqdm import tqdm\n\nimport numpy as np\nfrom numpy.random import choice\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\nfrom sklearn.metrics import cohen_kappa_score\n\nimport PIL\nfrom PIL import Image\nimport cv2\n\nimport ipywidgets as widgets\nfrom ipywidgets import interactive\n\nimport torch\nfrom torch import nn\nfrom torch.nn import functional as F\nfrom torch.utils.data import Dataset, DataLoader\nfrom torchvision import transforms, models as md\n\nimport sys\nsys.path.append(\'../input/efficientnet/efficientnet-pytorch/EfficientNet-PyTorch/\')\nfrom efficientnet_pytorch import EfficientNet')


# # Data pre-processing

# In[ ]:


DATA_DIR = '../input/aptos2019-blindness-detection'

train_dir = join(DATA_DIR, 'train_images')
label_df  = pd.read_csv(join(DATA_DIR, 'train.csv'))


def train_validation_split(df, val_fraction=0.1):
    val_ids  = np.random.choice(df.id_code, size=int(len(df) * val_fraction))
    val_df   = df.query('id_code     in @val_ids')
    train_df = df.query('id_code not in @val_ids')
    return train_df, val_df


train_df, val_df = train_validation_split(label_df)
print(train_df.shape, val_df.shape)
train_df.head()


# In[ ]:


def crop_image_from_gray(img,tol=7):
    """
    This function from:
    https://www.kaggle.com/ratthachat/aptos-updatedv14-preprocessing-ben-s-cropping
    """
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img


cv_to_pil = transforms.ToPILImage()

    
def center_crop(image: PIL.Image):
    """
    Only gets center square (of rectangular images) - no resizing
    => diffently sized square images
    """
    old_width, old_heigh = image.size
    new_size = min(old_width, old_heigh)
    
    margin_x = (old_width - new_size) // 2
    margin_y = (old_heigh - new_size) // 2
    
    left   = margin_x
    right  = margin_x + new_size
    top    = margin_y
    bottom = margin_y + new_size
    
    return image.crop( (left, top, right, bottom) )


def process_image_ratio_invariant(cv2_image, size=256, do_center_crop=True):
    
    image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    #image = cv2.resize(image, (size, size))  # this would distort eyeball shape
    
    if do_center_crop is False:
        return image
    
    # crop the largest possible square from the center
    pil_img = cv_to_pil(image)
    pil_img = center_crop(pil_img)
    image   = np.array(pil_img).copy()
    
    # now we have quadratic, but differently sized images
    # => resize without altering the shape of the eyeball
    image = cv2.resize(image, (size, size))
    
    # add gaussian blur with sigma proportional to new size:
    image = cv2.addWeighted (image, 4, cv2.GaussianBlur(image, (0, 0) , size/30) , -4 ,128)
        
    return cv_to_pil(image)


# # Load data + augmentation
# 
# ### Data augmentation
# 
# Both training-, and test-images are subjected to random horizontal flips and rotations. This inceases the diversity of the training data is hopefully decreases any systematic difference that might exist between training- and test-images.
# 
# The range of rotation between -20 and +20 degrees was chosen because the images show only a limited degree of rotation equivariance. (If an image is rotate by a few degrees, it looks just like another regular image. If it it's rotated by ~90 degrees it's doesn't look like a regular image any more).
# 
# ### In-memory dataset
# 
# The pre-processed data is stored in memory. While that takes a little while, it allows much faster training (less than a minute per epoch).

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n    \nclass Diabetic_Retionopathy_Data(Dataset):\n    \n    def __init__(self,\n                 image_dir: str,\n                 label_df: pd.DataFrame,\n                 train=True,\n                 transform=transforms.ToTensor(),\n                 sample_n=None,\n                 in_memory=False,\n                 write_images=False):\n        """\n        @ image_dir:   path to directory with images\n        @ label_df:    df with image id (str) and label (0/1) - only for labeled test-set\n        @ transforms:  image transformation; by default no transformation\n        @ sample_n:    if not None, only use that many observations\n        """\n        self.image_dir = image_dir\n        self.transform = transform\n        self.train     = train\n        self.in_memory = in_memory\n        \n        if sample_n:\n            label_df  = label_df.sample(n=min(sample_n, len(label_df)))\n            \n        ids            = set(label_df.id_code)\n        self.img_files = [f for f in os.listdir(image_dir) if f.split(\'.\')[0] in ids]\n        label_df.index = label_df.id_code\n        self.label_df  = label_df.drop(\'id_code\', axis=1)\n        \n        if in_memory:\n            \n            self.id2image = {}\n            for i, file_name in enumerate(self.img_files):\n                \n                if i and i % 500 == 0:\n                    print(f\'{i} / {len(self.img_files)}\')\n                \n                image = self._read_process_image(join(image_dir, file_name))\n                id_   = file_name.split(\'.\')[0]\n                self.id2image[id_] = image\n                \n                if write_images:\n                    image.save(file_name)\n                    \n        print(f\'Initialized datatset with {len(self.img_files)} images.\\n\')\n        \n    @staticmethod\n    def _read_process_image(file_path: str, size=256):\n        image = cv2.imread(file_path)        \n        return process_image_ratio_invariant(image, size=size)        \n\n    def __getitem__(self, idx):\n\n        file_name = self.img_files[idx]\n        id_ = file_name.split(\'.\')[0]\n        \n        if self.in_memory:\n            img = self.id2image[id_]\n        else:\n            img = self._read_process_image(join(self.image_dir, file_name))\n        \n        X   = self.transform(img)\n        \n        if self.train:\n            y = float(self.label_df.loc[id_].diagnosis)\n            return X, y, id_\n        else:\n            return X, id_\n    \n    def __len__(self):\n        return len(self.img_files)\n\n\nclass RandomCenterCrop(transforms.CenterCrop):\n    """\n    Crops the PIL Image at the center.\n    :param: min_size, max_size: range of crop-size randomly within [min_size, max_size]\n    """\n    def __init__(self, min_size: int, max_size: int):\n        self.min_size = min_size\n        self.max_size = max_size\n        \n    def __call__(self, img):\n        """\n        Args:\n            img (PIL Image): Image to be cropped.\n        Returns:\n            PIL Image: Cropped image.\n        """\n        size = np.random.randint(self.min_size, self.max_size + 1)\n        crop = transforms.CenterCrop( (size, size) )\n        return crop(img)\n\n    def __repr__(self):\n        return f\'{self.__class__.__name__}: (min-size={self.min_size}, max-size={self.max_size})\'\n\n\nbatchsize = 16\n\n# due to the large amount of data, random transformations might not be necessary...\ntrain_transform = transforms.Compose([\n    RandomCenterCrop(min_size=200, max_size=256),\n    transforms.Resize( (256, 256) ),\n    transforms.RandomHorizontalFlip(),\n    transforms.RandomRotation( (-20, 20) ),  \n    transforms.ToTensor(),\n    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])\n])\n\ntrain = Diabetic_Retionopathy_Data(train_dir,\n                                   train_df,\n                                   transform=train_transform,\n                                   in_memory=True,\n                                   write_images=False)\nval   = Diabetic_Retionopathy_Data(train_dir,\n                                   val_df,\n                                   transform=train_transform,\n                                   in_memory=True,\n                                   write_images=False)\n\ntrain_loader = DataLoader(train, batch_size=batchsize, num_workers=4, shuffle=True)\nval_loader   = DataLoader(val,   batch_size=batchsize, num_workers=3, shuffle=False)\n\nX, y, _ = next(iter(val_loader))\nprint(f\'batch-dimension:\\nX = {X.shape},\\ny = {y.shape}\')\nprint(f\'number of batches:\\ntrain: {len(train_loader)}\\nvalidation: {len(val_loader)}\')')


# ### Check pre-processing: compare raw vs. pre-processed images

# In[ ]:


def show_processed_images(image_dir, n=5, label_df=None, tf=None):
    
    sample_files = np.random.choice(os.listdir(image_dir), size=n)
    
    for file_name in sample_files:
        
        if label_df is not None:
            id_ = file_name.split('.')[0]
            diagnosis = label_df.query('id_code == @id_').diagnosis.item()
        else:
            diagnosis = 'unknown'
        
        image     = cv2.imread(join(image_dir, file_name))
        raw_image = cv_to_pil(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if tf is not None:
            processed_image = tf(join(image_dir, file_name))
        
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(10, 5)
        
        ax1.imshow(raw_image)
        if tf is not None:
            ax2.imshow(processed_image)
        ax1.set_title('raw')
        ax2.set_title('processed')
        
        fig.suptitle(f'diagnosis = {diagnosis}')            
        plt.show()
        
    
print('TRAINING DATA:')
show_processed_images(join(DATA_DIR, 'train_images'),
                      label_df=pd.concat([train_df, val_df]),
                      tf=train._read_process_image)
print('TEST DATA:')
show_processed_images(join(DATA_DIR, 'test_images'),
                      tf=train._read_process_image)


# # Define model

# In[ ]:


def count_parameters(model: nn.Module):
    return sum([np.prod(x.shape) for x in model.parameters()])


def print_lr_schedule(lr: float, decay: float, num_epochs=20):
    print('\nlearning-rate schedule:')
    for i in range(num_epochs):
        if i % 2 == 0:
            print(f'{i}\t{lr:.6f}')
        lr = lr* decay


net = EfficientNet.from_name('efficientnet-b0')
net.load_state_dict(torch.load('../input/efficientnet-pytorch/efficientnet-b0-08094119.pth'))

num_in_features = net._fc.in_features
net._fc = nn.Linear(num_in_features, 1)

print(f'number of parameters: {count_parameters(net)}')
net.train()
net.cuda()

loss_function = nn.MSELoss()
lr            = 0.0015
lr_decay      = 0.97

optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

print_lr_schedule(lr, lr_decay)


# # Train net

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nbest_epoch_score = np.inf\nprint('epoch\\ttrain-MSE\\tval-MSE\\tq-kappa\\tlr\\t\\ttime [min]')\nprint('------------------------------------------------------------------')\n\nfor epoch in range(25):\n    \n    start = time.time()\n    train_loss = []\n    \n    for i, (X, y, id_) in enumerate(train_loader):\n        \n        net.train()\n        optimizer.zero_grad()\n\n        out  = net(X.cuda())\n        loss = loss_function(out, y.float().cuda().view(-1, 1))    \n        \n        train_loss.append(loss.item())\n        \n        loss.backward()\n        optimizer.step()\n    \n    validation_loss = []\n    predictions     = np.array([])\n    truth           = np.array([])\n    \n    for  X, y, id_ in val_loader:\n\n        net.eval()\n\n        out  = net(X.cuda())\n        loss = loss_function(out, y.float().cuda().view(-1, 1))\n        \n        validation_loss.append(loss.item())\n        predictions = np.append(predictions, out.detach().cpu().numpy())\n        truth       = np.append(truth, y.detach().cpu().numpy().astype(int))\n\n    current_lr = optimizer.param_groups[0]['lr']\n    scheduler.step()\n    qk = cohen_kappa_score(predictions.round().astype(int), truth, weights='quadratic')\n    duration = (time.time() - start) / 60\n    print(f'{epoch}:\\t{np.mean(train_loss):.4f}\\t\\t{np.mean(validation_loss):.4f}\\t{qk:.4f}\\t{current_lr:.6f}\\t{duration:.2f}')\n    \n    if np.mean(validation_loss) < best_epoch_score:\n        torch.save(net.state_dict(), 'state_dict_best.pt')\n        best_epoch_score = np.mean(validation_loss)\n        best_epoch = epoch\n        \n\nprint(f'epoch with best validation-score: {best_epoch}')\n\nplt.hist(predictions, bins=5)\nplt.xlim(-1, 5)\nplt.title('distribution of predictions\\n(before rounding)')\nplt.show()\n\nplt.hist(train_df.diagnosis.values, bins=5)\nplt.xlim(-1, 5)\nplt.title('distribution of labels')\nplt.show()")


# # Make test-set predictions

# In[ ]:


test_dir = join(DATA_DIR, 'test_images')
test_df  = pd.read_csv(join(DATA_DIR, 'test.csv'))
test_df.head(3)


# ### Compare test- and training-images
# 
# Comparing test- and training-images side by side shows that, on average, test-images tend to have a higher zoom-level. The RandomCenterCrop applied to training-images only (as explained and implemented above) addresses this issue.

# In[ ]:


def sample_images(train_dir: str, test_dir: str, n=10):
    """
    Show n train- and test-images side by side.
    """
    train_files = choice(os.listdir(train_dir), size=n)
    test_files  = choice(os.listdir(test_dir),  size=n)
    images      = []
    
    for train_f, test_f in zip(train_files, test_files):
        train_img = Image.open(join(train_dir, train_f))
        test_img  = Image.open(join(test_dir,  test_f)) 
        images.append( (train_img, test_img) )
        
    def show_image(i):
        train_image, test_image = images[i]
        
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(20, 10)
        ax1.imshow(train_image)
        ax2.imshow(test_image)
        ax1.set_title('train')
        ax2.set_title('test')
        plt.show()
        
    return interactive(show_image, i=(0, n-1))
        
sample_images(train_dir=train_dir,
              test_dir=test_dir,
              n=10)


# ** Test-time augmentation **
# 
# The test-data is augmented in exactly the same way as the training data except for the RandomCenterCrop.

# In[ ]:


test_transform = transforms.Compose([
    transforms.Resize( (256, 256) ),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation( (-20, 20) ),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_ds = Diabetic_Retionopathy_Data(test_dir,
                                     test_df,
                                     transform=test_transform,
                                     train=False)

test_loader = DataLoader(test_ds, batch_size=batchsize)

X, _ = next(iter(test_loader))
print(f'batch-dimension:\nX = {X.shape}')


# **Make predictions**

# In[ ]:


net.load_state_dict(torch.load('state_dict_best.pt'))
net.eval()
net.cuda()
id2prediction = {}

for i, (X, id_) in enumerate(test_loader):
    out           = net(X.cuda())
    preds         = out.detach().cpu().numpy().ravel()
    id2prediction = {**id2prediction, **dict(zip( id_, preds.round().astype(int).tolist() ))}


# In[ ]:


submission_df = pd.read_csv(join(DATA_DIR, 'sample_submission.csv'))
submission_df.diagnosis = submission_df.id_code.map(id2prediction)

# limit predictions to interval [0, 4] !!
submission_df.diagnosis = submission_df.diagnosis.map(lambda p: max(p, 0))
submission_df.diagnosis = submission_df.diagnosis.map(lambda p: min(p, 4))

submission_df.to_csv('submission.csv', index=False)
display(submission_df.head())


# **Check the predictions**
# 
# Note that the test-data seemst to have a different diagnosis-distribution than the training data. Or maybe the predictions all very off. Or maybe both :-)

# In[ ]:


assert len(submission_df) == len(submission_df.dropna())
assert set(submission_df.diagnosis) == {0, 1, 2, 3, 4}
submission_df.diagnosis.hist()
plt.title('distribution of test-set predictions');

