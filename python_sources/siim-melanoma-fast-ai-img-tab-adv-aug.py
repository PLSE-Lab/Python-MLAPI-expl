#!/usr/bin/env python
# coding: utf-8

# ### **Thanks to nroman for the advanced hair & microscope view augmentation**

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


pip install image_tabular


# In[ ]:


from fastai.vision import *
from fastai.tabular import *
from image_tabular.core import *
from image_tabular.dataset import *
from image_tabular.model import *
from image_tabular.metric import *

# use gpu by default if available
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# In[ ]:


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")


# In[ ]:


data_path = Path("../input/resize-jpg-siimisic-melanoma-classification/640x640")
df_path = Path("../input/siim-isic-melanoma-classification")


# In[ ]:


train_df = pd.read_csv(df_path/"train.csv")
test_df = pd.read_csv(df_path/"test.csv")

print(len(train_df), len(test_df))


# In[ ]:


train_df.head()


# In[ ]:


# extremely unbalanced dataset, most of the images are benign
train_df["target"].value_counts(normalize=True)


# ## Image data

# In[ ]:


size = 256


# In[ ]:


cutout_frac = 0.25
p_cutout = 0.75
cutout_sz = round(size*cutout_frac)
cutout_tfm = cutout(n_holes=(1,1), length=(cutout_sz, cutout_sz), p=p_cutout)


# In[ ]:


#hair addition 
import cv2
from glob import glob

n_max=16     # the maximum number of hairs to augment
im_size=size  # all images are resized to this size

hair_images=glob('/kaggle/input/melanoma-hairs/*.png')

def _hair_aug_ocv(input_img):
    img1 = image2np(input_img)*255 # convert to numpy array in range 0-255
    img1 = img1.astype(np.uint8) # convert to int
#     print(img1)
    
    img=img1.copy()
    # Randomly choose the number of hairs to augment (up to n_max)
    n_hairs = random.randint(0, n_max)

    # If the number of hairs is zero then do nothing
    if not n_hairs:
        x = pil2tensor(img, dtype=np.float32)
        x.div_(255)
        return x

    # The image height and width (ignore the number of color channels)
    im_height, im_width, _ = img.shape 

    for _ in range(n_hairs):

        # Read a random hair image
        hair = cv2.imread(random.choice(hair_images)) 
        
        # Rescale the hair image to the right size (256 -- original size)
        scale=im_size/256
        hair = cv2.resize(hair, (int(scale*hair.shape[1]), int(scale*hair.shape[0])), 
                          interpolation=cv2.INTER_AREA)       

        # Flip it
        # flipcode = 0: flip vertically
        # flipcode > 0: flip horizontally
        # flipcode < 0: flip vertically and horizontally    
        hair = cv2.flip(hair, flipCode=random.choice([-1, 0, 1]))

        # Rotate it
        hair = cv2.rotate(hair, rotateCode=random.choice([cv2.ROTATE_90_CLOCKWISE,
                                                          cv2.ROTATE_90_COUNTERCLOCKWISE,
                                                          cv2.ROTATE_180
                                                         ])
                         )
        
        
        # The hair image height and width (ignore the number of color channels)
        h_height, h_width, _ = hair.shape

        # The top left coord's of the region of interest (roi)  
        # where the augmentation will be performed
        roi_h0 = random.randint(0, im_height - h_height)
        roi_w0 = random.randint(0, im_width - h_width)

        # The region of interest
        roi = img[roi_h0:(roi_h0 + h_height), roi_w0:(roi_w0 + h_width)]

        # Convert the hair image to grayscale
        hair2gray = cv2.cvtColor(hair, cv2.COLOR_BGR2GRAY)

        # If the pixel value is smaller than the threshold (10), it is set to 0 (black), 
        # otherwise it is set to a maximum value (255, white).
        # ret -- the list of thresholds (10 in this case)
        # mask -- the thresholded image
        # The original image must be a grayscale image
        # https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html
        ret, mask = cv2.threshold(hair2gray, 10, 255, cv2.THRESH_BINARY)

        # Invert the mask
        mask_inv = cv2.bitwise_not(mask)

        # Bitwise AND won't be performed where mask=0
        img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        hair_fg = cv2.bitwise_and(hair, hair, mask=mask)
        # Fixing colors
        hair_fg = cv2.cvtColor(hair_fg, cv2.COLOR_BGR2RGB)
        # Overlapping the image with the hair in the region of interest
        dst = cv2.add(img_bg, hair_fg)
        # Inserting the result in the original image
        img[roi_h0:roi_h0 + h_height, roi_w0:roi_w0 + h_width] = dst
        
    x = pil2tensor(img, dtype=np.float32)
    x.div_(255)
    return x 


# In[ ]:


# microscope view
p_micro = 0.3
def _microscope(input_img):
    img1 = image2np(input_img)*255 # convert to numpy array in range 0-255
    img1 = img1.astype(np.uint8) # convert to int
#     print(img1)
    
    img=img1.copy()

    if random.random() < p_micro:
        circle = cv2.circle((np.ones(img.shape) * 255).astype(np.uint8),
                        (img.shape[0]//2, img.shape[1]//2),
                        random.randint(img.shape[0]//2 - 3, img.shape[0]//2 + 15),
                        (0, 0, 0),
                        -1)

        mask = circle - 255
        img = np.multiply(img, mask)
    x = pil2tensor(img, dtype=np.float32)
    x.div_(255)
    return x
microscope = TfmPixel(_microscope)


# In[ ]:


hair_aug_ocv = TfmPixel(_hair_aug_ocv)
tfms = get_transforms(flip_vert=True, xtra_tfms = [cutout_tfm, hair_aug_ocv(),microscope()])


# In[ ]:


# idx for validation, shared by image and tabular data
val_idx = get_valid_index(train_df)
len(val_idx)


# In[ ]:


# load image data using train_df and prepare fastai LabelLists
image_data = (ImageList.from_df(train_df, path=data_path, cols="image_name",
                               folder="train", suffix=".jpg")
              .split_by_idx(val_idx)
              .label_from_df(cols="target")
              .transform(tfms, size=size))

# add test data so that we can make predictions
test_image_data = ImageList.from_df(test_df, path=data_path, cols="image_name",
                                    folder="test", suffix=".jpg")

image_data.add_test(test_image_data)


# In[ ]:


# show one example image
# print(image_data.train[0][1])
image_data.train[6][0]


# ## Tabular data

# In[ ]:


dep_var = 'target'
cat_names = ['sex', 'anatom_site_general_challenge']
cont_names = ['age_approx']
procs = [FillMissing, Categorify, Normalize]


# In[ ]:


tab_data = (TabularList.from_df(train_df, path=data_path, cat_names=cat_names, cont_names=cont_names, procs=procs)
                           .split_by_idx(val_idx)
                           .label_from_df(cols=dep_var))

# add test
tab_data.add_test(TabularList.from_df(test_df, cat_names=cat_names, cont_names=cont_names,
                                      processor = tab_data.train.x.processor))


# In[ ]:


# one example
tab_data.train[0]


# ## Integrate image and tabular data

# In[ ]:


integrate_train, integrate_valid, integrate_test = get_imagetabdatasets(image_data, tab_data)


# In[ ]:


# package train, valid, and test datasets into a fastai databunch
bs = 32

db = DataBunch.create(integrate_train, integrate_valid, integrate_test,
                      path=data_path, bs=bs)#.normalize(imagenet_stats)
db


# In[ ]:


# image normalization with imagenet_stats
db.norm, db.denorm = normalize_funcs_image_tab(*imagenet_stats)
db.add_tfm(db.norm)


# In[ ]:


# check the shape of one batch
x, y = next(iter(db.train_dl))
len(x)


# In[ ]:


# images
x[0].shape


# In[ ]:


# categorical and continuous tabular data 
x[1][0].shape, x[1][1].shape


# In[ ]:


# targets
y.shape


# ## Model that trains on image and tabular data simultaneously

# In[ ]:


# cnn model for images, use Resnet50 as an example
cnn_arch = models.resnet50

# cnn_out_sz is the output size of the cnn model that will be concatenated with tabular model output
cnn_out_sz = 256

# use fastai functions to get a cnn model
image_data_db = image_data.databunch()
image_data_db.c = cnn_out_sz
cnn_learn = cnn_learner(image_data_db, cnn_arch, ps=0.2)
cnn_model = cnn_learn.model


# In[ ]:


# get embedding sizes of categorical data
emb_szs = tab_data.train.get_emb_szs()

# output size of the tabular model that will be concatenated with cnn model output
tab_out_sz = 12

# use fastai functions to get a tabular model
tabular_model = TabularModel(emb_szs, len(cont_names), out_sz=tab_out_sz, layers=[12], ps=0.1)
tabular_model


# In[ ]:


# get an integrated model that combines the two components and concatenate their outputs
# which will pass through additional fully connected layers
integrate_model = CNNTabularModel(cnn_model,
                                  tabular_model,
                                  layers = [cnn_out_sz + tab_out_sz, 32],
                                  ps=0.2,
                                  out_sz=2).cuda()


# In[ ]:


# check model output dimension, should be (bs, 2)
integrate_model(*x).shape


# In[ ]:


# adjust loss function weight because the dataset is extremely unbalanced
weights = [1/(1-train_df["target"].mean()), 1/train_df["target"].mean()]
loss_func = CrossEntropyFlat(weight=torch.FloatTensor(weights).cuda())#.mixup()


# In[ ]:


# package everything in a fastai learner, add auc roc score as a metric
learn = Learner(db, integrate_model, metrics=[accuracy, ROCAUC()], loss_func=loss_func)


# In[ ]:


# organize layer groups in order to use differential learning rates provided by fastai
# the first two layer groups are earlier layers of resnet
# the last layer group consists of the fully connected layers of cnn model, tabular model,
# and final fully connected layers for the concatenated data
learn.layer_groups = [nn.Sequential(*flatten_model(cnn_learn.layer_groups[0][0])),
                      nn.Sequential(*flatten_model(cnn_learn.layer_groups[0][1])),
                      nn.Sequential(*(flatten_model(cnn_learn.layer_groups[0][2]) +
                                      flatten_model(integrate_model.tabular_model) +
                                      flatten_model(integrate_model.layers)))]


# ## Training

# In[ ]:


# find learning rate to train the last layer group first 
learn.model_dir='/kaggle/working/'
learn.model.cuda()
learn.lr_find()
learn.recorder.plot()


# In[ ]:


# train
learn.fit_one_cycle(4, 1e-2)


# In[ ]:


# unfreeze all layer groups to train the entire model using differential learning rates
learn.unfreeze()
learn.fit_one_cycle(8, slice(1e-6, 1e-4))


# ## Prediction

# In[ ]:


val_preds, val_labels = learn.get_preds(DatasetType.Test)
print_metrics(val_preds, val_labels)


# In[ ]:


# # make predictions for the test set
# preds, y = learn.get_preds(DatasetType.Test)


# In[ ]:


# submit predictions to kaggle
submit = pd.read_csv(data_path/"sample_submission.csv")
submit["target"] = preds[:, 1]
submit.to_csv("/kaggle/working/image_tab.csv", index=False)


# In[ ]:




