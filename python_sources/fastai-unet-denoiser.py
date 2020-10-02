#!/usr/bin/env python
# coding: utf-8

# # Denoising Document Backgrounds with Fastai Unet
# 
# ## Purpose
# 
# The purpose of the notebook is to demonstrate the Fastai Library to perform background removal in images using the Unet.  Most of this code comes from his [Super Resolution Notebook](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson7-superres.ipynb) which is exlained in this [fast.ai course video](https://www.youtube.com/watch?time_continue=4745&v=9spwoDYwW_I). This was adapted from another [unfinished notebook](https://www.kaggle.com/hahmed747/background-removal-using-fastai-unet-learner).
# 
# ## Opportunities
# 
# There are a few reasons why this setup as-is won't perform that well:
# * The pretrained vgg model is using photos, not text
# * The pretrained vgg model is using color images, not black and white
# * Insufficient experimentation of hyperparameters
# * The values have to get clipped to 0,1; seems like a bad sign that they are out of that range

# # Setup

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import pathlib
import fastai
from fastai import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.utils.mem import *

from torchvision.models import vgg16_bn
from subprocess import check_output


# # Data Import

# In[ ]:


input_path = Path('/kaggle/input/denoising-dirty-documents')
items = list(input_path.glob("*.zip"))
print([x for x in items])


# In[ ]:


import zipfile

for item in items:
    print(item)
    with zipfile.ZipFile(str(item), "r") as z:
        z.extractall(".")


# In[ ]:


bs, size = 4, 128
arch = models.resnet34
path_train = Path("train")
path_train_cleaned = Path("train_cleaned")
path_test = Path("test")
path_submission = Path("submission")


# # Data Processing

# In[ ]:


src = ImageImageList.from_folder(path_train).split_by_rand_pct(0.2, seed=42)


# In[ ]:


def get_data(src, bs, size):
    data = (
        src.label_from_func(lambda x: path_train_cleaned / x.name)
           .transform(get_transforms(max_zoom=2.), size=size, tfm_y=True)
           .databunch(bs=bs)       
           .normalize(imagenet_stats, do_y=True)
    )
    data.c = 3
    return data


# In[ ]:


data = get_data(src, bs, size)


# In[ ]:


# Show some validation examples
data.show_batch(ds_type=DatasetType.Valid, rows=2, figsize=(5, 5), title="Some image")


# ## Feature Loss

# In[ ]:


t = data.valid_ds[0][1].data
t = torch.stack([t,t])


# In[ ]:


def gram_matrix(x):
    n,c,h,w = x.size()
    x = x.view(n, c, -1)
    return (x @ x.transpose(1,2))/(c*h*w)


# In[ ]:


base_loss = F.l1_loss


# In[ ]:


vgg_m = vgg16_bn(True).features.cuda().eval()
requires_grad(vgg_m, False)


# In[ ]:


# Show the layers before all pooling layers, which turn out to be ReLU activations.
# This is right before the grid size changes in the VGG model, which we are using
# for feature generation.
blocks = [i-1 for i,o in enumerate(children(vgg_m)) if isinstance(o,nn.MaxPool2d)]
blocks, [vgg_m[i] for i in blocks]


# In[ ]:


class FeatureLoss(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts):
        """ m_feat is the pretrained model """
        super().__init__()
        self.m_feat = m_feat
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        # hooking grabs intermediate layers
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel',] + [f'feat_{i}' for i in range(len(layer_ids))
              ] + [f'gram_{i}' for i in range(len(layer_ids))]

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]
    
    def forward(self, input, target):
        # get features for target
        out_feat = self.make_features(target, clone=True)
        # features for input
        in_feat = self.make_features(input)
        # calc l1 pixel loss
        self.feat_losses = [base_loss(input,target)]
        # get l1 loss from all the block activations
        self.feat_losses += [base_loss(f_in, f_out)*w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.feat_losses += [base_loss(gram_matrix(f_in), gram_matrix(f_out))*w**2 * 5e3
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        # so we can show all the layer loss amounts
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)
    
    def __del__(self): self.hooks.remove()


# In[ ]:


feat_loss = FeatureLoss(vgg_m, blocks[2:5], [5,15,2])


# ## Train

# In[ ]:


wd = 1e-3
learn = unet_learner(data, arch, wd=wd, loss_func=feat_loss, callback_fns=LossMetrics,
                     blur=True, norm_type=NormType.Weight)
gc.collect();


# In[ ]:


learn.model_dir = Path('models').absolute()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


print(f"Validation set size: {len(data.valid_ds.items)}")


# In[ ]:


lr = 1e-3


# In[ ]:


def do_fit(save_name, lrs=slice(lr), pct_start=0.9):
    learn.fit_one_cycle(10, lrs, pct_start=pct_start)
    learn.save(save_name)
    learn.show_results(rows=1, imgsize=5)


# In[ ]:


do_fit('1a', slice(lr*10))


# In[ ]:


learn.unfreeze()


# In[ ]:


do_fit('1b', slice(1e-5, lr))


# In[ ]:


# Increase resolution of the images.
data = get_data(src, 12, size*2)


# In[ ]:


learn.data = data
learn.freeze()
gc.collect()


# In[ ]:


learn.load('1b');


# In[ ]:


do_fit('2a')


# In[ ]:


learn.unfreeze()


# In[ ]:


do_fit('2b', slice(1e-6,1e-4), pct_start=0.3)

# save entire configuration
#learn.export(file = model_path)


# ## Test on a Validation Example

# In[ ]:


fn = data.valid_ds.x.items[10]; fn


# In[ ]:


img = open_image(fn); img.shape


# In[ ]:


p,img_pred,b = learn.predict(img)
show_image(img, figsize=(8,5), interpolation='nearest');


# In[ ]:


Image(img_pred).show(figsize=(8,5))


# ## Processing Test Set
# 
# We could add the test set to the original DataBunch with `add_test_folder`, 
# ([as discussed](https://forums.fast.ai/t/beginner-question-how-to-predict-on-test-set/31179)) however this encounters a number of problems. Namely, we want to process
# our entire test images without cropping transformations, but they have difference
# sizes which means a DataBunch (minibatches) can't be used.  
# 
# There is an option to assign a test set when calling `load_learn` but this
# had problems as well.
# 
# We go with the simple inefficient route of processing them one by one.  This does require a [beautiful hack](https://forums.fast.ai/t/segmentation-mask-prediction-on-different-input-image-sizes/44389), however which is the first line below.

# In[ ]:


# Turn off resizing transformations for inference time.
# https://forums.fast.ai/t/segmentation-mask-prediction-on-different-input-image-sizes/44389
learn.data.single_ds.tfmargs['size'] = None


# In[ ]:


test_images = ImageImageList.from_folder(path_test)
print(test_images)


# In[ ]:


img = test_images[0]
img.show()
img.shape


# In[ ]:


p, img_pred, b = learn.predict(img)


# In[ ]:


def rgb2gray(_img):
    """ Convert from 3 channels to 1 channel """
    from skimage.color import rgb2gray as _rgb2gray

    # Rotate channels dimension to the end, per skimage's expectations
    _img_pred_np = _img.permute(1, 2, 0).numpy()
    _img_pred_2d = Tensor(_rgb2gray(_img_pred_np))
    # Add the channel dimension back
    _img_pred = _img_pred_2d.unsqueeze(0)
    return _img_pred
  
Image(rgb2gray(img_pred)).show(figsize=(8,5))


# In[ ]:


def write_image(fname, _img_tensor):
    _img_tensor = (_img_tensor * 255).to(dtype=torch.uint8)
    imwrite(path_submission/fname, _img_tensor.squeeze().numpy())


# In[ ]:


import csv
from imageio import imread, imwrite

path_submission.mkdir(exist_ok=True)

with Path('submission.csv').open('w', encoding='utf-8', newline='') as outf:
    writer = csv.writer(outf)
    writer.writerow(('id', 'value'))
    for i, fname in enumerate(path_test.glob("*.png")):
        img = open_image(fname)
        img_id = int(fname.name[:-4])
        print('Processing: {} '.format(img_id))
        # Predictions
        p, img_pred, b = learn.predict(img)
        # Convert to grayscale and clip out of range values.
        img_2d = rgb2gray(img_pred).clamp(0, 1)
        # Write an image file for examination.
        write_image(fname.name, img_2d)
        # Write to the submission file, in a very inefficient way.
        for r in range(img_2d.shape[1]):
            for c in range(img_2d.shape[2]):
                id = str(img_id)+'_'+str(r + 1)+'_'+str(c + 1)
                val = img_2d[0, r, c].item()
                writer.writerow((id, val))


# In[ ]:




