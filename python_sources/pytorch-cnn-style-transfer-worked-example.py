#!/usr/bin/env python
# coding: utf-8

# # PyTorch GAN Style Transfer Worked Example
# 
# This notebook is a run of the model presented in the PyTorch tutorial section [Neural Transfer Using PyTorch](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html).
# 
# First we reproduce this model, then we dissect it to understand how it works.
# 
# ## High-level description
# 
# This demo showcases building a generative adversarial network, or GAN, for performing style transfer.

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# In[ ]:


# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


style_img = image_loader("../input/two-images/picasso.jpg")
content_img = image_loader("../input/two-images/dancing.jpg")

assert style_img.size() == content_img.size(),     "we need to import style and content images of the same size"


# In[ ]:


class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


# In[ ]:


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


# In[ ]:


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


# In[ ]:


cnn = models.vgg19(pretrained=True).features.to(device).eval()


# In[ ]:


cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


# In[ ]:


content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    # At runtime, CNN is a pretrained VGG19 CNN network.
    cnn = copy.deepcopy(cnn)

    # Ship the Normalization module to CUDA. The mean and standard deviation values
    # used for normalization on the originating training dataset are well-known
    # values, specified manually as `cnn_normalization_mean` and `cnn_normalization_std`
    # in the code block above.
    #
    # In this demo, the Normalization module is applied just-in-time as a functional
    # layer of the network (first layer).
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    model = nn.Sequential(normalization)

    content_losses = []
    style_losses = []

    # This next code block does the bulk of the work building the new model.
    #
    # Recall that .children() iterates over model layers that are direct children
    # of the model, and not children-of-children, in case the model contains
    # submodel (e.g. layers that are themselves nn.Module modules) definitions.
    #
    # Recall that cnn is the VGG19 pretrain convolutional model.
    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        # The first layer simply puts names to things and replaces ReLU inplace
        # (which is optimized) with ReLU reallocated. This is a small optimization
        # being removed, and hence a small performance penalty, necessitated by
        # ContentLoss and StyleLoss not working well when inplace=True.
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        # add_module is a setter that is pretty much a setattr equivalent, used for
        # registering the layer with PyTorch.
        model.add_module(name, layer)

        # The next pair of boolean checks ship the layer from cnn to the output model, with
        # a content-dependent adjustment.
        #
        # Content loss is defined on just one layer in the network, conv_4. Style loss is
        # defined on every convolutional layer in the network, conv_1...conv_4. conv_4 has
        # both style loss and content loss defined on it.
        #
        # In both cases, the output vector of the model thus far is called on with .detach
        # applied to it. .detach is call which tells PyTorch that the given layer output
        # is not subject to gradient tracking and backpropogation, e.g. it is a functional
        # or frozen layer. Basically the same thing as setting .no_grad to True.
        #
        # The loss of choice (Content or Style) is then applied to the layer. The loss is
        # calculated by taking the difference between the corresponding target image as it
        # appears after passing thus far into the network (model(content_img).detach()) and
        # the current input image (at forward time).
        if name in content_layers:
            # model(content_img) executes *the entire model we have built so far* on the
            # input image. detach keeps the operations involved out of the computational
            # graph, e.g. no backpropogation.
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # Trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


# In[ ]:


input_img = content_img.clone()


# In[ ]:


# The convolutional layers in the resulting model are the trainable ones.
#
# The loss layers are pass-through. They propogate their input forward,
# whilst calculating a loss against target. The loss against target is
# what is used to perform backpropogation.
#
# This is an unusual organization, and means that ContentLoss et al are
# not true loss modules. Wouldn't it make more sense to run them at
# forward time? E.g.:
# https://discuss.pytorch.org/t/how-to-calculate-loss-related-to-intermediate-feed-forward-results/10278/2?u=residentmario
temp_model, _, _ = get_style_model_and_losses(
    cnn, cnn_normalization_mean, cnn_normalization_std,
    style_img, content_img
)
temp_model


# How were the layers at which the losses were attached chosen? This is unknown to me.
# 
# As for the definitions of the losses themselves, the original document has good descriptions:
# * https://pytorch.org/tutorials/advanced/neural_style_tutorial.html#content-loss
# * https://pytorch.org/tutorials/advanced/neural_style_tutorial.html#style-loss

# In[ ]:


[n for (n, pg) in list(temp_model.named_parameters()) if pg.requires_grad]


# LBFGS is the optimizer suggested by the original author of this implementation [here](https://discuss.pytorch.org/t/pytorch-tutorial-for-neural-transfert-of-artistic-style/336).

# In[ ]:


# The optimizer operates on *the input image* instead of on any intermediate weights.
# We need to run requires_grad_() here to set requires_grad=True.
#
# Since the optimizer is only being run on the input image, no training occurs on
# intermediate weights.
def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


# In[ ]:


def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=1000,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model.')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    # TODO: refactor out this [0] list nonsense
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # The input image can stray out of the (0, 1) range as a
            # result of our operations, so we have to clamp it so that
            # the result makes sense.
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            # Construct the cumulative loss function and backpropogate.
            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss
            style_score *= style_weight
            content_score *= content_weight
            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img


# In[ ]:


output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img)


# In[ ]:


unloader = transforms.ToPILImage()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated

imshow(output)

