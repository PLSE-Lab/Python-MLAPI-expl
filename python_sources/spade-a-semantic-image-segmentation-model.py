#!/usr/bin/env python
# coding: utf-8

# # SPADE, a semantic image segmentation model
# 
# ## High-level ideas
# 
# This notebook contains my notes on the paper ["Semantic Image Synthesis with Spatially-Adaptive Normalization"](https://arxiv.org/pdf/1903.07291.pdf).
# 
# * SPADE is a network block architecture that is proposed in this very recent 2019 paper as a methodology for improving the performance of the generative GAN architecture. The paper proposes, in abstract, a spatially adaptive learned transformation on the inputs that is proposed to deal with "feature washout" caused by the use of normalization layers.
# * SPADE allows both photorealistic output based on an input image segmentation as well as style manipulation with style input:
# 
# ![](https://i.imgur.com/nocpGqI.png)
# 
# * Normalization layers (e.g. `BatchNorm2d`) as originally formatted are unconditional, in the sense that they are purely functional. **Conditional normalization layers** are a batch or new proposed normalization layers which condition their output based on outside data and/or parameters. Generally speaking, they work by normalizing to 0 mean and unit variance, then *denormalizing* in some way.
# * For style transfer tasks, a popular methodology is tying the denormalization to a learned affine transformation that is parameterized based on a separate input image (the style image). Note that this is radically different from the "backpropogation on the input image" approach used in the original 2015 style transfer paper.
# * SPADE makes this denormalization spatially sensitive.
# 
# 
# * SPADE normalization, not reproduced here, basically boils down to "conditional batch normalization which varies on a per-pixel basis". It is implemented as a two-layer convolutional neural network (compare with batch normalization, which is a simple functional layer).
# * Besides its use of SPADE, GuaGAN is a very close copy of a previous well-known best-in-class generative model, `pix2pixHD`.
# * SPADE improves model performance by dealing with a specific problem. Image segmentation masks tend to have many flat areas with equivalent mask values and pixel values, which get normalized to all-zeros (or close to it). This causes signal collapse. In the extreme case of a completely single-label image the output is all zeros. By applying *spatial* denormalization *before* the usual normalization operation, SPADE avoids this problem and enables more realistic output in low-entropy settings.
# 
# ## Implementation details
# 
# * Recall that the fundamental contribution of this paper is the SPADE normalization block. As a weight normalizer, SPADE manipulates the values of the weight matrix.
# * First, it applies sychronized batch normalization to the data. Batch normalization is the standard original form of normalization, e.g. applied to the whole batch at once. When training a GAN on a single machine, the input is traditionally batched with a size of 1, so there isn't any *actual* intra-batch "cross pollination" going on. When training a GAN on multiple machines, the batch size is basically the number of machines (here "synchronized" means that batches are forward propogated using an all-gather pattern). So the batch part of the normalization only has an effect if training is multi-instance.
# * Next, the segmentation map *for the whole image* is processed. It is first downsampled to match the filter in dimensionality, then passed through a 2d convolutional layer with a 3x3 kernel and 128 feature maps, A second 3x3 kernel convolutional layer with a user determined feature map count is stacked on top of this, and the output is combined with the batch norm result in a multiplicative manner.
# * Finally, a convolutional network based on the segmentation map with a shared first convolutional layer but a different topmost convolutional layer is combined with the output of the previous operation in an additive manner.
# * The result is then outputted.
# * Effectively you are having the network learn how to denormalize the normalization layer effectively as part of the training process. Diagram:
# ![](https://i.imgur.com/GGyp49F.png)
# 
# 
# * Next we'll look at the organization of the ResNet block. Recall that a ResNet block is one whose output is a combination of the input values (skip connection) and the output of a stack of layers applied to those input values. Recall that this is advantageous because it creates an easy way for the model to do nothing: just zero out the non-skip layers of the model.
# * The SPADE ResNet block is organized: SPADE w/ ReLU on input, 3x3 Conv w/ SPADE normalization w/ ReLU, 3x3 Conv (again with ReLU activation, presumably?). The residual connection is then made, and it is additive. This skip connection is only purely additive if the number of filters/channels in the non-skip output is the same as the number in the non-skip input; otherwise, an additional SPADE-ReLU-3x3 Conv transform must be learned and applied to the input, and *that* result is added to the output. Again the diagram:
# 
# ![](https://i.imgur.com/VReEsAW.png)
# 
# 
# * Here's the full generator architecture. Gaussian noise is the input to a linear layer that transforms 256 input values to 16384 output values, which is reshaped into a `(1024, 4, 4)` random matrix.
# 
#   A SPADE ResNet block operates on this input, using convolutional layers with 1024 nodes (every `Conv2d` layer is fed a `4x4` input). Nearest-neighbor upsampling is then applied to the output to double the size of the output, which is propogated to the next SPADEResBlk. There are seven such layers, so the output is `4*2**7 = 512` in dimensionality.
# 
#   The output layer is a 3x3 Conv2d with three filters (R, G, and B) and tanh activation.
# 
#   Again the diagram:
# 
# ![](https://i.imgur.com/rU3JZQI.png)
# 
# * Here's the discriminator architecture. This architecture is a close copy of the architecture used by `pix2pixHD`, which itself derived it from another model called PatchGAN. The segmentation is concatenated with the image pixels as an additional channel. This is passed through four convolutional layers featuring 64, 128, 256, and 512 filters each. Each convolutional layer is 4x4 (presumably the filters are divided equally between the layers?), stride length 2, Leakly ReLU activation, and instance normalized. The output layer is a 4x4 convolutional layer with a single node. I guess that's attached to a linear layer not shown in the diagram?
# 
# ![](https://i.imgur.com/oZWeVn2.png)
# 
# * The last major component is the image encoder. This is a stack of 3x3 stride 2 convolutional layers of increasing size again featuring instance normalization and leaky ReLU. The output of the final convolutional layer is flattened before being passed to two linear featurization layers with 256 features, one finding the mean values and one finding the variance values.
# 
# ![](https://i.imgur.com/ljvIgpr.png)
# 
# * By putting all of these components together we arrive at the overall model architecture:
# 
# ![](https://i.imgur.com/bHVzSwp.png)
# 
#   First the image is decomposed by the image encoder into a 256-arity mean-and-variance feature vector. These vectors are used to compute the noise inputted to the generator via a "reparameterization trick". Basically, the reparameterization trick is a way of making the output of a random node, e.g. a Gaussian normal, backpropagation-friendly. I have a notebook on this here: ["Notes on the reparameterization trick"](https://www.kaggle.com/residentmario/notes-on-the-reparameterization-trick).
#   
#   The reparameterization trick is used to make the noise input to the generator a function of the original image. The image is encoded (using a CNN) into a set of 256 learned pair-features. The values of these features are used as the mean value and the standard deviation value parameters of a set of Guassian random functions which generate random noise input to the model part of the generator.
#   
#   Recall that a simple generator may take purely random noise as input. Shaping the noise using image input (such that images/segmentation map pairs that are close to one another in the feature space are close to one another in generator noise input) has been shown to have performance benefits in practice however.
#   
#   The generator operates on this noise input&mdash;plus the segmentation mask at many steps, via the SPADE normalization blocks. The SPADE normalization blocks are the *only* place in the model architecture where the segmentation maps are used. In other words the denormalization part of SPADE normalization is the only part of the model that is segmentation-aware! OK, recall that GANs in their pure form are an unsupervised learning algorithm, and this is what makes this one into a supervised one.
#   
#   The discriminator is fed (and operates on) the concatenation of the generator output plus the segmentation mask, and it outputs a "real or fake" classification.
#   
#   
# * The VAE in the GuaGAN architecture is the combination of the encoder and generator sub-architectures. The encoder compresses the input image into a randomized feature representation, the generator reconstructs its own understanding from that random encoded feature representation. The encoder is specifically optimized, via the choice of loss function (KL divergence), for *style encoding*. All of the semantic information about image layout is found in and optimized for in the SPADE normalization blocks.
# * This is advantagenous because it makes the model implicity "multi-modal" (able to create multiple different images based on a single input image).
# * Another effect is that an information encoder segment, typically present in the GAN architectures that are designed for this task, is absent: 
#   
#   > With the SPADE, there is no need to feed the segmentation map to the first layer of the generator, since the learned modulation parameters have encoded enough information about the label layout. Therefore, we discard encoder part of the generator, which is commonly used in recent architectures.
# 
# * At test time, the encoder is repurposed for style transfer. The style image is fed to the encoder (seemingly in place of the original image input), from which the style random seed data is generated, from which a generated output image is drawn.
# 
# 
# * The paper also contains some comments on the composite learning function, which I will skip over for the moment.
