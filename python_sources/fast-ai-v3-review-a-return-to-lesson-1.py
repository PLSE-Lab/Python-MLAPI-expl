#!/usr/bin/env python
# coding: utf-8

# # Lesson 1 Revisited - Fastai review

# **This notebook revisits the first lesson in the Fastai 2019 Part 1 course.** In this lesson, the Fastai team showed how we could create a remarkably good image classifier for pet breeds using only a few lines of code. While this was really cool, I thought it would be interesting to revisit the lesson and see some of the things that can go wrong when you're developing a model, rather than having everything work right away! Perhaps, as you start developing your own models, this will help you recognize some of the things that happen ...
# 
# Lesson goal - a quick review of the overall Fastai process for developing a model, including:
# - recognizing underfitting and overfitting
# - fastai innovations including `lr_find` and `fit_one_cycle`
# - transfer learning and fine tuning
# - how to adjust different types of regularization
# 

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai import *
from fastai.vision import *


# In[ ]:


bs = 64
# bs = 16   # uncomment this line if you run out of memory even after clicking Kernel->Restart


# ## Setting up the data

# We'll again use the [Oxford-IIIT Pet Dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/) by [O. M. Parkhi et al., 2012](http://www.robots.ox.ac.uk/~vgg/publications/2012/parkhi12a/parkhi12a.pdf) which features 12 cat breeds and 25 dogs breeds. However, we'll cut the dataset down to 11 breeds so we can run through the notebook faster.

# In[ ]:


path = untar_data(URLs.PETS); path


# In[ ]:


path_img = path/'images'
fnames = get_image_files(path_img)
fnames[:5]


# In[ ]:


fnames = sorted(fname for fname in fnames if fname.name[0].lower() in 'abc')
len(fnames)


# In[ ]:


np.random.seed(2)
pat = re.compile(r'/([^/]+)_\d+.jpg$')
data = ImageDataBunch.from_name_re(path_img, fnames, pat,
                                   ds_tfms=get_transforms(),
                                   size=224, bs=bs, num_workers=0
                                  ).normalize(imagenet_stats)
data


# In[ ]:


data.show_batch(rows=3, figsize=(7,6))


# In[ ]:


print(data.classes)
len(data.classes),data.c


# ## A bad model

# Let's define a simple linear model on all the input pixel values. This won't do well, since pet breed recognition isn't a linear function!

# In[ ]:


def badLearner() :    
    model = nn.Sequential(
        Flatten(),
        nn.Linear(224*224*3,data.c)
    )
    return Learner(data,model,metrics=accuracy)


# In[ ]:


def randomSeedForTraining(seed) :
    "This is to make the training demos below repeatable."
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

randomSeedForTraining(3141);
learn = badLearner()


# In[ ]:


print(learn.summary())


# ## Step 0: Loss values ridiculously big or blow up

# In[ ]:


learn.fit(3,1.0)


# Sometimes you'll see values that stay ridiculously high for the training loss column (this shows the average value of the loss function on the training set at each epoch). In extreme cases, you may see the loss values "blow up" or even wind up with mysterious printouts like `NaN`, which means the values can't even be represented anymore.
# 
# This shouldn't happen! Mathematically, the training process essentially consists of repeatedly changing your model's weights as follows:  
# `weights -= learning_rate*gradient`  
# where the gradient is a vector the same size as the weights that says which direction the loss function goes as we slightly change each of the weights. So, as long as your learning rate is small enough, the loss on your training set should go down pretty consistently until you get close to a minimum value. (One exception is that when using `fit_one_cycle` you may see the training loss go up for a while at the beginning of the training - this may actually help training by exploring the weight space more thoroughly).
# 
# If you do see ridiculously big values or a blow up, the most likely reason is that your learning rate is too high, but you also might have bugs in how your model is coded.
# 
# Luckily, Fastai provides the `lr_find()` function to help you pick a good learning rate!

# In[ ]:


randomSeedForTraining(314);
learn = badLearner()
learn.lr_find(); learn.recorder.plot()


# ## Step 1: Underfitting

# So, now that we have a good learning rate for our bad model, let's train it!

# In[ ]:


learn.fit(5,3e-5)


# What we see now is that the training loss does go down for a while and then approaches a minimum, but that minimum is still bad! This situation (not even doing well on your training set) is called **underfitting**. You can also start to look at the other two columns, which give you stats on the validation set after each epoch - these are examples that your model wasn't trained on - and see the performance on these is bad as well.
# 
# One explanation for underfitting is that your model isn't powerful enough, but there are a couple of other possibilities. Why else might you get something that looks like underfitting?
# 
# - Maybe what you're trying to predict isn't that predictable! If you tried to predict the results of a coin flip based on the coin flipper's age, weight, and so forth, you'd probably wind up with something that looks like underfitting.
# - Maybe you have a bug in constructing the dataset! Suppose you randomly mixed up the labels before applying them to the training data. The result would likely look like underfitting. Or, suppose you have a bug in your image preprocessing code and accidentally clip out much of the data relevant for prediction. This is why we look at the data! See the code above that calls `show_batch`.

# One confusing thing - you may see something to the effect that linear training provably produces an optimal result,
# while optimally training a neural network is intractable (NP-complete) even for very shallow architectures.
# What's going on here?
# 
# "A horse than can count to ten is a remarkable horse, not a remarkable mathematician." - Samuel Johnson?
# 
# Our model is a remarkable linear model (close to the best possible), but not a remarkable predictor!
# Our function isn't close to linear, so even the optimal (best possible) linear model will be a lousy predictor.
# On the other hand, when we train a neural network, while we may not get an optimal result for our neural network architecture,
# we will in practice do much better than the linear model.

# ## Step 2: Overfitting

# Since we're underfitting, let's try using a more powerful model! We'll use a powerful neural network model called Resnet34, but unlike in the original lesson we'll start from scratch (no pretraining).

# In[ ]:


randomSeedForTraining(3141);
learn = create_cnn(data, models.resnet34, metrics=accuracy, pretrained=False)


# In[ ]:


learn.lr_find(); learn.recorder.plot()


# In[ ]:


learn.fit(8,5e-3)


# Now we see a different pattern. The loss values for the training set (the leftmost column) go down steadily, but those for the validation set don't seem to progress over a considerable number of epochs, or may even get worse. This indicates **overfitting**. Intuitively, overfitting can happen when our model makes its classification function too complicated, for example by using some irrelevant feature that randomly happens to be associated with a category in our training set. This is likely to occur when we use an architecture that has lots of capacity for complication, and we don't have enough training data to make those incorrect associations unlikely.
# 
# Unfortunately, many ML problems have complicated classification functions that probably require a model with lots of capacity for complication in order to do well, and in many cases training data is expensive and hard to get so you don't have that much to work with! So for many realistic problems we may face overfitting when we try to use powerful models.
# 
# The "traditional" ML response is to try simpler models. Fastai disagrees with this in an interesting way - they say to use more powerful models, and then fix the overfitting problem. In fact, they say that **you should always try to get to the point of overfitting** - in other words, if you don't get to the point of overfitting you're probably leaving something on the table! This strikes me as similar to business negotiators who believe that if you don't get walked out on in disgust at some point, you haven't negotiated hard enough.
# 
# So, how do we fix overfitting when it happens? The most reliable way to fix it is to get more data! But, as we just said, that may not be possible. So, the rest of this revisited lesson basically talks about what else you can do - in other words, how to use powerful models when you don't have enough data to train them from scratch. In a nutshell there are two things that can help a lot - **transfer learning** and **regularization** - and Fastai makes heavy use of both.

# ## Step 3 - Transfer Learning  

# In transfer learning, we use a powerful model trained on a task that's related to what we want,
# but for which there's a lot more data available, so we can train the model well.
# We then modify the pretrained model a little to adapt it to our task,
# and train the modified part to do our task -
# we can avoid overfitting in this stage because the part we're training is relatively small.
# In a third step, we may fine-tune the whole model, but here we'll again be in danger of overfitting.
# Fastai has a clever way to do this fine-tuning that usually improves performance a little.
# In fact, Fastai is generally set up to do transfer learning very conveniently and with good performance.
# 
# A classic use of transfer learning is in image classification, so let's go through that in some detail
# (for a use of transfer learning in another domain, see the ULMFit algorithm in lesson 3).
# We'll again use the powerful NN model Resnet34, but this time we won't add `pretrained=False`,
# so we'll get a version of the model which has been pretrained on the Imagenet dataset
# (a large training set of images containing different objects that's used in ML competitions).

# In[ ]:


randomSeedForTraining(3141);
learn = create_cnn(data, models.resnet34, metrics=accuracy)


# Let's look at what's going on here - how was this model created?

# In[ ]:


print(learn.summary())


# The part of the model that was newly created for our problem is the last few lines. The main items we added are the two Linear layers:
# ```
# Linear               [64, 512]            524800     True
# ```
# and
# ```
# Linear               [64, 11]             5643       True
# ```
# that go from the activations in the pretrained network to
# a new hidden layer of 512 nodes,
# and then to an output layer of 11 nodes (we can tell how big we need to make
# this output layer from the dataset we passed to `create_cnn` - in this case
# we need 11 outputs since we're
# doing classification into 11 categories).
# 
# These new parts are the ones that we'll now train -
# the rest of the network will be frozen to the pretrained weights from training on Imagenet.
# Note that `create_cnn` allows for a considerable amount of customization
# in how to modify the pretrained network; for example, you can specify a different set of linear hidden layers using `lin_ftrs=`,
# or even specify your own model for the added part using `custom_head=`. For details, see the documentation!
# 
# Note: `create_cnn` has been changed to `cnn_learner` in later versions of Fastai.
# 
# Note: you can easily see the documentation for any function by running `doc(fname)` -
# this will pop up a summary, and give you links to the full documentation and to the source code
# (which can also be instructive)!
# 
# Let's try training with transfer learning.
# We'll also start using `fit_one_cycle`, which is a modified form of training that changes the learning rate over time,
# gradually increasing it during the first part of training and reducing it toward the end.
# This has been found to generally improve the final results.

# In[ ]:


learn.fit_one_cycle(3,5e-3)


# And now we suddenly have pretty good performance!
# Let's save the model at this stage so we can revert to it below after we screw things up ...

# In[ ]:


learn.save('stage-1')


# In the interest of time I'll skip the part of Lesson 1 where we look at
# the results in detail using `ClassificationInterpretation`,
# but you should definitely check that out if you haven't yet!
# 
# Let's look at the fine-tuning process. What we want to do is tune the rest of the model in addition
# to the part we added. What happens if we do this in a simple way?

# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(1,5e-3)


# Looks like we got worse! And this isn't too surprising, because we're now training a much larger number of weights (since we unfroze the pretrained weights). So we can't usually improve performance if we just train the whole network at the same rate as before. But, it turns out we can often improve performance a bit by training the whole network "just a little" - i.e. fine-tuning it. Fastai provides a clever way of doing this called discriminative learning rates. Instead of a single learning rate, we pass two learning rates, the first smaller than the second (we choose these learning rates as described in lesson 1 - the first is usually 1/10 of the learning rate used in the first training stage above, while the second is chosen by running `lr_find` again). Fastai then uses the smaller learning rate on the shallower layers of the network, and the larger rate on the deeper layers - this seems to work best in practice. Let's try this.

# In[ ]:


learn.load('stage-1')  # forget about the training above and revert to stage 1
learn.unfreeze()
learn.lr_find(); learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(2,slice(1e-5,5e-4))


# This concludes our quick review of transfer learning and fine-tuning, which is the first way Fastai lets us use more powerful models on relatively small training sets. But, there's more going on under the hood, and some of it may help you further improve performance in many cases. So, let's move on to the second bag of tricks - **regularization**.

# ## Regularization
# 
# Regularization isn't a specific method, but a catch-all term for a group of methods.
# It refers to a bunch of different techniques that **prevent or reduce overfitting**.
# Fastai makes heavy use of several of these by default,
# applied unobtrusively under the hood.
# 
# We'll quickly review a grab-bag of these regularization methods,
# commenting in particular on how and when you may want to try changing the defaults to help performance.

# ### BatchNorm
# 
# This one is a bit mysterious - a combination of regularization and training helper,
# and it's not clear why it has a regularizing effect.
# To apply it, we take for each activation the mean and variance over each mini-batch,
# and do a modified form of normalization.
# We then do a linear rescaling using two **trainable** weights for each activation.
# This may actually be the more important step, since (intuitively) it makes it
# easier for the training to rescale activations into a better range -
# otherwise this could require modifying many shallower weights in a complicated way.
# 
# We won't talk about BatchNorm much more because:
# - There's not much call for varying it - you almost always want to use it, because it almost always improves training! If you look at summaries of practically any model in Fastai, you'll see that it's generally invoked after a layer's activation function (you'll see ReLU, then BatchNorm).
# - It's not really clear how it works (the original paper said it "reduces covariate shift", but that turned out to be wrong).

# ### Weight Decay
# 
# Weight decay (AKA L2 regularization) is a more understandable form of regularization,
# and also one that could be worth experimenting with.
# It works by simply adding to the loss function a factor `wd` * the sum-of-squares of all your model's weights.
# Effectively, this penalizes your model for having weights significantly different from zero -
# the higher the value of wd, the greater the penalty, and the more the regularizing effect.
# 
# So, if you see overfitting you may want to try a higher value of wd,
# and if you think your model may be underfitting you can try a lower value.
# Fastai makes this easy - functions that construct a learner have an optional `wd=` argument.
# The default value for wd is 0.01, which is quite low -
# it's usually worth trying some higher values (0.05, 0.1, 0.2). For example:

# In[ ]:


randomSeedForTraining(3141);
learn = create_cnn(data, models.resnet34, metrics=accuracy, wd=0.1)
learn.fit_one_cycle(3,5e-3)


# ### Dropout
# 
# Dropout is a slightly more mysterious form of regularization - it works by, at each minibatch,
# "dropping" (zeroing out) a random selection of the activations, as specified by a probability p.
# This has been found to reduce overfitting; intuitively, it may prevent the model from
# depending on any particular activation for too much of its prediction.
# The higher the probability, the stronger the regularizing effect,
# so again if you see overfitting you may want to try a higher dropout probability,
# and if you think your model may be underfitting you may want to try a lower one.
# 
# In the example above, dropout is used on the two new Linear layers (see the model summary above).
# You can set the dropout probability for these layers using the `ps=` argument to `create_cnn/cnn_learner` - if this is a float, the final layer will have that probability and
# the previous layers will have `ps/2`; you can also use a list to set the probability directly
# for each of the new layers.
# 
# The default value for `ps` is 0.5. Example of using a lower value (less regularization):

# In[ ]:


randomSeedForTraining(3141);
learn = create_cnn(data, models.resnet34, metrics=accuracy, ps=[0.1,0.2])
learn.fit_one_cycle(3,5e-3)


# ### Data Augmentation
# 
# This means we modify our training data to generate new examples,
# hopefully ones that are "realistic".
# If done well, this is almost like getting more training data for free.
# And, as we said above the best way to prevent overfitting is to get more data!
# 
# Fastai has a good built-in system for doing this for images,
# and the defaults are set up to work well for the common case
# of photos taken from handheld cameras.
# If you do want to get the best possible performance,
# it's worth understanding how this works in some detail,
# and how you might want to change things for different image types.
# 
# The basic ML training step is to take a mini-batch
# (a random selection of `batchsize` examples) from the training set,
# calculate the loss function and gradient,
# and update weights using the gradient.
# When we do this for images, fastai randomly selects and applies a group of transforms
# to each image before feeding it into the mini-batch,
# as specified by the `tfms=` argument when the dataset is created.
# You'll generally use `get_transforms()` to create this argument, and this default
# will work well for common photos.
# However, if you want to get the best possible performance it may be worth looking
# at the documentation for `get_transforms`
# 
# https://docs.fast.ai/vision.transform.html#get_transforms
# 
# and carefully considering what transforms to use - specifically, how realistic
# is each transform for the type of images you'll be predicting?
# Some examples:
# - `do_flip`, `flip_vert` say whether to apply random horizontal and vertical flips. For "regular" photos, you probably want horizontal but not vertical flips (the default). For satellite images, both. For chest X-rays, neither!
# - `max_warp` applies a perspective warp that's similar to taking a photo from a higher or lower angle. Regular photos - yes; Satellite images - no.
# - `max_lighting` randomly changes lighting and contrast. To predict photos taken by cameraphone you probably want a fair amount of this; for photos taken in a studio, not as much.
# 
# Example of using different arguments to `get_transforms`:

# In[ ]:


data = ImageDataBunch.from_name_re(path_img, fnames, pat,
                                   ds_tfms=get_transforms(flip_vert=True,max_warp=0.1),
                                   size=224, bs=bs, num_workers=0
                                  ).normalize(imagenet_stats)

