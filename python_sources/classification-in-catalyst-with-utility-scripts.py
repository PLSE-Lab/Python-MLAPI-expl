#!/usr/bin/env python
# coding: utf-8

# ## General information
# 
# It is usually a good idea to write modular code, if we want reproducibility, ability to easily change the code and conveniently reuse the same code for multiple purposes.
# 
# Currently there is[ a utility scripts competition](https://www.kaggle.com/general/109651) and have decided to contribute.
# 
# This kernel is inspired by this work: https://www.kaggle.com/samusram/cloud-classifier-for-post-processing
# 
# The main idea is also training a classifier for postprocessing. I combine it with models trained in my previous kernel: https://www.kaggle.com/artgor/segmentation-in-pytorch-using-convenient-tools/
# 
# In this kernel I once again use the following libraries (or ideas from them):
# 
# * [albumentations](https://github.com/albu/albumentations): this is a great library for image augmentation which makes it easier and more convenient
# * [catalyst](https://github.com/catalyst-team/catalyst): this is a great library which makes using PyTorch easier, helps with reprodicibility and contains a lot of useful utils
# * [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch): this is a great library with convenient wrappers for models, losses and other useful things
# * [pytorch-toolbelt](https://github.com/BloodAxe/pytorch-toolbelt): this is a great library with many useful shortcuts for building pytorch models
# 
# But in this case all the functions and classes are imported from my utility script: https://www.kaggle.com/artgor/pytorch-utils-for-images
# 
# It has functions to:
# - get models, optimizers and other things necessary for training;
# - get dataloaders;
# - train model;
# - make visualizations;
# - and so on.
# 
# This code isn't ideal and I continue working on improving it, but I actually use it to train models locally.
# 
# Also I want to say, that this code is split into several scripts: for models, augmentations, dataset and so on separately. But I think it is better to keep it in a single kaggle utility script.
# 
# P. S. This code can be also used for severstal competition, you would only need to modify `prepare_loaders` function ;)
# ![](https://cdn.technologynetworks.com/tn/images/thumbs/jpeg/640_360/repeatability-vs-reproducibility-317157.jpg)

# In[ ]:


# imports
import torch
import gc
from pytorch_utils_for_images import *


# ## Setting up parameters
# 
# At first it is a good idea to set up random seeds.

# In[ ]:


SEED = 42

set_global_seed(SEED)
prepare_cudnn(deterministic=True)


# Now I'm setting various parameters for functions. Description for most of these parameters can be found in my utility script.
# 
# When I train models locally, I prefer to either run the code in command line and pass parameters there or to set up config files.
# 
# With this approach it is easy to switch between classification and segmentation tasks, between model architectures and encoders.

# In[ ]:


task = 'classification'
encoder = 'densenet169'
encoder_weights = 'imagenet'
batch_size = 8
path = '../input/understanding_cloud_organization'
num_workers = 0
segm_type = 'Unet'
activation = None
n_classes = 4
loss = 'BCE'
gradient_accumulation = 'False'
num_epochs = 20
lr = 1e-4


# ## Preparing for training
# 
# Now we need to get everything necessary for training the model:
# - get data loaders with training and validation datasets;
# - get the model itself. It is densenet169 with custom head right now (several linear layers with batch norm and dropout);
# - get optimizer, scheduler and criterion;
# - get catalyst callbacks;
# 
# We have the masks, but for multilabel classification we would need labels instead, I do it as in this great kernel: https://www.kaggle.com/samusram/cloud-classifier-for-post-processing (in my function `prepare_loaders`)
# ```python
#     train = pd.read_csv(f'{path}/train.csv')
#     train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
#     train['im_id'] = train['Image_Label'].apply(lambda x: x.split('_')[0])
#     if task == 'classification':
#         train_df = train[~train['EncodedPixels'].isnull()]
#         classes = train_df['label'].unique()
#         train_df = train_df.groupby('im_id')['label'].agg(set).reset_index()
#         for class_name in classes:
#             train_df[class_name] = train_df['label'].map(lambda x: 1 if class_name in x else 0)
# 
#         img_2_ohe_vector = {img: np.float32(vec) for img, vec in zip(train_df['im_id'], train_df.iloc[:, 2:].values)}
# ````

# In[ ]:


preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)
loaders = prepare_loaders(path=path, bs=batch_size,
                          num_workers=num_workers, preprocessing_fn=preprocessing_fn, preload=False, task=task,
                          image_size=(224, 224))
test_loader = loaders['test']
del loaders['test']

model = get_model(model_type=segm_type, encoder=encoder, encoder_weights=encoder_weights,
                  activation=activation, task=task, n_classes=n_classes, head='simple')


# In[ ]:


optimizer = get_optimizer(optimizer='RAdam', lookahead=False, model=model, separate_decoder=False, lr=lr, lr_e=lr)


# In[ ]:


scheduler = ReduceLROnPlateau(optimizer, factor=0.7, patience=2)
criterion = get_loss(loss)
criterion


# In[ ]:


if task == 'segmentation':
    callbacks = [DiceCallback(), EarlyStoppingCallback(patience=5, min_delta=0.001), CriterionCallback()]
elif task == 'classification':
    callbacks = [AUCCallback(class_names=['Fish', 'Flower', 'Gravel', 'Sugar'], num_classes=4), EarlyStoppingCallback(patience=5, min_delta=0.001), CriterionCallback(), CustomCheckpointCallback()]

runner = SupervisedRunner()


# ## Exploring augmentations with albumentations
# 
# One of important things while working with images is choosing good augmentations. There are a lot of them, let's have a look at augmentations from albumentations!
# 
# For classification the position of clouds is less important, so we could use more agressive augmentations.

# In[ ]:


train = pd.read_csv('../input/understanding_cloud_organization/train.csv')
train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
train['im_id'] = train['Image_Label'].apply(lambda x: x.split('_')[0])


# In[ ]:


image_name = '8242ba0.jpg'
image = get_img(image_name, '../input/understanding_cloud_organization/train_images')
mask = make_mask(train, image_name)


# In[ ]:


plot_with_augmentation(image, mask, albu.HorizontalFlip(p=1))


# In[ ]:


plot_with_augmentation(image, mask, albu.Blur(p=1))


# In[ ]:


plot_with_augmentation(image, mask, albu.CLAHE(p=1))


# In[ ]:


plot_with_augmentation(image, mask, albu.ShiftScaleRotate(p=1))


# ## Training the model
# 

# In[ ]:


for i, param in list((model.named_parameters()))[:-10]:
    param.requires_grad = False


# In[ ]:


logdir = './logs/classification'

#num_epochs = 1
runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            loaders=loaders,
            callbacks=callbacks,
            logdir=logdir,
            num_epochs=num_epochs,
            verbose=True
        )


# We can see not only mean AUC, but also AUC for each class.

# In[ ]:


plot_metrics(
    logdir='../input/cloud-segmentation-model', 
    # specify which metrics we want to plot
    metrics=["loss", "auc/_mean", "auc/class_Fish", "auc/class_Flower", "auc/class_Gravel", "auc/class_Sugar", "_base/lr"]
)


# Let's get predictions and the original labels.

# In[ ]:


valid_predictions = runner.predict_loader(
    model, loaders["valid"],
    resume=f"{logdir}/checkpoints/best.pth", verbose=True
)

y_valid = []
for img, label in loaders["valid"].dataset:
    y_valid.append(label)
y_valid = np.array(y_valid)


# In[ ]:


valid_predictions = sigmoid(valid_predictions)


# Let's have a look at precision-recall curves!

# In[ ]:


class_dict = {0: 'Fish', 1: 'Flower', 2: 'Gravel', 3: 'Sugar'}
fig, ax = plt.subplots(figsize = (16, 12))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plot_precision_recall(y_valid[:, i], valid_predictions[:, i], title=class_dict[i])


# Looks quite good for a first attempt.
# 
# For classification we would need to find some thresholds, so that we have labels and not probabilites.

# In[ ]:


class_thresholds = {}
for i in range(4):
    print(f"Class: {class_dict[i]}")
    t = find_threshold(y_valid[:, i], valid_predictions[:, i])
    print()
    class_thresholds[i] = t


# For comparison I also want to try a function from this kernel: https://www.kaggle.com/samusram/cloud-classifier-for-post-processing

# In[ ]:


def get_threshold_for_recall(y_true, y_pred, class_i, recall_threshold=0.95, precision_threshold=0.94, plot=False):
    precision, recall, thresholds = metrics.precision_recall_curve(y_true[:, class_i], y_pred[:, class_i])
    i = len(thresholds) - 1
    best_recall_threshold = None
    while best_recall_threshold is None:
        next_threshold = thresholds[i]
        next_recall = recall[i]
        if next_recall >= recall_threshold:
            best_recall_threshold = next_threshold
        i -= 1
        
    # consice, even though unnecessary passing through all the values
    best_precision_threshold = [thres for prec, thres in zip(precision, thresholds) if prec >= precision_threshold][0]
    
    if plot:
        plt.figure(figsize=(10, 7))
        plt.step(recall, precision, color='r', alpha=0.3, where='post')
        plt.fill_between(recall, precision, alpha=0.3, color='r')
        plt.axhline(y=precision[i + 1])
        recall_for_prec_thres = [rec for rec, thres in zip(recall, thresholds) 
                                 if thres == best_precision_threshold][0]
        plt.axvline(x=recall_for_prec_thres, color='g')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.legend(['PR curve', 
                    f'Precision {precision[i + 1]: .2f} corresponding to selected recall threshold',
                    f'Recall {recall_for_prec_thres: .2f} corresponding to selected precision threshold'])
        plt.title(f'Precision-Recall curve for Class {class_dict[class_i]}')
    return best_recall_threshold, best_precision_threshold

recall_thresholds = dict()
precision_thresholds = dict()
for i, class_name in tqdm.tqdm(enumerate(class_dict.items())):
    recall_thresholds[class_name], precision_thresholds[class_name] = get_threshold_for_recall(y_valid, valid_predictions, i, plot=True)
    
print('recall_thresholds', recall_thresholds)


# In[ ]:


recall_thresholds = {k[0]: v for k, v in recall_thresholds.items()}


# ## Predicting masks
# 
# Now we will need to load a segmentation model to make predictions. This can be easily done.

# In[ ]:


# freeing memory
torch.cuda.empty_cache()
gc.collect()
del runner


# In[ ]:


task = 'segmentation'
encoder = 'resnet50'
batch_size = 8

model = get_model(model_type=segm_type, encoder=encoder, encoder_weights=encoder_weights,
                  activation=activation, task=task, n_classes=n_classes, head='custom')
loaders = prepare_loaders(path=path, bs=batch_size,
                          num_workers=num_workers, preprocessing_fn=preprocessing_fn, preload=False, task=task)

del loaders['train']
del loaders['test']
checkpoint_path = '../input/cloud-segmentation-model/best_full.pth'
checkpoint = utils.load_checkpoint(checkpoint_path)
model.cuda()
utils.unpack_checkpoint(checkpoint, model=model)
runner = SupervisedRunner(model=model)


# In[ ]:


loaders = {"infer": loaders['valid']}
runner.infer(
    model=runner.model,
    loaders=loaders,
    callbacks=[
        CheckpointCallback(
            resume=checkpoint_path),
        InferCallback()
    ],
)

valid_masks = []
probabilities = np.zeros((len(runner.callbacks[0].predictions['logits']) * 4, 350, 525))
for i, (batch, output) in enumerate(zip(
    loaders['infer'].dataset, runner.callbacks[0].predictions["logits"])):
    image, mask = batch
    for m in mask:
        if m.shape != (350, 525):
            m = cv2.resize(m, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
        valid_masks.append(m)

    for j, probability in enumerate(output):
        if probability.shape != (350, 525):
            probability = cv2.resize(probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
        probabilities[i * 4 + j, :, :] = probability


# ## Dice of raw predictions

# In[ ]:


valid_ids = loaders['infer'].dataset.img_ids


# In[ ]:


original_dices = []
d = []
for ind, (i, j) in enumerate(zip(probabilities, valid_masks)):
    if (i.sum() == 0) & (j.sum() == 0):
        d.append(1)
    else:
        d.append(dice(i, j))
    if len(d) == 4:
        d = [valid_ids[ind // 4]] + d
        original_dices.append(d)
        d = []


# In[ ]:


original_dices = pd.DataFrame(original_dices)
original_dices.columns = ['img', 'Fish', 'Flower', 'Gravel', 'Sugar']
original_dices['total_dice'] = original_dices[['Fish', 'Flower', 'Gravel', 'Sugar']].sum(1)
original_dices['mean_dice'] = original_dices['total_dice'] / 4
original_dices.head()


# In[ ]:


for c in ['Fish', 'Flower', 'Gravel', 'Sugar']:
    print(f"Mean dice for {c} is {original_dices[c].mean():.4f}")


# We can see that the raw predictions give quite a bad dice, which is to be expected. Let's try post-processing!

# ## Post processing

# In[ ]:


class_params = {}
for class_id in range(4):
    print(class_id)
    attempts = []
    for t in range(0, 100, 10):
        t /= 100
        for ms in [0, 100, 1000, 5000, 10000, 11000, 14000, 15000, 16000, 18000, 19000, 20000, 21000, 23000, 25000, 27000, 30000, 50000]:
            masks = []
            for i in range(class_id, len(probabilities), 4):
                probability = probabilities[i]
                predict, num_predict = post_process(sigmoid(probability), t, ms)
                masks.append(predict)

            d = []
            for i, j in zip(masks, valid_masks[class_id::4]):
                if (i.sum() == 0) & (j.sum() == 0):
                    d.append(1)
                else:
                    d.append(dice(i, j))

            attempts.append((t, ms, np.mean(d)))

    attempts_df = pd.DataFrame(attempts, columns=['threshold', 'size', 'dice'])

    attempts_df = attempts_df.sort_values('dice', ascending=False)
    print(attempts_df.head())
    best_threshold = attempts_df['threshold'].values[0]
    best_size = attempts_df['size'].values[0]

    class_params[class_id] = (best_threshold, best_size)

print(class_params)


# In[ ]:


processed_dices = []
d = []
for ind, (i, j) in enumerate(zip(probabilities, valid_masks)):
    i, num_predict = post_process(sigmoid(i), class_params[ind % 4][0],
                                                       class_params[ind % 4][1])
    if (i.sum() == 0) & (j.sum() == 0):
        d.append(1)
    else:
        d.append(dice(i, j))
    if len(d) == 4:
        d = [valid_ids[ind // 4]] + d
        processed_dices.append(d)
        d = []


# In[ ]:


processed_dices = pd.DataFrame(processed_dices)
processed_dices.columns = ['img', 'Fish', 'Flower', 'Gravel', 'Sugar']
processed_dices['total_dice'] = processed_dices[['Fish', 'Flower', 'Gravel', 'Sugar']].sum(1)
processed_dices['mean_dice'] = processed_dices['total_dice'] / 4
processed_dices.head()


# In[ ]:


for c in ['Fish', 'Flower', 'Gravel', 'Sugar']:
    print(f"Mean dice for {c} is {processed_dices[c].mean():.4f}")


# Of course, now dice is much better.

# In[ ]:


fig, ax = plt.subplots(figsize = (12, 6))
plt.subplot(1, 2, 1)
plt.hist(original_dices['Fish'], label='Fish');
plt.hist(original_dices['Flower'], label='Flower');
plt.hist(original_dices['Gravel'], label='Gravel');
plt.hist(original_dices['Sugar'], label='Sugar');
plt.title('Original dices');

plt.subplot(1, 2, 2)
plt.hist(processed_dices['Fish'], label='Fish');
plt.hist(processed_dices['Flower'], label='Flower');
plt.hist(processed_dices['Gravel'], label='Gravel');
plt.hist(processed_dices['Sugar'], label='Sugar');
plt.title('Processed dices');
plt.legend();


# It seems that the main improvement is thanks to making some masks empty correctly! But is this always so? Let's have a look at images which had mean dice increased/decreased the most.

# In[ ]:


processed_dices.columns = [f'{col}_processed' for col in processed_dices.columns]
dices = pd.merge(original_dices, processed_dices, left_on='img', right_on='img_processed')
dices['dice_diff'] = dices['mean_dice_processed'] - dices['mean_dice']


# In[ ]:


dices.sort_values('dice_diff').head()


# In[ ]:


dices.sort_values('dice_diff', ascending=False).head()


# It seems that our post processing sometimes created additional problems. One of the ways to fix it is to apply a classifier for post-processing.

# ## Using classifier

# In[ ]:


# converting probabilities from classifier to labels
binary_classifier_predictions = np.zeros_like(valid_predictions)
for i in range(4):
    binary_classifier_predictions[:, i] = (valid_predictions[:, i] > class_thresholds[i]) * 1
    
binary_classifier_predictions = binary_classifier_predictions.reshape(-1, 1)


# In[ ]:


classified_dices = []
d = []
for ind, (i, j) in enumerate(zip(probabilities, valid_masks)):
    i, num_predict = post_process(sigmoid(i), class_params[ind % 4][0],
                                                       class_params[ind % 4][1])
    i = i if binary_classifier_predictions[ind] == 1 else i * 0
    if (i.sum() == 0) & (j.sum() == 0):
        d.append(1)
    else:
        d.append(dice(i, j))
    if len(d) == 4:
        d = [valid_ids[ind // 4]] + d
        classified_dices.append(d)
        d = []


# In[ ]:


classified_dices = pd.DataFrame(classified_dices)
classified_dices.columns = ['img', 'Fish', 'Flower', 'Gravel', 'Sugar']
classified_dices['total_dice'] = classified_dices[['Fish', 'Flower', 'Gravel', 'Sugar']].sum(1)
classified_dices['mean_dice'] = classified_dices['total_dice'] / 4
classified_dices.head()


# In[ ]:


for c in ['Fish', 'Flower', 'Gravel', 'Sugar']:
    print(f"Mean dice for {c} is {classified_dices[c].mean():.4f}")


# ## Using different classifier

# In[ ]:


# converting probabilities from classifier to labels
binary_classifier_predictions = np.zeros_like(valid_predictions)
for i in range(4):
    binary_classifier_predictions[:, i] = (valid_predictions[:, i] > recall_thresholds[i]) * 1
    
binary_classifier_predictions = binary_classifier_predictions.reshape(-1, 1)

classified_dices = []
d = []
for ind, (i, j) in enumerate(zip(probabilities, valid_masks)):
    i, num_predict = post_process(sigmoid(i), class_params[ind % 4][0],
                                                       class_params[ind % 4][1])
    i = i if binary_classifier_predictions[ind] == 1 else i * 0
    if (i.sum() == 0) & (j.sum() == 0):
        d.append(1)
    else:
        d.append(dice(i, j))
    if len(d) == 4:
        d = [valid_ids[ind // 4]] + d
        classified_dices.append(d)
        d = []
        
classified_dices = pd.DataFrame(classified_dices)
classified_dices.columns = ['img', 'Fish', 'Flower', 'Gravel', 'Sugar']
classified_dices['total_dice'] = classified_dices[['Fish', 'Flower', 'Gravel', 'Sugar']].sum(1)
classified_dices['mean_dice'] = classified_dices['total_dice'] / 4
classified_dices.head()


# In[ ]:


for c in ['Fish', 'Flower', 'Gravel', 'Sugar']:
    print(f"Mean dice for {c} is {classified_dices[c].mean():.4f}")


# ## Ideas for improvement
# 
# This was only one of the ways of improving the score, here are some more ideas:
# - try applying classifier at first and then finding thresholds for post-processing;
# - train segmentation only on images on masks;
# - while optimizing classifier thresholds and post processing, do it at the same time, trying to maximize dice;
# - use different way to find classifier thresholds;
# - use some ideas from severstal competition;

# ## Making predictions

# In[ ]:


# predict with classifier
torch.cuda.empty_cache()
del runner

task = 'classification'
encoder = 'densenet169'
batch_size = 8

model = get_model(model_type=segm_type, encoder=encoder, encoder_weights=encoder_weights,
                  activation=activation, task=task, n_classes=n_classes)
loaders = prepare_loaders(path=path, bs=batch_size,
                          num_workers=num_workers, preprocessing_fn=preprocessing_fn, preload=False, task=task,
                          image_size=(224, 224))
del loaders['train']
del loaders['valid']
checkpoint_path = f"{logdir}/checkpoints/best.pth"
runner = SupervisedRunner()
test_predictions = runner.predict_loader(
    model, loaders["test"],
    resume=checkpoint_path, verbose=True
)


# In[ ]:


# convert classifier predictions into labels
binary_classifier_predictions = np.zeros_like(test_predictions)
for i in range(4):
    binary_classifier_predictions[:, i] = (test_predictions[:, i] > class_thresholds[i]) * 1
    
binary_classifier_predictions = binary_classifier_predictions.reshape(-1, 1)


# In[ ]:


binary_classifier_predictions1 = np.zeros_like(test_predictions)
for i in range(4):
    binary_classifier_predictions1[:, i] = (test_predictions[:, i] > recall_thresholds[i]) * 1
    
binary_classifier_predictions1 = binary_classifier_predictions1.reshape(-1, 1)


# In[ ]:


torch.cuda.empty_cache()
del runner
del test_predictions
del probabilities
del model
gc.collect()


# In[ ]:


task = 'segmentation'
encoder = 'resnet50'
batch_size = 8

model = get_model(model_type=segm_type, encoder=encoder, encoder_weights=encoder_weights,
                  activation=activation, task=task, n_classes=n_classes)
loaders = prepare_loaders(path=path, bs=batch_size,
                          num_workers=num_workers, preprocessing_fn=preprocessing_fn, preload=False, task=task)


# In[ ]:


loaders['test'] = test_loader
checkpoint_path = '../input/cloud-segmentation-model/best_full.pth'
checkpoint = utils.load_checkpoint(checkpoint_path)
model.cuda()
utils.unpack_checkpoint(checkpoint, model=model)
runner = SupervisedRunner(model=model)


# In[ ]:


encoded_pixels = []
encoded_pixels1 = []
image_id = 0
for _, test_batch in enumerate(loaders['test']):
    runner_out = runner.predict_batch({"features": test_batch[0].cuda()})['logits']
    for _, batch in enumerate(runner_out):
        for probability in batch:

            probability = probability.cpu().detach().numpy()
            if probability.shape != (350, 525):
                probability = cv2.resize(probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
                prediction, num_predict = post_process(sigmoid(probability), class_params[image_id % 4][0],
                                                   class_params[image_id % 4][1])
                prediction = prediction if binary_classifier_predictions[image_id] == 1 else prediction * 0
                prediction1 = prediction if binary_classifier_predictions1[image_id] == 1 else prediction * 0
            if num_predict == 0:
                encoded_pixels.append('')
                encoded_pixels1.append('')
            else:
                r = mask2rle(prediction)
                encoded_pixels.append(r)
                r1 = mask2rle(prediction1)
                encoded_pixels1.append(r1)
            image_id += 1

sub = pd.read_csv(f'{path}/sample_submission.csv')
sub['EncodedPixels'] = encoded_pixels
sub.to_csv(f'submission.csv', columns=['Image_Label', 'EncodedPixels'], index=False)
sub['EncodedPixels'] = encoded_pixels1
sub.to_csv(f'submission1.csv', columns=['Image_Label', 'EncodedPixels'], index=False)

