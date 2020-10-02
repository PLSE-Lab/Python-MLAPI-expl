#!/usr/bin/env python
# coding: utf-8

# # Overview
# ## Model
# The model choice is a recursive U-Net where the output segmentation is fed into the same model again as a _seed_ segmentation for making the next prediction. The idea is that maybe the model can get iteratively better by knowing what the last model produced. The parameter update is quite complicated since with 4 loops it is going through the same model 4 times (a bit RNN-like), but it might work?

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (8, 8)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 14
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.style.use('ggplot')
sns.set_style("whitegrid", {'axes.grid': False})


# # Setup and Loading

# In[ ]:


import pandas as pd
import numpy as np
from skimage.color import label2rgb, gray2rgb
from skimage.transform import resize
from skimage.io import imread
from pathlib import Path
from tqdm import tqdm
tqdm.pandas()


# In[ ]:


img_dir = Path('..') / 'input' / 'pulmonary-chest-xray-abnormalities'
img_df = pd.DataFrame(dict(path = list(img_dir.glob('**/*.*'))))
img_df['img_id'] = img_df['path'].map(lambda x: x.stem)
img_df['folder'] = img_df['path'].map(lambda x: x.parent.stem)
img_df['group'] = img_df['path'].map(lambda x: x.parent.parent.stem)
img_df = img_df[img_df['path'].map(lambda x: x.suffix[1:].lower() in {'png', 'jpg'})]
img_df


# In[ ]:


IMG_SIZE = (1024, 1024)
def read_seg_map(in_row):
    """Read segmentation maps as images"""
    right_img = imread(in_row['rightMask'], as_gray=True)
    left_img = imread(in_row['leftMask'], as_gray=True)
    comb_img = np.clip(255.0*(right_img+left_img), 0, 255).astype('uint8')
    rs_img = resize(comb_img, IMG_SIZE)
    return np.expand_dims(rs_img, -1)
colorize = lambda x: (gray2rgb(x)*255).clip(0, 255).astype('uint8')[:, :, :3]


# In[ ]:


img_pairs_df = img_df.    pivot_table(
        columns='folder', 
        index='img_id', 
        values='path', 
        aggfunc='first').\
    dropna().\
    sample(50, random_state=2019)
img_pairs_df['segmap'] = img_pairs_df.progress_apply(read_seg_map, axis=1)
img_pairs_df['rgb_image'] = img_pairs_df['CXR_png'].progress_map(lambda x: 
                                                                 colorize(
                                                                     resize(
                                                                         imread(x, as_gray=True),
                                                                         IMG_SIZE
                                                                     ) 
                                                                 )
                                                                )
print(img_pairs_df.shape[0])
img_pairs_df.sample(1)


# In[ ]:


def show_row(n_axs, c_row, channel_wise=True):
    (ax1, ax2, ax3) = n_axs
    ax1.imshow(c_row['rgb_image'].squeeze())
    ax1.axis('off')
    
    segmap = c_row['segmap']
    col_map = (segmap[:, :, 0]>0.5).astype('int')
    ax2.imshow(col_map)
    ax2.axis('off')
    
    ax3.imshow(label2rgb(image=c_row['rgb_image'].squeeze(), label=col_map.astype('int'), bg_label=0))
    ax3.axis('off')
    
fig, m_axs = plt.subplots(3, min(len(img_pairs_df), 3), figsize=(15, 5))

for (c_idx, c_row), n_axs in zip(img_pairs_df.iterrows(), m_axs.T):
    show_row(n_axs, c_row)


# In[ ]:


from sklearn.model_selection import train_test_split
train_df, valid_df = train_test_split(img_pairs_df, test_size=0.2, random_state=2019)
train_df = train_df.copy()
valid_df = valid_df.copy() # make the datasets independent of the input


# ## Deep Learning Section

# In[ ]:


from keras import layers, models
from keras.utils.vis_utils import model_to_dot
from IPython.display import SVG
import keras.backend as K
def dice_score(y_true, y_pred):
    """
    A simple DICE implementation without any weighting
    """
    y_t = K.batch_flatten(y_true)
    y_p = K.batch_flatten(y_pred)
    return 2.0 * K.sum(y_t * y_p) / (K.sum(y_t) + K.sum(y_p) + K.epsilon())
def dice_loss(y_true, y_pred):
    """
    A simple inverted dice to use as a loss function
    """
    return 1 - dice_score(y_true, y_pred)


# ## Setup the Augmentation

# In[ ]:


from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,    
    CenterCrop,    
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion, 
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,    
    RandomGamma,
    HueSaturationValue,
    MedianBlur, 
    MotionBlur,
    Blur,
    RandomFog,
    Rotate
)


# In[ ]:


print(img_pairs_df['segmap'].iloc[0].max())
tile_size = (256, 256)
aug = Compose([
    Rotate(limit=15, p=0.5),
    OneOf([
        ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        GridDistortion(p=0.5),
        OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)                  
        ], p=0.8),
    RandomFog(p=0.1),
    CLAHE(p=0.1),
    OneOf([
        RandomBrightnessContrast(p=0.5),    
        RandomGamma(p=0.5),
        HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=0, p=0.5),
        HueSaturationValue(hue_shift_limit=30, sat_shift_limit=10, val_shift_limit=0, p=0.25),
    ]),
    Blur(p=0.2),
    RandomSizedCrop(min_max_height=(128, 512), width=tile_size[0], height=tile_size[1])
])


# In[ ]:


aug_df = pd.concat([
    train_df.sample(n=16, replace=True, random_state=i).\
        apply(
            lambda x: pd.Series(
                aug(image=x['rgb_image'], 
                    mask=(x['segmap']>0.5).astype('uint8')
                   )
            ), 1) 
    for i in tqdm(range(16))], ignore_index=True).\
    rename(columns={'image': 'rgb_image', 'mask': 'segmap'})


# In[ ]:


sample_aug_df = aug_df.sample(12)
fig, m_axs = plt.subplots(3, len(sample_aug_df), figsize=(15, 5))
for (c_idx, c_row), n_axs in zip(sample_aug_df.iterrows(), m_axs.T):
    show_row(n_axs, c_row)


# # Create the static datasets

# In[ ]:


X_train = np.stack(aug_df['rgb_image'], 0)
y_train = np.stack(aug_df['segmap'], 0)
print(X_train.shape, y_train.shape)


# In[ ]:


X_valid = np.stack(valid_df['rgb_image'], 0)
y_valid = np.stack(valid_df['segmap'], 0)
print(X_valid.shape, y_valid.shape)


# # Build and Train Models

# ## Training Code

# In[ ]:


from IPython.display import clear_output
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
def get_callbacks(in_model):
    weight_path="{}_weights.best.hdf5".format(in_model.name)

    checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                                 save_best_only=True, mode='min', save_weights_only = True)

    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
    early = EarlyStopping(monitor="val_loss", 
                          mode="min", 
                          patience=15) # probably needs to be more patient, but kaggle time is limited
    return [checkpoint, early, reduceLROnPlat], weight_path

def fit_model(in_model, epochs=50, batch_size=16, loss_func='binary_crossentropy'):
    in_model.compile(loss=loss_func, metrics=['binary_accuracy', dice_score, 'mae'], optimizer='adam')
    callback_list, weight_path = get_callbacks(in_model)
    out_results = in_model.fit(X_train, y_train, 
                                   epochs=epochs,
                                   batch_size=batch_size,
                                   validation_data=(X_valid, y_valid), 
                                   callbacks=callback_list)
    in_model.load_weights(weight_path)
    in_model.save(weight_path.replace('_weights', '_model'))
    clear_output()
    v_keys = [k for k in out_results.history.keys() if 'val_{}'.format(k) in out_results.history.keys()]
    fig, m_axs = plt.subplots(1, len(v_keys), figsize=(16, 4))
    for c_key, c_ax in zip(v_keys, m_axs):
        c_ax.plot(out_results.history[c_key], 'r-', label='Training')
        val_vec = out_results.history['val_{}'.format(c_key)]
        c_ax.plot(val_vec, 'b-', label='Validation (Best: {:2.2%})'.format(np.nanmin(val_vec) if 'loss' in c_key else np.nanmax(val_vec)))
        c_ax.set_title(c_key)
        c_ax.legend()
    fig.savefig(weight_path.replace('.hdf5', '.png'))
    return out_results


# ## Result Code

# In[ ]:


def show_training(in_model):
    sample_aug_df['predictions'] = [x for x in in_model.predict(np.stack(sample_aug_df['rgb_image'], 0))]
    fig, m_axs = plt.subplots(5, len(sample_aug_df), figsize=(15, 10))
    m_axs[1, 0].set_title('Ground-Truth')
    m_axs[3, 0].set_title('Prediction')
    for (c_idx, c_row), (ax1, ax2, ax3, ax4, ax5) in zip(sample_aug_df.iterrows(), 
                                     m_axs.T):
        show_row((ax1, ax2, ax3), c_row)
        show_row((ax1, ax4, ax5), {'rgb_image': c_row['rgb_image'], 'segmap': c_row['predictions']})
def show_validation(in_model):
    valid_df['predictions'] = [x for x in in_model.predict(np.stack(valid_df['rgb_image'], 0))]
    fig, m_axs = plt.subplots(5, len(valid_df), figsize=(8, 12))
    m_axs[1, 0].set_title('Ground-Truth')
    m_axs[3, 0].set_title('Prediction')
    for (c_idx, c_row), (ax1, ax2, ax3, ax4, ax5) in zip(valid_df.iterrows(), 
                                     m_axs.T):
        show_row((ax1, ax2, ax3), c_row)
        show_row((ax1, ax4, ax5), {'rgb_image': c_row['rgb_image'], 'segmap': c_row['predictions']})
        


# # Simple U-Net

# In[ ]:


from keras.models import Model
from keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, UpSampling2D, Input, concatenate
from keras import layers

def conv2d_block(
    inputs, 
    use_batch_norm=True, 
    dropout=0.3, 
    filters=16, 
    kernel_size=(3,3), 
    activation='leakyrelu', 
    kernel_initializer='he_normal', 
    double_layer=True,
    padding='same'):
    c = inputs
    for _ in range(2 if double_layer else 1):
        c = Conv2D(filters, kernel_size, activation='linear', kernel_initializer=kernel_initializer, padding=padding, use_bias=not use_batch_norm) (inputs)
        if use_batch_norm:
            c = BatchNormalization()(c)
        if dropout > 0.0:
            c = Dropout(dropout)(c)
        if activation.lower().startswith('leaky'):
            c = layers.LeakyReLU(0.1)(c)
        else:
            c = layers.Activation(activation)(c)
    return c

def basic_unet(
    input_shape,
    num_classes=1,
    dropout=0.0, 
    filters=64,
    num_layers=4,
    use_deconv=False,
    crop_output=True,
    output_name='OutMask',
    output_activation='sigmoid'): # 'sigmoid' or 'softmax'
    """taken from https://github.com/karolzak/keras-unet"""
    # Build U-Net model
    inputs = Input(input_shape)
    x = BatchNormalization()(inputs)   

    down_layers = []
    for l in range(num_layers):
        x = conv2d_block(inputs=x, filters=filters, use_batch_norm=False, dropout=0.0, padding='same')
        down_layers.append(x)
        x = MaxPooling2D((2, 2), strides=2) (x)
        filters = filters*2 # double the number of filters with each layer

    x = Dropout(dropout)(x)
    x = conv2d_block(inputs=x, filters=filters, use_batch_norm=False, dropout=0.0, padding='same')

    for conv in reversed(down_layers):
        filters //= 2 # decreasing number of filters with each layer 
        if use_deconv:
            x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same') (x)
        else:
            x = UpSampling2D((2, 2))(x)
        
        x = concatenate([x, conv])
        x = conv2d_block(inputs=x, filters=filters, use_batch_norm=False, dropout=0.0, padding='same')
    
    outputs = Conv2D(num_classes, (1, 1), 
                     activation=output_activation, 
                     name=output_name if not crop_output else 'Pre_{}'.format(output_name)) (x)    
    if crop_output:
        lay_17_crop = layers.Cropping2D((2**num_layers, 2**num_layers))(outputs)
        outputs = layers.ZeroPadding2D((2**num_layers, 2**num_layers), name=output_name)(lay_17_crop)

    model = Model(inputs=[inputs], outputs=[outputs], name='VanillaUNET')
    return model


# # Vanilla U-Net

# In[ ]:


simple_unet = basic_unet((None, None, 3), num_classes=1, num_layers=4, dropout=0.1, filters=32)


# In[ ]:


dot_mod = model_to_dot(simple_unet, show_shapes=True, show_layer_names=False)
dot_mod.set_rankdir('UD')
SVG(dot_mod.create_svg())


# In[ ]:


fit_model(simple_unet, epochs=50, loss_func='binary_crossentropy', batch_size=16)


# In[ ]:


plt.imshow(simple_unet.predict(np.random.uniform(0, 255, size=(1, 128, 128, 3)))[0, :, :, 0])


# In[ ]:


show_training(simple_unet)


# In[ ]:


show_validation(simple_unet)


# # Recursive Model
# The recursive model starts with a simple U-Net to get the initial segmentation and then 

# In[ ]:


pre_unet = basic_unet((None, None, 3), num_classes=1, num_layers=2, filters=8)
pre_unet.name = "FirstGuess"
pre_unet.summary()


# In[ ]:


def recursive_unet_block(
    input_shape,
    num_classes=1,
    dropout=0.0, 
    filters=16,
    num_layers=4,
    use_deconv=False,
    crop_output=True,
    resnet_style=False):
    """a unet model that can be run multiple times"""
    # Build U-Net model
    inputs = Input(input_shape, name='InImage')
    bn_img = BatchNormalization()(inputs)
    last_mask = Input(input_shape[:2]+(num_classes,), name='InMask')
    down_layers = [[], []]
    
    c_pair = [bn_img, last_mask]
    for l in range(num_layers):
        for i, x in enumerate(c_pair):
            if (l>0) and (i==0): # combine the mask and input channels
                x = concatenate(c_pair)
            
            x = conv2d_block(inputs=x, filters=filters, use_batch_norm=False, dropout=0.0, padding='same')
            down_layers[i].append(x)
            x = MaxPooling2D((2, 2), strides=2)(x)
            c_pair[i] = x
        
        filters = filters*2 # double the number of filters with each layer
    
    x = concatenate(c_pair)
    x = Dropout(dropout)(x)
    x = conv2d_block(inputs=x, filters=filters, use_batch_norm=False, dropout=0.0, padding='same')

    for conv_img, conv_mask in zip(reversed(down_layers[0]), reversed(down_layers[1])):
        filters //= 2 # decreasing number of filters with each layer 
        if use_deconv:
            x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same') (x)
        else:
            x = UpSampling2D((2, 2))(x)
        
        x = concatenate([x, conv_img, conv_mask])
        x = conv2d_block(inputs=x, filters=filters, use_batch_norm=False, dropout=0.0, padding='same')
    
    outputs = Conv2D(num_classes, (1, 1), activation='tanh' if resnet_style else 'sigmoid') (x)   
    
    if resnet_style:
        outputs = layers.add([last_mask, outputs]) # resnet style
    
    if crop_output:
        lay_17_crop = layers.Cropping2D((2**num_layers, 2**num_layers))(outputs)
        outputs = layers.ZeroPadding2D((2**num_layers, 2**num_layers), name='OutRecursive')(lay_17_crop)
    
    model = Model(inputs=[inputs, last_mask], outputs=[outputs], name='RecursiveBlock')
    return model


# In[ ]:


rec_block = recursive_unet_block(input_shape=(None, None, 3), num_classes=1, num_layers=4, filters=16)


# In[ ]:


dot_mod = model_to_dot(rec_block, show_shapes=True, show_layer_names=False)
dot_mod.set_rankdir('UD')
dot_mod.write_svg('rec_block.svg')
SVG(dot_mod.create_svg())


# ### Glue the beast together

# In[ ]:


def build_rec_unet(recursive_steps):
    in_img = Input((None, None, 3))
    first_seg = pre_unet(in_img)
    out_seg_list = [first_seg]
    for i in range(recursive_steps):
        noisy_seg = layers.GaussianNoise(0.1)(out_seg_list[-1]) # jitter things a bit
        out_seg_list.append(rec_block([in_img, noisy_seg]))
    for i in range(len(out_seg_list)):
        # name the outputs sensibly
        out_seg_list[i] = layers.Lambda(lambda x: x, name='Iter{:02}'.format(i))(out_seg_list[i])
    return Model(inputs=[in_img], outputs=out_seg_list, name='RecUNET')
rec_unet = build_rec_unet(4)


# In[ ]:


dot_mod = model_to_dot(rec_unet, show_shapes=True, show_layer_names=True)
dot_mod.set_rankdir('LR')
dot_mod.write_svg('rec_unet.svg')
SVG(dot_mod.create_svg())


# In[ ]:


rec_unet.summary()


# ### Progressively Growing Training
# Here we train the model one output at a time starting with the simple first layer and then progressively adding outputs and de-weighting the earlier stages. The idea should be to get a more stable result than trying the train the whole mess at once.

# In[ ]:


from keras import losses
def fit_rec_model(in_model, 
                  epochs=50, 
                  batch_size=8,
                 loss_weight_min=0.1):
    rec_steps = len(in_model.outputs)
    # stagger losses from being mostly dice to being mostly BCE
    loss_funcs = {
        'Iter{:02d}'.format(i): lambda x,y: (1-k)*dice_loss(x,y)+k*losses.binary_crossentropy(x,y) 
        for i,k in enumerate(np.linspace(loss_weight_min, 1-loss_weight_min, rec_steps))
    }
    all_results = []
    # progressively grow the training
    for c_step in range(rec_steps):
        base_weights = np.zeros(rec_steps)
        base_weights[:c_step] = 1
        base_weights[c_step] = 3
        loss_weights = base_weights/np.sum(base_weights)

        in_model.compile(loss=loss_funcs, 
                         metrics=['binary_accuracy', dice_score, 'mae'], 
                         optimizer='adam', 
                         loss_weights=loss_weights.tolist())

        callback_list, weight_path = get_callbacks(in_model)

        out_results = in_model.fit(X_train, [y_train]*(rec_steps), 
                                       epochs=epochs,
                                       batch_size=batch_size,
                                       validation_data=(X_valid, [y_valid]*(rec_steps)), 
                                       callbacks=callback_list)
        all_results += [out_results]
        in_model.load_weights(weight_path)
    
        clear_output()
        v_keys = [k for k in out_results.history.keys() if 'val_{}'.format(k) in out_results.history.keys()]
        fig, m_axs = plt.subplots(1, len(v_keys), figsize=(4*len(v_keys), 4))
        for c_key, c_ax in zip(v_keys, m_axs):
            c_ax.plot(out_results.history[c_key], 'r-', label='Training')
            val_vec = out_results.history['val_{}'.format(c_key)]
            c_ax.plot(val_vec, 'b-', label='Validation (Best: {:2.2%})'.format(np.nanmin(val_vec) if 'loss' in c_key else np.nanmax(val_vec)))
            c_ax.set_title(c_key)
            c_ax.legend()
        fig.savefig(weight_path.replace('.hdf5', '.png'))

        in_model.save(weight_path.replace('_weights', '_model'))
    out_df = pd.concat([pd.DataFrame(keras_history.history).                            assign(train_block=i).                            reset_index().                            rename(columns={'index': 'inner_epoch'})
                        for i, keras_history in enumerate(all_results)],
                      ignore_index=True).\
        reset_index().\
        rename(columns={'index': 'epoch'})
    return out_df


# In[ ]:


rec_fit_df = fit_rec_model(rec_unet, epochs=50)
rec_fit_df


# In[ ]:


full_fit_df = pd.melt(rec_fit_df, id_vars=['train_block','epoch', 'inner_epoch']).query('variable!="lr"')
full_fit_df['split'] = full_fit_df['variable'].map(lambda x: 'validation' if x.startswith('val_') else 'training')
full_fit_df['clean_variable'] = full_fit_df['variable'].map(lambda x: x.replace('val_', ''))
full_fit_df['model'] = full_fit_df['clean_variable'].map(lambda x: x.split('_')[0] if '_' in x else 'all')
full_fit_df['variable'] = full_fit_df['clean_variable'].map(lambda x: '_'.join(x.split('_')[1:]) if '_' in x else x)
full_fit_df.to_csv('rec_values.csv', index=False)
full_fit_df.head(5)


# In[ ]:


sns.catplot(data=full_fit_df.query('variable!="loss"'), 
            x='epoch', 
            y='value', 
            col='split', 
            hue='model', 
            row='variable', 
            kind='point', 
            sharey='row')


# In[ ]:


sns.catplot(data=full_fit_df.query('variable!="loss"'), 
            x='model', 
            y='value', 
            col='split', 
            hue='epoch', 
            row='variable', 
            kind='point', 
            sharey='row')


# In[ ]:


full_fit_df.query('variable=="binary_accuracy"').query('split=="validation"').pivot_table(columns='model', values='value', index=['epoch'])


# ### Make a simple, single input, single output model

# In[ ]:


in_vec = Input((None, None, 3))
multi_out = rec_unet(in_vec)
simple_rec_unet = Model(inputs=[in_vec], 
                        outputs=[multi_out[-1]])


# In[ ]:


show_training(simple_rec_unet)


# In[ ]:


show_validation(simple_rec_unet)


# ## Show other components

# In[ ]:


show_training(pre_unet)


# In[ ]:


def show_multistep(in_df, samples=2, steps=3):
    fig, m_axs = plt.subplots((2+steps)*2+1, samples, figsize=(6*samples, 4*((2+steps)*2+1)))
    m_axs[1, 0].set_title('Ground-Truth')
    
    for (c_idx, c_row), n_axs in zip(in_df.sample(samples, random_state=0).iterrows(), 
                                     m_axs.T):
        ax1 = n_axs[0]
        show_row([ax1]+n_axs[1:3].tolist(), c_row)
        in_img = np.expand_dims(c_row['rgb_image'], 0)
        last_seg = pre_unet.predict(in_img)
        show_row([ax1]+n_axs[3:5].tolist(), {'rgb_image': c_row['rgb_image'], 'segmap': last_seg[0]})
        m_axs[3, 0].set_title('Simple')
        for i in range(2, 2+steps):
            last_seg = rec_block.predict([in_img, last_seg])
            j = 2*i+1
            show_row([ax1]+n_axs[j:j+2].tolist(), {'rgb_image': c_row['rgb_image'], 'segmap': last_seg[0]})
            
            m_axs[j, 0].set_title('Rec#{}'.format(i-2))
            m_axs[j+1, 0].set_title('Rec#{}'.format(i-2))


# In[ ]:


show_multistep(sample_aug_df, samples=4, steps=4)


# In[ ]:


show_multistep(valid_df, steps=4)


# In[ ]:





# In[ ]:




