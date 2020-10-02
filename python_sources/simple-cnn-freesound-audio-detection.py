#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import json
import tensorflow as tf
import tensorflow.keras.layers as ls
from tensorflow.contrib.framework.python.ops import audio_ops as audio
from sklearn.preprocessing import MultiLabelBinarizer


# In[2]:


os.listdir('../input')


# In[3]:


do_run=False


# In[4]:


if not do_run:
    print(os.listdir('../input/cnnfat2019'))
    train_curated_csv = '../input/freesound-audio-tagging-2019/train_curated.csv'
    train_noisy_csv = '../input/freesound-audio-tagging-2019/train_noisy.csv'
    test_csv = '../input/freesound-audio-tagging-2019/test.csv'
    train_curated_path = '../input/freesound-audio-tagging-2019/train_curated/'
    train_noisy_path = '../input/freesound-audio-tagging-2019/train_noisy/'
else:
    train_curated_csv = '../input/train_curated.csv'
    train_noisy_csv = '../input/train_noisy.csv'
    test_csv = '../input/test.csv'
    train_curated_path = '../input/train_curated/'
    train_noisy_path = '../input/train_noisy/'


# In[ ]:


# Read train files
train_curated_df = pd.read_csv(train_curated_csv)
train_noisy_df = pd.read_csv(train_noisy_csv)

# Append path to train files
train_curated_df['fname'] = train_curated_path + train_curated_df['fname']
train_noisy_df['fname'] = train_noisy_path + train_noisy_df['fname']

# Generate labels dict
labels = np.concatenate(train_curated_df['labels'].str.split(','))
unique_labels = np.unique(labels)
labels_dict = {label:i for i, label in enumerate(unique_labels)}

# Generate Val curated df
split = int(0.2 * train_curated_df.shape[0])
indices = np.random.permutation(train_curated_df.shape[0])
val_indices = indices[:split]
train_indices = indices[split:]
val_curated_df = train_curated_df[train_curated_df.index.isin(val_indices)]
train_curated_df = train_curated_df[train_curated_df.index.isin(train_indices)]

# Generate Train df
train_df = pd.concat([train_curated_df, train_noisy_df])

# Remove unwanted labels from noisy data
noisy_labels = train_noisy_df['labels'].str.split(',')
new_noisy_labels = []
for labels in noisy_labels:
    new_noisy_labels.append(','.join([label for label in labels if label in labels_dict]))
train_noisy_df['labels'] = new_noisy_labels

# Saving files
if not os.path.isdir('./preprocessed'):
    os.makedirs('./preprocessed')
train_curated_df.to_csv('./preprocessed/train_curated.csv', index=False)
train_noisy_df.to_csv('./preprocessed/train_noisy.csv', index=False)
val_curated_df.to_csv('./preprocessed/val_curated.csv', index=False)
train_df.to_csv('./preprocessed/train.csv', index=False)

with open('./preprocessed/labels_dict.json', 'w+') as fp:
    json.dump(labels_dict, fp)


# In[ ]:


del train_curated_df
del train_df
del train_noisy_df
del train_indices
del labels_dict
import gc; gc.collect()


# In[ ]:


def random_rolls(audio):
    random_num = tf.random.uniform((1,), 0, 10)[0]
    cond = random_num >= 5
    roll_amount = tf.random.uniform((1,), 1200, 2000)[0]
    new_audio = tf.cond(
        cond, lambda: tf.manip.roll(audio, tf.cast(roll_amount, tf.int32), 0),
        lambda: audio)
    return new_audio

def random_speedx(audio):
    def get_audio():
        factor = tf.random.uniform((1,), 0.7, 1.7)[0]
        indices = tf.round(tf.range(0, tf.shape(audio)[0], factor))
        indices = tf.boolean_mask(indices, indices < tf.cast(tf.shape(audio)[0], tf.float32))
        sound = tf.gather(audio, tf.cast(indices, tf.int32))
        return sound

    random_num = tf.random.uniform((1,), 0, 10)[0]
    cond = random_num >= 5
    return tf.cond(cond, get_audio, lambda: audio)

def wav_to_spectogram(wav_filename, random_perturbs=False):
    wav_bytes = tf.read_file(wav_filename)
    wav_decoder = audio.decode_wav(wav_bytes, 1)
    if random_perturbs:
        sound = random_speedx(random_rolls(wav_decoder.audio))
    else:
        sound = wav_decoder.audio
    spectogram = audio.audio_spectrogram(
        sound, window_size=1024, stride=64)
    
    minimum =  tf.minimum(spectogram, 255.)
    expand_dims = tf.expand_dims(minimum, -1)
    resize = tf.image.resize_bilinear(expand_dims, [300, 300])
    squeeze = tf.squeeze(resize, 0)

    return squeeze

def csv_pipe(csv, labels_dict):
    df = pd.read_csv(csv)
    df = df.sample(frac=1).reset_index(drop=True)
    
    files = df['fname']
    labels = df['labels'].str.split(',')

    binarizer = MultiLabelBinarizer(list(labels_dict.keys()))
    labels_1hot = binarizer.fit_transform(labels).astype(np.int32)
    return files, labels_1hot

def wav_data_generators(csv, labels_dict, batch_size=32, shuffle=False, repeat=True):
    files, labels = csv_pipe(csv, labels_dict)

    wav_to_spectogram_applier = lambda file, label: (wav_to_spectogram(file, random_perturbs=shuffle), label)

    dataset = tf.data.Dataset.from_tensor_slices((files, labels))
    dataset = dataset.map(wav_to_spectogram_applier)

    if shuffle:
        dataset = dataset.shuffle(800)

    dataset = dataset.batch(batch_size).prefetch(10)

    if repeat:
        dataset = dataset.repeat()

    return dataset


# In[ ]:


def conv_relu_bn(filters, kernels, strides, padding='valid', in_shape=None):
    if in_shape is not None:
        conv = ls.Conv2D(filters, kernels, strides, padding=padding, input_shape=in_shape)
    else:
        conv = ls.Conv2D(filters, kernels, strides, padding=padding)
    return tf.keras.models.Sequential([
        conv,
        ls.ReLU(),
        ls.BatchNormalization()
    ])
    

class Model(tf.keras.Model):
    def __init__(self, in_shape, n_classes=80):
        super(Model, self).__init__()
        self.conv_relu_bn1 = conv_relu_bn(64, 3, 1, padding='same', in_shape=in_shape)
        self.conv_relu_bn2 = conv_relu_bn(64, 3, 2)
        self.conv_relu_bn3 = conv_relu_bn(128, 3, 1, padding='same')
        self.conv_relu_bn4 = conv_relu_bn(128, 3, 2)
        self.conv_relu_bn5 = conv_relu_bn(256, 3, 1, padding='same')
        self.conv_relu_bn6 = conv_relu_bn(256, 3, 2)
        self.conv_relu_bn7 = conv_relu_bn(512, 3, 1, padding='same')
        self.conv_relu_bn8 = conv_relu_bn(512, 3, 2)
        self.gpool = ls.GlobalAveragePooling2D()
        self.classifier = ls.Dense(n_classes)

    def call(self, inputs):
        conv_relu_bn1 = self.conv_relu_bn1(inputs)
        conv_relu_bn2 = self.conv_relu_bn2(conv_relu_bn1)
        conv_relu_bn3 = self.conv_relu_bn3(conv_relu_bn2)
        conv_relu_bn4 = self.conv_relu_bn4(conv_relu_bn3)
        conv_relu_bn5 = self.conv_relu_bn5(conv_relu_bn4)
        conv_relu_bn6 = self.conv_relu_bn6(conv_relu_bn5)
        conv_relu_bn7 = self.conv_relu_bn7(conv_relu_bn6)
        conv_relu_bn8 = self.conv_relu_bn8(conv_relu_bn7)
        pooled = self.gpool(conv_relu_bn8)
        return self.classifier(pooled)


# In[ ]:


class Trainor:
    def __init__(self,
                 train_csv,
                 val_csv,
                 labels_dict,
                 lr=1e-3,
                 batch_size=8,
                 shuffle=True,
                 model_dir='ds2'):
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        self.model_dir = model_dir
        self.model = Model(in_shape=(300, 300, 1))
        self._define_dataset_iterators(
            train_csv, val_csv, labels_dict, batch_size, shuffle)

        logits = self.model(self.features)
        self.probas = tf.nn.sigmoid(logits)
        preds = self.probas > 0.5

        self.loss = tf.losses.sigmoid_cross_entropy(self.labels, logits)
        self.accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.cast(preds, tf.int32), self.labels), tf.float32))
        optimizer = tf.train.RMSPropOptimizer(lr)
        self.global_step = tf.train.get_or_create_global_step()
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
        
        self.sess = tf.Session()
        self._define_summaries()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self._may_be_load_model()

    def _define_summaries(self):
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)
        flip = tf.image.flip_left_right(self.features)
        transpose = tf.image.transpose_image(flip)
        tf.summary.image('spectogram', transpose)
        self.merged = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.model_dir, self.sess.graph)

    def _define_dataset_iterators(self, train_csv, val_csv, labels_dict, batch_size, shuffle):
        train_dataset = wav_data_generators(
            train_csv, labels_dict, batch_size, shuffle)
        val_dataset = wav_data_generators(
            val_csv, labels_dict, batch_size, shuffle)
        iter = tf.data.Iterator.from_structure(
            train_dataset.output_types, train_dataset.output_shapes)

        self.features, self.labels = iter.get_next()
        self.train_init_op = iter.make_initializer(train_dataset)
        self.val_init_op = iter.make_initializer(val_dataset)

    def _may_be_load_model(self):
        #if os.path.isdir(self.model_dir):
        if os.path.isdir('../input/cnnfat2019'):
            #files = os.listdir(self.model_dir)
            files = os.listdir('../input/cnnfat2019')
            steps = [int(file.split('-')[-1].split('.')[0]) for file in files if 'model' in file]
            if len(steps) > 0:
                #self._load_model(os.path.join(self.model_dir, 'model.ckpt'), max(steps))
                self._load_model(os.path.join('../input/cnnfat2019', 'model.ckpt'), max(steps))

    def _save_model(self):
        return self.saver.save(
            self.sess, os.path.join(self.model_dir, 'model.ckpt'),
            global_step=self.global_step)

    def _load_model(self, ckpt_path, step):
        self.saver.restore(self.sess, ckpt_path + '-{}'.format(step))

    def train(self, steps=500, ckpt_every_n_steps=100, log_every_n_steps=1500, log_val_n_steps=5):
        writer = tf.summary.FileWriter(self.model_dir, self.sess.graph)
        self.sess.run(self.train_init_op)
        for step in range(1, steps+1):
            
            summary, _ = self.sess.run([self.merged, self.train_op])
            self.summary_writer.add_summary(summary, step)

            if step % log_every_n_steps == 0:
                self.sess.run(self.val_init_op)
                val_losses, val_accs = [], []
                for _ in range(log_val_n_steps):
                    l, a = self.sess.run([self.loss, self.accuracy])
                    val_losses.append(l)
                    val_accs.append(a)
                print('STEP: {}, Loss: {:.4f}, Acc: {:.4f}'.format(
                    step, np.mean(val_losses), np.mean(val_accs)))
                self.sess.run(self.train_init_op)

            if step % ckpt_every_n_steps == 0:
                path = self._save_model()
                print('STEP: {}, Model saved at: {}'.format(step, path))
        path = self._save_model()
        print('Final Model saved at: {}'.format(step, path))

    def __del__(self):
        self.sess.close()


# In[ ]:


with open('./preprocessed/labels_dict.json') as fp:
    labels_dict = json.load(fp)


# In[ ]:


# if do_run:
trainor = Trainor(
    './preprocessed/train.csv',
    './preprocessed/val_curated.csv',
labels_dict)


# In[ ]:


# if do_run:
trainor.train(3000)


# In[ ]:


# if not do_run:
#     trainor = Trainor(
#         './preprocessed/val_curated.csv',
#         './preprocessed/train.csv',
#         labels_dict)


# In[ ]:


# if not do_run:
#     trainor.train(700)


# In[ ]:


class Predictor:
    def __init__(self,
                 files,
                 lr=1e-3,
                 batch_size=16,
                 model_dir='ds2'):
        self.model_dir = model_dir
        self.model = Model((300, 300, 1))
        self._define_dataset_iterator(
            files, batch_size)

        logits = self.model(self.features)
        self.probas = tf.nn.sigmoid(logits)
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self._load_model()

    def _define_dataset_iterator(self, files, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices((files,))
        dataset = dataset.map(wav_to_spectogram)

        dataset = dataset.batch(batch_size).prefetch(10)

        dataset = dataset.repeat(1)
        self.features = dataset.make_one_shot_iterator().get_next()

    def _load_model(self):
        files = os.listdir(self.model_dir)
        steps = [int(file.split('-')[-1].split('.')[0]) for file in files if 'model' in file]
        ckpt_path = os.path.join(self.model_dir, 'model.ckpt')
        self.saver.restore(self.sess, ckpt_path + '-{}'.format(max(steps)))

    def pred_generator(self):
        try:
            while True:
                yield self.sess.run(self.probas)
        except:
            print('Prediction over')

    def __del__(self):
        self.sess.close()


# In[ ]:


tf.reset_default_graph()
if do_run:
    orig_files = os.listdir('../input/test')
    files = [os.path.join('../input/test', file) for file in orig_files]
    predictor = Predictor(files)
else:
    orig_files = os.listdir('../input/freesound-audio-tagging-2019/test')
    files = [os.path.join('../input/freesound-audio-tagging-2019/test', file) for file in orig_files]
    predictor = Predictor(files)


# In[ ]:


pred_gen = predictor.pred_generator()


# In[ ]:


pred_dict = {}
for label in labels_dict:
    pred_dict[label] = []

for probs in pred_gen:
    for label in labels_dict:
        pred_dict[label].extend(list(probs[:, labels_dict[label]]))


# In[ ]:


pred_dict['fname'] = orig_files
df = pd.DataFrame(pred_dict, columns=['fname'].extend(list(labels_dict.keys())))
df.to_csv('submission.csv', index=False)

